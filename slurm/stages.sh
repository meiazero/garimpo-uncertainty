#!/usr/bin/env bash
# Sourced by pilot.sbatch and full.sbatch: same stages, same scripts, only the values differ.
#
# Finished stages skip themselves. After fixing code, FORCE_FROM=<stage> redoes that stage and those
# after it; `train` runs again only when named, so a downstream fix never retrains.
# The sweep's beta_max is the search range, so it covers the domain the binary search walks.

STAGE_ORDER=(train infer fit_directions layer_monotonicity choose_layer endpoint_monotonicity
    compass_scores calibrate generate_intervals compute_metrics)
COMMON=(--checkpoint "$RUN_DIR/best_model.pt" --run-dir "$RUN_DIR" --batch-size "$EVAL_BATCH_SIZE"
    --num-workers "$NUM_WORKERS")

if [ -n "${FORCE_FROM:-}" ] && [[ " ${STAGE_ORDER[*]} " != *" $FORCE_FROM "* ]]; then
    echo "FORCE_FROM=$FORCE_FROM is not one of: ${STAGE_ORDER[*]}" >&2
    exit 1
fi

is_forced() {
    [ -n "${FORCE_FROM:-}" ] || return 1
    local stage
    for stage in "${STAGE_ORDER[@]}"; do
        [ "$stage" = "$FORCE_FROM" ] && return 0
        [ "$stage" = "$1" ] && return 1
    done
    return 1
}

run_stage() {
    local stage=$1 script=$2
    shift 2
    local flag=()
    if is_forced "$stage"; then
        flag=(--force)
    fi
    uv run --no-sync python "scripts/$script.py" "$@" "${flag[@]}"
}

monotonicity_args() {
    MONOTONICITY_ARGS=("${COMMON[@]}" --beta-max-logits "$BETA_RANGE_LOGITS"
        --beta-max-features "$BETA_RANGE_FEATURES" --steps "$SWEEP_STEPS"
        --cells-per-role "$SWEEP_CELLS" --seed "$SEED"
        --direction-temperature "$DIRECTION_TEMPERATURE")
}

run_validate() {
    uv run --no-sync python scripts/validate_data.py --train-fraction "$TRAIN_FRACTION" \
        --subset-seed "$SUBSET_SEED" --report "$VALIDATION_REPORT" \
        "${VALIDATE_ARGS[@]}" ${FORCE_VALIDATE:+--force}
}

# Everything up to the layer, chosen from the validation role by choose_layer.py.
run_diagnostics() {
    monotonicity_args
    # Passed whenever set, so train.py alone decides what fits the architecture.
    local encoder_weights=()
    [ -z "${ENCODER_WEIGHTS:-}" ] || encoder_weights=(--encoder-weights "$ENCODER_WEIGHTS")
    run_stage train train --arch "$ARCH" "${ARCH_ARGS[@]}" "${encoder_weights[@]}" --run-dir "$RUN_DIR" \
        --train-fraction "$TRAIN_FRACTION" --subset-seed "$SUBSET_SEED" --epochs "$EPOCHS" \
        --batch-size "$BATCH_SIZE" --lr "$LR" --weight-decay "$WEIGHT_DECAY" \
        --loss "$LOSS" --seed "$SEED" --num-workers "$NUM_WORKERS"
    run_stage infer infer "${COMMON[@]}"
    run_stage fit_directions fit_directions "${COMMON[@]}" --layers "${LAYERS[@]}" \
        --svm-c "$SVM_C" --svm-class-weight "$SVM_CLASS_WEIGHT" \
        --svm-pixels-per-cell "$SVM_PIXELS_PER_CELL" --seed "$SEED" \
        --direction-temperature "$DIRECTION_TEMPERATURE"
    run_stage layer_monotonicity verify_monotonicity "${MONOTONICITY_ARGS[@]}" --use layer \
        --layers "${LAYERS[@]}"
    run_stage choose_layer choose_layer --run-dir "$RUN_DIR" \
        --min-fraction-ascending "$MIN_FRACTION_ASCENDING"
}

run_calibrated_stages() {
    monotonicity_args
    local layer read_layer='import json, sys; print(json.load(open(sys.argv[1]))["layer"])'
    layer=$(uv run --no-sync python -c "$read_layer" "$RUN_DIR/chosen_layer.json")
    run_stage endpoint_monotonicity verify_monotonicity "${MONOTONICITY_ARGS[@]}" --use endpoint \
        --layers "$layer"
    run_stage compass_scores compass_scores "${COMMON[@]}" --layer "$layer" \
        --beta-range-logits "$BETA_RANGE_LOGITS" --beta-range-features "$BETA_RANGE_FEATURES" \
        --search-threshold-logits "$SEARCH_THRESHOLD_LOGITS" \
        --search-threshold-features "$SEARCH_THRESHOLD_FEATURES" \
        --direction-temperature "$DIRECTION_TEMPERATURE"
    run_stage calibrate calibrate --run-dir "$RUN_DIR" --cells "$CELLS" \
        --score-floor-ha "$SCORE_FLOOR_HA" --seed "$SEED" --lightgbm-jobs "$NUM_WORKERS"
    run_stage generate_intervals generate_intervals "${COMMON[@]}" --layer "$layer"
    run_stage compute_metrics compute_metrics --run-dir "$RUN_DIR" \
        --boundary-tolerance-px "$BOUNDARY_TOLERANCE_PX" --seed "$SEED"
}
