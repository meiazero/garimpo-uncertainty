"""The automatic layer choice on hand-written diagnostics, with no model behind them.

Ways it could fail, each a scenario below: choosing on calibration or test cells, choosing a layer
whose sweep is not ascending, letting one direction stand for both, inventing a layer when none
qualifies, choosing among layers the two tables disagree on, and an order-dependent tie.
"""

import json
from pathlib import Path

import pandas as pd

from tests.e2e import support

VARIANCE = {"up.0": 0.95, "up.1": 0.80, "up.2": 0.60}


def write_run(
    run_dir: Path,
    ascending: dict[tuple[str, str], float],
    variance: dict[str, float] = VARIANCE,
    extra_rows: tuple[dict, ...] = (),
) -> Path:
    """Write the two tables choose_layer reads, as fit_directions and the layer sweep write them."""
    (run_dir / "directions").mkdir(parents=True)
    pd.DataFrame(
        {"layer": list(variance), "pca_explained_variance": list(variance.values())}
    ).to_csv(run_dir / "directions" / "diagnostics.csv", index=False)
    rows = [
        {"layer": "logits", "direction": "uniform", "role": "validation", "fraction_ascending": 1.0}
    ]
    rows += [
        {"layer": layer, "direction": direction, "role": "validation", "fraction_ascending": value}
        for (layer, direction), value in ascending.items()
    ]
    pd.DataFrame([*rows, *extra_rows]).to_csv(run_dir / "monotonicity_layer.csv", index=False)
    return run_dir


def both(values: dict[str, float]) -> dict[tuple[str, str], float]:
    return {(layer, d): value for layer, value in values.items() for d in ("pca", "svm")}


def choose(run_dir: Path, minimum: str = "1.0", check: bool = True):
    args = ["--run-dir", str(run_dir), "--min-fraction-ascending", minimum]
    return support.run("choose_layer.py", *args, data_root=run_dir, check=check)


def chosen(run_dir: Path) -> dict:
    return json.loads((run_dir / "chosen_layer.json").read_text())


def test_highest_variance_among_ascending_layers(tmp_path: Path) -> None:
    """up.0 explains the most variance but is not ascending everywhere, so up.1 wins."""
    run_dir = write_run(tmp_path, both({"up.0": 0.9, "up.1": 1.0, "up.2": 1.0}))
    choose(run_dir)
    record = chosen(run_dir)
    assert record["layer"] == "up.1"
    assert record["min_fraction_ascending"] == 1.0
    qualifies = {c["layer"]: c["qualifies"] for c in record["candidates"]}
    assert qualifies == {"up.0": False, "up.1": True, "up.2": True}


def test_the_threshold_is_inclusive_and_chooses_by_variance(tmp_path: Path) -> None:
    run_dir = write_run(tmp_path, both({"up.0": 0.9, "up.1": 1.0, "up.2": 1.0}))
    choose(run_dir, minimum="0.9")
    assert chosen(run_dir)["layer"] == "up.0"


def test_one_direction_failing_disqualifies_the_layer(tmp_path: Path) -> None:
    """compass_j_pca and compass_j_svm both run on the chosen layer, so both must be ascending."""
    ascending = both({"up.1": 1.0, "up.2": 1.0}) | {("up.0", "pca"): 1.0, ("up.0", "svm"): 0.5}
    run_dir = write_run(tmp_path, ascending)
    choose(run_dir)
    assert chosen(run_dir)["layer"] == "up.1"


def test_a_layer_with_one_direction_swept_refuses(tmp_path: Path) -> None:
    """Its svm fraction is missing, not passing: a minimum over what is present would let it in."""
    run_dir = write_run(tmp_path, both({"up.1": 1.0, "up.2": 1.0}) | {("up.0", "pca"): 1.0})
    result = choose(run_dir, check=False)
    assert result.returncode != 0
    assert not (run_dir / "chosen_layer.json").exists()


def test_a_forced_rerun_that_refuses_leaves_no_old_choice(tmp_path: Path) -> None:
    """The pipeline check prints chosen_layer.json, so an old one would read as today's choice."""
    run_dir = write_run(tmp_path, both({"up.0": 0.9, "up.1": 1.0, "up.2": 1.0}))
    choose(run_dir)
    args = ["--run-dir", str(run_dir), "--min-fraction-ascending", "1.0", "--force"]
    (run_dir / "monotonicity_layer.csv").write_text(
        (run_dir / "monotonicity_layer.csv").read_text().replace(",1.0\n", ",0.5\n")
    )
    result = support.run("choose_layer.py", *args, data_root=run_dir, check=False)
    assert result.returncode != 0
    assert not (run_dir / "chosen_layer.json").exists()


def test_only_the_validation_role_is_read(tmp_path: Path) -> None:
    """Test rows that would make up.0 qualify must not reach the choice."""
    leaked = tuple(
        {"layer": "up.0", "direction": d, "role": role, "fraction_ascending": 1.0}
        for d in ("pca", "svm")
        for role in ("calibration", "test")
    )
    run_dir = write_run(tmp_path, both({"up.0": 0.5, "up.1": 1.0, "up.2": 1.0}), extra_rows=leaked)
    choose(run_dir)
    assert chosen(run_dir)["layer"] == "up.1"


def test_no_qualifying_layer_refuses_and_names_the_best(tmp_path: Path) -> None:
    run_dir = write_run(tmp_path, both({"up.0": 0.5, "up.1": 0.7, "up.2": 0.6}))
    result = choose(run_dir, check=False)
    assert result.returncode != 0
    assert not (run_dir / "chosen_layer.json").exists()
    record = json.loads((run_dir / "status" / "choose_layer.json").read_text())
    assert record["status"] == "failed"
    assert "up.1" in record["error"]
    assert "MIN_FRACTION_ASCENDING" in record["error"]


def test_layers_the_tables_disagree_on_refuse(tmp_path: Path) -> None:
    """A layer with directions but no sweep has no monotonicity evidence; it must not be chosen."""
    run_dir = write_run(tmp_path, both({"up.1": 1.0, "up.2": 1.0}))
    result = choose(run_dir, check=False)
    assert result.returncode != 0
    record = json.loads((run_dir / "status" / "choose_layer.json").read_text())
    assert "up.0" in record["error"]


def test_a_tie_goes_to_the_first_layer_of_the_diagnostics(tmp_path: Path) -> None:
    variance = {"up.2": 0.8, "up.1": 0.8, "up.0": 0.5}
    run_dir = write_run(tmp_path, both(dict.fromkeys(variance, 1.0)), variance=variance)
    choose(run_dir)
    assert chosen(run_dir)["layer"] == "up.2"


def test_a_new_threshold_refuses_the_old_choice(tmp_path: Path) -> None:
    """Later stages ran on the old layer, so changing the rule must not pass silently."""
    run_dir = write_run(tmp_path, both({"up.0": 0.9, "up.1": 1.0, "up.2": 1.0}))
    choose(run_dir)
    choose(run_dir)
    result = choose(run_dir, minimum="0.9", check=False)
    assert result.returncode != 0
    assert "--force-from choose_layer" in result.stderr
    assert chosen(run_dir)["layer"] == "up.1"
