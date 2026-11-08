"""End-to-end checks on synthetic data, on CPU, through the same code a pod runs.

Covered: select_tiles on the real split, the versioned tile lists, the cell store, the download on
local rasters (tile_origin, process_tile, _read_band, _boa_offset, download_scenes, the lists and
scenes commands), validation with check_split_counts, every later stage, and two training fractions
sharing one data root.
Not covered: real HTTP (STAC, SAS tokens, 429 retries, Zenodo), remote reads, real Sentinel-2
scenes, the shell scripts in slurm/ including FORCE_FROM and the EXIT trap, the metadata and weights
download commands, GPU, the real ImageNet and Sentinel-2 MoCo files (synthetic ones stand in).
Artifacts stay in tests/artifacts/.
"""
