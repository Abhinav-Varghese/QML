"""
extract_raw_predictions.py
---------------------------
Standalone post-processing script: walks a scan_root (same layout as
scan_experiments.py), finds every run's C_evaluations.npz and
Q_evaluations.npz, and merges them into a single HDF5 file (ROC_data.hdf5)
with separate groups per substrate:

    /C/y_true              (n_samples,)          uint8
    /C/losses              (n_samples, n_runs_C) float64
    /C/run_names           (n_runs_C,)           variable-length UTF-8 strings
    /C/train_num_samples   (n_runs_C,)           int64  -- from configUsed.yml

    /Q/y_true              (n_samples,)          uint8
    /Q/losses              (n_samples, n_runs_Q) float64
    /Q/run_names           (n_runs_Q,)           variable-length UTF-8 strings
    /Q/train_num_samples   (n_runs_Q,)           int64  -- from configUsed.yml

Folder layout expected under scan_root:
    scan_root/code_snapshot/outputs/<date>/scan:(...)/<run_folder>/<substrate>_evaluations.npz
    scan_root/code_snapshot/outputs/<date>/scan:(...)/<run_folder>/configUsed.yml

Each run folder can contain either or both of C_evaluations.npz and
Q_evaluations.npz. Before adding a run's losses column, its y_true is
checked against that substrate's reference y_true (the first run seen for
that substrate); mismatches raise immediately and abort the merge.

Column order is NOT assumed to correspond to any particular
train_num_samples grouping -- each run's actual data.train_num_samples is
read from its own configUsed.yml and stored alongside losses/run_names, so
downstream selection (e.g. "give me all runs where train_num_samples==1000")
should filter on that array rather than assume fixed-size contiguous blocks.
"""

import argparse
import sys
from pathlib import Path

import h5py
import numpy as np
import yaml

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable=None, *args, **kwargs):
        if iterable is None:
            return iterable
        return iterable

SUBSTRATES = ["C", "Q"]


def find_scan_roots(date_root: str | Path) -> list[Path]:
    date_root = Path(date_root)
    if not date_root.is_dir():
        return []

    scan_roots = []
    for d in sorted(date_root.iterdir()):
        if d.is_dir() and (d / "code_snapshot" / "outputs").is_dir():
            scan_roots.append(d)
    return scan_roots


def find_runfile_dirs(scan_root: str | Path) -> list[Path]:
    scan_root = Path(scan_root)
    outputs_dir = scan_root / "code_snapshot" / "outputs"
    if not outputs_dir.is_dir():
        return []

    date_dirs = sorted(d for d in outputs_dir.iterdir() if d.is_dir())

    candidates = set(date_dirs)
    for d in tqdm(date_dirs, desc="Scanning date dirs", unit="dir"):
        candidates.update(c for c in d.iterdir() if c.is_dir())

    candidates = sorted(candidates)

    matches = []
    for d in tqdm(candidates, desc="Checking candidate dirs", unit="dir"):
        try:
            has_experiments = any(
                _looks_like_experiment_folder(sub) for sub in d.iterdir() if sub.is_dir()
            )
        except (PermissionError, FileNotFoundError):
            has_experiments = False
        if has_experiments:
            matches.append(d)

    return sorted(matches)


def _looks_like_experiment_folder(d: Path) -> bool:
    parts = d.name.split(":")
    if len(parts) != 3:
        return False
    try:
        int(parts[0])
    except ValueError:
        return False
    return True


def _run_index(name: str) -> float:
    """
    Extracts the leading integer index from a run name like
    'scan:(...)/4:C:Done' -> 4, for numeric sorting. Falls back to +inf
    (sorts last) if the pattern doesn't match, so malformed names don't
    crash the sort.
    """
    run_folder_name = name.split("/")[-1]
    idx_str = run_folder_name.split(":")[0]
    try:
        return int(idx_str)
    except ValueError:
        return float("inf")


def find_evaluation_files(runfile_dirs: list[Path]) -> dict[str, list[tuple[str, Path]]]:
    """
    find_runfile_dirs already returns directories one level ABOVE the run
    folders (e.g. .../scan:(UnsupCvsQ_strong_weak_C_C)), since that's the
    level whose immediate children match the 'N:label:status' pattern.
    So here we iterate each runfile_dir's own children directly as run
    folders, rather than descending an extra level.
    """
    found: dict[str, list[tuple[str, Path]]] = {s: [] for s in SUBSTRATES}

    for runfile_dir in runfile_dirs:
        run_folders = sorted(d for d in runfile_dir.iterdir() if d.is_dir())
        for run_folder in run_folders:
            for substrate in SUBSTRATES:
                npz_path = run_folder / f"{substrate}_evaluations.npz"
                if npz_path.exists():
                    name = f"{runfile_dir.name}/{run_folder.name}"
                    found[substrate].append((name, npz_path))

    # Sort numerically by the run's leading index (e.g. '4:C:Done' -> 4),
    # rather than lexicographically, so columns end up ordered 0,1,2,3,...
    # (This is purely cosmetic ordering now -- correctness of grouping by
    # train_num_samples comes from the config-read value, not this order.)
    for substrate in SUBSTRATES:
        found[substrate].sort(key=lambda entry: _run_index(entry[0]))

    return found


def _read_train_num_samples(npz_path: Path) -> int:
    """
    Reads data.train_num_samples from the configUsed.yml sitting alongside
    a run's *_evaluations.npz file. Raises FileNotFoundError / KeyError
    (propagated up) if the config or the key is missing, so a run with no
    reliable train_num_samples never silently ends up mis-grouped.
    """
    config_path = npz_path.parent / "configUsed.yml"
    if not config_path.exists():
        raise FileNotFoundError(f"Missing configUsed.yml next to {npz_path}")

    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f) or {}

    try:
        return int(cfg["data"]["train_num_samples"])
    except (KeyError, TypeError) as exc:
        raise KeyError(
            f"Could not read data.train_num_samples from {config_path}"
        ) from exc

def _read_encoder_shape(npz_path: Path) -> str:
    """
    Reads circuit.encoder_shape from the configUsed.yml sitting alongside a
    run's *_evaluations.npz file, and returns it as a string (e.g.
    "[24, 32, 32, 32]") so it can be stored as a plain HDF5 string dataset
    and compared/filtered on exactly, without needing ragged/object arrays.
    Quantum runs may not have this key (it's a classical-only config field);
    returns "" in that case rather than raising, so quantum runs aren't
    dropped.
    """
    config_path = npz_path.parent / "configUsed.yml"
    if not config_path.exists():
        raise FileNotFoundError(f"Missing configUsed.yml next to {npz_path}")

    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f) or {}

    try:
        shape = cfg["circuit"]["encoder_shape"]
    except (KeyError, TypeError):
        return ""

    return str(shape)

def _collect_substrate(eval_files: list[tuple[str, Path]], substrate: str):
    """Returns (ref_y_true, run_names, losses_cols, train_num_samples_list,
    encoder_shape_list) for one substrate.
    """
    ref_y_true = None
    ref_run_name = None
    run_names = []
    losses_cols = []
    train_num_samples_list = []
    encoder_shape_list = []

    for run_name, npz_path in tqdm(eval_files, desc=f"Reading {substrate} runs", unit="run"):
        data = np.load(npz_path)
        y_true = data["y_true"].astype(np.uint8)
        losses = data["losses"].astype(np.float64)

        if ref_y_true is None:
            ref_y_true = y_true
            ref_run_name = run_name
        elif y_true.shape != ref_y_true.shape or not np.array_equal(y_true, ref_y_true):
            raise ValueError(
                f"[{substrate}] y_true mismatch: run '{run_name}' (shape {y_true.shape}) "
                f"does not match reference run '{ref_run_name}' (shape {ref_y_true.shape}). "
                f"Aborting merge — all runs for a given substrate must share the same eval set."
            )

        train_num_samples = _read_train_num_samples(npz_path)
        encoder_shape = _read_encoder_shape(npz_path)

        run_names.append(run_name)
        losses_cols.append(losses)
        train_num_samples_list.append(train_num_samples)
        encoder_shape_list.append(encoder_shape)

    return ref_y_true, run_names, losses_cols, train_num_samples_list, encoder_shape_list


def build_raw_predictions_h5(
    scan_root: Path,
    out_path: Path,
    compression: str = "gzip",
    compression_opts: int = 4,
) -> None:
    runfile_dirs = find_runfile_dirs(scan_root)
    if not runfile_dirs:
        print(f"[warn] No runfiles dirs found under {scan_root}, skipping", file=sys.stderr)
        return

    eval_files = find_evaluation_files(runfile_dirs)

    if not any(eval_files.values()):
        print(f"[warn] No *_evaluations.npz files found under {scan_root}, skipping", file=sys.stderr)
        return

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(out_path, "w") as h5f:
        h5f.attrs["scan_root"] = str(scan_root)

        for substrate in SUBSTRATES:
            files = eval_files[substrate]
            if not files:
                print(f"[info] No {substrate}_evaluations.npz files found for {scan_root.name}, skipping group", file=sys.stderr)
                continue

            ref_y_true, run_names, losses_cols, train_num_samples_list, encoder_shape_list = _collect_substrate(files, substrate)

            if not run_names:
                print(f"[warn] No valid {substrate} runs collected for {scan_root}, skipping group", file=sys.stderr)
                continue

            losses_matrix = np.column_stack(losses_cols)

            grp = h5f.create_group(substrate)
            grp.create_dataset("y_true", data=ref_y_true, compression=compression, compression_opts=compression_opts)
            grp.create_dataset("losses", data=losses_matrix, compression=compression, compression_opts=compression_opts)
            grp.create_dataset(
                "run_names",
                data=np.array(run_names, dtype=h5py.string_dtype(encoding="utf-8")),
            )
            grp.create_dataset(
                "train_num_samples",
                data=np.array(train_num_samples_list, dtype=np.int64),
            )
            grp.create_dataset(
                "encoder_shape",
                data=np.array(encoder_shape_list, dtype=h5py.string_dtype(encoding="utf-8")),
            )
            grp.attrs["n_runs"] = len(run_names)
            grp.attrs["n_samples"] = ref_y_true.shape[0]

            # Sanity check: counts per (train_num_samples, encoder_shape) combo
            combo_keys = list(zip(train_num_samples_list, encoder_shape_list))
            unique_combos, counts = np.unique(
                [f"{tns}|{shp}" for tns, shp in combo_keys], return_counts=True
            )
            counts_str = ", ".join(f"{c_key}:{c}" for c_key, c in zip(unique_combos, counts))
            print(
                f"[info] {substrate}: saved {len(run_names)} run(s) "
                f"(train_num_samples|encoder_shape counts -> {counts_str})",
                file=sys.stderr,
            )

    print(f"[info] Saved -> {out_path}", file=sys.stderr)


def main():
    parser = argparse.ArgumentParser(
        description="Merge per-run C/Q_evaluations.npz files into a single HDF5 (y_true + losses matrix per substrate)."
    )
    parser.add_argument(
        "--scan_root",
        required=True,
        default=None,
        metavar="DIR",
        help="Single scan_root directory, e.g. .../UnsupCvsQ_strong_weak_C",
    )
    parser.add_argument(
        "--scan_date_root",
        default=None,
        metavar="DIR",
        help="Batch mode: process every scan_root found under this date folder.",
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        metavar="DIR",
        help="Output directory. Defaults to each scan_root itself.",
    )
    args = parser.parse_args()

    if not args.scan_root and not args.scan_date_root:
        parser.error("Provide one of --scan_root or --scan_date_root")

    if args.scan_date_root:
        date_root = Path(args.scan_date_root)
        scan_roots = find_scan_roots(date_root)
        if not scan_roots:
            parser.error(f"No valid scan_root directories found under {date_root}")
        for scan_root in scan_roots:
            out_dir = Path(args.output_dir) if args.output_dir else scan_root
            prefix = out_dir.name
            build_raw_predictions_h5(scan_root, out_dir / f"{prefix}_ROC_data.hdf5")
        return

    scan_root = Path(args.scan_root)
    out_dir = Path(args.output_dir) if args.output_dir else scan_root
    prefix = out_dir.name
    build_raw_predictions_h5(scan_root, out_dir / f"{prefix}_ROC_data.hdf5")


if __name__ == "__main__":
    main()