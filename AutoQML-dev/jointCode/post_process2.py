"""
scan_experiments.py
-------------------
(see module docstring in original file)
"""

import argparse
import sys
from pathlib import Path
from functools import reduce
from typing import Iterable, Union
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
from sklearn.metrics import roc_auc_score
from scipy.stats import wasserstein_distance
import pandas as pd
import yaml

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable=None, *args, **kwargs):
        if iterable is None:
            return iterable
        return iterable

try:
    _YAML_LOADER = yaml.CSafeLoader
except AttributeError:
    _YAML_LOADER = yaml.SafeLoader

SKIP_KEYS = {
    "data.data_path",
    "data.save_path",
    "misc.scan_id",
    "misc.scan_index",
    "misc.scan_keys",
    "misc.scan_params",
    "data.HLS4ML_features",
    "data.HLS4ML_signal",
    "data.HLS4ML_background",
    "data.LHCOlympics_features",
    "data.ds_lims",
    "misc.folder_tag",
    "misc.scan_params.bs_substrate_type",
    "misc.scan_params.random_seed",
    "misc.scan_params.wires",
}


def _get_nested(d: dict, dotted_key: str, default=None):
    keys = dotted_key.split(".")
    try:
        return reduce(lambda acc, k: acc[k], keys, d)
    except (KeyError, TypeError):
        return default


def _flatten_keys(d: dict, prefix: str = "") -> list[str]:
    keys = []
    for k, v in d.items():
        full_key = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict):
            keys.extend(_flatten_keys(v, full_key))
        else:
            keys.append(full_key)
    return keys


def _parse_folder_name(name: str):
    parts = name.split(":")
    if len(parts) >= 2:
        try:
            return int(parts[0]), parts[1]
        except ValueError:
            pass
    return None, name


def _load_yaml(path: Path) -> dict:
    try:
        with path.open() as f:
            data = yaml.load(f, Loader=_YAML_LOADER) or {}
        return data if isinstance(data, dict) else {}
    except FileNotFoundError:
        return {}
    except yaml.YAMLError as exc:
        print(f"  [warn] YAML parse error in {path}: {exc}", file=sys.stderr)
        return {}


def _looks_like_experiment_folder(d: Path) -> bool:
    parts = d.name.split(":")
    if len(parts) != 3:
        return False
    try:
        int(parts[0])
    except ValueError:
        return False
    return True


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


def find_scan_roots(date_root: str | Path) -> list[Path]:
    date_root = Path(date_root)
    if not date_root.is_dir():
        return []

    scan_roots = []
    for d in sorted(date_root.iterdir()):
        if d.is_dir() and (d / "code_snapshot" / "outputs").is_dir():
            scan_roots.append(d)
    return scan_roots


def _load_test_metrics(folder: Path) -> dict:
    out = {"test_auc": None, "test_wasserstein": None}
    npz_files = list(folder.glob("*_evaluations.npz"))
    if not npz_files:
        return out
    if len(npz_files) > 1:
        print(f"  [warn] Multiple *_evaluations.npz found in {folder.name}, using first: {npz_files[0].name}", file=sys.stderr)

    try:
        data = np.load(npz_files[0])
        losses = data["losses"].astype(np.float64)
        y_true = data["y_true"].astype(np.uint8)
        out["test_auc"] = roc_auc_score(y_true, losses)
        # Raw W1 in loss units: only comparable within a model family / loss definition
        out["test_wasserstein"] = wasserstein_distance(losses[y_true == 0], losses[y_true == 1])
    except Exception as exc:
        print(f"  [warn] Failed to compute test metrics from {npz_files[0]}: {exc}", file=sys.stderr)
    return out


def build_experiment_df(
    scan_dir: Union[str, Path, Iterable[Union[str, Path]]],
    keys: list[str] | None = None,
) -> pd.DataFrame:
    if isinstance(scan_dir, (str, Path)):
        scan_dirs = [Path(scan_dir)]
    else:
        scan_dirs = [Path(d) for d in scan_dir]

    scan_dirs = [d for d in scan_dirs if d.is_dir()]
    if not scan_dirs:
        raise NotADirectoryError(
            f"No valid scan directories found among: {scan_dir}"
        )

    all_folders = []
    for scan_d in scan_dirs:
        all_folders.extend([d for d in scan_d.iterdir() if d.is_dir()])

    seen_names = {}
    for d in all_folders:
        seen_names.setdefault(d.name, []).append(d)
    dupes = {name: paths for name, paths in seen_names.items() if len(paths) > 1}
    if dupes:
        print(
            f"[warn] {len(dupes)} experiment folder name(s) appear more than once "
            f"across merged directories (possible duplication from a manual move/rsync). "
            f"Keeping all copies as separate rows — inspect these if results look off:",
            file=sys.stderr,
        )
        for name, paths in list(dupes.items())[:10]:
            print(f"    {name}: {[str(p) for p in paths]}", file=sys.stderr)
        if len(dupes) > 10:
            print(f"    ... and {len(dupes) - 10} more", file=sys.stderr)

    all_folders = sorted(
        all_folders,
        key=lambda d: (_parse_folder_name(d.name)[0] or float("inf"), d.name),
    )

    if not all_folders:
        print(f"[warn] No sub-directories found in {scan_dirs}", file=sys.stderr)
        return pd.DataFrame()

    def _read_one(folder: Path):
        """Read a single experiment folder. Returns None if summary.yml is missing."""
        summary_path = folder / "summary.yml"
        if not summary_path.exists():
            return None

        summary = _load_yaml(summary_path)
        best_val_loss = summary.get("Best_val_loss", None)
        best_epoch = summary.get("Best_epoch", None)
        best_val_auc = summary.get("Best_val_auc", None)
        num_trainable = summary.get("NumTrainable", None)
        test_metrics = _load_test_metrics(folder)

        config = _load_yaml(folder / "configUsed.yml")
        idx, label = _parse_folder_name(folder.name)

        return (folder, idx, label, best_val_loss, best_val_auc, best_epoch, test_metrics, num_trainable, config)

    # Read all folders concurrently (I/O-bound: yaml/npz reads + roc_auc_score
    # on each). Threads (not processes) are fine here since the work is I/O-
    # bound and releases the GIL during file reads / numpy ops.
    #
    # Results are collected into a position-indexed list, keyed by each
    # folder's position in `all_folders`, so we can restore the exact
    # original order afterward regardless of which thread finishes first.
    # This matters for two reasons: (1) reproducibility of which folder
    # "wins" if duplicate names exist upstream, and (2) auto-discovered key
    # ordering below uses folder_configs' position to break ties, so that
    # ordering must stay deterministic across runs.
    n_skipped = 0
    results: list = [None] * len(all_folders)

    with ThreadPoolExecutor(max_workers=min(32, len(all_folders) or 1)) as executor:
        future_to_pos = {
            executor.submit(_read_one, folder): pos
            for pos, folder in enumerate(all_folders)
        }
        for future in tqdm(
            as_completed(future_to_pos), total=len(future_to_pos),
            desc="Reading experiment folders", unit="exp",
        ):
            pos = future_to_pos[future]
            results[pos] = future.result()

    folder_configs = []
    for r in results:
        if r is None:
            n_skipped += 1
        else:
            folder_configs.append(r)

    if n_skipped:
        print(f"[info] Skipped {n_skipped} folder(s) with no summary.yml", file=sys.stderr)

    if not folder_configs:
        print(f"[warn] No folders with summary.yml found in {scan_dirs}", file=sys.stderr)
        return pd.DataFrame()

    if keys is None:
        seen = {}
        for i, (_, _, _, _, _, _, _, _, config) in enumerate(folder_configs):
            for key in _flatten_keys(config):
                if key not in SKIP_KEYS and key not in seen:
                    seen[key] = i
        keys = sorted(seen.keys(), key=lambda k: (k.split(".")[0], seen[k], k))
        print(f"[info] Auto-discovered {len(keys)} keys", file=sys.stderr)

    rows = []
    for folder, idx, label, best_val_loss, best_val_auc, best_epoch, test_metrics, num_trainable, config in folder_configs:
        extracted = {key: _get_nested(config, key) for key in keys}
        rows.append({
            "index": idx,
            "label": label,
            "folder": folder.name,
            "Best_val_loss": best_val_loss,
            "Best_val_auc": best_val_auc,
            "Best_epoch": best_epoch,
            **test_metrics,
            "NumTrainable": num_trainable,
            **extracted,
        })

    df = pd.DataFrame(rows)

    fixed_cols = ["index", "label", "folder", "Best_val_loss", "Best_val_auc", "Best_epoch",
                  "test_auc", "test_wasserstein", "NumTrainable"]
    ordered_cols = fixed_cols + [k for k in keys if k in df.columns]
    df = df[ordered_cols]

    return df.sort_values("index", na_position="last").reset_index(drop=True)


def merge_experiments(
    df: pd.DataFrame,
    same_settings: list[str],
    merge_over: list[str],
) -> pd.DataFrame:
    value_cols = [
        c for c in df.columns if c not in same_settings and c not in merge_over
    ]

    df = df.copy()
    df["_group_rank"] = df.groupby(same_settings, sort=False).cumcount()

    merge_wide = df.pivot_table(
        index=same_settings, columns="_group_rank", values=merge_over, aggfunc="first"
    )
    merge_wide.columns = [f"{col}_{rank}" for col, rank in merge_wide.columns]

    value_wide = df.pivot_table(
        index=same_settings, columns="_group_rank", values=value_cols, aggfunc="first"
    )
    value_wide.columns = [f"{col}_{rank}" for col, rank in value_wide.columns]

    merged = pd.concat([merge_wide, value_wide], axis=1).reset_index()

    auc_cols = sorted([
        c for c in merged.columns
        if c.startswith(("Best_val_loss_", "test_auc_", "test_wasserstein_"))
    ])
    seed_cols = sorted(
        [c for c in merged.columns if any(c.startswith(f"{m}_") for m in merge_over)]
    )
    other_cols = [
        c for c in merged.columns if c not in same_settings + auc_cols + seed_cols
    ]

    return merged[same_settings + auc_cols + seed_cols + other_cols]


def _process_scan_root(
    scan_root: Path,
    keys: list[str] | None,
    same_settings: list[str] | None,
    merge_over: list[str] | None,
    output_dir: Path | None,
    output_filename: str | None,
    no_print: bool,
) -> None:
    """
    Runs find_runfile_dirs -> build_experiment_df -> (optional merge) -> save
    for a single scan_root, and prints/saves results. Shared logic used by
    both the single --scan_root path and the --scan_date_root batch path.
    """
    runfile_dirs = find_runfile_dirs(scan_root)
    if not runfile_dirs:
        print(f"[warn] No code_snapshot/outputs/*/runfiles directories found under {scan_root}, skipping", file=sys.stderr)
        return

    if len(runfile_dirs) > 1:
        print(
            f"[info] Found {len(runfile_dirs)} runfiles dirs across date "
            f"boundaries, merging: {[str(d) for d in runfile_dirs]}",
            file=sys.stderr,
        )

    df = build_experiment_df(scan_dir=runfile_dirs, keys=keys)

    if df.empty:
        print(f"[warn] Empty DataFrame for {scan_root.name}, skipping save", file=sys.stderr)
        return

    df_raw = None
    if same_settings and merge_over:
        df_raw = df.copy()
        df = merge_experiments(df, same_settings=same_settings, merge_over=merge_over)

    resolved_output_dir = output_dir if output_dir else scan_root
    resolved_output_filename = output_filename or f"{scan_root.name}.csv"

    resolved_output_dir.mkdir(parents=True, exist_ok=True)

    out_path = resolved_output_dir / resolved_output_filename
    df.to_csv(out_path, index=False)
    print(f"[info] Saved to {out_path}", file=sys.stderr)

    if df_raw is not None:
        raw_name = resolved_output_filename.replace(".csv", "_raw.csv")
        df_raw.to_csv(resolved_output_dir / raw_name, index=False)
        print(f"[info] Raw saved to {resolved_output_dir / raw_name}", file=sys.stderr)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Extract experiment configs + AUC scores into a DataFrame / CSV. "
            "Works for quantum, classical, or mixed scan directories."
        )
    )
    parser.add_argument(
        "--scan_root",
        default=None,
        metavar="DIR",
        help=(
            "Scan root directory (the folder containing code_snapshot/), "
            "e.g. /vols/cms/lgc124/QMLoutputs/quantum_scan_logs/2026-07-06/fixedRatioSampleScan. "
            "Automatically globs code_snapshot/outputs/*/runfiles across ALL date "
            "sub-directories (handles overnight scans that spill into the next day), "
            "and by default writes the output CSV as <scan_root>/<scan_root_name>.csv. "
        ),
    )
    parser.add_argument(
        "--scan_date_root",
        default=None,
        metavar="DIR",
        help=(
            "Batch mode: point at a date folder containing MULTIPLE scan_root "
            "directories (each with its own code_snapshot/outputs/). Runs the "
            "same extraction independently on every scan_root found directly "
            "beneath it, saving one CSV per scan as <scan_date_root>/<scan_name>/<scan_name>.csv "
            "(same layout as running --scan_root on each individually). "
            "Mutually exclusive with --scan_root/--scan_dir."
        ),
    )
    parser.add_argument(
        "--scan_dir",
        default=None,
        help=(
            "Legacy mode: point directly at a single experiment-folders directory "
            "(no date-glob, no auto-naming). Ignored if --scan_root is given."
        ),
    )
    parser.add_argument(
        "--keys",
        nargs="+",
        required=False,
        default=None,
        metavar="KEY",
        help=(
            "Dot-separated config keys to extract. "
            "If not provided, all keys are auto-discovered from configUsed.yml files."
        ),
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        metavar="DIR",
        help=(
            "Output directory. CSV saved as <output_dir>/<output_filename>. "
            "If --scan_root is given and this is omitted, defaults to <scan_root> itself. "
            "In --scan_date_root batch mode, this is ignored per-scan (each scan's CSV "
            "always saves inside its own scan_root folder) unless explicitly set, in "
            "which case ALL scans' CSVs are saved into this single shared directory."
        ),
    )
    parser.add_argument(
        "--output_filename",
        default=None,
        metavar="FILENAME",
        help=(
            "Name of the output CSV file. If --scan_root is given and this is "
            "omitted, defaults to '<scan_root_name>.csv' (e.g. fixedRatioSampleScan.csv). "
            "Falls back to 'results.csv' otherwise. In --scan_date_root batch mode, this "
            "is ignored per-scan unless explicitly set (each scan defaults to its own name)."
        ),
    )
    parser.add_argument(
        "--same_settings",
        nargs="+",
        default=None,
        metavar="COL",
        help="Columns that must match for rows to be merged.",
    )
    parser.add_argument(
        "--merge_over",
        nargs="+",
        default=None,
        metavar="COL",
        help="Columns to collapse over (e.g. misc.random_seed).",
    )
    parser.add_argument(
        "--no_print",
        action="store_true",
        help="Suppress printing the DataFrame to stdout",
    )
    args = parser.parse_args()

    if not args.scan_root and not args.scan_dir and not args.scan_date_root:
        parser.error("Provide one of --scan_root (preferred), --scan_dir, or --scan_date_root")

    if args.scan_date_root:
        date_root = Path(args.scan_date_root)
        scan_roots = find_scan_roots(date_root)

        if not scan_roots:
            parser.error(
                f"No valid scan_root directories (with code_snapshot/outputs/) found under {date_root}"
            )

        print(f"[info] Found {len(scan_roots)} scan(s) under {date_root}:", file=sys.stderr)
        for sr in scan_roots:
            print(f"    {sr.name}", file=sys.stderr)

        explicit_output_dir = Path(args.output_dir) if args.output_dir else None

        for scan_root in scan_roots:
            print(f"\n[info] === Processing scan: {scan_root.name} ===", file=sys.stderr)
            _process_scan_root(
                scan_root=scan_root,
                keys=args.keys,
                same_settings=args.same_settings,
                merge_over=args.merge_over,
                output_dir=explicit_output_dir,
                output_filename=args.output_filename,
                no_print=args.no_print,
            )
        return

    if args.scan_root:
        scan_root = Path(args.scan_root)
        _process_scan_root(
            scan_root=scan_root,
            keys=args.keys,
            same_settings=args.same_settings,
            merge_over=args.merge_over,
            output_dir=Path(args.output_dir) if args.output_dir else None,
            output_filename=args.output_filename,
            no_print=args.no_print,
        )
        return

    scan_dir_arg = args.scan_dir
    output_dir = Path(args.output_dir) if args.output_dir else None
    output_filename = args.output_filename or "results.csv"

    df = build_experiment_df(scan_dir=scan_dir_arg, keys=args.keys)

    df_raw = None
    if args.same_settings and args.merge_over:
        df_raw = df.copy()
        df = merge_experiments(
            df, same_settings=args.same_settings, merge_over=args.merge_over
        )

    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)

        out_path = output_dir / output_filename
        df.to_csv(out_path, index=False)
        print(f"\nSaved to {out_path}", file=sys.stderr)

        if df_raw is not None:
            raw_name = output_filename.replace(".csv", "_raw.csv")
            df_raw.to_csv(output_dir / raw_name, index=False)
            print(f"Raw saved to {output_dir / raw_name}", file=sys.stderr)


if __name__ == "__main__":
    main()