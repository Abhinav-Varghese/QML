"""
Merge two .h5 files, keeping only:
    - truth_labels
    - jetConstituentsList
    - particleFeatureNames
"""

import h5py
import numpy as np
from pathlib import Path

INPUT_PATHS = [
    "/vols/cms/bmaier/flat_val/TTBar_/TTBar__flat_pt_sample.h5",
    "/vols/cms/bmaier/flat_test/TTBar_/TTBar__flat_pt_sample.h5",
    "/vols/cms/bmaier/flat_train/TTBar_/TTBar__flat_pt_sample.h5",
    "/vols/cms/bmaier/flat_val/ZJetsToNuNu/ZJetsToNuNu_flat_pt_sample.h5",
    "/vols/cms/bmaier/flat_test/ZJetsToNuNu/ZJetsToNuNu_flat_pt_sample.h5",
    "/vols/cms/bmaier/flat_train/ZJetsToNuNu/ZJetsToNuNu_flat_pt_sample.h5",
]
OUTPUT_PATH = "/vols/cms/lgc124/QMLoutputs/TTBar_+ZJetsToNuNU_flat_merged.h5"

# Datasets to concatenate along axis 0 (one entry per jet)
CONCAT_DATASETS = ["truth_labels", "jetConstituentsList"]

# Datasets that are same in both files (copy once)
COPY_ONCE_DATASETS = ["particleFeatureNames"]


def merge_h5_files(input_paths, output_path):
    input_paths = [Path(p) for p in input_paths]
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(output_path, "w") as fout:

        # --- Copy once datasets (e.g. feature name strings) ---
        # These are the same in both files, so just take them from the first
        with h5py.File(input_paths[0], "r") as f:
            for name in COPY_ONCE_DATASETS:
                fout.create_dataset(name, data=f[name][...])
                print(f"Copied '{name}' from first file: shape {f[name].shape}")

        # --- Concatenate per-jet datasets across all files ---
        # Load everything into memory then write — simple and fine for ~10k jets
        for name in CONCAT_DATASETS:
            arrays = []
            for p in input_paths:
                with h5py.File(p, "r") as f:
                    arrays.append(f[name][...])
                    print(f"Read '{name}' from {p.name}: shape {f[name].shape}")

            merged = np.concatenate(arrays, axis=0)
            fout.create_dataset(name, data=merged, compression="gzip", compression_opts=4)
            print(f"Written '{name}' -> merged shape {merged.shape}\n")

    print(f"Done! Merged file: {output_path}")


if __name__ == "__main__":
    merge_h5_files(INPUT_PATHS, OUTPUT_PATH)