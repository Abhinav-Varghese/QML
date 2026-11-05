# Variational Autoencoder

## Overview
This project implements a flexible photonic autoencoder alongside a classical benchmark. Experiments can be launched either locally or on a cluster via run.sh, which executes `main.py`.

Configuration is handled through `config.py`, which loads parameters from `config.yaml`, defining the specific settings for each run. The photonic circuit is constructed in `circuits.py`, combining encoder and base components.

Training logic is implemented in `training.py`, featuring a flexible forward pass. Data loading is managed by `data_loader.py`. Visualisation utilities are provided in `plotting.py`, while `utils.py` contains general helper functions (e.g. directory management and random seed control).

## Installation
```bash
conda env create -f environment.yml
conda activate AutoQMLEnv
```

## Usage
### Condor (single run)
- Ensure all `config.yaml` settings are as intended
```bash
condor_submit singleJob.job
```
### Condor (multi run)
- Ensure all `config.yaml` and `scan_params.yml` settings are as intended
- Ensure the "queue" argument in `multiJob.job` is equal to the number of permutations being scanned.
```bash
condor_submit multiJob.job
```
### Local
- Ensure all `config.yaml` settings are as intended
```bash
./run.sh
```

## Expected Output

Each run should create a new directory named `outputs/{date}/{time}:{UUID}:{optional tag}-{multiscan ID}`, appended with `:Done` upon completion. This directory contains:

- **Plots/**
    - `{feature}.png` - Input data plots showing training (bg) and validation (bg + sg) data.
    - `{circuit}.png` - 5 plots for the `hetFid` quantum loss function (due to the heterodyne implementation), or 1 for all other cases.
    - `ROC.png` + `LossHist.png` - Produced upon completion using validation data.
- **`configUsed.yml`** - Config settings used for the run. If running a multiscan, this reflects the final config with the scan permutation applied.
- **`params.pth`** - Final PyTorch parameters, usable to reload the best model. Note these are PyTorch parameters and not necessarily gate parameters, as the circuit may apply parameter scaling (e.g. gates with angles may multiply the parameter by 2π).
- **`evaluations.csv`** - Final loss and validation data labels from the final model evaluation, used for further plotting and post-processing.
- **`summary.yml`** - Quick-assessment snapshot including initial and final parameters, number of features and trainable parameters, and current epoch and timestamp for monitoring progress during training.