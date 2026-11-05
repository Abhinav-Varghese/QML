################################################################
import sys
from pathlib import Path
_root = Path(__file__).parent.parent  # Setting root
sys.path.append(str(_root / "jointCode"))
sys.path.append(str(_root / "qubitCode"))
import time
import torch
torch.set_default_dtype(torch.float64)
from omegaconf import OmegaConf
from qubit_config import CONFIG
from data_loader import TTBarZJetsData
from utils import createRunDir, saveUsefulInfo, createSubRNGS, finishRunDir, decode_arg
from plotting import Plotter
from qubit_circ import buildQubit
import numpy as np
from qubit_training import Trainer

start = time.perf_counter()

################################################################
# Override (if scan)(allows passing in a copied version of config for scans)
overrides = [decode_arg(arg) for arg in sys.argv[1:] if "=" in arg and not arg.startswith("--")]

OmegaConf.set_struct(CONFIG, False)
CONFIG = OmegaConf.merge(CONFIG, OmegaConf.from_dotlist(overrides))
OmegaConf.set_struct(CONFIG, True)

#### ---- Create run directory ---- ####
run_path = createRunDir(CONFIG)

#### ---- Initialise useful saving function ---- ####
infoSave = saveUsefulInfo(run_path)

#### ---- Create sub-RNGs ---- ####
rngs = createSubRNGS(CONFIG)

#### ---- Create model ---- ####
model = buildQubit(CONFIG, rngs)

#### ---- Circuit plotting ---- ####
# This must run before any training call, as it actually initialises the model
plotter = Plotter(run_path, CONFIG, "qubit")
plotter.plotQubit(model)

#### ---- Load all data (bg+sg) and plot ---- ####
dataloader = TTBarZJetsData(CONFIG)
rng = np.random.default_rng(CONFIG.misc.random_seed)

if CONFIG.training.paradigm == "unsupervised":
    X_train, X_val, X_te, y_te = dataloader.createUnsupervisedDatasets(rng)
else:
    raise ValueError("cfg.training.paradigm should be 'unsupervised'!")

# Data Plotting
(plotter.plot_input_features(X_train, X_te, y_te)) 

#### ---- Train autoencoder and plot metrics ---- ####
trainer = Trainer(CONFIG, rngs, run_path, infoSave)
if CONFIG.training.paradigm == "unsupervised":
    trainer.train_unsupervised_qubit(X_train, X_val, X_te, y_te, model)
else:
    raise ValueError("Paradigm should be 'unsupervised'!")
    
plotter.plotMetrics(["lr"])
plotter.plotLoss()

# Shouldn't be used for unsupervised!
plotter.plotFinalROC()
plotter.plotFinalLoss()

#### ---- Rename directory upon finish ---- ####
finishRunDir(run_path)
print(f"Time Taken: {time.perf_counter()-start}")
