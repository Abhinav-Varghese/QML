################################################################
import time
from q_training import Trainer
import torch
from q_circ import make_params
from omegaconf import OmegaConf
from plotting import Plotter
from q_config import CONFIG
from data_loader import TTBarZJetsData
from utils import createRunDir, saveUsefulInfo, createSubRNGS, finishRunDir, decode_arg
import numpy as np
import sys
from pathlib import Path

_root = Path(__file__).parent.parent  # Setting root
sys.path.append(str(_root / "jointCode"))
sys.path.append(str(_root / "quantumCode"))

torch.set_default_dtype(torch.float64)

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

#### ---- Load all data (bg+sg) and plot ---- ####
dataloader = TTBarZJetsData(CONFIG)
rng = np.random.default_rng(CONFIG.misc.random_seed)

if CONFIG.training.paradigm == "unsupervised":
    X_train, X_val, X_te, y_te = dataloader.createUnsupervisedDatasets(rng)
elif CONFIG.training.paradigm == "supervised":
    X_train, y_train, X_val, y_val, X_te, y_te = dataloader.createSupervisedDatasets(rng)
else:
    raise ValueError("cfg.training.paradigm should be 'unsupervised' or 'supervised'!")

#### ---- Generate params ---- ####
params, pairs = make_params(CONFIG, rngs)

#### ---- Circuit and data plotting ---- ####
plotter = Plotter(run_path, CONFIG, "quantum")
plotter.plotCirc(params, pairs)
plotter.plot_input_features(X_train, X_te, y_te)

#### ---- Train autoencoder and plot metrics ---- ####
trainer = Trainer(CONFIG, rngs, run_path, infoSave)
if CONFIG.training.paradigm == "unsupervised":
    trainer.train_QAE(X_train, X_val, X_te, y_te, params, pairs)
elif CONFIG.training.paradigm == "supervised":
    trainer.train_CVVC(X_train, y_train, X_val, y_val, X_te, y_te, params, pairs)
else:
    raise ValueError("cfg.training.paradigm should be 'unsupervised' or 'supervised'!")
    
plotter.plotMetrics(["lr"])
plotter.plotLoss()
plotter.plotN()
plotter.plotBSmixing()

# Shouldn't be used for unsupervised!
plotter.plotFinalROC()
plotter.plotFinalLoss()

# #### ---- Rename directory upon finish ---- ####
finishRunDir(run_path)
print(f"Time Taken: {time.perf_counter()-start}")
