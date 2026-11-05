################################################################
import time
import torch
from omegaconf import OmegaConf
from c_training import Trainer
from c_config import CONFIG
from data_loader import TTBarZJetsData
from utils import createRunDir, saveUsefulInfo, createSubRNGS, finishRunDir, decode_arg
from c_circ import buildClassical
# plotting must be imported before c_circ (torch) to avoid libstdc++ conflict
from plotting import Plotter
import numpy as np
import sys
from pathlib import Path
_root = Path(__file__).parent.parent  # Setting root
sys.path.append(str(_root / "jointCode"))
sys.path.append(str(_root / "classicalCode"))
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

#### ---- Create classical model ---- ####
model = buildClassical(CONFIG, rngs)
model.param_generator()

#### ---- Circuit plotting ---- ####
plotter = Plotter(run_path, CONFIG, "classical")
plotter.plotClassical()

#### ---- Load all data (bg+sg) and plot ---- ####
dataloader = TTBarZJetsData(CONFIG)
rng = np.random.default_rng(CONFIG.misc.random_seed)

if CONFIG.training.paradigm == "unsupervised":
    X_train, X_val, X_te, y_te = dataloader.createUnsupervisedDatasets(rng)
elif CONFIG.training.paradigm == "supervised":
    X_train, y_train, X_val, y_val, X_te, y_te = dataloader.createSupervisedDatasets(rng)
else:
    raise ValueError("cfg.training.paradigm should be 'unsupervised' or 'supervised'!")

# Data Plotting
(plotter.plot_input_features(X_train, X_te, y_te))

# Flatten data to match classical 3N encoding
X_train = X_train.reshape(X_train.shape[0], -1)
X_val = X_val.reshape(X_val.shape[0], -1)
X_te = X_te.reshape(X_te.shape[0], -1)

#### ---- Train autoencoder and plot metrics ---- ####
trainer = Trainer(CONFIG, rngs, run_path, infoSave)
if CONFIG.training.paradigm == "unsupervised":
    trainer.train_unsupervised_classical(X_train, X_val, X_te, y_te, model)
elif CONFIG.training.paradigm == "supervised":
    trainer.train_supervised_classical(X_train, y_train, X_val, y_val, X_te, y_te, model)
else:
    raise ValueError("Paradigm should be 'unsupervised' or 'supervised'!")
    
plotter.plotMetrics(["lr"])
plotter.plotLoss()

# Shouldn't be used for unsupervised!
plotter.plotFinalROC()
plotter.plotFinalLoss()

#### ---- Rename directory upon finish ---- ####
finishRunDir(run_path)
print(f"Time Taken: {time.perf_counter()-start}")
