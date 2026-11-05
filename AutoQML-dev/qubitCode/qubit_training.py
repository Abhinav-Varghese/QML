from tqdm import tqdm
import numpy as np
import os
import pandas as pd
import time
from sklearn.metrics import roc_curve
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
import copy
import sys
from pathlib import Path
_root = Path(__file__).parent.parent  # Setting root
sys.path.append(str(_root / "jointCode"))
import scheduler


class Trainer():
    """
    General training code for QML autoencoder.
    Train function takes in y_train (despite being unused) as batch-iteration code was designed to take x and y (for validation later)
    Training uses batch_averaged loss, whereas evaluation uses a loss per datapoint
    """

    def __init__(self, cfg, rngs, run_path, infoSave):
        self.cfg = cfg
        self.rngs = rngs
        self.run_path = Path(run_path)

        self.best_loss = np.inf
        self.best_epoch = 0
        self.infoSave = infoSave

        # Initialize CSVs for per-epoch logging and final evaluation
        self.logging_path = os.path.join(self.run_path, "QB_logging.csv")
        self.evaluation_path = os.path.join(self.run_path, "QB_evaluations.npz")

        self.masters_rng = np.random.default_rng(cfg.misc.random_seed + 101)
        return

    def save_params(self, model, name):
        # A) Torch format - for loading back
        torch.save(model.state_dict(), self.run_path / f"{name}_params.pt")

        # B) Human readable
        with open(self.run_path / f"{name}_params.txt", "w") as f:
            for k, v in model.state_dict().items():
                f.write(f"{k}: {v.squeeze().tolist()}\n")

    def iterate_batches(self, X, batch_size, shuffle, y=None):
        """
        Generator that yields batches of X (and y if provided).
        Takes shuffle argument:
        - Train -> should shuffle
        - Val/test -> no need to shuffle
        """
        N = len(X)
        indices = np.arange(N)
        if shuffle:
            self.masters_rng.shuffle(indices)

        for start in range(0, N, batch_size):
            batch_idx = indices[start: start + batch_size]
            if y is None:
                yield X[batch_idx]
            else:
                yield X[batch_idx], y[batch_idx]

#########################################################################################
# UNSUPERVISED
#########################################################################################
    def train_unsupervised_qubit(self, X_train, X_val, X_te, y_te, model):

        device = torch.device("cpu")
        model.to(device)

        trainable = [p for p in model.parameters() if p.requires_grad]
        no_params = sum(p.numel() for p in trainable)
        self.infoSave.add("NumTrainable", no_params)
        self.infoSave.add("NumJetConstituents", self.cfg.circuit.constituents)

        # Control/baseline: nothing to train, just evaluate the fixed circuit
        if no_params == 0:
            self.save_params(model, "init")
            self.save_params(model, "best")      # final eval below expects best_params.pt
            val_loss = self.qubit_unsupervised_validation(model, X_val)
            pd.DataFrame([{"epoch": 0, "train_loss": None, "val_loss": val_loss,
                        "lr": None, "duration": None}]).to_csv(self.logging_path, index=False)
            self.infoSave.add("Best_epoch", 0)
            self.infoSave.add("Best_val_loss", val_loss)
            self.infoSave.add("StopReason", "No trainable parameters (baseline)")
            self.qubit_unsupervised_test(model, X_te, y_te)
            return

        # Optimiser
        opt = torch.optim.Adam(model.parameters(), lr=self.cfg.training.lr_start, weight_decay=self.cfg.training.opt_weight_decay)

        # LR Scheduler
        lr_sch = scheduler.Scheduler(self.cfg, opt)

        #### Saving initial ###
        pd.DataFrame(columns=["epoch", "train_loss", "val_loss", "lr", "duration"]).to_csv(self.logging_path, index=False)
        self.save_params(model, "init")
        val_loss = self.qubit_unsupervised_validation(model, X_val)
        pd.DataFrame([{"epoch": 0, "train_loss": None, "val_loss": val_loss, "lr": lr_sch.get_lr(), "duration": None}]).to_csv(self.logging_path, mode="a", header=False, index=False)

        # Write baseline best
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            self.best_epoch = 0
            self.save_params(model, "best")

        ######################################################################################################
        # --Start Training Loop--------------------------------------------------------------------------------
        ######################################################################################################
        bad = 0
        progress_bar = tqdm(range(1, self.cfg.training.epochs + 1), desc="Training")
        for epoch in progress_bar:
            model.train()
            total_loss = 0.0

            ######################################
            # --Start Per Epoch--------------------
            ######################################
            start_epoch = time.perf_counter()

            for x_batch in self.iterate_batches(X_train, self.cfg.training.batch_size, shuffle=True):
                ############################
                # --Start Per Batch----------
                ############################
                x_batch = torch.as_tensor(x_batch,dtype=torch.float64,device=device)

                batch_Fs = model(x_batch)           # SWAP test expval(Z) = F
                batch_loss = (1-batch_Fs).mean()     # average loss over batch

                # backward pass / update
                opt.zero_grad()
                batch_loss.backward()
                opt.step()

                with torch.no_grad():
                    total_loss += (1 - batch_Fs).sum().item()

                ############################
                # --End Per Batch----------
                ############################

            epoch_loss = total_loss / len(X_train)

            # Validate every X epochs
            if epoch % self.cfg.training.val_every == 0:
                val_loss = self.qubit_unsupervised_validation(model, X_val)

                # LR Scheduling
                lr_sch.step(val_loss)

                # Early stopping
                if val_loss < (self.best_loss - self.cfg.training.min_delta):
                    self.best_loss = val_loss
                    self.best_epoch = epoch
                    self.save_params(model, "best")
                    bad = 0
                else:
                    bad += 1
                    if bad >= self.cfg.training.es_patience:
                        self.infoSave.add("StopReason", f"Patience limit met at epoch {epoch}")
                        break

            epoch_duration = time.perf_counter() - start_epoch
            # Log rest every epoch anyway
            pd.DataFrame([{"epoch": epoch, "train_loss": epoch_loss, "val_loss": (val_loss if epoch % self.cfg.training.val_every == 0 else None),
               "lr": lr_sch.get_lr(), "duration": epoch_duration}]).to_csv(self.logging_path, mode="a", header=False, index=False)
            progress_bar.set_postfix(tr_loss=f"{epoch_loss:.4f}")
            ######################################
            # --End Per Epoch--------------------
            ######################################

        # Final evaluation
        self.infoSave.add("Best_epoch", self.best_epoch)
        self.infoSave.add("Best_val_loss", self.best_loss)
        best = torch.load(self.run_path / "best_params.pt", map_location=device)
        model.load_state_dict(best)
        self.qubit_unsupervised_test(model, X_te, y_te)
        ####################################################################################################
        # --End Training loop-------------------------------------------------------------------------------
        ####################################################################################################
        return

    def qubit_unsupervised_validation(self, model, X_val):
        device = torch.device("cpu")
        model.to(device)

        total_loss = 0.0
        N = len(X_val)

        model.eval()

        with torch.no_grad():
            for x_batch in self.iterate_batches(X_val, self.cfg.training.batch_size, shuffle=False):
                x_batch = torch.as_tensor(x_batch, dtype=torch.float64, device=device)
                
                batch_Fs = model(x_batch)          # one F per jet
                batch_losses = 1 - batch_Fs        # one loss per jet

                total_loss += batch_losses.sum().item()

        return total_loss / N 

    def qubit_unsupervised_test(self, model, X_te, y_te):
        all_losses = []
        all_y_true = []

        device = torch.device("cpu")
        model.eval()

        with torch.no_grad():
            for x_batch, y_batch in self.iterate_batches(X_te, self.cfg.training.batch_size, shuffle=False, y=y_te):

                x_batch = torch.as_tensor(x_batch, dtype=torch.float64, device=device)

                batch_Fs = model(x_batch)       # one F per jet
                batch_losses = 1 - batch_Fs     # one anomaly score per jet

                all_losses.append(batch_losses.cpu())
                all_y_true.append(torch.as_tensor(y_batch).cpu())

        # One loss and one label for every test jet
        losses = torch.cat(all_losses).numpy().astype(np.float64)
        y_true = torch.cat(all_y_true).numpy().astype(np.uint8)

        np.savez_compressed(self.evaluation_path, losses=losses, y_true=y_true)