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

        self.best_params = None
        self.best_loss = float("inf")
        self.best_epoch = 0
        self.best_auc = 0.0
        self.infoSave = infoSave

        # Initialize CSVs for per-epoch logging and final evaluation
        self.logging_path = os.path.join(self.run_path, "C_logging.csv")
        self.evaluation_path = os.path.join(self.run_path, "C_evaluations.npz")

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
    def train_unsupervised_classical(self, X_train, X_val, X_te, y_te, model):
        device = torch.device("cpu")
        model.to(device)

        # Optimiser
        opt = torch.optim.Adam(model.parameters(), lr=self.cfg.training.lr_start,
                               weight_decay=self.cfg.training.opt_weight_decay)

        # LR Scheduler
        lr_sch = scheduler.Scheduler(self.cfg, opt)

        # Save num params
        no_params = sum(p.numel() for p in model.encoder_layers.parameters()) + \
            sum(p.numel() for p in model.decoder_layers.parameters())
        self.infoSave.add("NumTrainable", no_params)
        self.infoSave.add("NumJetConstituents", self.cfg.circuit.encoder_shape[0] // 3)
        self.infoSave.add("NumFeatures", self.cfg.circuit.encoder_shape[0])

        #### TRAINING ###
        pd.DataFrame(columns=["epoch", "train_loss", "val_loss", "lr",
                     "duration"]).to_csv(self.logging_path, index=False)
        self.save_params(model, "init")
        val_loss = self.classical_unsupervised_validation(model, X_val)
        pd.DataFrame([{"epoch": 0, "train_loss": None, "val_loss": val_loss, "lr": lr_sch.get_lr(
        ), "duration": None}]).to_csv(self.logging_path, mode="a", header=False, index=False)
        ######################################################################################################
        # --Start Training Loop--------------------------------------------------------------------------------
        ######################################################################################################
        bad = 0
        criterion = torch.nn.MSELoss()
        progress_bar = tqdm(range(1, self.cfg.training.epochs + 1), desc="Training")
        for epoch in progress_bar:

            loss_sum = 0.0

            start_epoch = time.perf_counter()
            for x_batch in self.iterate_batches(X_train, self.cfg.training.batch_size, shuffle=True):

                x_batch = torch.tensor(x_batch, dtype=torch.float64, device=device)

                opt.zero_grad()
                output = model(x_batch)             # [batch_size, features]
                # output = torch.sigmoid(output)      # bad for MSE loss?
                loss = criterion(output, x_batch)
                loss.backward()
                opt.step()

                loss_sum += loss.item() * len(x_batch)

            epoch_loss = loss_sum / len(X_train)

            # Validate every X epochs
            if epoch % self.cfg.training.val_every == 0:
                val_loss = self.classical_unsupervised_validation(model, X_val)

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
                        self.infoSave.add(
                            "StopReason", f"Patience limit met at epoch {epoch}"
                        )
                        break

            epoch_duration = time.perf_counter() - start_epoch
            # Log rest every epoch anyway
            pd.DataFrame(
                [
                    {
                        "epoch": epoch,
                        "train_loss": epoch_loss,
                        "val_loss": (
                            val_loss
                            if epoch % self.cfg.training.val_every == 0
                            else None
                        ),
                        "lr": lr_sch.get_lr(),
                        "duration": epoch_duration,
                    }
                ]
            ).to_csv(self.logging_path, mode="a", header=False, index=False)
            progress_bar.set_postfix(tr_loss=f"{epoch_loss:.4f}")
            # --End Per Epoch--------------------

        # Final evaluation
        self.infoSave.add("Best_epoch", self.best_epoch)
        self.infoSave.add("Best_val_loss", self.best_loss)
        best_state = torch.load(self.run_path / "best_params.pt")
        model.load_state_dict(best_state)
        self.classical_unsupervised_test(model, X_te, y_te)
        ######################################################################################################
        # --End Training Loop----------------------------------------------------------------------------------
        ######################################################################################################
        return

    def classical_unsupervised_validation(self, model, X_val):

        device = next(model.parameters()).device
        criterion = torch.nn.MSELoss(reduction="sum")

        all_losses = []

        with torch.no_grad():
            for x_batch in self.iterate_batches(X_val, self.cfg.training.batch_size, shuffle=False):

                x_batch = torch.tensor(x_batch, dtype=torch.float64, device=device)

                output = model(x_batch)
                loss = criterion(output, x_batch)

                all_losses.append(loss.item())

        # MSELoss(reduction="sum") sums over both samples and features
        # so divide by N * n_features to get mean per element
        # For val we need one loss epoch
        N = len(X_val)
        n_features = X_val.shape[1:]
        return sum(all_losses) / (N * int(np.prod(n_features)))  

    def classical_unsupervised_test(self, model, X_te, y_te):
        all_losses = []
        all_y_true = []

        device = next(model.parameters()).device

        with torch.no_grad():
            for x_batch, y_batch in self.iterate_batches(X_te, self.cfg.training.batch_size, shuffle=False, y=y_te):
                x_batch = torch.tensor(x_batch, dtype=torch.float64, device=device)

                output = model(x_batch)

                per_feature_error = (output - x_batch) ** 2
                batch_losses = per_feature_error.mean(dim=1)

                all_losses.append(batch_losses.detach().cpu())
                all_y_true.append(y_batch.detach().cpu())    

        # For test we need one loss per jet
        losses = torch.cat(all_losses).numpy().astype(np.float64)     # (N,)
        y_true = torch.cat(all_y_true).numpy().astype(np.uint8)

        np.savez_compressed(self.evaluation_path, losses=losses, y_true=y_true)

#########################################################################################
# SUPERVISED (need to fix epoch averaging, last batch gets weighted more)
#########################################################################################
    def train_supervised_classical(self, X_train, y_train, X_val, y_val, X_te, y_te, model):
        device = torch.device("cpu")
        model.to(device)

        # Optimiser
        opt = torch.optim.Adam(model.parameters(), lr=self.cfg.training.lr_start,weight_decay=self.cfg.training.opt_weight_decay)

        # LR Scheduler
        lr_sch = scheduler.Scheduler(self.cfg, opt)

        # Loss Function
        loss_fn = nn.BCEWithLogitsLoss(reduction="mean")

        # Save num params
        no_params = sum(
            p.numel() for p in model.classifier.parameters()
            if p.requires_grad
        ) + sum(
            p.numel() for p in model.encoder_layers.parameters()
            if p.requires_grad
        )
        self.infoSave.add("NumTrainable", no_params)
        self.infoSave.add("NumJetConstituents", self.cfg.circuit.encoder_shape[0] // 3)
        self.infoSave.add("NumFeatures", self.cfg.circuit.encoder_shape[0])

        #### TRAINING ###
        pd.DataFrame(columns=["epoch", "train_loss", "val_loss", "val_auc", "lr", "duration"]).to_csv(self.logging_path, index=False)
        self.save_params(model, "init")
        val_loss, val_auc = self.classical_supervised_validation(model, X_val, y_val)
        pd.DataFrame(
            [
                {
                    "epoch": 0,
                    "train_loss": None,
                    "val_loss": val_loss,
                    "val_auc": val_auc,
                    "lr": lr_sch.get_lr(),
                    "duration": None,
                }
            ]
        ).to_csv(self.logging_path, mode="a", header=False, index=False)
        ######################################################################################################
        # --Start Training Loop--------------------------------------------------------------------------------
        ######################################################################################################
        bad = 0
        progress_bar = tqdm(range(1, self.cfg.training.epochs + 1), desc="Training")
        for epoch in progress_bar:

            all_batch_losses = []

            start_epoch = time.perf_counter()
            for x_batch, y_batch in self.iterate_batches(X_train, self.cfg.training.batch_size, shuffle=True,y=y_train):

                x_batch = torch.tensor(x_batch, dtype=torch.float64, device=device)
                y_batch = y_batch.to(dtype=torch.float64, device=device)    # already tensor

                opt.zero_grad()
                output = model(x_batch)             # [batch_size, features]
                loss = loss_fn(output, y_batch)     # 1 scalar per batch
                loss.backward()
                opt.step()

                all_batch_losses.append(loss)

            epoch_loss = torch.mean(torch.stack(all_batch_losses)).item()

            # Validate every X epochs
            if epoch % self.cfg.training.val_every == 0:
                val_loss,val_auc = self.classical_supervised_validation(model, X_val, y_val)

                # LR Scheduling
                lr_sch.step(val_loss)

                # Early stopping
                if val_loss < (self.best_loss - self.cfg.training.min_delta):
                    self.best_loss = val_loss
                    self.best_epoch = epoch
                    self.best_auc = val_auc
                    self.save_params(model, "best")
                    bad = 0
                else:
                    bad += 1
                    if bad >= self.cfg.training.es_patience:
                        self.infoSave.add(
                            "StopReason", f"Patience limit met at epoch {epoch}"
                        )
                        break

            epoch_duration = time.perf_counter() - start_epoch
            # Log rest every epoch anyway
            pd.DataFrame(
                [
                    {
                        "epoch": epoch,
                        "train_loss": epoch_loss,
                        "val_loss": (
                            val_loss
                            if epoch % self.cfg.training.val_every == 0
                            else None
                        ),
                        "val_auc": (
                            val_auc
                            if epoch % self.cfg.training.val_every == 0
                            else None
                        ),
                        "lr": lr_sch.get_lr(),
                        "duration": epoch_duration,
                    }
                ]
            ).to_csv(self.logging_path, mode="a", header=False, index=False)
            progress_bar.set_postfix(tr_loss=f"{epoch_loss:.4f}")
            # --End Per Epoch--------------------

        # Final evaluation
        self.infoSave.add("Best_epoch", self.best_epoch)
        self.infoSave.add("Best_val_loss", self.best_loss)
        self.infoSave.add("Best_val_auc", self.best_auc)
        best_state = torch.load(self.run_path / "best_params.pt")
        model.load_state_dict(best_state)
        self.classical_supervised_test(model, X_te, y_te)
        ######################################################################################################
        # --End Training Loop----------------------------------------------------------------------------------
        ######################################################################################################
        return

    def classical_supervised_validation(self, model, X_val, y_val):

        device = next(model.parameters()).device

        loss_fn = nn.BCEWithLogitsLoss(reduction="sum") # allows manual averaging per epoch, not batch-size related

        all_losses = []
        all_logits = []
        all_y_true = []

        with torch.no_grad():
            for x_batch, y_batch in self.iterate_batches(X_val, self.cfg.training.batch_size, shuffle=False, y=y_val):

                x_batch = torch.tensor(x_batch, dtype=torch.float64, device=device)
                y_batch = y_batch.to(dtype=torch.float64, device=device) 

                output = model(x_batch)
                loss = loss_fn(output, y_batch)

                all_losses.append(loss.item())
                all_logits.append(output.detach().cpu())
                all_y_true.append(y_batch.detach().cpu())

        val_loss = sum(all_losses) / len(y_val)

        logits_all = torch.cat(all_logits).numpy().astype(np.float64)
        y_true_all = torch.cat(all_y_true).numpy().astype(np.uint8)
        val_auc = roc_auc_score(y_true_all, logits_all)

        return val_loss, val_auc

    def classical_supervised_test(self, model, X_te, y_te):
        all_scores = []
        all_y_true = []

        device = next(model.parameters()).device

        with torch.no_grad():
            for x_batch, y_batch in self.iterate_batches(X_te, self.cfg.training.batch_size, shuffle=False, y=y_te):
                x_batch = torch.tensor(x_batch, dtype=torch.float64, device=device)
                output = model(x_batch).squeeze(-1)

                all_scores.append(output.detach().cpu())
                all_y_true.append(y_batch.detach().cpu())

        scores = torch.cat(all_scores).numpy().astype(np.float64)
        y_true = torch.cat(all_y_true).numpy().astype(np.uint8)

        np.savez_compressed(self.evaluation_path, losses=scores, y_true=y_true)
