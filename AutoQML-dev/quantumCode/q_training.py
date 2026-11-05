import time
import torch
from q_circ import QAE_loss,CVVC_loss
import pandas as pd
import numpy as np
import os
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
import sys
from pathlib import Path
_root = Path(__file__).parent.parent  # Setting root
sys.path.append(str(_root / "jointCode"))
sys.path.append(str(_root / "quantumCode"))
torch.set_default_dtype(torch.float64)
import scheduler
from q_params import clamp_params


class Trainer:

    def __init__(self, cfg, rngs, run_path, infoSave):
        self.cfg = cfg
        self.rngs = rngs
        self.run_path = Path(run_path)
        self.all_wires = list(range(cfg.circuit.wires))

        self.best_loss = float("inf")
        self.best_epoch = 0
        self.best_F = 0.0           # tracked for QAE - as final loss values are on diff scales, and no auc available
        self.best_auc = 0.0         # tracked for CVVC - as can use val_auc
        self.infoSave = infoSave

        # Initialize CSVs for per-epoch logging and final evaluation
        self.logging_path = os.path.join(self.run_path, "Q_logging.csv")
        self.evaluation_path = os.path.join(self.run_path, "Q_evaluations.npz")

        self.masters_rng = np.random.default_rng(cfg.misc.random_seed + 101)

        return

    def save_params(self, params, pairs, name):

        # Count trainable parameters
        no_params = 0
        for v in params.values():
            if isinstance(v, dict):
                for vv in v.values():
                    if isinstance(vv, torch.nn.Parameter) and vv.requires_grad:
                        no_params += vv.numel()
            else:
                if isinstance(v, torch.nn.Parameter) and v.requires_grad:
                    no_params += v.numel()

        # A) Torch format - for loading back
        torch.save({"params": params, "pairs": pairs, },
                   self.run_path / f"{name}_params.pt",)

        # B) Human readable
        with open(self.run_path / f"{name}_params.txt", "w") as f:
            f.write("=== params ===\n")
            for k, v in params.items():
                if isinstance(v, dict):
                    f.write(f"  {k}:\n")
                    for kk, vv in v.items():
                        data = vv.data if isinstance(
                            vv, torch.nn.Parameter) else vv
                        f.write(f"    {kk}: {data.squeeze().tolist()}\n")
                else:
                    data = v.data if isinstance(v, torch.nn.Parameter) else v
                    f.write(f"  {k}: {data.squeeze().tolist()}\n")
            f.write("\n")
        
        return no_params

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

    def flatten_params(self, d):  # for optimiser
        for v in d.values():
            if isinstance(v, dict):
                yield from self.flatten_params(v)
            else:
                yield v

    ## TEST
    def shuffle_wires(self, X, rng):
        perm = np.argsort(rng.random(X.shape[:2]), axis=1)   # independent permutation per event
        return X[np.arange(len(X))[:, None], perm]
#########################################################################################
# UNSUPERVISED - Quantum Autoencoder
#########################################################################################

    def train_QAE(self, X_train, X_val, X_te, y_te, params, pairs):
        """
        Method for training QML autoendoers, using flexible "forward" method.
        """

        # Clamp params here to ensure any initial params are in box before first forward pass
        clamp_params(params)

        all_params = list(self.flatten_params(params))

        opt = torch.optim.Adam(all_params, lr=self.cfg.training.lr_start,
                               weight_decay=self.cfg.training.opt_weight_decay)

        # LR Scheduler
        lr_sch = scheduler.Scheduler(self.cfg, opt)

        if self.cfg.training.shuffle_particles:
            wire_rng_val = np.random.default_rng(self.cfg.misc.random_seed + 200)
            wire_rng_te = np.random.default_rng(self.cfg.misc.random_seed + 201)
            X_val = self.shuffle_wires(X_val, wire_rng_val)
            X_te = self.shuffle_wires(X_te, wire_rng_te)

        # Save initial
        pd.DataFrame(columns=["epoch", "train_loss", "val_loss", "val_F", "lr", "meanN", "maxN", "epoch_duration"]).to_csv(self.logging_path, index=False)
        n_params = self.save_params(params, pairs, "init")
        val_loss,val_F = self.QAE_validation(X_val, params, pairs)
        pd.DataFrame([{"epoch": 0, "train_loss": None, "val_loss": val_loss, "val_F":val_F, "lr": lr_sch.get_lr(),"meanN": None, 
                        "maxN": None, "duration": None}]).to_csv(self.logging_path, mode="a", header=False, index=False)
        # Write baseline best
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            self.best_F = val_F
            self.best_epoch = 0
            self.save_params(params, pairs, "best")
        ######################################################################################################
        # --Start Training Loop--------------------------------------------------------------------------------
        ######################################################################################################
        bad = 0
        progress_bar = tqdm(range(1, self.cfg.training.epochs + 1), desc="Training")

        if self.cfg.training.shuffle_particles:
            wire_rng_train = np.random.default_rng(self.cfg.misc.random_seed + 202)

        for epoch in progress_bar:

            if self.cfg.training.shuffle_particles:
                X_epoch = self.shuffle_wires(X_train, wire_rng_train)
            
            loss_sum = 0.0
            n_sum = 0.0 

            ######################################
            # --Start Per Epoch--------------------
            ######################################
            start_epoch = time.perf_counter()
            epoch_n_max = float('-inf')
            
            for x_batch in self.iterate_batches(X_epoch if self.cfg.training.shuffle_particles else X_train, 
                                                self.cfg.training.batch_size, 
                                                shuffle=True):

                ############################
                # --Start Per Batch----------
                ############################
                losses, _, n = QAE_loss(self.cfg, x_batch, params, pairs)
                loss = losses.mean()

                # backward pass / update
                opt.zero_grad()
                loss.backward()
                opt.step()

                # clamp params to box before next forward pass
                clamp_params(params)

                with torch.no_grad():
                    loss_sum += loss.item() * len(x_batch)
                    n_sum += n.mean().item() * len(x_batch) 
                    epoch_n_max = max(epoch_n_max, n.max().item())

                ############################
                # --End Per Batch----------
                ############################

            epoch_loss = loss_sum / len(X_train)
            epoch_n_mean = n_sum / len(X_train)

            # Validate every X epochs
            if epoch % self.cfg.training.val_every == 0:
                val_loss, val_F = self.QAE_validation(X_val, params, pairs)

                # LR Scheduling
                lr_sch.step(val_loss)

                # Early stopping
                if val_loss < (self.best_loss - self.cfg.training.min_delta):
                    self.best_loss = val_loss
                    self.best_F = val_F 
                    self.best_epoch = epoch
                    self.save_params(params, pairs, "best")
                    bad = 0
                else:
                    bad += 1
                    if bad >= self.cfg.training.es_patience:
                        self.infoSave.add("StopReason", f"Patience limit met at epoch {epoch}")
                        break

            epoch_duration = time.perf_counter() - start_epoch
            # Log rest every epoch anyway
            pd.DataFrame([{"epoch": epoch, 
               "train_loss": epoch_loss, 
               "val_loss": (val_loss if epoch % self.cfg.training.val_every == 0 else None),
               "val_F": (val_F if epoch % self.cfg.training.val_every == 0 else None),
               "lr": lr_sch.get_lr(), 
               "meanN": epoch_n_mean, 
               "maxN": epoch_n_max,
               "duration": epoch_duration}]).to_csv(self.logging_path, mode="a", header=False, index=False)
            progress_bar.set_postfix(tr_loss=f"{epoch_loss:.4f}")
            ######################################
            # --End Per Epoch--------------------
            ######################################

        # Final evaluation
        self.infoSave.add("Best_epoch", self.best_epoch)
        self.infoSave.add("Best_val_loss", self.best_loss)
        self.infoSave.add("Best_val_F", self.best_F)
        self.infoSave.add("NumTrainable", n_params)
        best = torch.load(self.run_path / "best_params.pt")
        self.QAE_test(X_te, y_te, best["params"], best["pairs"])
        ####################################################################################################
        # --End Training loop--------------------------------------------------------------------------------
        ####################################################################################################
        return

    def QAE_validation(self, X_val, params, pairs):
        all_losses = []
        all_F = []

        with torch.no_grad():
            for x_batch in self.iterate_batches(X_val, self.cfg.training.batch_size, shuffle=False):
                losses, F, _ = QAE_loss(self.cfg, x_batch, params, pairs)
                all_losses.append(losses.sum().item())
                all_F.append(F.sum().item()) 

        N = len(X_val)
        return sum(all_losses) / N, sum(all_F) / N      # weighted equally over all data, not just per batch

    def QAE_test(self, X_te, y_te, params, pairs):
        all_losses = []
        all_y_true = []

        with torch.no_grad():
            for x_batch, y_batch in self.iterate_batches(X_te, self.cfg.training.batch_size, shuffle=False, y=y_te):
                batch_losses,_, _ = QAE_loss(self.cfg, x_batch, params, pairs)
                all_losses.append(batch_losses.detach().cpu())
                all_y_true.append(y_batch.detach().cpu()) 

        # Flatten across batches — safe regardless of whether last batch is smaller
        losses = torch.cat(all_losses).numpy().astype(np.float64)   # (N,)
        y_true = torch.cat(all_y_true).numpy().astype(np.uint8)  

        np.savez_compressed(self.evaluation_path, losses=losses, y_true=y_true)

#########################################################################################
# SUPERVISED - Continuous-variable Variational Classifier (need to fix loss averaging!)
#########################################################################################

    def train_CVVC(self, X_train, y_train, X_val, y_val, X_te, y_te, params, pairs):
        """
        Method for training QML autoendoers, using flexibile "forward" method.
        """

        # Clamp params here to ensure any initial params are in box before first forward pass
        clamp_params(params)

        all_params = list(self.flatten_params(params))

        opt = torch.optim.Adam(all_params, lr=self.cfg.training.lr_start,
                               weight_decay=self.cfg.training.opt_weight_decay)

        # LR Scheduler
        lr_sch = scheduler.Scheduler(self.cfg, opt)

        # Save initial
        pd.DataFrame(columns=["epoch", "train_loss", "val_loss", "val_auc","lr", "meanN", "maxN", "epoch_duration"]).to_csv(self.logging_path, index=False)
        n_params = self.save_params(params, pairs, "init")
        val_loss, val_auc = self.CVVC_validation(X_val, y_val, params, pairs)
        pd.DataFrame([{"epoch": 0, "train_loss": None, "val_loss": val_loss, "val_auc": val_auc, "lr": lr_sch.get_lr(),
                     "meanN": None, "maxN": None, "duration": None}]).to_csv(self.logging_path, mode="a", header=False, index=False)
        # Write baseline best
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            self.best_epoch = 0
            self.best_auc = val_auc
            self.save_params(params, pairs, "best")
        ######################################################################################################
        # --Start Training Loop--------------------------------------------------------------------------------
        ######################################################################################################
        criterion = torch.nn.BCELoss(reduction="mean") if self.cfg.training.loss_fn == "F" else torch.nn.BCEWithLogitsLoss(reduction="mean")
        bad = 0
        progress_bar = tqdm(range(1, self.cfg.training.epochs + 1), desc="Training")
        for epoch in progress_bar:

            all_batch_losses = []
            all_batch_n_means = []
            ######################################
            # --Start Per Epoch--------------------
            ######################################
            start_epoch = time.perf_counter()
            epoch_n_max = float('-inf')

            for x_batch, y_batch in self.iterate_batches(X_train, self.cfg.training.batch_size, shuffle=True, y=y_train):
                ############################
                # --Start Per Batch----------
                ############################
                logits, n = CVVC_loss(self.cfg, x_batch, params, pairs)

                loss = criterion(logits, y_batch.double())

                # backward pass / update
                opt.zero_grad()
                loss.backward()
                opt.step()

                # clamp params to box before next forward pass
                clamp_params(params)

                with torch.no_grad():
                    all_batch_losses.append(loss.item())
                    all_batch_n_means.append(n.mean().item())
                    epoch_n_max = max(epoch_n_max, n.max().item())

                ############################
                # --End Per Batch----------
                ############################
            epoch_loss = sum(all_batch_losses) / len(all_batch_losses)
            epoch_n_mean = sum(all_batch_n_means) / len(all_batch_n_means)

            # Validate every X epochs
            if epoch % self.cfg.training.val_every == 0:
                val_loss, val_auc = self.CVVC_validation(X_val, y_val, params, pairs)

                # LR Scheduling
                lr_sch.step(val_loss)

                # Early stopping
                if val_loss < (self.best_loss - self.cfg.training.min_delta):
                    self.best_loss = val_loss
                    self.best_epoch = epoch
                    self.best_auc = val_auc
                    self.save_params(params, pairs, "best")
                    bad = 0
                else:
                    bad += 1
                    if bad >= self.cfg.training.es_patience:
                        self.infoSave.add(
                            "StopReason", f"Patience limit met at epoch {epoch}")
                        break

            epoch_duration = time.perf_counter() - start_epoch
            # Log rest every epoch anyway
            pd.DataFrame([{"epoch": epoch, "train_loss": epoch_loss, 
            "val_loss": (val_loss if epoch % self.cfg.training.val_every == 0 else None),
            "val_auc": (val_auc if epoch % self.cfg.training.val_every == 0 else None),
                           "lr": lr_sch.get_lr(), "meanN": epoch_n_mean, "maxN": epoch_n_max, "duration": epoch_duration}]).to_csv(self.logging_path, mode="a", header=False, index=False)
            progress_bar.set_postfix(tr_loss=f"{epoch_loss:.4f}")
            ######################################
            # --End Per Epoch--------------------
            ######################################

        # Final evaluation
        self.infoSave.add("Best_epoch", self.best_epoch)
        self.infoSave.add("Best_val_loss", self.best_loss)
        self.infoSave.add("Best_val_auc", self.best_auc)
        self.infoSave.add("NumTrainable", n_params)
        best = torch.load(self.run_path / "best_params.pt")
        self.CVVC_test(X_te, y_te, best["params"], best["pairs"])
        ####################################################################################################
        # --End Training loop--------------------------------------------------------------------------------
        ####################################################################################################
        return

    def CVVC_validation(self, X_val, y_val, params, pairs):
        all_losses = []
        all_logits = []
        all_y_true = []

        criterion = torch.nn.BCELoss(reduction="sum") if self.cfg.training.loss_fn == "F" else torch.nn.BCEWithLogitsLoss(reduction="sum")

        with torch.no_grad():
            for x_batch, y_batch in self.iterate_batches(X_val, self.cfg.training.batch_size, shuffle=False, y=y_val):
                logits, _ = CVVC_loss(self.cfg, x_batch, params, pairs)
                loss = criterion(logits, y_batch.double())

                all_losses.append(loss.item())
                all_logits.append(logits.detach().cpu())
                all_y_true.append(y_batch.detach().cpu())

        val_loss = sum(all_losses) / len(y_val)

        logits_all = torch.cat(all_logits).numpy().astype(np.float64)
        y_true_all = np.concatenate([y.numpy() for y in all_y_true]).astype(np.uint8)
        val_auc = roc_auc_score(y_true_all, logits_all)

        return val_loss, val_auc

    def CVVC_test(self, X_te, y_te, params, pairs):
        all_logits = []
        all_y_true = []

        with torch.no_grad():
            for x_batch, y_batch in self.iterate_batches(X_te, self.cfg.training.batch_size, shuffle=False, y=y_te):
                logits, _ = CVVC_loss(self.cfg, x_batch, params, pairs)

                all_logits.append(logits.detach().cpu())
                all_y_true.append(y_batch.detach().cpu())

        # Flatten across batches
        logits = torch.cat(all_logits).numpy().astype(np.float64)  # (N,)
        y_true = torch.cat(all_y_true).numpy().astype(np.uint8)  # (N,)

        np.savez_compressed(self.evaluation_path, losses=logits, y_true=y_true)
