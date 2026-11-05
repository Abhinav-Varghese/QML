import numpy as np
import h5py
import os
import torch
import gc
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
#####################################


class TTBarZJetsData:
    """
    Process:
    - Init loads the data (load_data) and removes particles over N_CONSTITUENTS.
    - Init also filters out jets with padded particles < N_CONSTITUENTS
      (so every jet now has exactly N_CONSTITUENTS real particles)
    - CreateUnsupervisedDatasets produces train/val/test split indices, by shuffling all data (fixed seed) and carving val and 
      test out from the end. This means val and test are fixed, regardless of training_samples requested. Then, training_samples 
      are taken randomly (seeded) from the remaining bg data. The remaining bg data pool is also returned.
    - These indices are then used to create train/val/test datasets.
    - These datasets are then normalised and clipped. Using the remaining bg data pool, 99th percentiles are found for |eta| and 
      |phi|, whilst the maximum pt_frac is found. Eta and Phi are then clipped [-a, a] / a to [-1,1] and ([-a, a] / a) * pi
      respectively, whilst pt_frac is simply sqrt(pt_frac / pt_frac(max)).
      
      The percentiles and max are found per constituent, then the largest is taken across all constituents. This is because each particle
      has its own width, so the eta 99th percentile of particle 1 is different to particle 7, which is different to 99th percentile over
      all particles. If we used the all-particles 99th percentile, it would cut lots of the widest-range particle. However this means 
      the narrow-range particles don't use much of the range. We could use 99 percentile of all particles, which would clip lots of the
      wide-range particles but make general best use of the range.
    """

    # This is the number of particles kept in each jet (others removed)
    # This is also the number of particles below which jets with padded particles are removed
    N_CONSTITUENTS = 8
    R_JET = 0.8

    def __init__(self, cfg):
        self.cfg = cfg
        dataset_path = os.path.join("/vols/cms/lgc124/QMLoutputs/TTBar_+ZJetsToNuNU_flat_merged.h5")
        self.const, self.truth, self.jet_pt = self.load_data(dataset_path)

        self._filter_low_multiplicity_jets()

        # The number of constituents needed
        if hasattr(cfg.circuit, "wires"):
            self.n = cfg.circuit.wires
            self.paradigm = "Qumode"

        elif hasattr(cfg.circuit, "encoder_shape"):
            self.n = cfg.circuit.encoder_shape[0] // 3
            self.paradigm = "Classical"

        else:
            self.n = cfg.circuit.constituents
            self.paradigm = "Qubit"
            if (cfg.circuit.constituents * 3) > (2**cfg.circuit.qubits):
                raise ValueError("Incorrect number of qubits/constituents!")
            
        if self.n > self.N_CONSTITUENTS:
            raise ValueError(f"Requested {self.n} constituents but data is cut to {self.N_CONSTITUENTS}")

        self.n_train = cfg.data.train_num_samples
        self.n_val = cfg.data.val_num_samples
        self.n_test = cfg.data.test_num_samples

    def load_data(self, path):
        """
        Loads and removes particles > N_CONSTITUENTS
        (Afterwards, all jets have N_CONSTITUENTS particles)
        """
        with h5py.File(path, "r") as f:
            const = f["jetConstituentsList"][...].astype(np.float64)
            truth = np.rint(f["truth_labels"][...]).astype(np.int64)
            names = [
                x.decode() if isinstance(x, (bytes, np.bytes_)) else str(x)
                for x in f["particleFeatureNames"][()]
            ]

        # Resolve ALL feature indices dynamically -- do not assume a fixed
        # column order anywhere else in the class.
        self.eta_idx = names.index("eta")
        self.phi_idx = names.index("phi")
        self.pt_idx = names.index("pt")

        print("\nInitial Feature Limits")
        for name, idx in [("eta", self.eta_idx), ("phi", self.phi_idx), ("pt", self.pt_idx)]:
            values = const[:, :, idx]
            finite = values[np.isfinite(values)]

            if finite.size:
                print(
                    f"{name}: "
                    f"min={finite.min():.6f}, "
                    f"max={finite.max():.6f}, "
                    f"p1={np.percentile(finite, 1):.6f}, "
                    f"p99={np.percentile(finite, 99):.6f}"
                )
            else:
                print(f"{name}: no finite values")

        # Sort each jet by pt (highest first), keep top N, drop the rest
        pt_all = const[:, :, self.pt_idx]
        key = np.where(np.isfinite(pt_all) & (pt_all > 0), pt_all, -np.inf)     # finite/positive pt takes pt value, rest is given -inf (to remove from top)
        order = np.argsort(-key, axis=1)[:, :self.N_CONSTITUENTS]        # keeps the top 8 indices
        const = np.take_along_axis(const, order[:, :, None], axis=1)            # uses the indices to actually cut out rest [jets, 8, features]

        ### FROM HERE ON, ALL JETS HAVE ONLY 8 PARTICLES
        
        # Find the total pt per jet (for later creation of pt_frac)
        constituent_pt = const[:, :, self.pt_idx]
        valid = np.isfinite(constituent_pt) & (constituent_pt > 0)
        jet_pt = np.where(valid, constituent_pt, 0.0).sum(axis=1)

        return const, truth, jet_pt

    def _filter_low_multiplicity_jets(self):
        """
        Whereas load_data removes constituents, this function removes full jets with < N_CONSTITUENTS real (non-padded) particles
        """
        pt = self.const[:, :, self.pt_idx]
        eta = self.const[:, :, self.eta_idx]
        phi = self.const[:, :, self.phi_idx]
        valid = np.isfinite(pt) & (pt > 0) & np.isfinite(eta) & np.isfinite(phi)
        n_valid = valid.sum(axis=1)

        keep = n_valid >= self.N_CONSTITUENTS
        n_dropped = int((~keep).sum())
        n_total = len(keep)
        print(
            f"[TTBarZJetsData] Dropping {n_dropped}/{n_total} jets "
            f"({n_dropped / n_total * 100:.3f}%) with < {self.N_CONSTITUENTS} "
            f"valid constituents."
        )

        self.const = self.const[keep]
        self.truth = self.truth[keep]
        self.jet_pt = self.jet_pt[keep]

    def build_X(self, const, jet_pt, indices, k):
        """
        Builds [jets, k, 3] of (eta, pt_frac, phi) from the top-k pt-sorted constituents.
        pt_frac = constituent pt / jet pt (jet pt = sum over the kept constituents).
        """
        c = const[indices, :k]                                   # [jets, k, features], already pt-sorted
        eta     = c[:, :, self.eta_idx]
        pt_frac = c[:, :, self.pt_idx] / (jet_pt[indices, None] + 1e-12)
        phi     = c[:, :, self.phi_idx]
        return np.stack([eta, pt_frac, phi], axis=-1)

    def global_shuffle(self, indices, seed):
        rng = np.random.default_rng(seed)
        shuffled = indices.copy()
        rng.shuffle(shuffled)
        return shuffled
    
    def split_train_val_test(self, rng, unsupervised):
        # ------------------------------------------------------------------
        # Fixed val/test carve-out, taken from the END of the shuffled pool
        # so they never move regardless of train_num_samples. Train is then
        # sliced from the FRONT of whatever remains, up to self.n_train.
        # ------------------------------------------------------------------
        bkg = np.where(self.truth == self.truth.min())[0]
        sig = np.where(self.truth == self.truth.max())[0]
        
        # Fixed global dataset shuffle (decoupled from seed)
        global_seed = 20

        # We first shuffle globally within bkg and sig to prevent in-class ordering
        # This is seed independent, such that splitting off test/val are fixed across seeds
        bkg = self.global_shuffle(bkg, global_seed)
        sig = self.global_shuffle(sig, global_seed)
        
        if unsupervised:
            n_test_each = self.n_test // 2   # test is 50/50 bg/sig
            n_val = self.n_val               # val is bg-only

            te_bkg = bkg[-n_test_each:]                         # Deterministic test-bg
            te_sig = sig[-n_test_each:]                         # Deterministic test-sg
            vl_bkg = bkg[-(n_test_each + n_val):-n_test_each]   # Deterministic val-bg

            bkg_pool = bkg[:-(n_test_each + n_val)]
            
            if self.n_train > len(bkg_pool):
                raise ValueError(f"Requested train_num_samples={self.n_train} background "
                f"jets but only {len(bkg_pool)} remain after val/test carve-out.")

            # Take random/seeded training data
            tr_bkg = rng.permutation(bkg_pool)[:self.n_train]

            tr_idx = tr_bkg
            vl_idx = vl_bkg
            te_idx = np.concatenate([te_bkg, te_sig])

            # We secondly shuffle test as is concatenated bkg+sig so needs shuffling again
            # This is seed-independent, data is already fixed, just to intertwine bg/sg
            te_idx = self.global_shuffle(np.concatenate([te_bkg, te_sig]), global_seed)

            return tr_idx, vl_idx, te_idx

        else:
            n_val_each = self.n_val // 2
            n_test_each = self.n_test // 2
            n_train_each = self.n_train // 2

            # Deterministic
            te_bkg = bkg[-n_test_each:]
            te_sig = sig[-n_test_each:]
            vl_bkg = bkg[-(n_test_each + n_val_each):-n_test_each]
            vl_sig = sig[-(n_test_each + n_val_each):-n_test_each]

            remaining_bkg = bkg[:-(n_test_each + n_val_each)]
            remaining_sig = sig[:-(n_test_each + n_val_each)]
            
            if n_train_each > len(remaining_bkg) or n_train_each > len(remaining_sig):
                raise ValueError(
                    f"Requested train_num_samples={self.n_train} "
                    f"({n_train_each} per class) but only "
                    f"{len(remaining_bkg)} bkg / {len(remaining_sig)} sig remain "
                    f"after val/test carve-out."
                )

            tr_bkg = rng.permutation(remaining_bkg)[:n_train_each]
            tr_sig = rng.permutation(remaining_sig)[:n_train_each]
            
            # So currently test and val were selected from a fixed dataset, train was selected at random
            # Now we concatenate bkg and sig, these just need to be shuffled generally/intertwined
            tr_idx = self.global_shuffle(np.concatenate([tr_bkg, tr_sig]), global_seed)
            vl_idx = self.global_shuffle(np.concatenate([vl_bkg, vl_sig]), global_seed)
            te_idx = self.global_shuffle(np.concatenate([te_bkg, te_sig]), global_seed)

            # Also return remaining pool such that normalisation is over fixed set (so normalisation doesn't change if 10k or 100k train requested)
            return tr_idx, vl_idx, te_idx, np.concatenate([remaining_bkg, remaining_sig])

    def normalisation(self, X):
        """
        Here the features are scaled before being delivered/plotted. Performed here to keep
        centralised.

        NOTE: The normalisation is fully data independent to avoid bias. We don't use stats from the full training pool, as this
        would be using data we are not necessarily training on, providing additional information. We also don't use stats from the
        seeded training set, as this could scale the same data point differently, depending on the jets selected with it.
        """
        X = X.copy()

        eta, pt_frac, phi = X[:, :, 0], X[:, :, 1], X[:, :, 2]

        # Create 3 fresh channels
        out = np.empty(X.shape[:2] + (3,), dtype=X.dtype)

        # Pt_frac 
        # pt_frac = np.sqrt(pt_frac / p["pt_frac_max"])    # [0, 1], same map for every slot
        # pt_frac = np.sqrt(pt_frac)    # [0, 1], same map for every slot

        # We clip |eta| and |phi| at R=0.8 or 1.
        # This keeps all events and leaves all values within the standard jet radius unchanged.
        # Technically the jet radius is on eta and phi combined, but we are not enforcing a condition on R,
        # we are simply staurating eta and phi at the jet radius, to then rescale to [-1,1]
        # This is the looser condition, as (0.7,0.7) has dR=0.99, so our cut on eta, phi is less strict
        eta = np.clip(eta, -1, 1)
        # Phi initial range already < pi so no wrapping needed
        phi = np.clip(phi, -1, 1)

        # Fixed linear map (to reach -1,1)
        # eta = eta / self.R_JET
        # phi = phi / self.R_JET

        if self.paradigm == "Qumode":
            disp_factor = self.cfg.data.d_mult
            sqze_factor = self.cfg.data.s_mult

            out[:, :, 0] = sqze_factor * eta           # eta -> now in squeeze
            out[:, :, 1] = disp_factor * pt_frac       # pt -> displacement
            out[:, :, 2] = phi * np.pi                 # phi

        elif self.paradigm == "Classical":

            out[:, :, 0] = eta
            out[:, :, 1] = pt_frac
            out[:, :, 2] = phi
        
        elif self.paradigm == "Qubit":

            out[:, :, 0] = eta
            out[:, :, 1] = pt_frac
            out[:, :, 2] = phi

        else:
            raise ValueError("Paradigm should be Qumode, Classical or Qubit!")

        return out

    def free_raw_data(self):
        for attr in ("const", "truth", "jet_pt"):
            if hasattr(self, attr):
                delattr(self, attr)
        gc.collect()

    def createUnsupervisedDatasets(self, rng):

        # Train, val, test split (indices)
        tr_idx, vl_idx, te_idx = self.split_train_val_test(rng, unsupervised=True)

        # Train, val, test split
        X_tr_raw = self.build_X(self.const, self.jet_pt, tr_idx, k=self.n)
        X_vl_raw = self.build_X(self.const, self.jet_pt, vl_idx, k=self.n)
        X_te_raw = self.build_X(self.const, self.jet_pt, te_idx, k=self.n)

        # Normalise using scaler pool
        X_tr = self.normalisation(X_tr_raw)
        X_vl = self.normalisation(X_vl_raw)
        X_te = self.normalisation(X_te_raw)

        y_te = torch.from_numpy((self.truth[te_idx] == self.truth.max()).astype(np.int64))
        gc.collect()

        print("\nFinal (Train) Feature Limits")
        for i, name in enumerate(["eta", "pt_frac", "phi"]):
            values = X_tr[:, :, i]
            finite = values[np.isfinite(values)]
            if finite.size:
                print(
                    f"{name}: "
                    f"min={finite.min():.6f}, "
                    f"max={finite.max():.6f}, "
                    f"p1={np.percentile(finite, 1):.6f}, "
                    f"p99={np.percentile(finite, 99):.6f}"
                )
            else:
                print(f"{name}: no finite values")

        return X_tr, X_vl, X_te, y_te

    def createSupervisedDatasets(self, rng):
        tr_idx, vl_idx, te_idx, scaler_pool = self.split_train_val_test(rng, unsupervised=False)

        all_idx = np.unique(np.concatenate([tr_idx, vl_idx, te_idx]))
        X_all = self.build_X(self.const, self.jet_pt, all_idx, k=self.n)
        r2i = {int(i): r for r, i in enumerate(all_idx)}

        def G(idx):
            return X_all[np.array([r2i[int(i)] for i in idx])]

        # Normalise over fixed pool, independent of no.train requested
        X_tr = self.normalisation(G(tr_idx), fit_idx=scaler_pool)
        X_vl = self.normalisation(G(vl_idx), fit_idx=scaler_pool)
        X_te = self.normalisation(G(te_idx), fit_idx=scaler_pool)

        y_tr = torch.from_numpy((self.truth[tr_idx] == self.truth.max()).astype(np.int64))
        y_vl = torch.from_numpy((self.truth[vl_idx] == self.truth.max()).astype(np.int64))
        y_te = torch.from_numpy((self.truth[te_idx] == self.truth.max()).astype(np.int64))

        del X_all, r2i
        gc.collect()
        
        return X_tr, y_tr, X_vl, y_vl, X_te, y_te