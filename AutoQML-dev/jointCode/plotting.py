import sys
from pathlib import Path
import torch
_root = Path(__file__).parent.parent
sys.path.append(str(_root / "jointCode"))
sys.path.append(str(_root / "quantumCode"))
import os
import pandas as pd
import numpy as np
from cycler import cycler
from sklearn.metrics import roc_curve, auc
from utils import YoudenJ
import matplotlib.pyplot as plt
import pennylane as qml
from scipy.stats import wasserstein_distance
import re
import ast

plt.rcParams.update(
    {
        "text.usetex": False,
        "mathtext.fontset": "dejavuserif",
        "font.family": "DejaVu Serif",
        "axes.labelsize": 22,
        "xtick.labelsize": 22,
        "ytick.labelsize": 22,
        "legend.fontsize": 18,
        "lines.linewidth": 3,
        "lines.markersize": 8,
        "legend.title_fontsize": 18,
        "xaxis.labellocation": "right",
        "yaxis.labellocation": "top",
        # tick params
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.major.width": 2,
        "ytick.major.width": 2,
        "xtick.major.size": 8,
        "ytick.major.size": 8,
        # legend
        "legend.loc": "upper left",
        "legend.frameon": False,
        # grid
        "axes.grid": True,
        "grid.linestyle": "--",
        "grid.alpha": 0.5,
        # colours (https://www.nceas.ucsb.edu/sites/default/files/2022-06/Colorblind%20Safe%20Color%20Schemes.pdf)
        "axes.prop_cycle": cycler(
            color=[
                (148 / 255, 203 / 255, 236 / 255),
                (194 / 255, 106 / 255, 119 / 255),
                (93 / 255, 168 / 255, 153 / 255),
                (220 / 255, 205 / 255, 125 / 255),
                (159 / 255, 74 / 255, 150 / 255),
                (0 / 255, 114 / 255, 178 / 255),
            ]
        ),
    }
)
my_cmap = [
    "#FF7165",
    "#9B6DFF",
    "#E5C04A",
    "#dc267f",
    "#78C2EC",
    "#2A9D55",
    "#4A2FE2" 
]

class Plotter:
    def __init__(self, run_path, cfg, paradigm):
        self.run_path = run_path
        self.plots_path = os.path.join(self.run_path, "Plots")
        os.makedirs(self.plots_path, exist_ok=True)
        self.cfg = cfg
        self.paradigm = paradigm

        # General prefix to use for clarity
        if paradigm == "quantum":
            self.plot_prefix = "Q"
        elif paradigm == "classical":
            self.plot_prefix = "C"
        elif paradigm == "qubit":
            self.plot_prefix = "QB"
        else:
            raise ValueError(
                "paradigm should be either 'classical' or 'quantum' or 'qubit'!"
            )

        # Note paths to be filled during training, accessible for plotting after
        self.logging_path = os.path.join(
            self.run_path, f"{self.plot_prefix}_logging.csv"
        )
        self.evaluations_path = os.path.join(
            self.run_path, f"{self.plot_prefix}_evaluations.npz"
        )
        return

    def plotCirc(self, params, pairs):
        from q_circ import get_op_log
        sections = get_op_log(self.cfg, params, pairs)

        # Name wires
        # if self.cfg.training.paradigm == "unsupervised":
        #     trash = set(self.cfg.circuit.trash_modes)
        #     wire_labels = [(f"(tr){i}"if i in trash else f"(lat){sum(j<i and j not in trash for j in range(i))}")
        #                    for i in range(self.cfg.circuit.wires)]
        # else:
        wire_labels = list(range(self.cfg.circuit.wires))
        dev = qml.device("default.gaussian", wires=wire_labels)

        @qml.qnode(dev)
        def dummy_circuit():
            for i, section in enumerate(sections):
                for name, wires in section:
                    if name == "D":
                        qml.Displacement(0.0, 0.0, wires=wire_labels[wires[0]])

                    elif name == "S":
                        qml.Squeezing(0.0, 0.0, wires=wire_labels[wires[0]])

                    elif name == "R":
                        qml.Rotation(0.0, wires=wire_labels[wires[0]])

                    elif name == "BS":
                        qml.Beamsplitter(
                            0.0, 0.0, wires=[wire_labels[w] for w in wires]
                        )
                    elif name == "CX":
                        qml.ControlledAddition(0.0, wires=[wire_labels[w] for w in wires])
                if i < len(sections):
                    qml.Barrier(wire_labels)
            if self.cfg.training.paradigm == "unsupervised":
                trash_wire_labels = [wire_labels[i] for i in self.cfg.circuit.trash_modes]
                return [qml.expval(qml.QuadX(w)) for w in trash_wire_labels]
            else:
                return qml.state()

        fig, ax = qml.draw_mpl(dummy_circuit, wire_order=wire_labels)()
        fig.savefig(f"{self.plots_path}/Circuit.png", dpi=150, bbox_inches="tight")

    def plotMetrics(self, metrics):

        # Load CSV
        df = pd.read_csv(self.logging_path)

        if df.empty:
            print("No metrics to plot")
            return

        for metric in metrics:
            # Plot metric vs epoch
            fig, ax = plt.subplots()

            # Plot main line
            ax.plot(df["epoch"], df[metric], marker="o")

            ax.set_xlabel("Epoch")
            ax.set_ylabel(f"{metric}")
            ax.set_title("Training Metrics vs Epoch")
            ax.grid(True)

            plt.tight_layout()
            fig.savefig(
                f"{self.plots_path}/{self.plot_prefix}_{metric}.png",
                dpi=300,
                bbox_inches="tight",
            )
            plt.close(fig)
        return

    def plot_input_features(self, X_train, X_te, y_te):

        feat_names = ["eta", "pt_frac", "phi"]
        labels = [r"$\eta$", r"$p_T/p_T^{jet}$", r"$\phi$"]

        def trim(bins, vals):
            # keep only the bin edges spanning this dataset's own min/max
            i0 = np.searchsorted(bins, vals.min(), side="right") - 1
            i1 = np.searchsorted(bins, vals.max(), side="left")
            return bins[i0:i1 + 1]

        N = X_train.shape[1]
        n_cols = 4
        n_rows = int(np.ceil(N / n_cols))

        for feat_idx, (name, label) in enumerate(zip(feat_names, labels)):

            fig, axes = plt.subplots(
                n_rows, n_cols,
                figsize=(4 * n_cols, 3.5 * n_rows),
                squeeze=False
            )
            axes_flat = axes.ravel()

            for constituent_idx in range(N):
                ax = axes_flat[constituent_idx]

                train_vals = X_train[:, constituent_idx, feat_idx]
                te_bg_vals = X_te[y_te == 0, constituent_idx, feat_idx]
                te_sg_vals = X_te[y_te == 1, constituent_idx, feat_idx]

                all_vals = np.concatenate([train_vals, te_bg_vals, te_sg_vals])
                bins = np.histogram_bin_edges(all_vals, bins=30)      # shared bins

                ax.hist(
                    te_bg_vals, bins=trim(bins, te_bg_vals), density=True,     # <-- trimmed
                    alpha=0.45, color="steelblue", label="Test BG"
                )
                ax.hist(
                    te_sg_vals, bins=trim(bins, te_sg_vals), density=True,     # <-- trimmed
                    alpha=0.45, color="tomato", label="Test SG"
                )
                ax.hist(
                    train_vals, bins=trim(bins, train_vals), density=True,     # <-- trimmed
                    histtype="step", linewidth=2,
                    color="black", linestyle="--", label="Train"
                )

                ax.set_title(f"Constituent {constituent_idx}", fontsize=9)
                ax.set_xlabel(label, fontsize=8)
                ax.set_ylabel("Density", fontsize=8)
                ax.tick_params(labelsize=7)
                ax.grid(True, alpha=0.3)

                if constituent_idx == 0:
                    ax.legend(fontsize=7)

            for idx in range(N, len(axes_flat)):
                axes_flat[idx].set_visible(False)

            fig.suptitle(
                f"{label} — per-constituent distributions\n"
                f"(Train outline, Test BG/SG filled)",
                fontsize=12, y=1.01
            )

            plt.tight_layout()
            fig.savefig(
                f"{self.plots_path}/{name}_per_constituent.png",
                dpi=300,
                bbox_inches="tight"
            )
            plt.close(fig)

    def plotLoss(self):
        df = pd.read_csv(self.logging_path)

        if df.empty:
            print("No metrics to plot")
            return

        val_df = df.dropna(subset=["val_loss"])

        fig, ax = plt.subplots()

        ax.plot(df["epoch"], df["train_loss"], label="Train")
        ax.plot(val_df["epoch"], val_df["val_loss"], marker="o", label="Val")

        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_yscale("log")
        ax.set_title("Loss Monitoring")
        ax.grid(True)
        ax.legend(loc="upper right")
        ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))

        plt.tight_layout()
        fig.savefig(
            f"{self.plots_path}/{self.plot_prefix}_Losses.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close(fig)
        return

    def plotN(self):
        df = pd.read_csv(self.logging_path)

        if df.empty:
            print("No metrics to plot")
            return

        fig, ax = plt.subplots()

        ax.plot(df["epoch"], df["meanN"], label="Mean")
        ax.plot(df["epoch"], df["maxN"], label="Max")

        ax.set_xlabel("Epoch")
        ax.set_ylabel("Photon Number")
        ax.set_yscale("log")
        ax.set_title("Photon Number Monitoring")
        ax.grid(True)
        ax.legend(loc="upper right")
        ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))

        plt.tight_layout()
        fig.savefig(
            f"{self.plots_path}/{self.plot_prefix}_PhotonNumber.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close(fig)
        return

    # Final Plots (TEST)
    def plotBSmixing(self):

        def load_theta(filename):
            with open(os.path.join(self.run_path, filename)) as f:
                text = f.read()
            match = re.search(r"bs_theta:\s*(\[\[.*?\]\])", text, re.S)
            if match is None:
                return None
            theta = np.asarray(ast.literal_eval(match.group(1))).ravel()
            theta = np.mod(theta, np.pi)
            return np.minimum(theta, np.pi - theta)

        theta_init = load_theta("init_params.txt")
        theta_best = load_theta("best_params.txt")

        if theta_init is None or theta_best is None:
            return None

        bins = np.linspace(0, np.pi / 2, 13)
        fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)

        for ax, theta, title in zip(
            axes,
            [theta_init, theta_best],
            ["Initial parameters", "Best parameters"],
        ):
            ax.hist(theta, bins=bins, histtype="step", linewidth=1.8)
            ax.set_xlim(0, np.pi / 2)
            ax.set_xticks([0, np.pi / 4, np.pi / 2])
            ax.set_xticklabels([r"$0$", r"$\pi/4$", r"$\pi/2$"])
            ax.set_xlabel(r"Folded beamsplitter angle $\theta$")
            ax.set_title(title)
            ax.grid(True)

        axes[0].set_ylabel("Number of beamsplitters")
        fig.suptitle("Beamsplitter Mixing Distribution")
        plt.tight_layout()

        fig.savefig(
            f"{self.plots_path}/{self.plot_prefix}_BSMixingHist.png",
            dpi=300, bbox_inches="tight",
        )
        plt.close(fig)

        return theta_init, theta_best

    def plotFinalLoss(self):

        # Load npz
        data = np.load(self.evaluations_path)
        losses = data["losses"]
        y_true = data["y_true"]

        best_threshold = YoudenJ(losses, y_true)

        # Separate bg and sg
        bg_losses = losses[y_true == 0]
        sg_losses = losses[y_true == 1]

        # Wasserstein-1
        w1_raw = wasserstein_distance(bg_losses, sg_losses)   # loss units, within-family only

        fig, ax = plt.subplots()

        bins = np.linspace(losses.min(), losses.max(), 101)

        ax.hist(
            bg_losses,
            bins=bins,
            alpha=1,
            histtype="step",
            label="Background",
            density=True,
        )
        ax.hist(
            sg_losses, bins=bins, alpha=1, histtype="step", label="Signal", density=True
        )

        ax.axvline(best_threshold, color="m", ls="--")

        ax.set_xlabel("Loss")
        ax.set_ylabel("Density")
        ax.set_title(f"Loss Distribution (BG vs SG)\nW1 = {w1_raw:.4g}")
        ax.grid(True)
        ax.legend(loc="upper right")

        plt.tight_layout()
        fig.savefig(
            f"{self.plots_path}/{self.plot_prefix}_FinalLossHist.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close(fig)
        return w1_raw

    def plotFinalROC(self):

        # Load npz
        data = np.load(self.evaluations_path)
        losses = data["losses"]
        y_true = data["y_true"]

        fpr, tpr, _ = roc_curve(y_true, losses)
        roc_auc = auc(fpr, tpr)

        fig, ax = plt.subplots(figsize=(8, 8))

        ax.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
        ax.plot([0, 1], [0, 1], "--", label="Random")

        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("ROC Curve")
        ax.grid(True)
        ax.legend(loc="lower right")

        plt.tight_layout()
        fig.savefig(
            f"{self.plots_path}/{self.plot_prefix}_FinalROC.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close(fig)

    def plotNormalisedLoss(self):

        # Load npz
        data = np.load(self.evaluations_path)
        losses = data["losses"]
        y_true = data["y_true"]

        lo, hi = losses.min(), losses.max()
        scores = (losses - lo) / (hi - lo)

        bg_scores = scores[y_true == 0]
        sg_scores = scores[y_true == 1]

        best_score = YoudenJ(scores, y_true)

        fig, ax = plt.subplots()

        bins = np.linspace(0, 1, 101)

        ax.hist(
            bg_scores,
            bins=bins,
            alpha=1,
            histtype="step",
            label="Background",
            density=True,
        )
        ax.hist(
            sg_scores, bins=bins, alpha=1, histtype="step", label="Signal", density=True
        )

        ax.axvline(best_score, color="m", ls="--")

        ax.set_xlim(0, np.percentile(scores, 99))
        ax.set_xlabel("MinMax Normalised Loss")
        ax.set_ylabel("Density")
        ax.set_title("Normalised Loss Distribution (BG vs SG)")
        ax.grid(True)
        ax.legend(loc="upper right")

        plt.tight_layout()
        fig.savefig(
            f"{self.plots_path}/{self.plot_prefix}_NormLossHist.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close(fig)
        return

    #######################################################

    def plotClassical(self):
        encoder_shape = self.cfg.circuit.encoder_shape

        if self.cfg.training.paradigm == "supervised":
            full_layers = encoder_shape + [1]
        else:
            decoder_shape = encoder_shape[::-1][1:]
            full_layers = encoder_shape + decoder_shape

        n_layers = len(full_layers)
        n_encoder = len(encoder_shape)

        palette = my_cmap

        color_for_layer = {}
        if self.cfg.training.paradigm == "unsupervised":
            for i in range(n_encoder - 1):
                c = palette[i % len(palette)]
                color_for_layer[i] = c
                color_for_layer[n_layers - 1 - i] = c
            bottleneck_color = palette[(n_encoder - 1) % len(palette)]
            color_for_layer[n_encoder - 1] = bottleneck_color
        else:
            for i in range(n_layers):
                color_for_layer[i] = palette[i % len(palette)]

        y_spacing = 1.2  # >1 spreads neurons apart vertically; tune to taste

        fig, ax = plt.subplots(figsize=(11, 5 * y_spacing))
        max_neurons = max(full_layers)

        for i, neurons in enumerate(full_layers):
            y_offset = (max_neurons - neurons) / 2 * y_spacing
            y = [yi * y_spacing + y_offset for yi in range(neurons)]
            x = [i] * neurons
            ax.scatter(x, y, s=200, zorder=2, color=color_for_layer[i])

            ax.text(i, -1.5 * y_spacing, str(neurons), ha="center", va="top", fontsize=14)

            if i < n_layers - 1:
                next_neurons = full_layers[i + 1]
                next_y_offset = (max_neurons - next_neurons) / 2 * y_spacing
                for yi in y:
                    for yj in range(next_neurons):
                        ax.plot(
                            [i, i + 1],
                            [yi, yj * y_spacing + next_y_offset],
                            color="gray",
                            alpha=0.3,
                            zorder=1,
                        )

        ax.set_xlim(-0.5, n_layers - 0.5)
        ax.set_ylim(-3 * y_spacing, max_neurons * y_spacing + 1)

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["bottom"].set_visible(False)
        ax.spines["left"].set_visible(False)

        ax.set_xticks(range(n_layers))
        ax.tick_params(axis="x", length=0, labelbottom=False)
        ax.xaxis.grid(True, color="gray", alpha=0.2, linewidth=0.8)

        ax.set_yticks([])

        ax.set_axisbelow(True)

        plt.savefig(f"{self.plots_path}/C_circuit.png", dpi=600, bbox_inches="tight")
        plt.close(fig)

        print("- Classical model plot created")
        return

    #######################################################

    def plotQubit(self, model):
        """Draw a buildQubit model's circuit with dummy inputs."""

        n_const = self.cfg.circuit.constituents   # whatever your config calls it (8)
        dummy_x = torch.rand(1, n_const, 3)

        fig, ax = qml.draw_mpl(model.circuit, show_all_wires=True)(dummy_x)
        save_path = os.path.join(self.plots_path, "QB_circuit.png")
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        return fig, ax
