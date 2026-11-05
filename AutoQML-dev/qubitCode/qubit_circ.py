import pennylane as qml
import torch
import itertools
from qubit_params import PARAM_SPECS, init_param
from torch import nn

class buildQubit(nn.Module):
    def __init__(self, cfg, rngs):
        super().__init__()

        # Definitions
        self.cfg = cfg
        self.n_qubits = cfg.circuit.qubits          # number of "data" qubits
        self.trash = list(cfg.circuit.trash)        # explicit trash wire indices, e.g. [1, 3]
        self.n_trash = len(self.trash)
        self.gen = rngs["circuit"]

        self.qubits = list(range(self.n_qubits))                          # all data wires

        # total wires needed for the device
        if cfg.circuit.measurement == "SWAP":

            # reference wires + ancilla
            self.reference = list(range(self.n_qubits, self.n_qubits + self.n_trash))
            self.ancilla = self.n_qubits + self.n_trash

            self.total_wires = self.n_qubits + self.n_trash + 1

        elif cfg.circuit.measurement == "vacProb":
            self.total_wires = self.n_qubits
        else:
            raise ValueError("cfg.circuit.measurement must be 'SWAP' or 'vacProb'!")

        self.dev = qml.device("default.qubit",wires=self.total_wires)

        # Params
        self.params = nn.ParameterDict()

        # Instantiate
        self.circuit = qml.qnode(self.dev, interface="torch")(self.buildCirc)

    def buildEncoding(self, x):

        ######## x: (batch, n_qubits, 3)
        # x = nn.functional.pad(x, (0, 1)).reshape(x.shape[0], -1) # 4-slot pad (eta, pt, phi, 0, eta, pt, phi, 0, ...)

        x = nn.functional.pad(x.reshape(x.shape[0], -1), (0, self.cfg.circuit.constituents))   # padding at end (eta, pt, phi, eta, pt, phi...0, 0, 0)

        # x = nn.functional.pad(x.reshape(x.shape[0], -1), (0, 8))
        # x = torch.flip(x, dims=[1])              # zeros first, and the data order is reversed

        # Print Amplitudes
        # norms = torch.linalg.vector_norm(x, dim=-1, keepdim=True)
        # amps = x / norms.clamp(min=1e-12)
        # features = amps[:, :24].reshape(-1, 8, 3)
        # padding = amps[:, 24:]
        # print("Mean absolute amplitude [eta, pt, phi]:",
        #     features.abs().mean(dim=(0, 1)))
        # print("Probability contribution [eta, pt, phi]:",
        #     features.square().sum(dim=(0, 1)) / amps.square().sum())
        # print("Padding probability:", padding.square().sum().item() / amps.shape[0])

        # Embedding
        qml.AmplitudeEmbedding(
            features=x,
            wires=range(self.n_qubits),
            normalize=True,    # L2 normalise
            )

    def buildLocalGates(self,idx):

        # Only initialise if called, but only once
        if "lg" not in self.params:
            n_lg_blocks = self.cfg.circuit.layer_structure.count("lg")
            self.params["lg"] = init_param("lg", (n_lg_blocks, self.n_qubits, 3), self.gen)

        lg = self.params["lg"]
        for q in self.qubits:
            qml.Rot(lg[idx, q, 0], lg[idx, q, 1], lg[idx, q, 2], wires=q)
            
    def buildSubstrate(self):
        substrate = self.cfg.circuit.entangling_substrate

        if substrate == "all":
            pairs = itertools.combinations(self.qubits, 2)
            for control, target in pairs:
                qml.CNOT(wires=[control, target])
        
        elif substrate == "ring":
            for i, control in enumerate(self.qubits):
                target = self.qubits[(i + 1) % len(self.qubits)]
                qml.CNOT(wires=[control, target])

    def buildMeasurement(self):
        if self.cfg.circuit.measurement == "SWAP":

            # SWAP
            for t, r in zip(self.trash, self.reference):
                qml.CSWAP(wires=[self.ancilla, t, r])

            # Measurement
            qml.Hadamard(wires=self.ancilla)
            return qml.expval(qml.PauliZ(self.ancilla))
        
        # If vacProb calculation
        else:
            return qml.probs(wires=self.trash)

    def buildCirc(self, x):

        # Encoding
        self.buildEncoding(x)
        if self.cfg.circuit.measurement == "SWAP":
            qml.Hadamard(wires=self.ancilla)

        lg_idx = 0
        for block in self.cfg.circuit.layer_structure:
            if block == "ent":
                # Entangling
                self.buildSubstrate()

            elif block == "lg":
                # Local gates
                self.buildLocalGates(lg_idx)
                lg_idx += 1
            else:
                raise ValueError(f"Unknown block type: {block}")

        # Measurement
        return self.buildMeasurement()

    def __call__(self, x):
        out = self.circuit(x)

        # If vacProb, take vacuum prob only
        if self.cfg.circuit.measurement == "vacProb":
            out = out[..., 0]
        return out