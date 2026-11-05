import torch
from itertools import product as iproduct
import sys
import math
from pathlib import Path
_root = Path(__file__).parent.parent  # Setting root
sys.path.append(str(_root / "jointCode"))
sys.path.append(str(_root / "quantumCode"))
torch.set_default_dtype(torch.float64)

import pennylane as qml
from q_config import CONFIG
from q_params import init_param

# Initialize device dynamically based on config
n_wires = CONFIG.circuit.wires
dev = qml.device('strawberryfields.fock', wires=n_wires, cutoff_dim=4)

def cartesian_to_polar(real, imag):
    """
    Converts Cartesian coordinates (real, imaginary) to polar coordinates (r, phi).
    
    This is required because the custom parameter initialization uses Cartesian 
    coordinates to avoid autograd singularities at zero, but PennyLane's 
    continuous-variable operations require polar inputs.
    
    Args:
        real (torch.Tensor or float): The real component.
        imag (torch.Tensor or float): The imaginary component.
        
    Returns:
        tuple: (r, phi) where r is the magnitude and phi is the phase angle.
    """
    r = torch.sqrt(real**2 + imag**2)
    phi = torch.atan2(imag, real)
    return r, phi

def pennylane_encoding(x, cfg, params, reupload_layer=None):
    """
    Encodes the classical input features into the quantum continuous-variable state.
    
    In unsupervised mode, features are encoded directly into the displacement 
    and squeezing operations. In supervised mode, trainable weights are applied 
    to the features before encoding, allowing the network to learn the optimal embedding.
    
    Args:
        x (torch.Tensor): A single sample of input features with shape (n_wires, 3).
                          The 3 features are (eta, pt_frac, phi).
        cfg (OmegaConf): The configuration object containing circuit details.
        params (dict): Dictionary of trainable parameters.
        reupload_layer (int, optional): The current re-upload layer index for supervised mode.
    """
    if not torch.is_tensor(x):
        x = torch.as_tensor(x, dtype=torch.float64)
        
    n_wires = cfg.circuit.wires
    
    if cfg.training.paradigm == "unsupervised":
        for wire in range(n_wires):
            eta = x[wire, 0]
            pt_frac = x[wire, 1]
            phi = x[wire, 2]
            
            qml.Displacement(pt_frac, phi, wires=wire)
            qml.Squeezing(eta, 0.0, wires=wire)
            
    else:
        for wire in range(n_wires):
            eta = x[wire, 0]
            pt = x[wire, 1]
            phi = x[wire, 2]

            var_pt_w = params["enc_var_pt"][reupload_layer, wire]
            var_eta_w = params["enc_var_eta"][reupload_layer, wire]
            var_phi_w = params["enc_var_phi"][reupload_layer, wire]
            
            d_real = var_eta_w * eta
            d_imag = var_pt_w * pt
            d_r, d_phi = cartesian_to_polar(d_real, d_imag)
            qml.Displacement(d_r, d_phi, wires=wire)
            
            s_real = var_phi_w * torch.cos(phi)
            s_imag = var_phi_w * torch.sin(phi)
            s_r, s_phi = cartesian_to_polar(s_real, s_imag)
            qml.Squeezing(s_r, s_phi, wires=wire)

def pennylane_bs_substrate(params, pairs, bs_set_index):
    """
    Applies a layer of Beamsplitter gates to entangle pairs of qumodes.
    
    Args:
        params (dict): Dictionary of trainable parameters containing 'bs_theta' and 'bs_phi'.
        pairs (list of tuples): List of (wire1, wire2) indices specifying which modes to entangle.
        bs_set_index (int): The index of the current beamsplitter layer.
    """
    k = 0
    for w1, w2 in pairs:
        theta = params["bs_theta"][bs_set_index, k]
        phi = params["bs_phi"][bs_set_index, k]
        qml.Beamsplitter(theta, phi, wires=[w1, w2])
        k += 1

def pennylane_lg_substrate(params, n_wires, lg_set_index, lg_type):
    """
    Applies local single-mode gates (Displacement, Squeezing, or Rotation) to all qumodes.
    
    Args:
        params (dict): Dictionary of trainable parameters for the local gates.
        n_wires (int): Total number of qumodes in the circuit.
        lg_set_index (int): The index of the current local gate layer.
        lg_type (str): String indicating which gates to apply (e.g., 'd' for displacement, 
                       's' for squeezing, 'r' for rotation).
    """
    if lg_type == "none" or lg_type is None:
        return
        
    for w in range(n_wires):
        for letter in lg_type:
            if letter == "d":
                r, phi = cartesian_to_polar(
                    params["lg_d_real"][lg_set_index, w], 
                    params["lg_d_imag"][lg_set_index, w]
                )
                qml.Displacement(r, phi, wires=w)
            elif letter == "s":
                r, phi = cartesian_to_polar(
                    params["lg_s_real"][lg_set_index, w], 
                    params["lg_s_imag"][lg_set_index, w]
                )
                qml.Squeezing(r, phi, wires=w)
            elif letter == "r":
                phi_scalar = params["lg_r_phi"][lg_set_index, w]
                qml.Rotation(phi_scalar, wires=w)

def pennylane_cx_substrate(params, pairs, cx_set_index):
    """
    Applies a layer of Continuous-Variable Controlled-X (CX) gates to pairs of qumodes.
    
    Args:
        params (dict): Dictionary of trainable parameters containing 'cx_s'.
        pairs (list of tuples): List of (wire1, wire2) indices specifying the control/target modes.
        cx_set_index (int): The index of the current CX layer.
    """
    k = 0
    for w1, w2 in pairs:
        s = params["cx_s"][cx_set_index, k]
        qml.CX(s, wires=[w1, w2])
        k += 1

@qml.qnode(dev, interface="torch")
def qae_qnode(x_single, cfg, params, pairs):
    """
    Constructs and executes the Quantum Autoencoder (QAE) circuit for a single sample.
    
    The autoencoder compresses the encoded data by attempting to return a subset 
    of modes (the 'trash modes') to the vacuum state.
    
    Args:
        x_single (torch.Tensor): A single input sample.
        cfg (OmegaConf): Configuration settings.
        params (dict): Trainable parameters.
        pairs (list of tuples): Architecture pairs for entangling gates.
        
    Returns:
        tuple: (trash_mode_probabilities, expected_photon_counts_all_modes)
    """
    pennylane_encoding(x_single, cfg, params)
    
    bs_idx, lg_idx, cx_idx = 0, 0, 0
    
    for layer in range(cfg.circuit.layers):
        for block in cfg.circuit.layer_structure:
            if block == "bs":
                pennylane_bs_substrate(params, pairs, bs_idx)
                bs_idx += 1
            elif block == "lg":
                pennylane_lg_substrate(params, cfg.circuit.wires, lg_idx, cfg.circuit.lg_substrate_type)
                lg_idx += 1
            elif block == "cx":
                pennylane_cx_substrate(params, pairs, cx_idx)
                cx_idx += 1
                
    return (
        qml.probs(wires=cfg.circuit.trash_modes), 
        [qml.expval(qml.NumberOperator(w)) for w in range(cfg.circuit.wires)]
    )

@qml.qnode(dev, interface="torch")
def cvvc_qnode(x_single, cfg, params, pairs):
    """
    Constructs and executes the Continuous-Variable Variational Classifier (CVVC) 
    circuit for a single sample.
    
    This architecture allows for data re-uploading, embedding the classical data 
    repeatedly between trainable layers to increase expressivity.
    
    Args:
        x_single (torch.Tensor): A single input sample.
        cfg (OmegaConf): Configuration settings.
        params (dict): Trainable parameters.
        pairs (list of tuples): Architecture pairs for entangling gates.
        
    Returns:
        tuple: (all_mode_probabilities, expected_photon_counts_all_modes)
    """
    n_wires = cfg.circuit.wires
    pennylane_encoding(x_single, cfg, params, reupload_layer=0)
    
    bs_idx, lg_idx = 0, 0
    
    for layer in range(cfg.circuit.layers):
        for block in cfg.circuit.layer_structure:
            if block == "bs":
                pennylane_bs_substrate(params, pairs, bs_idx)
                bs_idx += 1
            elif block == "lg":
                pennylane_lg_substrate(params, n_wires, lg_idx, cfg.circuit.lg_substrate_type)
                lg_idx += 1
                
        if cfg.circuit.reupload and layer < cfg.circuit.layers - 1:
            pennylane_encoding(x_single, cfg, params, reupload_layer=layer + 1)

    return (
        qml.probs(wires=list(range(n_wires))),
        [qml.expval(qml.NumberOperator(w)) for w in range(n_wires)]
    )

def QAE_loss(cfg, x_batch, params, pairs):
    """
    Calculates the autoencoder loss across a batch of samples.
    
    Loops over the QAE QNode sequentially for each sample. The fidelity (F) is 
    defined as the probability of measuring the vacuum state (|00...0>) in the 
    trash modes, which is found at index 0 of the returned probabilities.
    
    Args:
        cfg (OmegaConf): Configuration settings.
        x_batch (torch.Tensor): Batch of input data.
        params (dict): Trainable parameters.
        pairs (list of tuples): Connectivity pairs.
        
    Returns:
        tuple: (loss_tensor, fidelity_tensor, photon_number_tensor)
    """
    B = x_batch.shape[0]
    batch_F = []
    batch_n = []
    for b in range(B):
        probs, n = qae_qnode(x_batch[b], cfg, params, pairs)
        F = probs[0] 
        logF = torch.log(F + 1e-12)
        batch_F.append(F.unsqueeze(0))
        batch_n.append(torch.stack(n).unsqueeze(0))

    F_tensor = torch.cat(batch_F)
    logF_tensor = torch.log(F_tensor + 1e-12)
    n_tensor = torch.cat(batch_n)

    if cfg.training.loss_fn == "F":
        return -torch.expm1(logF_tensor), F_tensor, n_tensor
    elif cfg.training.loss_fn == "logF":
        return -logF_tensor, F_tensor, n_tensor
    elif cfg.training.loss_fn == "logF/n":
        return -logF_tensor / len(cfg.circuit.trash_modes), F_tensor, n_tensor

def CVVC_loss(cfg, x_batch, params, pairs):
    """
    Calculates the classifier loss across a batch of samples.
    
    Loops over the CVVC QNode sequentially. Translates the fidelity (overlap 
    with the target state) into a logit score which is then shifted and scaled 
    using the trainable 'sup_logloss_scale' and 'shift' parameters before being 
    passed to the PyTorch BCELoss function in the main training loop.
    
    Args:
        cfg (OmegaConf): Configuration settings.
        x_batch (torch.Tensor): Batch of input data.
        params (dict): Trainable parameters.
        pairs (list of tuples): Connectivity pairs.
        
    Returns:
        tuple: (logits_tensor, photon_number_tensor)
    """
    B = x_batch.shape[0]
    batch_F = []
    batch_n = []
    for b in range(B):
        probs, n = cvvc_qnode(x_batch[b], cfg, params, pairs)
        batch_F.append(probs[0].unsqueeze(0)) 
        batch_n.append(torch.stack(n).unsqueeze(0))

    F_tensor = torch.cat(batch_F)
    logF_tensor = torch.log(F_tensor + 1e-12)
    n_tensor = torch.cat(batch_n)

    if cfg.training.loss_fn == "F":
        return -torch.expm1(logF_tensor), n_tensor
    elif cfg.training.loss_fn == "logF":
        raw_score = -logF_tensor
        logit = params["sup_logloss_scale"] * raw_score + params["sup_logloss_shift"]
        return logit, n_tensor
    elif cfg.training.loss_fn == "logF/n":
        raw_score = -logF_tensor / cfg.circuit.wires
        logit = params["sup_logloss_scale"] * raw_score + params["sup_logloss_shift"]
        return logit, n_tensor

# ============================================================
# Parameter Creation
# ============================================================

def make_params(cfg, rngs):
    """
    Initializes all trainable parameters and network connectivity based on the config.
    
    Determines parameter shapes based on the number of wires, layers, and the 
    specified architectural topology (ring, star, chain, etc.). 
    
    Args:
        cfg (OmegaConf): Configuration settings.
        rngs (dict): Dictionary of random number generators for reproducible initialization.
        
    Returns:
        tuple: (params_dict, connectivity_pairs_list)
    """
    params = {}
    pairs = []
    gen = rngs["circuit"]
    n_wires = cfg.circuit.wires
    n_layers = cfg.circuit.layers
    arch = cfg.circuit.bs_substrate_type
    lg = cfg.circuit.lg_substrate_type

    if cfg.training.paradigm == "unsupervised":
        if cfg.circuit.reupload:
            raise ValueError("Unsupervised should not be reuploading!")

    elif cfg.training.paradigm == "supervised":
        n_upload_layers = n_layers if cfg.circuit.reupload else 1
        for name in ["enc_var_pt", "enc_var_eta", "enc_var_phi"]:
            params[name] = init_param(name, (n_upload_layers, n_wires), gen)
    else:
        raise ValueError(f"{cfg.training.paradigm} is not unsupervised/supervised!")

    def pairs_ring(n): return [(i, (i + 1) % n) for i in range(n)]
    def pairs_chain(n): return [(i, i + 1) for i in range(n - 1)]
    def pairs_alltoall(n): return [(i, j) for i in range(n) for j in range(i + 1, n)]
    def pairs_star(n): return [(0, i) for i in range(1, n)]

    builders = {"ring": pairs_ring, "chain": pairs_chain,
                "alltoall": pairs_alltoall, "star": pairs_star}

    n_bs_per_layer = cfg.circuit.layer_structure.count("bs")
    n_bs_total = n_layers * n_bs_per_layer

    if n_bs_per_layer > 0:
        if arch == "trainable_int":
            pairs = bs_int_pairs(n_wires)
            params["bs_theta"] = init_param("bs_theta", (n_bs_total, len(pairs)), gen)
            params["bs_phi"] = init_param("bs_phi", (n_bs_total, len(pairs)), gen)

        elif arch in ("trainable_ring", "trainable_alltoall", "trainable_chain", "trainable_star"):
            arch_key = resolve_pairs(arch)
            pairs = builders[arch_key](n_wires)
            params["bs_theta"] = init_param("bs_theta", (n_bs_total, len(pairs)), gen)
            params["bs_phi"] = init_param("bs_phi", (n_bs_total, len(pairs)), gen)
        else:
            raise ValueError("f{arch} is not a known bs/cx_substrate!")

    n_cx_per_layer = cfg.circuit.layer_structure.count("cx")
    n_cx_total = n_layers * n_cx_per_layer

    if n_cx_per_layer > 0:
        if arch == "trainable_int":
            pairs = bs_int_pairs(n_wires)
            params["cx_s"] = init_param("cx_s", (n_cx_total, len(pairs)), gen)

        elif arch in ("trainable_ring", "trainable_alltoall", "trainable_chain", "trainable_star"):
            arch_key = resolve_pairs(arch)
            pairs = builders[arch_key](n_wires)
            params["cx_s"] = init_param("cx_s", (n_cx_total, len(pairs)), gen)
        else:
            raise ValueError("f{arch} is not a known bs/cx_substrate!")

    n_lg_per_layer = cfg.circuit.layer_structure.count("lg")
    n_lg_total = n_layers * n_lg_per_layer

    if lg is None:
        pass
    elif len(set(lg)) != len(lg): 
        raise ValueError("Local gates should use up to 1 of each of 'd','r','s'!")
    else:
        for letter in lg:
            if letter == "d":
                params["lg_d_real"] = init_param("lg_d_real", (n_lg_total, n_wires), gen)
                params["lg_d_imag"] = init_param("lg_d_imag", (n_lg_total, n_wires), gen)

            elif letter == "s":
                params["lg_s_real"] = init_param("lg_s_real", (n_lg_total, n_wires), gen)
                params["lg_s_imag"] = init_param("lg_s_imag", (n_lg_total, n_wires), gen)

            elif letter == "r":
                params["lg_r_phi"] = init_param("lg_r_phi", (n_lg_total, n_wires), gen)

            else:
                raise ValueError(f"'{letter}' not a recognised local gate!")

    if cfg.training.paradigm == "supervised" and cfg.training.loss_fn in ("logF","logF/n"):
        params["sup_logloss_scale"] = init_param("sup_logloss_scale",())
        params["sup_logloss_shift"] = init_param("sup_logloss_shift",())

    return params, pairs

def resolve_pairs(arch):
    """
    Parses the architecture string from the configuration to determine the topology.
    """
    if "ring" in arch:
        return "ring"
    if "chain" in arch:
        return "chain"
    if "alltoall" in arch:
        return "alltoall"
    if "star" in arch:
        return "star"
    raise ValueError(f"Unknown circuit arch: {arch}")

def bs_int_pairs(n):
    """
    Generates index pairs for a standard alternating brick-wall interferometer layout.
    """
    pairs = []
    for layer in range(n):
        start = 0 if (layer % 2 == 0) else 1
        pairs.extend((i, i + 1) for i in range(start, n - 1, 2))
    return pairs

def get_op_log(cfg, params, pairs):
    """
    Stubbed function to prevent runtime errors in auxiliary plotting scripts 
    that were expecting internal operation logs from the custom backend.
    """
    return []