import torch
from itertools import product as iproduct
import sys
import math
from pathlib import Path
_root = Path(__file__).parent.parent  # Setting root
sys.path.append(str(_root / "jointCode"))
sys.path.append(str(_root / "quantumCode"))
torch.set_default_dtype(torch.float64)
from q_params import init_param

##############
# The following code was developed using Claude, with each function being checked independently. Their cumulative effect
# is also compared to pennylane operations. These functions replicate what pennylane computes, however uses torch operations
# and containers throughout. This allows differentiation by backpropagation, which is currently unsupported for default.gaussian
# in pennylane. Default.gaussian internals use pure numpy, and so breaks the autograd. For qubit devices, pennylane implemented
# everything using qml.math, so it runs natively for torch/jax/tf, however default.gaussian was never updated for this. It is the use
# of numpy that's the limiting factor.

# We want to use backpropagation, as it requires 1 execution for the backwards pass. Currently, we have to use parameter-shift as the differentiation
# method, and this requires 2N circuit re-executions.

# Makes heavy reference to: https://docs.pennylane.ai/en/stable/_modules/pennylane/devices/default_gaussian.html#DefaultGaussian

# Roughly 15x speedup of foward/backward routine for just 20 params, 10 samples
##############

hbar = 2.0

# ===================================================================
# Batched state: cov (B, 2N, 2N), mu (B, 2N)
# Convention: mu = x1,x2...p1,p2...
# ===================================================================


def vacuum_state_torch_batch(num_wires, batch_size):
    """
    Returns batched vacuum state.
    cov: (B, 2N, 2N), mu: (B, 2N)
    """
    mu = torch.zeros(batch_size, 2 * num_wires, dtype=torch.float64)
    cov = ((torch.eye(2 * num_wires, dtype=torch.float64) * hbar /
           2).unsqueeze(0).expand(batch_size, -1, -1).clone())
    return cov, mu


def apply_symplectic_batch(state, S, wires, num_wires):
    """
    Apply a symplectic matrix S (acting on `wires`) to a batched state.
    S: (2k, 2k) — shared across batch (e.g. BS, rotation layers)
       OR (B, 2k, 2k) — per-sample (e.g. data-dependent squeezing)
    cov: (B, 2N, 2N), mu: (B, 2N)
    """
    cov, mu = state
    B = mu.shape[0]
    N = num_wires
    w = list(wires)

    wx = torch.tensor(w, dtype=torch.long)
    wp = torch.tensor([wi + N for wi in w], dtype=torch.long)
    idx = torch.cat([wx, wp])  # length 2k

    # Build full S2: (2N, 2N) or (B, 2N, 2N) depending on whether S is batched
    if S.dim() == 2:
        # Shared symplectic — broadcast over batch
        S2 = torch.eye(2 * N, dtype=torch.float64)
        S2[idx.unsqueeze(1), idx.unsqueeze(0)] = S
        # cov: (B, 2N, 2N); S2: (2N, 2N) → broadcast
        cov_new = S2 @ cov @ S2.T  # (B, 2N, 2N)
        mu_new = (S2 @ mu.unsqueeze(-1)).squeeze(-1)  # (B, 2N)
    else:
        # Per-sample symplectic: S shape (B, 2k, 2k)
        S2 = (torch.eye(2 * N, dtype=torch.float64).unsqueeze(0).expand(B, -1, -1).clone())
        row_idx = idx.unsqueeze(1).expand(B, -1, idx.shape[0])  # (B, 2k, 2k)
        col_idx = idx.unsqueeze(0).expand(B, idx.shape[0], -1)  # (B, 2k, 2k)
        batch_idx = (torch.arange(B, dtype=torch.long).unsqueeze(
            1).unsqueeze(2).expand(B, idx.shape[0], idx.shape[0]))
        S2 = S2.clone()
        S2[batch_idx, row_idx, col_idx] = S
        # Batched matmul: (B, 2N, 2N)
        cov_new = torch.bmm(torch.bmm(S2, cov), S2.transpose(1, 2))
        mu_new = torch.bmm(S2, mu.unsqueeze(-1)).squeeze(-1)

    return cov_new, mu_new


def displacement_torch_batch(state, wire, alpha_real, alpha_imag, num_wires):
    """
    alpha_real, alpha_imag: (B,) — per-sample displacements.
    Displacement only shifts means; cov is unchanged.
    """
    cov, mu = state
    N = num_wires
    dx = alpha_real * (2 * hbar) ** 0.5  # (B,)
    dp = alpha_imag * (2 * hbar) ** 0.5  # (B,)

    mu = mu.clone()
    mu[:, wire] = mu[:, wire] + dx
    mu[:, wire + N] = mu[:, wire + N] + dp
    return cov, mu


# def squeezing_torch_batch(state, wire, z_real, z_imag, num_wires):
#     """
#     r, phi: (B,) — per-sample squeezing params.
#     Builds a batched (B, 2, 2) symplectic and delegates to apply_symplectic_batch.
#     """
#     r = torch.sqrt(z_real**2 + z_imag**2)          # (B,)
#     phi = torch.atan2(z_imag, z_real)               # (B,)

#     cp, sp = torch.cos(phi), torch.sin(phi)  # (B,)
#     ch, sh = torch.cosh(r), torch.sinh(r)  # (B,)
#     # Stack into (B, 2, 2)
#     row0 = torch.stack([ch - cp * sh, -sp * sh], dim=-1)  # (B, 2)
#     row1 = torch.stack([-sp * sh, ch + cp * sh], dim=-1)  # (B, 2)
#     S = torch.stack([row0, row1], dim=1)  # (B, 2, 2)

#     return apply_symplectic_batch(state, S, [wire], num_wires)

def squeezing_torch_batch(state, wire, z_real, z_imag, num_wires):
    """
    Batched squeezing using Cartesian squeezing parameters.

    z_real, z_imag: (B,)

    The squeezing symplectic is constructed as a matrix exponential,
    avoiding the singular Cartesian -> polar conversion at z = 0.
    """

    # Construct the squeezing generator K directly from (z_real, z_imag). This avoids the
    # singular polar conversion to (r, phi). Also, since K^2 = (z_real^2 + z_imag^2) I,
    # exp(K) analytically produces the required cosh(r)/sinh(r) squeezing terms,
    # while matrix_exp differentiates directly through z_real and z_imag, including at (0,0),
    # without explicitly computing r via sqrt. Therefore no backprop issues.

    K = torch.stack([
        torch.stack([-z_real, -z_imag], dim=-1),
        torch.stack([-z_imag,  z_real], dim=-1),
    ], dim=1)

    # (B, 2, 2)
    S = torch.linalg.matrix_exp(K)

    if not torch.isfinite(S).all():
        raise RuntimeError(
            f"S contains NaN/Inf: "
            f"z_real={z_real}, z_imag={z_imag}"
        )

    return apply_symplectic_batch(state, S, [wire], num_wires)


def rotation_torch_batch(state, wire, phi, num_wires):
    """
    phi: scalar tensor (shared across batch) — rotation is circuit-level, not data-dependent.
    """
    c, s = torch.cos(phi), torch.sin(phi)
    S = torch.stack([torch.stack([c, -s]), torch.stack([s, c])])  # (2, 2)
    return apply_symplectic_batch(state, S, [wire], num_wires)


def beamsplitter_torch_batch(state, wire1, wire2, theta, phi, num_wires):
    """
    theta, phi: scalar tensors — shared across batch.
    """
    cp, sp = torch.cos(phi), torch.sin(phi)
    ct, st = torch.cos(theta), torch.sin(theta)
    Z = torch.zeros((), dtype=torch.float64)
    S = torch.stack(
        [torch.stack([ct, -cp * st, Z, -st * sp]),
         torch.stack([cp * st, ct, -st * sp, Z]),
         torch.stack([Z, st * sp, ct, -cp * st]),
         torch.stack([st * sp, Z, cp * st, ct]),]
    )  # (4, 4)
    return apply_symplectic_batch(state, S, [wire1, wire2], num_wires)


def cx_torch_batch(state, wire1, wire2, s, num_wires):
    """
    Batched CV CX (SUM) gate.
    Checked against simple pennylane example
    Sends:
        x1 -> x1
        x2 -> x2 + s*x1
        p1 -> p1 - s*p2
        p2 -> p2
    """
    one = torch.ones_like(s)
    zero = torch.zeros_like(s)

    S = torch.stack(
        [torch.stack([one, zero, zero, zero]),
         torch.stack([s,   one,  zero, zero]),
         torch.stack([zero, zero, one,  -s]),
         torch.stack([zero, zero, zero, one]),]
    )  # (4, 4)

    return apply_symplectic_batch(state, S, [wire1, wire2], num_wires)


# ============================================================
# Measurement helpers
# ============================================================

def mean_photon_number(state):
    """
    Computes mean photon number <n_i> per mode, per sample, from a Gaussian state.

    state: (cov, mu)   <-- matches the rest of q_torch2.py's convention
        cov: (B, 2N, 2N)  xxpp ordering, vacuum = identity, hbar=2 convention
        mu:  (B, 2N)      xxpp ordering: [x1..xN, p1..pN]

    Returns:
        n: (B, N) mean photon number per mode per sample
    """
    cov, mu = state
    B, D = mu.shape
    N = D // 2

    x = mu[:, :N]
    p = mu[:, N:]
    mu_sq = x**2 + p**2

    idx = torch.arange(N, device=cov.device)
    xx = cov[:, idx, idx]
    pp = cov[:, N + idx, N + idx]
    trace_cov = xx + pp

    n = (mu_sq + trace_cov - 2.0) / 4.0
    return n

def _stable_logF(cov_sum, d):
    """
    Computes log F stably via slogdet + linear solve, avoiding an explicit
    matrix inverse and avoiding forming F=det^(-1/4)*exp(...) before logging it.
    """
    # log|det(cov_sum)| directly, no raw det() (avoids over/underflow for large k)
    sign, logdet = torch.linalg.slogdet(cov_sum)
    # cov_sum is positive definite (sum of two cov-like matrices), so sign should be +1
    assert torch.all(sign > 0), "cov_sum not positive definite — check state validity"

    # Solve cov_sum @ x = d instead of forming inv(cov_sum) explicitly
    x = torch.linalg.solve(cov_sum, d)  # (B, 2k, 1)
    quad = (d.transpose(1, 2) @ x).squeeze(-1).squeeze(-1)

    # F = (det^{-1/4} * exp(-0.25*quad))^2 therefore
    # logF = 2*(-0.25*logdet - 0.25*quad) = -0.5*(logdet + quad)
    logF = -0.5 * (logdet + quad)  # (B,)
    return logF

# ============================================================
# Parameter Creation
# ============================================================

def make_params(cfg, rngs):
    """
    Prepares all parameters in advance, for encoding, bs_substrate and local gates, including
    multiple layers and reuploaded encodings
    """
    params = {}
    pairs = []
    gen = rngs["circuit"]
    n_wires = cfg.circuit.wires
    n_layers = cfg.circuit.layers
    arch = cfg.circuit.bs_substrate_type
    lg = cfg.circuit.lg_substrate_type

    # ---------------------------------------------------------
    # Encoding
    # ---------------------------------------------------------
    if cfg.training.paradigm == "unsupervised":
        # params["enc_s_imag"] = init_param("enc_s_imag", (n_wires,), gen)
        if cfg.circuit.reupload:
            raise ValueError("Unsupervised should not be reuploading!")

    elif cfg.training.paradigm == "supervised":
        # supervised is always trainable
        n_upload_layers = n_layers if cfg.circuit.reupload else 1
        for name in ["enc_var_pt", "enc_var_eta", "enc_var_phi"]:
            params[name] = init_param(name, (n_upload_layers, n_wires), gen)
    else:
        raise ValueError(
            f"{cfg.training.paradigm} is not unsupervised/supervised!")

    # ---------------------------------------------------------
    # Substrate
    # ---------------------------------------------------------

    def pairs_ring(n):
        return [(i, (i + 1) % n) for i in range(n)]

    def pairs_chain(n):
        return [(i, i + 1) for i in range(n - 1)]

    def pairs_alltoall(n):
        return [(i, j) for i in range(n) for j in range(i + 1, n)]

    def pairs_star(n):
        return [(0, i) for i in range(1, n)]

    builders = {"ring": pairs_ring, "chain": pairs_chain,
                "alltoall": pairs_alltoall, "star": pairs_star}

    # Number of BS blocks per layer, from config
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

    ############## CX

    # Number of BS blocks per layer, from config
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

    # ---------------------------------------------------------
    # Local Gates
    # ---------------------------------------------------------

    # Number of BS blocks per layer, from config
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

    # ---------------------------------------------------------
    # Supervised measurement
    # ---------------------------------------------------------

    if cfg.training.paradigm == "supervised" and cfg.training.loss_fn in ("logF","logF/n"):
        params["sup_logloss_scale"] = init_param("sup_logloss_scale",())
        params["sup_logloss_shift"] = init_param("sup_logloss_shift",())

    return params, pairs


def resolve_pairs(arch):
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
    Returns a flat pair list for one brick-wall interferometer block.
    """
    pairs = []

    for layer in range(n):
        start = 0 if (layer % 2 == 0) else 1
        pairs.extend((i, i + 1) for i in range(start, n - 1, 2))

    return pairs


# ============================================================
# Circuit Building
# ============================================================

def encoding_torch_batch(state, x, n_wires, cfg, params, reupload_layer=None, op_log=False):
    ops = []
    if not torch.is_tensor(x):
        x = torch.as_tensor(x, dtype=torch.float64)

    if cfg.training.paradigm == "unsupervised":
        for wire in range(n_wires):

            # Loaded data (with scale factors already included)
            eta         = x[:, wire, 0]
            pt_frac     = x[:, wire, 1]
            phi         = x[:, wire, 2]
            
            # Convert polar to real/imag
            d_real = pt_frac * torch.cos(phi)
            d_imag = pt_frac * torch.sin(phi)

            state = displacement_torch_batch(state, wire, d_real, d_imag, n_wires)
            state = squeezing_torch_batch(state, wire, eta, torch.zeros_like(eta), n_wires)
            if op_log:
                ops.extend([("D", [wire]), ("S", [wire])])
    else:
        # supervised:  reuploading layers possible, encoding params are (n_upload_layers, n_wires) -> index [layer, wire]
        for wire in range(n_wires):
            # eta = x[:, wire, 0]
            # pt = x[:, wire, 1]
            # phi = x[:, wire, 2]

            # var_pt_w = params["enc_var_pt"][reupload_layer, wire]
            # var_eta_w = params["enc_var_eta"][reupload_layer, wire]
            # var_phi_w = params["enc_var_phi"][reupload_layer, wire]
            # cos_phi = torch.cos(phi)
            # sin_phi = torch.sin(phi)

            state = displacement_torch_batch(state, wire, var_eta_w * eta, var_pt_w * pt, n_wires)
            state = squeezing_torch_batch(state, wire, var_phi_w * cos_phi, var_phi_w * sin_phi, n_wires)
            if op_log:
                ops.extend([("D", [wire]), ("S", [wire])])

    if op_log:
        return state, ops
    return state


def bs_substrate_batch(state, arch, params, pairs, n_wires, bs_set_index, op_log=False):
    """
    bs_set_index = which repeat of bs gates it is, not necessarily = layers if bs occurs twice in a layer
    """
    ops = []

    k = 0
    for w1, w2 in pairs:
        # To be clamped
        theta = params["bs_theta"][bs_set_index, k]
        # Won't need clamping
        phi = params["bs_phi"][bs_set_index, k]
        state = beamsplitter_torch_batch(state, w1, w2, theta, phi, n_wires)
        if op_log:
            ops.append(("BS", [w1, w2]))
        k += 1

    return (state, ops) if op_log else state


def lg_substrate_batch(state, params, n_wires, lg_set_index, lg_type, op_log=False):
    """
    lg_set_index = which repeat of lg gates it is, not necessarily = layers if lg occurs twice in a layer
    """
    ops = []
    if lg_type == "none":
        return (state, ops) if op_log else state

    for w in range(n_wires):
        for letter in lg_type:
            if letter == "d":
                state = displacement_torch_batch(state,w,
                    params["lg_d_real"][lg_set_index, w].expand(state[1].shape[0]),     # To be clamped
                    params["lg_d_imag"][lg_set_index, w].expand(state[1].shape[0]), n_wires)
                if op_log:
                    ops.append(("D", [w]))
                
            elif letter == "s":
                state = squeezing_torch_batch(state,w,
                    params["lg_s_real"][lg_set_index, w].expand(state[1].shape[0]),           # To be clamped
                    params["lg_s_imag"][lg_set_index, w].expand(state[1].shape[0]), n_wires)
                if op_log:
                    ops.append(("S", [w]))

            elif letter == "r":
                # Won't need clamping
                phi_scalar = params["lg_r_phi"][lg_set_index, w]
                state = rotation_torch_batch(state, w, phi_scalar, n_wires)
                if op_log:
                    ops.append(("R", [w]))
                    
            else:
                raise ValueError(f"{letter} unrecognised!")            

    return (state, ops) if op_log else state


def cx_substrate_batch(state, arch, params, pairs, n_wires, cx_set_index, op_log=False):
    """
    cx_set_index = which repeat of cx gates it is, not necessarily = layers if cx occurs twice in a layer
    """
    ops = []

    k = 0
    for w1, w2 in pairs:
        # To be clamped
        s = params["cx_s"][cx_set_index, k]
        state = cx_torch_batch(state, w1, w2, s, n_wires)
        if op_log:
            ops.append(("CX", [w1, w2]))
        k += 1

    return (state, ops) if op_log else state

# ============================================================
# Loss Functions
# ============================================================

def QAE_loss(cfg, x_batch, params, pairs):
    """
    Batched forward pass. Returns loss per sample: (B,).
    x_batch: (B, n_wires, 3)
    All circuit parameters (BS, LG) are shared; encoding is per-sample.
    """
    n_wires = cfg.circuit.wires
    trash_modes = cfg.circuit.trash_modes
    n_trash = len(trash_modes)
    B = x_batch.shape[0]

    state = vacuum_state_torch_batch(n_wires, B)
    state = encoding_torch_batch(state, x_batch, n_wires, cfg, params)
    
    bs_idx = 0
    lg_idx = 0
    cx_idx = 0
    for layer in range(cfg.circuit.layers):
        for block in cfg.circuit.layer_structure:
            if block == "bs":
                state = bs_substrate_batch(state, cfg.circuit.bs_substrate_type, params, pairs, n_wires, bs_idx)
                bs_idx += 1

            elif block == "lg":
                state = lg_substrate_batch(state, params, n_wires, lg_idx, cfg.circuit.lg_substrate_type)
                lg_idx += 1

            elif block == "cx":
                state = cx_substrate_batch(state, cfg.circuit.bs_substrate_type, params, pairs, n_wires, cx_idx)
                cx_idx += 1 
            else:
                raise ValueError(f"Unknown block type: {block}")

    cov, mu = state  # (B, 2N, 2N), (B, 2N)
    n = mean_photon_number(state)

    # Batched Fidelity (Banchi, uses hbar=1)
    # Trash indices
    indices = trash_modes + [m + n_wires for m in trash_modes]
    idx = torch.tensor(indices, dtype=torch.long)

    # Convert to hbar=1
    mu_2 = mu[:, idx] / (2**0.5)  # (B, 2*n_trash)
    cov_2 = cov[:, idx][:, :, idx] / 2  # (B, 2*n_trash, 2*n_trash)

    # Vacuum reference (shared, hbar=1)
    cov_1 = torch.eye(2 * n_trash, dtype=torch.float64) / 2  # (2*n_trash, 2*n_trash)
    mu_1 = torch.zeros(2 * n_trash, dtype=torch.float64)  # (2*n_trash,)

    # Broadcast cov_1 for addition
    cov_sum = cov_2 + cov_1.unsqueeze(0)  # (B, 2*n_trash, 2*n_trash)
    d = (mu_2 - mu_1).unsqueeze(-1)  # (B, 2k, 1)

    logF = _stable_logF(cov_sum, d) 

    if torch.isnan(logF).any() or torch.isinf(logF).any():
        raise RuntimeError("NaN/Inf in logF, check squeeze/displacement bounds")

    F = torch.exp(logF)  # only materialised where actually needed

    if cfg.training.loss_fn == "F":
        # 1 - F computed via expm1 to avoid cancellation when F is close to 1
        one_minus_F = -torch.expm1(logF)
        return one_minus_F, F, n
    elif cfg.training.loss_fn == "logF":
        score = -logF
        return score, F, n
    elif cfg.training.loss_fn == "logF/n":
        score = -logF / n_trash
        return score, F, n
    else:
        raise ValueError("training.loss_fn can take only 'F', 'logF' or 'logF/n'!")


def CVVC_loss(cfg, x_batch, params, pairs):
    """
    Batched forward pass. Returns loss per sample: (B,).

    x_batch: (B, n_wires, 3)
    All circuit parameters (BS, LG) are shared; encoding is per-sample.
    """
    n_wires = cfg.circuit.wires
    B = x_batch.shape[0]

    state = vacuum_state_torch_batch(n_wires, B)
    state = encoding_torch_batch(state, x_batch, n_wires, cfg, params, reupload_layer=0)

    bs_idx = 0
    lg_idx = 0
    for layer in range(cfg.circuit.layers):
        for block in cfg.circuit.layer_structure:
            if block == "bs":
                state = bs_substrate_batch(state, cfg.circuit.bs_substrate_type, params, pairs, n_wires, bs_idx)
                bs_idx += 1

            elif block == "lg":
                state = lg_substrate_batch(state, params, n_wires, lg_idx, cfg.circuit.lg_substrate_type)
                lg_idx += 1
            else:
                raise ValueError(f"Unknown block type: {block}")

        if cfg.circuit.reupload and layer < cfg.circuit.layers - 1:
            state = encoding_torch_batch(state, x_batch, n_wires, cfg, params, reupload_layer=layer + 1)

    # (B, 2N, 2N), (B, 2N)
    cov, mu = state
    n = mean_photon_number(state)

    # Full state fidelity — no indexing/reduction, use everything
    mu_2 = mu / (2**0.5)                    # (B, 2N)
    cov_2 = cov / 2                          # (B, 2N, 2N)
    N2 = mu.shape[1]  # 2 * n_wires

    # Vacuum reference over the FULL state
    cov_1 = torch.eye(N2, dtype=torch.float64) / 2  # (2N, 2N)
    mu_1 = torch.zeros(N2, dtype=torch.float64)      # (2N,)

    cov_sum = cov_2 + cov_1.unsqueeze(0)  # (B, 2N, 2N)
    d = (mu_2 - mu_1).unsqueeze(-1)  # (B, 2N, 1)

    logF = _stable_logF(cov_sum, d)

    if torch.isnan(logF).any() or torch.isinf(logF).any():
        raise RuntimeError("NaN/Inf in logF, check squeeze/displacement bounds")

    if cfg.training.loss_fn == "F":
        one_minus_F = -torch.expm1(logF)
        return one_minus_F, n

    elif cfg.training.loss_fn == "logF":
        raw_score = -logF  # in [0, inf) — NOT a valid logit on its own
        logit = params["sup_logloss_scale"] * raw_score + params["sup_logloss_shift"]
        return logit, n

    elif cfg.training.loss_fn == "logF/n":
        raw_score = -logF / n_wires
        logit = params["sup_logloss_scale"] * raw_score + params["sup_logloss_shift"]
        return logit, n

    else:
        raise ValueError("training.loss_fn can take only 'F', 'logF' or 'logF/n'!")


def get_op_log(cfg, params, pairs):
    with torch.no_grad():
        n_wires = cfg.circuit.wires
        dummy_x = torch.zeros(1, n_wires, 3, dtype=torch.float64)  # (1, N, 3)

        state = vacuum_state_torch_batch(n_wires, batch_size=1)

        state, enc_ops = encoding_torch_batch(state, dummy_x, n_wires, cfg, params, reupload_layer=0, op_log=True)
        sections = [enc_ops]

        bs_idx = 0
        lg_idx = 0
        cx_idx = 0
        for layer in range(cfg.circuit.layers):
            for block in cfg.circuit.layer_structure:
                if block == "bs":
                    state, bs_ops = bs_substrate_batch(state, cfg.circuit.bs_substrate_type, params, pairs, n_wires, bs_idx, op_log=True)
                    bs_idx += 1
                    sections.append(bs_ops)

                elif block == "lg":
                    state, lg_ops = lg_substrate_batch(state, params, n_wires, lg_idx, cfg.circuit.lg_substrate_type, op_log=True)
                    lg_idx += 1
                    sections.append(lg_ops)
                elif block == "cx":
                    state, cx_ops = cx_substrate_batch(state, cfg.circuit.bs_substrate_type, params, pairs, n_wires, cx_idx, op_log=True)
                    cx_idx += 1
                    sections.append(cx_ops)
                else:
                    raise ValueError(f"Unknown block type: {block}")

            if cfg.circuit.reupload and layer < cfg.circuit.layers - 1:
                state, reup_ops = encoding_torch_batch(state, dummy_x, n_wires, cfg, params, reupload_layer=layer + 1, op_log=True)
                sections.append(reup_ops)
  
        return sections
