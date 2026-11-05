import math
import sys
from collections import namedtuple
from pathlib import Path

import torch

_root = Path(__file__).parent.parent
sys.path.append(str(_root / "jointCode"))
sys.path.append(str(_root / "quantumCode"))
torch.set_default_dtype(torch.float64)

try:
    from q_params import init_param
except ImportError:  # lets the simulator be imported/tested without the rest of the project
    init_param = None

##############
# Differentiable Fock-space CV simulator in pure torch (no PennyLane / Strawberry Fields).
#
# * The state is pure (all gates are unitary, input is vacuum): a complex tensor of shape
#   (B, c, c, ..., c), one axis per mode, c = cfg.circuit.cutoff_dim. Memory is B * c^N * 16 bytes.
# * Every gate is a (truncated) matrix applied to one or two axes, so autograd gives exact gradients
#   of the *truncated* circuit with a single backward pass, batched over samples.
# * Conventions match PennyLane (hbar = 2): D(a)=exp(a a^dag - a* a), S(z)=exp((z* a^2 - z a^dag^2)/2),
#   R(phi)=exp(i phi n), BS(theta,phi)=exp(theta(e^{i phi} a b^dag - e^{-i phi} a^dag b)),
#   CX(s)=exp(-i s x(x)p / hbar), Kerr(k)=exp(i k n^2), CubicPhase(g)=exp(i g x^3 / (3 hbar)).
# * Truncation: D, S, Kerr, R, CubicPhase and CX are built in an enlarged space (cutoff + pad) and then cut to
#   cutoff x cutoff, which keeps their matrix elements accurate for states well inside the cutoff. BS is
#   exponentiated directly on cutoff^2, so it is exact for total photon number < cutoff. Check
#   truncation_error(...) -- a leak of norm means the cutoff is too small for your encoding/gates.
#
# Circuit blocks in cfg.circuit.layer_structure: "bs", "lg", "cx", "kerr", "cp".
# Parameter names (build_params): enc_var_{pt,eta,phi}, bs_theta, bs_phi, cx_s, lg_d_{real,imag},
# lg_s_{real,imag}, lg_r_phi, kerr_kappa, cp_gamma, sup_logloss_{scale,shift}.
##############

HBAR = 2.0
CDT = torch.complex128


# ===================================================================
# Gate matrices
# ===================================================================

class FockBackend:
    """Precomputed constants + differentiable gate matrices for a given cutoff."""

    def __init__(self, cutoff, pad=None, device="cpu"):
        c = int(cutoff)
        pad = c if pad is None else int(pad)
        cb = c + pad
        self.c, self.cb, self.device = c, cb, device

        n_big = torch.arange(cb, dtype=torch.float64, device=device)
        a_big = torch.diag(torch.sqrt(n_big[1:]), diagonal=1).to(CDT)
        ad_big = a_big.T.contiguous()
        self.a_big, self.ad_big = a_big, ad_big
        self.aa_big = a_big @ a_big
        self.adad_big = ad_big @ ad_big
        self.n = n_big[:c]  # photon numbers on the kept space (real)

        s = math.sqrt(HBAR / 2)
        x_big = s * (a_big + ad_big)
        p_big = -1j * s * (a_big - ad_big)
        self.x_lam, self.x_Q = torch.linalg.eigh(x_big.real)  # real symmetric
        self.p_mu, self.p_R = torch.linalg.eigh(p_big)        # complex hermitian

        a, ad = a_big[:c, :c], ad_big[:c, :c]
        self.bs_ab = torch.kron(a, ad)  # a (x) b^dag
        self.bs_ba = torch.kron(ad, a)  # a^dag (x) b

    # ---- single-mode, dense (..., c, c); params may be scalars or (B,) ----
    def displacement(self, re, im):
        alpha = torch.complex(re, im)[..., None, None]
        G = alpha * self.ad_big - alpha.conj() * self.a_big
        return torch.linalg.matrix_exp(G)[..., :self.c, :self.c]

    def squeezing(self, zr, zi):
        # Cartesian z = zr + i zi, no polar conversion -> gradients are fine at z = 0
        z = torch.complex(zr, zi)[..., None, None]
        G = 0.5 * (z.conj() * self.aa_big - z * self.adad_big)
        return torch.linalg.matrix_exp(G)[..., :self.c, :self.c]

    def cubic_phase(self, gamma):
        phase = torch.exp(1j * gamma[..., None] * self.x_lam ** 3 / (3 * HBAR))  # (..., cb)
        Q = self.x_Q.to(CDT)
        V = (Q * phase[..., None, :]) @ Q.T
        return V[..., :self.c, :self.c]

    # ---- single-mode, diagonal (..., c) ----
    def rotation(self, phi):
        return torch.exp(1j * phi[..., None] * self.n)

    def kerr(self, kappa):
        return torch.exp(1j * kappa[..., None] * self.n ** 2)

    # ---- two-mode, (c, c, c, c) indexed [a, b, A, B] (out a,b; in A,B), scalar params ----
    def beamsplitter(self, theta, phi):
        G = theta * (torch.exp(1j * phi) * self.bs_ab - torch.exp(-1j * phi) * self.bs_ba)
        return torch.linalg.matrix_exp(G).reshape(self.c, self.c, self.c, self.c)

    def cx(self, s):
        # exp(-i s x1 p2 / hbar) via eigenbases of x and p (built in the padded space, cut to c)
        E = torch.exp(-1j * s * self.x_lam[:, None] * self.p_mu[None, :] / HBAR)  # (cb, cb)
        Qs = self.x_Q[:self.c].to(CDT)
        Rs = self.p_R[:self.c]
        M = torch.einsum("ai,ij,Ai->aAj", Qs, E, Qs.conj())
        return torch.einsum("aAj,bj,Bj->abAB", M, Rs, Rs.conj())


_BACKENDS = {}


def get_backend(cfg, device="cpu"):
    c = int(cfg.circuit.cutoff_dim)
    pad = getattr(cfg.circuit, "expm_pad", None)
    key = (c, pad, str(device))
    if key not in _BACKENDS:
        _BACKENDS[key] = FockBackend(c, pad, device)
    return _BACKENDS[key]


# ===================================================================
# Applying gates to the state  psi: (B, c, ..., c)
# ===================================================================

def apply_dense_1(psi, U, w):
    """U: (c, c) shared or (B, c, c) per-sample."""
    m = psi.movedim(w + 1, -1)
    if U.dim() == 2:
        out = m @ U.transpose(-1, -2)
    else:
        B = m.shape[0]
        out = torch.bmm(m.reshape(B, -1, m.shape[-1]), U.transpose(1, 2)).reshape(m.shape)
    return out.movedim(-1, w + 1)


def apply_diag_1(psi, d, w):
    """d: (c,) shared or (B, c) per-sample."""
    m = psi.movedim(w + 1, -1)
    if d.dim() == 2:
        d = d.reshape(d.shape[0], *([1] * (m.dim() - 2)), d.shape[-1])
    return (m * d).movedim(-1, w + 1)


def apply_2(psi, U4, w1, w2):
    m = psi.movedim((w1 + 1, w2 + 1), (-2, -1))
    out = torch.einsum("...AB,abAB->...ab", m, U4)
    return out.movedim((-2, -1), (w1 + 1, w2 + 1))


# ===================================================================
# Parameters
# ===================================================================

def get_pairs(arch, n):
    if arch == "trainable_int":
        pairs = []
        for layer in range(n):
            start = 0 if layer % 2 == 0 else 1
            pairs.extend((i, i + 1) for i in range(start, n - 1, 2))
        return pairs
    if "ring" in arch:
        return [(i, (i + 1) % n) for i in range(n)]
    if "chain" in arch:
        return [(i, i + 1) for i in range(n - 1)]
    if "alltoall" in arch:
        return [(i, j) for i in range(n) for j in range(i + 1, n)]
    if "star" in arch:
        return [(0, i) for i in range(1, n)]
    raise ValueError(f"Unknown circuit arch: {arch}")


def _lg_letters(cfg):
    lg = cfg.circuit.lg_substrate_type
    letters = "" if lg in (None, "none") else lg
    if len(set(letters)) != len(letters):
        raise ValueError("Local gates should use up to 1 of each of 'd','r','s'!")
    for ch in letters:
        if ch not in "dsr":
            raise ValueError(f"'{ch}' not a recognised local gate!")
    return letters


def build_params(cfg, rngs):
    """Creates every trainable tensor. Returns (params, pairs)."""
    if init_param is None:
        raise ImportError("q_params.init_param is needed to create parameters")
    gen = rngs["circuit"]
    N, L = cfg.circuit.wires, cfg.circuit.layers
    ls = list(cfg.circuit.layer_structure)
    params = {}

    if cfg.training.paradigm == "unsupervised":
        if cfg.circuit.reupload:
            raise ValueError("Unsupervised should not be reuploading!")
    elif cfg.training.paradigm == "supervised":
        n_up = L if cfg.circuit.reupload else 1
        for name in ("enc_var_pt", "enc_var_eta", "enc_var_phi"):
            params[name] = init_param(name, (n_up, N), gen)
    else:
        raise ValueError(f"{cfg.training.paradigm} is not unsupervised/supervised!")

    pairs = []
    if any(b in ls for b in ("bs", "cx")):
        pairs = get_pairs(cfg.circuit.bs_substrate_type, N)

    n_bs = L * ls.count("bs")
    if n_bs:
        params["bs_theta"] = init_param("bs_theta", (n_bs, len(pairs)), gen)
        params["bs_phi"] = init_param("bs_phi", (n_bs, len(pairs)), gen)

    n_cx = L * ls.count("cx")
    if n_cx:
        params["cx_s"] = init_param("cx_s", (n_cx, len(pairs)), gen)

    n_lg = L * ls.count("lg")
    for letter in _lg_letters(cfg):
        if n_lg == 0:
            break
        if letter == "d":
            params["lg_d_real"] = init_param("lg_d_real", (n_lg, N), gen)
            params["lg_d_imag"] = init_param("lg_d_imag", (n_lg, N), gen)
        elif letter == "s":
            params["lg_s_real"] = init_param("lg_s_real", (n_lg, N), gen)
            params["lg_s_imag"] = init_param("lg_s_imag", (n_lg, N), gen)
        else:
            params["lg_r_phi"] = init_param("lg_r_phi", (n_lg, N), gen)

    n_kerr = L * ls.count("kerr")
    if n_kerr:
        params["kerr_kappa"] = init_param("kerr_kappa", (n_kerr, N), gen)
    n_cp = L * ls.count("cp")
    if n_cp:
        params["cp_gamma"] = init_param("cp_gamma", (n_cp, N), gen)

    if cfg.training.paradigm == "supervised" and cfg.training.loss_fn in ("logF", "logF/n"):
        params["sup_logloss_scale"] = init_param("sup_logloss_scale", (), gen)
        params["sup_logloss_shift"] = init_param("sup_logloss_shift", (), gen)

    return params, pairs


# ===================================================================
# Circuit description (backend independent) and execution
# ===================================================================

Gate = namedtuple("Gate", ["name", "wires", "args"])
# args: tuple of (param_name, row, col) references; for "ENC" it is (reupload_layer,)


def build_spec(cfg, pairs):
    """List of sections (one per block), each a list of Gate."""
    N, L = cfg.circuit.wires, cfg.circuit.layers
    letters = _lg_letters(cfg)
    reupload = bool(cfg.circuit.reupload)

    def enc(layer):
        return [Gate("ENC", (w,), (layer,)) for w in range(N)]

    spec = [enc(0)]
    cnt = {"bs": 0, "lg": 0, "cx": 0, "kerr": 0, "cp": 0}
    for layer in range(L):
        for block in cfg.circuit.layer_structure:
            if block not in cnt:
                raise ValueError(f"Unknown block type: {block}")
            i = cnt[block]
            sec = []
            if block == "bs":
                for k, (w1, w2) in enumerate(pairs):
                    sec.append(Gate("BS", (w1, w2), (("bs_theta", i, k), ("bs_phi", i, k))))
            elif block == "cx":
                for k, (w1, w2) in enumerate(pairs):
                    sec.append(Gate("CX", (w1, w2), (("cx_s", i, k),)))
            elif block == "lg":
                for w in range(N):
                    for ch in letters:
                        if ch == "d":
                            sec.append(Gate("D", (w,), (("lg_d_real", i, w), ("lg_d_imag", i, w))))
                        elif ch == "s":
                            sec.append(Gate("S", (w,), (("lg_s_real", i, w), ("lg_s_imag", i, w))))
                        else:
                            sec.append(Gate("R", (w,), (("lg_r_phi", i, w),)))
            elif block == "kerr":
                sec = [Gate("K", (w,), (("kerr_kappa", i, w),)) for w in range(N)]
            elif block == "cp":
                sec = [Gate("CP", (w,), (("cp_gamma", i, w),)) for w in range(N)]
            cnt[block] += 1
            spec.append(sec)
        if reupload and layer < L - 1:
            spec.append(enc(layer + 1))
    return spec


def _p(params, ref):
    key, i, j = ref
    return params[key][i, j]


def _apply_gate(psi, g, params, x, cfg, be):
    w = g.wires[0]
    name = g.name

    if name == "ENC":
        eta, pt, phi = x[:, w, 0], x[:, w, 1], x[:, w, 2]
        if cfg.training.paradigm == "unsupervised":
            # displacement alpha = pt * exp(i phi); squeezing z = eta (real, may be negative)
            D = be.displacement(pt * torch.cos(phi), pt * torch.sin(phi))
            S = be.squeezing(eta, torch.zeros_like(eta))
        else:
            layer = g.args[0]
            v_pt = params["enc_var_pt"][layer, w]
            v_eta = params["enc_var_eta"][layer, w]
            v_phi = params["enc_var_phi"][layer, w]
            D = be.displacement(v_eta * eta, v_pt * pt)
            S = be.squeezing(v_phi * torch.cos(phi), v_phi * torch.sin(phi))
        return apply_dense_1(apply_dense_1(psi, D, w), S, w)

    if name == "D":
        return apply_dense_1(psi, be.displacement(_p(params, g.args[0]), _p(params, g.args[1])), w)
    if name == "S":
        return apply_dense_1(psi, be.squeezing(_p(params, g.args[0]), _p(params, g.args[1])), w)
    if name == "R":
        return apply_diag_1(psi, be.rotation(_p(params, g.args[0])), w)
    if name == "K":
        return apply_diag_1(psi, be.kerr(_p(params, g.args[0])), w)
    if name == "CP":
        return apply_dense_1(psi, be.cubic_phase(_p(params, g.args[0])), w)
    if name == "BS":
        U = be.beamsplitter(_p(params, g.args[0]), _p(params, g.args[1]))
        return apply_2(psi, U, g.wires[0], g.wires[1])
    if name == "CX":
        return apply_2(psi, be.cx(_p(params, g.args[0])), g.wires[0], g.wires[1])
    raise ValueError(f"Unknown gate {name}")


def simulate(cfg, x, params, spec, backend=None):
    """Runs the circuit on a batch x (B, N, 3). Returns psi: (B, c, ..., c) complex."""
    x = torch.as_tensor(x, dtype=torch.float64)
    N = cfg.circuit.wires
    backend = backend or get_backend(cfg, x.device)
    c = backend.c
    psi = torch.zeros((x.shape[0],) + (c,) * N, dtype=CDT, device=x.device)
    psi[(slice(None),) + (0,) * N] = 1.0
    for section in spec:
        for g in section:
            psi = _apply_gate(psi, g, params, x, cfg, backend)
    return psi


# ===================================================================
# Measurements
# ===================================================================

def probabilities(psi):
    return psi.real ** 2 + psi.imag ** 2


def mean_photon_number(prob, backend):
    """<n_i> per mode: (B, N)."""
    N = prob.dim() - 1
    out = []
    for w in range(N):
        dims = tuple(d for d in range(1, N + 1) if d != w + 1)
        pw = prob.sum(dim=dims) if dims else prob
        out.append(pw @ backend.n)
    return torch.stack(out, dim=1)


def vacuum_fidelity(prob, wires):
    """<0|rho_wires|0>: probability of vacuum on `wires`, summed over everything else. (B,)"""
    N = prob.dim() - 1
    wires = set(wires)
    idx = (slice(None),) + tuple(0 if w in wires else slice(None) for w in range(N))
    sel = prob[idx]
    return sel.reshape(prob.shape[0], -1).sum(1)


def truncation_error(cfg, x_batch, params, pairs):
    """1 - <psi|psi> per sample: norm lost beyond the cutoff. Keep this ~1e-3 or smaller."""
    with torch.no_grad():
        spec = build_spec(cfg, pairs)
        prob = probabilities(simulate(cfg, x_batch, params, spec))
        return 1.0 - prob.reshape(prob.shape[0], -1).sum(1)


# ===================================================================
# Losses (same signatures / returns as the earlier versions)
# ===================================================================

def _checked(t, what):
    if not torch.isfinite(t).all():
        raise RuntimeError(f"NaN/Inf in {what}, check encoding scales / cutoff_dim")
    return t


def QAE_loss(cfg, x_batch, params, pairs):
    """x_batch: (B, N, 3). Returns (score (B,), F (B,), n (B, N))."""
    trash = list(cfg.circuit.trash_modes)
    spec = build_spec(cfg, pairs)
    be = get_backend(cfg, torch.as_tensor(x_batch).device)
    prob = probabilities(simulate(cfg, x_batch, params, spec, be))
    n = mean_photon_number(prob, be)
    F = _checked(vacuum_fidelity(prob, trash), "F")

    if cfg.training.loss_fn == "F":
        return 1.0 - F, F, n
    logF = torch.log(F.clamp_min(1e-300))
    if cfg.training.loss_fn == "logF":
        return -logF, F, n
    if cfg.training.loss_fn == "logF/n":
        return -logF / len(trash), F, n
    raise ValueError("training.loss_fn can take only 'F', 'logF' or 'logF/n'!")


def CVVC_loss(cfg, x_batch, params, pairs):
    """Full-state vacuum fidelity classifier. Returns (score or logit (B,), n (B, N))."""
    N = cfg.circuit.wires
    spec = build_spec(cfg, pairs)
    be = get_backend(cfg, torch.as_tensor(x_batch).device)
    prob = probabilities(simulate(cfg, x_batch, params, spec, be))
    n = mean_photon_number(prob, be)
    F = _checked(vacuum_fidelity(prob, range(N)), "F")

    if cfg.training.loss_fn == "F":
        return 1.0 - F, n
    logF = torch.log(F.clamp_min(1e-300))
    if cfg.training.loss_fn == "logF":
        raw = -logF
    elif cfg.training.loss_fn == "logF/n":
        raw = -logF / N
    else:
        raise ValueError("training.loss_fn can take only 'F', 'logF' or 'logF/n'!")
    return params["sup_logloss_scale"] * raw + params["sup_logloss_shift"], n


def get_op_log(cfg, params, pairs):
    """Sections of (gate name, wires), as in the earlier versions."""
    sections = []
    for sec in build_spec(cfg, pairs):
        ops = []
        for g in sec:
            if g.name == "ENC":
                ops.extend([("D", [g.wires[0]]), ("S", [g.wires[0]])])
            else:
                ops.append((g.name, list(g.wires)))
        sections.append(ops)
    return sections
