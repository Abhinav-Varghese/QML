import torch
import math

def uniform_init(shape, gen, low, high):
    """
    Initialises uniformly across range
    """
    return torch.empty(shape).uniform_(low, high, generator=gen)

def phase_init(shape, gen):
    """Unconstrained/periodic phase param: uniform over a full period."""
    return torch.empty(shape).uniform_(0, 2 * torch.pi, generator=gen)

def constant_init(shape, value):
    """Initialises every element to a fixed constant value (no RNG needed)."""
    return torch.full(shape, value, dtype=torch.float64)

# Contains parameter names, bounds and inits
PARAM_SPECS = {
    # Affine transformation for log
    "sup_logloss_scale": {"bounds": None,                    "init": lambda shape, gen: constant_init(shape, 1.0)}, # always 1 to start
    "sup_logloss_shift": {"bounds": None,                    "init": lambda shape, gen: constant_init(shape, 0.0)}, # always 0 to start

    # Encoding params (supervised)
    "enc_var_pt":        {"bounds": (0.0,  2.0),             "init": lambda shape, gen: uniform_init(shape, gen, 0.0,  2.0)}, # * data (sup)
    "enc_var_eta":       {"bounds": (0.0,  2.0),             "init": lambda shape, gen: uniform_init(shape, gen, 0.0,  2.0)}, # * data (sup)
    "enc_var_phi":       {"bounds": (0.0,  2.0),             "init": lambda shape, gen: uniform_init(shape, gen, 0.0,  2.0)}, # * data (sup)

    # BS params
    "bs_theta":          {"bounds": None,                    "init": lambda shape, gen: phase_init(shape, gen)},
    "bs_phi":            {"bounds": None,                    "init": lambda shape, gen: phase_init(shape, gen)},
    # "bs_theta": {"bounds": (math.pi / 4, math.pi / 4),     "init": lambda shape, gen: constant_init(shape, math.pi / 4)},
    # "bs_phi":   {"bounds": (math.pi / 2, math.pi / 2),     "init": lambda shape, gen: constant_init(shape, math.pi / 2)},

    # CX params
    "cx_s":              {"bounds": (-2, 2.0),                "init": lambda shape, gen: uniform_init(shape, gen, -1, 1.0)},

    # Initialise trainables below to small values, to prevent large initial photon numbers and F~0
    # LG params
    "lg_r_phi":          {"bounds": None,                    "init": lambda shape, gen: phase_init(shape, gen)},
    "lg_s_real":         {"bounds": (-2, 2.0),             "init": lambda shape, gen: uniform_init(shape, gen, -1, 1.0)},
    "lg_s_imag":         {"bounds": (0, 0),                  "init": lambda shape, gen: constant_init(shape, 0)},
    "lg_d_real":         {"bounds": (-2, 2),               "init": lambda shape, gen: uniform_init(shape, gen, -1, 1.0)},
    "lg_d_imag":         {"bounds": (0, 0),                  "init": lambda shape, gen: constant_init(shape, 0)},
}

def init_param(name, shape, gen=None):
    spec = PARAM_SPECS[name]
    value = spec["init"](shape, gen)
    bounds = spec["bounds"]

    # A parameter is frozen if its lower and upper bounds are equal
    # (e.g. (0, 0) for the imaginary local-gate parts)
    if bounds is None:
        is_frozen = False
    else:
        lower, upper = bounds
        is_frozen = (lower == upper)

    return torch.nn.Parameter(value, requires_grad=not is_frozen)

def clamp_params(params):
    with torch.no_grad():
        for name, p in params.items():

            if name not in PARAM_SPECS:
                raise KeyError(
                    f"Param '{name}' not found in PARAM_BOUNDS! Add bounds or set to None if intentionally unbounded!")

            bounds = PARAM_SPECS[name]["bounds"]
            if bounds is not None:
                p.data.clamp_(bounds[0], bounds[1])
