import torch
import math

def uniform_w_init(shape, gen):
    """Sample f uniformly in (1, 1+2π), then invert sigmoid to get w init"""
    f = torch.empty(shape).uniform_(1 + 1e-4, 1 + 2 * torch.pi - 1e-4, generator=gen)
    sig = (f - 1) / (2 * torch.pi)          # sigmoid(w) recovered
    w = torch.log(sig / (1 - sig))          # logit
    return w

def phase_init(shape, gen):
    """Unconstrained/periodic phase param: uniform over a full period."""
    return torch.empty(shape).uniform_(0, 2 * torch.pi, generator=gen)

# Contains parameter names, bounds and inits
PARAM_SPECS = {
    # LG params
    "lg":           {"init": lambda shape, gen: phase_init(shape, gen)},
}

def init_param(name, shape, gen=None):
    return torch.nn.Parameter(PARAM_SPECS[name]["init"](shape, gen))
