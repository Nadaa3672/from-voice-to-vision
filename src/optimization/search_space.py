"""
Shared search space and objective for the bio-inspired optimizers.

All three optimizers (PSO, FSO, GA) work in a normalized [0,1]^D space so they
are directly comparable. `decode` maps a normalized vector to CNN hyper-parameters;
`encode` is the inverse (used to SEED the search with the strong default config).
`make_objective` returns u -> robust validation accuracy (fitness to MAXIMISE).
"""
import numpy as np

from src import config

# name, low, high, log-scale?, integer?
# Bounds kept in a sensible regime (net not too small, lr not too high) so the
# search can't wander into fast-but-weak configurations.
PARAM_SPACE = [
    ("lr",           3e-4, 3e-3, True,  False),
    ("dropout",      0.20, 0.50, False, False),
    ("weight_decay", 1e-5, 1e-3, True,  False),
    ("batch_size",   16,   48,   False, True),
    ("width",        24,   48,   False, True),
]
DIM = len(PARAM_SPACE)

# The known-good default — used to seed the optimizers.
DEFAULT_HP = {"lr": 1e-3, "dropout": 0.3, "weight_decay": 1e-4, "batch_size": 32, "width": 32}


def decode(u) -> dict:
    """Map a normalized vector u in [0,1]^DIM to a hyper-parameter dict."""
    u = np.clip(np.asarray(u, dtype=float), 0.0, 1.0)
    hp = {}
    for value, (name, low, high, is_log, is_int) in zip(u, PARAM_SPACE):
        if is_log:
            lo, hi = np.log10(low), np.log10(high)
            val = 10 ** (lo + value * (hi - lo))
        else:
            val = low + value * (high - low)
        if is_int:
            val = int(round(val))
        hp[name] = val
    return hp


def encode(hp) -> np.ndarray:
    """Inverse of decode: hyper-parameter dict -> normalized vector in [0,1]^DIM."""
    u = []
    for name, low, high, is_log, is_int in PARAM_SPACE:
        v = float(hp[name])
        if is_log:
            frac = (np.log10(v) - np.log10(low)) / (np.log10(high) - np.log10(low))
        else:
            frac = (v - low) / (high - low)
        u.append(np.clip(frac, 0.0, 1.0))
    return np.array(u, dtype=float)


def default_u() -> np.ndarray:
    """Normalized vector of the default config (seed for the search)."""
    return encode(DEFAULT_HP)


def make_objective(splits, epochs: int = 18, patience: int = 5, verbose: bool = False):
    """
    Build the fitness function. Each call trains a short CNN (with augmentation +
    delta channels) and returns the robust validation accuracy (mean of top-3 epochs).
    Cached by rounded vector to avoid recomputation. Requires torch (Colab).
    """
    from src.models import cnn
    import torch

    cache = {}
    history_evals = []

    def objective(u):
        key = tuple(np.round(np.asarray(u, dtype=float), 3))
        if key in cache:
            return cache[key]
        hp = decode(u)
        out = cnn.train_cnn(splits, hp, epochs=epochs, patience=patience,
                            deltas=True, augment=True, verbose=False)
        fit = out["robust_val_acc"]
        cache[key] = fit
        history_evals.append({"hp": hp, "val_acc": fit})
        if verbose:
            print(f"   eval  robust_val={fit:.3f}  hp={hp}")
        del out
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return fit

    objective.cache = cache
    objective.history_evals = history_evals
    return objective
