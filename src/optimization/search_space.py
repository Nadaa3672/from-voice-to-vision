"""
Shared search space and objective for the bio-inspired optimizers.

All three optimizers (PSO, FSO, GA) work in a normalized [0,1]^D space so they
are directly comparable. `decode` maps a normalized vector to the CNN
hyper-parameters; `make_objective` returns a function u -> validation accuracy
(the fitness to MAXIMISE), training a short CNN for each candidate.
"""
import numpy as np

from src import config

# name, low, high, log-scale?, integer?
PARAM_SPACE = [
    ("lr",           1e-4, 5e-3, True,  False),
    ("dropout",      0.10, 0.60, False, False),
    ("weight_decay", 1e-6, 1e-3, True,  False),
    ("batch_size",   16,   64,   False, True),
    ("width",        8,    32,   False, True),
]
DIM = len(PARAM_SPACE)


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


def make_objective(splits, epochs: int = 12, patience: int = 4, verbose: bool = False):
    """
    Build the fitness function. Each call trains a short CNN and returns the best
    validation accuracy. Results are cached by rounded vector to avoid recomputation
    (e.g. GA elites). Requires torch (available on Colab).
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
        out = cnn.train_cnn(splits, hp, epochs=epochs, patience=patience, verbose=False)
        # robust fitness (mean of top-3 val epochs) generalises better than a lucky peak
        fit = out["robust_val_acc"]
        cache[key] = fit
        history_evals.append({"hp": hp, "val_acc": fit})
        if verbose:
            print(f"   eval  val_acc={fit:.3f}  hp={hp}")
        # free GPU memory between candidates
        del out
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return fit

    objective.cache = cache
    objective.history_evals = history_evals
    return objective
