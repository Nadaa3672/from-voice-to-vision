"""
Real-coded Genetic Algorithm (GA) — maximises the objective in [0,1]^dim.

Ingredients:
  * tournament selection
  * BLX-alpha (blend) crossover
  * Gaussian mutation with clipping to [0,1]
  * elitism (best individuals carried over unchanged)
"""
import numpy as np


def _tournament(pop, fit, rng, t=3):
    idx = rng.integers(0, len(pop), size=t)
    return pop[idx[np.argmax(fit[idx])]].copy()


def _blx_crossover(p1, p2, rng, alpha=0.3):
    lo = np.minimum(p1, p2)
    hi = np.maximum(p1, p2)
    d = hi - lo
    c1 = rng.uniform(lo - alpha * d, hi + alpha * d)
    c2 = rng.uniform(lo - alpha * d, hi + alpha * d)
    return np.clip(c1, 0, 1), np.clip(c2, 0, 1)


def _mutate(ind, rng, rate=0.2, sigma=0.1):
    mask = rng.random(len(ind)) < rate
    ind = ind + mask * rng.normal(0, sigma, size=len(ind))
    return np.clip(ind, 0, 1)


def optimize(objective, dim, pop_size=6, n_iter=6,
             cx_rate=0.9, mut_rate=0.2, mut_sigma=0.1, elite=1, tournament=3,
             seed=42, verbose=True, seed_u=None):
    rng = np.random.default_rng(seed)
    P = rng.random((pop_size, dim))
    if seed_u is not None:                       # seed one individual with the good default
        P[0] = np.clip(np.asarray(seed_u, dtype=float), 0.0, 1.0)
    fit = np.array([objective(x) for x in P])

    best_i = int(fit.argmax())
    best_u, best_fit = P[best_i].copy(), float(fit[best_i])
    history = [best_fit]
    if verbose:
        print(f"[GA] gen 0  best={best_fit:.3f}")

    for gen in range(n_iter):
        order = np.argsort(fit)[::-1]
        newP = [P[order[i]].copy() for i in range(elite)]        # elitism
        while len(newP) < pop_size:
            p1, p2 = _tournament(P, fit, rng, tournament), _tournament(P, fit, rng, tournament)
            if rng.random() < cx_rate:
                c1, c2 = _blx_crossover(p1, p2, rng)
            else:
                c1, c2 = p1.copy(), p2.copy()
            newP.append(_mutate(c1, rng, mut_rate, mut_sigma))
            if len(newP) < pop_size:
                newP.append(_mutate(c2, rng, mut_rate, mut_sigma))
        P = np.array(newP)
        fit = np.array([objective(x) for x in P])

        gen_best = int(fit.argmax())
        if fit[gen_best] > best_fit:
            best_u, best_fit = P[gen_best].copy(), float(fit[gen_best])
        history.append(best_fit)
        if verbose:
            print(f"[GA] gen {gen+1}/{n_iter}  best={best_fit:.3f}")

    return {"best_u": best_u, "best_fit": best_fit, "history": history, "name": "GA"}
