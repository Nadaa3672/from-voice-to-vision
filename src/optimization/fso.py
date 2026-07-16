"""
Flock of Starlings Optimization (FSO) — maximises the objective in [0,1]^dim.

Inspired by starling murmurations: each bird coordinates with a FIXED NUMBER of
*topological* nearest neighbours (~6-7, the "magic number" from Ballerini et al.),
independent of their metric distance. On top of the classic PSO pulls toward the
personal best and global best, FSO adds an ALIGNMENT term that matches the average
velocity of a particle's k nearest neighbours — the hallmark of flocking:

    v = w*v + c1*r1*(pbest - x) + c2*r2*(gbest - x) + c3*r3*(mean_neighbour_v - v)

This is what distinguishes FSO from plain PSO and is the novel ingredient of the project.
"""
import numpy as np


def _knn_indices(X, k):
    """For each particle, indices of its k nearest neighbours (topological, by distance)."""
    n = len(X)
    diff = X[:, None, :] - X[None, :, :]
    dist = np.sqrt((diff ** 2).sum(-1))
    np.fill_diagonal(dist, np.inf)
    return np.argsort(dist, axis=1)[:, :k]


def optimize(objective, dim, n_particles=6, n_iter=6,
             w=0.6, c1=1.5, c2=1.5, c3=1.0, k=None, vmax=0.3, seed=42, verbose=True, seed_u=None):
    rng = np.random.default_rng(seed)
    if k is None:
        k = min(6, n_particles - 1)      # starling "magic number", capped by pop size

    X = rng.random((n_particles, dim))
    if seed_u is not None:               # start one bird at the good default
        X[0] = np.clip(np.asarray(seed_u, dtype=float), 0.0, 1.0)
    V = (rng.random((n_particles, dim)) * 2 - 1) * 0.1

    fit = np.array([objective(x) for x in X])
    pbest, pbest_fit = X.copy(), fit.copy()
    g = int(pbest_fit.argmax())
    gbest, gbest_fit = pbest[g].copy(), float(pbest_fit[g])
    history = [gbest_fit]
    if verbose:
        print(f"[FSO] init  best={gbest_fit:.3f}  (k={k} neighbours)")

    for it in range(n_iter):
        nn = _knn_indices(X, k)
        mean_neigh_v = V[nn].mean(axis=1)      # avg velocity of the k neighbours
        r1, r2, r3 = (rng.random((n_particles, dim)) for _ in range(3))
        V = (w * V
             + c1 * r1 * (pbest - X)
             + c2 * r2 * (gbest - X)
             + c3 * r3 * (mean_neigh_v - V))   # alignment with the flock
        V = np.clip(V, -vmax, vmax)
        X = np.clip(X + V, 0.0, 1.0)

        fit = np.array([objective(x) for x in X])
        improved = fit > pbest_fit
        pbest[improved], pbest_fit[improved] = X[improved], fit[improved]
        g = int(pbest_fit.argmax())
        if pbest_fit[g] > gbest_fit:
            gbest, gbest_fit = pbest[g].copy(), float(pbest_fit[g])
        history.append(gbest_fit)
        if verbose:
            print(f"[FSO] iter {it+1}/{n_iter}  best={gbest_fit:.3f}")

    return {"best_u": gbest, "best_fit": gbest_fit, "history": history, "name": "FSO"}
