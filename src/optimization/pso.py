"""
Particle Swarm Optimization (PSO) — maximises the objective in [0,1]^dim.

Velocity update (classic global-best PSO):
    v = w*v + c1*r1*(pbest - x) + c2*r2*(gbest - x)
"""
import numpy as np


def optimize(objective, dim, n_particles=6, n_iter=6,
             w=0.6, c1=1.5, c2=1.5, vmax=0.3, seed=42, verbose=True):
    rng = np.random.default_rng(seed)
    X = rng.random((n_particles, dim))
    V = (rng.random((n_particles, dim)) * 2 - 1) * 0.1

    fit = np.array([objective(x) for x in X])
    pbest, pbest_fit = X.copy(), fit.copy()
    g = int(pbest_fit.argmax())
    gbest, gbest_fit = pbest[g].copy(), float(pbest_fit[g])
    history = [gbest_fit]
    if verbose:
        print(f"[PSO] init  best={gbest_fit:.3f}")

    for it in range(n_iter):
        r1, r2 = rng.random((n_particles, dim)), rng.random((n_particles, dim))
        V = w * V + c1 * r1 * (pbest - X) + c2 * r2 * (gbest - X)
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
            print(f"[PSO] iter {it+1}/{n_iter}  best={gbest_fit:.3f}")

    return {"best_u": gbest, "best_fit": gbest_fit, "history": history, "name": "PSO"}
