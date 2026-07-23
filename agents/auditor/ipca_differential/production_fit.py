"""
production_fit.py — the ONE production estimator rule (spec §5.2, D-A52).

A single call site for producing a fitted state Θ̂. "This exact rule produces Θ̂_N, every
Θ̂_{N\\b}, every calibration fit, and every stability-diagnostic replicate. A diagnostic that
resamples a different estimator estimates a different estimand." Nothing else in the extension
may call ``fit_ipca`` directly — so the estimator can later become best-of-M **without changing
any call site**.

Current form (n_initialisations = 1): a single SVD cold start. The §12.5 fit-timing measurement
decides whether best-of-M is affordable; when it is, ``lambda.n_initialisations`` grows and the
best-of-M branch below is exercised. Either definition; never both (the "compute escape hatch").

Returns a ``FrozenIPCAState`` — the fit is frozen at the point of production, so the rest of the
pipeline can only ever see the immutable, hashed artefact.
"""

from __future__ import annotations

import numpy as np

from agents.auditor.thresholds import IPCALambda
from agents.quant.library.ipca import (
    IPCAFit,
    SuffStats,
    build_sufficient_stats,
    fit_ipca,
    total_r2,
)
from agents.quant.library.ipca_feed import IPCAFeed

from .frozen_state import FrozenIPCAState


def _fit_once(stats: SuffStats, lam: IPCALambda, gamma0=None, F0=None) -> IPCAFit:
    """One ALS fit under λ (restricted α=0). Cold SVD start unless a warm (gamma0, F0) is given."""
    return fit_ipca(
        stats,
        K=lam.factor_count,
        alpha=False,                      # restricted α=0 specification (category 3)
        weighting=lam.month_weighting,
        tol=lam.als_tolerance,
        max_iter=lam.als_max_iter,
        gamma0=gamma0,
        F0=F0,
    )


def _best_of_m(stats: SuffStats, lam: IPCALambda) -> IPCAFit:
    """Best-of-M production rule (§5.2): fit from M initialisations, select the lowest final ALS
    objective (equivalently the highest managed-portfolio total R², which is monotone in −SSR),
    deterministic tie-break by initialisation order. Init 0 is the SVD cold start; inits 1..M-1
    are seeded random orthonormal starts. Not exercised while n_initialisations == 1."""
    T, L, _ = stats.W.shape
    K = lam.factor_count
    candidates: list[tuple[float, int, IPCAFit]] = []
    for j in range(lam.n_initialisations):
        if j == 0:
            fit = _fit_once(stats, lam)                       # SVD cold start
        else:
            seed = lam.initialisation_seeds[j % len(lam.initialisation_seeds)] + j
            rng = np.random.default_rng(seed)
            g0, _ = np.linalg.qr(rng.standard_normal((L, K)))
            f0 = rng.standard_normal((K, T))
            fit = _fit_once(stats, lam, gamma0=g0, F0=f0)
        # ALS objective is monotone decreasing in total_r2; select the max (tie-break: order j).
        candidates.append((total_r2(stats, fit), j, fit))
    # highest total_r2 wins; ties broken by lowest initialisation index (stable, deterministic).
    best = max(candidates, key=lambda c: (c[0], -c[1]))
    return best[2]


def production_fit(feed: IPCAFeed, lam: IPCALambda) -> FrozenIPCAState:
    """Produce a frozen fitted state Θ̂ from a panel-state feed under the §5.2 production rule.
    The SINGLE place a fit is produced in the extension."""
    stats = build_sufficient_stats(feed.Z, feed.R, feed.months)
    if lam.n_initialisations <= 1:
        fit = _fit_once(stats, lam)
    else:
        fit = _best_of_m(stats, lam)
    return FrozenIPCAState.freeze(fit, lam)


def production_fit_seeded(feed: IPCAFeed, lam: IPCALambda, seed: int) -> FrozenIPCAState:
    """Fit from ONE seeded random initialisation (not the SVD cold start). Used ONLY by the
    multi-start range diagnostic to probe ALS local-optimum sensitivity (§5.3) — NOT the production
    estimator (which is ``production_fit``). Distinct local optima can differ in span, and span
    differences move alpha; the multi-start range is precisely that detector."""
    stats = build_sufficient_stats(feed.Z, feed.R, feed.months)
    T, L, _ = stats.W.shape
    rng = np.random.default_rng(seed)
    gamma0, _ = np.linalg.qr(rng.standard_normal((L, lam.factor_count)))
    F0 = rng.standard_normal((lam.factor_count, T))
    return FrozenIPCAState.freeze(_fit_once(stats, lam, gamma0=gamma0, F0=F0), lam)
