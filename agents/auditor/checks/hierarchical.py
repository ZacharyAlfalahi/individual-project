"""
hierarchical.py — the cross-strategy hierarchical model: magnitude, heterogeneity
and prevalence (§8.2).

The only component that formally evaluates whether per-strategy effects generalise.
Two levels, with the crucial asymmetry (§8.2.1):

    Ê_{i,s} | E_{i,s} ~ N(E_{i,s}, v_{i,s})   measurement — v KNOWN (from the bootstrap)
    E_{i,s}          ~ N(μ_i, τ_i²)           population  — τ ESTIMATED, diagonal (D-A32)

A one-level model treats the noisy estimate as the truth, so τ_i is inflated by
measurement noise (§8.2.1); the measurement level is what makes a wide τ a finding
rather than an artefact. The population covariance is DIAGONAL — an unrestricted
5×5 T is ~20 parameters from ~5 vectors and would report the prior (D-A32) — so
the model factorises across coordinates.

Two estimands, two denominators (§8.2.3, D-A31):
  * μ_i, τ_i  — over SUSCEPTIBLE strategies (runnable and NOT a proven no-op);
    labelled "conditional magnitude among susceptible strategies".
  * prevalence  π̃^corpus_i(ϑ) = (1/n_i) Σ_s P(|E_{i,s}| > ϑ | data), denominator =
    ALL runnable strategies within COMPLETE audits. A proven no-op contributes 0
    to the numerator and 1 to the denominator — a degenerate posterior at zero
    (§8.2.3), the honest contribution, NOT an exclusion.

Fitted with a Metropolis-within-Gibbs sampler (numpy only): the measurement
variances decouple the coordinates given the diagonal population, so each
coordinate is the classic hierarchical normal ("eight schools"), fit
independently. A wide τ posterior at S≈5 under a weakly-informative half-Normal
prior is the correct answer, and reporting it wide is honest (§8.2.4).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class StrategyEffect:
    """One strategy's estimate of one coordinate for the hierarchy. `is_no_op` marks
    a proven no-op (degenerate posterior at zero — enters prevalence as P=0 exactly)."""

    strategy_label: str
    estimate: float          # Ê_{i,s}
    variance: float          # v_{i,s} (measurement, from the bootstrap)
    is_no_op: bool = False


@dataclass(frozen=True)
class HierarchyPriors:
    mu_scale: float = 1.0        # μ ~ N(0, mu_scale²), weakly informative
    tau_scale: float = 0.1       # τ ~ HalfNormal(tau_scale), weakly informative (§8.2.4)


@dataclass(frozen=True, eq=False)
class CoordinateHierarchy:
    coordinate: object
    n_runnable: int
    n_susceptible: int
    mu_mean: float
    mu_ci_low: float
    mu_ci_high: float
    tau_mean: float
    tau_ci_low: float
    tau_ci_high: float
    prevalence: float            # π̃^corpus_i(ϑ)
    vartheta: float

    def to_dict(self) -> dict:
        return {
            "n_runnable": self.n_runnable,
            "n_susceptible": self.n_susceptible,
            "mu_mean": self.mu_mean,
            "mu_ci": [self.mu_ci_low, self.mu_ci_high],
            "tau_mean": self.tau_mean,
            "tau_ci": [self.tau_ci_low, self.tau_ci_high],
            "prevalence": self.prevalence,
            "vartheta": self.vartheta,
            "label_note": "μ, τ are conditional on susceptibility; prevalence spans all runnable",
        }


def _log_tau_posterior(tau: float, resid_sq_sum: float, n: int, tau_scale: float) -> float:
    """log p(τ | {E_s}, μ) ∝ Σ_s logN(E_s; μ, τ²) + logHalfNormal(τ; tau_scale)."""
    if tau <= 0:
        return -math.inf
    ll = -0.5 * n * math.log(2 * math.pi * tau * tau) - resid_sq_sum / (2 * tau * tau)
    lprior = -tau * tau / (2 * tau_scale * tau_scale)
    return ll + lprior


def fit_coordinate_hierarchy(
    effects: Sequence[StrategyEffect],
    *,
    vartheta: float,
    coordinate: object = None,
    priors: HierarchyPriors | None = None,
    n_iter: int = 3000,
    burn: int = 1000,
    seed: int = 0,
) -> CoordinateHierarchy:
    """Fit one coordinate's hierarchical normal model and compute prevalence.

    μ, τ are estimated over SUSCEPTIBLE strategies; prevalence is over ALL runnable
    (no-ops contribute exactly 0 to the numerator, 1 to the denominator)."""
    priors = priors or HierarchyPriors()
    runnable = list(effects)
    n_runnable = len(runnable)
    susceptible = [e for e in runnable if not e.is_no_op]
    S = len(susceptible)

    if S == 0:
        # every runnable strategy is a proven no-op => prevalence 0, μ/τ undefined.
        return CoordinateHierarchy(
            coordinate, n_runnable, 0, float("nan"), float("nan"), float("nan"),
            float("nan"), float("nan"), float("nan"), 0.0, vartheta,
        )

    y = np.array([e.estimate for e in susceptible], dtype=float)
    v = np.array([max(e.variance, 1e-12) for e in susceptible], dtype=float)
    rng = np.random.default_rng(seed)

    # Initialise.
    mu = float(np.mean(y))
    tau = max(float(np.std(y)) if S > 1 else priors.tau_scale, 1e-3)
    E = y.copy()

    mu_draws, tau_draws = [], []
    E_draws = np.empty((n_iter - burn, S))
    tau_step = 0.3  # log-scale RW step

    for it in range(n_iter):
        # 1. E_s | μ, τ, Ê_s  (conjugate normal)
        prec = 1.0 / v + 1.0 / (tau * tau)
        post_var = 1.0 / prec
        post_mean = post_var * (y / v + mu / (tau * tau))
        E = rng.normal(post_mean, np.sqrt(post_var))

        # 2. μ | {E_s}, τ  (conjugate normal, prior N(0, mu_scale²))
        prior_prec = 1.0 / (priors.mu_scale ** 2)
        mu_prec = S / (tau * tau) + prior_prec
        mu_var = 1.0 / mu_prec
        mu_mean = mu_var * (np.sum(E) / (tau * tau))
        mu = float(rng.normal(mu_mean, math.sqrt(mu_var)))

        # 3. τ | {E_s}, μ  (Metropolis on log τ, half-Normal prior)
        resid_sq = float(np.sum((E - mu) ** 2))
        cur_lp = _log_tau_posterior(tau, resid_sq, S, priors.tau_scale)
        prop = tau * math.exp(rng.normal(0.0, tau_step))
        prop_lp = _log_tau_posterior(prop, resid_sq, S, priors.tau_scale)
        # log-scale RW Jacobian: + log(prop) - log(tau)
        if math.log(rng.uniform()) < (prop_lp - cur_lp + math.log(prop) - math.log(tau)):
            tau = prop

        if it >= burn:
            mu_draws.append(mu)
            tau_draws.append(tau)
            E_draws[it - burn] = E

    mu_arr = np.array(mu_draws)
    tau_arr = np.array(tau_draws)

    # Prevalence: susceptible strategies contribute P(|E_s|>ϑ) from their posterior
    # E draws; no-ops contribute 0. Denominator = all runnable.
    p_material = np.mean(np.abs(E_draws) > vartheta, axis=0)  # per susceptible strategy
    prevalence = float(np.sum(p_material)) / n_runnable

    lo_mu, hi_mu = np.percentile(mu_arr, [2.5, 97.5])
    lo_tau, hi_tau = np.percentile(tau_arr, [2.5, 97.5])
    return CoordinateHierarchy(
        coordinate=coordinate,
        n_runnable=n_runnable,
        n_susceptible=S,
        mu_mean=float(np.mean(mu_arr)),
        mu_ci_low=float(lo_mu), mu_ci_high=float(hi_mu),
        tau_mean=float(np.mean(tau_arr)),
        tau_ci_low=float(lo_tau), tau_ci_high=float(hi_tau),
        prevalence=prevalence,
        vartheta=vartheta,
    )


def fit_hierarchy(
    corpus: Mapping[object, Sequence[StrategyEffect]],
    *,
    vartheta: float,
    priors: HierarchyPriors | None = None,
    n_iter: int = 3000,
    burn: int = 1000,
    seed: int = 0,
) -> dict:
    """Fit every coordinate's hierarchy. `corpus` maps a coordinate (e.g. a
    first-order toggle) to its per-strategy effects (COMPLETE audits only — the
    caller enforces the conditioning rule, §8.2.3b)."""
    out: dict = {}
    for j, (coord, effects) in enumerate(corpus.items()):
        out[coord] = fit_coordinate_hierarchy(
            effects, vartheta=vartheta, coordinate=coord, priors=priors,
            n_iter=n_iter, burn=burn, seed=seed + j,
        )
    return out
