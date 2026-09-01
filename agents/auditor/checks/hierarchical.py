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

Fitted with a Metropolis-within-Gibbs sampler (numpy only). Two entry points:

  * fit_joint_hierarchy (PRIMARY, D-A32) keeps the FULL k×k bootstrap measurement
    covariance V̂_s, so a strategy's correlated cell effects are modelled jointly and
    sampling noise is not misread as heterogeneity. The off-diagonals couple a
    strategy's coordinates in the E-step; μ_i, τ_i stay per-coordinate because the
    population T is diagonal.
  * fit_coordinate_hierarchy / fit_hierarchy (the diagonal-measurement SENSITIVITY):
    a scalar measurement variance per coordinate decouples the coordinates, so each is
    the classic hierarchical normal ("eight schools") fit independently. This is exactly
    fit_joint_hierarchy restricted to a diagonal V̂_s.

A wide τ posterior at S≈5 under a weakly-informative half-Normal prior is the correct
answer, and reporting it wide is honest (§8.2.4).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

from .bayes import regularise_covariance


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
    # ((ϑ, π̃(ϑ)), ...) neighbouring-threshold sensitivity (§8.2.3/§9); empty when no grid
    # was requested. Contains the headline ϑ when populated.
    prevalence_sweep: tuple = ()

    def to_dict(self) -> dict:
        d = {
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
        if self.prevalence_sweep:
            d["prevalence_sweep"] = [[v, p] for v, p in self.prevalence_sweep]
        return d


def _prevalence_sweep(abs_draw_arrays, n_runnable: int, grid) -> tuple:
    """π̃(ϑ) at each grid ϑ from the per-susceptible |E| posterior draws (§8.2.3/§9): each
    susceptible strategy contributes P(|E|>ϑ), no-ops contribute 0, denominator = n_runnable.
    Empty grid → empty sweep (the headline prevalence is always reported separately)."""
    return tuple(
        (float(v), sum(float(np.mean(a > v)) for a in abs_draw_arrays) / n_runnable)
        for v in grid
    )


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
    vartheta_grid: Sequence[float] = (),
) -> CoordinateHierarchy:
    """Fit one coordinate's hierarchical normal model and compute prevalence.

    μ, τ are estimated over SUSCEPTIBLE strategies; prevalence is over ALL runnable
    (no-ops contribute exactly 0 to the numerator, 1 to the denominator). When
    `vartheta_grid` is non-empty the coordinate also carries π̃(ϑ) at every grid ϑ
    (§8.2.3/§9 sensitivity), recomputed from the same posterior draws."""
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
            prevalence_sweep=tuple((float(v), 0.0) for v in vartheta_grid),
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
    abs_arrays = [np.abs(E_draws[:, j]) for j in range(S)]
    prevalence = sum(float(np.mean(a > vartheta)) for a in abs_arrays) / n_runnable
    prev_sweep = _prevalence_sweep(abs_arrays, n_runnable, vartheta_grid)

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
        prevalence_sweep=prev_sweep,
    )


def fit_hierarchy(
    corpus: Mapping[object, Sequence[StrategyEffect]],
    *,
    vartheta: float,
    priors: HierarchyPriors | None = None,
    n_iter: int = 3000,
    burn: int = 1000,
    seed: int = 0,
    vartheta_grid: Sequence[float] = (),
) -> dict:
    """Fit every coordinate's hierarchy. `corpus` maps a coordinate (e.g. a
    first-order toggle) to its per-strategy effects (COMPLETE audits only — the
    caller enforces the conditioning rule, §8.2.3b)."""
    out: dict = {}
    for j, (coord, effects) in enumerate(corpus.items()):
        out[coord] = fit_coordinate_hierarchy(
            effects, vartheta=vartheta, coordinate=coord, priors=priors,
            n_iter=n_iter, burn=burn, seed=seed + j, vartheta_grid=vartheta_grid,
        )
    return out


@dataclass(frozen=True)
class StrategyVector:
    """One strategy's JOINT estimate over its PRESENT coordinates (k may be < 5 when a
    correction is structurally non-applicable and stripped upstream — e.g. `str` with
    `meas_err` removed → k=4). `covariance` is the k×k bootstrap measurement covariance
    V̂_s (D-A32, off-diagonals kept). `no_op_mask[j]` marks coordinate j a proven no-op
    (degenerate at zero: contributes P=0 to its numerator and 1 to its denominator)."""

    strategy_label: str
    coords: tuple                # length k — coordinate ids, in row/col order of covariance
    estimate: np.ndarray         # (k,)   Ê_{·,s}
    covariance: np.ndarray       # (k, k) V̂_s
    no_op_mask: np.ndarray       # (k,)   bool


@dataclass(frozen=True)
class JointHierarchy:
    """The result of fit_joint_hierarchy: the per-coordinate hierarchies PLUS the recorded
    per-strategy measurement-covariance regularisations (D-A34 — a regularisation applied
    silently is an undocumented researcher degree of freedom, so it is disclosed here just
    as bayes.py discloses its single-strategy one)."""

    coordinates: dict            # coord -> CoordinateHierarchy
    regularisations: dict        # strategy_label -> CovarianceRegularisation

    def to_dict(self) -> dict:
        return {
            "coordinates": {str(c): h.to_dict() for c, h in self.coordinates.items()},
            "regularisations": {k: r.to_dict() for k, r in self.regularisations.items()},
        }


def fit_joint_hierarchy(
    strategies: Sequence[StrategyVector],
    *,
    vartheta: float,
    epsilon: float,
    priors: HierarchyPriors | None = None,
    n_iter: int = 3000,
    burn: int = 1000,
    seed: int = 0,
    vartheta_grid: Sequence[float] = (),
) -> JointHierarchy:
    """Fit the two-level model with the FULL measurement covariance (D-A32):

        Ê_s | E_s ~ N(E_s, V̂_s)   measurement — V̂_s the k×k bootstrap covariance (full)
        E_{i,s}   ~ N(μ_i, τ_i²)   population   — diagonal, so μ_i, τ_i are per-coordinate

    The off-diagonal measurement covariance couples a strategy's coordinates in the E-step;
    the diagonal population keeps μ_i, τ_i per-coordinate. Each V̂_s is regularised by the
    pre-registered symmetrise → eigenvalue-floor-at-ε recipe (D-A34) via the SAME
    `bayes.regularise_covariance` the single-strategy layer uses, so both V̂ consumers floor
    identically; ε is threshold-sourced by the caller (never hardcoded) and the applied
    regularisation is recorded per strategy. Returns a JointHierarchy carrying one
    CoordinateHierarchy per coordinate with the D-A31 prevalence denominator (no-op → 0
    numerator / 1 denominator; a structurally non-applicable coordinate is absent from
    `coords` and never counted). On a diagonal V̂_s this reduces to fit_coordinate_hierarchy
    in distribution (the two samplers draw RNG differently, so they agree only up to
    Monte-Carlo error, not bit-for-bit)."""
    priors = priors or HierarchyPriors()
    rng = np.random.default_rng(seed)

    # Coordinate order = first appearance across strategies.
    coords: list = []
    for sv in strategies:
        for c in sv.coords:
            if c not in coords:
                coords.append(c)

    # Per-strategy precompute over SUSCEPTIBLE coords: the measurement sub-covariance,
    # eigenvalue-floored at ε and inverted from the same eigendecomposition (D-A34), with
    # the applied regularisation recorded so it is never a silent degree of freedom.
    packs: list[dict] = []
    regularisations: dict = {}
    for sv in strategies:
        k = len(sv.coords)
        sus = [j for j in range(k) if not bool(sv.no_op_mask[j])]
        if not sus:
            continue  # all present coords are no-ops: contributes only to denominators
        sub_coords = [sv.coords[j] for j in sus]
        yhat = np.asarray(sv.estimate, dtype=float)[sus]
        V = np.asarray(sv.covariance, dtype=float)[np.ix_(sus, sus)]
        _, Vinv, reg = regularise_covariance(V, epsilon)
        regularisations[sv.strategy_label] = reg
        packs.append({"coords": sub_coords, "yhat": yhat, "Vinv": Vinv, "E": yhat.copy()})

    # Initialise population params per coordinate over the susceptible estimates.
    est_by_coord: dict = {c: [] for c in coords}
    for pk in packs:
        for c, e in zip(pk["coords"], pk["yhat"]):
            est_by_coord[c].append(e)
    mu = {c: (float(np.mean(v)) if v else 0.0) for c, v in est_by_coord.items()}
    tau = {c: max(float(np.std(v)) if len(v) > 1 else priors.tau_scale, 1e-3)
           for c, v in est_by_coord.items()}

    mu_draws = {c: [] for c in coords}
    tau_draws = {c: [] for c in coords}
    abs_draws = {}  # (coord, strategy-pack-index) -> list[|E|]  for susceptible pairs
    tau_step = 0.3

    for it in range(n_iter):
        # 1. E_s | μ, τ, Ê_s  — multivariate conjugate, per strategy.
        for pk in packs:
            sub = pk["coords"]
            mu_A = np.array([mu[c] for c in sub])
            tau_A = np.array([tau[c] for c in sub])
            Lam = pk["Vinv"] + np.diag(1.0 / (tau_A * tau_A))
            post_cov = np.linalg.inv(Lam)
            post_cov = 0.5 * (post_cov + post_cov.T)  # symmetrise against round-off
            rhs = pk["Vinv"] @ pk["yhat"] + mu_A / (tau_A * tau_A)
            post_mean = post_cov @ rhs
            pk["E"] = np.atleast_1d(rng.multivariate_normal(post_mean, post_cov))

        # 2/3. μ_i, τ_i per coordinate (diagonal population) over susceptible strategies.
        for c in coords:
            E_c = np.array([pk["E"][pk["coords"].index(c)] for pk in packs if c in pk["coords"]])
            if E_c.size == 0:
                continue
            S = E_c.size
            prior_prec = 1.0 / (priors.mu_scale ** 2)
            mu_prec = S / (tau[c] * tau[c]) + prior_prec
            mu_var = 1.0 / mu_prec
            mu[c] = float(rng.normal(mu_var * (np.sum(E_c) / (tau[c] * tau[c])), math.sqrt(mu_var)))

            resid_sq = float(np.sum((E_c - mu[c]) ** 2))
            cur_lp = _log_tau_posterior(tau[c], resid_sq, S, priors.tau_scale)
            prop = tau[c] * math.exp(rng.normal(0.0, tau_step))
            prop_lp = _log_tau_posterior(prop, resid_sq, S, priors.tau_scale)
            if math.log(rng.uniform()) < (prop_lp - cur_lp + math.log(prop) - math.log(tau[c])):
                tau[c] = prop

        if it >= burn:
            for c in coords:
                mu_draws[c].append(mu[c])
                tau_draws[c].append(tau[c])
            for sidx, pk in enumerate(packs):
                for local, c in enumerate(pk["coords"]):
                    abs_draws.setdefault((c, sidx), []).append(abs(pk["E"][local]))

    out: dict = {}
    for c in coords:
        # Denominator = strategies where c is PRESENT (susceptible OR a no-op).
        n_runnable = sum(1 for sv in strategies if c in sv.coords)
        sus_sidx = [sidx for sidx, pk in enumerate(packs) if c in pk["coords"]]
        n_sus = len(sus_sidx)
        if n_sus == 0:
            out[c] = CoordinateHierarchy(
                c, n_runnable, 0, float("nan"), float("nan"), float("nan"),
                float("nan"), float("nan"), float("nan"), 0.0, vartheta,
                prevalence_sweep=tuple((float(v), 0.0) for v in vartheta_grid),
            )
            continue
        abs_arrays = [np.asarray(abs_draws[(c, sidx)]) for sidx in sus_sidx]
        prevalence = sum(float(np.mean(a > vartheta)) for a in abs_arrays) / n_runnable
        prev_sweep = _prevalence_sweep(abs_arrays, n_runnable, vartheta_grid)
        mu_arr = np.asarray(mu_draws[c])
        tau_arr = np.asarray(tau_draws[c])
        lo_mu, hi_mu = np.percentile(mu_arr, [2.5, 97.5])
        lo_tau, hi_tau = np.percentile(tau_arr, [2.5, 97.5])
        out[c] = CoordinateHierarchy(
            coordinate=c,
            n_runnable=n_runnable,
            n_susceptible=n_sus,
            mu_mean=float(np.mean(mu_arr)),
            mu_ci_low=float(lo_mu), mu_ci_high=float(hi_mu),
            tau_mean=float(np.mean(tau_arr)),
            tau_ci_low=float(lo_tau), tau_ci_high=float(hi_tau),
            prevalence=prevalence,
            vartheta=vartheta,
            prevalence_sweep=prev_sweep,
        )
    return JointHierarchy(coordinates=out, regularisations=regularisations)
