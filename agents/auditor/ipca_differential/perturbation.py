"""
perturbation.py — stochastic perturbation robustness (spec §6.3, D-A51).

The mean-zero-noise run RETAINED under its HONEST name. It adds symmetric mean-zero noise to the
panel returns and re-fits, reporting the distribution of the interaction bracket I under
perturbation as a robustness DESCRIPTIVE. A systematic non-null centre is EXPECTED and interpreted
as the variance/attenuation channel — it is **never** a false positive (``is_fpr=False``). This is
the honest counterpart to the withdrawn mean-zero placebo (§6.2): E[ε]=0 does not give E[I]=0
through a nonlinear fitted model, so a non-zero centre here is a real channel, not an FPR failure.

Secondary check (§6.3): the coverage of zero by the §5.3 conditional bootstrap intervals within the
matched-twin null samples — diagnosing the intervals actually shipped. Under the exchangeable null a
well-calibrated (1−α) interval should cover zero ≈ (1−α) of the time.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from agents.auditor.thresholds import (
    IPCABootstrapConfig,
    IPCAFprConfig,
    IPCALambda,
    IPCAPerturbationConfig,
    IPCAProjectionGate,
)
from agents.quant.library.ipca import ContractViolation
from agents.quant.library.ipca_feed import IPCAFeed

from .differential import differential_from_feeds
from .randomisation_fpr import _panels_from_label, make_matched_twin_dataset


def _perturb_feed(feed: IPCAFeed, rng: np.random.Generator, noise_sd: float) -> IPCAFeed:
    """Add symmetric mean-zero noise to the returns (the measurement-error channel). Z is untouched."""
    R = [r + rng.normal(0.0, noise_sd, r.shape[0]) for r in feed.R]
    return feed._replace(R=R)


@dataclass(frozen=True, eq=False)
class PerturbationRobustness:
    """The distribution of I under mean-zero perturbation. A robustness descriptive, NEVER an FPR."""

    n_draws: int
    noise_sd: float
    i_obs: float
    i_mean: float                    # the perturbed centre — a non-null value is EXPECTED
    i_std: float
    i_min: float
    i_max: float
    n_usable: int
    is_fpr: bool = False
    interpretation: str = (
        "distribution of I under mean-zero perturbation (variance/attenuation channel); a non-null "
        "centre is EXPECTED and interpreted, never a false positive (§6.3)"
    )

    def to_dict(self) -> dict:
        return {
            "perturbation_robustness": {
                "n_draws": self.n_draws,
                "noise_sd": self.noise_sd,
                "i_obs": self.i_obs,
                "i_mean": self.i_mean,
                "i_std": self.i_std,
                "i_min": self.i_min,
                "i_max": self.i_max,
                "n_usable": self.n_usable,
                "is_fpr": self.is_fpr,
                "interpretation": self.interpretation,
            }
        }


def perturbation_robustness(
    feed_n: IPCAFeed,
    feed_b: IPCAFeed,
    anchor: pd.Series,
    lam: IPCALambda,
    gate: IPCAProjectionGate,
    cfg: IPCAPerturbationConfig,
    *,
    i_obs: float | None = None,
    seed: int = 0,
) -> PerturbationRobustness:
    """Re-fit under mean-zero perturbation ``n_draws`` times and report the distribution of I. The
    anchor (fixed test asset) is held constant; only the panels are perturbed and re-fit."""
    if i_obs is None:
        i_obs = differential_from_feeds(
            "perturb", "perturb", feed_n, feed_b, anchor, lam, gate
        ).interaction_bracket_raw.value

    rng = np.random.default_rng(seed)
    draws: list[float] = []
    for _ in range(cfg.n_draws):
        fn = _perturb_feed(feed_n, rng, cfg.noise_sd)
        fb = _perturb_feed(feed_b, rng, cfg.noise_sd)
        try:
            res = differential_from_feeds("perturb", "perturb", fn, fb, anchor, lam, gate)
            draws.append(res.interaction_bracket_raw.value)
        except (ContractViolation, np.linalg.LinAlgError):
            draws.append(float("nan"))

    finite = np.array([x for x in draws if x == x], dtype=np.float64)
    if finite.size:
        i_mean, i_std = float(finite.mean()), float(finite.std(ddof=0))
        i_min, i_max = float(finite.min()), float(finite.max())
    else:
        i_mean = i_std = i_min = i_max = float("nan")
    return PerturbationRobustness(
        n_draws=cfg.n_draws, noise_sd=cfg.noise_sd, i_obs=float(i_obs),
        i_mean=i_mean, i_std=i_std, i_min=i_min, i_max=i_max, n_usable=int(finite.size),
    )


@dataclass(frozen=True, eq=False)
class ZeroCoverage:
    """Coverage of zero by the §5.3 conditional bootstrap intervals across matched-twin null
    samples (§6.3 secondary check). Under a true null a (1−α) interval should cover 0 ≈ (1−α)."""

    r_datasets: int
    alpha: float
    n_cover: int
    n_usable: int
    coverage_fraction: float

    def to_dict(self) -> dict:
        return {
            "bootstrap_zero_coverage": {
                "r_datasets": self.r_datasets,
                "alpha": self.alpha,
                "n_cover": self.n_cover,
                "n_usable": self.n_usable,
                "coverage_fraction": self.coverage_fraction,
            }
        }


def bootstrap_zero_coverage(
    fpr_cfg: IPCAFprConfig,
    bootstrap_cfg: IPCABootstrapConfig,
    lam: IPCALambda,
    gate: IPCAProjectionGate,
    *,
    seed: int,
    r: int | None = None,
) -> ZeroCoverage:
    """For each matched-twin null dataset, run the differential WITH the §5.3 bootstrap and check
    whether the bracket interval covers zero. Diagnoses the intervals actually shipped."""
    n_r = fpr_cfg.r_datasets if r is None else r
    n_cover = 0
    n_usable = 0
    for rr in range(n_r):
        twin = make_matched_twin_dataset(fpr_cfg.twin_dgp, lam, seed=seed + rr)
        feed0, feed1 = _panels_from_label(twin, np.zeros(twin.n_bonds, dtype=int))
        res = differential_from_feeds(
            "placebo", "placebo", feed0, feed1, twin.anchor, lam, gate,
            bootstrap=bootstrap_cfg, bootstrap_seed=seed + 100_000 + rr,
        )
        eff = res.interaction_bracket_raw
        if eff.interval is None:                 # bootstrap refused on this sample — not usable
            continue
        n_usable += 1
        lo, hi = eff.interval
        if lo <= 0.0 <= hi:
            n_cover += 1
    coverage = (n_cover / n_usable) if n_usable else float("nan")
    return ZeroCoverage(
        r_datasets=n_r, alpha=bootstrap_cfg.alpha, n_cover=n_cover,
        n_usable=n_usable, coverage_fraction=coverage,
    )
