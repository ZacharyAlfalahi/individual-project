"""
compression.py — the higher-order compression-adequacy statistic D (§8.1).

The body reports a compression of the saturated decomposition — main effects plus
the confirmatory two-way E. Does that compression discard material higher-order
structure? The saturated transform computes EVERY coefficient (§5.2), so the
question has a direct, exact answer:

    D = Σ_{|T|≥3} E_T²  /  Σ_{|T|≥1} E_T²

— the share of total effect energy carried by third- and higher-order terms. This
replaces the earlier "PPC", which named no statistic, threshold or replicated
quantity (D-A41): an estimate-level diagnostic is not a posterior predictive
check, and D answers the same question with an honest name.

Uncertainty is the bootstrap distribution of D (apply the transform within each
replicate). Verdict against a pre-registered D_max: interval below D_max →
compression adequate; otherwise a finding (the appendix coefficients are promoted
to the body). Failure of the compression does not invalidate the saturated
analysis — only skipping the check is not reportable.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np

from .bootstrap import BootstrapResult

_DENOM_FLOOR = 1e-18  # below this there is no effect energy at all => D := 0


def compression_statistic(doe: Mapping[frozenset, float]) -> float:
    """D from a DOE-effect map over subsets. Returns 0.0 when there is no
    first-or-higher-order effect energy (a clean/degenerate lattice)."""
    num = 0.0
    den = 0.0
    for T, e in doe.items():
        order = len(T)
        if order >= 1:
            sq = float(e) ** 2
            den += sq
            if order >= 3:
                num += sq
    if den < _DENOM_FLOOR:
        return 0.0
    return num / den


def compression_distribution(bootstrap: BootstrapResult) -> np.ndarray:
    """The bootstrap distribution of D, computed from the per-replicate DOE draws
    (which cover every subset)."""
    B = bootstrap.n_replicates
    num = np.zeros(B)
    den = np.zeros(B)
    for T, draws in bootstrap.doe_draws.items():
        order = len(T)
        if order >= 1:
            sq = np.asarray(draws) ** 2
            den += sq
            if order >= 3:
                num += sq
    return np.where(den < _DENOM_FLOOR, 0.0, num / np.where(den < _DENOM_FLOOR, 1.0, den))


@dataclass(frozen=True, eq=False)
class CompressionResult:
    d_point: float
    d_ci_low: float
    d_ci_high: float
    d_max: float
    adequate: bool              # interval below D_max
    draws: np.ndarray

    def to_dict(self) -> dict:
        return {
            "d_point": self.d_point,
            "d_ci_low": self.d_ci_low,
            "d_ci_high": self.d_ci_high,
            "d_max": self.d_max,
            "compression_adequate": self.adequate,
        }


def run_compression(
    doe_point: Mapping[frozenset, float],
    bootstrap: BootstrapResult,
    *,
    d_max: float,
    alpha: float = 0.05,
) -> CompressionResult:
    """Compute D, its bootstrap interval, and the adequacy verdict against D_max.
    `adequate` iff the whole interval sits below D_max (§8.1)."""
    d_point = compression_statistic(doe_point)
    draws = compression_distribution(bootstrap)
    lo, hi = np.percentile(draws, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return CompressionResult(
        d_point=d_point,
        d_ci_low=float(lo),
        d_ci_high=float(hi),
        d_max=d_max,
        adequate=bool(hi < d_max),
        draws=draws,
    )
