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

Degenerate case. When the lattice carries no first-or-higher-order effect energy
(a flat response surface), the ratio is 0/0 and D is UNDEFINED. It is recorded as
NaN, never as 0: a surface with no structure has no higher-order structure to
hide, so compression is trivially adequate, but "no higher-order structure"
(a real D of 0) must be distinguished from "no structure at all" (D undefined).
Silently reporting 0 would fill an undefined quantity with a zero — the very error
the applicability discipline forbids — and, at the replicate level, would pull the
bootstrap interval toward zero and bias the adequacy verdict toward "adequate".
Degenerate replicates are therefore NaN and excluded from the interval.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping

import numpy as np

from .bootstrap import BootstrapResult

_DENOM_FLOOR = 1e-18  # below this there is no effect energy at all => D is undefined (NaN)


def compression_statistic(doe: Mapping[frozenset, float]) -> float:
    """D from a DOE-effect map over subsets. Returns NaN when there is no
    first-or-higher-order effect energy (a flat/degenerate lattice): the ratio is
    0/0 and undefined, and must not be silently reported as 0."""
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
        return float("nan")
    return num / den


def compression_distribution(bootstrap: BootstrapResult) -> np.ndarray:
    """The bootstrap distribution of D, computed from the per-replicate DOE draws
    (which cover every subset). Degenerate replicates (no effect energy) are NaN,
    not 0, so they are excluded from the interval rather than pulling it toward
    zero."""
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
    with np.errstate(invalid="ignore", divide="ignore"):
        ratio = num / den
    return np.where(den < _DENOM_FLOOR, np.nan, ratio)


@dataclass(frozen=True, eq=False)
class CompressionResult:
    d_point: float
    d_ci_low: float
    d_ci_high: float
    d_max: float
    adequate: bool              # degenerate, or finite interval below D_max
    degenerate: bool            # no first-or-higher-order effect energy => D undefined (NaN)
    draws: np.ndarray

    def to_dict(self) -> dict:
        return {
            "d_point": self.d_point,
            "d_ci_low": self.d_ci_low,
            "d_ci_high": self.d_ci_high,
            "d_max": self.d_max,
            "compression_adequate": self.adequate,
            "compression_degenerate": self.degenerate,
        }


def run_compression(
    doe_point: Mapping[frozenset, float],
    bootstrap: BootstrapResult,
    *,
    d_max: float,
    alpha: float = 0.05,
) -> CompressionResult:
    """Compute D, its bootstrap interval, and the adequacy verdict against D_max.
    `adequate` iff the lattice is degenerate (no structure to compress) OR the whole
    finite interval sits below D_max (§8.1). A degenerate lattice yields an
    undefined D (NaN), reported as such rather than as 0; its degenerate bootstrap
    replicates are NaN and excluded from the interval."""
    d_point = compression_statistic(doe_point)
    degenerate = not math.isfinite(d_point)
    draws = compression_distribution(bootstrap)
    finite = draws[np.isfinite(draws)]
    if finite.size == 0:
        lo = hi = float("nan")
    else:
        lo, hi = np.percentile(finite, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    adequate = bool(degenerate or (math.isfinite(hi) and hi < d_max))
    return CompressionResult(
        d_point=d_point,
        d_ci_low=float(lo),
        d_ci_high=float(hi),
        d_max=d_max,
        adequate=adequate,
        degenerate=degenerate,
        draws=draws,
    )
