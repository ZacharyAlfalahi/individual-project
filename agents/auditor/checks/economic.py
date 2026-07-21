"""
economic.py — economic significance (§9).

Three tiers of significance: statistical (distinguishable from zero), practical
(large enough to matter, §9 thresholds), and economic (what it costs a
practitioner: Δ annual return, Δ alpha, Δ Sharpe, Δ IR, Δ max drawdown, Δ
turnover). Practical effect sizes are classified Negligible / Small / Moderate /
Large against thresholds PRE-COMMITTED with a cited source — a threshold chosen
after seeing results is exactly the researcher degree of freedom this project
exists to expose.

Deflated Sharpe: the subtle one (O-A4). The lattice cells are NOT independent
trials — they are one strategy under registered interventions — so the DSR trial
count is the STRATEGY-LEVEL discovery count, identical across all cells, never 2^k.
Implemented via the Probabilistic Sharpe Ratio and the expected-maximum-Sharpe
deflation (Bailey & López de Prado, 2014).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

from ..schemas.lattice_types import MetricSet
from .stats import normal_cdf, normal_ppf

Band = Literal["Negligible", "Small", "Moderate", "Large"]

_EULER_MASCHERONI = 0.5772156649015329


@dataclass(frozen=True)
class EconomicBands:
    """Pre-registered practical-significance thresholds on |effect| for one metric
    (§9). Each must carry a cited source or ex-ante justification where committed."""

    small: float
    moderate: float
    large: float

    def __post_init__(self) -> None:
        if not (0 < self.small < self.moderate < self.large):
            raise ValueError(
                f"bands must satisfy 0 < small < moderate < large; got {self}"
            )


def classify_effect(value: float, bands: EconomicBands) -> Band:
    """Classify |value| against the pre-registered bands."""
    v = abs(value)
    if v < bands.small:
        return "Negligible"
    if v < bands.moderate:
        return "Small"
    if v < bands.large:
        return "Moderate"
    return "Large"


# ---------------------------------------------------------------------------
# Deflated Sharpe
# ---------------------------------------------------------------------------

def probabilistic_sharpe_ratio(
    sr_periodic: float,
    n_obs: int,
    *,
    benchmark: float = 0.0,
    skew: float = 0.0,
    kurtosis: float = 3.0,
) -> float:
    """PSR: P(true per-period Sharpe > benchmark) given the estimate's uncertainty
    (Bailey & López de Prado). `sr_periodic` is the per-period Sharpe (NOT
    annualised); `kurtosis` is non-excess (3 = normal)."""
    if n_obs < 2:
        return float("nan")
    denom = 1.0 - skew * sr_periodic + (kurtosis - 1.0) / 4.0 * sr_periodic ** 2
    if denom <= 0:
        return float("nan")
    z = (sr_periodic - benchmark) * math.sqrt(n_obs - 1) / math.sqrt(denom)
    return normal_cdf(z)


def expected_max_sharpe(n_trials: int, sr_std: float) -> float:
    """The expected maximum per-period Sharpe from `n_trials` independent trials
    with cross-trial Sharpe standard deviation `sr_std` — the deflation benchmark
    SR0 (Bailey & López de Prado). Requires n_trials >= 2."""
    if n_trials < 2:
        raise ValueError(f"n_trials must be >= 2 for a deflation benchmark; got {n_trials}")
    g = _EULER_MASCHERONI
    a = normal_ppf(1.0 - 1.0 / n_trials)
    b = normal_ppf(1.0 - 1.0 / (n_trials * math.e))
    return sr_std * ((1.0 - g) * a + g * b)


def deflated_sharpe_ratio(
    sr_periodic: float,
    n_obs: int,
    n_trials: int,
    *,
    sr_std: float,
    skew: float = 0.0,
    kurtosis: float = 3.0,
) -> float:
    """DSR = PSR evaluated against the expected-maximum-Sharpe deflation benchmark
    for `n_trials` (the strategy-level discovery count, O-A4). n_trials == 1 =>
    no deflation (benchmark 0)."""
    benchmark = 0.0 if n_trials <= 1 else expected_max_sharpe(n_trials, sr_std)
    return probabilistic_sharpe_ratio(
        sr_periodic, n_obs, benchmark=benchmark, skew=skew, kurtosis=kurtosis
    )


# ---------------------------------------------------------------------------
# Economic deltas + assembled result
# ---------------------------------------------------------------------------

def economic_deltas(uncorrected: MetricSet, corrected: MetricSet) -> dict:
    """The practitioner-facing deltas (corrected minus as-published) on the shared
    metrics (§9). Annualised-return and Sharpe deltas are the most cited."""
    return {
        "delta_annual_return": corrected.annualised_average - uncorrected.annualised_average,
        "delta_mean_return": corrected.average - uncorrected.average,
        "delta_sharpe": corrected.sharpe - uncorrected.sharpe,
        "delta_volatility": corrected.bumpiness - uncorrected.bumpiness,
    }


@dataclass(frozen=True)
class EconomicResult:
    endpoint_gap: float
    gap_band: Band
    deltas: dict
    deflated_sharpe_corrected: float
    n_trials: int

    def to_dict(self) -> dict:
        return {
            "endpoint_gap": self.endpoint_gap,
            "gap_band": self.gap_band,
            "deltas": self.deltas,
            "deflated_sharpe_corrected": self.deflated_sharpe_corrected,
            "n_trials": self.n_trials,
        }


def run_economic(
    uncorrected: MetricSet,
    corrected: MetricSet,
    *,
    gap_bands: EconomicBands,
    n_trials: int,
    sr_std: float,
    skew: float = 0.0,
    kurtosis: float = 3.0,
) -> EconomicResult:
    """Classify the endpoint gap, compute the practitioner deltas, and the deflated
    Sharpe of the corrected endpoint. `gap` uses the annualised-return metric.

    `sr_std` must be the PER-PERIOD (monthly) cross-trial Sharpe standard deviation —
    it is compared against `sr_periodic = average/bumpiness`, also per-period. Passing
    an annualised value would inflate the deflation benchmark by √months_per_year."""
    gap = corrected.annualised_average - uncorrected.annualised_average
    deltas = economic_deltas(uncorrected, corrected)
    # per-period Sharpe of the corrected endpoint = mean / sd (native units).
    sr_periodic = (
        corrected.average / corrected.bumpiness
        if corrected.bumpiness and not math.isnan(corrected.bumpiness)
        else float("nan")
    )
    dsr = deflated_sharpe_ratio(
        sr_periodic, corrected.n_months, n_trials, sr_std=sr_std,
        skew=skew, kurtosis=kurtosis,
    )
    return EconomicResult(
        endpoint_gap=gap,
        gap_band=classify_effect(gap, gap_bands),
        deltas=deltas,
        deflated_sharpe_corrected=dsr,
        n_trials=n_trials,
    )
