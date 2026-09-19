"""
inference.py — inference routed by estimand (§7.1).

The routing table (normative):

  * Mean-return contrasts (marginals, E, h, φ of the MEAN return) — HAC / Newey-West
    on the paired monthly series, PLUS the synchronised block bootstrap.
  * Sharpe, IR, max-drawdown, deflated Sharpe (nonlinear metrics) — the block
    bootstrap is THE method, not a robustness column.

"HAC where the estimand is a mean of observables; bootstrap everywhere else."
There is no monthly difference series whose HAC t tests a Shapley value of the
Sharpe ratio (§7.1) — the Sharpe is a nonlinear functional, so neither it nor a
linear combination of such functionals is the mean of anything observable per
month.

Key identity that makes the HAC branch exact: for the MEAN metric, every §5
quantity is linear in the cell means, and each cell mean is itself a monthly mean,
so a quantity q(Y) equals mean_t q(r_t) — the mean of the monthly series obtained
by applying q's transform to each month's cross-cell return vector. That monthly
series is what the HAC t-statistic tests.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd

from agents.quant.library.characteristic_sort import summarize_returns

from ..schemas.lattice_types import CellReturns
from ..schemas.toggle import ToggleId
from .algebra import doe_effects, saturated_basis, walsh_coefficients
from .bootstrap import BootstrapResult
from .stats import two_sided_p
from .support import primary_metric_vector, return_matrix

# Metrics that are linear in the monthly returns (a mean of observables) => HAC
# applies. Everything else is a nonlinear functional => bootstrap only.
MEAN_METRICS: frozenset[str] = frozenset({"average", "annualised_average"})

#: Default inert-coordinate tolerance for direct (synthetic / unit-test) callers. The audited path
#: takes it from ``AuditorConfig.inert_relative_tol``, which ``from_thresholds`` reads fail-loud from
#: ``auditor.inert_relative_tol`` — so a production run never scores against an unregistered value.
#: A DOE coordinate whose monthly contrast series is zero at floating-point
#: precision has no estimand: every month's cross-cell contrast cancels exactly and what remains is
#: round-off (many orders of magnitude below the return scale). A test statistic formed from that residue is noise,
#: so the coordinate is reported as inert (p = 1, t = 0) rather than tested. The tolerance is relative to
#: the largest cell metric value in the run; a real effect of one basis point per month (1e-4) sits eight
#: orders of magnitude above it.
INERT_RELATIVE_TOL = 1e-12


def _metric_scale(Y) -> float:
    vals = np.asarray(list(Y.values()) if isinstance(Y, dict) else Y, dtype=float)
    finite = vals[np.isfinite(vals)]
    return max(1.0, float(np.max(np.abs(finite)))) if finite.size else 1.0


def is_inert(values, scale: float, tol: float = INERT_RELATIVE_TOL) -> bool:
    """True iff every finite value lies within ``tol * scale`` of zero (an empty or all-NaN
    series is not inert: it is missing, not zero)."""
    vals = np.asarray(values, dtype=float)
    finite = vals[np.isfinite(vals)]
    return bool(finite.size) and float(np.max(np.abs(finite))) <= tol * scale


@dataclass(frozen=True)
class InferenceResult:
    coordinate: frozenset
    metric: str
    point: float
    method: str                 # "HAC" | "bootstrap"
    p_value: float
    ci_low: float
    ci_high: float
    t_stat: float | None = None
    n_obs: int | None = None    # months (HAC branch)
    inert: bool = False         # identically-zero contrast: not tested (p = 1, t = 0)


def _monthly_doe_series(
    R: np.ndarray, keys: Sequence[frozenset], months: pd.DatetimeIndex,
    toggles: Sequence[ToggleId],
) -> dict[frozenset, pd.Series]:
    """For each subset T, the monthly series E_T(r_t) — apply the DOE transform to
    each month's cross-cell return vector. mean_t of this series equals E_T(Y)."""
    idx = pd.DatetimeIndex(months).sort_values()
    per_month: dict[frozenset, list] = {}
    for ti in range(R.shape[0]):
        Yt = {keys[j]: float(R[ti, j]) for j in range(len(keys))}
        Et = doe_effects(walsh_coefficients(Yt, toggles))
        for T, val in Et.items():
            per_month.setdefault(T, []).append(val)
    return {T: pd.Series(vals, index=idx) for T, vals in per_month.items()}


def _bootstrap_p(draws: np.ndarray) -> float:
    """A two-sided bootstrap p-value: 2 · min(P(draw>0), P(draw<0)), floored at 1/B.
    A bootstrap p-value cannot resolve below 1/n_replicates, so reporting an exact 0
    would overstate the precision the resampling can support."""
    n = len(draws)
    frac_pos = float(np.mean(draws > 0))
    frac_neg = float(np.mean(draws < 0))
    p = min(1.0, 2.0 * min(frac_pos, frac_neg))
    return max(1.0 / n, p) if n else float("nan")


def infer_doe_effects(
    cells: Sequence[CellReturns],
    months: pd.DatetimeIndex,
    toggles: Sequence[ToggleId],
    bootstrap: BootstrapResult,
    *,
    metric: str,
    coordinates: Sequence[frozenset] | None = None,
    months_per_year: int = 12,
    alpha: float = 0.05,
    inert_relative_tol: float = INERT_RELATIVE_TOL,
) -> dict[frozenset, InferenceResult]:
    """Infer each DOE coordinate, routed by estimand. `coordinates` defaults to
    every subset; pass the confirmatory set to restrict. HAC is used iff `metric`
    is a mean of observables; otherwise the bootstrap is the method."""
    Y = primary_metric_vector(cells, months, metric, months_per_year=months_per_year)
    basis = saturated_basis(Y, toggles)
    coords = list(coordinates) if coordinates is not None else list(basis.doe)
    doe_ci = bootstrap.doe_ci(alpha)

    results: dict[frozenset, InferenceResult] = {}

    scale = _metric_scale(Y)
    if metric in MEAN_METRICS:
        keys, R = return_matrix(cells, months)
        series = _monthly_doe_series(R, keys, months, toggles)
        for T in coords:
            summ = summarize_returns(series[T], None, months_per_year)
            lo, hi = doe_ci.get(T, (float("nan"), float("nan")))
            if is_inert(series[T], scale, inert_relative_tol):
                results[T] = InferenceResult(
                    coordinate=T, metric=metric, point=basis.doe[T], method="HAC",
                    p_value=1.0, ci_low=lo, ci_high=hi, t_stat=0.0, n_obs=summ["n_months"], inert=True,
                )
                continue
            # A coordinate whose monthly DOE series is constant but non-zero has zero
            # within-sample variance and yields a NaN HAC t (and NaN p); read it off its
            # bootstrap CI (carried below).
            t = summ["t_stat"]
            results[T] = InferenceResult(
                coordinate=T, metric=metric, point=basis.doe[T], method="HAC",
                p_value=two_sided_p(t), ci_low=lo, ci_high=hi,
                t_stat=t, n_obs=summ["n_months"],
            )
    else:
        for T in coords:
            draws = bootstrap.doe_draws.get(T)
            lo, hi = doe_ci.get(T, (float("nan"), float("nan")))
            inert = (draws is not None and is_inert(draws, scale, inert_relative_tol)
                     and abs(float(basis.doe[T])) <= inert_relative_tol * scale)
            results[T] = InferenceResult(
                coordinate=T, metric=metric, point=basis.doe[T], method="bootstrap",
                p_value=(1.0 if inert else _bootstrap_p(draws) if draws is not None else float("nan")),
                ci_low=lo, ci_high=hi, t_stat=None, n_obs=None, inert=inert,
            )
    return results
