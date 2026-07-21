"""
recovery_sweep.py — the survivorship recovery-rate sensitivity protocol (build
step 12b, §10.5).

The survivorship number depends on an assumption nobody can pin down — what a
defaulted bond is worth when it never trades again. That assumption is a dial, and
the protocol makes it impossible to turn after seeing the answer: the recovery rate
ρ is SWEPT over a pre-registered grid and the whole curve is the reported result.

  * Only Group B (NaN terminal, no trade) rows are imputed; Group A (observed
    price) rows are NEVER touched — imputing over real data is fabrication (§10.5.6).
  * The estimand is named (§10.5.3): the primary is the survivorship leave-one-out
    with all other corrections ON — Y(N) − Y(N∖{surv}), evaluated at each ρ.
  * Monotonicity is NOT assumed (§10.5.2): membership churn, breakpoints and metric
    nonlinearity can make the curve non-monotone; the robustness rule admits every
    shape and reports it.

This module owns the imputation transform, the curve classification, and the sweep
driver; the caller supplies the effect function that runs the named estimand at a
given ρ (so the sweep stays decoupled from the full lattice run).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import pandas as pd

# Pre-registered default grid (§10.5.4). The RANGE is immutable; only the
# resolution may be cut under compute pressure (§14), never the range.
DEFAULT_RECOVERY_GRID: tuple[float, ...] = tuple(round(0.1 * i, 1) for i in range(11))


def impute_group_b(
    panel: pd.DataFrame,
    rho: float,
    *,
    distress_exits: Sequence[str] = ("defaulted",),
) -> pd.DataFrame:
    """Impute Group-B terminal returns at recovery rate ρ. A Group-B row is a
    distress-terminal row whose corrected return is NaN (no trade). Its imputed
    terminal return is ρ − 1 (ρ=1 → ~0, ρ=0 → −100%). Group-A rows (observed
    return) are untouched — the function only writes where ret_corr is NaN."""
    if not (0.0 <= rho <= 1.0):
        raise ValueError(f"recovery rate ρ must be in [0,1]; got {rho}")
    panel = panel.copy()
    is_distress = panel["exit_reason"].isin(list(distress_exits))
    group_b = is_distress & panel["ret_corr"].isna()
    imputed = rho - 1.0
    for col in ("ret_raw", "ret_corr"):
        if col in panel.columns:
            panel.loc[group_b, col] = imputed
    for col in ("xret_raw", "xret_corr"):
        if col in panel.columns:
            rf = panel.loc[group_b, "rf_monthly"] if "rf_monthly" in panel.columns else 0.0
            panel.loc[group_b, col] = imputed - rf
    return panel


@dataclass(frozen=True)
class RecoveryPoint:
    rho: float
    effect: float
    ci_low: float
    ci_high: float


def _sign(x: float, tol: float = 1e-12) -> int:
    return 0 if abs(x) <= tol else (1 if x > 0 else -1)


def classify_curve(effects: Sequence[float]) -> str:
    """Classify the sweep per the §10.5.4 robustness table:
      sign_stable · one_crossover · non_monotone_sign_stable · multiple_crossings."""
    signs = [_sign(e) for e in effects if _sign(e) != 0]
    crossings = sum(1 for a, b in zip(signs, signs[1:]) if a != b)
    monotone = all(
        b >= a for a, b in zip(effects, effects[1:])
    ) or all(b <= a for a, b in zip(effects, effects[1:]))
    # `monotone` only REFINES the sign-stable (no-crossing) case: a monotone curve that
    # passes through zero genuinely changes sign, so it is a crossover, not sign_stable.
    if crossings == 0:
        return "sign_stable" if monotone else "non_monotone_sign_stable"
    if crossings == 1:
        return "one_crossover"
    return "multiple_crossings"


@dataclass(frozen=True, eq=False)
class RecoverySweepResult:
    grid: tuple[float, ...]
    points: tuple[RecoveryPoint, ...]
    classification: str
    sign_stable: bool
    crossover_rhos: tuple[float, ...]
    headline_rule: str          # "DRR_A5" | "FIXED_RECOVERY"
    headline_rho: float | None  # null under DRR_A5 (§10.5.5)

    def to_dict(self) -> dict:
        return {
            "grid": list(self.grid),
            "effect_by_rho": [
                {"rho": p.rho, "effect": p.effect, "ci_low": p.ci_low, "ci_high": p.ci_high}
                for p in self.points
            ],
            "classification": self.classification,
            "sign_stable": self.sign_stable,
            "crossover_rho": list(self.crossover_rhos),
            "headline_rule": self.headline_rule,
            "headline_rho": self.headline_rho,
        }


def run_recovery_sweep(
    effect_fn: Callable[[float], tuple[float, float, float]],
    *,
    grid: Sequence[float] = DEFAULT_RECOVERY_GRID,
    headline_rule: str = "FIXED_RECOVERY",
    headline_rho: float | None = 0.4,
) -> RecoverySweepResult:
    """Sweep the named estimand over the pre-registered grid. `effect_fn(rho)`
    returns (effect, ci_low, ci_high) for that recovery rate. `headline_rho` must
    be None under the DRR_A5 price-based rule (§10.5.5)."""
    if headline_rule == "DRR_A5" and headline_rho is not None:
        raise ValueError("headline_rho must be None under the DRR_A5 price-based rule (§10.5.5)")
    if headline_rule == "FIXED_RECOVERY" and headline_rho is None:
        raise ValueError("FIXED_RECOVERY requires a pre-registered headline_rho")

    points = []
    for rho in grid:
        effect, lo, hi = effect_fn(rho)
        points.append(RecoveryPoint(rho, effect, lo, hi))

    effects = [p.effect for p in points]
    classification = classify_curve(effects)
    signs = [_sign(e) for e in effects]
    crossover_rhos = tuple(
        points[i + 1].rho
        for i in range(len(signs) - 1)
        if signs[i] != 0 and signs[i + 1] != 0 and signs[i] != signs[i + 1]
    )
    return RecoverySweepResult(
        grid=tuple(grid),
        points=tuple(points),
        classification=classification,
        sign_stable=classification in ("sign_stable", "non_monotone_sign_stable"),
        crossover_rhos=crossover_rhos,
        headline_rule=headline_rule,
        headline_rho=headline_rho,
    )
