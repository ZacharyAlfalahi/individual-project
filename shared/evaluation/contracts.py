"""
contracts.py — typed results, enums, and refusal codes for the shared economic
diagnostics (shared/evaluation, spec §3). Supersedes the Devil's Advocate agent (D4).

Every numeric the Reporter may print is a NAMED attribute on a frozen dataclass —
no dicts of floats computed at render time (spec §3.1). A diagnostic that cannot be
computed returns a result object with `estimable=False` (or `evaluable=False`) and a
`refusal_code`, NOT an exception and NOT NaN (D-E3): "not estimable" is counted, not
crashed. There is deliberately NO verdict boolean — no `is_crowded`, `is_spanned`,
`passes` (D-E7); the aggregate is `SharedEvaluationResult`, never
`AdversarialDiagnostics` (D-E6); the cost field is `cost_adjusted_*`, never `net_*`
(D-E10).

House style (mirrors agents/quant/config/refusal.py): stdlib `@dataclass(frozen=True)`,
`class X(str, Enum)`, hand-written `to_dict()` returning a JSON/YAML-safe dict. No
Pydantic. Ordered pair-collections are stored as `tuple[tuple[str, T], ...]` so the
frozen dataclasses stay hashable; `to_dict()` renders them as JSON objects.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum


# ---------------------------------------------------------------------------
# Enums (spec §3.2)
# ---------------------------------------------------------------------------

class RefusalCode(str, Enum):
    INSUFFICIENT_OBSERVATIONS = "insufficient_observations"            # n < min_obs
    RANK_DEFICIENT = "rank_deficient"                                  # exact collinearity
    NO_COMMON_SAMPLE = "no_common_sample"                              # alignment empty
    ZERO_TURNOVER = "zero_turnover"                                    # break-even undefined
    NON_POSITIVE_GROSS = "non_positive_gross"                          # break-even undefined
    NON_POSITIVE_TURNOVER_INTERCEPT = "non_positive_turnover_intercept"  # break-even undefined
    MISSING_RATING = "missing_rating"                                  # cost bucket unassignable
    MISSING_MACRO_OBSERVATION = "missing_macro_observation"            # regime unassignable
    PREDICTION_NOT_EVALUABLE = "prediction_not_evaluable"             # no falsifiable contrast
    ARTEFACT_CAPABILITY_MISSING = "artefact_capability_missing"        # drifted/target weights absent
    DEVELOPMENT_SCOPE_DIAGNOSTIC = "development_scope_diagnostic"      # A7: floor-invoking inference not computed on the holdout window


class TurnoverMethod(str, Enum):
    DRIFT_ADJUSTED = "drift_adjusted"
    TARGET_TO_TARGET_PROXY = "target_to_target_proxy"


class WeightingScheme(str, Enum):
    PAR = "par"
    EQUAL = "equal"


class CostUnit(str, Enum):
    ONE_WAY = "one_way"
    ROUND_TRIP = "round_trip"


class BreakEvenStatus(str, Enum):
    FOUND = "found"
    UNDEFINED = "undefined"        # carries a RefusalCode


class EvaluationScope(str, Enum):
    CONFIRMATORY = "confirmatory"    # pre-registered, primary
    SUPPLEMENTARY = "supplementary"  # post-lock addition, non-gating
    AUDIT_VARIANT = "audit_variant"  # deliberately flawed original, retained for measurement


class RegimeRole(str, Enum):
    FORMATION = "formation"          # determines positions; strictly past-only
    EVALUATION = "evaluation"        # decomposition only; frozen threshold


class SampleWindow(str, Enum):
    """Which evaluation sample a result was computed on (amendment A7).

    The pre-registered 60-month floors were calibrated for the ~240-month development
    window; the holdout window (2022–2025, 48 months per SC-SCI-10) is shorter than
    every floor BY CONSTRUCTION of the walk-forward split. A7 scopes by claim type:
    floor-invoking inference (the spanning/crowding regression, the conditional-alpha
    supplementary) is refused on HOLDOUT with DEVELOPMENT_SCOPE_DIAGNOSTIC; mean/sign
    point estimates remain computable but carry `short_sample` when below a floor, and
    the reporting layer licenses point-estimate/direction sentences only. DEVELOPMENT
    is the default everywhere so pre-A7 call sites are byte-identical.
    """

    DEVELOPMENT = "development"
    HOLDOUT = "holdout"


# ---------------------------------------------------------------------------
# to_dict helpers
# ---------------------------------------------------------------------------

def _pairs_to_dict(pairs: tuple[tuple[str, object], ...]) -> dict:
    """Render an ordered tuple-of-pairs as a JSON object (insertion order preserved)."""
    return {str(k): v for k, v in pairs}


def _enum_val(x):
    return x.value if isinstance(x, Enum) else x


def _jsonsafe(value):
    """Map non-finite floats (NaN/inf) to None so a to_dict() is strict-JSON serialisable
    (`json.dumps(..., allow_nan=False)`): a bare `NaN`/`Infinity` token must never reach
    the Reporter's numeric verifier. Recurses into dicts/lists. Applied at the aggregate
    boundary; the producing modules also null non-finite values at construction."""
    if isinstance(value, bool):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {k: _jsonsafe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_jsonsafe(v) for v in value]
    return value


# ---------------------------------------------------------------------------
# Provenance (spec §3.3) — mandatory on every estimated result. shared/evaluation
# may NOT compute a standard error / CI / Sharpe / p-value itself; it may only call
# a versioned shared function and stamp which one produced the number.
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class EstimatorProvenance:
    estimator_id: str                       # e.g. "newey_west_hac"
    estimator_version: str                  # module reference or content hash
    lag_rule: str                           # "floor(T**0.25)"
    lag_used: int
    ddof: int
    annualisation: str | None = None
    bootstrap_seed: int | None = None
    bootstrap_replications: int | None = None

    def to_dict(self) -> dict:
        return {
            "estimator_id": self.estimator_id,
            "estimator_version": self.estimator_version,
            "lag_rule": self.lag_rule,
            "lag_used": self.lag_used,
            "ddof": self.ddof,
            "annualisation": self.annualisation,
            "bootstrap_seed": self.bootstrap_seed,
            "bootstrap_replications": self.bootstrap_replications,
        }


# ---------------------------------------------------------------------------
# Sample alignment (spec §3.4) — month-level mandatory, bond-level recorded (D-E5).
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SampleAlignment:
    as_published_n_months: int
    corrected_n_months: int
    common_n_months: int
    as_published_only_months: tuple[str, ...]
    corrected_only_months: tuple[str, ...]
    bond_count_by_month_as_published: tuple[tuple[str, int], ...]
    bond_count_by_month_corrected: tuple[tuple[str, int], ...]
    common_month_index: tuple[str, ...]

    def to_dict(self) -> dict:
        return {
            "as_published_n_months": self.as_published_n_months,
            "corrected_n_months": self.corrected_n_months,
            "common_n_months": self.common_n_months,
            "as_published_only_months": list(self.as_published_only_months),
            "corrected_only_months": list(self.corrected_only_months),
            "bond_count_by_month_as_published": _pairs_to_dict(
                self.bond_count_by_month_as_published
            ),
            "bond_count_by_month_corrected": _pairs_to_dict(
                self.bond_count_by_month_corrected
            ),
            "common_month_index": list(self.common_month_index),
        }


@dataclass(frozen=True)
class PairedDifference:
    """Both endpoint estimates on their natural samples AND the paired estimate on the
    common months (alignment rule 1). The alignment record travels with the result
    (alignment rule 3) — never logged and discarded."""

    estimable: bool
    refusal_code: RefusalCode | None
    a_label: str
    b_label: str
    # natural-sample endpoint means (each on its own full sample)
    mean_a_natural: float | None
    mean_b_natural: float | None
    n_a_natural: int
    n_b_natural: int
    # paired estimate on common months
    mean_a_common: float | None
    mean_b_common: float | None
    paired_mean_difference: float | None      # mean(a - b) over common months
    n_common: int
    alignment: SampleAlignment

    def to_dict(self) -> dict:
        return {
            "estimable": self.estimable,
            "refusal_code": _enum_val(self.refusal_code),
            "a_label": self.a_label,
            "b_label": self.b_label,
            "mean_a_natural": self.mean_a_natural,
            "mean_b_natural": self.mean_b_natural,
            "n_a_natural": self.n_a_natural,
            "n_b_natural": self.n_b_natural,
            "mean_a_common": self.mean_a_common,
            "mean_b_common": self.mean_b_common,
            "paired_mean_difference": self.paired_mean_difference,
            "n_common": self.n_common,
            "alignment": self.alignment.to_dict(),
        }


# ---------------------------------------------------------------------------
# Spanning (spec §3.5, §4) — estimates + conditioning diagnostics only. NO verdict
# boolean (D-E7). Conditioning diagnostics are MANDATORY, never omitted (D-E9).
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SpanningResult:
    estimable: bool
    refusal_code: RefusalCode | None
    # estimates
    alpha_monthly: float | None
    alpha_se: float | None
    t_hac: float | None
    ci_low: float | None
    ci_high: float | None
    betas: tuple[tuple[str, float], ...]
    r_squared: float | None
    # sample
    n_obs: int
    n_controls: int
    min_obs: int                             # the pre-registered floor this result was judged against
    sample_id: str
    control_set_id: str
    control_set_hash: str
    control_correction_state: str            # "corrected" — pre-registered
    # conditioning diagnostics — MANDATORY, never omitted. Values are finite-or-None
    # (perfect collinearity / singular design null out rather than emit NaN/inf).
    condition_number: float | None
    design_rank: int | None
    vif_by_control: tuple[tuple[str, float | None], ...]
    leave_one_control_out_alpha: tuple[tuple[str, float | None], ...]
    max_abs_pairwise_corr: float | None
    # governance
    scope: EvaluationScope
    provenance: EstimatorProvenance
    window: SampleWindow = SampleWindow.DEVELOPMENT   # A7; defaulted so pre-A7 construction is unchanged

    def to_dict(self) -> dict:
        return {
            "estimable": self.estimable,
            "refusal_code": _enum_val(self.refusal_code),
            "alpha_monthly": self.alpha_monthly,
            "alpha_se": self.alpha_se,
            "t_hac": self.t_hac,
            "ci_low": self.ci_low,
            "ci_high": self.ci_high,
            "betas": _pairs_to_dict(self.betas),
            "r_squared": self.r_squared,
            "n_obs": self.n_obs,
            "n_controls": self.n_controls,
            "min_obs": self.min_obs,
            "sample_id": self.sample_id,
            "control_set_id": self.control_set_id,
            "control_set_hash": self.control_set_hash,
            "control_correction_state": self.control_correction_state,
            "condition_number": self.condition_number,
            "design_rank": self.design_rank,
            "vif_by_control": _pairs_to_dict(self.vif_by_control),
            "leave_one_control_out_alpha": _pairs_to_dict(self.leave_one_control_out_alpha),
            "max_abs_pairwise_corr": self.max_abs_pairwise_corr,
            "scope": _enum_val(self.scope),
            "provenance": self.provenance.to_dict(),
            "window": _enum_val(self.window),
        }


# ---------------------------------------------------------------------------
# Costs (spec §3.5, §6). Field named cost_adjusted_*, never net_* (D-E10).
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CostScenarioResult:
    scenario_id: str                    # "kpp_comparable" | "project_registered"
    cost_bps_ig: float
    cost_bps_hy: float
    unit: CostUnit
    source_citation: str                # MUST be non-empty (enforced at construction)
    cost_adjusted_mean_monthly: float
    cost_adjusted_sharpe: float | None
    cost_drag_bps_monthly: float

    def __post_init__(self) -> None:
        if not isinstance(self.source_citation, str) or not self.source_citation.strip():
            raise ValueError(
                f"CostScenarioResult {self.scenario_id!r} has an empty source_citation; "
                f"a cost scenario must carry a non-empty citation (D-E12)"
            )

    def to_dict(self) -> dict:
        return {
            "scenario_id": self.scenario_id,
            "cost_bps_ig": self.cost_bps_ig,
            "cost_bps_hy": self.cost_bps_hy,
            "unit": _enum_val(self.unit),
            "source_citation": self.source_citation,
            "cost_adjusted_mean_monthly": self.cost_adjusted_mean_monthly,
            "cost_adjusted_sharpe": self.cost_adjusted_sharpe,
            "cost_drag_bps_monthly": self.cost_drag_bps_monthly,
        }


@dataclass(frozen=True)
class CostResult:
    estimable: bool
    refusal_code: RefusalCode | None
    # turnover
    turnover_method: TurnoverMethod
    weighting_scheme: WeightingScheme
    turnover_mean: float | None
    turnover_median: float | None
    turnover_series_id: str
    proxy_bias_direction: str | None    # "understates" | "approximately_unbiased" | "unknown"
    # scenarios
    scenarios: tuple[CostScenarioResult, ...]
    # break-even
    break_even_mean_cost_bps: float | None
    break_even_mean_status: BreakEvenStatus
    break_even_alpha_cost_bps: float | None
    break_even_alpha_status: BreakEvenStatus
    break_even_alpha_multiplier: float | None    # lambda vs registered schedule
    alpha_gross_intercept: float | None          # numerator, reported for audit
    alpha_turnover_intercept: float | None       # denominator, reported for audit
    break_even_unit: CostUnit
    # coverage
    rating_coverage_fraction: float
    months_with_missing_rating: int
    scope: EvaluationScope
    provenance: EstimatorProvenance

    def to_dict(self) -> dict:
        return {
            "estimable": self.estimable,
            "refusal_code": _enum_val(self.refusal_code),
            "turnover_method": _enum_val(self.turnover_method),
            "weighting_scheme": _enum_val(self.weighting_scheme),
            "turnover_mean": self.turnover_mean,
            "turnover_median": self.turnover_median,
            "turnover_series_id": self.turnover_series_id,
            "proxy_bias_direction": self.proxy_bias_direction,
            "scenarios": [s.to_dict() for s in self.scenarios],
            "break_even_mean_cost_bps": self.break_even_mean_cost_bps,
            "break_even_mean_status": _enum_val(self.break_even_mean_status),
            "break_even_alpha_cost_bps": self.break_even_alpha_cost_bps,
            "break_even_alpha_status": _enum_val(self.break_even_alpha_status),
            "break_even_alpha_multiplier": self.break_even_alpha_multiplier,
            "alpha_gross_intercept": self.alpha_gross_intercept,
            "alpha_turnover_intercept": self.alpha_turnover_intercept,
            "break_even_unit": _enum_val(self.break_even_unit),
            "rating_coverage_fraction": self.rating_coverage_fraction,
            "months_with_missing_rating": self.months_with_missing_rating,
            "scope": _enum_val(self.scope),
            "provenance": self.provenance.to_dict(),
        }


@dataclass(frozen=True)
class PositionArtefactCapability:
    """Result of the §6.2 capability probe. `supports_drift_adjusted` is derived — it
    requires target weights AND prior/pre-trade drifted holdings to both be present."""

    has_target_weights: bool
    has_prior_holdings: bool
    has_realised_return_since_rebalance: bool
    has_cashflow_events: bool           # maturity, default, coupon
    has_entry_exit_dates: bool
    has_rating_at_trade_date: bool
    supports_drift_adjusted: bool       # derived

    def to_dict(self) -> dict:
        return {
            "has_target_weights": self.has_target_weights,
            "has_prior_holdings": self.has_prior_holdings,
            "has_realised_return_since_rebalance": self.has_realised_return_since_rebalance,
            "has_cashflow_events": self.has_cashflow_events,
            "has_entry_exit_dates": self.has_entry_exit_dates,
            "has_rating_at_trade_date": self.has_rating_at_trade_date,
            "supports_drift_adjusted": self.supports_drift_adjusted,
        }


# ---------------------------------------------------------------------------
# Regimes (spec §3.5, §7). Prediction contrast vs corrected parent (D-E16); hit
# counts only, no interval (D-E15).
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PredictionAssessment:
    evaluable: bool
    refusal_code: RefusalCode | None
    predicted_sign: int | None                  # -1, 0, +1
    observed_sign: int | None
    hit: bool | None
    mechanically_implied_by_template: bool
    contrast_definition: str                    # human-readable, frozen pre-holdout
    contrast_statistic: str                     # "conditional_mean_difference" (primary)
    delta_mean_vs_parent_in_target_state: float | None
    delta_mean_vs_parent_in_other_state: float | None
    n_months_target_state: int
    n_months_other_state: int
    # optional supplementary, populated only if both states clear min_obs_conditional
    delta_alpha_vs_parent_in_target_state: float | None = None
    conditional_alpha_control_set_id: str | None = None
    conditional_alpha_estimable: bool = False

    def to_dict(self) -> dict:
        return {
            "evaluable": self.evaluable,
            "refusal_code": _enum_val(self.refusal_code),
            "predicted_sign": self.predicted_sign,
            "observed_sign": self.observed_sign,
            "hit": self.hit,
            "mechanically_implied_by_template": self.mechanically_implied_by_template,
            "contrast_definition": self.contrast_definition,
            "contrast_statistic": self.contrast_statistic,
            "delta_mean_vs_parent_in_target_state": self.delta_mean_vs_parent_in_target_state,
            "delta_mean_vs_parent_in_other_state": self.delta_mean_vs_parent_in_other_state,
            "n_months_target_state": self.n_months_target_state,
            "n_months_other_state": self.n_months_other_state,
            "delta_alpha_vs_parent_in_target_state": self.delta_alpha_vs_parent_in_target_state,
            "conditional_alpha_control_set_id": self.conditional_alpha_control_set_id,
            "conditional_alpha_estimable": self.conditional_alpha_estimable,
        }


@dataclass(frozen=True)
class RegimeResult:
    applicable: bool                    # False unless a registered state prediction exists
    formation_regime_id: str
    formation_regime_role: RegimeRole
    evaluation_regime_id: str
    months_high: int
    months_low: int
    mean_return_high: float | None
    mean_return_low: float | None
    high_minus_low: float | None
    prediction: PredictionAssessment
    macro_data_contract_id: str
    scope: EvaluationScope
    provenance: EstimatorProvenance
    window: SampleWindow = SampleWindow.DEVELOPMENT   # A7; defaulted so pre-A7 construction is unchanged
    # A7 degrade path: True when computed on the holdout window with a regime state
    # below the pre-registered conditional floor — the reporting layer then licenses
    # point-estimate/direction sentences only. Always False on the development window.
    short_sample: bool = False

    def to_dict(self) -> dict:
        return {
            "applicable": self.applicable,
            "formation_regime_id": self.formation_regime_id,
            "formation_regime_role": _enum_val(self.formation_regime_role),
            "evaluation_regime_id": self.evaluation_regime_id,
            "months_high": self.months_high,
            "months_low": self.months_low,
            "mean_return_high": self.mean_return_high,
            "mean_return_low": self.mean_return_low,
            "high_minus_low": self.high_minus_low,
            "prediction": self.prediction.to_dict(),
            "macro_data_contract_id": self.macro_data_contract_id,
            "scope": _enum_val(self.scope),
            "provenance": self.provenance.to_dict(),
            "window": _enum_val(self.window),
            "short_sample": self.short_sample,
        }


# ---------------------------------------------------------------------------
# Aggregate (spec §3.5) — SharedEvaluationResult, never AdversarialDiagnostics (D-E6).
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SharedEvaluationResult:
    candidate_id: str
    parent_id: str | None
    stage: str                          # "G3" | "G4" | "auditor"
    alignment: SampleAlignment | None
    spanning: SpanningResult
    costs: CostResult
    regimes: RegimeResult
    config_hash: str
    code_version: str
    window: SampleWindow = SampleWindow.DEVELOPMENT   # A7; defaulted so pre-A7 construction is unchanged

    def to_dict(self) -> dict:
        return _jsonsafe({
            "candidate_id": self.candidate_id,
            "parent_id": self.parent_id,
            "stage": self.stage,
            "alignment": self.alignment.to_dict() if self.alignment is not None else None,
            "spanning": self.spanning.to_dict(),
            "costs": self.costs.to_dict(),
            "regimes": self.regimes.to_dict(),
            "config_hash": self.config_hash,
            "code_version": self.code_version,
            "window": _enum_val(self.window),
        })
