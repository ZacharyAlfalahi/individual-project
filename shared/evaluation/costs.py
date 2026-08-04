"""
costs.py — turnover, cost-adjusted returns, and break-even cost (shared/evaluation, §6).

Two layers, per the plan's costs/turnover split:

  * PURE MATH (built + fully unit-tested now): cost-adjusted returns, the closed-form
    break-evens, the lambda multiplier, and the assumed-turnover-grid fallback. Every
    function takes an INJECTED turnover series — no engine dependency. `cost_result_from_turnover`
    assembles a full `CostResult` from injected turnover; it is the one-line plug-in for
    a passive weight-export layer (D-E11 step 1).

  * TURNOVER SOURCE (typed refusal stub): the current run artefact (`StrategyResult`)
    persists only `strategy_ret` + `n_bonds` — NO per-position weights — so neither the
    drift-adjusted nor the target-to-target proxy turnover is computable (P4). `evaluate_costs`
    probes the artefact and returns a real `CostResult(estimable=False,
    refusal_code=ARTEFACT_CAPABILITY_MISSING)` that flows to a licensed sentence. Never
    a NaN, never an exception (D-E3).

Field named cost_adjusted_*, never net_* (D-E10). Break-even alpha denominator is the
CONTROL-ADJUSTED turnover intercept, never unconditional mean turnover (D-E13). No AUM
capacity ceiling (D-E/A4): no size-dependent impact input exists.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from agents.quant.library.characteristic_sort import regress_on_benchmark, summarize_returns

from .contracts import (
    BreakEvenStatus,
    CostResult,
    CostScenarioResult,
    CostUnit,
    EstimatorProvenance,
    EvaluationScope,
    PositionArtefactCapability,
    RefusalCode,
    TurnoverMethod,
    WeightingScheme,
)
from .thresholds import CostsConfig, load_costs_config

_BPS = 1.0e4


def _fon(x) -> float | None:
    """Finite-or-None: null non-finite floats at construction so results are JSON-safe."""
    return float(x) if x is not None and np.isfinite(x) else None


def _cost_provenance(lag_used: int, estimator_id: str = "newey_west_hac") -> EstimatorProvenance:
    # ddof describes the break-even headline numbers, which come from the HAC
    # `regress_on_benchmark` intercepts (no small-sample df correction -> ddof=0). The
    # scenario Sharpes reuse `summarize_returns` (ddof=1) as a secondary quantity.
    return EstimatorProvenance(
        estimator_id=estimator_id,
        estimator_version="characteristic_sort.regress_on_benchmark+summarize_returns",
        lag_rule="floor(T**0.25)",
        lag_used=int(lag_used),
        ddof=0,
        annualisation="sqrt(months_per_year)",
    )


def _normalise(series: pd.Series) -> pd.Series:
    if not isinstance(series, pd.Series):
        raise TypeError("series must be a pandas Series indexed by date")
    idx = pd.to_datetime(series.index).astype("datetime64[ns]")
    s = pd.Series(series.to_numpy(dtype=float), index=idx)
    s = s[s.notna()]
    # Defensive: collapse a doubled month-end label (keep-last) so `_common`'s reindex
    # never raises on duplicate labels (invariant #1).
    return s[~s.index.duplicated(keep="last")]


def _common(a: pd.Series, b: pd.Series) -> tuple[pd.Series, pd.Series]:
    """Both series restricted to their common month-end dates (aligned, same order)."""
    sa, sb = _normalise(a), _normalise(b)
    common = sa.index.intersection(sb.index)
    return sa.reindex(common), sb.reindex(common)


def _refused_cost(
    code: RefusalCode,
    *,
    turnover_method: TurnoverMethod,
    weighting_scheme: WeightingScheme,
    turnover_series_id: str,
    proxy_bias_direction: str | None,
    scope: EvaluationScope,
) -> CostResult:
    """A typed, all-None CostResult refusal — no scenarios, no break-even, no NaN."""
    return CostResult(
        estimable=False,
        refusal_code=code,
        turnover_method=turnover_method,
        weighting_scheme=weighting_scheme,
        turnover_mean=None,
        turnover_median=None,
        turnover_series_id=turnover_series_id,
        proxy_bias_direction=proxy_bias_direction,
        scenarios=tuple(),
        break_even_mean_cost_bps=None,
        break_even_mean_status=BreakEvenStatus.UNDEFINED,
        break_even_alpha_cost_bps=None,
        break_even_alpha_status=BreakEvenStatus.UNDEFINED,
        break_even_alpha_multiplier=None,
        alpha_gross_intercept=None,
        alpha_turnover_intercept=None,
        break_even_unit=CostUnit.ONE_WAY,
        rating_coverage_fraction=0.0,
        months_with_missing_rating=0,
        scope=scope,
        provenance=_cost_provenance(0, estimator_id="none"),
    )


# ---------------------------------------------------------------------------
# Position-artefact capability probe (§6.2)
# ---------------------------------------------------------------------------

def probe_position_artefacts(run_artefact) -> PositionArtefactCapability:
    """Duck-typed probe over a run artefact (a `StrategyResult` dataclass or a dict).
    `supports_drift_adjusted` requires BOTH target weights AND prior/pre-trade drifted
    holdings — the current artefact carries neither, so it is False."""

    def _present(v) -> bool:
        # Present = not None and (not a container, or a non-empty one) — an empty
        # DataFrame/dict/list is NOT "has weights".
        if v is None:
            return False
        try:
            return len(v) > 0
        except TypeError:
            return True

    def _has(*names: str) -> bool:
        if run_artefact is None:
            return False
        for name in names:
            v = run_artefact.get(name) if isinstance(run_artefact, dict) else getattr(run_artefact, name, None)
            if _present(v):
                return True
        return False

    has_target = _has("target_weights", "weights")
    has_prior = _has("prior_holdings", "pretrade_weights", "drifted_weights")
    return PositionArtefactCapability(
        has_target_weights=has_target,
        has_prior_holdings=has_prior,
        has_realised_return_since_rebalance=_has("realised_return_since_rebalance"),
        has_cashflow_events=_has("cashflow_events"),
        has_entry_exit_dates=_has("entry_exit_dates"),
        has_rating_at_trade_date=_has("rating_at_trade_date"),
        supports_drift_adjusted=has_target and has_prior,
    )


# ---------------------------------------------------------------------------
# Pure math (injected turnover)
# ---------------------------------------------------------------------------

def cost_adjusted_series(gross: pd.Series, turnover: pd.Series, cost_bps: float) -> pd.Series:
    """cost_adjusted_t = gross_t - (cost_bps / 1e4) * turnover_t, on the common months.
    The cost is applied verbatim in the caller's unit convention — never silently
    converted between one-way and round-trip (the JNPS-convention hazard, P5/D-E12)."""
    g, to = _common(gross, turnover)
    return g - (cost_bps / _BPS) * to


def break_even_mean(gross: pd.Series, turnover: pd.Series) -> tuple[float | None, BreakEvenStatus, RefusalCode | None]:
    """Mean-return break-even c_BE^mu = mean(r_gross) / mean(TO), in bps."""
    g, to = _common(gross, turnover)
    if len(g) == 0 or float(to.mean()) <= 0.0:
        return None, BreakEvenStatus.UNDEFINED, RefusalCode.ZERO_TURNOVER
    if float(g.mean()) <= 0.0:
        return None, BreakEvenStatus.UNDEFINED, RefusalCode.NON_POSITIVE_GROSS
    return float(g.mean() / to.mean()) * _BPS, BreakEvenStatus.FOUND, None


def break_even_alpha(
    gross: pd.Series, turnover: pd.Series, factors: pd.DataFrame, *, nw_lags: int | None
) -> dict:
    """Alpha break-even c_BE^alpha = alpha_hat_r / alpha_hat_TO (bps), where alpha_hat_r is
    the intercept of gross-on-controls and alpha_hat_TO the intercept of turnover-on-controls
    on the SAME sample and design matrix (D-E13). Exact: OLS is linear in the LHS, so
    alpha_hat(c) = alpha_hat_r - c*alpha_hat_TO is affine in c and has at most one root."""
    g, to = _common(gross, turnover)  # identical sample for both intercepts
    reg_r = regress_on_benchmark(g, factors, nw_lags=nw_lags)
    reg_to = regress_on_benchmark(to, factors, nw_lags=nw_lags)
    alpha_r = float(reg_r["alpha"])
    alpha_to = float(reg_to["alpha"])
    lag_used = int(reg_r["nw_lags_used"])
    out = {
        "alpha_gross_intercept": alpha_r,
        "alpha_turnover_intercept": alpha_to,
        "n_obs": int(reg_r["n_obs"]),
        "lag_used": lag_used,
    }
    if not np.isfinite(alpha_r) or not np.isfinite(alpha_to):
        # The regression could not be fitted (no common sample, or T <= k). Report the
        # honest cause, NOT ZERO_TURNOVER (which would misname it).
        code = RefusalCode.NO_COMMON_SAMPLE if out["n_obs"] == 0 else RefusalCode.INSUFFICIENT_OBSERVATIONS
        out.update(value_bps=None, status=BreakEvenStatus.UNDEFINED, refusal=code)
        return out
    if alpha_r <= 0.0:
        out.update(value_bps=None, status=BreakEvenStatus.UNDEFINED, refusal=RefusalCode.NON_POSITIVE_GROSS)
        return out
    if alpha_to <= 0.0:
        out.update(
            value_bps=None,
            status=BreakEvenStatus.UNDEFINED,
            refusal=RefusalCode.NON_POSITIVE_TURNOVER_INTERCEPT,
        )
        return out
    out.update(value_bps=(alpha_r / alpha_to) * _BPS, status=BreakEvenStatus.FOUND, refusal=None)
    return out


def lambda_break_even(alpha_gross_intercept: float, alpha_turnover_intercept: float, registered_bps: float) -> float | None:
    """Dimensionless lambda: the strategy breaks even at lambda times the registered cost
    schedule. lambda = (alpha_r / alpha_TO) / (registered_bps / 1e4)."""
    if alpha_turnover_intercept <= 0.0 or registered_bps <= 0.0:
        return None
    return (alpha_gross_intercept / alpha_turnover_intercept) / (registered_bps / _BPS)


def break_even_grid(gross: pd.Series, turnover_grid: tuple[float, ...]) -> tuple[tuple[float, float | None], ...]:
    """Fallback when NO measured turnover exists: for each assumed constant monthly
    turnover rate g, the break-even cost = mean(r_gross) / g (bps). Non-empty and honest
    where a single measured number is unavailable (spec §6 fallback)."""
    gg = _normalise(gross)
    mg = float(gg.mean()) if len(gg) else float("nan")
    out: list[tuple[float, float | None]] = []
    for g in turnover_grid:
        out.append((float(g), (mg / g) * _BPS if np.isfinite(mg) and g > 0 else None))
    return tuple(out)


def _build_scenarios(
    gross: pd.Series,
    turnover: pd.Series,
    config: CostsConfig,
    *,
    nw_lags: int | None,
    months_per_year: int,
) -> tuple[CostScenarioResult, ...]:
    """One `CostScenarioResult` per USABLE scenario. Scenarios with an unresolved source
    (P5) are skipped, never emitted. A scenario with ig != hy needs bucketed (per-rating)
    turnover, which the artefact cannot supply, so only single-rate scenarios are computed."""
    g, to = _common(gross, turnover)
    to_mean = float(to.mean()) if len(to) else float("nan")
    results: list[CostScenarioResult] = []
    for spec in config.scenarios:
        if not spec.usable or spec.ig_bps != spec.hy_bps:
            continue
        rate = spec.ig_bps
        adj = cost_adjusted_series(g, to, rate)
        sharpe = summarize_returns(adj, nw_lags, months_per_year)["sharpe"] if len(adj) else float("nan")
        results.append(
            CostScenarioResult(
                scenario_id=spec.scenario_id,
                cost_bps_ig=spec.ig_bps,
                cost_bps_hy=spec.hy_bps,
                unit=spec.unit,
                source_citation=spec.source,
                cost_adjusted_mean_monthly=float(adj.mean()) if len(adj) else float("nan"),
                cost_adjusted_sharpe=_fon(sharpe),
                cost_drag_bps_monthly=rate * to_mean,
            )
        )
    return tuple(results)


def cost_result_from_turnover(
    gross: pd.Series,
    turnover: pd.Series,
    factors: pd.DataFrame,
    *,
    config: CostsConfig | None = None,
    weighting_scheme: WeightingScheme,
    turnover_method: TurnoverMethod,
    turnover_series_id: str,
    proxy_bias_direction: str | None = None,
    nw_lags: int | None = None,
    months_per_year: int = 12,
    scope: EvaluationScope = EvaluationScope.SUPPLEMENTARY,
    rating_coverage_fraction: float = 1.0,
    months_with_missing_rating: int = 0,
) -> CostResult:
    """Full estimable `CostResult` from an INJECTED turnover series. This is the plug-in
    the passive-weight-export adjudication (D-E11) unblocks; it is fully testable now."""
    if config is None:
        config = load_costs_config()
    g, to = _common(gross, turnover)
    # No common months -> a typed refusal, NOT an estimable result with all-NaN scenarios.
    if len(g) == 0:
        return _refused_cost(
            RefusalCode.NO_COMMON_SAMPLE,
            turnover_method=turnover_method,
            weighting_scheme=weighting_scheme,
            turnover_series_id=turnover_series_id,
            proxy_bias_direction=proxy_bias_direction,
            scope=scope,
        )
    scenarios = _build_scenarios(g, to, config, nw_lags=nw_lags, months_per_year=months_per_year)

    be_mean_bps, be_mean_status, _ = break_even_mean(g, to)
    ba = break_even_alpha(g, to, factors, nw_lags=nw_lags)

    return CostResult(
        estimable=True,
        refusal_code=None,
        turnover_method=turnover_method,
        weighting_scheme=weighting_scheme,
        turnover_mean=_fon(to.mean()),
        turnover_median=_fon(to.median()),
        turnover_series_id=turnover_series_id,
        proxy_bias_direction=proxy_bias_direction,
        scenarios=scenarios,
        break_even_mean_cost_bps=be_mean_bps,
        break_even_mean_status=be_mean_status,
        break_even_alpha_cost_bps=ba["value_bps"],
        break_even_alpha_status=ba["status"],
        break_even_alpha_multiplier=None,  # requires the registered TWO-rate schedule + bucketed turnover (blocked)
        alpha_gross_intercept=_fon(ba["alpha_gross_intercept"]),
        alpha_turnover_intercept=_fon(ba["alpha_turnover_intercept"]),
        break_even_unit=CostUnit.ONE_WAY,
        rating_coverage_fraction=rating_coverage_fraction,
        months_with_missing_rating=months_with_missing_rating,
        scope=scope,
        provenance=_cost_provenance(ba["lag_used"]),
    )


def evaluate_costs(
    run_artefact,
    *,
    weighting_scheme: WeightingScheme = WeightingScheme.PAR,
    scope: EvaluationScope = EvaluationScope.SUPPLEMENTARY,
) -> CostResult:
    """Turnover-source entry point. Probes the run artefact; the current artefact carries
    no per-position weights, so this returns a real ARTEFACT_CAPABILITY_MISSING refusal
    (the selected method is the proxy, since drift-adjusted is unsupported — but even the
    proxy needs target weights the artefact lacks)."""
    probe = probe_position_artefacts(run_artefact)
    if probe.supports_drift_adjusted:
        # Fail loud (never silently refuse): the artefact now carries weights, so refusing
        # here would flatter costs (report gross-only) — the exact bias the project
        # criticises. The D-E11 plug-in is to compute turnover and call
        # cost_result_from_turnover; wire it rather than reaching this guard.
        raise RuntimeError(
            "probe reports drift-adjusted turnover is now available — route through "
            "cost_result_from_turnover with the exported weights (D-E11 plug-in), "
            "do not call evaluate_costs"
        )
    return _refused_cost(
        RefusalCode.ARTEFACT_CAPABILITY_MISSING,
        turnover_method=TurnoverMethod.TARGET_TO_TARGET_PROXY,
        weighting_scheme=weighting_scheme,
        turnover_series_id="unavailable",
        proxy_bias_direction="unknown",
        scope=scope,
    )
