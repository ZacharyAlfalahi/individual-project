"""
regimes.py — regime decomposition + falsifiable prediction contrast (shared/evaluation, §7).

Three regime roles, per D-E14 — but this pass builds only what is estimable now:

  * EVALUATION regime (BUILT): a FROZEN development median (resolved from extension_1,
    never restated), used to DECOMPOSE realised returns comparably across periods. This
    is exactly the right construction for decomposition, where comparability is the point.

  * PREDICTION CONTRAST (BUILT, pure): the scored quantity is the contrast against the
    corrected parent WITHIN the pre-specified state, as a conditional MEAN difference, not
    a conditional alpha (D-E16) — a conditional alpha needs a control regression inside a
    regime subsample, which is not estimable under the 60-month guard on a holdout slice.
    Hit test uses a SIGN only. No population interval (D-E15): proposals share a parent,
    a template, and the same calendar months, so a Wilson interval would imply a
    population claim the design cannot support — report exact counts + a per-parent
    breakdown instead.

  * FORMATION regime (DEFERRED, A3 + P2): the expanding, strictly past-only median that
    determines positions. `expanding_past_only_median` is provided as a pure, look-ahead-
    guarded function (and is tested by the look-ahead guard), but it is NOT ADOPTED as the
    operational/confirmatory construction: adopting it over extension_1's frozen median is
    a dated pre-holdout amendment (A3), and its burn-in cost depends on pre-2002 macro
    history (P2). Until then the retained fixed-median construction runs as the AUDIT_VARIANT.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .contracts import (
    EstimatorProvenance,
    EvaluationScope,
    PredictionAssessment,
    RefusalCode,
    RegimeResult,
    RegimeRole,
    SampleWindow,
)
from .thresholds import RegimesConfig, load_regimes_config


def _regime_provenance() -> EstimatorProvenance:
    return EstimatorProvenance(
        estimator_id="conditional_mean_difference",
        estimator_version="shared.evaluation.regimes",
        lag_rule="none",
        lag_used=0,
        ddof=0,
        annualisation=None,
    )


def _normalise(series: pd.Series) -> pd.Series:
    if not isinstance(series, pd.Series):
        raise TypeError("series must be a pandas Series indexed by date")
    idx = pd.to_datetime(series.index).astype("datetime64[ns]")
    s = pd.Series(series.to_numpy(dtype=float), index=idx)
    s = s[s.notna()]
    # Defensive: collapse a doubled month-end label (keep-last) so the intersection +
    # reindex paths never raise on duplicate labels (invariant #1).
    return s[~s.index.duplicated(keep="last")]


def _bool_series(series: pd.Series) -> pd.Series:
    idx = pd.to_datetime(series.index).astype("datetime64[ns]")
    # Route through float so a NaN maps to False (NOT True): np.asarray(nan, dtype=bool)
    # would silently mark a missing month as "in the target state".
    f = np.asarray(series.to_numpy(), dtype=float)
    return pd.Series(np.where(np.isnan(f), False, f != 0.0), index=idx)


# ---------------------------------------------------------------------------
# Formation regime — DEFERRED (A3 + P2). Pure, look-ahead-guarded; NOT adopted.
# ---------------------------------------------------------------------------

def expanding_past_only_median(
    spread: pd.Series, *, min_history_months: int, lag_months: int
) -> pd.Series:
    """Expanding, STRICTLY past-only median of the macro spread, lagged by the macro
    release delay — the formation-regime threshold. At month t the threshold uses only
    observations dated at or before (t - lag_months) AND strictly before t: the current
    month's spread is not yet observed, so including it would be look-ahead — the exact
    bias class the Auditor exists to detect. Below `min_history_months` of usable history
    the threshold is NaN (burn-in).

    NOT ADOPTED as the operational formation regime: that is amendment A3 (pending) and its
    burn-in depends on pre-2002 macro history (P2). Provided pure + tested (look-ahead guard).

    Positional (not date-arithmetic): for a gap-free monthly series the threshold at month i
    is the median of positions [0, i - lag_months), i.e. strictly-past observations backed off
    by the release lag. Positional indexing is used deliberately — `t - DateOffset(months=k)`
    lands on inconsistent days for a month-end index (Feb 28 - 1mo = Jan 28, which would drop a
    Jan 31 observation). Exact calendar-release timing is part of the A3 macro-data contract.
    """
    s = _normalise(spread).sort_index()
    vals = s.to_numpy()
    n = len(vals)
    out = np.full(n, np.nan)
    for i in range(n):
        hi = max(i - lag_months, 0)      # exclusive end; positions [0, hi) are all < i
        window = vals[:hi]
        if len(window) >= min_history_months:
            out[i] = float(np.median(window))
    return pd.Series(out, index=s.index)


# ---------------------------------------------------------------------------
# Prediction contrast (BUILT, pure) — vs corrected parent, within the pre-specified state.
# ---------------------------------------------------------------------------

def prediction_contrast(
    extension: pd.Series,
    parent: pd.Series,
    target_state: pd.Series,
    *,
    predicted_sign: int | None,
    mechanically_implied: bool,
    contrast_definition: str,
) -> PredictionAssessment:
    """Signed conditional mean difference of the extension vs the corrected parent, in the
    target state and the other state (D-E16). `target_state` is a boolean Series (True =
    the pre-specified target state). Returns PREDICTION_NOT_EVALUABLE if there is no
    falsifiable contrast (no registered sign, or a state with zero months)."""
    e = _normalise(extension)
    p = _normalise(parent)
    ts = _bool_series(target_state)
    common = e.index.intersection(p.index).intersection(ts.index)

    target_idx = common[ts.reindex(common).to_numpy()]
    other_idx = common[~ts.reindex(common).to_numpy()]
    n_t, n_o = len(target_idx), len(other_idx)

    def _not_evaluable() -> PredictionAssessment:
        return PredictionAssessment(
            evaluable=False,
            refusal_code=RefusalCode.PREDICTION_NOT_EVALUABLE,
            predicted_sign=predicted_sign,
            observed_sign=None,
            hit=None,
            mechanically_implied_by_template=mechanically_implied,
            contrast_definition=contrast_definition,
            contrast_statistic="conditional_mean_difference",
            delta_mean_vs_parent_in_target_state=None,
            delta_mean_vs_parent_in_other_state=None,
            n_months_target_state=n_t,
            n_months_other_state=n_o,
        )

    if predicted_sign not in (-1, 0, 1) or n_t == 0 or n_o == 0:
        return _not_evaluable()

    delta_t = float(e.reindex(target_idx).mean() - p.reindex(target_idx).mean())
    delta_o = float(e.reindex(other_idx).mean() - p.reindex(other_idx).mean())
    observed_sign = int(np.sign(delta_t))
    return PredictionAssessment(
        evaluable=True,
        refusal_code=None,
        predicted_sign=predicted_sign,
        observed_sign=observed_sign,
        hit=(observed_sign == predicted_sign),
        mechanically_implied_by_template=mechanically_implied,
        contrast_definition=contrast_definition,
        contrast_statistic="conditional_mean_difference",
        delta_mean_vs_parent_in_target_state=delta_t,
        delta_mean_vs_parent_in_other_state=delta_o,
        n_months_target_state=n_t,
        n_months_other_state=n_o,
    )


# ---------------------------------------------------------------------------
# Evaluation-regime decomposition + assembled RegimeResult (BUILT).
# ---------------------------------------------------------------------------

def evaluate_regimes(
    candidate_returns: pd.Series,
    spread: pd.Series,
    *,
    parent_returns: pd.Series | None = None,
    config: RegimesConfig | None = None,
    predicted_sign: int | None = None,
    mechanically_implied: bool = False,
    contrast_definition: str = "delta_vs_corrected_parent_in_high_spread_state",
    scope: EvaluationScope = EvaluationScope.SUPPLEMENTARY,
    window: SampleWindow = SampleWindow.DEVELOPMENT,
) -> RegimeResult:
    """Decompose the candidate's realised returns by the FROZEN-median evaluation regime
    (high = spread > frozen median), and — where a falsifiable state prediction is
    registered and the corrected parent is supplied — score the prediction contrast. The
    formation regime is referenced for identity only; it is DEFERRED (A3 + P2).

    The decomposition means and the sign contrast are deliberately mean/sign statistics
    (D-E16) and REMAIN computable on `window=HOLDOUT`; per A7 the result then carries
    `short_sample=True` when either regime state falls below `min_obs_conditional`,
    which licenses point-estimate/direction sentences only at the reporting layer."""
    if config is None:
        config = load_regimes_config()

    cand = _normalise(candidate_returns)
    spr = _normalise(spread)
    common = cand.index.intersection(spr.index)
    c = cand.reindex(common)
    high_mask = pd.Series(spr.reindex(common).to_numpy() > config.evaluation_median, index=common)

    high_ret = c[high_mask.to_numpy()]
    low_ret = c[~high_mask.to_numpy()]
    months_high, months_low = int(len(high_ret)), int(len(low_ret))
    mean_high = float(high_ret.mean()) if months_high else None
    mean_low = float(low_ret.mean()) if months_low else None
    high_minus_low = (mean_high - mean_low) if (mean_high is not None and mean_low is not None) else None

    if parent_returns is not None and predicted_sign is not None:
        prediction = prediction_contrast(
            c,
            parent_returns,
            high_mask,
            predicted_sign=predicted_sign,
            mechanically_implied=mechanically_implied,
            contrast_definition=contrast_definition,
        )
        applicable = True
    else:
        prediction = PredictionAssessment(
            evaluable=False,
            refusal_code=RefusalCode.PREDICTION_NOT_EVALUABLE,
            predicted_sign=predicted_sign,
            observed_sign=None,
            hit=None,
            mechanically_implied_by_template=mechanically_implied,
            contrast_definition=contrast_definition,
            contrast_statistic="conditional_mean_difference",
            delta_mean_vs_parent_in_target_state=None,
            delta_mean_vs_parent_in_other_state=None,
            n_months_target_state=months_high,
            n_months_other_state=months_low,
        )
        applicable = False

    short_sample = window is SampleWindow.HOLDOUT and (
        months_high < config.min_obs_conditional or months_low < config.min_obs_conditional
    )
    return RegimeResult(
        applicable=applicable,
        formation_regime_id="expanding_past_only_median:DEFERRED(A3+P2)",
        formation_regime_role=RegimeRole.FORMATION,
        evaluation_regime_id=f"frozen_dev_median={config.evaluation_median}",
        months_high=months_high,
        months_low=months_low,
        mean_return_high=mean_high,
        mean_return_low=mean_low,
        high_minus_low=high_minus_low,
        prediction=prediction,
        macro_data_contract_id=config.macro_data_contract_id,
        scope=scope,
        provenance=_regime_provenance(),
        window=window,
        short_sample=short_sample,
    )


def not_applicable_result(
    *,
    config: RegimesConfig | None = None,
    scope: EvaluationScope = EvaluationScope.SUPPLEMENTARY,
    mechanically_implied: bool = False,
    reason: str = "no_spread_series_supplied",
    window: SampleWindow = SampleWindow.DEVELOPMENT,
) -> RegimeResult:
    """A not-applicable RegimeResult for the orchestrator when no macro spread series is
    supplied (nothing to decompose). Carries a real PREDICTION_NOT_EVALUABLE refusal."""
    if config is None:
        config = load_regimes_config()
    prediction = PredictionAssessment(
        evaluable=False,
        refusal_code=RefusalCode.PREDICTION_NOT_EVALUABLE,
        predicted_sign=None,
        observed_sign=None,
        hit=None,
        mechanically_implied_by_template=mechanically_implied,
        contrast_definition=reason,
        contrast_statistic="conditional_mean_difference",
        delta_mean_vs_parent_in_target_state=None,
        delta_mean_vs_parent_in_other_state=None,
        n_months_target_state=0,
        n_months_other_state=0,
    )
    return RegimeResult(
        applicable=False,
        formation_regime_id="expanding_past_only_median:DEFERRED(A3+P2)",
        formation_regime_role=RegimeRole.FORMATION,
        evaluation_regime_id=f"frozen_dev_median={config.evaluation_median}",
        months_high=0,
        months_low=0,
        mean_return_high=None,
        mean_return_low=None,
        high_minus_low=None,
        prediction=prediction,
        macro_data_contract_id=config.macro_data_contract_id,
        scope=scope,
        provenance=_regime_provenance(),
        window=window,
    )
