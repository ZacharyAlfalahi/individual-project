"""
spanning.py — typed factor-spanning contract (shared/evaluation, spec §4).

A typed SUPERSET of the existing `crowding.py` Layer-1 diagnostic. It reuses the SAME
pre-registered corrected six-factor bundle (`load_crowding_factor_bundle`) and the SAME
audited regression (`characteristic_sort.regress_on_benchmark`), so the alpha / betas /
HAC t-stats are byte-identical to `crowding_diagnostic` (a characterisation test pins
this). What it adds is the typed `SpanningResult`: refusal codes, MANDATORY conditioning
diagnostics (condition number, design rank, VIF, leave-one-control-out alpha, max
pairwise correlation — D-E9), a HAC confidence interval, and estimator provenance.

There is deliberately NO verdict boolean (D-E7): the pre-registered "declare crowded if
alpha insignificant at 5%" rule accepts the null from a failure to reject on a selected,
correlated, finite sample — the exact inferential error this project criticises. Removing
the field makes the unlicensed sentence unwriteable.

The engine's HAC lag is INJECTED via the existing `nw_lags` parameter (the pre-registered
`floor(T**0.25)` rule, resolved by `crowding._resolve_nw_lags`); no engine code is
touched and existing outputs are byte-identical. `shared/evaluation` computes no standard
error itself — it calls the versioned engine and stamps the provenance.
"""

from __future__ import annotations

import hashlib
import json

import numpy as np
import pandas as pd

from agents.quant.library.characteristic_sort import regress_on_benchmark

from .contracts import (
    EstimatorProvenance,
    EvaluationScope,
    RefusalCode,
    SpanningResult,
)
from .crowding import _resolve_nw_lags, load_crowding_factor_bundle
from .thresholds import CrowdingConfig, load_crowding_config

_Z_95 = 1.959963984540054  # two-sided 95% normal quantile


def _finite_or_none(x) -> float | None:
    """Null non-finite floats at construction so the result is JSON-safe and honest:
    perfect collinearity (VIF) and a singular design (condition number) null out rather
    than store NaN/inf."""
    return float(x) if x is not None and np.isfinite(x) else None


def _control_set_identity(config: CrowdingConfig) -> tuple[str, str]:
    """A readable id and a byte-stable hash for the (corrected) control set. Identity is
    the pre-registered factor set + correction state + HAC rule — never a hash of the
    realised data frame."""
    control_set_id = "corrected:" + ",".join(config.factor_set)
    payload = json.dumps(
        {
            "factor_set": list(config.factor_set),
            "correction_state": "corrected",
            "hac_lag_rule": config.hac_lag_rule,
        },
        sort_keys=True,
    )
    control_set_hash = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return control_set_id, control_set_hash


def _provenance(config: CrowdingConfig, lag_used: int) -> EstimatorProvenance:
    lag_rule = (
        "floor(T**0.25)"
        if config.hac_lag_rule == "floor_t_pow_0.25"
        else "newey_west_auto:floor(4*(T/100)**(2/9))"
    )
    return EstimatorProvenance(
        estimator_id="newey_west_hac",
        estimator_version="characteristic_sort.regress_on_benchmark",
        lag_rule=lag_rule,
        lag_used=int(lag_used),
        ddof=0,  # engine HAC uses no small-sample df correction
        annualisation=None,
    )


def _r_squared(y: np.ndarray, X: np.ndarray) -> float | None:
    """OLS R^2 on the same design matrix the engine fits (descriptive only)."""
    sst = float(np.sum((y - y.mean()) ** 2))
    if sst <= 0.0:
        return None
    try:
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    except np.linalg.LinAlgError:
        return None
    ssr = float(np.sum((y - X @ beta) ** 2))
    return 1.0 - ssr / sst


def _vif_by_control(controls: np.ndarray) -> list[float | None]:
    """VIF_i = 1/(1 - R^2_i), R^2_i from regressing control i on the other controls (+
    intercept). Perfect collinearity / constant column -> None (undefined). Near-
    collinearity -> large but finite, the informative signal."""
    T, k = controls.shape
    out: list[float | None] = []
    for i in range(k):
        target = controls[:, i]
        others = np.delete(controls, i, axis=1)
        X = np.column_stack([np.ones(T), others]) if others.shape[1] else np.ones((T, 1))
        sst = float(np.sum((target - target.mean()) ** 2))
        if sst <= 0.0:
            out.append(None)
            continue
        try:
            beta, *_ = np.linalg.lstsq(X, target, rcond=None)
        except np.linalg.LinAlgError:
            out.append(None)
            continue
        ssr = float(np.sum((target - X @ beta) ** 2))
        r2 = 1.0 - ssr / sst
        out.append(None if r2 >= 1.0 else 1.0 / (1.0 - r2))
    return out


def _max_abs_pairwise_corr(augmented: np.ndarray) -> float | None:
    """Max |correlation| among all columns of [candidate | controls] (candidate INCLUDED,
    self-pairs excluded). Including the candidate is what makes 'candidate identical to a
    control' (its own corrected parent, by pre-registration) surface as ~1 — the
    substantive near-singularity D-E9 is about."""
    _, m = augmented.shape
    if m < 2:
        return None
    C = np.corrcoef(augmented, rowvar=False)
    offdiag = C[~np.eye(m, dtype=bool)]
    finite = np.abs(offdiag)[np.isfinite(offdiag)]
    return float(np.max(finite)) if finite.size else None


def _leave_one_control_out_alpha(
    candidate: pd.Series, factors: pd.DataFrame, factor_cols: list[str], nw_lags: int | None
) -> tuple[tuple[str, float | None], ...]:
    """Alpha when each control in turn is dropped — the cheapest instrument for how much
    of the result rests on one (possibly collinear) column (D-E9). Undefined for a single
    control (dropping it leaves an intercept-only model the engine refuses)."""
    if len(factor_cols) < 2:
        return tuple()
    out: list[tuple[str, float | None]] = []
    for dropped in factor_cols:
        reduced = factors.drop(columns=[dropped])
        reg = regress_on_benchmark(candidate, reduced, nw_lags=nw_lags)
        out.append((dropped, _finite_or_none(reg["alpha"])))
    return tuple(out)


def spanning_regression(
    candidate_returns: pd.Series,
    *,
    config: CrowdingConfig | None = None,
    factors: pd.DataFrame | None = None,
    min_obs: int | None = None,
    scope: EvaluationScope = EvaluationScope.SUPPLEMENTARY,
) -> SpanningResult:
    """Typed factor-spanning regression of one candidate against the fixed corrected
    control set.

    `candidate_returns` is a monthly (month-end) excess-return Series indexed by date.
    `config`/`factors` default to the pre-registered `crowding:` block when omitted
    (passing `factors` avoids re-reading the parquets). `min_obs` defaults to
    `config.min_obs` (the pre-registered 60-month floor). Never raises on data shape —
    degenerate cases return a typed refusal, not NaN or an exception (D-E3).
    """
    if not isinstance(candidate_returns, pd.Series):
        raise TypeError("candidate_returns must be a pandas Series indexed by date")
    if config is None:
        config = load_crowding_config()
    if factors is None:
        factors = load_crowding_factor_bundle(config)

    factor_cols = list(config.factor_set)
    n_controls = len(factor_cols)
    mo = config.min_obs if min_obs is None else int(min_obs)
    control_set_id, control_set_hash = _control_set_identity(config)

    # Normalise the candidate index and replicate the engine's inner-join + dropna so the
    # T here is exactly the sample the regression fits on.
    idx = pd.to_datetime(candidate_returns.index).astype("datetime64[ns]")
    cand = pd.Series(candidate_returns.to_numpy(), index=idx)
    y_df = pd.DataFrame({"date": cand.index, "_y": cand.to_numpy()})
    merged = y_df.merge(factors, on="date", how="inner").dropna(subset=["_y", *factor_cols])
    T = len(merged)
    nw_lags = _resolve_nw_lags(config.hac_lag_rule, T)
    sample_id = (
        f"{merged['date'].min().date()}..{merged['date'].max().date()}:T{T}"
        if T > 0
        else "empty"
    )

    def _refusal(code: RefusalCode, *, cond, rank, vif, loo, max_corr, lag_used) -> SpanningResult:
        return SpanningResult(
            estimable=False,
            refusal_code=code,
            alpha_monthly=None,
            alpha_se=None,
            t_hac=None,
            ci_low=None,
            ci_high=None,
            betas=tuple(),
            r_squared=None,
            n_obs=T,
            n_controls=n_controls,
            min_obs=mo,
            sample_id=sample_id,
            control_set_id=control_set_id,
            control_set_hash=control_set_hash,
            control_correction_state="corrected",
            condition_number=cond,
            design_rank=rank,
            vif_by_control=vif,
            leave_one_control_out_alpha=loo,
            max_abs_pairwise_corr=max_corr,
            scope=scope,
            provenance=_provenance(config, lag_used),
        )

    # (1) Insufficient observations — refuse before fitting; no conditioning diagnostics
    #     (there is no design to condition). n_obs populated (behaviour table §4.3).
    if T < mo:
        lag_used = 0 if nw_lags is None else max(0, min(int(nw_lags), max(0, T - 1)))
        return _refusal(
            RefusalCode.INSUFFICIENT_OBSERVATIONS,
            cond=None, rank=None, vif=tuple(), loo=tuple(), max_corr=None, lag_used=lag_used,
        )

    # Assemble the design matrix X = [1 | controls] on the fitted sample.
    controls = np.column_stack([merged[c].to_numpy(dtype=float) for c in factor_cols])
    y = merged["_y"].to_numpy(dtype=float)
    X = np.column_stack([np.ones(T), controls])
    condition_number = _finite_or_none(float(np.linalg.cond(X)))
    design_rank = int(np.linalg.matrix_rank(X))
    vif = tuple(zip(factor_cols, _vif_by_control(controls)))
    loo = _leave_one_control_out_alpha(cand, factors, factor_cols, nw_lags)
    max_corr = _max_abs_pairwise_corr(np.column_stack([y, controls]))

    # (2) Rank deficiency — never drop a column (D-E9); surface it as a diagnostic. This
    #     also captures T <= k (cannot fit OLS). Conditioning diagnostics ARE populated.
    if design_rank < n_controls + 1:
        lag_used = 0 if nw_lags is None else max(0, min(int(nw_lags), T - 1))
        return _refusal(
            RefusalCode.RANK_DEFICIENT,
            cond=condition_number, rank=design_rank, vif=vif, loo=loo,
            max_corr=max_corr, lag_used=lag_used,
        )

    # (3) Estimable — reuse the audited engine verbatim, then enrich.
    reg = regress_on_benchmark(cand, factors, nw_lags=nw_lags)
    alpha = float(reg["alpha"])
    alpha_t = float(reg["alpha_t"])
    lag_used = int(reg["nw_lags_used"])
    alpha_se = alpha / alpha_t if np.isfinite(alpha_t) and alpha_t != 0.0 else None
    ci_low = alpha - _Z_95 * alpha_se if alpha_se is not None else None
    ci_high = alpha + _Z_95 * alpha_se if alpha_se is not None else None
    betas = tuple((c, float(reg["betas"][c])) for c in factor_cols)

    return SpanningResult(
        estimable=True,
        refusal_code=None,
        alpha_monthly=_finite_or_none(alpha),
        alpha_se=alpha_se,
        t_hac=_finite_or_none(alpha_t),
        ci_low=ci_low,
        ci_high=ci_high,
        betas=betas,
        r_squared=_finite_or_none(_r_squared(y, X)),
        n_obs=int(reg["n_obs"]),
        n_controls=n_controls,
        min_obs=mo,
        sample_id=sample_id,
        control_set_id=control_set_id,
        control_set_hash=control_set_hash,
        control_correction_state="corrected",
        condition_number=condition_number,
        design_rank=design_rank,
        vif_by_control=vif,
        leave_one_control_out_alpha=loo,
        max_abs_pairwise_corr=max_corr,
        scope=scope,
        provenance=_provenance(config, lag_used),
    )
