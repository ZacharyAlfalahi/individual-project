"""
test_shared_eval_costs.py — the cost math + the turnover-source refusal stub. Includes the
D-E13 fixture: the control-adjusted break-even (alpha_r/alpha_TO) differs measurably from
the naive alpha_r/mean(TO), and the affine identity alpha(c) = alpha_r - c*alpha_TO holds
to machine precision. Failure-dominant: the artefact refusal, all three break-even edge
cases, the blocked scenario.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from agents.quant.library.characteristic_sort import regress_on_benchmark
from shared.evaluation.contracts import (
    BreakEvenStatus,
    RefusalCode,
    TurnoverMethod,
    WeightingScheme,
)
from shared.evaluation.costs import (
    break_even_alpha,
    break_even_grid,
    break_even_mean,
    cost_result_from_turnover,
    evaluate_costs,
    probe_position_artefacts,
)

from _shared_eval_fixtures import candidate, factor_frame, turnover_series  # noqa: E402


# ---------------------------------------------------------------------------
# turnover source: the P4 refusal stub (returns a real object, never raises)
# ---------------------------------------------------------------------------

def test_probe_current_artefact_has_no_weights() -> None:
    class FakeStrategyResult:  # mirrors the real StrategyResult surface (no weights)
        monthly_returns = pd.DataFrame({"date": [], "strategy_ret": [], "n_bonds": []})
        summary = {}
        bookkeeping = {}

    cap = probe_position_artefacts(FakeStrategyResult())
    assert not cap.supports_drift_adjusted
    assert not cap.has_target_weights and not cap.has_prior_holdings


def test_evaluate_costs_refuses_without_weights() -> None:
    cr = evaluate_costs(run_artefact=None)
    assert not cr.estimable and cr.refusal_code is RefusalCode.ARTEFACT_CAPABILITY_MISSING
    assert cr.turnover_method is TurnoverMethod.TARGET_TO_TARGET_PROXY  # selected method recorded
    assert cr.turnover_mean is None and cr.scenarios == ()


def test_evaluate_costs_fails_loud_if_weights_appear() -> None:
    # The guard must fail LOUD (never silently refuse and flatter costs) if a future
    # artefact carries weights but the plug-in was not wired.
    artefact = {"target_weights": {"a": 0.5}, "prior_holdings": {"a": 0.4}}
    with pytest.raises(RuntimeError, match="cost_result_from_turnover"):
        evaluate_costs(run_artefact=artefact)


# ---------------------------------------------------------------------------
# D-E13: closed-form break-even, correct denominator, affine identity
# ---------------------------------------------------------------------------

def test_break_even_alpha_affine_identity_and_differs_from_naive() -> None:
    fr = factor_frame(T=200, seed=1)
    gross = candidate(fr, a0=0.002, betas={"mktb": 0.5}, noise=0.0005, seed=2)
    to = turnover_series(fr, base=0.2, cov_factor="mktb", cov=0.6, seed=11)  # covaries w/ a control

    ba = break_even_alpha(gross, to, fr, nw_lags=3)
    ar, at = ba["alpha_gross_intercept"], ba["alpha_turnover_intercept"]
    assert ba["status"] is BreakEvenStatus.FOUND

    # affine identity to machine precision: alpha(c) = alpha_r - c*alpha_TO
    for c in (0.01, 0.03, 0.08):
        reg_c = regress_on_benchmark(pd.Series(gross.to_numpy() - c * to.to_numpy(), index=gross.index), fr, nw_lags=3)
        assert reg_c["alpha"] == pytest.approx(ar - c * at, abs=1e-12)

    naive_bps = ar / float(to.mean()) * 1e4
    correct_bps = ba["value_bps"]
    assert not math.isclose(correct_bps, naive_bps, rel_tol=1e-6)  # the denominators genuinely differ


def test_break_even_mean_edge_cases() -> None:
    fr = factor_frame(T=120, seed=3)
    gross = candidate(fr, a0=0.002, betas={}, noise=0.0, seed=4)
    idx = gross.index
    # zero turnover
    v, status, code = break_even_mean(gross, pd.Series(np.zeros(120), index=idx))
    assert status is BreakEvenStatus.UNDEFINED and code is RefusalCode.ZERO_TURNOVER
    # non-positive gross (mean <= 0) but positive turnover
    neg = pd.Series(np.full(120, -0.001), index=idx)
    v, status, code = break_even_mean(neg, pd.Series(np.full(120, 0.2), index=idx))
    assert status is BreakEvenStatus.UNDEFINED and code is RefusalCode.NON_POSITIVE_GROSS


def test_break_even_alpha_non_positive_turnover_intercept() -> None:
    fr = factor_frame(T=200, seed=5)
    gross = candidate(fr, a0=0.002, betas={"mktb": 0.3}, noise=0.0003, seed=6)
    # turnover with a clearly NEGATIVE control-adjusted intercept.
    to = pd.Series(-0.05 + 0.3 * fr["mktb"].to_numpy(), index=fr["date"])
    ba = break_even_alpha(gross, to, fr, nw_lags=3)
    assert ba["status"] is BreakEvenStatus.UNDEFINED
    assert ba["refusal"] is RefusalCode.NON_POSITIVE_TURNOVER_INTERCEPT


# ---------------------------------------------------------------------------
# scenario registry: usable computed, blocked scenario never emitted
# ---------------------------------------------------------------------------

def test_only_usable_scenario_emitted_and_full_result() -> None:
    fr = factor_frame(T=200, seed=7)
    gross = candidate(fr, a0=0.003, betas={"mktb": 0.4}, noise=0.0004, seed=8)
    to = turnover_series(fr, seed=9)
    cr = cost_result_from_turnover(
        gross, to, fr,
        weighting_scheme=WeightingScheme.EQUAL,
        turnover_method=TurnoverMethod.TARGET_TO_TARGET_PROXY,
        turnover_series_id="synthetic",
        nw_lags=3,
    )
    assert cr.estimable
    # project_registered is blocked on citation (P5) -> not emitted; only kpp_comparable.
    ids = [s.scenario_id for s in cr.scenarios]
    assert ids == ["kpp_comparable"]
    assert cr.scenarios[0].unit.value == "one_way"
    assert cr.turnover_mean is not None and cr.turnover_median is not None


def test_disjoint_sample_refuses_not_estimable_with_nans() -> None:
    # MINOR regression: no common months -> a typed refusal, never estimable-with-NaN.
    fr = factor_frame(T=120, seed=20)
    gross = candidate(fr, seed=21)
    to = pd.Series(np.full(6, 0.2), index=pd.date_range("2050-01-31", periods=6, freq="ME"))
    cr = cost_result_from_turnover(
        gross, to, fr,
        weighting_scheme=WeightingScheme.PAR,
        turnover_method=TurnoverMethod.TARGET_TO_TARGET_PROXY,
        turnover_series_id="disjoint", nw_lags=3,
    )
    assert not cr.estimable and cr.refusal_code is RefusalCode.NO_COMMON_SAMPLE
    assert cr.scenarios == () and cr.turnover_mean is None
    import json
    json.dumps(cr.to_dict(), allow_nan=False)  # strict-JSON safe


def test_break_even_grid_fallback() -> None:
    fr = factor_frame(T=120, seed=10)
    gross = candidate(fr, a0=0.002, betas={}, noise=0.0, seed=11)
    grid = (0.1, 0.2, 0.5)
    out = break_even_grid(gross, grid)
    assert [g for g, _ in out] == list(grid)
    # break-even cost = mean(gross)/g (bps), monotone decreasing in g.
    vals = [v for _, v in out]
    assert vals[0] > vals[1] > vals[2] > 0
