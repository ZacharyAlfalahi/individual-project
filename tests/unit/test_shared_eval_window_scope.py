"""
test_shared_eval_window_scope.py — amendment A7: the 60-month floors are scoped to
development-window diagnostics by claim type.

HOLDOUT window: the spanning/crowding HAC regression is refused with a typed
DEVELOPMENT_SCOPE_DIAGNOSTIC (never computed — its floor cannot be met on the 48-month
holdout by construction of the walk-forward split, SC-SCI-10), while the regime mean
decomposition and sign contrast (deliberately mean/sign statistics, D-E16) remain
computable and carry `short_sample` when a state is below the conditional floor.
DEVELOPMENT window: byte-identical to pre-A7 behaviour, including the config_hash.
"""

from __future__ import annotations

import inspect
import json

from shared.evaluation import crowding as crowding_mod
from shared.evaluation.contracts import RefusalCode, SampleWindow
from shared.evaluation.orchestrator import evaluate
from shared.evaluation.regimes import evaluate_regimes, not_applicable_result
from shared.evaluation.reporting_rules import (
    FORBIDDEN_STRINGS,
    regime_sentence,
    spanning_sentence,
)
from shared.evaluation.spanning import spanning_regression
from shared.evaluation.thresholds import (
    CostsConfig,
    CostScenarioSpec,
    CostUnit,
    RegimesConfig,
)

from _shared_eval_fixtures import candidate, crowding_config, factor_frame, spread_series  # noqa: E402


def _costs_cfg() -> CostsConfig:
    return CostsConfig(
        scenarios=(
            CostScenarioSpec("kpp_comparable", 19.0, 19.0, CostUnit.ONE_WAY, "KPP (2023)", True),
        ),
        break_even_alpha_denominator="control_adjusted_turnover_intercept",
        assumed_turnover_grid=(0.1, 0.2),
        gross_exposure_convention=2,
    )


def _regimes_cfg() -> RegimesConfig:
    return RegimesConfig(0.935, "test#median", "baa_aaa_v1", 60)


def _assert_licensed(sentence: str) -> None:
    low = sentence.lower()
    for bad in FORBIDDEN_STRINGS:
        assert bad not in low, f"forbidden string {bad!r} in {sentence!r}"


# ---------------------------------------------------------------------------
# Spanning: holdout -> typed development-scope refusal, BEFORE any factor load.
# ---------------------------------------------------------------------------

def test_holdout_spanning_refuses_without_loading_factors() -> None:
    fr = factor_frame(T=48, seed=1)
    y = candidate(fr, betas={"mktb": 0.5}, seed=2)
    # bundles in the fixture config point at parquets that DO NOT exist: if the
    # implementation regressed to loading factors before the window check, this raises.
    r = spanning_regression(y, config=crowding_config(), window=SampleWindow.HOLDOUT)
    assert not r.estimable
    assert r.refusal_code is RefusalCode.DEVELOPMENT_SCOPE_DIAGNOSTIC
    assert r.window is SampleWindow.HOLDOUT
    assert r.min_obs == 60 and r.sample_id == "not_computed:holdout_window"
    assert r.alpha_monthly is None and r.t_hac is None and r.betas == tuple()
    json.dumps(r.to_dict(), allow_nan=False)
    assert r.to_dict()["window"] == "holdout"


def test_development_default_and_explicit_are_identical() -> None:
    fr = factor_frame(T=200, seed=1)
    y = candidate(fr, betas={"mktb": 0.5}, seed=2)
    r_default = spanning_regression(y, config=crowding_config(), factors=fr)
    r_dev = spanning_regression(
        y, config=crowding_config(), factors=fr, window=SampleWindow.DEVELOPMENT
    )
    assert r_default.estimable and r_default.window is SampleWindow.DEVELOPMENT
    assert r_default.to_dict() == r_dev.to_dict()


# ---------------------------------------------------------------------------
# Regimes: mean/sign contrast stays computable on holdout; short_sample is
# floor-driven (a state below min_obs_conditional), not a blanket window flag.
# ---------------------------------------------------------------------------

def test_holdout_regimes_short_sample_below_floor() -> None:
    fr = factor_frame(T=48, seed=1)
    y = candidate(fr, betas={"mktb": 0.5}, seed=2)
    parent = candidate(fr, a0=0.001, betas={"mktb": 0.5}, seed=3)
    spread = spread_series(T=48)
    r = evaluate_regimes(
        y, spread, parent_returns=parent, config=_regimes_cfg(), predicted_sign=1,
        window=SampleWindow.HOLDOUT,
    )
    assert r.applicable and r.prediction.evaluable       # the sign contrast IS scored
    assert r.months_high < 60 and r.months_low < 60
    assert r.short_sample and r.window is SampleWindow.HOLDOUT
    json.dumps(r.to_dict(), allow_nan=False)


def test_holdout_regimes_no_flag_when_both_states_clear_floor() -> None:
    fr = factor_frame(T=200, seed=1)
    y = candidate(fr, betas={"mktb": 0.5}, seed=2)
    parent = candidate(fr, a0=0.001, betas={"mktb": 0.5}, seed=3)
    spread = spread_series(T=200)                        # ~100 months per state
    r = evaluate_regimes(
        y, spread, parent_returns=parent, config=_regimes_cfg(), predicted_sign=1,
        window=SampleWindow.HOLDOUT,
    )
    assert r.months_high >= 60 and r.months_low >= 60
    assert not r.short_sample


def test_development_regimes_never_flag_short_sample() -> None:
    fr = factor_frame(T=48, seed=1)
    y = candidate(fr, betas={"mktb": 0.5}, seed=2)
    parent = candidate(fr, a0=0.001, betas={"mktb": 0.5}, seed=3)
    r = evaluate_regimes(
        y, spread_series(T=48), parent_returns=parent, config=_regimes_cfg(), predicted_sign=1,
    )
    assert not r.short_sample and r.window is SampleWindow.DEVELOPMENT


# ---------------------------------------------------------------------------
# Orchestrator: composition + config-hash identity preservation.
# ---------------------------------------------------------------------------

def test_orchestrator_holdout_composition_and_hash() -> None:
    fr = factor_frame(T=48, seed=1)
    y = candidate(fr, betas={"mktb": 0.5}, seed=2)
    parent = candidate(fr, a0=0.001, betas={"mktb": 0.5}, seed=3)
    kw = dict(
        stage="holdout", candidate_id="ext_1", parent_id="p", parent_returns=parent,
        spread=spread_series(T=48), predicted_sign=1,
        crowding_config=crowding_config(), crowding_factors=fr,
        costs_config=_costs_cfg(), regimes_config=_regimes_cfg(),
    )
    hold = evaluate(y, window=SampleWindow.HOLDOUT, **kw)
    dev_default = evaluate(y, **kw)
    dev_explicit = evaluate(y, window=SampleWindow.DEVELOPMENT, **kw)

    assert hold.window is SampleWindow.HOLDOUT
    assert hold.spanning.refusal_code is RefusalCode.DEVELOPMENT_SCOPE_DIAGNOSTIC
    assert hold.regimes.short_sample and hold.regimes.prediction.evaluable
    json.dumps(hold.to_dict(), allow_nan=False)

    # A7 hash rule: DEVELOPMENT (default or explicit) preserves the pre-A7 identity;
    # HOLDOUT enters the hash.
    assert dev_default.config_hash == dev_explicit.config_hash
    assert hold.config_hash != dev_default.config_hash


# ---------------------------------------------------------------------------
# Reporting: the refusal and the short-sample qualifier stay licensed.
# ---------------------------------------------------------------------------

def test_scope_refusal_sentence_licensed() -> None:
    fr = factor_frame(T=48, seed=1)
    y = candidate(fr, betas={"mktb": 0.5}, seed=2)
    r = spanning_regression(y, config=crowding_config(), window=SampleWindow.HOLDOUT)
    s = spanning_sentence(r)
    assert "development-window diagnostic" in s and "not computed on the holdout" in s
    _assert_licensed(s)


def test_short_sample_regime_sentence_point_estimate_only() -> None:
    fr = factor_frame(T=48, seed=1)
    y = candidate(fr, betas={"mktb": 0.5}, seed=2)
    parent = candidate(fr, a0=0.001, betas={"mktb": 0.5}, seed=3)
    r = evaluate_regimes(
        y, spread_series(T=48), parent_returns=parent, config=_regimes_cfg(), predicted_sign=1,
        window=SampleWindow.HOLDOUT,
    )
    s = regime_sentence(r)
    assert "short sample" in s and "no inferential claim" in s
    _assert_licensed(s)
    # the development sentence carries no qualifier
    r_dev = evaluate_regimes(
        y, spread_series(T=48), parent_returns=parent, config=_regimes_cfg(), predicted_sign=1,
    )
    assert "short sample" not in regime_sentence(r_dev)


# ---------------------------------------------------------------------------
# Tripwires: DEVELOPMENT is the default everywhere; the Layer-1 crowding lens
# (G4, development-side by construction) has no window parameter — holdout use
# is barred at the typed spanning seam, not silently permitted here.
# ---------------------------------------------------------------------------

def test_window_defaults_are_development() -> None:
    for fn in (evaluate, spanning_regression, evaluate_regimes, not_applicable_result):
        default = inspect.signature(fn).parameters["window"].default
        assert default is SampleWindow.DEVELOPMENT, fn.__name__


def test_crowding_diagnostic_has_no_window_parameter() -> None:
    params = inspect.signature(crowding_mod.crowding_diagnostic).parameters
    assert "window" not in params
