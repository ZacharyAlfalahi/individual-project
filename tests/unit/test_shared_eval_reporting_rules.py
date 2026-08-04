"""
test_shared_eval_reporting_rules.py — the deterministic sentence templates (D-E17). The
load-bearing test: NO generated sentence — across every estimable/refusal branch —
contains a forbidden overclaiming string. Also pins the licensed wording for the
CI-covers-zero vs CI-excludes-zero branches.
"""

from __future__ import annotations

from shared.evaluation.contracts import TurnoverMethod, WeightingScheme
from shared.evaluation.reporting_rules import (
    FORBIDDEN_STRINGS,
    cost_sentence,
    regime_sentence,
    spanning_sentence,
)
from shared.evaluation.costs import cost_result_from_turnover, evaluate_costs
from shared.evaluation.regimes import evaluate_regimes
from shared.evaluation.spanning import spanning_regression
from shared.evaluation.thresholds import RegimesConfig

from _shared_eval_fixtures import (  # noqa: E402
    FROZEN_MEDIAN,
    candidate,
    crowding_config,
    factor_frame,
    spread_series,
    turnover_series,
)


def _cfg() -> RegimesConfig:
    return RegimesConfig(
        evaluation_median=FROZEN_MEDIAN,
        evaluation_median_source="test#median",
        macro_data_contract_id="baa_aaa_v1",
        min_obs_conditional=60,
    )


def _all_sentences() -> list[str]:
    fr = factor_frame(T=200, seed=1)
    y = candidate(fr, a0=0.002, betas={"mktb": 0.5}, noise=0.0005, seed=2)
    to = turnover_series(fr, seed=3)
    spread = spread_series(T=200)
    parent = candidate(fr, a0=0.001, betas={"mktb": 0.5}, noise=0.0005, seed=4)
    cfg = crowding_config()

    sp_ok = spanning_regression(y, config=cfg, factors=fr)
    sp_insufficient = spanning_regression(y.iloc[:40], config=cfg, factors=fr)
    frdup = factor_frame(T=120, seed=7)
    frdup["mom6"] = frdup["str"]
    sp_rank = spanning_regression(candidate(frdup, seed=8), config=crowding_config(min_obs=10), factors=frdup)

    cost_refusal = evaluate_costs(run_artefact=None)
    cost_ok = cost_result_from_turnover(
        y, to, fr,
        weighting_scheme=WeightingScheme.EQUAL,
        turnover_method=TurnoverMethod.TARGET_TO_TARGET_PROXY,
        turnover_series_id="syn", nw_lags=3,
    )

    reg_applicable = evaluate_regimes(y, spread, parent_returns=parent, config=_cfg(), predicted_sign=1)
    reg_not_applicable = evaluate_regimes(y, spread, config=_cfg())

    return [
        spanning_sentence(sp_ok),
        spanning_sentence(sp_insufficient),
        spanning_sentence(sp_rank),
        cost_sentence(cost_refusal),
        cost_sentence(cost_ok),
        regime_sentence(reg_applicable),
        regime_sentence(reg_not_applicable),
    ]


def test_no_forbidden_string_in_any_generated_sentence() -> None:
    for s in _all_sentences():
        low = s.lower()
        for bad in FORBIDDEN_STRINGS:
            assert bad not in low, (bad, s)


def test_spanning_ci_covers_zero_says_not_established() -> None:
    # a pure-noise candidate -> alpha CI straddles zero -> "not established".
    fr = factor_frame(T=200, seed=20)
    y = candidate(fr, a0=0.0, betas={f: 0.0 for f in ("mktb", "drf", "crf", "lrf", "str", "mom6")}, noise=0.02, seed=21)
    sp = spanning_regression(y, config=crowding_config(), factors=fr)
    s = spanning_sentence(sp)
    if sp.ci_low is not None and sp.ci_low <= 0.0 <= sp.ci_high:
        assert "not established" in s
    else:
        assert "not established" not in s


def test_none_ci_says_could_not_be_established_not_affirmative() -> None:
    # MAJOR-2 regression: estimable but the HAC SE is not computable (T - k <= 1) must NOT
    # fall through to the affirmative "was X per month" sentence, and must leak no `nan`.
    fr = factor_frame(T=8, seed=40)
    y = candidate(fr, a0=0.003, betas={}, noise=0.0, seed=41)
    sp = spanning_regression(y, config=crowding_config(min_obs=7), factors=fr)
    assert sp.estimable and sp.ci_low is None and sp.t_hac is None
    s = spanning_sentence(sp)
    assert "could not be established" in s and "nan" not in s.lower()
    for bad in FORBIDDEN_STRINGS:
        assert bad not in s.lower()


def test_insufficient_obs_sentence_reads_actual_min_obs() -> None:
    # the threshold in the sentence is sourced from the result, not hard-coded.
    fr = factor_frame(T=200, seed=42)
    sp = spanning_regression(candidate(fr, seed=43).iloc[:40], config=crowding_config(min_obs=48), factors=fr)
    assert "48-month" in spanning_sentence(sp)


def test_cost_refusal_sentence_mentions_weights() -> None:
    s = cost_sentence(evaluate_costs(run_artefact=None))
    assert "weights" in s.lower() and "not computed" in s.lower()


def test_insufficient_obs_sentence_is_licensed() -> None:
    fr = factor_frame(T=200, seed=30)
    sp = spanning_regression(candidate(fr, seed=31).iloc[:40], config=crowding_config(min_obs=60), factors=fr)
    s = spanning_sentence(sp)
    assert "not estimable" in s and "60-month" in s
