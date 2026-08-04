"""
test_shared_eval_integration.py — orchestrator.evaluate composes the three diagnostics
into one SharedEvaluationResult, with blocked sub-results carrying REAL typed refusals
(the end-to-end typed flow works now, before any Auditor/Scientist wiring). JSON round-
trips; config_hash is deterministic; stage/scope are stamped.
"""

from __future__ import annotations

import json

from shared.evaluation.contracts import EvaluationScope, RefusalCode
from shared.evaluation.orchestrator import CODE_VERSION, evaluate
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
            CostScenarioSpec("project_registered", 40.0, 80.0, CostUnit.ROUND_TRIP, "UNRESOLVED_pending_citation", False),
        ),
        break_even_alpha_denominator="control_adjusted_turnover_intercept",
        assumed_turnover_grid=(0.1, 0.2),
        gross_exposure_convention=2,
    )


def _regimes_cfg() -> RegimesConfig:
    return RegimesConfig(0.935, "test#median", "baa_aaa_v1", 60)


def _evaluate(**kw):
    fr = factor_frame(T=200, seed=1)
    y = candidate(fr, a0=0.002, betas={"mktb": 0.5}, noise=0.0005, seed=2)
    return evaluate(
        y,
        stage="G4",
        candidate_id="ext_1",
        crowding_config=crowding_config(),
        crowding_factors=fr,
        costs_config=_costs_cfg(),
        regimes_config=_regimes_cfg(),
        **kw,
    ), fr, y


def test_full_compose_with_all_inputs() -> None:
    fr = factor_frame(T=200, seed=1)
    y = candidate(fr, a0=0.002, betas={"mktb": 0.5}, noise=0.0005, seed=2)
    parent = candidate(fr, a0=0.001, betas={"mktb": 0.5}, noise=0.0005, seed=3)
    spread = spread_series(T=200)
    res = evaluate(
        y, stage="G4", candidate_id="ext_1", parent_id="mktb_parent",
        parent_returns=parent, spread=spread, predicted_sign=1,
        crowding_config=crowding_config(), crowding_factors=fr,
        costs_config=_costs_cfg(), regimes_config=_regimes_cfg(),
    )
    assert res.spanning.estimable
    assert not res.costs.estimable and res.costs.refusal_code is RefusalCode.ARTEFACT_CAPABILITY_MISSING
    assert res.regimes.applicable and res.regimes.prediction.evaluable
    assert res.alignment is not None and res.alignment.common_n_months == 200
    assert res.stage == "G4" and res.code_version == CODE_VERSION
    json.dumps(res.to_dict(), allow_nan=False)  # whole aggregate is strict-JSON safe


def test_missing_optional_inputs_yield_typed_not_crash() -> None:
    # no parent, no spread, no run artefact -> alignment None, regime not-applicable,
    # cost refusal — never a crash.
    res, _, _ = _evaluate()
    assert res.alignment is None
    assert not res.regimes.applicable
    assert res.regimes.prediction.refusal_code is RefusalCode.PREDICTION_NOT_EVALUABLE
    assert not res.costs.estimable
    json.dumps(res.to_dict(), allow_nan=False)


def test_aggregate_strict_json_safe_with_degenerate_spanning() -> None:
    # MAJOR-1 at the aggregate boundary: a rank-deficient spanning block (NaN VIF, huge/inf
    # condition number) must not make the SharedEvaluationResult JSON-unsafe.
    fr = factor_frame(T=120, seed=50)
    fr["mom6"] = fr["str"]  # singular control design
    y = candidate(fr, seed=51)
    res = evaluate(
        y, stage="auditor", candidate_id="ext_deg",
        crowding_config=crowding_config(min_obs=10), crowding_factors=fr,
        costs_config=_costs_cfg(), regimes_config=_regimes_cfg(),
    )
    assert not res.spanning.estimable and res.spanning.refusal_code is RefusalCode.RANK_DEFICIENT
    json.dumps(res.to_dict(), allow_nan=False)


def test_config_hash_deterministic_and_scope_stamped() -> None:
    res_a, _, _ = _evaluate(scope=EvaluationScope.SUPPLEMENTARY)
    res_b, _, _ = _evaluate(scope=EvaluationScope.SUPPLEMENTARY)
    res_c, _, _ = _evaluate(scope=EvaluationScope.CONFIRMATORY)
    assert res_a.config_hash == res_b.config_hash          # deterministic
    assert res_a.config_hash != res_c.config_hash          # scope enters the hash
    assert res_a.spanning.scope is EvaluationScope.SUPPLEMENTARY
    assert res_c.spanning.scope is EvaluationScope.CONFIRMATORY
