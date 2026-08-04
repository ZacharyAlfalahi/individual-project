"""
test_shared_eval_contracts.py — the typed contracts hold their invariants: no verdict
boolean (D-E7), cost_adjusted_* naming never net_* (D-E10), non-empty citation enforced
(D-E12), enums + to_dict JSON-serialisable.
"""

from __future__ import annotations

import json

import pytest

from shared.evaluation.contracts import (
    CostResult,
    CostScenarioResult,
    CostUnit,
    RefusalCode,
    SpanningResult,
)


def test_spanning_result_has_no_verdict_boolean() -> None:
    # D-E7: the unlicensed label must not exist as a field.
    fields = set(SpanningResult.__dataclass_fields__)
    for forbidden in ("is_crowded", "is_spanned", "passes", "spanned", "crowded"):
        assert forbidden not in fields


def test_no_net_field_anywhere_costs() -> None:
    # D-E10: never `net_*`; the cost-adjusted naming must be used instead.
    for cls in (CostResult, CostScenarioResult):
        for f in cls.__dataclass_fields__:
            assert not f.startswith("net_"), (cls.__name__, f)
    assert "cost_adjusted_mean_monthly" in CostScenarioResult.__dataclass_fields__


def test_cost_scenario_rejects_empty_citation() -> None:
    # D-E12: a scenario must carry a non-empty source citation.
    with pytest.raises(ValueError, match="source_citation"):
        CostScenarioResult(
            scenario_id="x",
            cost_bps_ig=19.0,
            cost_bps_hy=19.0,
            unit=CostUnit.ONE_WAY,
            source_citation="   ",
            cost_adjusted_mean_monthly=0.0,
            cost_adjusted_sharpe=None,
            cost_drag_bps_monthly=0.0,
        )


def test_cost_scenario_accepts_real_citation_and_serialises() -> None:
    s = CostScenarioResult(
        scenario_id="kpp_comparable",
        cost_bps_ig=19.0,
        cost_bps_hy=19.0,
        unit=CostUnit.ONE_WAY,
        source_citation="KPP (2023)",
        cost_adjusted_mean_monthly=0.001,
        cost_adjusted_sharpe=0.5,
        cost_drag_bps_monthly=3.8,
    )
    d = s.to_dict()
    assert d["unit"] == "one_way"  # enum serialised to value
    json.dumps(d)


def test_refusal_codes_are_string_enums() -> None:
    assert RefusalCode.ARTEFACT_CAPABILITY_MISSING.value == "artefact_capability_missing"
    assert isinstance(RefusalCode.RANK_DEFICIENT.value, str)
