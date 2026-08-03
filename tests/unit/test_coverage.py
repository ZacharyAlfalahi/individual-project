"""Layered RQ2 coverage (evaluation contract §5.2) — agents/quant/config/coverage.py."""

import pytest

from agents.quant.config.coverage import (
    LAYERS,
    REFUSAL_LAYER,
    layer_of,
    layered_coverage,
)
from agents.quant.config.refusal import RefusalCode


def test_every_refusal_code_has_a_layer():
    """Completeness: all eight typed codes map to exactly one §5.2 layer (the module
    also asserts this at import — this pins it as a test, so an additive RefusalCode
    member cannot silently escape the coverage classifier)."""
    assert set(REFUSAL_LAYER) == set(RefusalCode)
    assert set(REFUSAL_LAYER.values()) <= set(LAYERS)


def test_layer_of_accepts_code_or_string_and_rejects_unknown():
    assert layer_of(RefusalCode.MISSING_BINDING) == "binding"
    assert layer_of("REVIEW_REQUIRED") == "semantic"
    assert layer_of("OUT_OF_ENUM_WEIGHTING") == "execution"
    with pytest.raises(ValueError):
        layer_of("NOT_A_CODE")


def test_anchor_set_all_supported_is_unit_coverage():
    """𝒞 = 3 supported anchors ⇒ ℰ = ℬ = 𝒮 = 𝒞 ⇒ every C_* = 1.0, no refusals."""
    cov = layered_coverage([{"refusal_codes": []} for _ in range(3)])
    assert cov["n_candidates"] == cov["n_executed"] == 3
    assert cov["C_semantic"] == cov["C_binding"] == cov["C_execution"] == cov["C_end_to_end"] == 1.0
    assert cov["refusals_by_layer"] == {"semantic": 0, "binding": 0, "execution": 0}


def test_nested_fractions_and_product_identity():
    """One failure per layer + two executed (n=5): fractions nest and
    C_end_to_end == C_semantic * C_binding * C_execution."""
    cov = layered_coverage([
        {"refusal_codes": ["REVIEW_REQUIRED"]},        # semantic
        {"refusal_codes": ["MISSING_BINDING"]},        # binding
        {"refusal_codes": ["OUT_OF_ENUM_WEIGHTING"]},  # execution
        {"refusal_codes": []},
        {"refusal_codes": []},
    ])
    assert (cov["n_candidates"], cov["n_semantic"], cov["n_binding"], cov["n_executed"]) == (5, 4, 3, 2)
    assert cov["C_semantic"] == 4 / 5
    assert cov["C_binding"] == 3 / 4
    assert cov["C_execution"] == 2 / 3
    assert cov["C_end_to_end"] == pytest.approx(cov["C_semantic"] * cov["C_binding"] * cov["C_execution"])
    assert cov["refusals_by_layer"] == {"semantic": 1, "binding": 1, "execution": 1}


def test_strategy_fails_at_earliest_layer():
    """A strategy with codes at several layers fails at the EARLIEST (semantic < binding
    < execution) — it is not double-counted, and it drops out of 𝒮."""
    cov = layered_coverage([{"refusal_codes": ["OUT_OF_ENUM_WEIGHTING", "REVIEW_REQUIRED"]}])
    assert cov["refusals_by_layer"] == {"semantic": 1, "binding": 0, "execution": 0}
    assert cov["n_semantic"] == 0


def test_empty_denominator_is_none_not_fabricated_one():
    """If 𝒮 is empty, C_binding is undefined — returned as None, never a fake 1.0."""
    cov = layered_coverage([{"refusal_codes": ["REVIEW_REQUIRED"]}])
    assert cov["C_semantic"] == 0.0
    assert cov["C_binding"] is None
    assert cov["C_execution"] is None
