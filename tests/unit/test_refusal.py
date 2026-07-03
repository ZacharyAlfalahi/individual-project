"""
Unit tests for the refusal-vs-raise taxonomy: a legit-but-unrepresentable
strategy returns a ConfigRefusal (recorded for RQ2); malformed agent output
raises.
"""

import pytest

from agents.quant.config import (
    Binding,
    ConfigRefusal,
    Evidence,
    Inherited,
    RefusalCode,
    build_quant_config,
)


def _score():
    return Binding("var_5pct", "BOUND", Evidence(column="var_5pct", note="5% VaR"))


# --- each refusal code ------------------------------------------------------

def test_missing_score_binding_refuses():
    r = build_quant_config(
        "s1",
        Binding(None, "MISSING", Evidence(note="no panel column for the ESG-tilt signal")),
    )
    assert isinstance(r, ConfigRefusal)
    assert r.code is RefusalCode.MISSING_BINDING
    assert r.field == "score"
    assert r.strategy_id == "s1"


def test_missing_control_binding_refuses():
    r = build_quant_config(
        "s2",
        _score(),
        control=Binding(None, "MISSING", Evidence(note="no column for the described control axis")),
    )
    assert isinstance(r, ConfigRefusal)
    assert r.code is RefusalCode.MISSING_BINDING
    assert r.field == "control"


def test_out_of_enum_weighting_refuses():
    r = build_quant_config(
        "s3",
        _score(),
        weighting=Inherited("market_value", "STATED", Evidence(quote="value-weighted by market value")),
    )
    assert isinstance(r, ConfigRefusal)
    assert r.code is RefusalCode.OUT_OF_ENUM_WEIGHTING


def test_unsupported_trim_variant_refuses():
    r = build_quant_config(
        "s4",
        _score(),
        trim=Inherited(
            {"method": "truncate", "bounds": {"type": "percentile", "lo": 0.01, "hi": 0.99}},
            "STATED", Evidence(quote="winsorised at 1/99 percentiles"),
        ),
    )
    assert isinstance(r, ConfigRefusal)
    assert r.code is RefusalCode.UNSUPPORTED_TRIM_VARIANT


def test_control_plus_multimonth_hold_refuses():
    r = build_quant_config(
        "s5",
        _score(),
        control=Binding("rating", "BOUND", Evidence(column="rating", note="rating axis")),
        holding_period=Inherited(6, "STATED", Evidence(quote="held six months")),
    )
    assert isinstance(r, ConfigRefusal)
    assert r.code is RefusalCode.UNSUPPORTED_COMBINATION
    assert r.field == "holding_period"


# --- ConfigRefusal.to_dict is serialisation-safe (F8) ----------------------

def test_refusal_to_dict_roundtrip_with_tuple_candidates():
    r = ConfigRefusal(
        "s6", RefusalCode.MISSING_BINDING, "control", "ambiguous+missing",
        Evidence(candidates=("a", "b"), chosen="a", note="searched"),
    )
    d = r.to_dict()
    assert d["code"] == "MISSING_BINDING"
    assert d["evidence"]["candidates"] == ["a", "b"]  # tuple -> list


# --- malformed-supported trim raises (NOT a refusal) -----------------------

def test_supported_trim_without_bounds_raises():
    with pytest.raises(ValueError):
        build_quant_config(
            "s7",
            _score(),
            trim=Inherited(
                {"method": "truncate", "bounds": {"type": "absolute"}},  # no lo/hi
                "STATED", Evidence(quote="truncate"),
            ),
        )


# --- T10: remaining trim-triage refusal branches ---------------------------

def test_unsupported_trim_target_refuses():
    r = build_quant_config(
        "s8", _score(),
        trim=Inherited(
            {"method": "truncate", "target": "xret", "bounds": {"type": "absolute", "lo": -0.1}},
            "STATED", Evidence(quote="trim excess return"),
        ),
    )
    assert isinstance(r, ConfigRefusal)
    assert r.code is RefusalCode.UNSUPPORTED_TRIM_VARIANT


def test_unsupported_trim_sample_refuses():
    r = build_quant_config(
        "s9", _score(),
        trim=Inherited(
            {"method": "truncate", "bounds": {"type": "absolute", "lo": -0.1},
             "sample": "by_month_cross_section"},
            "STATED", Evidence(quote="per-month trim"),
        ),
    )
    assert isinstance(r, ConfigRefusal)
    assert r.code is RefusalCode.UNSUPPORTED_TRIM_VARIANT


def test_trim_bounds_not_mapping_refuses():
    r = build_quant_config(
        "s10", _score(),
        trim=Inherited(
            {"method": "truncate", "bounds": "nope"},
            "STATED", Evidence(quote="malformed bounds"),
        ),
    )
    assert isinstance(r, ConfigRefusal)
    assert r.code is RefusalCode.UNSUPPORTED_TRIM_VARIANT


# --- T10: control x holding>1 also refuses for an AMBIGUOUS-resolved control -

def test_ambiguous_control_plus_multimonth_hold_refuses():
    r = build_quant_config(
        "s11", _score(),
        control=Binding(
            "rating", "AMBIGUOUS",
            Evidence(candidates=("rating", "rating_numeric"), chosen="rating", note="chose rating"),
        ),
        holding_period=Inherited(6, "STATED", Evidence(quote="held six months")),
    )
    assert isinstance(r, ConfigRefusal)
    assert r.code is RefusalCode.UNSUPPORTED_COMBINATION
