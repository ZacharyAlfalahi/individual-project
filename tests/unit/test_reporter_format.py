"""Format-only conversions and fmt() (docs/reporter/reporter_spec_v0.2.md §6)."""

from __future__ import annotations

import pytest

from agents.reporter.format import fmt, scale_for_unit, to_bps, to_percent
from shared.reporting.canonical import NanValue
from shared.reporting.claims import Unit


def test_decimal_rounding():
    assert fmt(0.0123456, precision=4) == "0.0123"


def test_percent_scale():
    assert fmt(0.0123, precision=2, scale=100.0) == "1.23"


def test_bps_scale():
    assert fmt(0.0005, precision=1, scale=10000.0) == "5.0"


def test_negative_zero_normalises():
    assert fmt(-0.0) == "0.0"
    assert fmt(-0.00001, precision=2) == "0.00"


def test_nan_renders_as_token():
    assert fmt(NanValue()) == "nan"
    assert fmt(float("nan")) == "nan"


def test_bool_rejected():
    with pytest.raises(TypeError):
        fmt(True, precision=2)


def test_string_returned_verbatim():
    assert fmt("equal_average") == "equal_average"


def test_shortest_roundtrip_when_no_precision():
    assert fmt(0.5) == "0.5"
    assert fmt(3) == "3"


def test_unit_scales():
    assert scale_for_unit(Unit.PERCENT) == 100.0
    assert scale_for_unit(Unit.BPS) == 10000.0
    assert scale_for_unit(Unit.DECIMAL) == 1.0


def test_conversion_helpers():
    assert to_percent(0.0123) == pytest.approx(1.23)
    assert to_bps(0.0005) == pytest.approx(5.0)
