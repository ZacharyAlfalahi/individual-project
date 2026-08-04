"""Canonical-JSON contract (docs/reporter/reporter_spec_v0.2.md §10.2): determinism, NaN sentinel, -0.0, NFC."""

from __future__ import annotations

import math

import pytest

from shared.reporting.canonical import (
    CanonicalisationError,
    NanValue,
    canonical_hash,
    canonical_json,
)


def test_key_order_is_lexicographic_and_insertion_independent():
    a = {"b": 1, "a": 2, "c": 3}
    b = {"c": 3, "a": 2, "b": 1}
    assert canonical_json(a) == canonical_json(b) == '{"a":2,"b":1,"c":3}'


def test_separators_are_tight():
    assert canonical_json({"x": [1, 2]}) == '{"x":[1,2]}'


def test_float_nan_encodes_as_sentinel():
    assert canonical_json(float("nan")) == '{"__nan__":true,"reason":null}'


def test_nanvalue_carries_its_reason():
    out = canonical_json(NanValue(reason="zero variance"))
    assert out == '{"__nan__":true,"reason":"zero variance"}'


def test_infinity_is_rejected():
    with pytest.raises(CanonicalisationError):
        canonical_json(math.inf)
    with pytest.raises(CanonicalisationError):
        canonical_json(-math.inf)


def test_negative_zero_normalises_to_zero():
    assert canonical_json(-0.0) == canonical_json(0.0) == "0.0"
    assert canonical_json({"x": -0.0}) == '{"x":0.0}'


def test_strings_are_nfc_normalised():
    decomposed = "café"  # e + combining acute
    composed = "café"  # é
    assert decomposed != composed
    assert canonical_json({"k": decomposed}) == canonical_json({"k": composed})


def test_non_string_key_is_rejected():
    with pytest.raises(CanonicalisationError):
        canonical_json({1: "a"})


def test_bool_stays_bool_not_int():
    assert canonical_json({"t": True, "f": False}) == '{"f":false,"t":true}'


def test_hash_is_stable_across_key_order():
    assert canonical_hash({"a": 1, "b": 2}) == canonical_hash({"b": 2, "a": 1})


def test_unsupported_type_raises():
    with pytest.raises(CanonicalisationError):
        canonical_json({"x": object()})
