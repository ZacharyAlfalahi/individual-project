"""Word-selection total functions (INV-6): keyed on sign/enum, explicit failure branch."""

from __future__ import annotations

import pytest

from agents.reporter.words import (
    audit_scope_word,
    effect_direction_word,
    metric_direction_word,
    sign_word,
    verdict_word,
)
from shared.reporting.canonical import NanValue


def test_metric_direction():
    assert metric_direction_word(0.5) == "higher"
    assert metric_direction_word(-0.5) == "lower"
    assert metric_direction_word(0.0) == "unchanged"


def test_effect_direction():
    assert effect_direction_word(0.5) == "raises"
    assert effect_direction_word(-0.5) == "lowers"
    assert effect_direction_word(0.0) == "leaves unchanged"


def test_sign_word():
    assert sign_word(1) == "positive"
    assert sign_word(0) == "zero"
    assert sign_word(-2) == "negative"


def test_nan_has_no_direction():
    with pytest.raises(ValueError):
        metric_direction_word(float("nan"))
    with pytest.raises(ValueError):
        effect_direction_word(NanValue())


def test_audit_scope_words():
    assert audit_scope_word("COMPLETE") == "complete"
    assert audit_scope_word("PARTIAL") == "partial"
    assert audit_scope_word("REFUSED") == "refused"
    with pytest.raises(ValueError):
        audit_scope_word("SOMETHING")


def test_verdict_words():
    assert verdict_word("PASS") == "passed"
    assert verdict_word("FAIL") == "failed"
    with pytest.raises(ValueError):
        verdict_word("MAYBE")
