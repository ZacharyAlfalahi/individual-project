"""Unit tests for the negative-control specificity gate (§10.3, D20 §4.5 remediation).

The `separated` mode makes the negative control FALSIFIABLE: it PASSES iff its whole
corrected-vs-uncorrected bootstrap interval lies strictly within ±vartheta. These tests
pin the pass/fail boundary and the guards, so the "cannot fail" defect stays fixed.
"""

import pytest

from agents.auditor.validation.hypothesis_registry import (
    FactorHypothesis,
    HypothesisRegistryError,
    NegativeControlVerdict,
    evaluate_negative_control,
)

VARTHETA = 0.001  # auditor.practical_significance.vartheta (the pre-registered floor)


def _control(mode: str = "separated") -> FactorHypothesis:
    return FactorHypothesis(
        factor_id="traded_liquidity",
        dominant_bias="none",
        expected_sign=0,
        magnitude_mode=mode,
        expected_magnitude_range=None,
        is_locked=True,
        status="negative_control",
        source="s",
        registered_commit="abc",
    )


def test_pass_interval_within_band():
    v = evaluate_negative_control(_control(), -0.0004, 0.0006, VARTHETA)
    assert isinstance(v, NegativeControlVerdict)
    assert v.passed is True
    assert v.absolute_gap == pytest.approx(0.0006)


def test_fail_high_side():
    # The interval reaches above +vartheta — the control moved as much as a real bias.
    v = evaluate_negative_control(_control(), 0.0003, 0.0015, VARTHETA)
    assert v.passed is False
    assert v.absolute_gap == pytest.approx(0.0015)


def test_fail_low_side():
    v = evaluate_negative_control(_control(), -0.0020, -0.0002, VARTHETA)
    assert v.passed is False
    assert v.absolute_gap == pytest.approx(0.0020)


def test_boundary_touch_fails_strict():
    # ci_high exactly == vartheta must FAIL (strict inequality — conservative).
    v = evaluate_negative_control(_control(), -0.0001, VARTHETA, VARTHETA)
    assert v.passed is False


def test_boundary_touch_low_side_fails_strict():
    # ci_low exactly == -vartheta must FAIL too (strict, symmetric with the high side).
    v = evaluate_negative_control(_control(), -VARTHETA, 0.0001, VARTHETA)
    assert v.passed is False


def test_rejects_nan_vartheta():
    # The guard is written `not vartheta > 0.0` specifically so NaN rejects rather than
    # slipping through — pin it so a future simplification to `vartheta <= 0` can't regress.
    with pytest.raises(HypothesisRegistryError, match="positive"):
        evaluate_negative_control(_control(), -0.0001, 0.0001, float("nan"))


def test_rejects_nan_interval_bound():
    # The ordering guard `not ci_low <= ci_high` also rejects NaN in either bound.
    with pytest.raises(HypothesisRegistryError, match="ci_low"):
        evaluate_negative_control(_control(), -0.0001, float("nan"), VARTHETA)


def test_wide_interval_about_zero_fails():
    # Near-zero point but a wide interval poking past ±vartheta FAILS — intended:
    # specificity not established with precision is not certified.
    v = evaluate_negative_control(_control(), -0.0030, 0.0031, VARTHETA)
    assert v.passed is False
    assert v.absolute_gap == pytest.approx(0.0031)


def test_absolute_gap_takes_max_magnitude():
    v = evaluate_negative_control(_control(), -0.0008, 0.0002, VARTHETA)
    assert v.absolute_gap == pytest.approx(0.0008)


def test_rejects_non_separated_mode():
    with pytest.raises(HypothesisRegistryError, match="separated"):
        evaluate_negative_control(_control(mode="near_zero"), -0.0001, 0.0001, VARTHETA)


def test_rejects_nonpositive_vartheta():
    with pytest.raises(HypothesisRegistryError, match="positive"):
        evaluate_negative_control(_control(), -0.0001, 0.0001, 0.0)


def test_rejects_inverted_interval():
    with pytest.raises(HypothesisRegistryError, match="ci_low"):
        evaluate_negative_control(_control(), 0.0005, -0.0005, VARTHETA)
