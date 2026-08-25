"""Unit tests for the §4.7 arm-2 known-error positive control (BBW lead/lag).

Deterministic: a synthetic ``run_leadlag_gate.py`` report is graded against a self-contained
thresholds fixture, so the test depends on neither the real thresholds file nor a built panel.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from agents.auditor.validation.known_error_control import (  # noqa: E402
    evaluate_known_error_control,
    load_lead_lag_criterion,
)

# collapse_max 0.7 / restore_min 0.90 — the pre-registered band (validation.gate_thresholds.lead_lag).
_THRESHOLDS = """
validation:
  gate_thresholds:
    lead_lag:
      collapse_max: 0.7
      restore_min: 0.90
"""


@pytest.fixture()
def thresholds_path(tmp_path) -> Path:
    p = tmp_path / "thresholds.yaml"
    p.write_text(_THRESHOLDS)
    return p


def _arm(defective: float, restored: float) -> dict:
    return {"corr_correct_vs_defective": defective, "corr_correct_vs_restored": restored}


def _report(drf_arm: dict, **others: dict) -> dict:
    return {"arms": {"drf": drf_arm, **others}}


def test_load_criterion_reads_registered_band(thresholds_path):
    c = load_lead_lag_criterion(thresholds_path)
    assert c.collapse_max == 0.7 and c.restore_min == 0.90


def test_load_criterion_fail_loud_when_absent(tmp_path):
    p = tmp_path / "empty.yaml"
    p.write_text("validation: {}\n")
    with pytest.raises(RuntimeError, match="unregistered"):
        load_lead_lag_criterion(p)


def test_pass_when_defect_reproduced_and_corrected(thresholds_path):
    # drf collapses (0.26 < 0.7) and the round-trip restores (1.0 >= 0.90): the control fires.
    v = evaluate_known_error_control(
        _report(_arm(0.26, 1.0), crf=_arm(0.44, 0.99), lrf=_arm(0.30, 0.95)),
        thresholds_path=thresholds_path,
    )
    assert v.passed is True
    assert v.primary.collapsed and v.primary.restored
    assert len(v.corroborators) == 2 and all(c.passed for c in v.corroborators)
    assert v.to_dict()["defect"] == "lead_lag_look_ahead"


def test_fail_when_no_collapse(thresholds_path):
    # The as-published arm does NOT decorrelate (0.85 >= 0.7): the documented defect was not reproduced.
    v = evaluate_known_error_control(_report(_arm(0.85, 1.0)), thresholds_path=thresholds_path)
    assert v.passed is False and v.primary.collapsed is False and v.primary.restored is True


def test_fail_when_not_restored(thresholds_path):
    # The correction fails to restore the factor (0.50 < 0.90): reproduced but not fixed.
    v = evaluate_known_error_control(_report(_arm(0.26, 0.50)), thresholds_path=thresholds_path)
    assert v.passed is False and v.primary.collapsed is True and v.primary.restored is False


def test_primary_gates_even_if_corroborators_pass(thresholds_path):
    # crf/lrf corroborate but do NOT rescue a failed drf primary — the control is drf-gated.
    v = evaluate_known_error_control(
        _report(_arm(0.85, 1.0), crf=_arm(0.44, 1.0), lrf=_arm(0.30, 1.0)),
        thresholds_path=thresholds_path,
    )
    assert v.passed is False and all(c.passed for c in v.corroborators)


def test_missing_drf_arm_raises(thresholds_path):
    with pytest.raises(ValueError, match="missing the 'drf' arm"):
        evaluate_known_error_control({"arms": {"crf": _arm(0.44, 1.0)}}, thresholds_path=thresholds_path)
