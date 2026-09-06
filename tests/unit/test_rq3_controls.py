"""RQ3 controls exporter — positive (known-error) + negative (specificity) controls. Offline."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import run_rq3_controls as RC  # noqa: E402


def _gate_report(defective=0.25, restored=1.0):
    arm = {"corr_correct_vs_defective": defective, "corr_correct_vs_restored": restored}
    return {"run_timestamp": "t", "git_commit": "g",
            "arms": {"drf": dict(arm), "crf": dict(arm), "lrf": dict(arm)}}


# --- positive (known-error) control -------------------------------------------------

def test_positive_control_passes_on_reproduced_and_fixed_defect():
    out = RC.positive_control(_gate_report())
    assert out["passed"] is True and out["basis"] == "real_dev_data"
    assert out["primary"]["collapsed"] and out["primary"]["restored"]


def test_positive_control_fails_when_defect_not_reproduced():
    # as-published corr does NOT collapse (0.95 >= collapse_max 0.7) -> control fails
    out = RC.positive_control(_gate_report(defective=0.95))
    assert out["passed"] is False


def test_positive_control_missing_drf_arm_fails_loud():
    with pytest.raises(RC.RQ3ControlsError, match="drf"):
        RC.positive_control({"arms": {"crf": {"corr_correct_vs_defective": 0.2,
                                              "corr_correct_vs_restored": 1.0}}})


# --- negative (specificity) control -------------------------------------------------
# The real dev-data negative-control computation (~13s) is validated in
# test_negative_control_run.py; here we inject a fixture so the CLI/assembly tests stay fast.

def _neg_fixture(**_kw) -> dict:
    return {
        "control": "negative_control_specificity", "factor": "traded_liquidity",
        "basis": "real_dev_data", "status": "fired",
        "ci_low": -0.0001, "ci_high": 0.0004, "vartheta": 0.001, "passed": True,
        "absolute_gap": 0.0004, "point_gap_mean": 0.00013,
        "bootstrap": {"n_months_common": 233, "n_replicates": 1000, "block_length_months": 6,
                      "effective_blocks": 38, "seed": 0},
        "estimand": {"run_once": True}, "gate": {"mode": "separated", "vartheta": 0.001},
    }


def test_vartheta_loader_fails_loud_on_missing_block(tmp_path):
    import yaml
    bad = tmp_path / "th.yaml"
    bad.write_text(yaml.safe_dump({"auditor": {}}))
    with pytest.raises(RC.RQ3ControlsError):
        RC.load_vartheta(bad)


def test_build_controls_assembles_both_with_injected_negative():
    r = RC.build_controls(_gate_report(), negative_control_fn=_neg_fixture)
    assert r["positive_control"]["passed"] is True
    assert r["negative_control"]["basis"] == "real_dev_data"
    assert r["negative_control"]["status"] == "fired"
    assert isinstance(r["negative_control"]["passed"], bool)


def test_main_missing_leadlag_fails_loud(tmp_path):
    with pytest.raises(RC.RQ3ControlsError, match="missing"):
        RC.main(["--leadlag", str(tmp_path / "nope.json"), "--out", str(tmp_path / "o.json")])


def test_main_writes_zero_spend_result(tmp_path, monkeypatch):
    import json
    monkeypatch.setattr(RC, "negative_control", _neg_fixture)   # skip the ~13s dev run
    lead = tmp_path / "leadlag.json"
    lead.write_text(json.dumps(_gate_report()))
    out = tmp_path / "controls.json"
    assert RC.main(["--leadlag", str(lead), "--out", str(out)]) == 0
    r = json.loads(out.read_text())
    assert r["provenance"]["dev_only"] is True and r["provenance"]["model_calls"] == 0
    assert r["negative_control"]["basis"] == "real_dev_data"
