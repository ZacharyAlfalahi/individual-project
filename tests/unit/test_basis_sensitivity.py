"""Basis-sensitivity exporter (RQ2 §5.2) — pure consolidation, offline, dev-only."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import run_basis_sensitivity as BS  # noqa: E402


def _tr_report():
    return {"zero_coupon_invariance": {"raw_ok": True, "corr_ok": True, "z_bond_months_checked": 337031}}


def _accrual():
    return {
        "levels_corrected": {"crf_clean_pct": -0.1373, "crf_total_pct": 0.1358,
                             "lrf_clean_pct": 0.0002, "lrf_total_pct": 0.0403,
                             "criterion": "CRF sign turns positive"},
        "differentials_static": {"mom6_ep_minus_ea_clean_pct": 0.064, "mom6_ep_minus_ea_total_pct": 0.066,
                                 "criterion": "sign-invariant bias gates unchanged"},
        "fallback_exposure": {"eligible_fallback_pct": 0.039},
    }


def test_build_captures_sign_flip_and_exact_identity():
    r = BS.build_basis_sensitivity(_tr_report(), _accrual())
    assert r["zero_coupon_identity"]["exact"] is True
    assert r["level_sensitivity"]["crf_sign_flip"] is True          # clean<0, total>0
    assert r["level_sensitivity"]["basis_sensitive"] is True
    assert r["differential_invariance"]["basis_invariant"] is True


def test_no_sign_flip_when_both_same_sign():
    acc = _accrual()
    acc["levels_corrected"]["crf_clean_pct"] = 0.05                 # both positive -> no flip
    r = BS.build_basis_sensitivity(_tr_report(), acc)
    assert r["level_sensitivity"]["crf_sign_flip"] is False


def test_identity_not_exact_when_a_side_fails():
    tr = _tr_report()
    tr["zero_coupon_invariance"]["corr_ok"] = False
    r = BS.build_basis_sensitivity(tr, _accrual())
    assert r["zero_coupon_identity"]["exact"] is False


@pytest.mark.parametrize("mutate", [
    lambda tr, ac: tr.pop("zero_coupon_invariance"),
    lambda tr, ac: ac.pop("levels_corrected"),
    lambda tr, ac: ac.pop("differentials_static"),
])
def test_missing_block_fails_loud(mutate):
    tr, ac = _tr_report(), _accrual()
    mutate(tr, ac)
    with pytest.raises(BS.BasisSensitivityError):
        BS.build_basis_sensitivity(tr, ac)


def test_missing_source_file_fails_loud(tmp_path):
    with pytest.raises(BS.BasisSensitivityError, match="missing"):
        BS._read_json(tmp_path / "nope.json", "total-return report")


def test_end_to_end_writes_result_no_holdout(tmp_path):
    import json
    (tmp_path / "tr.json").write_text(json.dumps(_tr_report()))
    (tmp_path / "ac.json").write_text(json.dumps(_accrual()))
    out = tmp_path / "basis.json"
    rc = BS.main(["--tr-report", str(tmp_path / "tr.json"), "--accrual", str(tmp_path / "ac.json"),
                 "--out", str(out)])
    assert rc == 0
    r = json.loads(out.read_text())
    assert r["provenance"]["dev_only"] is True and r["provenance"]["model_calls"] == 0
    assert "holdout" not in out.read_text().lower()
