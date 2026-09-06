"""RQ3 validation-fixture exporter — recovery-of-planted-signal blocks. Offline, synthetic-only.

The extracted DGPs are the SAME the test batteries assert against, so these check the exporter
surfaces the recovered numbers correctly (planted alpha recovered, null clean, IPCA subspace/factor
recovered, shuffled-date null collapses, decimal-shift recall 100%).
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import run_rq3_validation_export as VE  # noqa: E402


# --- fast blocks --------------------------------------------------------------------

def test_characteristic_sort_recovers_planted_alpha_and_is_null_clean():
    cs = VE.characteristic_sort_recovery()
    p = cs["planted_long_short_alpha"]
    assert p["within_3se"] is True and p["significant"] is True       # planted signal recovered
    assert cs["null_no_signal"]["no_spurious_signal"] is True          # no false signal
    b = cs["benchmark_regression"]
    assert abs(b["recovered_beta"] - b["planted_beta"]) < 0.1          # beta recovered


def test_meas_err_decimal_shift_recall_is_total():
    me = VE.meas_err_recall()
    assert me["recall"] == 1.0                                         # both slips detected
    assert me["values_recovered"] is True and me["clean_values_untouched"] is True


def test_recovery_sweep_is_sign_stable_over_preregistered_grid():
    sw = VE.recovery_sweep_block()
    assert sw["sign_stable"] is True and sw["grid_is_preregistered"] is True


def test_calibration_pointer_shape():
    cp = VE.calibration_pointer()
    assert cp["component"] == "auditor.layer_b_calibration"
    assert isinstance(cp["present"], bool)


# --- IPCA block + full export (slower: runs the IPCA fits) --------------------------

def test_ipca_recovery_recovers_subspace_and_collapses_on_shuffle():
    ip = VE.ipca_recovery()
    assert ip["subspace_recovery"]["snr_40"]["within_bound"] is True   # tight subspace at high SNR
    assert ip["subspace_recovery"]["snr_4"]["within_bound"] is True
    assert ip["factor_alignment"]["above_bound"] is True               # factors align (R²>0.95)
    assert ip["shuffled_date_null"]["intact_above_half"] is True
    assert ip["shuffled_date_null"]["shuffled_collapses"] is True      # shuffle destroys OOS R²


def test_full_export_writes_zero_spend_dev_only(tmp_path):
    import json
    out = tmp_path / "rq3_validation.json"
    assert VE.main(["--out", str(out)]) == 0
    r = json.loads(out.read_text())
    assert set(r) >= {"characteristic_sort_recovery", "ipca_recovery", "recovery_sweep",
                      "measurement_error_recall", "calibration_pointer", "provenance"}
    assert r["provenance"]["dev_only"] is True and r["provenance"]["synthetic_dgp_only"] is True
    assert r["provenance"]["model_calls"] == 0 and r["provenance"]["spend_usd"] == 0.0
    assert "holdout" not in out.read_text().lower()
