"""System-level integration arms: the clean-variant integration null must manufacture no 
effect on clean data, and the integrated known-error positive control must reproduce the 
documented lead/lag defect on LIVE Quant->Auditor output and restore it. Deterministic, $0,
dev/synthetic only (no /data/holdout/).
"""
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from run_integration_arms import run_clean_null, run_integrated_leadlag_positive  # noqa: E402


def test_clean_variant_integration_null_manufactures_no_effect():
    r = run_clean_null(seed=2026, replicates=40)
    # The instrument stays quiet on a clean panel: every toggle within +/-vartheta, every CI covers 0.
    assert r["all_toggles_within_vartheta"] is True
    assert r["all_ci_cover_zero"] is True
    assert r["passed"] is True
    # The survivorship toggle (the one the T4b positive plants) collapses to ~0 here.
    assert abs(r["toggles"]["survivorship"]["doe_bp"]) < r["vartheta_bp"]


@pytest.mark.skipif(
    not (REPO_ROOT / "data" / "development" / "monthly_panel_maximal.parquet").exists(),
    reason="requires the licensed dev panel (data/development/) — not shipped; see README")
def test_integrated_known_error_positive_control_detects_and_restores():
    r = run_integrated_leadlag_positive()
    assert r["passed"] is True                       # graded by the pre-registered known-error gate
    drf = r["arms"]["drf"]                            # drf is the primary (gating) anchor
    assert drf["corr_correct_vs_defective"] < 0.7     # defect reproduced (collapse below the band)
    assert drf["corr_correct_vs_restored"] >= 0.90    # correction restores the factor
    assert r["pipeline"] == "quant_to_auditor_live"   # honest scope: live Quant->Auditor, not full path
