"""RQ4 funnel POSITIVE CONTROL — the funnel has discriminating power.

A planted strong extension (a synthetic known-answer signal) must ADVANCE through the real
funnel (BH-FDR -> CPCV -> G5), and a no-signal extension must be REJECTED. This is the counterpart
to the real-data result (0 / 24 advanced): together they show that null is a TRUE null, not a dead
pipeline. Deterministic, synthetic, no real data, never touches the holdout.
"""
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from run_rq4_positive_control import run_positive_control  # noqa: E402


def test_funnel_advances_planted_strong_and_rejects_noise():
    r = run_positive_control()

    # POWER: the planted strong extension advances, with a decisive (not marginal) margin.
    planted = r["planted_strong"]
    assert planted["advanced"] is True
    assert planted["p_bh"] is not None and planted["p_bh"] < 0.01      # far inside the q=0.10 bar
    assert planted["cpcv_median_sharpe"] is not None and planted["cpcv_median_sharpe"] > 0
    assert r["power_planted_advances"] is True

    # SPECIFICITY: the no-signal extension is rejected (not a false positive).
    noise = r["noise"]
    assert noise["advanced"] is False
    assert noise["p_bh"] is not None and noise["p_bh"] > 0.10          # nowhere near significant
    assert r["specificity_noise_rejected"] is True

    # Provenance guardrails: this instrument never masquerades as real evidence.
    assert r["synthetic"] and not r["reads_real_data"] and not r["touches_holdout"]
    assert r["cost_usd"] == 0.0
