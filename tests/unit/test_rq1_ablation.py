"""B7: the RQ1 ablation-grid driver (scripts/run_rq1_ablation.py)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT))

from scripts.run_rq1_ablation import _ship, score_grid                    # noqa: E402

_RUNS = _REPO_ROOT / "runs" / "g3_v3"
needs_runs = pytest.mark.skipif(not (_RUNS / "bbw" / "raw" / "raw_model_a.jsonl").exists(),
                                reason="recorded dev run not on disk")

_A = {"answered": True, "quote": "qa", "value": "v"}
_B = {"answered": True, "quote": "qb", "value": "v"}
_SILENT = {"answered": False, "quote": None, "value": None}


def test_ship_full_system_needs_agreement_and_both_locates():
    assert _ship(_A, _B, True, True, "v", "v", dual=True, quote=True) == (True, "v")
    assert _ship(_A, _B, True, False, "v", "v", dual=True, quote=True) == (False, None)
    assert _ship(_A, _B, True, True, "v", "w", dual=True, quote=True) == (False, None)
    assert _ship(_A, _SILENT, True, False, "v", None, dual=True, quote=True) == (False, None)


def test_ship_quote_off_ignores_location():
    assert _ship(_A, _B, False, False, "v", "v", dual=True, quote=False) == (True, "v")


def test_ship_single_model_prefers_a_then_b_deterministically():
    assert _ship(_A, _B, True, True, "va", "vb", dual=False, quote=True) == (True, "va")
    assert _ship(_SILENT, _B, False, True, None, "vb", dual=False, quote=True) == (True, "vb")
    # quote gate on: an unlocated model_a is skipped in favour of a located model_b
    assert _ship(_A, _B, False, True, "va", "vb", dual=False, quote=True) == (True, "vb")
    assert _ship(_A, _B, False, False, "va", "vb", dual=False, quote=True) == (False, None)
    assert _ship(_A, _B, False, False, "va", "vb", dual=False, quote=False) == (True, "va")


@needs_runs
def test_grid_monotonicity_on_the_recorded_run():
    """Removing a discipline can only widen what ships: coverage must be
    monotone non-decreasing from the full system to neither."""
    rows = {r["combo"]: r for r in score_grid("drf", _RUNS / "bbw")}
    cov = {c: r["coverage"][0] for c, r in rows.items()}
    assert cov["full system (dual + quote gate)"] <= cov["dual-model only (no quote gate)"]
    assert cov["dual-model only (no quote gate)"] <= cov["neither discipline"]
    assert cov["quote gate only (single-model)"] <= cov["neither discipline"]
    # the replayed full system matches the production shipping decision count
    universe = rows["full system (dual + quote gate)"]["coverage"][1]
    assert universe == 44
