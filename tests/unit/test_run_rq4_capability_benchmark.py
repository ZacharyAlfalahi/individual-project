"""Tests for scripts/run_rq4_capability_benchmark.py — the SC-SCI-16 capability benchmark.

Load-bearing guarantees, all STRUCTURAL (no real-data run here — the driver's own run_cell
self-verify covers numeric fidelity at run time):
(1) the holdout can never be touched — no rehearsal import, no holdout path, no one-shot holdout bridge;
(2) the generative wall holds — the driver constructs NO LLM source (deferred, not wired);
(3) exactly the registered parameters — k=5, m=6, entry bypass recorded, per-parent
    holding/lookback constants match the gold specs;
(4) artifacts can never clobber the recorded funnel outputs;
(5) the seam parameter defaults are 1 (recorded-behaviour regression bar).
"""

from __future__ import annotations

import inspect
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
for _p in (str(_REPO_ROOT), str(_REPO_ROOT / "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import run_rq4_capability_benchmark as C  # noqa: E402
import run_rq4_exhaustive_benchmark as B  # noqa: E402
from agents.scientist.experimentalist.execution_verifier import execute_g1b  # noqa: E402

_SOURCE = Path(C.__file__).read_text(encoding="utf-8")


def test_holdout_is_structurally_unreachable():
    assert "run_oneshot_holdout" not in _SOURCE     # no rehearsal/holdout bridge import
    assert "_run_rehearsal" not in _SOURCE
    assert "data/holdout" not in _SOURCE
    assert "SCIENTIST_HOLDOUT_UNLOCK" not in _SOURCE


def test_generative_wall_no_llm_source_constructed():
    assert "LLMResearcherSource" not in _SOURCE     # deferred pending Phase-F approval, not wired
    assert "build_phase_d_clients" not in _SOURCE
    assert "build_phase_f_clients" not in _SOURCE
    assert "DEFERRED" in _SOURCE                    # ...and the deferral is recorded, not silent


def test_anchor_constants_match_gold_specs():
    assert C._ANCHORS["str"] == {"holding_period": 1, "signal_lookback": 1}
    assert C._ANCHORS["drf"] == {"holding_period": 1, "signal_lookback": 36}
    assert C._ANCHORS["mom6"] == {"holding_period": 6, "signal_lookback": 7}


def test_registered_parameters_are_defaults():
    sig = inspect.signature(C.run_capability)
    assert sig.parameters["k"].default == 5
    assert sig.parameters["m"].default == 6


def test_artifact_name_cannot_collide_with_funnel_outputs():
    p = C.output_path(Path("/x"), "drf", "minilm")
    assert p.name.startswith("rq4_capability_")     # funnel writes rq4_funnel_*; disjoint prefix
    assert "rq4_funnel" not in p.name


def test_seam_defaults_are_recorded_behaviour():
    assert inspect.signature(execute_g1b).parameters["holding_period"].default == 1
    assert inspect.signature(B.evaluate_batch).parameters["holding_period"].default == 1


def test_directional_ceiling_respects_parent_direction():
    rows = [
        {"status": "evaluated", "alpha_t": 1.2, "id": "pos"},
        {"status": "evaluated", "alpha_t": -0.9, "id": "neg"},
        {"status": "evaluated", "alpha_t": 0.1, "id": "mid"},
        {"status": "duplicate"},
    ]
    up = C.directional_ceiling([dict(r) for r in rows], direction=1)
    assert up["id"] == "pos"
    down = C.directional_ceiling([dict(r) for r in rows], direction=-1)
    assert down["id"] == "neg"                 # improvement for a negative-premium parent


def test_directional_ceiling_ranks_and_empty():
    rows = [{"status": "evaluated", "alpha_t": t} for t in (2.0, -1.0, 0.5)]
    C.directional_ceiling(rows, direction=-1)
    assert [r["rank_in_canonical"] for r in rows] == [3, 1, 2]
    assert C.directional_ceiling([{"status": "refused"}], direction=1) is None
