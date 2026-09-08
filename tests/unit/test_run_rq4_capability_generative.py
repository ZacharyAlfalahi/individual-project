"""Tests for scripts/run_rq4_capability_generative.py — the SC-SCI-17 generative sibling.

Load-bearing guarantees, all STRUCTURAL (no live vendor call here — a real run makes live calls;
numeric fidelity is covered by the driver's own run_cell self-verify at run time):
(1) the holdout can never be touched — no rehearsal import, no holdout path, no one-shot holdout bridge;
(2) the generative arm IS wired (the inverse of SC-SCI-16's deferral guard) — LLMResearcherSource
    + build_phase_f_clients, over a dedicated parent-specific cache disjoint from the str funnel;
(3) SC-SCI-16 stays FROZEN — this sibling reuses its helpers (the SAME objects), never forks them,
    and never imports/mutates its driver's behaviour;
(4) exactly the registered parameters — k=5, m=6 defaults, entry bypass recorded;
(5) artifacts can never clobber the funnel OR the SC-SCI-16 capability outputs.
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
import run_rq4_capability_generative as G  # noqa: E402

_SOURCE = Path(G.__file__).read_text(encoding="utf-8")


def test_holdout_is_structurally_unreachable():
    assert "run_oneshot_holdout" not in _SOURCE
    assert "_run_rehearsal" not in _SOURCE
    assert "data/holdout" not in _SOURCE
    assert "SCIENTIST_HOLDOUT_UNLOCK" not in _SOURCE
    # the artefact records the shut holdout, in words
    assert "HOLDOUT STAYS SHUT" in _SOURCE or "NOT OPENED" in _SOURCE


def test_generative_arm_is_wired():
    # the inverse of SC-SCI-16's deferral guard: here the generative source IS constructed.
    assert "LLMResearcherSource" in _SOURCE
    assert "build_phase_f_clients" in _SOURCE
    assert "ResponseCache" in _SOURCE


def test_generative_cache_is_parent_specific_and_disjoint():
    # a dedicated cache root, not the committed str funnel cache
    assert G._CACHE_ROOT != (_REPO_ROOT / "runs" / "rq4_funnel" / "cache")
    assert "rq4_capability_generative" in str(G._CACHE_ROOT)


def test_sc_sci_16_helpers_are_reused_not_forked():
    # the sibling binds SC-SCI-16's FROZEN objects, so the two can never drift apart.
    assert G.C._ANCHORS is C._ANCHORS
    assert G.C.corrected_parent is C.corrected_parent
    assert G.C.directional_ceiling is C.directional_ceiling


def test_registered_parameters_are_defaults():
    sig = inspect.signature(G.run_capability_generative)
    assert sig.parameters["k"].default == 5
    assert sig.parameters["m"].default == 6


def test_entry_bypass_and_amendment_recorded():
    assert "entry_bypassed" in _SOURCE
    assert "SC-SCI-17" in _SOURCE


def test_artifact_name_disjoint_from_funnel_and_sc_sci_16():
    p = G.output_path(Path("/x"), "drf", "minilm")
    assert p.name.startswith("rq4_capability_generative_")
    assert "rq4_funnel" not in p.name
    # SC-SCI-16 writes rq4_capability_<anchor>_...; the generative prefix is a strict superset
    # string, so guard the full stems are not equal for the same args.
    assert p.name != C.output_path(Path("/x"), "drf", "minilm").name


def test_default_anchors_are_the_two_non_entering_parents():
    # parsed from the argparse default; str is excluded (it is the registered funnel)
    parser = [a for a in inspect.getsource(G.main).splitlines() if "default=" in a and "anchors" in a]
    assert parser, "anchors argument must carry a default"
    assert '"drf"' in parser[0] and '"mom6"' in parser[0]
