"""Strictness-frontier simulation (scripts/analyze_strictness_frontier.py).

Pure-logic tests for the ship rule and the spec-variant builder, plus an optional
artifact-backed integration test that pins the frontier output (produced by the
script from the scoped-extraction run archives; skipped when those gitignored
archives are absent).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT))

from scripts.analyze_strictness_frontier import (  # noqa: E402
    _RUN_DIRS,
    never_emitted_detail,
    shippable_value,
    simulate_level,
    value_nodes_of,
)


def _rec(reason, na, nb, field="weight_timing"):
    return {"field": field, "final_tag": "UNKNOWN", "final_reason": reason,
            "normalised_a": na, "normalised_b": nb,
            "model_a": {"quote": "qa"}, "model_b": {"quote": "qb"}}


# --- the ship rule ----------------------------------------------------------

def test_l1_ships_only_agreed_quote_failures():
    assert shippable_value(_rec("quote_match_failure", "x", "x"), "L1") == (True, "x", False)
    # latent disagreement: qmf label, values differ -> never ships, flagged
    assert shippable_value(_rec("quote_match_failure", "x", "y"), "L1") == (False, None, True)
    assert shippable_value(_rec("single_response", "x", None), "L1") == (False, None, False)
    assert shippable_value(_rec("disagreement", "x", "y"), "L1") == (False, None, False)


def test_l0_ships_singles_but_never_value_conflicts():
    assert shippable_value(_rec("single_response", "x", None), "L0") == (True, "x", False)
    assert shippable_value(_rec("single_response", None, "y"), "L0") == (True, "y", False)
    assert shippable_value(_rec("quote_match_failure", "x", None), "L0") == (True, "x", False)
    assert shippable_value(_rec("quote_match_failure", "x", "y"), "L0") == (False, None, True)
    assert shippable_value(_rec("disagreement", "x", "y"), "L0") == (False, None, False)
    assert shippable_value(_rec("not_stated", None, None), "L0") == (False, None, False)


# --- the spec-variant builder ----------------------------------------------

def _spec_dict():
    unk = lambda: {"value": None, "tag": "UNKNOWN",  # noqa: E731
                   "evidence": {"note": "n", "unknown_reason": "quote_match_failure"}}
    return {"part2": {
        "weight_timing": unk(),
        "rf_convention": unk(),
        "legs": [{"bucketing_method": unk(),
                  "sort_signal": {"concept_id": {"value": "x", "tag": "STATED"}}}],
        "combiner": {"kind": unk()},
    }}


def test_value_nodes_maps_commons_legs_and_combiner():
    nodes = value_nodes_of(_spec_dict())
    assert set(nodes) == {"weight_timing", "rf_convention", "bucketing_method", "combiner"}
    # structural fields are never simulatable
    assert "sort_signal" not in nodes and "legs" not in nodes


def test_simulate_level_ships_as_design_and_counts():
    trace = {"records": [
        _rec("quote_match_failure", "formation", "formation", field="weight_timing"),
        _rec("quote_match_failure", "a", "b", field="rf_convention"),     # latent
        _rec("single_response", "equal_count", None, field="bucketing_method"),
    ]}
    sim, n, latent = simulate_level(_spec_dict(), trace, "L1")
    assert (n, latent) == (1, 1)
    assert sim["part2"]["weight_timing"]["tag"] == "DESIGN"          # never STATED
    assert sim["part2"]["weight_timing"]["value"] == "formation"
    assert "frontier-sim L1" in sim["part2"]["weight_timing"]["evidence"]["note"]
    assert sim["part2"]["legs"][0]["bucketing_method"]["tag"] == "UNKNOWN"  # L1: no singles

    sim0, n0, latent0 = simulate_level(_spec_dict(), trace, "L0")
    assert (n0, latent0) == (2, 1)
    assert sim0["part2"]["legs"][0]["bucketing_method"]["value"] == "equal_count"

    sim2, n2, latent2 = simulate_level(_spec_dict(), trace, "L2")
    assert (n2, latent2) == (0, 1)                                   # census only
    assert sim2["part2"]["weight_timing"]["tag"] == "UNKNOWN"        # untouched


def test_simulate_never_touches_the_input_dict():
    d = _spec_dict()
    before = json.dumps(d, sort_keys=True)
    simulate_level(d, {"records": [_rec("single_response", "x", None)]}, "L0")
    assert json.dumps(d, sort_keys=True) == before


def test_value_nodes_rejects_ambiguous_names():
    d = _spec_dict()
    d["part2"]["bucketing_method"] = {"value": None, "tag": "UNKNOWN",
                                      "evidence": {"note": "n"}}  # collides with leg
    with pytest.raises(RuntimeError, match="more than one spec location"):
        value_nodes_of(d)


# --- never_emitted_detail (fixture-driven) ----------------------------------

def _never_emitted_fixture(tmp_path, event_detail, n_events=1):
    """A tiny run dir + gold: 2 strategy constructions, 1 emitted, n emission
    events. Returns (run_dirs, dir_paper, gold_paths)."""
    import yaml

    run = tmp_path / "toy"
    (run / "raw").mkdir(parents=True)
    gold = tmp_path / "enum_toy.yaml"
    gold.write_text(yaml.safe_dump({"gold": True, "paper_id": "TOY_1", "constructions": [
        {"name": "Kept", "quote": "q1", "class": "strategy"},
        {"name": "Dropped", "quote": "q2", "class": "strategy"},
    ]}), encoding="utf-8")
    (run / "spec_0.json").write_text(json.dumps(
        {"header": {"paper_id": "TOY_1", "strategy_label": {"value": "Kept"}}}),
        encoding="utf-8")
    (run / "events.json").write_text(json.dumps(
        [{"kind": "LibrarianEmissionError", "detail": event_detail}] * n_events),
        encoding="utf-8")
    for model in ("a", "b"):
        recs = []
        for cname, kind_val, axis_val in (("Kept", "single", None),
                                          ("Dropped", "conditional", f"axis_{model}")):
            recs.append({"field": "formation_structure", "parsed": {"value": "x"}})
            recs.append({"field": "sort_kind", "parsed": {"value": kind_val}})
            recs.append({"field": "control_axis", "parsed": {"concept_id": axis_val}})
        (run / "raw" / f"raw_model_{model}.jsonl").write_text(
            "\n".join(json.dumps(r) for r in recs) + "\n", encoding="utf-8")
    return [run], {"toy": "TOY_1"}, {"TOY_1": gold}


def test_never_emitted_attributes_answers_and_guard1_disposition(tmp_path):
    args = _never_emitted_fixture(tmp_path, "sort_kind clash (Guard 1, schema-v1.1 §3)")
    rows = never_emitted_detail(*args)
    assert [r["name"] for r in rows] == ["Dropped"]        # never the emitted one
    row = rows[0]
    assert row["disposition"] == "guard1_emission_refusal"
    assert row["sort_kind"] == {"a": "conditional", "b": "conditional"}
    assert row["control_axis"] == {"a": "axis_a", "b": "axis_b"}


def test_never_emitted_generic_disposition_for_non_guard1(tmp_path):
    args = _never_emitted_fixture(tmp_path, "some other validation failure")
    rows = never_emitted_detail(*args)
    assert rows[0]["disposition"] == "emission_refusal"    # never mislabeled Guard-1


def test_never_emitted_fails_loud_on_event_count_mismatch(tmp_path):
    args = _never_emitted_fixture(tmp_path, "x (Guard 1)", n_events=2)
    with pytest.raises(RuntimeError, match="refuse to label"):
        never_emitted_detail(*args)


# --- machine-local integration pin ------------------------------------------

@pytest.mark.skipif(
    not all(p.exists() for p in _RUN_DIRS)
    or not (_REPO_ROOT / "results" / "strictness_frontier.json").exists(),
    reason="scoped-extraction run archives / frontier artifact absent on this machine",
)
def test_recorded_frontier_artifact_pins():
    d = json.loads((_REPO_ROOT / "results" / "strictness_frontier.json")
                   .read_text(encoding="utf-8"))
    assert d["cross_pin"]["n_specs"] == 23
    for level in ("L2", "L1", "L0"):
        assert d["frontier"][level]["n_compiled"] == 0       # flat at zero
        assert d["frontier"][level]["n_executed"] == 0
        assert d["frontier"][level]["frontier_fir_events"] == []
    # 6 dfps Guard-1 (5 in-denominator + the excluded 153-family) + 3 bbw Guard-1
    assert len(d["never_emitted"]) == 9
