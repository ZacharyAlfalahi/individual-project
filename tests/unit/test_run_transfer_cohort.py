"""Unit tests for the transfer-cohort harness (scripts/run_transfer_cohort.py).

Pure-core tests on synthetic specs/records only -- no real data, no LLM, no
extraction, no adapter (the bind stage is exercised by the coverage-driver suite).
Follows the repo convention: bare-module import, synthetic fixtures under tmp_path.
"""
import importlib.util
import json
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location(
    "rtc", _REPO / "scripts" / "run_transfer_cohort.py")
rtc = importlib.util.module_from_spec(_spec)
sys.modules["rtc"] = rtc
_spec.loader.exec_module(rtc)


def _stated(value, quote=None, cs=None):
    d = {"tag": "STATED", "value": value}
    if quote:
        d["evidence"] = {"quote": quote,
                         "locator": {"page": 1, "char_start": cs, "char_end": (cs or 0) + len(quote)}}
    return d


def _unknown(reason=None):
    d = {"tag": "UNKNOWN", "value": "None"}
    if reason:
        d["evidence"] = {"unknown_reason": reason}   # production nests it under evidence
    return d


def _con(index, coverage, quote_cert, clean, codes, sc=None, reasons=None):
    """A synthetic per-construction record (the shape evaluate_paper produces)."""
    return {
        "index": index,
        "behavioural": {"coverage": coverage, "quote_cert": quote_cert,
                        "unknown_reasons": reasons or {}},
        "binding": {"clean": clean, "refusal_codes": codes},
        "self_consistency": sc,
    }


def test_tag_leaves_walks_header_part1_part2_only():
    spec = {
        "header": {"a": _stated("x")},
        "part1": {"b": _unknown()},
        "part2": {"nested": {"c": _stated("y")}},
        "paper_facts": {"z": _stated("ignored")},  # NOT a fact-bearing sub-tree
    }
    leaves = rtc._tag_leaves(spec)
    assert len(leaves) == 3  # paper_facts is excluded


def test_behavioural_coverage_quotecert_reasons():
    spec = {"part1": {
        "a": _stated("x", quote="qqq", cs=0),   # STATED + located quote
        "b": _stated("y"),                        # STATED, no located quote
        "c": _unknown("paper_silent"),
        "d": _unknown("paper_silent"),
    }}
    b = rtc.behavioural(spec)
    assert (b["n_fields"], b["n_stated"], b["n_unknown"]) == (4, 2, 2)
    assert b["coverage"] == round(2 / 4, 4)
    assert b["quote_cert"] == round(1 / 2, 4)          # only 'a' is located
    # unknown_reason lives under evidence in the real schema (regression guard)
    assert b["unknown_reasons"] == {"paper_silent": 2}


def test_behavioural_unknown_reason_read_from_evidence():
    # A leaf carrying unknown_reason at the ROOT (the old wrong location) must NOT count.
    spec = {"part1": {"a": {"tag": "UNKNOWN", "value": "None", "unknown_reason": "paper_silent"}}}
    assert rtc.behavioural(spec)["unknown_reasons"] == {}


def test_spec_index_numeric_sort(tmp_path):
    paths = [tmp_path / "spec_2.json", tmp_path / "spec_10.json", tmp_path / "spec_1.json"]
    assert [rtc._spec_index(p) for p in sorted(paths, key=rtc._spec_index)] == [1, 2, 10]


def test_aggregate_construction_weighted_and_paper_attrition():
    recs = [
        # p1: no spec -> blocked at emission
        {"paper_id": "p1", "emitted": False, "n_constructions": 0,
         "n_constructions_bound_clean": 0, "constructions": [],
         "stage_reached": "ingestion", "blocked_at": "spec_emission"},
        # p2: two constructions, neither binds -> blocked at semantic_binding
        {"paper_id": "p2", "emitted": True, "n_constructions": 2,
         "n_constructions_bound_clean": 0,
         "constructions": [
             _con(0, 0.4, 1.0, False, ["MISSING_BINDING"]),
             _con(1, 0.5, None, False, ["REVIEW_REQUIRED"]),
         ],
         "stage_reached": "spec_emission", "blocked_at": "semantic_binding"},
        # p3: two constructions, one binds -> parks at the execution boundary
        {"paper_id": "p3", "emitted": True, "n_constructions": 2,
         "n_constructions_bound_clean": 1,
         "constructions": [
             _con(0, 0.6, 1.0, True, []),
             _con(1, 0.8, 0.5, False, ["MISSING_BINDING"]),
         ],
         "stage_reached": "semantic_binding", "blocked_at": None},
    ]
    agg = rtc.aggregate(recs)
    # paper-weighted attrition over the reachable stages (execution is never reached)
    assert agg["attrition"] == {"ingestion": 3, "spec_emission": 2, "semantic_binding": 1}
    assert agg["chain_localisation"] == {"spec_emission": 1, "semantic_binding": 1}
    assert agg["n_emitted"] == 2                        # papers emitting >= 1 construction
    assert agg["n_constructions"] == 4                  # 2 + 2 (construction-weighted denominator)
    assert agg["n_papers_bound_clean"] == 1
    assert agg["n_constructions_bound_clean"] == 1
    assert agg["reached_execution_boundary"] == 1       # NOT structurally 0
    # construction-weighted: mean over all 4 constructions, not per-paper
    assert agg["mean_coverage"] == round((0.4 + 0.5 + 0.6 + 0.8) / 4, 4)
    # quote_cert only where defined (3 of 4)
    assert agg["mean_quote_cert"] == round((1.0 + 1.0 + 0.5) / 3, 4)
    assert agg["binding_refusal_codes"] == {"MISSING_BINDING": 2, "REVIEW_REQUIRED": 1}


def test_self_consistency_from_trace(tmp_path):
    tp = tmp_path / "trace_0.json"
    tp.write_text(json.dumps({"records": [
        {"agreement": True, "final_reason": "quoted"},
        {"agreement": False, "final_reason": "not_stated"},
        {"agreement": True, "final_reason": "quoted"},
        {"agreement": False, "final_reason": "disagreement"},
    ]}))
    sc = rtc.self_consistency(tp)
    assert sc["n_fields"] == 4
    assert sc["self_consistency"] == round(2 / 4, 4)
    assert sc["reason_histogram"] == {"quoted": 2, "not_stated": 1, "disagreement": 1}


def test_self_consistency_none_without_trace(tmp_path):
    assert rtc.self_consistency(tmp_path / "trace_0.json") is None


def test_evaluate_paper_no_spec_is_blocked_at_emission(tmp_path):
    d = tmp_path / "p"
    d.mkdir()
    # no spec_*.json -> review/zero_specs; bind is never called (subs unused)
    rec = rtc.evaluate_paper("p", d, subs=None)
    assert rec["emitted"] is False
    assert rec["n_constructions"] == 0
    assert rec["constructions"] == []
    assert rec["stage_reached"] == "ingestion"
    assert rec["blocked_at"] == "spec_emission"


def test_evaluate_paper_enumerates_every_construction(tmp_path, monkeypatch):
    d = tmp_path / "paper"
    d.mkdir()
    # three constructions; only the middle one is all-STATED
    specs = {
        0: {"part1": {"a": _stated("x", quote="qq", cs=0), "b": _unknown("paper_silent")}},
        1: {"part1": {"a": _stated("y", quote="zz", cs=0)}},
        2: {"part1": {"a": _unknown("paper_silent")}},
    }
    for i, s in specs.items():
        (d / f"spec_{i}.json").write_text(json.dumps(s))
        (d / f"trace_{i}.json").write_text(
            json.dumps({"records": [{"agreement": True, "final_reason": "quoted"}]}))

    # bind is exercised by the coverage suite; stub it: clean iff no UNKNOWN leaf
    def fake_bind(spec_dict, subs):
        if any(leaf.get("tag") == "UNKNOWN" for leaf in rtc._tag_leaves(spec_dict)):
            return False, ["MISSING_BINDING"]
        return True, []
    monkeypatch.setattr(rtc, "bind", fake_bind)

    rec = rtc.evaluate_paper("paper", d, subs=None)
    assert rec["emitted"] is True
    assert rec["n_constructions"] == 3                       # ALL specs read, not just spec_0
    assert [c["index"] for c in rec["constructions"]] == [0, 1, 2]
    assert rec["n_constructions_bound_clean"] == 1           # only construction 1
    assert rec["stage_reached"] == "semantic_binding"        # best construction bound
    assert rec["blocked_at"] is None and rec["execution_status"] == "not_attempted"
    # every construction picked up its own trace
    assert all(c["self_consistency"] is not None for c in rec["constructions"])


def test_render_md_smoke():
    recs = [
        {"paper_id": "p2", "emitted": True, "n_constructions": 1,
         "n_constructions_bound_clean": 0,
         "constructions": [_con(0, 0.4, 1.0, False, ["MISSING_BINDING"])],
         "stage_reached": "spec_emission", "blocked_at": "semantic_binding"},
        {"paper_id": "p3", "emitted": True, "n_constructions": 2,
         "n_constructions_bound_clean": 1,
         "constructions": [_con(0, 0.6, 1.0, True, []),
                           _con(1, 0.8, 0.5, False, ["MISSING_BINDING"])],
         "stage_reached": "semantic_binding", "blocked_at": None},
    ]
    md = rtc.render_md("expansion", rtc.aggregate(recs), recs)
    assert "Transfer cohort (expansion)" in md
    assert "reached execution boundary" in md
    assert "execution boundary (not run)" in md   # clean-bind paper's per-paper outcome
    assert "| p3 | 2 |" in md          # per-paper construction count rendered
