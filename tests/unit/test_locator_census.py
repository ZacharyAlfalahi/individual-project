"""Locator census tests (scripts/run_locator_census.py) — tmp_path synthetic run
dirs and stub canonical texts; the census cross-pin fires before any canonical
text is loaded, so the fail-loud test never touches machine-local assets."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT))

from agents.librarian.config.canonical_text import CanonicalText            # noqa: E402
from evaluation.harness.run_artefacts import RunField, canonical_trace_json  # noqa: E402
from scripts.run_locator_census import (                                    # noqa: E402
    classify_quote,
    gate_failed_fields,
    read_raw_segmented,
    run_census,
)
from scripts.run_p6a_replay import _read_raw                                # noqa: E402


def _ct(pages: tuple[str, ...]) -> CanonicalText:
    return CanonicalText(
        source_pdf="stub.pdf", source_sha256="deadbeef",
        parser={"name": "stub", "version": "0"},
        normalisation={"ladder_level": "L1", "rules": []},
        pages=pages, status="stub",
    )


def _raw_line(field: str, quote: str | None, *, kind: str = "enum", value: str = "v") -> dict:
    return {"field": field, "kind": kind, "configured_model_id": "m", "answered": True,
            "parse_failed": False,
            "parsed": {"field": field, "answered": True, "value": value, "quote": quote}}


def _trace_record(field: str, quote: str | None) -> dict:
    return {"field": field,
            "model_a": {"answered": True, "quote": quote, "locate_result": None,
                        "model_id": "m"},
            "model_b": {"answered": True, "quote": quote, "locate_result": None,
                        "model_id": "m"},
            "normalised_a": "v", "normalised_b": "v",
            "final_tag": "UNKNOWN", "final_reason": "quote_match_failure",
            "ship_choice": None}


def _write_run_dir(tmp_path: Path, name: str, field_seqs: list[list[str]]) -> Path:
    run_dir = tmp_path / name
    (run_dir / "raw").mkdir(parents=True)
    all_lines = []
    for i, seq in enumerate(field_seqs):
        trace = {"header": {"paper_id": "STUB"},
                 "records": [_trace_record(f, f"quote for {f}") for f in seq]}
        (run_dir / f"trace_{i}.json").write_text(json.dumps(trace), encoding="utf-8")
        all_lines += [_raw_line(f, f"quote for {f}") for f in seq]
    for fname in ("raw_model_a.jsonl", "raw_model_b.jsonl"):
        (run_dir / "raw" / fname).write_text(
            "\n".join(json.dumps(line) for line in all_lines) + "\n", encoding="utf-8")
    return run_dir


def test_read_raw_segmented_matches_trace_order(tmp_path):
    # Two strategies sharing a field name — the exact case _read_raw refuses.
    run_dir = _write_run_dir(tmp_path, "multi", [["f1", "f2"], ["f1", "f3"]])
    segments = read_raw_segmented(run_dir)
    assert len(segments) == 2
    assert sorted(segments[0][0]) == ["f1", "f2"]
    assert sorted(segments[1][0]) == ["f1", "f3"]

    # Single-strategy dirs must agree byte-for-byte with the canonical reader.
    single = _write_run_dir(tmp_path, "single", [["g1", "g2", "g3"]])
    seg_a, seg_b = read_raw_segmented(single)[0]
    assert seg_a == _read_raw(single / "raw" / "raw_model_a.jsonl")
    assert seg_b == _read_raw(single / "raw" / "raw_model_b.jsonl")

    # A scrambled archive (raw order != trace order) fails loud.
    scrambled = _write_run_dir(tmp_path, "scrambled", [["h1", "h2"]])
    raw = scrambled / "raw" / "raw_model_a.jsonl"
    lines = raw.read_text(encoding="utf-8").strip().split("\n")
    raw.write_text("\n".join(reversed(lines)) + "\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="field order does not match"):
        read_raw_segmented(scrambled)

    # Non-contiguous trace numbering fails loud (segment position must equal
    # strategy index for callers that subscript by it).
    gapped = _write_run_dir(tmp_path, "gapped", [["k1"], ["k2"]])
    (gapped / "trace_1.json").rename(gapped / "trace_2.json")
    with pytest.raises(RuntimeError, match="non-contiguous trace numbering"):
        read_raw_segmented(gapped)

    # A repeated field name WITHIN one strategy fails loud (the field-keyed
    # mapping would otherwise silently drop a record).
    dup = _write_run_dir(tmp_path, "dup", [["d1", "d1"]])
    with pytest.raises(RuntimeError, match="repeats a field name"):
        read_raw_segmented(dup)


def _run_field(name: str, *, a_located: bool, b_located: bool,
               norm_a: str = "v", norm_b: str = "v") -> RunField:
    return RunField(
        field=name, a_answered=True, b_answered=True,
        a_quote=f"qa {name}", b_quote=f"qb {name}",
        a_located=a_located, b_located=b_located,
        a_model_id="m", b_model_id="m",
        normalised_a=norm_a, normalised_b=norm_b,
        final_tag="UNKNOWN", shipped_reason="r", ship_choice=None, not_extracted=False,
    )


def test_gate_failed_fields_selects_agree_qgf_only():
    art = SimpleNamespace(fields={
        "lost": _run_field("lost", a_located=False, b_located=True),
        "located": _run_field("located", a_located=True, b_located=True),
        "disagree": _run_field("disagree", a_located=False, b_located=True, norm_b="w"),
    })
    assert gate_failed_fields(art) == ["lost"]


def test_census_classifier_cascade():
    sentence = ("The distress factor earns a significant premium after controlling "
                "for duration and rating in every specification we run.")
    pages = (
        "The disas- ter came quickly to the bond market that year overall. " + sentence,
        "alpha beta gamma delta",
        "epsilon zeta eta theta",
    )
    ct = _ct(pages)

    assert classify_quote(
        ct, "The disaster came quickly to the bond market that year overall.",
        min_anchor_chars=10)["class"] == "l2_only"

    ct3 = _ct(("alpha beta gamma delta", "epsilon zeta eta theta", "iota kappa lambda mu"))
    assert classify_quote(
        ct3, "gamma delta epsilon zeta eta theta iota kappa",
        min_anchor_chars=10)["class"] == "beyond_adjacent_pair"

    prefix = sentence[:96]
    assert classify_quote(ct, prefix + " zzzz qqqq wwww",
                          min_anchor_chars=10)["class"] == "truncation_edge_drift"

    swapped = sentence.replace("controlling", "adjusting")
    row = classify_quote(ct, swapped, min_anchor_chars=10)
    assert row["class"] == "small_drift" and row["score"] >= 0.85

    unrelated = "Neural translation quality improves with wider context windows overall."
    assert classify_quote(ct, unrelated, min_anchor_chars=10)["class"] == "not_in_text"

    assert classify_quote(ct, None, min_anchor_chars=10)["class"] == "empty_quote"


def test_census_count_cross_pin_fails_loud(tmp_path):
    # A valid run dir whose recomputed agree_qgf count (1) mismatches a fake
    # committed report (5) -> RuntimeError before any canonical text is loaded.
    run_dir = tmp_path / "drr"
    run_dir.mkdir()
    trace = {"header": {"paper_id": "STUB"},
             "records": [_trace_record("f1", "some quote text")]}
    trace_json = canonical_trace_json(trace)
    (run_dir / "trace_0.json").write_text(trace_json, encoding="utf-8")
    spec = {"header": {"trace_sha256": hashlib.sha256(trace_json.encode()).hexdigest()}}
    (run_dir / "spec_0.json").write_text(json.dumps(spec), encoding="utf-8")

    fake_g3 = tmp_path / "g3.json"
    fake_g3.write_text(json.dumps(
        {"anchors": {"str": {"condition_incidence": {"agree_quote_gate_failed": 5}}}}),
        encoding="utf-8")

    with pytest.raises(RuntimeError, match="do not report"):
        run_census(tmp_path, fake_g3, ["str"], allow_non_reportable=True)
