#!/usr/bin/env python
"""
The locator census (DIAGNOSTIC; never a headline).

Classifies, descriptively, every non-locating quote on the reportable run's
``agree_quote_gate_failed`` fields (both models agreed on the value; at least
one quote failed the exact quote gate). The bands answer "what KIND of failure
is this?" — they are descriptive labels and are NEVER the acceptance rule
(that is the calibrated bar, chosen on independent gold-quote data; this census
may print target scores precisely because the bar rule cannot see them).

Reads the run archives left by a corpus extraction run and the committed
metrics report; both are gitignored and not shipped, so a clean clone cannot
run this until they are regenerated locally.

Target selection uses the RECORDED run artefacts (`load_run` → RunField
`.conditions`), i.e. literally the rows the committed g3 report counted — and
cross-pins its per-anchor counts against that report's `condition_incidence`
(`RuntimeError "... do not report"` on any mismatch, nothing written). The
quote texts also come from the recorded trace; no live model call, no LLM.

Also home of ``read_raw_segmented``, the multi-spec raw-archive reader the
re-scoring pass uses (`_read_raw` refuses duplicate field names, but a
multi-strategy dir's shared raw JSONL repeats each field once per strategy;
the archives store lines in exactly concatenated trace order, which this
reader verifies fail-loud rather than assumes).
"""

from __future__ import annotations

import argparse
import json
import sys
from difflib import SequenceMatcher
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from agents.librarian.config import load_canonical_text                    # noqa: E402
from agents.librarian.config.canonical_text import CanonicalText           # noqa: E402
from agents.librarian.config.normalise import normalise                    # noqa: E402
from evaluation.harness.relocate import score_quote                        # noqa: E402
from evaluation.harness.relocate_thresholds import (                       # noqa: E402
    load_relocator_calibration_config,
)
from evaluation.harness.reportability import require_reportable            # noqa: E402
from evaluation.harness.run_artefacts import RunArtefacts, load_run        # noqa: E402
from scripts.run_g3_score import DEFAULT_DIR_OF                            # noqa: E402
from scripts.run_librarian import PAPERS                                   # noqa: E402

# Descriptive score bands (never the acceptance rule).
SMALL_DRIFT_FLOOR = 0.85
PARAPHRASE_FLOOR = 0.50
EDGE_BLOCK_SHARE = 0.80

_PAPER_KEY_OF_DIR = {"bbw": "bbw", "jnps": "jnps", "drr": "drr"}


# ---------------------------------------------------------------------------
# Shared multi-spec raw reader (used by scripts/run_relocator_rescore.py).
# ---------------------------------------------------------------------------

def _parse_raw_line(rec: dict) -> dict:
    """One raw JSONL record -> the `_read_raw` output shape (same semantics:
    a parse-failed or unparsed record is a silent model)."""
    parsed = rec.get("parsed") or {}
    answered = bool(rec.get("answered")) and not rec.get("parse_failed") and bool(
        parsed.get("answered")
    )
    kind = rec.get("kind")
    if kind == "signal_ref":
        value = parsed.get("concept_id")
    elif kind == "method_summary":
        value = parsed.get("summary")
    else:
        value = parsed.get("value")
    return {
        "answered": answered,
        "value": value if answered else None,
        "quote": (parsed.get("quote") if answered and kind != "method_summary" else None),
        "kind": kind,
        "model_id": rec.get("configured_model_id"),
        "returned": rec.get("returned_model_version"),
    }


def read_raw_segmented(run_dir: str | Path) -> list[tuple[dict[str, dict], dict[str, dict]]]:
    """One ``(raw_a, raw_b)`` field-keyed mapping per strategy index.

    The raw JSONL files hold every strategy's calls concatenated in trace
    order (a verified property of the run archives — but verified HERE,
    per run, not assumed): segment ``i`` must reproduce ``trace_i``'s record
    field sequence exactly, and each line's answered bit and quote must match
    the trace's own per-model record. Any mismatch raises ``RuntimeError``."""
    run_dir = Path(run_dir)
    trace_paths = sorted(
        (p for p in run_dir.glob("trace_*.json") if p.stem.removeprefix("trace_").isdigit()),
        key=lambda p: int(p.stem.removeprefix("trace_")),
    )
    if not trace_paths:
        raise RuntimeError(f"{run_dir}: no trace_<n>.json files found")
    numbers = [int(p.stem.removeprefix("trace_")) for p in trace_paths]
    if numbers != list(range(len(numbers))):
        # Callers index the returned list by STRATEGY NUMBER (resolve_strategy's
        # spec_{i} index), which only equals the list position when the trace
        # numbering is contiguous from zero — make the coupling explicit.
        raise RuntimeError(
            f"{run_dir}: non-contiguous trace numbering {numbers}; segment position "
            "would not equal strategy index"
        )
    traces = [json.loads(p.read_text(encoding="utf-8")) for p in trace_paths]
    field_seqs = [[r["field"] for r in (t.get("records") or [])] for t in traces]

    def read_lines(name: str) -> list[dict]:
        path = run_dir / "raw" / name
        if not path.exists():
            raise RuntimeError(f"{run_dir}: missing raw archive {name}")
        with path.open("r", encoding="utf-8") as fh:
            return [json.loads(line) for line in fh]

    lines_a, lines_b = read_lines("raw_model_a.jsonl"), read_lines("raw_model_b.jsonl")
    total = sum(len(s) for s in field_seqs)
    for name, lines in (("raw_model_a.jsonl", lines_a), ("raw_model_b.jsonl", lines_b)):
        if len(lines) != total:
            raise RuntimeError(
                f"{run_dir}/{name}: {len(lines)} lines vs {total} records summed over "
                f"trace_*.json — the archive does not correspond to the traces"
            )

    out: list[tuple[dict[str, dict], dict[str, dict]]] = []
    cursor = 0
    for idx, (trace, seq) in enumerate(zip(traces, field_seqs)):
        seg: list[tuple[dict[str, dict], str]] = []
        for role, lines in (("a", lines_a), ("b", lines_b)):
            chunk = lines[cursor:cursor + len(seq)]
            got = [rec["field"] for rec in chunk]
            if got != seq:
                raise RuntimeError(
                    f"{run_dir}: raw_model_{role}.jsonl segment {idx} field order does not "
                    f"match trace_{idx}.json (first divergence at position "
                    f"{next(i for i, (g, s) in enumerate(zip(got, seq)) if g != s)})"
                )
            mapping = {rec["field"]: _parse_raw_line(rec) for rec in chunk}
            if len(mapping) != len(chunk):
                raise RuntimeError(
                    f"{run_dir}: raw_model_{role}.jsonl segment {idx} repeats a field "
                    "name within one strategy — the field-keyed mapping would silently "
                    "drop a record"
                )
            for rec, trec in zip(chunk, trace.get("records") or []):
                m = trec["model_a"] if role == "a" else trec["model_b"]
                parsed_line = mapping[rec["field"]]
                if parsed_line["answered"] != bool(m["answered"]):
                    raise RuntimeError(
                        f"{run_dir}: segment {idx} field {rec['field']!r} model_{role} "
                        f"answered bit diverges between raw archive and trace"
                    )
                if parsed_line["answered"] and parsed_line["kind"] != "method_summary" \
                        and parsed_line["quote"] != m.get("quote"):
                    raise RuntimeError(
                        f"{run_dir}: segment {idx} field {rec['field']!r} model_{role} "
                        f"quote diverges between raw archive and trace"
                    )
            seg.append((mapping, role))
        out.append((seg[0][0], seg[1][0]))
        cursor += len(seq)
    return out


# ---------------------------------------------------------------------------
# Census proper.
# ---------------------------------------------------------------------------

def gate_failed_fields(art: RunArtefacts) -> list[str]:
    """Field names carrying the recorded ``agree_quote_gate_failed`` condition,
    in deterministic name order — exactly the rows the g3 report counted."""
    return sorted(
        name for name, rec in art.fields.items()
        if "agree_quote_gate_failed" in rec.conditions
    )


def _longest_block(ct: CanonicalText, q_l1: str) -> tuple[int, int, int]:
    """(size, b_start, len(q_l1)) of the longest common block between the quote
    and any page / adjacent-pair join — for the edge-anchoring test only."""
    best = (0, 0)
    texts = [normalise(p, "L1") for p in ct.pages]
    texts += [normalise(ct.pages[i] + "\n" + ct.pages[i + 1], "L1")
              for i in range(len(ct.pages) - 1)]
    matcher = SequenceMatcher(autojunk=False)
    matcher.set_seq2(q_l1)
    for text in texts:
        matcher.set_seq1(text)
        m = matcher.find_longest_match(0, len(text), 0, len(q_l1))
        if m.size > best[0]:
            best = (m.size, m.b)
    return best[0], best[1], len(q_l1)


def classify_quote(ct: CanonicalText, quote: str | None, *, min_anchor_chars: int) -> dict:
    """First-match cascade over the descriptive bands."""
    if quote is None or not isinstance(quote, str) or quote.strip() == "":
        return {"class": "empty_quote", "score": 0.0, "quote_len_l1": 0, "window_head": None}
    q_l1 = normalise(quote, "L1")
    row: dict = {"quote_len_l1": len(q_l1)}

    if ct.locate(quote, level="L2") is not None:
        row.update({"class": "l2_only", "score": 1.0, "window_head": None})
        return row
    if q_l1 and q_l1 in normalise("\n".join(ct.pages), "L1"):
        row.update({"class": "beyond_adjacent_pair", "score": 1.0, "window_head": None})
        return row

    cand = score_quote(ct, quote, min_anchor_chars=min_anchor_chars)
    if cand is None:
        row.update({"class": "not_in_text", "score": 0.0, "window_head": None})
        return row
    score = cand.score
    head = None
    if cand.cross_page:
        text_l1 = normalise(ct.pages[cand.page] + "\n" + ct.pages[cand.page + 1], "L1")
    else:
        text_l1 = normalise(ct.pages[cand.page], "L1")
    head = text_l1[cand.l1_start:cand.l1_start + 80]

    size, b_start, q_len = _longest_block(ct, q_l1)
    edge_anchored = (b_start == 0) or (b_start + size == q_len)
    if size >= EDGE_BLOCK_SHARE * q_len and edge_anchored:
        cls = "truncation_edge_drift"
    elif score >= SMALL_DRIFT_FLOOR:
        cls = "small_drift"
    elif score >= PARAPHRASE_FLOOR:
        cls = "heavy_paraphrase"
    else:
        cls = "not_in_text"
    row.update({"class": cls, "score": round(score, 4), "window_head": head})
    return row


def run_census(run_root: Path, g3_json: Path, anchors: list[str],
               *, allow_non_reportable: bool) -> dict:
    committed = json.loads(g3_json.read_text(encoding="utf-8"))
    cfg = load_relocator_calibration_config()

    # Pass 1: select targets and cross-pin BEFORE loading any canonical text,
    # so a pin failure aborts without touching the (machine-local) frozen texts.
    loaded: list[tuple[str, Path, object, list[str]]] = []
    per_anchor_counts: dict[str, int] = {}
    for anchor in anchors:
        run_dir = run_root / DEFAULT_DIR_OF[anchor]
        art = load_run(run_dir)
        require_reportable(art.reportability, allow_non_reportable=allow_non_reportable)
        targets = gate_failed_fields(art)
        per_anchor_counts[anchor] = len(targets)

        pinned = committed["anchors"][anchor]["condition_incidence"].get(
            "agree_quote_gate_failed", 0)
        if len(targets) != pinned:
            raise RuntimeError(
                f"census cross-pin failed for {anchor!r}: recomputed "
                f"agree_quote_gate_failed count {len(targets)} vs committed {pinned} "
                f"({g3_json.name}) — the recorded run has drifted; do not report"
            )
        loaded.append((anchor, run_dir, art, targets))

    # Pass 2: classify every failing quote against its paper's canonical text.
    rows: list[dict] = []
    for anchor, run_dir, art, targets in loaded:
        paper_key = _PAPER_KEY_OF_DIR[run_dir.name]
        ct = load_canonical_text(_REPO_ROOT / PAPERS[paper_key]["canonical_text"])

        for name in targets:
            rec = art.fields[name]
            for role, answered, located, quote in (
                ("a", rec.a_answered, rec.a_located, rec.a_quote),
                ("b", rec.b_answered, rec.b_located, rec.b_quote),
            ):
                if not answered or located:
                    continue
                row = {"anchor": anchor, "field": name, "model": role}
                row.update(classify_quote(ct, quote, min_anchor_chars=cfg.min_anchor_chars))
                rows.append(row)

    by_class: dict[str, int] = {}
    for r in rows:
        by_class[r["class"]] = by_class.get(r["class"], 0) + 1
    return {
        "diagnostic": "Locator census — descriptive bands; never the acceptance rule",
        "g3_report": str(g3_json.relative_to(_REPO_ROOT)) if g3_json.is_absolute() else str(g3_json),
        "cross_pin": {"agree_quote_gate_failed": per_anchor_counts},
        "bands": {"small_drift_floor": SMALL_DRIFT_FLOOR, "paraphrase_floor": PARAPHRASE_FLOOR,
                  "edge_block_share": EDGE_BLOCK_SHARE},
        "by_class": dict(sorted(by_class.items())),
        "n_failing_quotes": len(rows),
        "rows": rows,
    }


def _render_md(result: dict) -> str:
    lines = [
        "# Locator census (diagnostic; descriptive bands, never the acceptance rule)",
        "",
        f"Cross-pin (agree_quote_gate_failed per anchor): {result['cross_pin']['agree_quote_gate_failed']}",
        f"Failing quotes classified: {result['n_failing_quotes']}",
        "",
        "| class | n |",
        "|---|---|",
    ]
    lines += [f"| {c} | {n} |" for c, n in result["by_class"].items()]
    lines += ["", "| anchor | field | model | class | score | len(L1) |", "|---|---|---|---|---|---|"]
    lines += [
        f"| {r['anchor']} | {r['field']} | {r['model']} | {r['class']} | {r['score']} | {r['quote_len_l1']} |"
        for r in result["rows"]
    ]
    lines.append("")
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Locator census (diagnostic).")
    ap.add_argument("--run-root", default="runs/corpus_anchors_report")
    ap.add_argument("--g3", default="results/g3_report.json")
    ap.add_argument("--anchors", default="str,drf,mom6")
    ap.add_argument("--allow-non-reportable", action="store_true")
    ap.add_argument("--json", required=True, dest="json_out")
    args = ap.parse_args(argv)

    run_root = Path(args.run_root)
    if not run_root.is_absolute():
        run_root = _REPO_ROOT / run_root
    g3_json = Path(args.g3)
    if not g3_json.is_absolute():
        g3_json = _REPO_ROOT / g3_json
    anchors = [a.strip() for a in args.anchors.split(",") if a.strip()]

    result = run_census(run_root, g3_json, anchors,
                        allow_non_reportable=args.allow_non_reportable)

    out = Path(args.json_out)
    if not out.is_absolute():
        out = _REPO_ROOT / out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    out.with_suffix(".md").write_text(_render_md(result), encoding="utf-8")
    print(f"[census] json -> {out}", file=sys.stderr)
    print(f"[census] md   -> {out.with_suffix('.md')}", file=sys.stderr)
    print(json.dumps(result["by_class"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
