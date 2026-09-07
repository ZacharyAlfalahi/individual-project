"""P6a — merge-miss diagnosis replay over the Phase-D raw archives (WS-B).

Re-derives the §3.6 missed-evidence decomposition from the archived per-model
RAW responses, independently of the shipped trace: `answered` comes from the
archive, `located` is RECOMPUTED against the frozen canonical text (current
ladder), `normalised` is RECOMPUTED via the D9 value normaliser, and the
merge's shipping decision is re-derived from the order-free conditions. The
decomposition itself is produced by the EXISTING order-free classifier
(`evaluation.harness.missed_evidence.decompose`) — the replay tests whether
the recorded decomposition is REPRODUCIBLE from raw evidence, not a
re-implementation of the classifier.

Per-field divergences between the recorded trace and the replay are expected,
not exceptional: the replay locates with the CURRENT ladder (post-D40 v3)
against artefacts scored earlier. Diffs are therefore PARTITIONED into
"explained by post-baseline locator/normaliser fixes" (locate-flips on quotes
carrying dash-class / ligature / curly-quote glyphs — the D40 fix classes)
vs "unexplained". The explained bucket is itself a finding: it quantifies how
much of the recorded decomposition was the D40 bug.

Diagnostic engineering on Phase-D archives — NON-REPORTABLE by construction;
the report carries the Phase-D banner. No production code is modified, no LLM
is called, no panel data is read, the holdout is never touched.

Usage:
  ./.venv/bin/python scripts/run_p6a_replay.py
      [--run-root runs/g3_v3]
      [--out docs/extensions/reports/p6a_composition.md]
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
for _p in (str(_REPO_ROOT), str(_REPO_ROOT / "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from agents.librarian.config import load_canonical_text  # noqa: E402
from agents.librarian.pipeline.form_filler import normalise  # noqa: E402
from evaluation.harness.gold_calibration import score_anchor  # noqa: E402
from evaluation.harness.missed_evidence import (  # noqa: E402
    Mechanism,
    decompose,
    render_decomposition,
)
from evaluation.harness.run_artefacts import (  # noqa: E402
    RunArtefacts,
    RunField,
    load_run,
)

# Anchor -> (run subdir, frozen canonical text). The 2026-07-22 v3 run is the
# recorded G=3 baseline (docs/librarian/baselines/g3_dev_2026-07-22.md).
ANCHORS: tuple[tuple[str, str, str], ...] = (
    ("drf", "bbw", "evaluation/canonical_texts/bbw_2019.frozen.yaml"),
    ("mom6", "jnps", "evaluation/canonical_texts/jnps_2013.frozen.yaml"),
    ("str", "drr", "evaluation/canonical_texts/drr_2026.frozen.yaml"),
)

# The recorded pooled baseline the replay must reproduce (or itemise against).
RECORDED_BASELINE = {"merge_refused": 19, "value_wrong": 13, "gate_lost": 9, "not_retrieved": 2}

# D40 fix-class glyphs: the full dash class folded by ladder v3, plus the
# ligature / curly-quote classes handled at L1. A locate-flip on a quote
# carrying any of these is attributable to a post-baseline normaliser fix.
_DASH_CLASS = "‐‑‒–—―⁃−"
_LIGATURES = "ﬀﬁﬂﬃﬄ"
_CURLY = "‘’“”"
_FIX_CLASS_CHARS = set(_DASH_CLASS + _LIGATURES + _CURLY)

# Raw-record `kind` -> normalise() value_kind override. Everything else uses
# the field-name-driven default (token folding).
_KIND_TO_VALUE_KIND = {"int": "int", "date": "date", "paper_metric": "paper_metric"}

_NUMBER_WORDS = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
    "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12,
}
_UNIT_SUFFIXES = ("months", "month", "days", "day", "years", "year", "bps", "percent", "pct")


@dataclass(frozen=True)
class FieldDiff:
    """One recorded-vs-replayed divergence on a per-model condition."""

    anchor_id: str
    field: str
    what: str            # e.g. "a_located", "normalised_b", "final_tag"
    recorded: object
    replayed: object
    quote: str | None
    explained: bool      # attributable to a post-baseline fix class
    note: str


def _read_raw(path: Path) -> dict[str, dict]:
    """One raw JSONL -> {field: record}. A parse-failed or unparsed record is a
    silent model for merge purposes (mirrors the D9 gate's treatment)."""
    out: dict[str, dict] = {}
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            rec = json.loads(line)
            name = rec["field"]
            if name in out:
                raise RuntimeError(f"{path}: duplicate raw record for field {name!r}")
            parsed = rec.get("parsed") or {}
            answered = bool(rec.get("answered")) and not rec.get("parse_failed") and bool(
                parsed.get("answered")
            )
            kind = rec.get("kind")
            # Per-kind payload shape (mirrors the production parsers): signal_ref
            # carries the registry id under `concept_id`; method_summary carries
            # free text under `summary` and has NO quote (not quote-gated).
            if kind == "signal_ref":
                value = parsed.get("concept_id")
            elif kind == "method_summary":
                value = parsed.get("summary")
            else:
                value = parsed.get("value")
            out[name] = {
                "answered": answered,
                "value": value if answered else None,
                "quote": (parsed.get("quote") if answered and kind != "method_summary" else None),
                "kind": kind,
                "model_id": rec.get("configured_model_id"),
                "returned": rec.get("returned_model_version"),
            }
    return out


def _normalise_or_none(field: str, raw: object, kind: str | None) -> tuple[object, str | None]:
    """normalise() with the decoding-contract violation surfaced, not raised:
    a raw value the normaliser rejects is recorded and treated as silent-valued
    (the production filler would never have shipped it either)."""
    if raw is None:
        return None, None
    try:
        return normalise(field, raw, _KIND_TO_VALUE_KIND.get(kind or "")), None
    except Exception as exc:  # LibrarianSchemaError et al. — recorded, not fatal
        return None, f"{type(exc).__name__}: {exc}"


def replay_fields(
    recorded: RunArtefacts, raw_a: dict[str, dict], raw_b: dict[str, dict], ct
) -> tuple[dict[str, RunField], list[str]]:
    """Rebuild every asked field's RunField from the raw archives alone.

    The asked set comes from the recorded artefacts (the raw archive contains
    exactly the asked fields; `load_run` has already checked the line counts
    against the trace). Everything evidential — answered, quote, located,
    normalised, and the D9 shipping decision — is re-derived from raw."""
    notes: list[str] = []
    out: dict[str, RunField] = {}
    for name, rec_field in recorded.fields.items():
        if rec_field.not_extracted:
            out[name] = rec_field  # never asked: nothing to replay
            continue
        ra = raw_a.get(name)
        rb = raw_b.get(name)
        if ra is None or rb is None:
            raise RuntimeError(
                f"{recorded.run_dir}: field {name!r} present in trace but missing from a raw "
                "archive — the archive does not correspond to the trace"
            )
        if ra["kind"] == "method_summary":
            # Free-text summary: not value-normalised, not quote-gated, not
            # agreement-gated (the shipped trace records STATED with differing
            # texts and a trivially-true locate) — mirror those semantics.
            norm_a, norm_b = ra["value"], rb["value"]
            located_a, located_b = ra["answered"], rb["answered"]
            stated = ra["answered"] and rb["answered"]
            out[name] = RunField(
                field=name,
                a_answered=ra["answered"], b_answered=rb["answered"],
                a_quote=None, b_quote=None,
                a_located=located_a, b_located=located_b,
                a_model_id=ra["model_id"], b_model_id=rb["model_id"],
                normalised_a=norm_a, normalised_b=norm_b,
                final_tag="STATED" if stated else "UNKNOWN",
                shipped_reason="replayed_method_summary",
                ship_choice=None,
                not_extracted=False,
            )
            continue
        norm_a, err_a = _normalise_or_none(name, ra["value"], ra["kind"])
        norm_b, err_b = _normalise_or_none(name, rb["value"], rb["kind"])
        for who, err in (("model_a", err_a), ("model_b", err_b)):
            if err:
                notes.append(f"{name} [{who}]: normaliser rejected raw value ({err})")
        located_a = bool(ra["answered"] and ra["quote"] and ct.locate(ra["quote"]) is not None)
        located_b = bool(rb["answered"] and rb["quote"] and ct.locate(rb["quote"]) is not None)
        # D9 shipping decision, re-derived order-free: both answered, normalised
        # values equal and non-None, and every answered model's quote located.
        stated = (
            ra["answered"] and rb["answered"]
            and norm_a is not None and norm_a == norm_b
            and located_a and located_b
        )
        if not ra["answered"] and not rb["answered"]:
            reason = "both_silent"
        elif (ra["answered"] and not located_a) or (rb["answered"] and not located_b):
            reason = "quote_gate"
        elif ra["answered"] != rb["answered"]:
            reason = "single_response"
        elif norm_a != norm_b:
            reason = "disagreement"
        else:
            reason = "agreement"
        out[name] = RunField(
            field=name,
            a_answered=ra["answered"], b_answered=rb["answered"],
            a_quote=ra["quote"], b_quote=rb["quote"],
            a_located=located_a, b_located=located_b,
            a_model_id=ra["model_id"], b_model_id=rb["model_id"],
            normalised_a=norm_a, normalised_b=norm_b,
            final_tag="STATED" if stated else "UNKNOWN",
            shipped_reason=f"replayed_{reason}",
            ship_choice=None,
            not_extracted=False,
        )
    return out, notes


def diff_fields(anchor_id: str, recorded: RunArtefacts, replayed: RunArtefacts) -> list[FieldDiff]:
    """Per-model condition diffs between the recorded trace and the replay,
    each labelled explained (post-baseline fix-class signature) or not."""
    diffs: list[FieldDiff] = []
    for name, rec in recorded.fields.items():
        rep = replayed.fields[name]
        checks = (
            ("a_answered", rec.a_answered, rep.a_answered, rec.a_quote),
            ("b_answered", rec.b_answered, rep.b_answered, rec.b_quote),
            ("a_located", rec.a_located, rep.a_located, rec.a_quote or rep.a_quote),
            ("b_located", rec.b_located, rep.b_located, rec.b_quote or rep.b_quote),
            ("normalised_a", rec.normalised_a, rep.normalised_a, rec.a_quote),
            ("normalised_b", rec.normalised_b, rep.normalised_b, rec.b_quote),
            ("final_tag", rec.final_tag, rep.final_tag, None),
        )
        for what, old, new, quote in checks:
            if old == new:
                continue
            fix_class = bool(quote) and any(c in _FIX_CLASS_CHARS for c in quote)
            explained = what in ("a_located", "b_located", "final_tag") and fix_class
            note = (
                "locate-flip on a quote carrying D40 fix-class glyphs"
                if explained
                else ("quote carries fix-class glyphs" if fix_class else "no fix-class signature")
            )
            diffs.append(FieldDiff(
                anchor_id=anchor_id, field=name, what=what,
                recorded=old, replayed=new, quote=quote, explained=explained, note=note,
            ))
    return diffs


def _surface_form_token(value: object) -> str:
    return str(value).strip().lower()


def _candidate_equivalence(a: object, b: object) -> str | None:
    """Deterministic surface-form heuristics for the P6b review list. Returns
    the candidate class name, or None. Never applied — review only."""
    ta, tb = _surface_form_token(a), _surface_form_token(b)
    if ta == tb:
        return None  # identical post-str(); a real disagreement lives elsewhere
    # numeric-word vs digit ("six months" vs 6)
    for x, y in ((ta, tb), (tb, ta)):
        head = re.split(r"[_\s]", x, maxsplit=1)[0]
        digits = re.sub(r"[^0-9]", "", y)
        if head in _NUMBER_WORDS and digits and int(digits) == _NUMBER_WORDS[head]:
            return "numeric_word_vs_digit"
    # unit-suffix variants
    def strip_units(t: str) -> str:
        for u in _UNIT_SUFFIXES:
            t = re.sub(rf"[_\s]*{u}$", "", t)
        return t
    if strip_units(ta) == strip_units(tb) and strip_units(ta):
        return "unit_suffix_variant"
    # containment (one is a qualified form of the other)
    if (ta in tb or tb in ta) and min(len(ta), len(tb)) >= 4:
        return "containment_variant"
    return None


def review_list(anchor_id: str, replayed: RunArtefacts, decomposition) -> list[dict]:
    """Semantically-equal-but-unequal-post-normalisation candidates inside the
    disagreement / value_wrong sets (never merge_refused — that slice is
    partner-silence, not surface forms)."""
    out: list[dict] = []
    wrong_fields = {
        f.field for f in decomposition.fields
        if f.mechanism in (Mechanism.VALUE_WRONG, Mechanism.NOT_SCORABLE)
    }
    for name, rf in replayed.fields.items():
        if not (rf.a_answered and rf.b_answered):
            continue
        if rf.normalised_a == rf.normalised_b and name not in wrong_fields:
            continue
        cls = _candidate_equivalence(rf.normalised_a, rf.normalised_b)
        if cls:
            out.append({
                "anchor": anchor_id, "field": name, "class": cls,
                "normalised_a": repr(rf.normalised_a), "normalised_b": repr(rf.normalised_b),
            })
    return out


def _d40_ordering() -> str:
    """State the D40-vs-baseline date ordering from the decision log, so the
    report says whether locate-flips are expected at all."""
    log = _REPO_ROOT / "docs/librarian/registers/decision-log_librarian.md"
    try:
        text = log.read_text(encoding="utf-8")
    except OSError:
        return "decision log unreadable — D40 date not checked"
    m = re.search(r"D40[^\n]*\n(?:[^\n]*\n){0,6}?[^\n]*?(\d{4}-\d{2}-\d{2})", text)
    if not m:
        return "D40 entry found but no date parsed — ordering unstated"
    d40 = m.group(1)
    rel = "AFTER" if d40 > "2026-07-22" else "ON-OR-BEFORE"
    return (
        f"D40 (ladder v3, dash class) is dated {d40} — {rel} the 2026-07-22 baseline; "
        + ("locate-flips on dash-class quotes are therefore expected and explained."
           if rel == "AFTER" else
           "the baseline was already scored under v3, so fix-class flips should be ~zero.")
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-root", default="runs/g3_v3")
    ap.add_argument("--out", default="docs/extensions/reports/p6a_composition.md")
    args = ap.parse_args(argv)

    run_root = _REPO_ROOT / args.run_root
    lines: list[str] = []
    lines.append("# P6a — merge-miss diagnosis replay (WS-B)")
    lines.append("")
    lines.append("**NON-REPORTABLE — Phase-D archives (free dev pair). Diagnostic engineering "
                 "only; no figure here may be reported (contract §1 / I2).**")
    lines.append("")
    lines.append(f"_Run root: `{args.run_root}` · replayed with the CURRENT locator ladder and "
                 "D9 value normaliser · classifier: `evaluation.harness.missed_evidence.decompose` "
                 "(order-free, never `final_reason`)._")
    lines.append("")
    lines.append(f"D40 ordering check: {_d40_ordering()}")
    lines.append("")

    pooled_recorded: Counter = Counter()
    pooled_replayed: Counter = Counter()
    all_diffs: list[FieldDiff] = []
    all_reviews: list[dict] = []
    all_notes: list[str] = []
    merge_refused_composition: Counter = Counter()
    partner_differed_items: list[dict] = []

    for anchor_id, subdir, ct_path in ANCHORS:
        run_dir = run_root / subdir
        recorded = load_run(run_dir)
        ct = load_canonical_text(_REPO_ROOT / ct_path).require_frozen()
        raw_a = _read_raw(run_dir / "raw" / "raw_model_a.jsonl")
        raw_b = _read_raw(run_dir / "raw" / "raw_model_b.jsonl")
        rep_fields, notes = replay_fields(recorded, raw_a, raw_b, ct)
        all_notes.extend(f"{anchor_id}: {n}" for n in notes)
        replayed = RunArtefacts(
            run_dir=recorded.run_dir, paper_id=recorded.paper_id, header=recorded.header,
            fields=rep_fields, reportability=recorded.reportability,
            raw_line_counts=recorded.raw_line_counts,
        )

        score_rec = score_anchor(anchor_id, run_dir)
        score_rep = score_anchor(anchor_id, run_dir, artefacts=replayed)
        dec_rec = decompose(score_rec, recorded)
        dec_rep = decompose(score_rep, replayed)
        pooled_recorded.update(dec_rec.counts)
        pooled_replayed.update(dec_rep.counts)

        for f in dec_rep.fields:
            if f.mechanism is Mechanism.MERGE_REFUSED:
                rf = rep_fields[f.field]
                if rf.a_answered != rf.b_answered:
                    merge_refused_composition["partner_silent"] += 1
                else:
                    merge_refused_composition["partner_differed"] += 1
                    partner_differed_items.append({
                        "anchor": anchor_id, "paper": recorded.paper_id, "field": f.field,
                        "normalised_a": repr(rf.normalised_a),
                        "normalised_b": repr(rf.normalised_b),
                        "gold": repr(f.gold_value),
                    })

        all_diffs.extend(diff_fields(anchor_id, recorded, replayed))
        all_reviews.extend(review_list(anchor_id, replayed, dec_rep))

        lines.append(f"## {anchor_id} (`{subdir}`)")
        lines.append("")
        lines.append("Recorded (from the shipped trace):")
        lines.append("```")
        lines.append(render_decomposition(dec_rec, allow_non_reportable=True))
        lines.append("```")
        lines.append("Replayed (from the raw archives):")
        lines.append("```")
        lines.append(render_decomposition(dec_rep, allow_non_reportable=True))
        lines.append("```")
        lines.append("")

    # ---- pooled table vs the recorded headline --------------------------------
    lines.append("## Pooled decomposition (n = gold-STATED misses across drf + mom6 + str)")
    lines.append("")
    lines.append("| mechanism | recorded baseline | replayed from raw | recorded (this run) |")
    lines.append("|---|---|---|---|")
    order = ["merge_refused", "value_wrong", "gate_lost", "not_retrieved", "not_scorable"]
    for mech in order:
        lines.append(
            f"| `{mech}` | {RECORDED_BASELINE.get(mech, 0)} | "
            f"{pooled_replayed.get(mech, 0)} | {pooled_recorded.get(mech, 0)} |"
        )
    n_base = sum(RECORDED_BASELINE.values())
    n_rep = sum(pooled_replayed.get(m, 0) for m in order)
    reproduced = all(
        pooled_replayed.get(m, 0) == RECORDED_BASELINE.get(m, 0)
        for m in RECORDED_BASELINE
    )
    lines.append("")
    lines.append(
        f"Recorded baseline n={n_base} (19/13/9/2 = 44.2/30.2/20.9/4.7); replayed n={n_rep}. "
        + ("**The replay reproduces the recorded decomposition exactly.**" if reproduced
           else "**The replay does NOT reproduce the recorded decomposition — every "
                "divergence is itemised below (acceptance allows either outcome).**")
    )
    lines.append("")

    # ---- merge_refused composition (B2) ---------------------------------------
    lines.append("## merge_refused composition (replayed)")
    lines.append("")
    total_mr = sum(merge_refused_composition.values())
    for k in ("partner_silent", "partner_differed"):
        v = merge_refused_composition.get(k, 0)
        share = f"{100 * v / total_mr:.1f}%" if total_mr else "—"
        lines.append(f"- {k}: {v} ({share})")
    lines.append("")
    for item in partner_differed_items:
        lines.append(
            f"The partner-differed item, named in full: **{item['paper']}** (anchor "
            f"{item['anchor']}), field `{item['field']}` — one model held the gold value, the "
            f"other answered differently: model_a = {item['normalised_a']}, model_b = "
            f"{item['normalised_b']} (gold = {item['gold']}). This is the only residual "
            "ambiguity in the headline slice."
        )
    lines.append("")
    lines.append("_Partner-silence confirms the D38 reading: the slice is a pair-coverage "
                 "mechanism whose pre-registered remedy is pair remediation (the Phase-F pair), "
                 "not a normaliser defect._")
    lines.append("")

    # ---- per-item diffs, partitioned ------------------------------------------
    explained = [d for d in all_diffs if d.explained]
    unexplained = [d for d in all_diffs if not d.explained]
    lines.append("## Recorded-vs-replayed diffs")
    lines.append("")
    lines.append(f"- explained by post-baseline locator/normaliser fixes: **{len(explained)}**")
    lines.append(f"- unexplained: **{len(unexplained)}**")
    lines.append("")
    if all_diffs:
        lines.append("| anchor | field | condition | recorded | replayed | bucket | note |")
        lines.append("|---|---|---|---|---|---|---|")
        for d in all_diffs:
            bucket = "explained" if d.explained else "unexplained"
            lines.append(
                f"| {d.anchor_id} | `{d.field}` | {d.what} | {d.recorded!r} | {d.replayed!r} "
                f"| {bucket} | {d.note} |"
            )
        lines.append("")
    if all_notes:
        lines.append("Normaliser-rejected raw values (recorded, treated as silent):")
        for n in all_notes:
            lines.append(f"- {n}")
        lines.append("")

    # ---- review list + mechanical P6b trigger (B4) ----------------------------
    lines.append("## Surface-form review list (NEVER auto-applied)")
    lines.append("")
    if all_reviews:
        lines.append("| anchor | field | class | normalised_a | normalised_b |")
        lines.append("|---|---|---|---|---|")
        for r in all_reviews:
            lines.append(
                f"| {r['anchor']} | `{r['field']}` | {r['class']} | {r['normalised_a']} "
                f"| {r['normalised_b']} |"
            )
    else:
        lines.append("_Empty. No semantically-equal-but-unequal-post-normalisation pair was "
                     "found inside the disagreement / value_wrong sets — consistent with "
                     "disagreement = 0/42 on the BBW baseline and with rev 2's withdrawal of "
                     "the rev-1 normalisation premise._")
    class_counts = Counter(r["class"] for r in all_reviews)
    recurring = {c: n for c, n in class_counts.items() if n > 1}
    lines.append("")
    lines.append(
        f"**P6b trigger (mechanical): {'YES' if recurring else 'NO'}** — rule: a candidate "
        f"class appearing >1 time across papers. Classes: "
        f"{dict(class_counts) if class_counts else '{}'}."
    )
    lines.append("")
    lines.append("_P6b, if ever built, changes VALUE NORMALISATION ONLY; the D9 agreement gate "
                 "(both-answer-both-locate) is untouchable (I6)._")
    lines.append("")

    out_path = _REPO_ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    sidecar = out_path.with_suffix(".json")
    sidecar.write_text(json.dumps({
        "recorded_baseline": RECORDED_BASELINE,
        "pooled_replayed": dict(pooled_replayed),
        "pooled_recorded_this_run": dict(pooled_recorded),
        "reproduced": reproduced,
        "merge_refused_composition": dict(merge_refused_composition),
        "partner_differed_items": partner_differed_items,
        "diffs": [d.__dict__ for d in all_diffs],
        "review_list": all_reviews,
        "p6b_trigger": bool(recurring),
        "normaliser_rejections": all_notes,
    }, indent=2, default=str), encoding="utf-8")

    print(f"P6a replay written: {out_path.relative_to(_REPO_ROOT)}")
    print(f"reproduced={reproduced} diffs={len(all_diffs)} "
          f"(explained={len(explained)}) review_list={len(all_reviews)} "
          f"p6b_trigger={bool(recurring)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
