#!/usr/bin/env python
"""
Integrated end-to-end traversal of the five-agent pipeline (2026-09-07).

One deterministic driver that walks every registered non-reject paper through
the pipeline's seams and records the TYPED outcome at each stage — the success
bar is "each stage yields its designed typed outcome", never "every paper
executes" (a live-extracted spec that refuses at the adapter on UNKNOWN density
is the system working: the bright line forbids overriding an UNKNOWN).

Stages per paper (anchor-only stages are `not_applicable` for scale papers):
  extraction  — the recorded Phase-F run's outcome (ok / review / zero_specs;
                the pinned label), spec count asserted against the pinned denominator
  compile     — every pinned spec through spec_from_dict -> adapt_spec (hash-
                verified standing subs) -> typed refusal row OR run_strategy on
                the corrected() dev panel -> executed row; events.json rows
                surface as typed not_run rows (mirrors run_t3_coverage)
  gold run    — the three anchors' gold-spec compile+run via run_quant.run_all
                into a fresh dir, compared field-exact against the recorded quant run
  gate12      — §7 gates 1-2 (rulebook byte-equality) via gate12_verdict,
                compared against the recorded results/g2_gates.json
  audit       — fresh full-lattice reports compared against the recorded audit run
                by recursive subset-equality (additive keys allowed + listed)
  rq4 entry   — the 3-condition entry rule via run_rq4_funnel.build_case on the
                FRESH audit reports (str enters; drf/mom6 typed not-entered)
  funnel      — the fresh replay byte-compared against the recorded artifact
  reporter    — check_run over the recorded pointers (writes nothing)

Zero LLM calls in this driver. The paid extraction, the funnel replay and the
audit run are separate explicit commands; this driver only consumes their
output dirs (--fresh-drr-dir / --funnel-file / --audit-dir).

All writes land strictly under --out. The recorded artefacts this driver
compares against are read-only inputs; a comparison failure is a RECORDED
FINDING (exit 2), never something to iterate away silently.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from agents.librarian.adapter.adapt import adapt_spec  # noqa: E402
from agents.librarian.pipeline.spec_loader import spec_from_dict  # noqa: E402
from agents.quant.config.runner import StrategyResult, run_strategy  # noqa: E402
from agents.quant.library.run_config import corrected  # noqa: E402
from evaluation.harness.round_trip import gate12_verdict  # noqa: E402
from scripts.run_quant import load_inputs, run_all, summarize_run  # noqa: E402
from scripts.run_t3_coverage import refusal_codes_of  # noqa: E402
from shared.licensed_inputs import require_licensed_input  # noqa: E402


# --------------------------------------------------------------------------
# The pinned 12-paper denominator (frozen; one line per registered non-reject
# paper). `outcome` is the pinned expected label for the run (not re-derived
# here); `n_specs` is asserted against the run dir so the denominator cannot
# shrink silently. The reject_* adversarial papers are out of scope (no recorded
# specs; their refusal behaviour is separately recorded RQ2 machinery).
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class PaperPin:
    paper_id: str
    run_dir: str            # recorded extraction run dir, repo-relative
    outcome: str            # "ok" | "review" | "zero_specs"
    n_specs: int
    note: str = ""


PINNED_SPECS: dict[str, PaperPin] = {
    "drr": PaperPin("DRR_2026", "runs/corpus_anchors_report/drr", "ok", 1),
    "bbw": PaperPin("BBW_2019", "runs/corpus_anchors_report/bbw", "ok", 1),
    "jnps": PaperPin("JNPS_2013", "runs/corpus_anchors_report/jnps", "ok", 1),
    "bbw2021": PaperPin("BBW_2021", "runs/corpus_corpus_report/bbw2021", "ok", 5,
                        "run of record for the RQ2 denominator (2026-09-03)"),
    "dfps": PaperPin("DFPS_2026", "runs/corpus_scoped/dfps", "ok", 21,
                     "scoped re-run; events.json carries the per-construction "
                     "assembly reviews (CI-9/CI-10)"),
    "kpp": PaperPin("KPP_2023", "runs/kpp_report", "ok", 1,
                    "estimation family; scored by run_kpp_score, not run"),
    "klz": PaperPin("KLZ_2017", "runs/corpus_prospective_report/klz", "ok", 1),
    "bektic": PaperPin("BEKTIC_2018", "runs/corpus_prospective_report/bektic", "ok", 1),
    "hvz": PaperPin("HVZ_2017", "runs/corpus_prospective_report/hvz", "review", 0),
    "cgnst": PaperPin("CGNST_2017", "runs/corpus_prospective_report/cgnst", "review", 0),
    "bwwss": PaperPin("BWWSS_2019", "runs/corpus_prospective_report/bwwss", "review", 0),
    "synth": PaperPin("SYNTH_2026", "runs/corpus_synth_report/synth", "zero_specs", 0,
                      "exit 3 / zero_specs is the recorded terminal outcome "
                      "(corpus_log.json), not exit 2 / review"),
}

# Adapter refusal multisets pinned for two anchors (DRR, KPP) as a regression bar;
# every other paper's compile result is recorded, not pinned.
EXPECTED_REFUSAL_PINS: dict[str, dict[str, int]] = {
    "drr": {"REVIEW_REQUIRED": 15},
    "kpp": {"REFUSED_ON_SILENCE": 2},
}

ANCHOR_STRATEGIES = ("str", "drf", "mom6")
GATE12_ANCHORS = ("str", "drf", "mom6", "crf")
PAPER_OF_STRATEGY = {"str": "drr", "drf": "bbw", "mom6": "jnps", "crf": "bbw"}

RECORDED_QUANT = "results/quant/recorded"
RECORDED_AUDIT = "results/auditor/recorded"
RECORDED_G2 = "results/g2_gates.json"
RECORDED_FUNNEL = "results/scientist/rq4_funnel/rq4_funnel_reported.json"

# run_log keys that legitimately differ between the recorded runs and a fresh
# run under the current tree (identity metadata + additive-schema hashes).
QUANT_RUN_LOG_ALLOWED = frozenset({"git_commit", "git_short", "thresholds_sha256"})
AUDIT_RUN_LOG_ALLOWED = frozenset(
    {"git_commit", "thresholds_auditor_hash", "baseline_signatures"})


def _jsonable(v: object) -> object:
    if isinstance(v, (str, int, float, bool)) or v is None:
        return v
    return str(v)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def _leaf_equal(a: object, b: object) -> bool:
    if isinstance(a, float) and isinstance(b, float):
        if math.isnan(a) and math.isnan(b):
            return True
        return a == b
    return a == b


# --------------------------------------------------------------------------
# Stage cores (each pure enough to unit-test with injected stubs / tmp dirs).
# --------------------------------------------------------------------------

def extraction_record(key: str, pin: PaperPin, repo_root: Path = _REPO_ROOT) -> dict:
    """The recorded extraction run's spec count, asserted against the pin so the
    traversal denominator cannot drift; the outcome is the pinned expected label
    (not re-derived from the run manifest here)."""
    run_dir = require_licensed_input(
        repo_root / pin.run_dir, f"{key} extraction run dir (local pipeline output, not shipped with the repository)")
    n_found = len(sorted(run_dir.glob("spec_*.json")))
    if n_found != pin.n_specs:
        raise RuntimeError(
            f"{key}: {n_found} spec_*.json under {pin.run_dir}, pinned "
            f"{pin.n_specs} -- the pinned denominator no longer matches the "
            "run dir; re-pin explicitly, never score a drifted set")
    row = {"paper": key, "paper_id": pin.paper_id, "run_dir": pin.run_dir,
           "outcome": pin.outcome, "n_specs": n_found, "note": pin.note}
    manifest = run_dir / "run_manifest.json"
    if manifest.exists():
        m = _load(manifest)
        prof = m.get("operational_profile", {})
        row["operational_profile"] = {
            "model_calls": prof.get("model_calls"),
            "wall_clock_seconds": prof.get("wall_clock_seconds"),
            "tokens": prof.get("tokens"),
        }
    return row


def compile_paper(key: str, pin: PaperPin, panel, subs, *,
                  artefact_dir: Path | None = None, repo_root: Path = _REPO_ROOT,
                  adapt=adapt_spec, run=run_strategy) -> tuple[list[dict], list[dict]]:
    """Every pinned spec through the adapter; typed refusal / executed rows,
    plus typed not_run rows for events.json assembly reviews. Mirrors
    run_t3_coverage.observed_rows; additionally writes per-spec adapt
    artefacts (refusal codes + AdaptResult.to_dict) for later Reporter
    config_refusal pointers."""
    run_dir = repo_root / pin.run_dir
    rows: list[dict] = []
    events: list[dict] = []
    for spec_path in sorted(run_dir.glob("spec_*.json")):
        spec = spec_from_dict(_load(spec_path))
        name = spec.header.strategy_label.value
        result = adapt(spec, standing_subs=subs)
        codes = refusal_codes_of(result) if result.refused else []
        if artefact_dir is not None:
            adir = artefact_dir / key
            adir.mkdir(parents=True, exist_ok=True)
            (adir / f"{spec_path.stem}_adapt.json").write_text(json.dumps(
                {"spec": spec_path.name, "name": name, "refused": result.refused,
                 "refusal_codes": codes, "adapt_result": result.to_dict()},
                indent=2, default=str), encoding="utf-8")
        if result.refused:
            rows.append({"spec": spec_path.name, "name": name,
                         "outcome": "refused", "refusal_codes": codes})
            continue
        run_result = run(result, panel)
        if not isinstance(run_result, StrategyResult):
            raise RuntimeError(
                f"{key}/{spec_path.name}: run_strategy returned "
                f"{type(run_result).__name__} for a non-refused compile")
        rows.append({"spec": spec_path.name, "name": name, "outcome": "executed",
                     "refusal_codes": [], "summary": summarize_run(run_result)})
    events_path = run_dir / "events.json"
    if events_path.exists():
        for ev in _load(events_path):
            if ev.get("kind") == "assembly_incomplete":
                rows.append({"spec": None, "name": ev.get("construction_name"),
                             "outcome": "not_run",
                             "extraction_event": "assembly_incomplete"})
            else:
                events.append(ev)
    return rows, events


def refusal_multiset(rows: list[dict]) -> dict[str, int]:
    out: dict[str, int] = {}
    for r in rows:
        for c in r.get("refusal_codes", []):
            out[c] = out.get(c, 0) + 1
    return out


def spec_field_diff(dict_a: dict, dict_b: dict) -> dict:
    """Field-level diff of two SERIALISED specs over every {tag, value} leaf.

    Walks the raw to_dict form, so no curated field list can hide a change
    (the gold-field walker misses e.g. claimed_headline_metric and the leg
    sort fields -- verified 2026-09-07 when it reported 0 diffs on specs the
    adapter refused differently). Evidence/provenance churn under an
    unchanged tag+value is counted separately; header run-identity strings
    (plain non-tagged leaves) are ignored. A divergence is a recorded finding
    about dual-LLM run-to-run variation, never a failure."""
    rows: list[dict] = []
    counts = {"n_fields": 0, "n_evidence_only": 0}

    def walk(a, b, path):
        if isinstance(a, dict) and isinstance(b, dict) and ("tag" in a or "tag" in b):
            counts["n_fields"] += 1
            tag_a, tag_b = a.get("tag"), b.get("tag")
            va, vb = a.get("value"), b.get("value")
            value_equal = bool(va == vb) if (tag_a == "STATED" and tag_b == "STATED") else None
            if tag_a != tag_b or value_equal is False:
                rows.append({"path": path, "tag_a": tag_a, "tag_b": tag_b,
                             "tag_changed": tag_a != tag_b, "value_equal": value_equal,
                             "value_a": _jsonable(va), "value_b": _jsonable(vb)})
            elif a.get("evidence") != b.get("evidence"):
                counts["n_evidence_only"] += 1
            return
        if isinstance(a, dict) and isinstance(b, dict):
            for k in sorted(set(a) | set(b)):
                walk(a.get(k), b.get(k), f"{path}.{k}")
        elif isinstance(a, list) and isinstance(b, list):
            for i in range(max(len(a), len(b))):
                walk(a[i] if i < len(a) else None,
                     b[i] if i < len(b) else None, f"{path}[{i}]")
        # plain non-tagged leaves (header run identity, hashes) are ignored

    walk(dict_a, dict_b, "")
    return {
        "n_fields": counts["n_fields"],
        "n_evidence_only": counts["n_evidence_only"],
        "n_tag_changed": sum(1 for r in rows if r["tag_changed"]),
        "n_stated_value_diff": sum(1 for r in rows if r["value_equal"] is False),
        "rows": rows,
    }


def entry_rows(audit_dir: Path, *, build_case_fn=None) -> dict[str, dict]:
    """The 3-condition RQ4 entry rule for all three anchors against the FRESH
    audit reports, via the funnel's own case builder (pure reuse). Note:
    build_case stamps corrected_run_ref (per-anchor, e.g. 'str_corrected')
    internally -- metadata only, never read by the rule; the row records the actual path."""
    if build_case_fn is None:
        from scripts.run_rq4_funnel import build_case as build_case_fn  # lazy: scientist stack
    out: dict[str, dict] = {}
    for a in ANCHOR_STRATEGIES:
        report = audit_dir / f"{a}_report.json"
        case, params = build_case_fn(
            report, strategy_id=a, case_id=f"e2e_{a}",
            corrected_quant_config_ref=f"qc_{a}_corrected")
        out[a] = {"entered": bool(case.failed_check_ids),
                  "failed_check_ids": sorted(case.failed_check_ids),
                  "theta": params.theta, "q": params.q,
                  "audit_report": str(report)}
    return out


# --------------------------------------------------------------------------
# Comparators (fresh vs recorded). Empirically calibrated: shared numeric
# leaves are bit-identical across the recorded runs; drift is a finding.
# --------------------------------------------------------------------------

def subset_mismatches(recorded, fresh, path: str = "") -> tuple[list[str], list[str]]:
    """Recursive subset-equality: every leaf present in `recorded` must exist
    in `fresh` with an exactly equal value; fresh-only keys are additive
    (allowed, listed). Lists are leaves-with-structure: same length, each
    element compared in place."""
    mismatches: list[str] = []
    additive: list[str] = []
    if isinstance(recorded, dict) and isinstance(fresh, dict):
        for k, v in recorded.items():
            if k not in fresh:
                mismatches.append(f"{path}.{k}: missing in fresh")
            else:
                m, a = subset_mismatches(v, fresh[k], f"{path}.{k}")
                mismatches += m
                additive += a
        additive += [f"{path}.{k}" for k in fresh if k not in recorded]
    elif isinstance(recorded, list) and isinstance(fresh, list):
        if len(recorded) != len(fresh):
            mismatches.append(f"{path}: list length {len(recorded)} != {len(fresh)}")
        else:
            for i, (rv, fv) in enumerate(zip(recorded, fresh)):
                m, a = subset_mismatches(rv, fv, f"{path}[{i}]")
                mismatches += m
                additive += a
    elif not _leaf_equal(recorded, fresh):
        mismatches.append(f"{path}: {recorded!r} != {fresh!r}")
    return mismatches, additive


def _compare_run_log(fresh: dict, recorded: dict, allowed: frozenset) -> dict:
    mismatches, informational = [], {}
    for k in sorted(set(fresh) | set(recorded)):
        if k in allowed:
            if fresh.get(k) != recorded.get(k):
                informational[k] = {"recorded": _jsonable(recorded.get(k)),
                                    "fresh": _jsonable(fresh.get(k))}
            continue
        if k not in recorded:
            informational[k] = {"recorded": None, "fresh": "additive"}
        elif k not in fresh or not _leaf_equal(recorded[k], fresh[k]):
            mismatches.append(f"run_log.{k}")
    return {"pass": not mismatches, "mismatches": mismatches,
            "allowed_diffs": informational}


def compare_quant(fresh_dir: Path, recorded_dir: Path) -> dict:
    """Per-anchor typed record fields + summary exactly equal; coverage fully
    equal; run_log equal outside the identity-metadata allowlist."""
    keys = ("status", "strategy_label", "variant", "n_legs", "combiner", "summary")
    mismatches = []
    for a in ANCHOR_STRATEGIES:
        fresh, recorded = _load(fresh_dir / f"{a}.json"), _load(recorded_dir / f"{a}.json")
        for k in keys:
            if not _leaf_equal(recorded.get(k), fresh.get(k)):
                mismatches.append(f"{a}.{k}: {recorded.get(k)!r} != {fresh.get(k)!r}")
    m, _ = subset_mismatches(_load(recorded_dir / "coverage.json"),
                             _load(fresh_dir / "coverage.json"), "coverage")
    mismatches += m
    log = _compare_run_log(_load(fresh_dir / "run_log.json"),
                           _load(recorded_dir / "run_log.json"), QUANT_RUN_LOG_ALLOWED)
    return {"pass": not mismatches and log["pass"],
            "mismatches": mismatches + log["mismatches"],
            "run_log_allowed_diffs": log["allowed_diffs"]}


def compare_audit(fresh_dir: Path, recorded_dir: Path) -> dict:
    """Recorded report ⊆ fresh report per anchor (additive keys allowed and
    listed); numeric_verification subset-equal; run_log allowlist as above."""
    mismatches, additive = [], {}
    for name in [f"{a}_report.json" for a in ANCHOR_STRATEGIES] + ["numeric_verification.json"]:
        m, a = subset_mismatches(_load(recorded_dir / name), _load(fresh_dir / name), name)
        mismatches += m
        if a:
            additive[name] = a
    log = _compare_run_log(_load(fresh_dir / "run_log.json"),
                           _load(recorded_dir / "run_log.json"), AUDIT_RUN_LOG_ALLOWED)
    return {"pass": not mismatches and log["pass"],
            "mismatches": mismatches + log["mismatches"],
            "additive_keys": additive, "run_log_allowed_diffs": log["allowed_diffs"]}


def compare_funnel(fresh_file: Path, recorded_file: Path) -> dict:
    fresh, recorded = _sha256(fresh_file), _sha256(recorded_file)
    return {"pass": fresh == recorded, "fresh_sha256": fresh,
            "recorded_sha256": recorded}


def compare_gate12(fresh: dict, recorded_file: Path) -> dict:
    recorded = _load(recorded_file)["anchors"]
    m, _ = subset_mismatches(recorded, fresh, "gate12")
    all_pass = all(v.get("pass") is True for v in fresh.values())
    return {"pass": not m and all_pass, "mismatches": m,
            "all_anchor_gates_pass": all_pass}


# --------------------------------------------------------------------------
# Matrix + rendering.
# --------------------------------------------------------------------------

def build_matrix(extractions: dict, compile_rows: dict, quant_records: dict | None,
                 gate12: dict | None, audit_ok: dict | None, entry: dict | None,
                 funnel_ok: bool | None, reporter: dict) -> list[dict]:
    strategies_of = {}
    for s, p in PAPER_OF_STRATEGY.items():
        strategies_of.setdefault(p, []).append(s)
    rows = []
    for key, pin in PINNED_SPECS.items():
        crows = compile_rows.get(key, [])
        counts = {o: sum(1 for r in crows if r["outcome"] == o)
                  for o in ("refused", "executed", "not_run")}
        row = {"paper": key, "paper_id": pin.paper_id,
               "extraction": extractions[key]["outcome"],
               "compile": ("no_specs (terminal at extraction)" if not crows else
                           ", ".join(f"{v} {k}" for k, v in counts.items() if v))}
        anchors = strategies_of.get(key, [])
        if anchors:
            runnable = [a for a in anchors if a in ANCHOR_STRATEGIES]
            row["gold_execution"] = (
                "pending" if quant_records is None else
                ", ".join(f"{a}: {quant_records[a]['status']}" for a in runnable))
            row["gate12"] = (
                "pending" if gate12 is None else
                ", ".join(f"{a}: {'pass' if gate12[a]['pass'] else 'FAIL'}"
                          for a in anchors))
            row["audit"] = (
                "pending" if audit_ok is None else
                ", ".join(f"{a}: {'report' if audit_ok.get(a) else 'MISSING'}"
                          for a in runnable))
            row["rq4_entry"] = (
                "pending" if entry is None else
                ", ".join(f"{a}: {'entered' if entry[a]['entered'] else 'not entered (typed)'}"
                          for a in runnable))
            row["funnel"] = ("n/a" if "str" not in runnable else
                             "pending" if funnel_ok is None else
                             "byte-identical replay" if funnel_ok else "DIVERGED")
            row["reporter"] = ", ".join(
                f"{a}: {reporter.get(a, 'unobserved (pointer absent)')}"
                for a in runnable)
        else:
            for stage in ("gold_execution", "gate12", "audit", "rq4_entry", "funnel"):
                row[stage] = "not_applicable"
            row["reporter"] = "unobserved (pointer absent)"
        rows.append(row)
    return rows


def render_summary_md(matrix: list[dict], comparisons: dict, drr_diff: dict | None,
                      kpp: dict | None) -> str:
    cols = ("paper", "extraction", "compile", "gold_execution", "gate12",
            "audit", "rq4_entry", "funnel", "reporter")
    lines = ["# Integrated end-to-end traversal", "",
             "Success bar: a designed TYPED outcome at every seam. Adapter",
             "refusals on UNKNOWN density are passes (the bright line forbids",
             "overriding an UNKNOWN); anchors additionally carry the Path-B",
             "join to the gold-spec execution.", "",
             "| " + " | ".join(cols) + " |",
             "|" + "|".join("---" for _ in cols) + "|"]
    for r in matrix:
        lines.append("| " + " | ".join(str(r.get(c, "")) for c in cols) + " |")
    lines += ["", "## Fresh-vs-recorded comparisons", ""]
    for name, c in comparisons.items():
        verdict = "PASS" if c.get("pass") else "FAIL (recorded finding)"
        lines.append(f"- **{name}**: {verdict}")
        for m in c.get("mismatches", [])[:10]:
            lines.append(f"  - {m}")
    if kpp is not None:
        lines += ["", f"## kpp estimation score: exit {kpp['exit']} -> {kpp['out']}"]
    if drr_diff is not None:
        lines += ["", "## Fresh drr extraction vs recorded spec_0.json",
                  "",
                  f"{drr_diff['n_fields']} tagged fields walked; "
                  f"{drr_diff['n_tag_changed']} tag changes; "
                  f"{drr_diff['n_stated_value_diff']} STATED-value differences; "
                  f"{drr_diff.get('n_evidence_only', 0)} evidence-only changes. "
                  "Divergence is a recorded dual-LLM variation finding, not a failure."]
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------
# Orchestration.
# --------------------------------------------------------------------------

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Integrated end-to-end pipeline traversal.")
    ap.add_argument("--out", default="runs/e2e")
    ap.add_argument("--fresh-drr-dir", default=None,
                    help="fresh Phase-F drr extraction dir (spec_0.json inside)")
    ap.add_argument("--funnel-file", default=None,
                    help="fresh funnel replay output (rq4_funnel_reported.json)")
    ap.add_argument("--audit-dir", default=None,
                    help="fresh full-audit dir with <anchor>_report.json files")
    args = ap.parse_args(argv)

    out = _REPO_ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    pending: list[str] = []
    comparisons: dict[str, dict] = {}

    panel, subs = load_inputs(corrected())

    # 1. extraction records + compile every pinned spec.
    extractions = {k: extraction_record(k, p) for k, p in PINNED_SPECS.items()}
    compile_rows, other_events, refusal_pin_checks = {}, {}, {}
    for key, pin in PINNED_SPECS.items():
        rows, events = compile_paper(key, pin, panel, subs, artefact_dir=out / "compile")
        compile_rows[key], other_events[key] = rows, events
        if key in EXPECTED_REFUSAL_PINS:
            got = refusal_multiset(rows)
            refusal_pin_checks[key] = {"expected": EXPECTED_REFUSAL_PINS[key],
                                       "got": got,
                                       "pass": got == EXPECTED_REFUSAL_PINS[key]}
    comparisons["adapter_refusal_pins"] = {
        "pass": all(c["pass"] for c in refusal_pin_checks.values()),
        "mismatches": [f"{k}: expected {c['expected']}, got {c['got']}"
                       for k, c in refusal_pin_checks.items() if not c["pass"]]}

    # 2. fresh drr extraction (if present): compile + field diff vs recorded.
    drr_diff = None
    if args.fresh_drr_dir:
        fresh_dir = _REPO_ROOT / args.fresh_drr_dir
        fresh_specs = sorted(fresh_dir.glob("spec_*.json"))
        if not fresh_specs:
            pending.append("fresh_drr (no spec_*.json yet)")
        else:
            fresh_pin = PaperPin("DRR_2026", str(fresh_dir.relative_to(_REPO_ROOT)),
                                 "ok", len(fresh_specs), "fresh in-session re-extraction")
            rows, _ = compile_paper("drr_fresh", fresh_pin, panel, subs,
                                    artefact_dir=out / "compile")
            compile_rows["drr_fresh"] = rows
            drr_diff = spec_field_diff(
                _load(fresh_specs[0]),
                _load(_REPO_ROOT / PINNED_SPECS["drr"].run_dir / "spec_0.json"))
            (out / "drr_spec_diff.json").write_text(
                json.dumps(drr_diff, indent=2, default=str), encoding="utf-8")
    else:
        pending.append("fresh_drr (not supplied)")

    # 3. gold-spec quant run (fresh) + comparison against the recorded run.
    quant_dir = out / "quant"
    run_all(ANCHOR_STRATEGIES, out_dir=quant_dir)
    quant_records = {a: _load(quant_dir / f"{a}.json") for a in ANCHOR_STRATEGIES}
    comparisons["quant_vs_recorded"] = compare_quant(
        quant_dir, require_licensed_input(_REPO_ROOT / RECORDED_QUANT,
                                          "recorded quant run (local pipeline output, not shipped with the repository)"))

    # 4. gate12 (§7 gates 1-2) + comparison against the recorded verdicts.
    gate12 = {a: gate12_verdict(a, subs) for a in GATE12_ANCHORS}
    (out / "g2_gates.json").write_text(json.dumps({"anchors": gate12}, indent=2,
                                                  default=str), encoding="utf-8")
    comparisons["gate12_vs_recorded"] = compare_gate12(
        gate12, require_licensed_input(_REPO_ROOT / RECORDED_G2,
                                       "recorded g2 gates (local pipeline output, not shipped with the repository)"))

    # 5. kpp estimation score (deterministic; --out is mandatory to avoid the
    # recorded-artifact default path).
    from scripts.run_kpp_score import main as kpp_main  # lazy
    kpp_out = out / "kpp_score.json"
    kpp = {"exit": kpp_main(["--run-dir", "runs/kpp_report", "--out", str(kpp_out)]),
           "out": str(kpp_out.relative_to(_REPO_ROOT))}

    # 6. audit comparison + RQ4 entry rows (needs the fresh audit reports).
    audit_ok = entry = None
    if args.audit_dir:
        audit_dir = _REPO_ROOT / args.audit_dir
        audit_ok = {a: (audit_dir / f"{a}_report.json").exists() for a in ANCHOR_STRATEGIES}
        if all(audit_ok.values()):
            comparisons["audit_vs_recorded"] = compare_audit(
                audit_dir, require_licensed_input(_REPO_ROOT / RECORDED_AUDIT,
                                                  "recorded audit run (local pipeline output, not shipped with the repository)"))
            entry = entry_rows(audit_dir)
        else:
            pending.append("audit (reports incomplete: "
                           + ", ".join(a for a, ok in audit_ok.items() if not ok) + ")")
            audit_ok = None
    else:
        pending.append("audit (not supplied)")

    # 7. funnel byte-identity.
    funnel_ok = None
    if args.funnel_file:
        funnel_file = _REPO_ROOT / args.funnel_file
        if funnel_file.exists():
            comparisons["funnel_byte_identity"] = compare_funnel(
                funnel_file, require_licensed_input(_REPO_ROOT / RECORDED_FUNNEL,
                                                    "recorded funnel artifact (local pipeline output, not shipped with the repository)"))
            funnel_ok = comparisons["funnel_byte_identity"]["pass"]
        else:
            pending.append("funnel (file absent)")
    else:
        pending.append("funnel (not supplied)")

    # 8. reporter pointer checks (writes nothing).
    from agents.reporter.cli import check_run  # lazy
    reporter = {}
    for a in ANCHOR_STRATEGIES:
        if not (_REPO_ROOT / "reporter" / "runs" / f"{a}.yaml").exists():
            reporter[a] = "unobserved (pointer absent)"
            pending.append(f"reporter {a} (pointer absent)")   # a required stage; absence => incomplete
            continue
        try:
            check_run(a)
            reporter[a] = "verified"
        except Exception as exc:  # a check failure is a recorded finding
            reporter[a] = f"FAILED: {type(exc).__name__}: {exc}"

    # 9. matrix + manifest + summary.
    matrix = build_matrix(extractions, compile_rows, quant_records, gate12,
                          audit_ok, entry, funnel_ok, reporter)
    all_pass = all(c.get("pass") for c in comparisons.values())
    reporter_ok = all(v == "verified" for v in reporter.values()
                      if v != "unobserved (pointer absent)")
    status = ("complete_pass" if all_pass and reporter_ok and not pending else
              "incomplete" if pending else "finding_recorded")
    manifest = {
        "run_timestamp": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "pending": pending,
        "pins": {k: vars(p) for k, p in PINNED_SPECS.items()},
        "extractions": extractions,
        "compile_rows": compile_rows,
        "adapter_refusal_pin_checks": refusal_pin_checks,
        "other_extraction_events": {k: v for k, v in other_events.items() if v},
        "quant_records": quant_records,
        "gate12": gate12,
        "kpp_score": kpp,
        "rq4_entry": entry,
        "reporter": reporter,
        "comparisons": comparisons,
        "drr_spec_diff_summary": (None if drr_diff is None else
                                  {k: drr_diff[k] for k in
                                   ("n_fields", "n_tag_changed",
                                    "n_stated_value_diff", "n_evidence_only")}),
        "matrix": matrix,
    }
    (out / "traversal_manifest.json").write_text(
        json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    (out / "traversal_summary.md").write_text(
        render_summary_md(matrix, comparisons, drr_diff, kpp), encoding="utf-8")

    print(json.dumps({"status": status, "pending": pending,
                      "comparisons": {k: c.get("pass") for k, c in comparisons.items()},
                      "out": str(out)}, indent=2))
    if status == "complete_pass":
        return 0
    return 3 if status == "incomplete" else 2


if __name__ == "__main__":
    raise SystemExit(main())
