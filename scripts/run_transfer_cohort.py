#!/usr/bin/env python
"""
Transfer-cohort evaluation (T2-SEL-8) -- gold-free chained-transfer metrics over
unseen ADMITTED papers.

Reads a corpus report directory of *existing* extraction outputs (one subdir per
paper, each with a ``run_manifest.json`` and, if any construction was emitted, one
``spec_i.json`` + ``trace_i.json`` pair per enumerated construction). For every
paper it evaluates EACH construction:

  * gold-free BEHAVIOURAL rates -- structural coverage (STATED / fact-bearing
    fields), quote-certification (STATED fields whose evidence carries a located
    quote), and the UNKNOWN-reason profile;
  * the SEMANTIC-BINDING outcome, computed by the *same* deterministic path the
    coverage driver uses (``spec_from_dict`` -> ``adapt_spec`` with the
    hash-verified standing register) -> a clean bind or a typed refusal code;
  * the field-fill self-consistency (dual-model agreement) from that construction's
    trace.

Aggregation is CONSTRUCTION-weighted for the throughput rates (each emitted
construction is one unit) and PAPER-weighted for the attrition / chain-localisation
funnel (a paper advances as far as its best construction).

Deliberate scope (per the 2026-09-07 expanded-transfer pre-registration):
NO gold answer key is used (so NO accuracy / over-claim -- those need a reference
key); NO paper enters the RQ3 census (D-A59) or the RQ4 denominator; the holdout
is untouched. Cohorts are reported SEPARATELY (original vs expansion) and never
merged -- pass ``--cohort`` to label the run.

Zero LLM calls in this driver -- it consumes extraction outputs produced by
separate explicit ``run_librarian`` commands.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from agents.librarian.adapter.adapt import adapt_spec  # noqa: E402
from agents.librarian.pipeline.spec_loader import spec_from_dict  # noqa: E402
from evaluation.harness.round_trip import load_verified_standing_subs  # noqa: E402
from scripts.run_t3_coverage import refusal_codes_of  # noqa: E402

# The REACHABLE chained stages, in order. There is no 'execution' stage here: this driver
# does not run the strategy, so a clean bind is COMPLETE-and-parked at the execution boundary
# (blocked_at=None, execution_status='not_attempted') rather than blocked -- flagged for a
# follow-up run, since the narrow audited registry makes a clean unseen bind the rare case. A
# paper advances as far as its best construction.
STAGES = ("ingestion", "spec_emission", "semantic_binding")


def _tag_leaves(spec_dict: dict) -> list[dict]:
    """Every fact-bearing Inherited leaf (carries ``tag`` + ``value``) under the
    header/part1/part2 sub-trees."""
    leaves: list[dict] = []

    def walk(o):
        if isinstance(o, dict):
            if "tag" in o and "value" in o:
                leaves.append(o)
            else:
                for v in o.values():
                    walk(v)
        elif isinstance(o, list):
            for v in o:
                walk(v)

    for k in ("header", "part1", "part2"):
        walk(spec_dict.get(k, {}))
    return leaves


def _located(leaf: dict) -> bool:
    ev = leaf.get("evidence") or {}
    loc = ev.get("locator") or {}
    return bool(ev.get("quote")) and loc.get("char_start") is not None


def behavioural(spec_dict: dict) -> dict:
    """Gold-free structural extraction behaviour -- needs no answer key."""
    leaves = _tag_leaves(spec_dict)
    n = len(leaves)
    stated = [leaf for leaf in leaves if leaf.get("tag") == "STATED"]
    unknown = [leaf for leaf in leaves if leaf.get("tag") == "UNKNOWN"]
    cert = [leaf for leaf in stated if _located(leaf)]
    reasons = [(leaf.get("evidence") or {}).get("unknown_reason") for leaf in unknown]
    return {
        "n_fields": n,
        "n_stated": len(stated),
        "n_unknown": len(unknown),
        "coverage": round(len(stated) / n, 4) if n else None,
        "quote_cert": round(len(cert) / len(stated), 4) if stated else None,
        "unknown_reasons": dict(Counter(r for r in reasons if r)),
    }


def bind(spec_dict: dict, subs) -> tuple[bool, list[str]]:
    """The semantic-binding stage, identical to run_t3_coverage: spec -> adapter."""
    result = adapt_spec(spec_from_dict(spec_dict), standing_subs=subs)
    return (not result.refused), refusal_codes_of(result)


def self_consistency(trace_path: Path) -> dict | None:
    """Stage-3 Librarian throughput for ONE construction -- dual-model field-fill
    agreement + the tag-reason histogram, read from that construction's field-fill
    trace (gold-free). None when the trace is absent or empty."""
    if not trace_path.exists():
        return None
    recs = json.loads(trace_path.read_text(encoding="utf-8")).get("records", [])
    if not recs:
        return None
    agree = sum(1 for r in recs if r.get("agreement") is True)
    return {
        "n_fields": len(recs),
        "self_consistency": round(agree / len(recs), 4),
        "reason_histogram": dict(Counter(r.get("final_reason") for r in recs)),
    }


def _spec_index(spec_path: Path) -> int:
    """``spec_12.json`` -> 12 (numeric order, so spec_10 sorts after spec_2)."""
    return int(spec_path.stem.split("_")[1])


def evaluate_paper(paper_id: str, run_dir: Path, subs) -> dict:
    """One record per paper. EVERY emitted construction (spec_i.json + trace_i.json)
    is evaluated; the paper advances as far as its best construction."""
    spec_paths = sorted(run_dir.glob("spec_*.json"), key=_spec_index)
    rec: dict = {"paper_id": paper_id}
    if not spec_paths:
        # extraction routed to review / emitted zero specs -> blocked at emission
        rec.update(emitted=False, n_constructions=0, n_constructions_bound_clean=0,
                   constructions=[], stage_reached="ingestion", blocked_at="spec_emission",
                   execution_status="not_reached")
        return rec
    constructions = []
    for sp in spec_paths:
        i = _spec_index(sp)
        spec_dict = json.loads(sp.read_text(encoding="utf-8"))
        clean, codes = bind(spec_dict, subs)
        constructions.append({
            "index": i,
            "behavioural": behavioural(spec_dict),
            "binding": {"clean": clean, "refusal_codes": codes},
            "self_consistency": self_consistency(run_dir / f"trace_{i}.json"),
        })
    n_clean = sum(1 for c in constructions if c["binding"]["clean"])
    rec.update(emitted=True, n_constructions=len(constructions),
               n_constructions_bound_clean=n_clean, constructions=constructions)
    if n_clean > 0:
        # reached the execution boundary; execution is not run by this driver (not "blocked")
        rec.update(stage_reached="semantic_binding", blocked_at=None,
                   execution_status="not_attempted")
    else:
        rec.update(stage_reached="spec_emission", blocked_at="semantic_binding",
                   execution_status="not_reached")
    return rec


def discover_papers(report_dir: Path) -> list[tuple[str, Path]]:
    """One (paper_id, dir) per subdir that carries a run_manifest.json."""
    out = []
    for sub in sorted(report_dir.iterdir()):
        if sub.is_dir() and (sub / "run_manifest.json").exists():
            out.append((sub.name, sub))
    return out


def aggregate(records: list[dict]) -> dict:
    n = len(records)
    reached = {s: 0 for s in STAGES}
    for r in records:
        # every paper reaches ingestion; then up to its stage_reached
        idx = STAGES.index(r["stage_reached"])
        for s in STAGES[: idx + 1]:
            reached[s] += 1
    emitters = [r for r in records if r["emitted"]]
    # CONSTRUCTION-weighted pools -- each emitted construction is one unit.
    cons = [c for r in emitters for c in r["constructions"]]
    covs = [c["behavioural"]["coverage"] for c in cons if c["behavioural"]["coverage"] is not None]
    qcs = [c["behavioural"]["quote_cert"] for c in cons if c["behavioural"]["quote_cert"] is not None]
    scs = [c["self_consistency"]["self_consistency"] for c in cons if c.get("self_consistency")]
    refusal_codes: Counter = Counter()
    reason_hist: Counter = Counter()
    unknown_reasons: Counter = Counter()
    for c in cons:
        refusal_codes.update(c["binding"]["refusal_codes"])
        unknown_reasons.update(c["behavioural"]["unknown_reasons"])
        if c.get("self_consistency"):
            reason_hist.update(c["self_consistency"]["reason_histogram"])
    n_papers_bound_clean = sum(1 for r in emitters if r["n_constructions_bound_clean"] > 0)
    return {
        "n_papers": n,
        "unit_note": "attrition + chain-localisation are PAPER-weighted (a paper advances "
                     "as far as its best construction); throughput rates are CONSTRUCTION-weighted.",
        "attrition": {s: reached[s] for s in STAGES},
        "chain_localisation": dict(Counter(r["blocked_at"] for r in records
                                           if r["blocked_at"] is not None)),
        "n_emitted": len(emitters),                       # papers emitting >= 1 construction
        "n_constructions": len(cons),
        "n_papers_bound_clean": n_papers_bound_clean,     # >= 1 construction bound clean
        "n_constructions_bound_clean": sum(r["n_constructions_bound_clean"] for r in emitters),
        "reached_execution_boundary": n_papers_bound_clean,   # parked at the boundary (not run)
        "mean_coverage": round(sum(covs) / len(covs), 4) if covs else None,
        "mean_quote_cert": round(sum(qcs) / len(qcs), 4) if qcs else None,
        "mean_self_consistency": round(sum(scs) / len(scs), 4) if scs else None,
        "extraction_reason_histogram": dict(reason_hist.most_common()),
        "binding_refusal_codes": dict(refusal_codes.most_common()),
        "unknown_reasons": dict(unknown_reasons.most_common()),
        # descriptive performance + auditor stages need a clean bind AND a follow-up execution
        # run, which this driver does not attempt -> null (NOT an observed 0).
        "n_descriptive_performance": None,
        "n_audited": None,
        "post_bind_status": "not_attempted",
    }


def _paper_means(rec: dict) -> tuple:
    """Per-paper mean coverage / quote_cert over that paper's constructions."""
    cons = rec.get("constructions") or []
    covs = [c["behavioural"]["coverage"] for c in cons if c["behavioural"]["coverage"] is not None]
    qcs = [c["behavioural"]["quote_cert"] for c in cons if c["behavioural"]["quote_cert"] is not None]
    mc = round(sum(covs) / len(covs), 4) if covs else "—"
    mq = round(sum(qcs) / len(qcs), 4) if qcs else "—"
    return mc, mq


def render_md(cohort: str, agg: dict, records: list[dict]) -> str:
    a = agg["attrition"]
    lines = [
        f"# Transfer cohort ({cohort}) -- gold-free chained-transfer metrics",
        "",
        f"Papers: **{agg['n_papers']}** | emitted ≥1 construction: {agg['n_emitted']} | "
        f"constructions: {agg['n_constructions']} | papers with ≥1 clean bind: "
        f"{agg['n_papers_bound_clean']} | constructions bound clean: "
        f"{agg['n_constructions_bound_clean']} | reached execution boundary "
        f"(not run): {agg['reached_execution_boundary']}",
        "",
        "## Chained attrition (unit = paper; a paper advances as far as its best construction)",
        "",
        "| Stage | Papers reaching |",
        "|---|---:|",
    ]
    for s in STAGES:
        lines.append(f"| {s} | {a[s]} |")
    lines += [
        "",
        "## Chain-localisation (earliest blocking stage, per paper)",
        "",
        "| Blocked at | Papers |",
        "|---|---:|",
    ]
    for k, v in sorted(agg["chain_localisation"].items()):
        lines.append(f"| {k} | {v} |")
    lines += [
        "",
        f"## Per-component throughput (over the {agg['n_constructions']} constructions, gold-free)",
        "",
        f"- Librarian — mean field-fill **self-consistency** (dual-model agreement): "
        f"**{agg.get('mean_self_consistency')}**",
        f"- Librarian — tag-reason histogram: {agg.get('extraction_reason_histogram') or '—'}",
        f"- Librarian — mean structural coverage (STATED / fields): "
        f"**{agg['mean_coverage']}**; quote-certification: **{agg['mean_quote_cert']}**",
        f"- Librarian — UNKNOWN-reason histogram: {agg.get('unknown_reasons') or '—'}",
        f"- Quant — binding **refusal histogram**: {agg['binding_refusal_codes'] or '—'}",
        "- Quant — descriptive performance / Auditor — audited: **not attempted** "
        "(both need a clean bind AND a follow-up execution run; execution is not attempted "
        "by this driver, so neither is measured -- 'not attempted', not an observed 0).",
        "",
        "## Per-paper",
        "",
        "| paper | #constr | emitted | mean coverage | mean quote_cert | clean binds | outcome |",
        "|---|---:|---|---:|---:|---:|---|",
    ]
    for r in records:
        mc, mq = _paper_means(r)
        nclean = r.get("n_constructions_bound_clean", 0)
        ncons = r.get("n_constructions", 0)
        outcome = r.get("blocked_at") or "execution boundary (not run)"
        lines.append(
            f"| {r['paper_id']} | {ncons} | {r['emitted']} | {mc} | {mq} | "
            f"{nclean}/{ncons} | {outcome} |"
        )
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--report-dir", required=True, type=Path,
                    help="corpus report dir (one subdir per paper)")
    ap.add_argument("--cohort", required=True,
                    help="cohort label, e.g. 'original' or 'expansion' — reported separately, never merged")
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args(argv)

    subs = load_verified_standing_subs()
    papers = discover_papers(args.report_dir)
    if not papers:
        print(f"no paper subdirs under {args.report_dir}", file=sys.stderr)
        return 2
    records = [evaluate_paper(pid, d, subs) for pid, d in papers]
    agg = aggregate(records)

    args.out.mkdir(parents=True, exist_ok=True)
    payload = {"cohort": args.cohort, "report_dir": str(args.report_dir),
               "aggregate": agg, "papers": records}
    (args.out / "transfer_cohort.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8")
    (args.out / "transfer_cohort.md").write_text(
        render_md(args.cohort, agg, records), encoding="utf-8")
    print(f"[{args.cohort}] {agg['n_papers']} papers ({agg['n_constructions']} constructions) | "
          f"attrition {agg['attrition']} | reached_execution_boundary={agg['reached_execution_boundary']}")
    print(f"  chain-localisation: {agg['chain_localisation']}")
    print(f"  mean_coverage={agg['mean_coverage']} mean_quote_cert={agg['mean_quote_cert']}")
    print(f"  -> {args.out}/transfer_cohort.{{json,md}}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
