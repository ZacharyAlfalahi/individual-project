#!/usr/bin/env python
"""
RQ1 scoring CLI (B3): one committed entry point from an extraction run root to the
reportable RQ1 table -- the G3 harness chain that until now ran only inside tests.

    ./.venv/bin/python scripts/run_g3_score.py --run-root runs/g3_2026-07-22_v3
    ./.venv/bin/python scripts/run_g3_score.py --run-root runs/report_final \
        --anchors str,drf,mom6,crf --out results/librarian/g3_report.md

Per anchor: load_run -> score_anchor (§ scoring unit) -> compute_metrics (§3.1)
+ build_agreement_table (§3.2) + decompose (§3.6); then aggregate (§3.3) over the
3-anchor set AND, when crf is scored, the 4-anchor set -- both headlines side by
side, per the contract. All headline rendering passes through require_reportable:
a Phase-D run refuses unless --allow-non-reportable is given (greppable, per §1).

Boundaries (deliberate, per the count authority):
  * lrf may be scored per-anchor (--include-lrf) but has NO pooled home -- it
    never enters aggregate() (which fails loud on it by design).
  * kpp is the disjoint non-pooled §3.7 path (score_kpp consumes specs, not run
    dirs); it is NOT scored here until the live-KPP adapter lands (Scope B).
  * synth is a T4(b) instrument with no _ANCHORS row; run_t4b_kat.py grades it.
  * Contract §3.4 (repeats / flip-rate) has no harness support anywhere yet --
    a known gap carried, not silently closed.

Anchor -> run-dir resolution: each anchor maps to its paper's run dir (drf/crf ->
bbw, mom6 -> jnps, str -> drr); a multi-spec dir resolves the strategy index by
matching the gold's strategy_label against each spec's trace header, and
--map anchor=dir[:index] overrides both.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from evaluation.gold_specs.gold_loader import load_gold_spec              # noqa: E402
from evaluation.harness.aggregation import (                              # noqa: E402
    ANCHOR_SET,
    ANCHOR_SET_WITH_CRF,
    aggregate,
    render_aggregate,
)
from evaluation.harness.agreement_calibration import (                    # noqa: E402
    build_agreement_table,
    render_agreement_table,
)
from evaluation.harness.calibration_report import (                       # noqa: E402
    compute_metrics,
    render_headline_table,
)
from evaluation.harness.gold_calibration import score_anchor              # noqa: E402
from evaluation.harness.missed_evidence import decompose, render_decomposition  # noqa: E402
from evaluation.harness.run_artefacts import ArtefactError, load_run      # noqa: E402

# Default run-dir name per anchor (the paper the anchor lives in; run dirs are
# conventionally named by the run_librarian PAPERS key).
DEFAULT_DIR_OF = {"drf": "bbw", "crf": "bbw", "lrf": "bbw", "mom6": "jnps", "str": "drr"}


def _spec_count(run_dir: Path) -> int:
    return len(sorted(run_dir.glob("spec_*.json")))


def _resolve_artefacts(anchor: str, run_dir: Path, index: int | None):
    """Load the anchor's strategy from a (possibly multi-spec) run dir.

    Explicit index wins; a single-spec dir is index 0; a multi-spec dir resolves
    by matching the gold's strategy_label against each spec's trace header --
    fail loud (listing what was found) rather than guessing."""
    if index is not None:
        return load_run(run_dir, strategy_index=index)
    n = _spec_count(run_dir)
    if n == 0:
        raise ArtefactError(f"{run_dir}: no spec_*.json found")
    if n == 1:
        return load_run(run_dir, strategy_index=0)

    gold_label = load_gold_spec(anchor).header.strategy_label
    found: dict[int, str] = {}
    for i in range(n):
        art = load_run(run_dir, strategy_index=i)
        label = art.header.get("strategy_label", "")
        found[i] = label
        if label == gold_label:
            return art
    raise ArtefactError(
        f"{run_dir}: {n} specs but none matches gold {anchor!r} strategy_label "
        f"{gold_label!r}; found {found}. Pass --map {anchor}=<dir>:<index>."
    )


def score_one(anchor: str, run_dir: Path, index: int | None, *, allow_non_reportable: bool):
    """The full per-anchor chain; returns (sections_markdown, json_record)."""
    art = _resolve_artefacts(anchor, run_dir, index)
    score = score_anchor(anchor, run_dir, artefacts=art)
    bundle = compute_metrics(score, art)
    table = build_agreement_table(score, art)
    decomp = decompose(score, art)

    md = "\n\n".join([
        f"## {anchor} ({bundle.paper_id}) -- {run_dir}",
        render_headline_table(bundle, allow_non_reportable=allow_non_reportable),
        render_agreement_table(table, allow_non_reportable=allow_non_reportable),
        render_decomposition(decomp, allow_non_reportable=allow_non_reportable),
    ])

    def _prop(p):
        return {"numerator": p.numerator, "denominator": p.denominator,
                "value": p.value, "interval": p.interval}

    record = {
        "run_dir": str(run_dir),
        "paper_id": bundle.paper_id,
        "reportable": bundle.reportability.reportable,
        "phase": bundle.reportability.phase,
        "universe": bundle.universe,
        "coverage": _prop(bundle.coverage),
        "selective_accuracy": _prop(bundle.selective_accuracy),
        "over_claim_rate": _prop(bundle.over_claim_rate),
        "abstention_rate": _prop(bundle.abstention_rate),
        "missed_evidence_rate": _prop(bundle.missed_evidence_rate),
        "outcome_distribution": dict(bundle.outcome_distribution),
        "condition_incidence": dict(bundle.condition_incidence),
        "missed_evidence_counts": {getattr(m, "value", m): c
                                   for m, c in decomp.counts.items()},
        "missed_evidence_dominant": (getattr(decomp.dominant, "value", decomp.dominant)
                                     if decomp.dominant is not None else None),
    }
    return md, record, bundle


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Score an extraction run against the RQ1 golds.")
    ap.add_argument("--run-root", required=True,
                    help="run root holding per-paper run dirs (e.g. runs/g3_2026-07-22_v3)")
    ap.add_argument("--anchors", default="str,drf,mom6",
                    help="comma-separated sort anchors to score (add crf for the 4-anchor "
                         "headline). Default: the 3-anchor set.")
    ap.add_argument("--include-lrf", action="store_true",
                    help="also score lrf per-anchor (negative-control construction; it has "
                         "no pooled home and never enters the aggregate).")
    ap.add_argument("--map", action="append", default=[], metavar="ANCHOR=DIR[:INDEX]",
                    help="override an anchor's run dir (relative to --run-root) and "
                         "optionally its strategy index.")
    ap.add_argument("--allow-non-reportable", action="store_true",
                    help="render Phase-D (dev) figures anyway -- dev-iteration reference "
                         "only, never project numbers (contract §1).")
    ap.add_argument("--out", default=None, help="also write the markdown report here")
    ap.add_argument("--json", default=None, dest="json_out",
                    help="also write the JSON sidecar here")
    args = ap.parse_args(argv)

    run_root = (_REPO_ROOT / args.run_root) if not Path(args.run_root).is_absolute() \
        else Path(args.run_root)
    anchors = [a.strip() for a in args.anchors.split(",") if a.strip()]
    if args.include_lrf and "lrf" not in anchors:
        anchors.append("lrf")

    overrides: dict[str, tuple[str, int | None]] = {}
    for m in args.map:
        anchor, _, target = m.partition("=")
        d, _, idx = target.partition(":")
        overrides[anchor.strip()] = (d.strip(), int(idx) if idx else None)

    sections: list[str] = [f"# G3 score -- {run_root.name}"]
    records: dict[str, dict] = {}
    bundles: dict[str, object] = {}
    for anchor in anchors:
        d, idx = overrides.get(anchor, (DEFAULT_DIR_OF.get(anchor, anchor), None))
        md, record, bundle = score_one(anchor, run_root / d, idx,
                                       allow_non_reportable=args.allow_non_reportable)
        sections.append(md)
        records[anchor] = record
        if anchor in ANCHOR_SET_WITH_CRF:   # lrf never enters an aggregate
            bundles[anchor] = bundle

    # §3.3: both headlines side by side where the denominator allows.
    agg_records: dict[str, dict] = {}
    for name, expected in (("3-anchor", ANCHOR_SET), ("4-anchor", ANCHOR_SET_WITH_CRF)):
        if set(expected) <= set(bundles):
            agg = aggregate({a: bundles[a] for a in expected}, anchors_expected=expected)
            sections.append(f"## §3.3 aggregate -- {name} headline\n\n"
                            + render_aggregate(agg, allow_non_reportable=args.allow_non_reportable))
            agg_records[name] = {
                "anchors": list(expected),
                "micro": {k: {"numerator": p.numerator, "denominator": p.denominator,
                              "value": p.value} for k, p in agg.micro.items()},
            }

    report = "\n\n".join(sections) + "\n"
    print(report)

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(report, encoding="utf-8")
        print(f"[g3_score] markdown -> {out}", file=sys.stderr)
    if args.json_out:
        jp = Path(args.json_out)
        jp.parent.mkdir(parents=True, exist_ok=True)
        jp.write_text(json.dumps({"run_root": str(run_root), "anchors": records,
                                  "aggregates": agg_records}, indent=2), encoding="utf-8")
        print(f"[g3_score] json -> {jp}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
