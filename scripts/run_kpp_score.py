#!/usr/bin/env python
"""
KPP §3.7 sub-metric scorer (Scope B hybrid, registered 2026-09-04): load the
run's emitted spec typed (spec_loader), score against the ratified KPP gold via
the Scope-A ``score_kpp`` (never pooled with the sort G3 number -- own table).

Denominator statement (the hybrid's honesty condition): the paid run asks 8 of
the 11 estimation fields; the 3 prose fields are registered UNKNOWN(not_extracted)
and appear as NOT_ASKED rows here -- never silently absent. The rubric re-score
to 11/11 upgrades this SAME run later (option (a) as future work).
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from agents.librarian.pipeline.spec_loader import spec_from_dict  # noqa: E402
from agents.librarian.schema.estimation_fields import (  # noqa: E402
    ESTIMATION_FIELDS,
    ESTIMATION_PROSE_FIELDS,
)
from evaluation.gold_specs.gold_loader import load_gold_spec  # noqa: E402
from evaluation.harness.estimation_scoring import score_kpp  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Score a KPP run against the ratified gold (§3.7).")
    ap.add_argument("--run-dir", default="runs/kpp_report")
    ap.add_argument("--out", default=None,
                    help="output JSON (default results/kpp_score_<runname>.json)")
    ap.add_argument("--rubric-judgements", default=None, dest="rubric_judgements",
                    help="rubric adjudication JSON (kpp_prose_rubric.md; only legal "
                         "once the rubric is IN FORCE): {\"judgements\": {field: "
                         "{\"correct\": bool, \"failed_clauses\": [...]}}}")
    args = ap.parse_args(argv)

    run_dir = Path(args.run_dir)
    spec_path = run_dir / "spec_0.json"
    if not spec_path.exists():
        print(f"[kpp_score] no spec_0.json in {run_dir} (typed exit upstream?)")
        return 1
    run_spec = spec_from_dict(json.loads(spec_path.read_text(encoding="utf-8")))
    gold = load_gold_spec("kpp")

    judgements = annotations = None
    if args.rubric_judgements:
        raw = json.loads(Path(args.rubric_judgements).read_text(encoding="utf-8"))
        entries = raw.get("judgements", {})
        bad = set(entries) - ESTIMATION_PROSE_FIELDS
        if bad:
            raise SystemExit(f"[kpp_score] rubric judgements for non-prose fields: {sorted(bad)}")
        judgements = {k: bool(v["correct"]) for k, v in entries.items()}
        # §2.4 non-scoring annotation: failed clauses ride the output, never a score.
        annotations = {k: list(v.get("failed_clauses", [])) for k, v in entries.items()}

    score = score_kpp(gold, run_spec, rubric_judgements=judgements)

    rows = [{"cell": c.cell, "key": str(c.key),
             "outcome": getattr(c.outcome, "value", c.outcome)}
            for c in score.cells]
    tally = Counter(r["outcome"] for r in rows)
    summary = {
        "denominator_statement": (
            f"{len(ESTIMATION_FIELDS) - len(ESTIMATION_PROSE_FIELDS)} of "
            f"{len(ESTIMATION_FIELDS)} estimation fields asked (Scope-B hybrid, "
            f"registered 2026-09-04); prose fields {sorted(ESTIMATION_PROSE_FIELDS)} "
            "NOT_ASKED pending the rubric"),
        "outcome_tally": dict(tally),
        "instruments": {"gold": score.n_gold_instruments,
                        "run": score.n_run_instruments,
                        "matched": score.n_matched_instruments},
        "never_pooled": True,
    }
    if annotations is not None:
        summary["rubric_annotations"] = annotations   # §2.4: diagnostic, non-scoring
    out = Path(args.out) if args.out else (
        _REPO_ROOT / "results" / f"kpp_score_{run_dir.name}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"summary": summary, "rows": rows}, indent=2),
                   encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"[kpp_score] -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
