#!/usr/bin/env python
"""
T5 clean-vs-perturbed differencing (prereg §4's registered robustness analysis,
run 2026-09-03): attribute every graded Arm-A row to BASELINE vs PERTURBATION.

The tally alone conflates two very different failure sources on the invariant
(structural) classes: a MISS where the CLEAN run also fails to ship the target
field is BASELINE-LIMITED (the extraction never shipped it, perturbed or not);
a MISS where the clean run ships it correctly is PERTURBATION-INDUCED (real
fragility -- the edit flipped a working extraction). Review-exit variants
(status ERROR, no spec) are their own bucket: the clean run exits 0, so a
perturbed run that cannot assemble at all is a perturbation-induced RUN break.

Inputs are stored artefacts only (deterministic, $0): the clean paid anchor runs
(runs/corpus_anchors_report/*) scored by the same score_anchor the grader uses,
plus results/t5/arm_a_robustness.json. Output: results/t5/differencing.{json,md}.
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from evaluation.harness.gold_calibration import score_anchor  # noqa: E402

CLEAN_DIR_OF = {"drf": "runs/corpus_anchors_report/bbw",
                "mom6": "runs/corpus_anchors_report/jnps",
                "str": "runs/corpus_anchors_report/drr"}


def clean_outcomes() -> dict[tuple[str, str], str]:
    """(anchor, field) -> the CLEAN paid run's scored outcome value."""
    out: dict[tuple[str, str], str] = {}
    for anchor, d in CLEAN_DIR_OF.items():
        score = score_anchor(anchor, _REPO_ROOT / d)
        for row in score.rows:
            leaf = row.dotted_path.split(".")[-1]      # part2.legs[0].long_leg -> long_leg
            out[(anchor, leaf)] = row.outcome.value
    return out


def main() -> int:
    rows = json.loads((_REPO_ROOT / "results/t5/arm_a_robustness.json").read_text())
    clean = clean_outcomes()

    graded, out_rows = Counter(), []
    for r in rows:
        if r["status"] == "SKIPPED_NON_SCOREABLE":
            continue
        # Compound targets: the graded row is the sheet's dotted-path field; use
        # the first component for the clean-side join (the graded target).
        field = r["field"].split("+")[0]
        clean_out = clean.get((r["anchor"], field))
        if r["status"] == "ERROR":
            attribution = "perturbation_broke_run"       # clean run exits 0
        elif r["status"] == "MATCH":
            attribution = "robust_or_expected"
        elif clean_out == "shipped_correct":
            attribution = "perturbation_induced"         # clean ships it; perturbed lost it
        elif clean_out is None:
            attribution = "no_clean_row"                 # target not in the clean gold universe
        else:
            attribution = f"baseline_limited({clean_out})"
        graded[(r["class_id"], r["status"], attribution)] += 1
        out_rows.append({**r, "clean_outcome": clean_out, "attribution": attribution})

    # Aggregate: per class, MISSes split into induced vs baseline-limited.
    per_class: dict[str, dict] = {}
    for (cls, status, attr), n in sorted(graded.items()):
        c = per_class.setdefault(cls, Counter())
        if status == "MISS":
            key = "miss_induced" if attr == "perturbation_induced" else "miss_baseline"
            c[key] += n
        elif status == "ERROR":
            c["run_broken"] += n
        else:
            c["match"] += n

    md = ["# T5 clean-vs-perturbed differencing (2026-09-03)", "",
          "| class | MATCH | MISS (perturbation-induced) | MISS (baseline-limited) | run broken |",
          "|---|---|---|---|---|"]
    tot = Counter()
    for cls in sorted(per_class):
        c = per_class[cls]
        md.append(f"| {cls} | {c['match']} | {c['miss_induced']} | {c['miss_baseline']} | {c['run_broken']} |")
        tot.update(c)
    md.append(f"| **total** | **{tot['match']}** | **{tot['miss_induced']}** | "
              f"**{tot['miss_baseline']}** | **{tot['run_broken']}** |")
    md += ["",
           "- *perturbation-induced*: the clean paid run ships this field correct; the perturbed run lost it — genuine fragility.",
           "- *baseline-limited*: the clean run itself does not ship the field (abstains/wrong) — the MISS restates baseline coverage, not fragility.",
           "- *run broken*: the perturbed variant could not assemble at all (review exit; the clean run exits 0).", ""]

    (_REPO_ROOT / "results/t5/differencing.json").write_text(
        json.dumps({"per_class": {k: dict(v) for k, v in per_class.items()},
                    "rows": out_rows}, indent=2), encoding="utf-8")
    (_REPO_ROOT / "results/t5/differencing.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
