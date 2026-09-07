#!/usr/bin/env python
"""
RQ1 safety-discipline ablation grid (B7 / Ch4 §4.3.3): re-score ARCHIVED raw
responses under the four combinations of the quote gate × the dual-model rule,
pricing each discipline's benefit (precision) against its cost (coverage).

    ./.venv/bin/python scripts/run_rq1_ablation.py --run-root runs/g3_v3 \
        --allow-non-reportable

No new extraction happens and no model is called: elicitation is held fixed, and
the grid records only what the disciplines DO to what is allowed to ship, exactly
as specified. Everything evidential is re-derived from the per-run raw archives
(the p6a replay machinery): `answered`/`value`/`quote` from the archive, `located`
recomputed against the frozen canonical text, `normalised` through the production
normaliser.

Shipping rule per combo (deterministic; single-model arms prefer model_a, then
model_b -- a fixed order, never a quality judgement):
  dual=ON,  quote=ON   the production D9 rule (both answered, normalised-equal,
                       both quotes located)
  dual=ON,  quote=OFF  both answered + normalised-equal (location ignored)
  dual=OFF, quote=ON   first model whose answer carries a LOCATED quote
  dual=OFF, quote=OFF  first model that answered at all

Reported per combo: coverage, selective accuracy, over-claim -- the risk-coverage
trade the disciplines buy. The full §3.1 bundle is NOT reported here: the combo
RunFields are scoring vehicles (the shipped value rides `normalised_a`), so the
condition-incidence and agreement views would be distorted; the three headline
proportions depend only on outcome classification and stay exact.

Phase discipline: identical to run_g3_score -- a Phase-D archive refuses to render
without --allow-non-reportable (contract §1).
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from agents.librarian.config import load_canonical_text                   # noqa: E402
from evaluation.harness.calibration_report import compute_metrics         # noqa: E402
from evaluation.harness.gold_calibration import score_anchor              # noqa: E402
from evaluation.harness.reportability import require_reportable           # noqa: E402
from evaluation.harness.run_artefacts import RunField, load_run           # noqa: E402
from scripts.run_g3_score import DEFAULT_DIR_OF                           # noqa: E402
from scripts.run_librarian import PAPERS                                  # noqa: E402
from scripts.run_p6a_replay import _normalise_or_none, _read_raw          # noqa: E402

COMBOS: tuple[tuple[str, bool, bool], ...] = (
    ("full system (dual + quote gate)", True, True),
    ("dual-model only (no quote gate)", True, False),
    ("quote gate only (single-model)", False, True),
    ("neither discipline", False, False),
)

# run-dir name -> canonical-text paper key (for locate recomputation).
_PAPER_KEY_OF_DIR = {"bbw": "bbw", "jnps": "jnps", "drr": "drr"}


def _ship(ra: dict, rb: dict, loc_a: bool, loc_b: bool,
          norm_a, norm_b, *, dual: bool, quote: bool):
    """(shipped?, shipped_value) under one combo. Deterministic a-then-b order."""
    if dual:
        if not (ra["answered"] and rb["answered"] and norm_a is not None and norm_a == norm_b):
            return False, None
        if quote and not (loc_a and loc_b):
            return False, None
        return True, norm_a
    candidates = ((ra["answered"], loc_a, norm_a), (rb["answered"], loc_b, norm_b))
    for answered, located, norm in candidates:
        if not answered or norm is None:
            continue
        if quote and not located:
            continue
        return True, norm
    return False, None


def replay_combo(recorded, raw_a, raw_b, ct, *, dual: bool, quote: bool):
    """Rebuild every asked field's RunField under one combo's shipping rule.

    The SHIPPED value rides normalised_a (the slot the scorer compares); these
    fields are scoring vehicles for outcome classification only."""
    fields: dict[str, RunField] = {}
    for name, rec in recorded.fields.items():
        if rec.not_extracted:
            fields[name] = rec
            continue
        ra, rb = raw_a.get(name), raw_b.get(name)
        if ra is None or rb is None:
            raise RuntimeError(f"{recorded.run_dir}: field {name!r} missing from a raw archive")

        if ra["kind"] == "method_summary":
            # Rubric field: excluded from the exact-match universe by the scorer;
            # ship semantics do not affect the three reported proportions.
            fields[name] = rec
            continue

        norm_a, _ = _normalise_or_none(name, ra["value"], ra["kind"])
        norm_b, _ = _normalise_or_none(name, rb["value"], rb["kind"])
        loc_a = bool(ra["answered"] and ra["quote"] and ct.locate(ra["quote"]) is not None)
        loc_b = bool(rb["answered"] and rb["quote"] and ct.locate(rb["quote"]) is not None)
        shipped, value = _ship(ra, rb, loc_a, loc_b, norm_a, norm_b, dual=dual, quote=quote)

        fields[name] = RunField(
            field=name,
            a_answered=ra["answered"], b_answered=rb["answered"],
            a_quote=ra["quote"], b_quote=rb["quote"],
            a_located=loc_a, b_located=loc_b,
            a_model_id=ra["model_id"], b_model_id=rb["model_id"],
            normalised_a=value if shipped else norm_a,
            normalised_b=norm_b,
            final_tag="STATED" if shipped else "UNKNOWN",
            shipped_reason="ablation_replay",
            ship_choice=None,
            not_extracted=False,
        )
    return dataclasses.replace(recorded, fields=fields)


def score_grid(anchor: str, run_dir: Path) -> list[dict]:
    recorded = load_run(run_dir)
    raw_a = _read_raw(run_dir / "raw" / "raw_model_a.jsonl")
    raw_b = _read_raw(run_dir / "raw" / "raw_model_b.jsonl")
    paper_key = _PAPER_KEY_OF_DIR.get(run_dir.name, run_dir.name)
    ct = load_canonical_text(_REPO_ROOT / PAPERS[paper_key]["canonical_text"])

    rows = []
    for label, dual, quote in COMBOS:
        variant = replay_combo(recorded, raw_a, raw_b, ct, dual=dual, quote=quote)
        bundle = compute_metrics(score_anchor(anchor, run_dir, artefacts=variant), variant)
        rows.append({
            "anchor": anchor, "combo": label, "dual": dual, "quote_gate": quote,
            "coverage": (bundle.coverage.numerator, bundle.coverage.denominator),
            "selective_accuracy": (bundle.selective_accuracy.numerator,
                                   bundle.selective_accuracy.denominator),
            "over_claim": (bundle.over_claim_rate.numerator,
                           bundle.over_claim_rate.denominator),
            "reportability": bundle.reportability,
        })
    return rows


def _pct(pair) -> str:
    k, n = pair
    return f"{k}/{n} = {100 * k / n:.1f}%" if n else f"{k}/{n} = --"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="RQ1 quote-gate x dual-model ablation grid.")
    ap.add_argument("--run-root", required=True)
    ap.add_argument("--anchors", default="str,drf,mom6")
    ap.add_argument("--allow-non-reportable", action="store_true")
    ap.add_argument("--json", default=None, dest="json_out")
    args = ap.parse_args(argv)

    run_root = (_REPO_ROOT / args.run_root) if not Path(args.run_root).is_absolute() \
        else Path(args.run_root)
    anchors = [a.strip() for a in args.anchors.split(",") if a.strip()]

    all_rows: list[dict] = []
    for anchor in anchors:
        rows = score_grid(anchor, run_root / DEFAULT_DIR_OF[anchor])
        require_reportable(rows[0]["reportability"],
                           allow_non_reportable=args.allow_non_reportable)
        all_rows.extend(rows)

    print(f"# RQ1 ablation grid -- {run_root.name}")
    if not all_rows[0]["reportability"].reportable:
        print(all_rows[0]["reportability"].banner)
    for anchor in anchors:
        print(f"\n## {anchor}")
        print(f"{'combo':38s} {'coverage':>18s} {'sel. accuracy':>18s} {'over-claim':>16s}")
        for r in (x for x in all_rows if x["anchor"] == anchor):
            print(f"{r['combo']:38s} {_pct(r['coverage']):>18s} "
                  f"{_pct(r['selective_accuracy']):>18s} {_pct(r['over_claim']):>16s}")

    # Pooled micro view per combo (numerators/denominators summed across anchors).
    print("\n## pooled (micro)")
    print(f"{'combo':38s} {'coverage':>18s} {'sel. accuracy':>18s} {'over-claim':>16s}")
    for label, _, _ in COMBOS:
        combo_rows = [r for r in all_rows if r["combo"] == label]
        pooled = {m: (sum(r[m][0] for r in combo_rows), sum(r[m][1] for r in combo_rows))
                  for m in ("coverage", "selective_accuracy", "over_claim")}
        print(f"{label:38s} {_pct(pooled['coverage']):>18s} "
              f"{_pct(pooled['selective_accuracy']):>18s} {_pct(pooled['over_claim']):>16s}")

    if args.json_out:
        out = Path(args.json_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        serialisable = [{k: v for k, v in r.items() if k != "reportability"}
                        for r in all_rows]
        out.write_text(json.dumps(serialisable, indent=2), encoding="utf-8")
        print(f"\n[ablation] json -> {out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
