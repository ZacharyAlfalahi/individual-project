#!/usr/bin/env python
"""
The L2-gate ablation (DIAGNOSTIC; never a headline).

Tests whether gate-failing
quotes locate VERBATIM at ladder level L2 (de-hyphenation) and fail only at
the recorded L1 operating point. Reads the run archives and recorded metrics
reports, which are gitignored and not shipped; a clean clone cannot run this
until they are regenerated locally.

Replays the archived responses through the recorded D9 merge rule with exactly
one change: the located test runs at ``level="L2"`` instead of the canonical
text's recorded level. L2 is a frozen level of the normalisation
ladder — there is no tunable constant anywhere in this instrument, no
calibration, and no bar. The production gate's level is a registered contract
and is NOT changed here.

Cross-pins (fail-loud, abort before writing): the recorded-level arm must
byte-reproduce the recorded shipped/abstained sets and metric
numerators/denominators for every archive.
Results are published as found.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from agents.librarian.config.canonical_text import CanonicalText           # noqa: E402
from evaluation.harness.calibration_report import compute_metrics          # noqa: E402
from evaluation.harness.gold_calibration import score_anchor               # noqa: E402
from evaluation.harness.reportability import require_reportable            # noqa: E402
from evaluation.harness.run_artefacts import RunArtefacts, RunField        # noqa: E402
from scripts.run_g3_score import DEFAULT_DIR_OF                            # noqa: E402
from scripts.run_locator_census import read_raw_segmented                  # noqa: E402
from scripts.run_p6a_replay import _normalise_or_none                      # noqa: E402
from evaluation.harness.run_artefacts import load_run                      # noqa: E402
from scripts.run_relocator_rescore import (                                # noqa: E402
    FOURANCHOR_G3,
    MASKED_G3,
    MASKED_INDEX_MAP,
    PRIMARY_G3,
    assert_newly_shipped_within_qgf,
    canonical_text_for,
    cross_pin_anchor,
    resolve_strategy,
)

BANNER = "L2-gate ablation — labelled diagnostic; never a headline"


def replay_at_level(recorded: RunArtefacts, raw_a: dict, raw_b: dict, ct: CanonicalText,
                    level: str | None) -> RunArtefacts:
    """The ablation full-system replay with the locate LEVEL as the only knob.

    ``level=None`` -> the recorded default (the text's own ladder level; the
    cross-pin arm, byte-identical to ``run_rq1_ablation.replay_combo(dual=True,
    quote=True)``). ``level="L2"`` -> the widened arm."""
    fields: dict[str, RunField] = {}
    for name, rec in recorded.fields.items():
        if rec.not_extracted:
            fields[name] = rec
            continue
        ra, rb = raw_a.get(name), raw_b.get(name)
        if ra is None or rb is None:
            raise RuntimeError(f"{recorded.run_dir}: field {name!r} missing from a raw archive")
        if ra["kind"] == "method_summary":
            fields[name] = rec
            continue

        norm_a, _ = _normalise_or_none(name, ra["value"], ra["kind"])
        norm_b, _ = _normalise_or_none(name, rb["value"], rb["kind"])
        loc_a = bool(ra["answered"] and ra["quote"]
                     and ct.locate(ra["quote"], level=level) is not None)
        loc_b = bool(rb["answered"] and rb["quote"]
                     and ct.locate(rb["quote"], level=level) is not None)
        shipped = bool(
            ra["answered"] and rb["answered"] and norm_a is not None and norm_a == norm_b
            and loc_a and loc_b
        )
        value = norm_a if shipped else None
        fields[name] = RunField(
            field=name,
            a_answered=ra["answered"], b_answered=rb["answered"],
            a_quote=ra["quote"], b_quote=rb["quote"],
            a_located=loc_a, b_located=loc_b,
            a_model_id=ra["model_id"], b_model_id=rb["model_id"],
            normalised_a=value if shipped else norm_a,
            normalised_b=norm_b,
            final_tag="STATED" if shipped else "UNKNOWN",
            shipped_reason="l2_ablation_replay",
            ship_choice=None,
            not_extracted=False,
        )
    return dataclasses.replace(recorded, fields=fields)


def _triple(anchor: str, run_dir: Path, variant: RunArtefacts) -> dict:
    bundle = compute_metrics(score_anchor(anchor, run_dir, artefacts=variant), variant)
    return {
        "coverage": [bundle.coverage.numerator, bundle.coverage.denominator],
        "selective_accuracy": [bundle.selective_accuracy.numerator,
                               bundle.selective_accuracy.denominator],
        "over_claim": [bundle.over_claim_rate.numerator, bundle.over_claim_rate.denominator],
    }


def ablate_anchor(anchor: str, run_dir: Path, recorded_anchor: dict,
                  *, allow_non_reportable: bool, index: int | None = None) -> dict:
    """``index`` pins the strategy where label matching cannot work (the masked
    archive); the cross-pin fails loud if the pinned index is wrong."""
    if index is None:
        art, index = resolve_strategy(anchor, run_dir)
    else:
        art = load_run(run_dir, strategy_index=index)
    require_reportable(art.reportability, allow_non_reportable=allow_non_reportable)
    raw_a, raw_b = read_raw_segmented(run_dir)[index]
    ct = canonical_text_for(run_dir)

    variant_off = replay_at_level(art, raw_a, raw_b, ct, None)
    pin_triple = cross_pin_anchor(anchor, variant_off, art, recorded_anchor, run_dir)

    variant_l2 = replay_at_level(art, raw_a, raw_b, ct, "L2")
    newly_shipped, agree_qgf = assert_newly_shipped_within_qgf(anchor, variant_l2, variant_off)

    rows_off = {r.dotted_path: r.outcome.name
                for r in score_anchor(anchor, run_dir, artefacts=variant_off).rows}
    rows_l2 = {r.dotted_path: r.outcome.name
               for r in score_anchor(anchor, run_dir, artefacts=variant_l2).rows}
    transitions = sorted(
        ({"dotted_path": p, "off": rows_off[p], "l2": rows_l2[p]}
         for p in rows_off if rows_off[p] != rows_l2[p]),
        key=lambda t: t["dotted_path"])

    return {
        "run_dir": str(run_dir.relative_to(_REPO_ROOT)),
        "strategy_index": index,
        "cross_pin_triple": pin_triple,
        "triple_off": _triple(anchor, run_dir, variant_off),
        "triple_l2": _triple(anchor, run_dir, variant_l2),
        "newly_shipped": newly_shipped,
        "n_agree_quote_gate_failed": len(agree_qgf),
        "outcome_transitions": transitions,
    }


def _pool(per_anchor: dict, key: str) -> dict:
    return {
        metric: [sum(a[key][metric][0] for a in per_anchor.values()),
                 sum(a[key][metric][1] for a in per_anchor.values())]
        for metric in ("coverage", "selective_accuracy", "over_claim")
    }


def _transition_counts(per_anchor: dict) -> dict:
    counts: dict[str, int] = {}
    for a in per_anchor.values():
        for t in a["outcome_transitions"]:
            k = f"{t['off']} -> {t['l2']}"
            counts[k] = counts.get(k, 0) + 1
    return dict(sorted(counts.items()))


def _four_anchor(primary: dict, robustness_4anchor: dict) -> dict:
    """The RQ1 reference-set pool over the four anchor golds
    (str + drf + mom6 + crf), scored at the same L2 de-hyphenation gate.

    str/drf/mom6 come straight from the ``primary`` section (their recorded
    ``corpus_anchors_report`` runs + PRIMARY_G3 golds); crf is grafted from the
    ``robustness_4anchor`` section (its ``bbw_4anchor_report`` run + FOURANCHOR_G3
    gold). No re-scoring — this pools the already-cross-pinned per-anchor triples,
    so the four-anchor number is byte-consistent with the two source sections."""
    per_anchor = {a: primary["per_anchor"][a] for a in ("str", "drf", "mom6")}
    per_anchor["crf"] = robustness_4anchor["per_anchor"]["crf"]
    return {
        "anchors": ["str", "drf", "mom6", "crf"],
        "per_anchor": per_anchor,
        "pooled_off": _pool(per_anchor, "triple_off"),
        "pooled_l2": _pool(per_anchor, "triple_l2"),
        "transition_counts": _transition_counts(per_anchor),
    }


def run_section(anchors_dirs: list[tuple[str, Path]], recorded_path: Path,
                *, allow_non_reportable: bool,
                index_map: dict[str, int] | None = None) -> dict:
    recorded = json.loads(recorded_path.read_text(encoding="utf-8"))
    per_anchor = {
        anchor: ablate_anchor(anchor, run_dir, recorded["anchors"][anchor],
                              allow_non_reportable=allow_non_reportable,
                              index=(index_map or {}).get(anchor))
        for anchor, run_dir in anchors_dirs
    }
    return {
        "recorded_report": str(recorded_path.relative_to(_REPO_ROOT)),
        "per_anchor": per_anchor,
        "pooled_off": _pool(per_anchor, "triple_off"),
        "pooled_l2": _pool(per_anchor, "triple_l2"),
        "transition_counts": _transition_counts(per_anchor),
    }


def _render_md(result: dict) -> str:
    def pct(pair):
        k, n = pair
        return f"{k}/{n} = {100 * k / n:.1f}%" if n else f"{k}/{n}"

    p = result["primary"]
    lines = [
        f"# {BANNER}",
        "",
        "| arm | coverage | sel. accuracy | over-claim |",
        "|---|---|---|---|",
        f"| recorded level (pin) | {pct(p['pooled_off']['coverage'])} | "
        f"{pct(p['pooled_off']['selective_accuracy'])} | {pct(p['pooled_off']['over_claim'])} |",
        f"| L2 gate | {pct(p['pooled_l2']['coverage'])} | "
        f"{pct(p['pooled_l2']['selective_accuracy'])} | {pct(p['pooled_l2']['over_claim'])} |",
        "",
        "Outcome transitions (off -> L2): "
        + (", ".join(f"{k}: {v}" for k, v in p["transition_counts"].items()) or "none"),
        "",
    ]
    fa = result.get("four_anchor")
    if fa is not None:
        lines += [
            "## four_anchor (RQ1 reference set: str + drf + mom6 + crf)",
            "",
            "| arm | coverage | sel. accuracy | over-claim |",
            "|---|---|---|---|",
            f"| recorded level (pin) | {pct(fa['pooled_off']['coverage'])} | "
            f"{pct(fa['pooled_off']['selective_accuracy'])} | {pct(fa['pooled_off']['over_claim'])} |",
            f"| L2 gate | {pct(fa['pooled_l2']['coverage'])} | "
            f"{pct(fa['pooled_l2']['selective_accuracy'])} | {pct(fa['pooled_l2']['over_claim'])} |",
            "",
            "Outcome transitions (off -> L2): "
            + (", ".join(f"{k}: {v}" for k, v in fa["transition_counts"].items()) or "none"),
            "",
        ]
    for section in ("robustness_4anchor", "robustness_masked"):
        r = result.get(section)
        if r is None:
            continue
        lines += [
            f"## {section}",
            "",
            f"pooled coverage {pct(r['pooled_off']['coverage'])} -> "
            f"{pct(r['pooled_l2']['coverage'])}; transitions: "
            + (", ".join(f"{k}: {v}" for k, v in r["transition_counts"].items()) or "none"),
            "",
        ]
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="L2-gate ablation (diagnostic).")
    ap.add_argument("--json", required=True, dest="json_out")
    ap.add_argument("--allow-non-reportable", action="store_true")
    ap.add_argument("--skip-robustness", action="store_true")
    args = ap.parse_args(argv)

    primary_dirs = [(a, _REPO_ROOT / "runs" / "corpus_anchors_report" / DEFAULT_DIR_OF[a])
                    for a in ("str", "drf", "mom6")]
    result: dict = {
        "diagnostic": BANNER,
        "primary": run_section(primary_dirs, _REPO_ROOT / PRIMARY_G3,
                               allow_non_reportable=args.allow_non_reportable),
    }
    if not args.skip_robustness:
        four_dir = _REPO_ROOT / "runs" / "bbw_4anchor_report"
        result["robustness_4anchor"] = run_section(
            [(a, four_dir) for a in ("drf", "crf", "lrf")],
            _REPO_ROOT / FOURANCHOR_G3,
            allow_non_reportable=args.allow_non_reportable)
        masked_dir = _REPO_ROOT / "runs" / "bbw_masked_report"
        result["robustness_masked"] = run_section(
            [("drf", masked_dir)], _REPO_ROOT / MASKED_G3,
            allow_non_reportable=args.allow_non_reportable,
            index_map=MASKED_INDEX_MAP)
        result["four_anchor"] = _four_anchor(
            result["primary"], result["robustness_4anchor"])

    out = Path(args.json_out)
    if not out.is_absolute():
        out = _REPO_ROOT / out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    out.with_suffix(".md").write_text(_render_md(result), encoding="utf-8")
    print(f"[l2_ablation] json -> {out}", file=sys.stderr)
    p = result["primary"]
    print(json.dumps({"pooled_off": p["pooled_off"], "pooled_l2": p["pooled_l2"],
                      "transitions": p["transition_counts"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
