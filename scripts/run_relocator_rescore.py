#!/usr/bin/env python
"""
The single relocate-then-certify re-scoring pass.

DIAGNOSTIC; NEVER A HEADLINE. One publish-as-found pass per frozen bar,
terminal — no second attempt at a different bar, ever. Reads the run archives
and recorded metrics reports, which are gitignored and not shipped; a clean
clone cannot run this until they are regenerated locally.

Replays the archived reportable run through the recorded D9 merge rule with
exactly one widening: a model quote passes the located test iff it EXACT-locates
(the production gate) OR the relocator accepts it at the frozen bar
(``librarian.relocator_diagnostic.accept_bar`` — the loader raises while it is
null, so this script structurally cannot run before calibration). Everything
else — normalisation, agreement, ship rule, scoring — is byte-identical to
``scripts/run_rq1_ablation.py``'s full-system replay, which reproduces the
recorded g3 headline exactly.

Cross-pins (fail-loud, abort before writing anything): the relocation-OFF arm
must reproduce, per archive, (1) every replayed field's shipped/abstained bit
and (2) the recorded coverage / selective-accuracy / over-claim numerators and
denominators. Primary = ``runs/corpus_anchors_report`` vs
``results/g3_report.json``; robustness = ``runs/bbw_4anchor_report``
(drf/crf/lrf) and ``runs/bbw_masked_report`` (drf, masked canonical text — the
text is routed through each archive's own ``run_manifest.json`` and its
recorded sha256 is verified against the on-disk file).
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import re
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from agents.librarian.config import load_canonical_text                    # noqa: E402
from agents.librarian.config.canonical_text import CanonicalText           # noqa: E402
from evaluation.gold_specs.gold_loader import load_gold_spec               # noqa: E402
from evaluation.harness.calibration_report import compute_metrics          # noqa: E402
from evaluation.harness.gold_calibration import score_anchor               # noqa: E402
from evaluation.harness.relocate import RelocateResult, relocate           # noqa: E402
from evaluation.harness.relocate_thresholds import (                       # noqa: E402
    RelocatorConfig,
    load_relocator_config,
)
from evaluation.harness.reportability import require_reportable            # noqa: E402
from evaluation.harness.run_artefacts import (                             # noqa: E402
    RunArtefacts,
    RunField,
    load_run,
)
from scripts.run_g3_score import DEFAULT_DIR_OF                            # noqa: E402
from scripts.run_locator_census import read_raw_segmented                  # noqa: E402
from scripts.run_p6a_replay import _normalise_or_none                      # noqa: E402

BANNER = "Relocate-then-certify — labelled diagnostic; never a headline"

# (section, run location, anchors, recorded record). The primary section's
# anchors live in per-paper subdirs of the run root; the robustness archives
# are single multi-spec dirs shared by their anchors.
PRIMARY_G3 = "results/g3_report.json"
FOURANCHOR_G3 = "results/g3_report_4anchor.json"
MASKED_G3 = "results/g3_masked_drf.json"

TRIPLE_KEYS = (("coverage", "coverage"), ("selective_accuracy", "selective_accuracy"),
               ("over_claim", "over_claim_rate"))

# The masked archive's strategy labels are entity-masked by design (the drf row
# reads "Downside Risk Factor (XF1)"), so gold-label matching cannot resolve it;
# the drf strategy sits at spec index 0, and the recorded-report cross-pin
# fails loud if this pin were ever the wrong strategy.
MASKED_INDEX_MAP: dict[str, int] = {"drf": 0}


def canonical_text_for(run_dir: Path) -> CanonicalText:
    """The archive's own canonical text, from its manifest, sha-verified.

    Exactly one ``canonical_texts`` entry must appear in the manifest's inputs;
    its recorded sha256 must match the on-disk file bytes (drift guard). This is
    what routes ``bbw_masked_report`` to the masked frozen text."""
    manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    inputs = manifest.get("inputs") or {}
    ct_entries = {p: sha for p, sha in inputs.items() if "canonical_texts" in p}
    if len(ct_entries) != 1:
        raise RuntimeError(
            f"{run_dir}: expected exactly one canonical-text input in run_manifest.json; "
            f"found {sorted(ct_entries)}"
        )
    rel_path, recorded_sha = next(iter(ct_entries.items()))
    path = _REPO_ROOT / rel_path
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual != recorded_sha:
        raise RuntimeError(
            f"{run_dir}: canonical text {rel_path} has drifted since the recorded run "
            f"(manifest {recorded_sha[:16]}..., on disk {actual[:16]}...); do not report"
        )
    return load_canonical_text(path)


def resolve_strategy(anchor: str, run_dir: Path) -> tuple[RunArtefacts, int]:
    """(artefacts, strategy_index) for the anchor in a possibly multi-spec dir,
    matched by the gold strategy_label — fail loud, never guessed.

    The gold header's ``strategy_label`` is an ``Inherited`` (its ``.value`` is
    the label string), while a trace header stores a plain string that may be a
    longer form of it (the real bbw archive stamps "Downside Risk Factor (DRF)"
    where the gold value is "DRF") — so matching is exact-on-the-unwrapped-value
    first, then unique substring containment; anything ambiguous raises."""
    n = len(sorted(run_dir.glob("spec_*.json")))
    if n == 0:
        raise RuntimeError(f"{run_dir}: no spec_*.json found")
    if n == 1:
        return load_run(run_dir), 0
    raw_label = load_gold_spec(anchor).header.strategy_label
    gold_label = getattr(raw_label, "value", raw_label)
    if not isinstance(gold_label, str) or gold_label == "":
        raise RuntimeError(f"gold {anchor!r} has no usable strategy_label value")
    arts: list[RunArtefacts] = []
    found: dict[int, str] = {}
    for i in range(n):
        art = load_run(run_dir, strategy_index=i)
        arts.append(art)
        found[i] = str(art.header.get("strategy_label", ""))
    token = re.compile(rf"\b{re.escape(gold_label)}\b")
    for predicate in (lambda label: label == gold_label,
                      lambda label: bool(token.search(label))):
        hits = [i for i, label in found.items() if predicate(label)]
        if len(hits) == 1:
            return arts[hits[0]], hits[0]
        if len(hits) > 1:
            raise RuntimeError(
                f"{run_dir}: gold {anchor!r} strategy_label {gold_label!r} matches "
                f"{len(hits)} specs {sorted(hits)}; found {found} — ambiguous, refusing"
            )
    raise RuntimeError(
        f"{run_dir}: {n} specs but none matches gold {anchor!r} strategy_label "
        f"{gold_label!r}; found {found}"
    )


def _reloc_entry(res: RelocateResult | None) -> dict | None:
    if res is None:
        return None
    entry = {"method": res.method, "score": round(res.score, 6), "reason": res.reason}
    if res.method == "relocated":
        # NB for consumers: on a cross-page relocation, l0_span indexes the raw
        # joined pair pages[page] + "\n" + pages[page + 1], not pages[page].
        entry.update({
            "locator": res.locator.to_dict(),
            "l0_span": list(res.l0_span),
            "cross_page": res.cross_page,
            "runner_up": round(res.runner_up, 6),
        })
    return entry


def replay_relocated(recorded: RunArtefacts, raw_a: dict, raw_b: dict, ct: CanonicalText,
                     cfg: RelocatorConfig | None) -> tuple[RunArtefacts, list[dict]]:
    """The ablation full-system replay with the located bit widened by ``cfg``.

    ``cfg is None`` -> the pure exact D9 replay (the cross-pin arm), semantics
    byte-identical to ``run_rq1_ablation.replay_combo(dual=True, quote=True)``.
    Otherwise each answered-with-quote model side passes iff exact-or-relocate
    accepts. Returns the variant plus a per-field relocation log."""
    fields: dict[str, RunField] = {}
    log: list[dict] = []
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

        res_a: RelocateResult | None = None
        res_b: RelocateResult | None = None
        if cfg is None:
            loc_a = bool(ra["answered"] and ra["quote"] and ct.locate(ra["quote"]) is not None)
            loc_b = bool(rb["answered"] and rb["quote"] and ct.locate(rb["quote"]) is not None)
        else:
            if ra["answered"] and ra["quote"]:
                res_a = relocate(ct, ra["quote"], bar=cfg.accept_bar,
                                 min_quote_chars=cfg.min_quote_chars,
                                 min_anchor_chars=cfg.min_anchor_chars)
            if rb["answered"] and rb["quote"]:
                res_b = relocate(ct, rb["quote"], bar=cfg.accept_bar,
                                 min_quote_chars=cfg.min_quote_chars,
                                 min_anchor_chars=cfg.min_anchor_chars)
            loc_a = bool(res_a is not None and res_a.accepted)
            loc_b = bool(res_b is not None and res_b.accepted)

        # The D9 dual + quote ship rule, byte-identical to run_rq1_ablation._ship.
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
            shipped_reason="relocator_replay",
            ship_choice=None,
            not_extracted=False,
        )
        if cfg is not None:
            log.append({
                "field": name,
                "a": _reloc_entry(res_a), "b": _reloc_entry(res_b),
                "shipped": shipped,
                "was_agree_qgf": "agree_quote_gate_failed" in rec.conditions,
            })
    return dataclasses.replace(recorded, fields=fields), log


def assert_shipped_set_matches(anchor: str, variant_off: RunArtefacts,
                               recorded: RunArtefacts, run_dir: Path) -> None:
    """Field-level half of the cross-pin: every replayed field's shipped bit
    must byte-match the recorded run's."""
    drifted = sorted(
        name for name, rec in recorded.fields.items()
        if variant_off.fields[name].shipped != rec.shipped
    )
    if drifted:
        raise RuntimeError(
            f"cross-pin failed for {anchor!r} ({run_dir}): the exact replay's "
            f"shipped/abstained set diverges from the recorded run on {drifted}; "
            "the archive or the replay semantics have drifted since the recorded "
            "run — do not report"
        )


def assert_newly_shipped_within_qgf(anchor: str, variant_on: RunArtefacts,
                                    variant_off: RunArtefacts
                                    ) -> tuple[list[str], set[str]]:
    """The strict-D9 invariant: relocation may only ever ship fields that were
    agree-but-quote-gate-failed under the exact replay. Returns
    (newly_shipped, agree_qgf); raises on any violation."""
    newly_shipped = sorted(
        name for name, rec in variant_on.fields.items()
        if rec.shipped and not variant_off.fields[name].shipped
    )
    agree_qgf = {
        name for name, rec in variant_off.fields.items()
        if "agree_quote_gate_failed" in rec.conditions
    }
    outside = sorted(set(newly_shipped) - agree_qgf)
    if outside:
        raise RuntimeError(
            f"{anchor!r}: relocation shipped fields outside the agree_quote_gate_failed "
            f"set ({outside}) — the strict D9 rule was violated; logic bug, do not report"
        )
    return newly_shipped, agree_qgf


def cross_pin_anchor(anchor: str, variant_off: RunArtefacts, recorded: RunArtefacts,
                     recorded_report: dict, run_dir: Path) -> dict:
    """The relocation-OFF arm must reproduce the recorded run exactly."""
    assert_shipped_set_matches(anchor, variant_off, recorded, run_dir)
    bundle = compute_metrics(score_anchor(anchor, run_dir, artefacts=variant_off), variant_off)
    triple = {}
    for ours, theirs in TRIPLE_KEYS:
        got = getattr(bundle, "over_claim_rate" if ours == "over_claim" else ours)
        want = recorded_report[theirs]
        if (got.numerator, got.denominator) != (want["numerator"], want["denominator"]):
            raise RuntimeError(
                f"cross-pin failed for {anchor!r}: replayed {ours} "
                f"{got.numerator}/{got.denominator} vs recorded "
                f"{want['numerator']}/{want['denominator']} — do not report"
            )
        triple[ours] = [got.numerator, got.denominator]

    # Incidence pin in the SAME universe the recorded report used: the scored
    # (gold-paired) field set of compute_metrics — NOT the raw trace field set,
    # which for a multi-leg anchor (crf) is strictly larger.
    recorded_qgf = (recorded_report.get("condition_incidence") or {}).get("agree_quote_gate_failed")
    if recorded_qgf is None:
        raise RuntimeError(
            f"cross-pin cannot run for {anchor!r}: the recorded report carries no "
            "condition_incidence.agree_quote_gate_failed — a silently skipped pin is "
            "no pin; do not report"
        )
    replayed_qgf = bundle.condition_incidence.get("agree_quote_gate_failed", 0)
    if replayed_qgf != recorded_qgf:
        raise RuntimeError(
            f"cross-pin failed for {anchor!r}: replayed agree_quote_gate_failed "
            f"incidence {replayed_qgf} vs recorded {recorded_qgf} — do not report"
        )
    return triple


def _triple_of(anchor: str, run_dir: Path, variant: RunArtefacts) -> dict:
    bundle = compute_metrics(score_anchor(anchor, run_dir, artefacts=variant), variant)
    return {
        "coverage": [bundle.coverage.numerator, bundle.coverage.denominator],
        "selective_accuracy": [bundle.selective_accuracy.numerator,
                               bundle.selective_accuracy.denominator],
        "over_claim": [bundle.over_claim_rate.numerator, bundle.over_claim_rate.denominator],
    }


def rescore_anchor(anchor: str, run_dir: Path, recorded_anchor: dict,
                   cfg: RelocatorConfig, *, allow_non_reportable: bool,
                   index: int | None = None) -> dict:
    """``index`` pins the strategy explicitly where label matching cannot work
    (the masked archive's labels are entity-masked by design); the cross-pin
    below still fails loud if the pinned index is the wrong strategy."""
    if index is None:
        art, index = resolve_strategy(anchor, run_dir)
    else:
        art = load_run(run_dir, strategy_index=index)
    require_reportable(art.reportability, allow_non_reportable=allow_non_reportable)
    raw_a, raw_b = read_raw_segmented(run_dir)[index]
    ct = canonical_text_for(run_dir)

    variant_off, _ = replay_relocated(art, raw_a, raw_b, ct, None)
    pin_triple = cross_pin_anchor(anchor, variant_off, art, recorded_anchor, run_dir)

    variant_on, reloc_log = replay_relocated(art, raw_a, raw_b, ct, cfg)
    # NB: this set lives in the RAW trace-field universe (the ship-rule
    # invariant's domain); the recorded-report incidence pin runs inside
    # cross_pin_anchor in the scored universe.
    newly_shipped, agree_qgf = assert_newly_shipped_within_qgf(anchor, variant_on, variant_off)

    # Outcome transitions off->on, matched on the gold dotted path through the
    # identical live scoring path (never reimplemented here).
    rows_off = {r.dotted_path: r.outcome.name
                for r in score_anchor(anchor, run_dir, artefacts=variant_off).rows}
    rows_on = {r.dotted_path: r.outcome.name
               for r in score_anchor(anchor, run_dir, artefacts=variant_on).rows}
    transitions = sorted(
        ({"dotted_path": p, "off": rows_off[p], "on": rows_on[p]}
         for p in rows_off if rows_off[p] != rows_on[p]),
        key=lambda t: t["dotted_path"])

    unrecovered = []
    for entry in reloc_log:
        if entry["was_agree_qgf"] and not entry["shipped"]:
            why = []
            for role in ("a", "b"):
                e = entry[role]
                if e is not None and not e["reason"].startswith(("exact", "accepted")):
                    why.append(f"quote_{role} {e['reason']} ({e['score']})")
            unrecovered.append({"field": entry["field"], "why": "; ".join(why) or "value path"})

    return {
        "run_dir": str(run_dir.relative_to(_REPO_ROOT)),
        "strategy_index": index,
        "cross_pin_triple": pin_triple,
        "triple_off": _triple_of(anchor, run_dir, variant_off),
        "triple_on": _triple_of(anchor, run_dir, variant_on),
        "newly_shipped": newly_shipped,
        "n_agree_quote_gate_failed": len(agree_qgf),
        "outcome_transitions": transitions,
        "unrecovered": unrecovered,
        "relocations": [e for e in reloc_log
                        if any(e[r] and e[r]["method"] == "relocated" for r in ("a", "b"))],
    }


def _pool(per_anchor: dict, key: str) -> dict:
    pooled = {}
    for metric in ("coverage", "selective_accuracy", "over_claim"):
        pooled[metric] = [
            sum(a[key][metric][0] for a in per_anchor.values()),
            sum(a[key][metric][1] for a in per_anchor.values()),
        ]
    return pooled


def _transition_counts(per_anchor: dict) -> dict:
    counts: dict[str, int] = {}
    for a in per_anchor.values():
        for t in a["outcome_transitions"]:
            k = f"{t['off']} -> {t['on']}"
            counts[k] = counts.get(k, 0) + 1
    return dict(sorted(counts.items()))


def run_section(anchors_dirs: list[tuple[str, Path]], recorded_path: Path,
                cfg: RelocatorConfig, *, allow_non_reportable: bool,
                index_map: dict[str, int] | None = None) -> dict:
    recorded = json.loads(recorded_path.read_text(encoding="utf-8"))
    per_anchor = {}
    for anchor, run_dir in anchors_dirs:
        per_anchor[anchor] = rescore_anchor(
            anchor, run_dir, recorded["anchors"][anchor], cfg,
            allow_non_reportable=allow_non_reportable,
            index=(index_map or {}).get(anchor))
    return {
        "recorded_report": str(recorded_path.relative_to(_REPO_ROOT)),
        "per_anchor": per_anchor,
        "pooled_off": _pool(per_anchor, "triple_off"),
        "pooled_on": _pool(per_anchor, "triple_on"),
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
        f"Bar {result['config']['accept_bar']} (calibration status "
        f"`{result['config']['calibration_status']}`); cross-pins passed for every archive.",
        "",
        "## Primary (reportable 3-anchor archive)",
        "",
        "| arm | coverage | sel. accuracy | over-claim |",
        "|---|---|---|---|",
        f"| recorded headline (pin) | {pct(p['pooled_off']['coverage'])} | "
        f"{pct(p['pooled_off']['selective_accuracy'])} | {pct(p['pooled_off']['over_claim'])} |",
        f"| relocate-then-certify | {pct(p['pooled_on']['coverage'])} | "
        f"{pct(p['pooled_on']['selective_accuracy'])} | {pct(p['pooled_on']['over_claim'])} |",
        "",
        "Outcome transitions (off -> on): "
        + (", ".join(f"{k}: {v}" for k, v in p["transition_counts"].items()) or "none"),
        "",
    ]
    for section in ("robustness_4anchor", "robustness_masked"):
        r = result.get(section)
        if r is None:
            continue
        lines += [
            f"## {section}",
            "",
            f"pooled off {pct(r['pooled_off']['coverage'])} -> on {pct(r['pooled_on']['coverage'])} "
            f"coverage; transitions: "
            + (", ".join(f"{k}: {v}" for k, v in r["transition_counts"].items()) or "none"),
            "",
        ]
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Relocate-then-certify re-scoring pass.")
    ap.add_argument("--json", required=True, dest="json_out")
    ap.add_argument("--allow-non-reportable", action="store_true")
    ap.add_argument("--skip-robustness", action="store_true")
    ap.add_argument("--protocol", choices=("qr1", "qr2"), default="qr1",
                    help="which registration's frozen bar governs the pass")
    args = ap.parse_args(argv)

    # Raises while the selected registration's accept_bar is null (stage gate).
    cfg = load_relocator_config(protocol=args.protocol)
    calib_glob = ("results/relocator_calibration.json" if args.protocol == "qr1"
                  else "results/relocator_calibration_qr2.json")
    calib_status = "unknown"
    calib_artifacts = sorted(_REPO_ROOT.glob(calib_glob))
    if calib_artifacts:
        calib = json.loads(calib_artifacts[-1].read_text(encoding="utf-8"))
        calib_status = calib.get("status", "unknown")

    primary_dirs = [(a, _REPO_ROOT / "runs" / "corpus_anchors_report" / DEFAULT_DIR_OF[a])
                    for a in ("str", "drf", "mom6")]
    result: dict = {
        "diagnostic": BANNER,
        "config": {
            "protocol": args.protocol,
            "accept_bar": cfg.accept_bar,
            "min_quote_chars": cfg.min_quote_chars,
            "min_anchor_chars": cfg.min_anchor_chars,
            "metric": cfg.metric,
            "thresholds_sha256": hashlib.sha256(
                (_REPO_ROOT / "docs" / "thresholds.yaml").read_bytes()).hexdigest(),
            "calibration_artifact": (str(calib_artifacts[-1].relative_to(_REPO_ROOT))
                                     if calib_artifacts else None),
            "calibration_status": calib_status,
        },
        "primary": run_section(primary_dirs, _REPO_ROOT / PRIMARY_G3, cfg,
                               allow_non_reportable=args.allow_non_reportable),
    }
    if not args.skip_robustness:
        four_dir = _REPO_ROOT / "runs" / "bbw_4anchor_report"
        result["robustness_4anchor"] = run_section(
            [(a, four_dir) for a in ("drf", "crf", "lrf")],
            _REPO_ROOT / FOURANCHOR_G3, cfg,
            allow_non_reportable=args.allow_non_reportable)
        masked_dir = _REPO_ROOT / "runs" / "bbw_masked_report"
        # The masked archive's labels are entity-masked (DRF -> "Downside Risk
        # Factor (XF1)" per the derived masked enum), so the drf strategy is
        # pinned to index 0 explicitly; the cross-pin against the recorded
        # masked report fails loud if that index were the wrong strategy.
        result["robustness_masked"] = run_section(
            [("drf", masked_dir)], _REPO_ROOT / MASKED_G3, cfg,
            allow_non_reportable=args.allow_non_reportable,
            index_map=MASKED_INDEX_MAP)

    out = Path(args.json_out)
    if not out.is_absolute():
        out = _REPO_ROOT / out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    out.with_suffix(".md").write_text(_render_md(result), encoding="utf-8")
    print(f"[rescore] json -> {out}", file=sys.stderr)
    p = result["primary"]
    print(json.dumps({"pooled_off": p["pooled_off"], "pooled_on": p["pooled_on"],
                      "transitions": p["transition_counts"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
