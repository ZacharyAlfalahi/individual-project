#!/usr/bin/env python
"""
Strictness-frontier simulation over the scoped-extraction run archives.

A post-run diagnostic: it reads the scoped-extraction archives left by a corpus
extraction run (produced by scripts/run_librarian_corpus.py; the raw archives are
gitignored and not shipped), so a clean clone cannot run it until those archives
are regenerated locally.

Motivating question: if the extraction shipping rule were relaxed level by
level, what would each level buy in RQ2 coverage? Levels:

  L2  ship on dual agreement + located quote      (the registered system, as run)
  L1  ship on dual agreement, quote gate waived
  L0  additionally accept a single model's answer (dual architecture abandoned)

Method — a full simulation, not refusal classification: for every emitted scoped
spec, the level variant is built deterministically from the spec's own recorded
trace (per-field normalised values, agreement flags, final reasons — no new model
calls), then pushed through the production adapter (`adapt_spec` + verified standing
substitutions), and, where it compiles, the production runner on the corrected
development panel. Cascade refusals, conflict overrides, factory defaults, and
binding therefore all resolve exactly as the production chain would — nothing is
inferred from detail strings.

Ship rules per level (only for fields the run left UNKNOWN in the review lane):
  L1: final_reason == quote_match_failure AND both models' normalised values agree
      -> ship the agreed value.
  L0: additionally single_response (ship the answering model's value), and
      quote_match_failure with only one normalised value present. A genuine
      disagreement ships at no level.

Scope and boundaries:
  * Only value fields are simulated (Part-2 commons, leg dials, combiner kind).
    Structural fields (sort signal, control axis, Part 1) are never altered, so a
    construction that failed Guard 1 at emission stays out -- this simulation
    relaxes the SHIP rule, it does not re-run extraction.
  * Librarian emit-validation is intentionally not re-run on simulated specs: L1/L0
    are precisely the worlds where the quote-evidence requirement is waived.
  * Simulated fields carry tag DESIGN with a "frontier-sim" evidence note (a
    deliberate analysis substitution; STATED's D7 locator requirement is the very
    discipline the relaxed levels waive, so claiming STATED would misstate
    provenance). They exist only inside this analysis and are never written into
    any run directory or coverage artifact.
  * L2 is cross-pinned: re-adapting the emitted specs must reproduce the committed
    coverage artifact's refusal codes exactly (fail-loud drift guard).
  * A must-refuse construction executing at a relaxed level is recorded as a
    frontier FIR event -- a diagnostic about the relaxed rule, never a programme
    FIR.

Deterministic; zero network; reads only run archives, dev panel, and registers.
Output: results/strictness_frontier.json + .md (diagnostic artifact; the
registered coverage artifacts are untouched).
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from collections import Counter
from pathlib import Path

import yaml

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from agents.librarian.adapter.adapt import adapt_spec                     # noqa: E402
from agents.librarian.pipeline.spec_loader import spec_from_dict          # noqa: E402
from agents.quant.config.runner import StrategyResult, run_strategy       # noqa: E402
from agents.quant.library.run_config import corrected                     # noqa: E402
from evaluation.harness.t3_coverage import load_coverage_labels           # noqa: E402
from scripts.run_quant import load_inputs                                 # noqa: E402

_RUN_DIRS = (
    _REPO_ROOT / "runs" / "corpus_scoped" / "bbw2021",
    _REPO_ROOT / "runs" / "corpus_scoped" / "dfps",
)
_COVERAGE_ARTIFACT = _REPO_ROOT / "results" / "rq2_coverage_scoped.json"
_GOLD = {
    "BBW_2021": _REPO_ROOT / "evaluation" / "gold_specs" / "enum_bbw_2021.yaml",
    "DFPS_2026": _REPO_ROOT / "evaluation" / "gold_specs" / "enum_dfps_2026.yaml",
}
# Structural / non-value trace fields never simulated (see module docstring).
_NEVER_SIMULATED = frozenset((
    "formation_structure", "asset_class", "method_summary",
    "sort_signal", "control_axis",
    "sample_start", "sample_end", "universe_filter", "claimed_headline_metric",
))
_LEVELS = ("L2", "L1", "L0")


def refusal_codes_of(result) -> list[str]:
    codes = [r.code.value for r in result.refusals]
    codes += [lc.result.code.value for lc in result.leg_calls
              if lc.refused and lc.result is not None]
    return sorted(codes)


def value_nodes_of(spec_dict: dict) -> dict:
    """{trace_field_name: mutable Inherited-dict node} for every simulatable value
    field of one spec dict (part2 commons, leg 0 dials, combiner kind). Fails loud
    if the flat name-space is ambiguous (a part2/leg/combiner name collision would
    silently route a shipped value to the wrong node)."""
    p2 = spec_dict["part2"]
    common = {name: node for name, node in p2.items()
              if isinstance(node, dict) and "tag" in node
              and name not in _NEVER_SIMULATED}
    leg = {name: node for name, node in p2["legs"][0].items()
           if isinstance(node, dict) and "tag" in node
           and name not in _NEVER_SIMULATED}
    overlap = (set(common) & set(leg)) | ({"combiner"} & (set(common) | set(leg)))
    if overlap:
        raise RuntimeError(
            f"simulatable field name(s) {sorted(overlap)} exist in more than one "
            "spec location -- the flat trace-field mapping is ambiguous; "
            "namespace it before simulating")
    nodes = {**common, **leg}
    kind = p2.get("combiner", {}).get("kind")
    if isinstance(kind, dict) and "tag" in kind:
        nodes["combiner"] = kind
    return nodes


def shippable_value(rec: dict, level: str):
    """(ship?, value, latent_disagreement?) for one UNKNOWN review-lane trace
    record at a level. A latent disagreement is a record whose final_reason is
    quote_match_failure (the merge's reason precedence) while the two models'
    normalised VALUES also differ -- unshippable at every level, and invisible
    to any reason-label census."""
    reason = rec.get("final_reason")
    na, nb = rec.get("normalised_a"), rec.get("normalised_b")
    latent = (reason == "quote_match_failure"
              and na is not None and nb is not None and na != nb)
    if level == "L1":
        if reason == "quote_match_failure" and na is not None and na == nb:
            return True, na, latent
        return False, None, latent
    if level == "L0":
        if reason in ("quote_match_failure", "single_response"):
            if na is not None and nb is not None:
                return (na == nb), (na if na == nb else None), latent
            picked = na if na is not None else nb
            return (picked is not None), picked, latent
        return False, None, latent
    return False, None, latent


def simulate_level(spec_dict: dict, trace: dict, level: str) -> tuple[dict, int, int]:
    """(level-variant spec dict, n fields simulated, n latent disagreements).
    L2 returns the dict as-is (latent count still measured, for the census)."""
    sim = copy.deepcopy(spec_dict)
    nodes = value_nodes_of(sim)
    n_shipped = n_latent = 0
    for rec in trace["records"]:
        field = rec["field"]
        node = nodes.get(field)
        if node is None or node.get("tag") != "UNKNOWN":
            continue
        ship, value, latent = shippable_value(rec, "L0" if level == "L2" else level)
        if latent:
            n_latent += 1
        if level == "L2" or not ship:
            continue
        # DESIGN, not STATED: a simulation-injected value is a deliberate
        # analysis substitution, and STATED's D7 locator requirement is exactly
        # the discipline these levels waive -- forging one would be dishonest.
        node["value"] = value
        node["tag"] = "DESIGN"
        node["evidence"] = {
            "note": (f"frontier-sim {level}: value shipped from the recorded "
                     f"trace (run outcome was UNKNOWN/{rec.get('final_reason')}; "
                     f"model quote did not satisfy the registered gate); "
                     "diagnostic simulation only, never a run artefact"),
        }
        n_shipped += 1
    return sim, n_shipped, n_latent


def analyze(panel, subs) -> dict:
    committed = json.loads(_COVERAGE_ARTIFACT.read_text(encoding="utf-8"))
    # Pin against EVERY observed row (an executed row pins to zero refusal codes),
    # so a legitimately-executed spec never trips a misleading "drift" error.
    committed_codes = {
        (row["paper_id"], row["name"]): sorted(row["refusal_codes"])
        for row in committed["observed"] if row["outcome"] in ("refused", "executed")
    }
    labels = load_coverage_labels()

    per_spec, n_pinned = [], 0
    for run_dir in _RUN_DIRS:
        for spec_path in sorted(run_dir.glob("spec_*.json")):
            idx = spec_path.stem.split("_")[1]
            trace = json.loads((run_dir / f"trace_{idx}.json").read_text(encoding="utf-8"))
            spec_dict = json.loads(spec_path.read_text(encoding="utf-8"))
            name = spec_dict["header"]["strategy_label"]["value"]
            if trace["header"]["strategy_label"] != name:
                raise RuntimeError(
                    f"{spec_path.name}/trace_{idx}.json label mismatch "
                    f"({name!r} vs {trace['header']['strategy_label']!r}) -- "
                    "spec/trace pairing broke; never simulate on the wrong trace")
            paper = spec_dict["header"]["paper_id"]
            label = labels.label_of.get((paper, name))
            row = {"paper_id": paper, "name": name, "label": label, "levels": {}}

            for level in _LEVELS:
                sim, n_shipped, n_latent = simulate_level(spec_dict, trace, level)
                result = adapt_spec(spec_from_dict(sim), standing_subs=subs)
                codes = refusal_codes_of(result)
                outcome = "refused"
                if not result.refused:
                    run_result = run_strategy(result, panel)
                    outcome = ("executed" if isinstance(run_result, StrategyResult)
                               else f"run_returned_{type(run_result).__name__}")
                row["levels"][level] = {
                    "n_fields_simulated": n_shipped,
                    "outcome": outcome,
                    "refusal_codes": dict(Counter(codes)),
                }
                if level == "L2":
                    row["n_latent_disagreements"] = n_latent
                if level == "L2":
                    if codes != committed_codes.get((paper, name)):
                        raise RuntimeError(
                            f"L2 cross-pin failed for {(paper, name)}: {codes} vs "
                            f"{committed_codes.get((paper, name))} -- adapter or "
                            "inputs drifted since the coverage run; do not report")
                    n_pinned += 1
            per_spec.append(row)

    if n_pinned != len(committed_codes):
        raise RuntimeError(f"cross-pinned {n_pinned} specs but the committed "
                           f"artifact records {len(committed_codes)} refused rows")

    frontier = {}
    for level in _LEVELS:
        compiled = [r for r in per_spec if r["levels"][level]["outcome"] != "refused"]
        executed = [r for r in per_spec if r["levels"][level]["outcome"] == "executed"]
        frontier[level] = {
            "n_compiled": len(compiled),
            "n_executed": len(executed),
            "executed": [{"name": r["name"], "paper_id": r["paper_id"],
                          "label": r["label"]} for r in executed],
            # A must-refuse spec COMPILING is already the safety event of
            # interest (the adapter's gate passed it); execution is not required.
            "frontier_fir_events": [
                {"name": r["name"], "paper_id": r["paper_id"],
                 "outcome": r["levels"][level]["outcome"]}
                for r in compiled if r["label"] == "refuse"],
        }
    return {"per_spec": per_spec, "frontier": frontier,
            "cross_pin": {"n_specs": n_pinned},
            "n_latent_disagreements_total": sum(
                r["n_latent_disagreements"] for r in per_spec)}


_DIR_PAPER = {"bbw2021": "BBW_2021", "dfps": "DFPS_2026"}


def never_emitted_detail(run_dirs=_RUN_DIRS, dir_paper=None, gold_paths=None) -> list[dict]:
    """Dual-model sort_kind/control_axis answers for the never-emitted
    constructions, reconstructed from the raw archives with framing asserts.

    Dispositions are cross-checked against the run's recorded events, never
    assigned by elimination alone: the never-emitted count must equal the
    emission-error event count, event-carried construction names (stamped for
    attribution) must match the derived set exactly, and the
    ``guard1_emission_refusal`` label is used only when the event's own detail
    names Guard 1 -- otherwise the generic ``emission_refusal`` is reported.

    NOTE the raw archives carry no per-record construction id, so the framing is
    positional (records attributed via the always-first ``formation_structure``
    delimiter, in gold order -- the assembler's deterministic iteration order);
    the per-model construction counts are asserted, and a framing drift raises
    rather than mis-attributing."""
    dir_paper = dict(_DIR_PAPER if dir_paper is None else dir_paper)
    gold_paths = dict(_GOLD if gold_paths is None else gold_paths)
    detail = []
    for run_dir in run_dirs:
        if run_dir.name not in dir_paper:
            raise RuntimeError(f"run dir {run_dir.name!r} has no registered paper "
                               "mapping -- extend _DIR_PAPER deliberately")
        paper = dir_paper[run_dir.name]
        emitted = set()
        for spec_path in sorted(run_dir.glob("spec_*.json")):
            d = json.loads(spec_path.read_text(encoding="utf-8"))
            if d["header"]["paper_id"] != paper:
                raise RuntimeError(f"{spec_path.name}: paper_id "
                                   f"{d['header']['paper_id']!r} != {paper!r}")
            emitted.add(d["header"]["strategy_label"]["value"])
        events_path = run_dir / "events.json"
        events = (json.loads(events_path.read_text(encoding="utf-8"))
                  if events_path.exists() else [])
        assembly_named = {e.get("construction_name") for e in events
                          if e.get("kind") == "assembly_incomplete"}
        emission_events = [e for e in events
                           if e.get("kind") == "LibrarianEmissionError"]
        gold = yaml.safe_load(gold_paths[paper].read_text(encoding="utf-8"))
        names = [c["name"] for c in gold["constructions"]
                 if c.get("class") == "strategy"]
        never = [n for n in names if n not in emitted and n not in assembly_named]

        # Disposition cross-checks: never assigned by elimination alone.
        if len(never) != len(emission_events):
            raise RuntimeError(
                f"{run_dir.name}: {len(never)} never-emitted constructions but "
                f"{len(emission_events)} emission-error events -- a construction "
                "was lost to something unrecorded; refuse to label it")
        event_names = {e.get("construction_name") for e in emission_events
                       if e.get("construction_name")}
        if event_names and event_names != set(never):
            raise RuntimeError(
                f"{run_dir.name}: emission-event construction names {sorted(event_names)} "
                f"!= derived never-emitted set {sorted(never)}")
        all_guard1 = all("Guard 1" in (e.get("detail") or "") for e in emission_events)
        disposition = ("guard1_emission_refusal" if all_guard1 and emission_events
                       else "emission_refusal")

        answers = {}
        for model in ("a", "b"):
            per, idx = {}, -1
            path = run_dir / "raw" / f"raw_model_{model}.jsonl"
            for line in path.open(encoding="utf-8"):
                rec = json.loads(line)
                if rec.get("field") == "formation_structure":
                    idx += 1
                if not (0 <= idx < len(names)):
                    raise RuntimeError(f"{path.name}: record outside construction "
                                       f"framing (idx={idx})")
                per.setdefault(names[idx], {})[rec["field"]] = rec.get("parsed")
            if idx + 1 != len(names):
                raise RuntimeError(f"{path.name}: framed {idx + 1} constructions; "
                                   f"gold lists {len(names)}")
            answers[model] = per

        def _val(model, name, field):
            p = (answers[model].get(name) or {}).get(field) or {}
            return p.get("value", p.get("raw", p.get("concept_id")))

        for name in never:
            detail.append({
                "paper_id": paper, "name": name,
                "disposition": disposition,
                "sort_kind": {m: _val(m, name, "sort_kind") for m in ("a", "b")},
                "control_axis": {m: _val(m, name, "control_axis") for m in ("a", "b")},
            })
    return detail


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="Strictness-frontier simulation over the run archives")
    ap.add_argument("--out", default=None,
                    help="output JSON (default results/strictness_frontier.json)")
    args = ap.parse_args(argv)

    panel, subs = load_inputs(corrected())
    core = analyze(panel, subs)
    guard1 = never_emitted_detail()

    result = {
        "component": "strictness_frontier_simulation",
        "method": ("per-level spec variants built from recorded traces; real "
                   "adapter + runner decide; L2 cross-pinned to the committed "
                   "coverage artifact; diagnostic only, never a headline"),
        "inputs": [str(p.relative_to(_REPO_ROOT)) for p in _RUN_DIRS]
                  + [str(_COVERAGE_ARTIFACT.relative_to(_REPO_ROOT))],
        **core,
        "never_emitted": guard1,
    }
    out = Path(args.out) if args.out else (
        _REPO_ROOT / "results" / "strictness_frontier.json")
    out.write_text(json.dumps(result, indent=2, sort_keys=True, default=str),
                   encoding="utf-8")

    md = ["# Strictness-frontier simulation (diagnostic; upper reaches of the dial)",
          "", "| Level | compiled | executed | frontier FIR events |", "|---|---|---|---|"]
    for level in _LEVELS:
        f = result["frontier"][level]
        md.append(f"| {level} | {f['n_compiled']} of {core['cross_pin']['n_specs']} "
                  f"| {f['n_executed']} | {len(f['frontier_fir_events'])} |")
    md += ["", "Never-emitted (Guard-1) dual-model answers:"]
    for g in guard1:
        md.append(f"- {g['name']}: sort_kind A/B = {g['sort_kind']['a']}/"
                  f"{g['sort_kind']['b']}; control_axis A/B = "
                  f"{g['control_axis']['a']}/{g['control_axis']['b']}")
    out.with_suffix(".md").write_text("\n".join(md) + "\n", encoding="utf-8")

    print(json.dumps({lvl: {k: v for k, v in result["frontier"][lvl].items()
                            if k != "executed"} for lvl in _LEVELS}, indent=2))
    print(f"[strictness_frontier] -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
