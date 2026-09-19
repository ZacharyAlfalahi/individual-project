#!/usr/bin/env python
"""
RQ2 coverage over the frozen 32-construction set (contract §5.2/§5.3; CI-5
denominator; built on the typed spec deserialiser).

Mirrors the anchors' own chain EXACTLY, run on the paid Phase-F emissions:
loaded emitted spec (spec_from_dict) -> `adapt_spec` with the hash-verified
standing subs -> refused ⇒ a TYPED refusal row; compiled ⇒ `run_strategy` on the
`corrected()` panel view (the identical loaders run_quant uses) ⇒ an executed
row. Papers whose extraction routed to review (dfps, exit 2: no spec set per
D31) contribute NOT_RUN rows for every registered construction — charged to the
end-to-end denominator, never laundered into a refusal layer (§8).

CI-9 (2026-09-06) additions, all additive to the scoring semantics above:
  * `--specs-dir` is repeatable, so a remediation run's second paper directory
    can be scored beside the first (each paper keeps its own emissions dir);
  * a run directory's `events.json` (written by run_librarian since CI-9) is
    consumed: an `assembly_incomplete` event yields a PER-CONSTRUCTION typed
    `not_run` row (`extraction_event` carries the kind) — still charged to the
    end-to-end denominator exactly like the blanket rows, never laundered into
    a refusal layer; only the attribution granularity changes;
  * the blanket NOT_RUN expansion applies only to constructions without a row
    already (no double counting), and a completeness check asserts every
    registered non-excluded construction has exactly one row before scoring.

Scoring is evaluation/harness/t3_coverage.score_corpus_coverage — the frozen
32 = 27 implement (FRR denominator) + 5 refuse (FIR / refusal-recall), the
153-family excluded by registration. The HEADLINE end-to-end rate is
C_end_to_end_full over the frozen 32; `layered_observed` is the observed-subset
diagnostic and is never headlined.

Zero LLM, zero network; deterministic given the run artefacts + dev panel.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from agents.librarian.adapter.adapt import adapt_spec  # noqa: E402
from agents.librarian.pipeline.spec_loader import spec_from_dict  # noqa: E402
from agents.quant.config.runner import StrategyResult, run_strategy  # noqa: E402
from agents.quant.library.run_config import corrected  # noqa: E402
from evaluation.harness.t3_coverage import (  # noqa: E402
    load_coverage_labels,
    score_corpus_coverage,
)
from scripts.run_quant import load_inputs, summarize_run  # noqa: E402

_DEFAULT_SPECS = _REPO_ROOT / "runs" / "corpus_corpus_report" / "bbw2021"
_NOT_RUN_PAPERS = {
    # dfps routed to review at extraction (exit 2): the
    # 'VaR' construction's sort_signal did not resolve, and v1 emits no partial
    # spec set (D31) -- every DFPS_2026 construction is therefore unobserved.
    "DFPS_2026": "extraction routed to review (D20/D31; no spec set emitted)",
}


def refusal_codes_of(result) -> list[str]:
    codes = [r.code.value for r in result.refusals]
    codes += [lc.result.code.value for lc in result.leg_calls
              if lc.refused and lc.result is not None]
    return codes


def observed_rows(specs_dirs, panel, subs,
                  not_run_papers=None) -> tuple[list[dict], dict]:
    """One outcome row per registered construction: adapt+run rows for the
    emitted specs, typed per-construction not_run rows for CI-9 assembly
    events, blanket not_run rows for the review-exit papers' remaining
    registered names."""
    if isinstance(specs_dirs, Path):        # back-compat: single-dir callers
        specs_dirs = [specs_dirs]
    if not_run_papers is None:
        not_run_papers = _NOT_RUN_PAPERS
    labels = load_coverage_labels()
    rows: list[dict] = []
    detail: dict[str, dict] = {}
    seen: set[tuple[str, str]] = set()

    def _add(row):
        key = (row["paper_id"], row["name"])
        if key in seen:
            raise RuntimeError(
                f"duplicate outcome row for {key} -- two run directories (or an "
                "events.json and a spec) claim the same construction; a build "
                "error to fix, never a row to drop silently")
        seen.add(key)
        rows.append(row)

    for specs_dir in specs_dirs:
        for spec_path in sorted(specs_dir.glob("spec_*.json")):
            spec = spec_from_dict(json.loads(spec_path.read_text(encoding="utf-8")))
            paper_id = spec.header.paper_id
            name = spec.header.strategy_label.value
            result = adapt_spec(spec, standing_subs=subs)
            if result.refused:
                codes = refusal_codes_of(result)
                _add({"paper_id": paper_id, "name": name,
                      "outcome": "refused", "refusal_codes": codes})
                detail[name] = {"disposition": "refused", "codes": codes,
                                "spec": spec_path.name}
                continue
            run_result = run_strategy(result, panel)
            if not isinstance(run_result, StrategyResult):
                raise RuntimeError(
                    f"{spec_path.name}: run_strategy returned "
                    f"{type(run_result).__name__} for a non-refused compile -- a "
                    "build error to fix, never an outcome to launder")
            _add({"paper_id": paper_id, "name": name,
                  "outcome": "executed", "refusal_codes": []})
            detail[name] = {"disposition": "executed", "spec": spec_path.name,
                            "summary": summarize_run(run_result)}

        # CI-9: per-construction assembly reviews recorded by run_librarian.
        events_path = specs_dir / "events.json"
        if events_path.exists():
            for ev in json.loads(events_path.read_text(encoding="utf-8")):
                if ev.get("kind") != "assembly_incomplete":
                    detail.setdefault("__events__", []).append(ev)
                    continue
                pid, name = ev.get("paper_id"), ev.get("construction_name")
                label = labels.label_of.get((pid, name))
                if label is None or label == "excluded":
                    # auxiliary / excluded constructions carry no denominator row
                    detail.setdefault("__events__", []).append(ev)
                    continue
                _add({"paper_id": pid, "name": name, "outcome": "not_run",
                      "refusal_codes": [],
                      "extraction_event": "assembly_incomplete"})
                detail[name] = {"disposition": "not_run",
                                "extraction_event": "assembly_incomplete",
                                "reason": ev.get("detail")}

    for paper_id, reason in not_run_papers.items():
        for (pid, name), label in sorted(labels.label_of.items()):
            if pid == paper_id and label != "excluded" and (pid, name) not in seen:
                _add({"paper_id": pid, "name": name,
                      "outcome": "not_run", "refusal_codes": []})
        detail[f"__{paper_id}__"] = {"disposition": "not_run", "reason": reason}

    registered = {k for k, label in labels.label_of.items() if label != "excluded"}
    missing = registered - seen
    if missing:
        raise RuntimeError(
            f"{len(missing)} registered construction(s) have no outcome row "
            f"(e.g. {sorted(missing)[:3]}) -- the headline is only computable "
            "over the complete frozen denominator; pass the missing paper's "
            "run directory or its --not-run-paper entry")
    return rows, detail


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="RQ2 coverage over the frozen 32-set.")
    ap.add_argument("--specs-dir", action="append", default=None,
                    help="run directory with spec_*.json (+ events.json); "
                         "repeatable (CI-9); default: the 2026-09-03 bbw2021 dir")
    ap.add_argument("--not-run-paper", action="append", default=None,
                    metavar="PAPER_ID=reason",
                    help="paper whose registered constructions (without a row "
                         "already) are charged not_run; default: the DFPS_2026 "
                         "2026-09-03 review exit")
    ap.add_argument("--out", default=None,
                    help="output JSON (default results/rq2_coverage.json; "
                         "never the registered 2026-09-04 artefact)")
    args = ap.parse_args(argv)

    specs_dirs = [Path(p) for p in (args.specs_dir or [str(_DEFAULT_SPECS)])]
    if args.not_run_paper is None:
        not_run = dict(_NOT_RUN_PAPERS)
    else:
        not_run = dict(entry.split("=", 1) for entry in args.not_run_paper)

    panel, subs = load_inputs(corrected())
    rows, detail = observed_rows(specs_dirs, panel, subs, not_run_papers=not_run)
    score = score_corpus_coverage(rows)

    out = Path(args.out) if args.out else (
        _REPO_ROOT / "results" / "rq2_coverage.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"score": score, "observed": rows,
                               "detail": detail}, indent=2, default=str),
                   encoding="utf-8")

    print(json.dumps(score, indent=2, default=str)[:2000])
    print(f"[t3_coverage] -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
