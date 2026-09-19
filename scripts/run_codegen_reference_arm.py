#!/usr/bin/env python
"""Codegen reference arm over the oracle set — CLI (no model calls, no generation).

Where the registered coverage-boundary Arm B is empty (the deterministic router compiles no
member of the scale-layer corpus), the two coverage-boundary comparators have nothing to
compare. This driver computes the SAME two comparators over the oracle set, from the
artefacts a P1 run produces:

  * inter-model agreement — the two models' generated series for the same strategy;
  * codegen-vs-compiler divergence — each model's series against the hand-built oracle.

It is a descriptive REFERENCE RATE, not the registered Arm B, and it never substitutes for
it: the population is the gold-specification oracle set, not the routed scale-layer corpus.

Reads `<basis>/codegen/p1_results_reported.json` + that run's sandbox outputs and archive,
and the basis's oracle series. Every series is hash-checked against the run P1 archived.
Writes `<basis>/codegen/reference_arm/reference_arm.{json,md}` and refuses to overwrite.
Development data only; the holdout is never opened.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
for _p in (str(_REPO_ROOT), str(_REPO_ROOT / "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from evaluation.codegen.jsonio import dump_json  # noqa: E402
from evaluation.codegen.oracles import load_oracle_series, oracle_path  # noqa: E402
from evaluation.codegen.p2_metrics import load_p2_scoring_thresholds  # noqa: E402
from evaluation.codegen.reference_arm import (  # noqa: E402
    build_reference_arm,
    load_generated_series,
    render_reference_report,
)
from scripts import basis_inputs  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--basis", choices=basis_inputs.BASES, required=True,
                    help="which basis's P1 run to score")
    ap.add_argument("--phase", default="reported", choices=("dev", "reported"))
    args = ap.parse_args(argv)

    root = basis_inputs.basis_dir(args.basis, "codegen")
    results_path = root / f"p1_results_{args.phase}.json"
    if not results_path.is_file():
        print(f"REFUSED: {results_path} does not exist — run the P1 ablation for this basis "
              "first.", file=sys.stderr)
        return 2
    out_dir = root / "reference_arm"
    out_json, out_md = out_dir / "reference_arm.json", out_dir / "reference_arm.md"
    existing = [p for p in (out_json, out_md) if p.exists()]
    if existing:
        print(f"REFUSED: {', '.join(str(p) for p in existing)} already exist(s) — a run never "
              "overwrites a recorded result; move or delete it to re-run.", file=sys.stderr)
        return 2

    p1 = json.loads(results_path.read_text(encoding="utf-8"))
    models = tuple(p1["models"])
    if len(models) != 2:
        print(f"REFUSED: expected a model PAIR, got {models}", file=sys.stderr)
        return 2
    # P1 recorded where it actually read and wrote. Scoring its runs against anything else —
    # a different sandbox, a different oracle — would silently compare the wrong things.
    p1_inputs = p1.get("inputs") or {}
    sandbox_root = root / "sandbox"
    recorded_sandbox = p1_inputs.get("sandbox_root")
    if recorded_sandbox and Path(recorded_sandbox) != sandbox_root:
        print(f"REFUSED: P1 recorded sandbox_root {recorded_sandbox} but this run resolves "
              f"{sandbox_root} — the outputs to score are not where this run is looking.",
              file=sys.stderr)
        return 2
    if p1.get("generation_errors"):
        print(f"REFUSED: the P1 run has {len(p1['generation_errors'])} generation error(s) — "
              "its cells are missing data, not measured failures, so an agreement denominator "
              "built from them would be wrong.", file=sys.stderr)
        return 2
    # Use the oracles P1 itself scored against (falling back to the basis default only when
    # the record carries no inputs.factors_dir), and pin them: the generated side is byte-pinned, so
    # the reference side must be too.
    recorded_factors = p1_inputs.get("factors_dir")
    factors_dir = Path(recorded_factors) if recorded_factors else basis_inputs.factors_dir(args.basis)
    oracle_pins = {}
    for strategy in sorted({r["strategy"] for r in p1["runs"]}):
        path = oracle_path(strategy, factors_dir)
        oracle_pins[strategy] = {"path": str(path),
                                 "sha256": basis_inputs.sha256(path) if path.is_file() else None}

    series, sources = load_generated_series(
        p1["runs"], sandbox_root=sandbox_root, archive_root=root / "archive",
        phase=args.phase)
    result = build_reference_arm(
        series, lambda s: load_oracle_series(s, factors_dir),
        load_p2_scoring_thresholds(), model_ids=models, sources=sources)

    payload = {
        "experiment": "codegen_reference_arm",
        "population": ("oracle set (gold specifications) — NOT the registered coverage-boundary "
                       "Arm B"),
        "basis": args.basis,
        "phase": args.phase,
        "models": list(models),
        "p1_results": str(results_path),
        "p1_panel_sha256": p1.get("panel_sha256"),
        "p1_reportable": p1.get("reportable"),
        "p1_sku_match": p1.get("sku_match"),
        "p1_generation_errors": len(p1.get("generation_errors") or []),
        "factors_dir": str(factors_dir),
        "oracle_pins": oracle_pins,
        "model_calls": 0,
        "spend_usd": 0.0,
        **result.to_dict(),
    }
    meta = {
        "p1 run": str(results_path),
        "panel sha256": (p1.get("panel_sha256") or "—")[:16],
        "oracles": str(factors_dir),
        "generation": "none — scored from P1's recorded outputs (0 model calls, $0.00)",
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    out_json.write_text(dump_json(payload), encoding="utf-8")
    out_md.write_text(render_reference_report(result, meta), encoding="utf-8")

    dist = result.distribution
    print(f"reference arm ({args.basis}): {len(result.agreements)} inter-model pairs, "
          f"{len(result.divergences)} codegen-vs-compiler rows")
    print(f"  agreement rate {dist.get('agreement_rate')} "
          f"({dist.get('n_agree')}/{dist.get('n_scored')}); "
          f"correlation median {dist.get('correlation_median')}")
    print(f"results written: {out_json}")
    print(f"report written:  {out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
