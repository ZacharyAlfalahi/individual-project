"""P1 codegen ablation (WS-C) — CLI.

`--dry-run` (default): renders and hashes every prompt, self-checks the sandbox
mechanism, and runs the oracle-vs-oracle sanity (each oracle round-tripped
through the output contract must score RUNS_RIGHT at the exact tier). Zero
generation calls, zero spend, no holdout contact.

`--execute [--phase dev|reported]`: refuses without a verified contract freeze
(I5) and the phase's credentials; then one-shot per (model, strategy),
sandboxed against the exported corr-family engine panel, scored vs the oracle,
archived, tabulated — under the contract §3/§9 budget guard.

  --phase dev       (DEFAULT for --execute): the FREE pair (Gemini 3.1-flash-lite
                    + Mistral-small). NON-reportable — the plumbing smoke test.
                    An accidental --execute can never hit the paid pair.
  --phase reported: the paid Phase-F pair (Claude Sonnet 4.6 + Gemini 3.5-flash),
                    under the $30/16k sub-cap. The reportable run.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
for _p in (str(_REPO_ROOT), str(_REPO_ROOT / "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from evaluation.codegen.ablation import run_scored_ablation  # noqa: E402
from evaluation.codegen.live_client import build_codegen_factory, load_budget  # noqa: E402
from evaluation.codegen.oracles import ORACLES, export_oracle_csv, load_oracle_series  # noqa: E402
from evaluation.codegen.panel_export import export_codegen_panel  # noqa: E402
from evaluation.codegen.runner import (  # noqa: E402
    STRATEGIES,
    contract_freeze_ok,
    load_models,
    prompt_asset_hashes,
    run_ablation,
)
from evaluation.codegen.sandbox import detect_mechanism, parse_output_csv  # noqa: E402
from evaluation.codegen.scoring import Verdict, load_scoring_thresholds, score_run  # noqa: E402


def _load_dotenv(path: Path) -> None:
    """Minimal .env loader: KEY=VALUE lines into os.environ (never overrides an existing var).
    Mirrors scripts/run_librarian._load_dotenv — the codegen CLI needs the env populated."""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip())


def _oracle_sanity(tmp_dir: Path) -> list[str]:
    """Round-trip every oracle through the output contract and the comparator."""
    lines: list[str] = []
    thresholds = load_scoring_thresholds()
    for strategy in ORACLES:
        csv_path = tmp_dir / f"oracle_{strategy}.csv"
        export_oracle_csv(strategy, csv_path)
        series = parse_output_csv(csv_path)
        result = score_run(strategy, "ok", None, load_oracle_series(strategy),
                           series, thresholds)
        status = "PASS" if (result.verdict is Verdict.RUNS_RIGHT and result.exact_tier) else "FAIL"
        lines.append(f"  oracle-vs-oracle {strategy}: {result.verdict.value} "
                     f"(exact_tier={result.exact_tier}) [{status}]")
        if status == "FAIL":
            raise SystemExit(f"oracle sanity failed for {strategy}: {result.to_dict()}")
    return lines


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true", default=True)
    mode.add_argument("--execute", dest="dry_run", action="store_false")
    ap.add_argument("--phase", choices=("dev", "reported"), default="dev",
                    help="dev = free non-reportable pair (default); reported = paid Phase-F pair")
    ap.add_argument("--scratch", default=str(_REPO_ROOT / "data" / "development" / "codegen"))
    args = ap.parse_args(argv)

    print("P1 codegen ablation — " + ("DRY RUN (zero generation calls)" if args.dry_run
                                      else f"EXECUTE (phase={args.phase})"))
    print(f"sandbox mechanism: {detect_mechanism()}")
    print("prompt assets:")
    for name, digest in prompt_asset_hashes().items():
        print(f"  {name}: {digest}")

    result = run_ablation(STRATEGIES, dry_run=True)
    print("prompt sha256 per strategy:")
    for strategy, digest in result["prompt_sha256"].items():
        print(f"  {strategy}: {digest}")

    scratch = Path(args.scratch)
    scratch.mkdir(parents=True, exist_ok=True)
    print("oracle-vs-oracle sanity:")
    for line in _oracle_sanity(scratch / "oracle_sanity"):
        print(line)

    if args.dry_run:
        print(f"phase_f pair (from thresholds): {result['models']}")
        print("dry run complete — no generation performed.")
        return 0

    # --- execute path -------------------------------------------------------
    ok, msg = contract_freeze_ok()
    if not ok:
        print(f"REFUSED: {msg}", file=sys.stderr)
        return 2

    _load_dotenv(_REPO_ROOT / ".env")
    models = load_models(args.phase)
    print(f"{args.phase} pair (from thresholds): {[m['model_id'] for m in models]}")
    missing = [m["api_key_env"] for m in models if not os.environ.get(m.get("api_key_env", ""))]
    if missing:
        print(f"REFUSED: missing {args.phase} credentials {missing} "
              f"(populate .env / export them)", file=sys.stderr)
        return 2

    budget = load_budget()
    if args.phase == "reported":
        print(f"REPORTED (paid) run under the contract §3/§9 sub-cap: "
              f"${budget.usd_cap:.0f} cumulative, {budget.per_call_output_token_cap} tokens/call.")
    else:
        print("DEV (free) run — NON-reportable plumbing smoke; spend logged, cap not binding.")

    print("exporting corr-family engine panel …")
    panel_path, panel_sha = export_codegen_panel()
    print(f"  {panel_path}  sha256={panel_sha[:16]}…")

    factory = build_codegen_factory(budget=budget, temperature=0.0)
    result = run_scored_ablation(STRATEGIES, phase=args.phase, client_factory=factory,
                                 budget=budget, panel_path=panel_path)

    print(f"verdicts: {result['verdict_counts']}")
    print(f"reportable={result['reportable']} (sku_match={result['sku_match']})")
    print(f"budget: spent ${result['budget']['spent_usd_estimate']:.4f} of "
          f"${result['budget']['usd_cap']:.0f} over {result['budget']['calls']} calls")

    out_path = scratch / f"p1_results_{args.phase}.json"
    out_path.write_text(json.dumps(result, indent=2, sort_keys=True, default=str), encoding="utf-8")
    print(f"results written: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
