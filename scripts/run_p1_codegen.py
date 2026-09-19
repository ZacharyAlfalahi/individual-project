"""P1 codegen ablation (WS-C) — CLI.

`--dry-run` (default): renders and hashes every prompt, self-checks the sandbox
mechanism, and runs the oracle-vs-oracle sanity (each oracle round-tripped
through the output contract must score RUNS_RIGHT at the exact tier). Zero
generation calls, zero spend, no holdout contact.

`--execute [--phase dev|reported]`: refuses without a verified contract freeze
(I5); then one-shot per (model, strategy), sandboxed against the exported
corr-family engine panel, scored vs the oracle, archived, tabulated — under the
contract §3/§9 budget guard.

  --phase dev       (DEFAULT for --execute): the FREE pair (Gemini 3.1-flash-lite
                    + Mistral-small). NON-reportable — the plumbing smoke test.
                    An accidental --execute can never hit the paid pair.
  --phase reported: the paid Phase-F pair (Claude Sonnet 4.6 + Gemini 3.5-flash),
                    under the $30/16k sub-cap. The reportable run.

LLM policy (design decision 2026-09-11; evaluation/codegen/preflight.py): cached
responses (runs/p1_codegen/cache, key = model + seed + prompt) are replayed
first. Before any live call the run counts its cache misses and estimates an
upper-bound spend per vendor; it REFUSES if the Anthropic or Gemini estimate
exceeds `--max-usd-anthropic` / `--max-usd-gemini` (default $10 each). A run
with zero misses is a pure replay: no .env read, no credentials, no live client.
The pre-flight, cache hits/misses and metered usage land in the results JSON.

Consistent-basis inputs (`--basis {total_return,clean}`): the engine panel is
exported from `scripts/basis_inputs.PANELS[basis][0]` into
`results/consistent_basis/<basis>/codegen/` (with the scratch,
sandbox and archive dirs beside it), and the oracles default to that basis's
factor series (`basis_inputs.factors_dir(basis)`). `--factors-dir` overrides the
oracle location on either path. No `--basis` => the default panel, paths and oracles.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
for _p in (str(_REPO_ROOT), str(_REPO_ROOT / "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from evaluation.codegen.ablation import run_scored_ablation  # noqa: E402
from evaluation.codegen.live_client import (  # noqa: E402
    BudgetExceededError,
    load_budget,
)
from evaluation.codegen.oracles import ORACLES, export_oracle_csv, load_oracle_series  # noqa: E402
from evaluation.codegen.panel_export import CODEGEN_PANEL, export_codegen_panel  # noqa: E402
from evaluation.codegen.preflight import (  # noqa: E402
    PER_VENDOR_USD_CEILING,
    metered_by_model,
    replay_first_factory,
    run_preflight,
)
from evaluation.codegen.runner import (  # noqa: E402
    STRATEGIES,
    build_prompt,
    contract_freeze_ok,
    load_models,
    prompt_asset_hashes,
    run_ablation,
)
from evaluation.codegen.sandbox import detect_mechanism, parse_output_csv  # noqa: E402
from evaluation.codegen.scoring import Verdict, load_scoring_thresholds, score_run  # noqa: E402
from scripts import basis_inputs  # noqa: E402

DEFAULT_SCRATCH = _REPO_ROOT / "data" / "development" / "codegen"
CACHE_ROOT = _REPO_ROOT / "runs" / "p1_codegen" / "cache"
SANDBOX_ROOT = _REPO_ROOT / "runs" / "p1_codegen" / "sandbox"
DEFAULT_MAX_USD = PER_VENDOR_USD_CEILING


@dataclass(frozen=True)
class P1Paths:
    """Where one invocation reads and writes. ``source_panel``/``factors_dir`` None = the
    default resolution inside panel_export / oracles (kept byte-for-byte for the default run)."""
    scratch: Path
    panel_out: Path
    source_panel: Path | None
    factors_dir: Path | None
    sandbox_root: Path            # the archive lands beside it (``sandbox_root.parent / "archive"``)


def resolve_paths(basis: str | None, scratch: str | None, factors_dir: str | None) -> P1Paths:
    """An explicit ``--scratch`` isolates EVERY output of the invocation (results JSON, oracle sanity, the
    exported engine panel, sandbox and archive) under that directory, so a scratch run never touches the
    default output tree; without it the default / basis locations apply."""
    if basis is None:
        scratch_p = Path(scratch) if scratch else DEFAULT_SCRATCH
        return P1Paths(
            scratch=scratch_p,
            panel_out=(scratch_p / CODEGEN_PANEL.name) if scratch else CODEGEN_PANEL,
            source_panel=None,
            factors_dir=Path(factors_dir) if factors_dir else None,
            sandbox_root=(scratch_p / "sandbox") if scratch else SANDBOX_ROOT,
        )
    root = basis_inputs.basis_dir(basis, "codegen")
    scratch_p = Path(scratch) if scratch else root
    return P1Paths(
        scratch=scratch_p,
        panel_out=scratch_p / CODEGEN_PANEL.name,
        source_panel=basis_inputs.PANELS[basis][0],
        factors_dir=Path(factors_dir) if factors_dir else basis_inputs.factors_dir(basis),
        sandbox_root=scratch_p / "sandbox",
    )


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


def _oracle_sanity(tmp_dir: Path, factors_dir: Path | None = None) -> list[str]:
    """Round-trip every oracle through the output contract and the comparator."""
    lines: list[str] = []
    thresholds = load_scoring_thresholds()
    for strategy in ORACLES:
        csv_path = tmp_dir / f"oracle_{strategy}.csv"
        export_oracle_csv(strategy, csv_path, factors_dir)
        series = parse_output_csv(csv_path)
        result = score_run(strategy, "ok", None, load_oracle_series(strategy, factors_dir),
                           series, thresholds)
        status = "PASS" if (result.verdict is Verdict.RUNS_RIGHT and result.exact_tier) else "FAIL"
        lines.append(f"  oracle-vs-oracle {strategy}: {result.verdict.value} "
                     f"(exact_tier={result.exact_tier}) [{status}]")
        if status == "FAIL":
            raise SystemExit(f"oracle sanity failed for {strategy}: {result.to_dict()}")
    return lines


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true", default=True)
    mode.add_argument("--execute", dest="dry_run", action="store_false")
    ap.add_argument("--phase", choices=("dev", "reported"), default="dev",
                    help="dev = free non-reportable pair (default); reported = paid Phase-F pair")
    ap.add_argument("--scratch", default=None,
                    help="oracle-sanity + results JSON dir (default data/development/codegen; "
                         "with --basis, results/consistent_basis/<basis>/codegen)")
    ap.add_argument("--basis", choices=basis_inputs.BASES, default=None,
                    help="export the engine panel from basis_inputs.PANELS[basis][0] into the "
                         "consistent-basis results tree (omit for the default panel)")
    ap.add_argument("--factors-dir", default=None,
                    help="oracle factor series dir (default data/development/factors; with "
                         "--basis, results/consistent_basis/<basis>/factors)")
    ap.add_argument("--max-usd-anthropic", type=float, default=DEFAULT_MAX_USD,
                    help="pre-flight refusal cap on estimated Anthropic spend (default $10)")
    ap.add_argument("--max-usd-gemini", type=float, default=DEFAULT_MAX_USD,
                    help="pre-flight refusal cap on estimated Gemini spend (default $10)")
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    paths = resolve_paths(args.basis, args.scratch, args.factors_dir)

    print("P1 codegen ablation — " + ("DRY RUN (zero generation calls)" if args.dry_run
                                      else f"EXECUTE (phase={args.phase})"))
    if args.basis is not None:
        print(f"basis: {args.basis}  (panel source {paths.source_panel}; "
              f"oracles {paths.factors_dir})")
    print(f"sandbox mechanism: {detect_mechanism()}")
    print("prompt assets:")
    for name, digest in prompt_asset_hashes().items():
        print(f"  {name}: {digest}")

    result = run_ablation(STRATEGIES, dry_run=True)
    print("prompt sha256 per strategy:")
    for strategy, digest in result["prompt_sha256"].items():
        print(f"  {strategy}: {digest}")

    scratch = paths.scratch
    scratch.mkdir(parents=True, exist_ok=True)
    print("oracle-vs-oracle sanity:")
    for line in _oracle_sanity(scratch / "oracle_sanity", paths.factors_dir):
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

    models = load_models(args.phase)
    print(f"{args.phase} pair (from thresholds): {[m['model_id'] for m in models]}")
    budget = load_budget()

    # Pre-flight (before any credential, client or call): replay-first census + spend estimate.
    caps = {"anthropic": args.max_usd_anthropic, "gemini": args.max_usd_gemini}
    try:
        preflight = run_preflight(STRATEGIES, models, cache_root=CACHE_ROOT, budget=budget,
                                  caps_usd=caps, prompt_fn=build_prompt)
    except BudgetExceededError as exc:           # an unpriced model cannot be bounded
        print(f"REFUSED (pre-flight): {exc}", file=sys.stderr)
        return 2
    print(f"pre-flight: {preflight['cache']['hits']} cache hits, "
          f"{preflight['cache']['misses']} misses; estimated spend "
          f"{preflight['estimate']['per_vendor_usd']} (caps {caps})")
    if preflight["refused"]:
        print("REFUSED (pre-flight, no call made): " + "; ".join(preflight["violations"]),
              file=sys.stderr)
        return 2

    live_models = preflight["live_models"]
    if live_models:
        _load_dotenv(_REPO_ROOT / ".env")
        missing = [m["api_key_env"] for m in models
                   if m["model_id"] in live_models and not os.environ.get(m.get("api_key_env", ""))]
        if missing:
            print(f"REFUSED: missing {args.phase} credentials {missing} "
                  f"(populate .env / export them)", file=sys.stderr)
            return 2
    else:
        print("pure cache replay — no credentials read, no live client constructed.")

    if args.phase == "reported":
        print(f"REPORTED (paid) run under the contract §3/§9 sub-cap: "
              f"${budget.usd_cap:.0f} cumulative, {budget.per_call_output_token_cap} tokens/call.")
    else:
        print("DEV (free) run — NON-reportable plumbing smoke; spend logged, cap not binding.")

    print("exporting corr-family engine panel …")
    panel_path, panel_sha = export_codegen_panel(paths.panel_out, source_panel=paths.source_panel)
    print(f"  {panel_path}  sha256={panel_sha[:16]}…")

    result = run_scored_ablation(STRATEGIES, phase=args.phase,
                                 client_factory=replay_first_factory(live_models, budget),
                                 budget=budget, panel_path=panel_path, cache_root=CACHE_ROOT,
                                 sandbox_root=paths.sandbox_root, factors_dir=paths.factors_dir)
    result["panel_sha256"] = panel_sha
    result["inputs"] = {
        "basis": args.basis,
        "source_panel": None if paths.source_panel is None else str(paths.source_panel),
        "source_panel_sha256": (None if paths.source_panel is None
                                else basis_inputs.sha256(paths.source_panel)),
        "factors_dir": None if paths.factors_dir is None else str(paths.factors_dir),
        "sandbox_root": str(paths.sandbox_root),
    }
    result["llm_policy"] = {"preflight": preflight,
                            "metered_by_model": metered_by_model(budget)}

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
