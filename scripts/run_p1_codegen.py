"""P1 codegen ablation (WS-C) — CLI.

`--dry-run` (the ONLY runnable mode until Phase-F credentials + the
signed frozen mini-contract exist): renders and hashes every prompt,
self-checks the sandbox mechanism, and runs the oracle-vs-oracle sanity
(each oracle round-tripped through the output contract must score RUNS_RIGHT
at the exact tier). Zero generation calls, zero spend, no holdout contact.

`--execute`: refuses without credentials AND a verified contract freeze
(I5); then one-shot per (model, strategy), sandboxed, scored, archived.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
for _p in (str(_REPO_ROOT), str(_REPO_ROOT / "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from evaluation.codegen.oracles import ORACLES, export_oracle_csv, load_oracle_series  # noqa: E402
from evaluation.codegen.runner import (  # noqa: E402
    STRATEGIES,
    contract_freeze_ok,
    load_phase_f_models,
    prompt_asset_hashes,
    run_ablation,
)
from evaluation.codegen.sandbox import detect_mechanism, parse_output_csv  # noqa: E402
from evaluation.codegen.scoring import Verdict, load_scoring_thresholds, score_run  # noqa: E402


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
    ap.add_argument("--scratch", default=str(_REPO_ROOT / "data" / "development" / "codegen"))
    args = ap.parse_args(argv)

    print("P1 codegen ablation — " + ("DRY RUN (zero generation calls)" if args.dry_run
                                      else "EXECUTE"))
    print(f"sandbox mechanism: {detect_mechanism()}")
    print("prompt assets:")
    for name, digest in prompt_asset_hashes().items():
        print(f"  {name}: {digest}")

    result = run_ablation(STRATEGIES, dry_run=True)
    print("prompt sha256 per strategy:")
    for strategy, digest in result["prompt_sha256"].items():
        print(f"  {strategy}: {digest}")
    print(f"phase_f pair (from thresholds): {result['models']}")

    scratch = Path(args.scratch)
    scratch.mkdir(parents=True, exist_ok=True)
    print("oracle-vs-oracle sanity:")
    for line in _oracle_sanity(scratch / "oracle_sanity"):
        print(line)

    if args.dry_run:
        print("dry run complete — no generation performed.")
        return 0

    ok, msg = contract_freeze_ok()
    if not ok:
        print(f"REFUSED: {msg}", file=sys.stderr)
        return 2
    missing = [m["api_key_env"] for m in load_phase_f_models()
               if not os.environ.get(m.get("api_key_env", ""))]
    if missing:
        print(f"REFUSED: missing Phase-F credentials {missing} (E3 approval gate)",
              file=sys.stderr)
        return 2
    print("REFUSED: live execution additionally requires the approved budget sub-cap "
          "recorded in the mini-contract; wire a real client_factory at that point.",
          file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
