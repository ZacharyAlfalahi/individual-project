#!/usr/bin/env python
"""
Check 5 (§7.5) driver — the multiple-testing flag, per strategy. Deterministic,
zero LLM, INFORMATIONAL (never gates routing; enters no differential — D7/A7).

Additive sidecar by design: this driver writes its own results file and touches
neither the audit lattice nor the registered report shape (the RQ3 run-of-record
predates check 5's build; folding the flag into the report dict would
change a frozen artefact shape for zero inferential gain).

Inputs per strategy are EXPLICIT and recorded in the output:
  * the strategy/factor NAME matched against the zoo (default: the --strategy
    id; override with --name when the paper's published factor label differs);
  * --t-stat: the paper-side t-statistic the band condition examines (the
    audited paper's claimed headline t; check 5 concerns the PAPER's discovery
    process, not our realised runs). Absent -> NOT_EVALUATED.
  * --free-parameters: the construction's free-parameter count (a paper-side
    fact; absent -> NOT_EVALUATED).

Constants come fail-loud from docs/thresholds.yaml auditor.mt_flag
(load_mt_flag_params); the zoo labels from the D5-resolved names.csv.

Usage:
  run_mt_flag.py --strategy drf [--name "drf"] [--t-stat 3.2] [--free-parameters 2]
                 [--out results/auditor/mt_flag_<strategy>.json]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from agents.auditor.checks.mt_flag import compute_mt_flag, load_zoo_names  # noqa: E402
from agents.auditor.thresholds import load_mt_flag_params  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Auditor check 5 (mt_flag), informational.")
    ap.add_argument("--strategy", required=True, help="strategy id (drf/mom6/str/...)")
    ap.add_argument("--name", default=None,
                    help="factor name for the zoo match (default: the strategy id)")
    ap.add_argument("--t-stat", type=float, default=None, dest="t_stat",
                    help="paper-side t-statistic (absent -> band NOT_EVALUATED)")
    ap.add_argument("--free-parameters", type=int, default=None, dest="free_parameters",
                    help="paper-side free-parameter count (absent -> NOT_EVALUATED)")
    ap.add_argument("--out", default=None,
                    help="output JSON (default results/auditor/mt_flag_<strategy>.json)")
    args = ap.parse_args(argv)

    params = load_mt_flag_params()
    zoo = load_zoo_names(_REPO_ROOT / params.zoo_names_path,
                         strip_trailing_asterisk=params.strip_trailing_asterisk)
    result = compute_mt_flag(
        args.name or args.strategy,
        t_stat=args.t_stat,
        free_parameters=args.free_parameters,
        zoo_names=zoo,
        t_band=params.t_band,
        max_free_parameters=params.max_free_parameters,
        aliases=params.aliases,
    )

    payload = {
        "strategy": args.strategy,
        "inputs": {"name": args.name or args.strategy, "t_stat": args.t_stat,
                   "free_parameters": args.free_parameters,
                   "zoo_names": params.zoo_names_path,
                   "t_band": list(params.t_band),
                   "max_free_parameters": params.max_free_parameters},
        **result.to_dict(),
    }
    out = Path(args.out) if args.out else (
        _REPO_ROOT / "results" / "auditor" / f"mt_flag_{args.strategy}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    print(f"[mt_flag] -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
