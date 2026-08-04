#!/usr/bin/env python
"""run_reporter.py — deterministic Reporter driver (render / publish / check / scaffold).

Usage:
  ./.venv/bin/python scripts/run_reporter.py scaffold <id> --spec <path> --paper BBW \
      --strategy drf --quant-dir results/quant/run_<sha> --audit-dir results/auditor/run_<sha>
  ./.venv/bin/python scripts/run_reporter.py check <id>      # verify in memory, write nothing
  ./.venv/bin/python scripts/run_reporter.py render <id>     # render into a scratch staging dir
  ./.venv/bin/python scripts/run_reporter.py publish         # publish every declared run

Mirrors the run_auditor.py driver idiom (repo-root sys.path insert, main() -> int exit code).
Subcommands are used because the Reporter genuinely has four modes.
"""

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from agents.reporter import cli  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_render = sub.add_parser("render", help="render + verify into a staging dir")
    p_render.add_argument("run_id")

    p_check = sub.add_parser("check", help="render + verify in memory (CI mode)")
    p_check.add_argument("run_id")

    sub.add_parser("publish", help="publish every declared run (atomic swap)")

    p_scaffold = sub.add_parser("scaffold", help="print a filled pointer file")
    p_scaffold.add_argument("run_id")
    p_scaffold.add_argument("--spec", required=True)
    p_scaffold.add_argument("--paper", required=True)
    p_scaffold.add_argument("--strategy", required=True)
    p_scaffold.add_argument("--phase", default="D", choices=("D", "F"))
    p_scaffold.add_argument("--quant-dir", default=None)
    p_scaffold.add_argument("--audit-dir", default=None)

    args = ap.parse_args()

    if args.cmd == "scaffold":
        print(
            cli.scaffold(
                reporter_run_id=args.run_id,
                paper_id=args.paper,
                strategy_id=args.strategy,
                phase=args.phase,
                spec=args.spec,
                quant_dir=args.quant_dir,
                audit_dir=args.audit_dir,
            ),
            end="",
        )
        return 0
    if args.cmd == "check":
        doc = cli.check_run(args.run_id)
        print(
            json.dumps(
                {
                    "run": args.run_id,
                    "disposition": doc.disposition.value,
                    "claims": len(doc.claims),
                    "verified": True,
                }
            )
        )
        return 0
    if args.cmd == "render":
        staging = cli.render_run(args.run_id)
        print(json.dumps({"run": args.run_id, "staging": str(staging)}))
        return 0
    if args.cmd == "publish":
        print(json.dumps(cli.publish_all(), indent=2))
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
