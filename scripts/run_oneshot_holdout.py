"""CLI entrypoint for the one-shot holdout builder/evaluator.

    ./.venv/bin/python scripts/run_oneshot_holdout.py --rehearsal      # dev pseudo-window, gate never opens
    ./.venv/bin/python scripts/run_oneshot_holdout.py                  # REAL one-shot — refused here

The one-shot holdout orchestration machinery, the gate checklist, the marker state machine, the seeded
build orchestration, and the evaluator all live in ``agents/scientist/experimentalist/oneshot_holdout/``
and are exercised end-to-end by ``tests/unit/test_oneshot_holdout_rehearsal.py`` against a synthetic
panel builder. Two integrations remain deliberately unwired in this component:

  * the DEVELOPMENT-pseudo-window panel builder over the real development parquets (needed for a
    real-data ``--rehearsal``); and
  * the HOLDOUT panel builder over ``data/holdout/`` — which is the gated one-shot itself.

The real run executes only on an explicit written go-ahead and is additionally blocked
on G6 survivors, the P3 moderate-prior artefact, and E9 (spec §7). This CLI refuses to fire it.
"""

from __future__ import annotations

import argparse
import sys


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="one-shot holdout builder/evaluator")
    parser.add_argument("--rehearsal", action="store_true",
                        help="run the full pipeline on a development pseudo-window (gate never opens)")
    args = parser.parse_args(argv)

    if args.rehearsal:
        print(
            "one-shot holdout --rehearsal: the orchestration + evaluator machinery is built and tested "
            "(tests/unit/test_oneshot_holdout_rehearsal.py runs the full rehearsal on a synthetic panel).\n"
            "Wiring the development-pseudo-window builder over the real development parquets is the "
            "remaining integration before a real-data rehearsal-green can be written.",
            file=sys.stderr,
        )
        return 2

    print(
        "one-shot holdout real run: REFUSED. This is the single code path that reads data/holdout/ and it "
        "executes only on an explicit written go-ahead, with the holdout panel builder wired and "
        "the preconditions met (G6 survivors, P3 moderate-prior artefact, E9). None of that is this "
        "component's job — build + rehearsal-green is where one-shot holdout stops (spec §7).",
        file=sys.stderr,
    )
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
