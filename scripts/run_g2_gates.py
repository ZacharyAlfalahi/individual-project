#!/usr/bin/env python
"""
RQ2 anchor fidelity gates 1-2 (contract §7) to a committed results file
(close-out item 5, 2026-09-05). Thin driver over the existing, tested
evaluation/harness/round_trip.gate12_verdict: for each anchor it adapts the
hand-authored gold once and reports gate 2 (rulebook byte-equality modulo the
authorised standing register) and the gate-1 holding-period invariant that
to_rulebook omits. Deterministic, zero LLM, zero holdout contact.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from evaluation.harness.round_trip import (  # noqa: E402
    gate12_verdict,
    load_verified_standing_subs,
)

ANCHORS = ("str", "drf", "mom6", "crf")


def main(argv=None) -> int:
    subs = load_verified_standing_subs()
    verdicts = {a: gate12_verdict(a, subs) for a in ANCHORS}
    out = _REPO_ROOT / "results" / "g2_gates.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"anchors": verdicts}, indent=2, default=str),
                   encoding="utf-8")
    for a, v in verdicts.items():
        print(f"{a}: gate1(holding)={v['holding_period_match']} "
              f"gate2(byte-equal)={v['rulebook_byte_equal']} pass={v['pass']}"
              + (f"  error={v['error']}" if v.get('error') else ""))
    print(f"[g2_gates] -> {out}")
    return 0 if all(v["pass"] for v in verdicts.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
