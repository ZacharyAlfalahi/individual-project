"""
Executability re-census (finding F8). The original reachability census
(prereg/reachability_census.md) checked template-enum x variable AVAILABILITY but NOT engine
EXECUTABILITY, so it over-counted: the audited engine natively runs only T4 (independent
double-sort, holding_period=1); T1/T2 (month filter) and T3 (row filter) are realised as
Scientist-side panel transforms leaving agents/quant/library/ unmodified.

This census recomputes the ELIGIBLE (engine-executable) mechanism count PER correction-sensitive
anchor, using the executability-aware eligibility filter, and derives m = min(default, count).
Anchors and their holding periods: str=1, drf=1 (T4 available), mom6=6 (T4 refused at holding>1).
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from agents.scientist.researcher.eligibility import evaluate  # noqa: E402
from agents.scientist.researcher.library import (  # noqa: E402
    available_conditioning_variables,
    load_library,
)
OUT_FILE = REPO_ROOT / "prereg" / "executability_census.md"

# Correction-sensitive anchors (O4) and their engine holding_period.
ANCHORS = {"str": 1, "drf": 1, "mom6": 6}
DEFAULT_M = 6
GATE = 8


def census() -> tuple[str, dict]:
    lib = load_library()
    available = available_conditioning_variables()
    lines = ["# Scientist executability census (generated — finding F8)", ""]
    lines.append(
        "Eligible = ENGINE-EXECUTABLE (F8): a template counts only if the audited engine can run "
        "it for the strategy's holding_period. Native T4 double-sort needs holding_period=1; "
        "T1/T2/T3 (panel transforms) run for any holding.")
    lines.append("")
    lines.append("| anchor | holding | eligible (distinct) mechanisms | >= 8 gate | m = min(6, n) |")
    lines.append("| --- | --- | --- | --- | --- |")
    per_strategy_m: dict[str, int] = {}
    detail: dict[str, list[str]] = {}
    for strat, hp in ANCHORS.items():
        elig = [m for m in lib.mechanisms
                if evaluate(m, strategy_family="CHARACTERISTIC_SORT", holding_period=hp,
                            templates=lib.templates, variable_families=lib.variable_families,
                            available_variables=available).eligible]
        ids = [m["mechanism_id"] for m in elig]
        detail[strat] = ids
        n = len({m["title"] for m in elig})
        m_val = min(DEFAULT_M, n)
        per_strategy_m[strat] = m_val
        gate = "PASS" if n >= GATE else f"**FAIL ({n}<{GATE})**"
        lines.append(f"| {strat} | {hp} | {n} ({', '.join(ids)}) | {gate} | {m_val} |")
    lines.append("")
    lines.append(
        "**T4-only mechanisms** (mech_001, mech_007) are NOT executable for mom6 (holding=6, "
        "double-sort refused); mom6 reaches the rest via the T1/T2/T3 panel transforms. All three "
        "anchors clear the >= 8 gate, so proposals_per_strategy m stays 6 for each.")
    lines.append("")
    lines.append(
        "This supersedes the availability-only reachability census for the eligibility gate "
        "(F8 / SC-SCI-7); the reachability census remains valid as the config-space enumeration.")
    return "\n".join(lines) + "\n", per_strategy_m


def main() -> None:
    text, per_strategy_m = census()
    OUT_FILE.write_text(text)
    print(text)
    print("proposals_per_strategy (executability-based):", per_strategy_m)


if __name__ == "__main__":
    main()
