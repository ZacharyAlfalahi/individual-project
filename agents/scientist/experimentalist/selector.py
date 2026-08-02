"""G5 — selection (spec §4/§9). LEXICOGRAPHIC, never a composite score: a weighted score would let
high returns compensate for methodological invalidity — the exact trade this project argues
against. Inputs are the CPCV-qualified survivors (already valid/audit-clean/BH-survivor/
CPCV-qualified), so the surviving tie-break is: higher MEDIAN CPCV-path Sharpe, then the SIMPLER
intervention (fewer config changes), then proposal_id (determinism). Advance at most `cap`.
"""

from __future__ import annotations

import math


def _finite(x) -> float:
    return x if isinstance(x, (int, float)) and math.isfinite(x) else float("-inf")


def select_g5(survivors: list[dict], *, cap: int) -> list[str]:
    """survivors: dicts with proposal_id, median_cpcv_sharpe, n_changes. Returns the advanced
    proposal_ids (<= cap), lexicographically ordered — NOT a composite score."""
    ordered = sorted(
        survivors,
        key=lambda s: (-_finite(s["median_cpcv_sharpe"]), s["n_changes"], s["proposal_id"]),
    )
    return [s["proposal_id"] for s in ordered[:cap]]
