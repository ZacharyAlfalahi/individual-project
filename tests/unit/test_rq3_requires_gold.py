"""
RQ3 anchor-membership invariant (D-A59).

A strategy enters the RQ3 prevalence denominator n_i and the confirmatory BH-FDR
family IFF it is a full anchor: a hand-authored StrategySpec gold
(evaluation/gold_specs/) PLUS a registered as-published baseline (a
config/hypothesis_registry.yaml row). A gold-free strategy has no paper-stated
baseline, so its bias-OFF state is compiler-inferred; pooling machine-inferred and
paper-stated baselines under one prevalence denominator n_i mixes two estimands.

Therefore every registered factor must have a StrategySpec gold — with ONE
grandfathered exception, the `traded_liquidity` negative control, which is a
synthetic specificity device (no replicated paper, hence no paper-stated baseline,
sign 0), not a corpus prevalence member.

This test makes it impossible to slip a gold-free scale-layer strategy into RQ3:
add it to the registry and it fails here until a StrategySpec gold exists (which
makes it an anchor). See docs/auditor/auditor_design.md (D-A59),
docs/auditor/confirmatory_prereg_proposal.md §4-ter, and
docs/evaluation/t2_selection_prereg.md §2.
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from agents.auditor.validation.hypothesis_registry import (  # noqa: E402
    load_hypothesis_registry,
)
from evaluation.gold_specs.gold_loader import _ANCHORS  # noqa: E402

# The sole deliberate gold-free registry row: the negative control is a synthetic
# specificity device, not a replicated paper, so it has no paper-stated baseline.
GRANDFATHERED_GOLDFREE = {"traded_liquidity"}

# StrategySpec gold files live here; _ANCHORS[fid]["file"] names each one.
GOLD_DIR = REPO_ROOT / "evaluation" / "gold_specs"


def _has_strategyspec_gold(factor_id: str) -> bool:
    """True iff a hand-authored StrategySpec gold FILE exists on disk for the factor
    (not merely a key in the _ANCHORS dict) — matching the invariant's claim that the
    gold exists in evaluation/gold_specs/. A registry row plus a stub _ANCHORS entry
    with no .md file must NOT pass."""
    meta = _ANCHORS.get(factor_id)
    if not isinstance(meta, dict) or "file" not in meta:
        return False
    return (GOLD_DIR / meta["file"]).exists()


def test_every_rq3_factor_has_a_strategyspec_gold():
    registry = load_hypothesis_registry()
    for fid, hyp in registry.items():
        if fid in GRANDFATHERED_GOLDFREE:
            # Grandfathered only as the sign-0 negative control, never a magnitude claim.
            assert hyp.expected_sign == 0, (
                f"{fid} is grandfathered gold-free only as the sign-0 negative control"
            )
            continue
        assert _has_strategyspec_gold(fid), (
            f"factor {fid!r} is registered for RQ3 (confirmatory/census) but has no "
            f"StrategySpec gold in evaluation/gold_specs/ — RQ3 membership requires a "
            f"hand-authored anchor gold (D-A59). Promote it by authoring the gold, or "
            f"it stays a scale-layer (RQ2/external) strategy."
        )


def test_grandfather_set_is_minimal():
    # Exactly the negative control(s) may be gold-free; keep the exception from growing
    # silently — any new gold-free row would enter the RQ3 denominator with a
    # compiler-inferred baseline (D-A59).
    registry = load_hypothesis_registry()
    goldfree = {fid for fid in registry if not _has_strategyspec_gold(fid)}
    assert goldfree == GRANDFATHERED_GOLDFREE, (
        f"gold-free registry rows changed: {sorted(goldfree)} "
        f"(expected {sorted(GRANDFATHERED_GOLDFREE)})"
    )
