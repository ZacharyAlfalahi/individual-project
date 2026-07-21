"""
Shared fixtures for the Auditor tests: a runnable single-leg strategy that sorts
on the synthetic panel's `score` column, plus the five all-runnable ToggleFacts.

Not a test module (leading underscore) — imported by the test_auditor_* files.
"""

from __future__ import annotations

from agents.auditor.schemas.toggle import TOGGLE_IDS, ToggleFacts
from agents.librarian.adapter.result import AdaptResult, CombinerInstruction, LegCall
from agents.quant.config import Binding, Evidence, Inherited, build_quant_config


def make_score_strategy(
    label: str = "synthetic",
    *,
    groups: int = 5,
    weighting: str = "equal",
    signal_lag: int = 0,
    trim: Inherited | None = None,
) -> AdaptResult:
    """A single-leg AdaptResult ranking on `score` (long top group, short bottom).
    The construction toggles (signal_lag, trim) are overridden per-cell by the
    cell runner; the values here are the strategy's base/as-published state."""
    score = Binding("score", "BOUND", Evidence(column="score"))
    cfg = build_quant_config(
        label,
        score,
        groups=Inherited(groups, "DESIGN", Evidence(note="test groups")),
        weighting=Inherited(weighting, "DESIGN", Evidence(note="test weighting")),
        signal_lag=Inherited(signal_lag, "DESIGN", Evidence(note="test base lag")),
        long_group=Inherited(groups - 1, "DESIGN", Evidence(note="top group")),
        short_group=Inherited(0, "DESIGN", Evidence(note="bottom group")),
        holding_period=Inherited(1, "DESIGN", Evidence(note="monthly rebalance")),
        trim=trim,
    )
    leg = LegCall(strategy_id=f"{label}::0", kwargs={}, result=cfg)
    return AdaptResult(
        strategy_label=label,
        leg_calls=(leg,),
        combiner=CombinerInstruction(kind="single_leg"),
    )


def all_runnable_facts() -> list[ToggleFacts]:
    """Five runnable toggle facts (a COMPLETE audit)."""
    return [ToggleFacts(t, runnable=True) for t in TOGGLE_IDS]
