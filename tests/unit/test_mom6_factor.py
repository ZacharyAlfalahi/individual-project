"""
Unit tests for the mom6 factor runner (scripts/build_mom6.py).

The signal maths is covered by test_mom6_signal.py and the overlap/engine by
their own suites; here we guard the new glue: the rulebook wiring (deciles, EW,
long the TOP decile, short the BOTTOM, skip = signal_lag) and the resulting leg
DIRECTION (winners − losers is positive when high-momentum bonds outperform).
A flipped long/short would silently invert the momentum sign.
"""

import sys
from pathlib import Path

import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))
from agents.quant.library.overlap import run_with_holding_period  # noqa: E402
import build_mom6 as bmf  # noqa: E402


def test_rulebook_wiring_from_config():
    cfg = {"n_groups": 10, "weighting": "equal", "skip_months": 1, "holding_months": 6}
    rb = bmf.mom6_rulebook(cfg)
    assert rb["score"] == "mom6"
    assert rb["groups"] == 10
    assert rb["weighting"] == "equal"
    assert rb["long_group"] == 9   # P10 winners
    assert rb["short_group"] == 0  # P1 losers
    assert rb["long_group"] > rb["short_group"]  # winners - losers, not inverted
    assert rb["signal_lag"] == 1   # the Jostova skip


def _me(s: str) -> pd.Timestamp:
    return pd.Timestamp(s) + pd.offsets.MonthEnd(0)


def test_leg_direction_is_winners_minus_losers():
    """High-mom6 bonds realise high next-month returns; the P(top)-P(bottom)
    spread must be POSITIVE. Uses groups=2, skip=0, H=1 for a hand-checkable
    case (H=1 reduces overlap to the base engine)."""
    spec = [("A", 0.5, 0.04), ("B", 0.4, 0.03), ("C", -0.1, -0.01), ("D", -0.2, -0.02)]
    rows = []
    for cusip, mom, r_next in spec:
        # Formation month: score present, own ret irrelevant.
        rows.append({"cusip": cusip, "date": _me("2010-01"), "ret": 0.0, "size": 1.0, "mom6": mom})
        # Realisation month: the held return; score no longer needed.
        rows.append({"cusip": cusip, "date": _me("2010-02"), "ret": r_next, "size": 1.0, "mom6": float("nan")})
    panel = pd.DataFrame(rows)

    cfg = {"n_groups": 2, "weighting": "equal", "skip_months": 0, "holding_months": 1}
    mr = run_with_holding_period(panel, bmf.mom6_rulebook(cfg), holding_period=1)

    row = mr.set_index("date").loc[_me("2010-02")]
    # long (top: A,B) mean 0.035; short (bottom: C,D) mean -0.015; spread = 0.05.
    assert row["strategy_ret"] == pytest.approx(0.05, abs=1e-12)
    assert row["strategy_ret"] > 0  # winners outperform losers → positive
