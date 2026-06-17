"""
Unit tests for the gamma / ILLIQ signal (scripts/build_gamma_illiq.py).

Covers:
  - gamma = -Cov(Δp_d, Δp_{d+1}) matches a hand-computed value, and an
    alternating (bid-ask-bounce) price path yields POSITIVE gamma (illiquid).
  - The >= min_pairs screen returns NaN when too few pairs.
  - The <= max_gap_bdays rule breaks the change chain (same data, different gap).
  - Changes never straddle a month boundary.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))
import build_gamma_illiq as bg  # noqa: E402


def _daily(cusip: str, days: list[str], logprices: list[float]) -> pd.DataFrame:
    return pd.DataFrame({
        "cusip_id": cusip,
        "trd_exctn_dt": pd.to_datetime(days),
        "price_vwap": np.exp(np.array(logprices, dtype=float)),
    })


def test_alternating_path_gives_positive_gamma_hand_value():
    # 8 consecutive business days (Jan 2010), log prices alternate 0/0.1 →
    # changes +.1,-.1,... (7 changes), pairs all (+.1,-.1) or (-.1,+.1):
    # 6 pairs, each x*y = -0.01, means 0 → Cov = -0.06/5 = -0.012 → gamma = +0.012.
    days = ["2010-01-04", "2010-01-05", "2010-01-06", "2010-01-07",
            "2010-01-08", "2010-01-11", "2010-01-12", "2010-01-13"]
    logp = [0.0, 0.1, 0.0, 0.1, 0.0, 0.1, 0.0, 0.1]
    out = bg.compute_gamma(_daily("X", days, logp), min_pairs=5, max_gap_bdays=7,
                           sign_multiplier=-1.0, cov_ddof=1)
    g = out.set_index("date").loc[pd.Timestamp("2010-01-31"), "gamma"]
    assert g == pytest.approx(0.012, abs=1e-9)
    assert g > 0  # bid-ask bounce → negative autocov → positive gamma = illiquid


def test_min_pairs_gate_returns_nan():
    # 5 days → 4 changes → 3 pairs < min_pairs(5) → NaN.
    days = ["2010-01-04", "2010-01-05", "2010-01-06", "2010-01-07", "2010-01-08"]
    logp = [0.0, 0.1, 0.0, 0.1, 0.0]
    out = bg.compute_gamma(_daily("X", days, logp), min_pairs=5, max_gap_bdays=7)
    g = out.set_index("date").loc[pd.Timestamp("2010-01-31"), "gamma"]
    assert np.isnan(g)


def test_gap_rule_breaks_the_chain():
    # Two runs (Jan 4-7 and Jan 25-27) separated by a >7-business-day gap.
    days = ["2010-01-04", "2010-01-05", "2010-01-06", "2010-01-07",
            "2010-01-25", "2010-01-26", "2010-01-27"]
    logp = [0.0, 0.1, 0.0, 0.1, 0.0, 0.1, 0.0]
    base = _daily("X", days, logp)
    # With the 7-bd gap rule, the Jan7→Jan25 change is invalid → only 3 valid
    # pairs → NaN under min_pairs=5.
    g_gapped = bg.compute_gamma(base, min_pairs=5, max_gap_bdays=7).set_index("date")
    assert np.isnan(g_gapped.loc[pd.Timestamp("2010-01-31"), "gamma"])
    # Relax the gap to span the chasm: the Jan7→Jan25 change becomes valid → 5
    # pairs → a finite gamma. Same data, only the gap rule changed.
    g_open = bg.compute_gamma(base, min_pairs=5, max_gap_bdays=100).set_index("date")
    assert np.isfinite(g_open.loc[pd.Timestamp("2010-01-31"), "gamma"])


def test_changes_do_not_straddle_months():
    # A December trade must not create a change in January's gamma. January alone
    # has 8 consecutive days (same as the hand-value test) → gamma 0.012.
    days = ["2009-12-31",  # prior-month trade — must be ignored for Jan
            "2010-01-04", "2010-01-05", "2010-01-06", "2010-01-07",
            "2010-01-08", "2010-01-11", "2010-01-12", "2010-01-13"]
    logp = [5.0,  # wild prior-month price; if it leaked into Jan it would distort
            0.0, 0.1, 0.0, 0.1, 0.0, 0.1, 0.0, 0.1]
    out = bg.compute_gamma(_daily("X", days, logp), min_pairs=5, max_gap_bdays=7).set_index("date")
    assert out.loc[pd.Timestamp("2010-01-31"), "gamma"] == pytest.approx(0.012, abs=1e-9)
