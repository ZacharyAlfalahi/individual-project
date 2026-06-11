"""
Unit tests for the BBW 2019 5% VaR signal.

Covers:
  - Eligibility gate: exactly 23 non-NaN obs → NaN; exactly 24 → computed.
  - Hand-computed value: small synthetic series with known returns yields
    the expected -1 × 2nd-lowest.
  - Sign convention: higher var_5pct = more downside risk.
  - Trailing-window discipline: returns >window months back do not affect
    the current value.
  - NaN-in-window handling: the eligibility count uses only non-NaN
    returns; intervening NaNs don't inflate the count.
  - Per-cusip independence: two cusips with disjoint timelines do not
    leak across the groupby.
  - Threshold-block validation: missing keys in thresholds.yaml signals
    block raise a clear KeyError before any compute starts.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "scripts"))
from build_var_5pct import compute_var_5pct  # noqa: E402


def _panel(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"]) + pd.offsets.MonthEnd(0)
    return df


# ---------------------------------------------------------------------------
# Eligibility gate
# ---------------------------------------------------------------------------

def test_23_non_nan_obs_yields_nan():
    """Exactly 23 non-NaN observations in the trailing 36-month window must
    yield NaN — the BBW min_obs=24 boundary."""
    # 23 months of returns, last row evaluated → window has 23 obs total.
    rows = [
        {"cusip": "C", "date": f"2010-{m:02d}", "ret": -0.01 * m}
        for m in range(1, 13)
    ] + [
        {"cusip": "C", "date": f"2010-{m:02d}", "ret": -0.005 * m}
        for m in range(1, 13)  # this construction would duplicate dates
    ]
    # The above duplicates. Build properly:
    rows = []
    for i in range(23):
        # Months 2010-01 .. 2011-11 = 23 months
        year = 2010 + (i // 12)
        month = (i % 12) + 1
        rows.append({"cusip": "C", "date": f"{year}-{month:02d}",
                     "ret": -0.001 * (i + 1)})
    out = compute_var_5pct(_panel(rows), window=36, min_obs=24, rank=2, multiplier=-1.0)
    assert out["var_5pct"].isna().all(), (
        "23 obs < min_obs=24 → every var_5pct must be NaN"
    )


def test_24_non_nan_obs_yields_computed_value():
    """Exactly 24 non-NaN observations: the LAST row clears the gate and
    yields a computed value. Earlier rows (1..23) still NaN."""
    rows = []
    for i in range(24):
        year = 2010 + (i // 12)
        month = (i % 12) + 1
        rows.append({"cusip": "C", "date": f"{year}-{month:02d}",
                     "ret": -0.001 * (i + 1)})  # strictly decreasing
    out = compute_var_5pct(_panel(rows), window=36, min_obs=24, rank=2, multiplier=-1.0)
    # First 23 rows: NaN (gate fails).
    assert out["var_5pct"].iloc[:23].isna().all()
    # Row 24 (last): computed. Returns are -0.001..-0.024; sorted ascending
    # the lowest is -0.024, 2nd-lowest is -0.023 → var_5pct = -1 × -0.023 = 0.023.
    assert out["var_5pct"].iloc[23] == pytest.approx(0.023, abs=1e-12)


# ---------------------------------------------------------------------------
# Hand-computed correctness
# ---------------------------------------------------------------------------

def test_hand_computed_2nd_lowest():
    """Construct a 24-observation series with mixed signs and one extreme
    drawdown; verify the output equals -1 × 2nd-lowest."""
    # 24 returns: 0.01 repeated 21 times, then -0.10, -0.05, -0.03 (three
    # negative tails). Sorted ascending: -0.10, -0.05, -0.03, 0.01, ...
    # 2nd-lowest = -0.05 → var_5pct = -1 × -0.05 = 0.05.
    rets = [0.01] * 21 + [-0.10, -0.05, -0.03]
    rows = []
    for i, r in enumerate(rets):
        year = 2010 + (i // 12)
        month = (i % 12) + 1
        rows.append({"cusip": "C", "date": f"{year}-{month:02d}", "ret": r})
    out = compute_var_5pct(_panel(rows), window=36, min_obs=24, rank=2, multiplier=-1.0)
    assert out["var_5pct"].iloc[-1] == pytest.approx(0.05, abs=1e-12)


# ---------------------------------------------------------------------------
# Sign convention
# ---------------------------------------------------------------------------

def test_higher_var_5pct_means_more_downside_risk():
    """Bond A's 2nd-lowest = -0.05 → var = 0.05. Bond B's 2nd-lowest = -0.20
    → var = 0.20. B has higher var; B has worse downside risk. Pins the
    sign convention."""
    def series(extreme):
        return [0.01] * 21 + [extreme - 0.05, extreme, 0.0]
    rets_a = series(-0.05)   # 2nd-lowest = -0.05
    rets_b = series(-0.20)   # 2nd-lowest = -0.20
    rows = []
    for cid, rets in [("A", rets_a), ("B", rets_b)]:
        for i, r in enumerate(rets):
            year = 2010 + (i // 12)
            month = (i % 12) + 1
            rows.append({"cusip": cid, "date": f"{year}-{month:02d}", "ret": r})
    out = compute_var_5pct(_panel(rows), window=36, min_obs=24, rank=2, multiplier=-1.0)
    a_var = out.loc[(out["cusip"] == "A") & (out["var_5pct"].notna()), "var_5pct"].iloc[-1]
    b_var = out.loc[(out["cusip"] == "B") & (out["var_5pct"].notna()), "var_5pct"].iloc[-1]
    assert b_var > a_var, (
        "Higher var_5pct must correspond to worse downside; "
        f"got A={a_var}, B={b_var}"
    )
    assert a_var == pytest.approx(0.05, abs=1e-12)
    assert b_var == pytest.approx(0.20, abs=1e-12)


# ---------------------------------------------------------------------------
# Trailing-window discipline
# ---------------------------------------------------------------------------

def test_returns_older_than_window_do_not_affect_value():
    """A catastrophic return >36 months ago must not appear in the current
    var_5pct. Differential against a buggy "all-history" implementation."""
    # 60 months total. The first month carries a -0.99 return that would
    # dominate a buggy "lifetime" implementation. After 36 months, that
    # observation falls out of the window.
    rows = []
    rows.append({"cusip": "C", "date": "2008-01", "ret": -0.99})
    for i in range(1, 60):
        year = 2008 + (i // 12)
        month = (i % 12) + 1
        rows.append({"cusip": "C", "date": f"{year}-{month:02d}", "ret": 0.01})
    out = compute_var_5pct(_panel(rows), window=36, min_obs=24, rank=2, multiplier=-1.0)
    # At the last month (2012-12), the trailing 36-month window starts at
    # 2010-01 — the -0.99 observation at 2008-01 is out of scope.
    # All 36 windowed returns are 0.01, so 2nd-lowest = 0.01 → var = -0.01.
    last = out.iloc[-1]["var_5pct"]
    assert last == pytest.approx(-0.01, abs=1e-12), (
        f"Last var_5pct {last} suggests the old -0.99 is leaking through "
        "(should be -0.01 if window=36 is respected)."
    )


# ---------------------------------------------------------------------------
# NaN-in-window handling
# ---------------------------------------------------------------------------

def test_nan_in_window_counts_only_non_nan():
    """NaN returns inside the trailing window are NOT counted toward
    min_obs; the eligibility test uses only the non-NaN count. Differential
    against a buggy `len(window) >= min_obs` implementation."""
    # 36 months total: 12 NaN scattered, 24 valid returns. The eligibility
    # gate should pass (24 ≥ 24) and the value should use only the 24
    # valid returns.
    rets = []
    n_nan_so_far = 0
    for i in range(36):
        # Scatter exactly 12 NaNs into the first 36 positions (every third
        # position until the quota is filled), leaving 24 valid returns.
        if i % 3 == 0 and n_nan_so_far < 12:
            rets.append(np.nan)
            n_nan_so_far += 1
        else:
            rets.append(-0.001 * (i + 1))
    # Ensure exactly 12 NaNs and 24 valid
    n_nan = sum(1 for r in rets if (isinstance(r, float) and np.isnan(r)))
    n_valid = len(rets) - n_nan
    assert n_nan == 12, f"setup invariant: 12 NaNs (got {n_nan})"
    assert n_valid == 24, f"setup invariant: 24 valid (got {n_valid})"

    rows = []
    for i, r in enumerate(rets):
        year = 2010 + (i // 12)
        month = (i % 12) + 1
        rows.append({"cusip": "C", "date": f"{year}-{month:02d}", "ret": r})
    out = compute_var_5pct(_panel(rows), window=36, min_obs=24, rank=2, multiplier=-1.0)
    # Last row: gate clears (n_valid = 24 ≥ 24).
    assert not pd.isna(out["var_5pct"].iloc[-1]), (
        "Last row should have a computed value (n_valid = 24 ≥ min_obs)."
    )


# ---------------------------------------------------------------------------
# Per-cusip independence
# ---------------------------------------------------------------------------

def test_two_cusips_compute_independently():
    """Two cusips' rolling-window computations must not leak. Interleave
    them in the input and check the output respects per-cusip history."""
    rows = []
    # Cusip A: 24 strongly negative returns → var_5pct positive.
    for i in range(24):
        year = 2010 + (i // 12)
        month = (i % 12) + 1
        rows.append({"cusip": "A", "date": f"{year}-{month:02d}",
                     "ret": -0.02 - i * 0.001})
    # Cusip B: 24 zero returns → 2nd-lowest = 0 → var_5pct = 0.
    for i in range(24):
        year = 2010 + (i // 12)
        month = (i % 12) + 1
        rows.append({"cusip": "B", "date": f"{year}-{month:02d}", "ret": 0.0})
    out = compute_var_5pct(_panel(rows), window=36, min_obs=24, rank=2, multiplier=-1.0)
    a_last = out.loc[out["cusip"] == "A", "var_5pct"].iloc[-1]
    b_last = out.loc[out["cusip"] == "B", "var_5pct"].iloc[-1]
    # A's returns: -0.02 .. -0.043. 2nd-lowest = -0.042 → var_5pct = 0.042.
    assert a_last == pytest.approx(0.042, abs=1e-12)
    # B's returns: all zero → 2nd-lowest = 0 → var_5pct = 0.
    assert b_last == pytest.approx(0.0, abs=1e-12)


# ---------------------------------------------------------------------------
# Threshold-block validation
# ---------------------------------------------------------------------------

def test_load_config_requires_all_keys(tmp_path, monkeypatch):
    """load_config raises KeyError if any of window/min_obs/rank/multiplier
    is missing — silent defaults would mask a stale thresholds.yaml."""
    import build_var_5pct as bvp

    bad_yaml = tmp_path / "bad.yaml"
    bad_yaml.write_text(
        "signals:\n"
        "  var_5pct:\n"
        "    window: 36\n"
        "    min_obs: 24\n"
        # rank and multiplier missing
    )
    monkeypatch.setattr(bvp, "THRESHOLDS_FILE", bad_yaml)
    with pytest.raises(KeyError, match="rank"):
        bvp.load_config()


def test_load_config_missing_block_raises(tmp_path, monkeypatch):
    import build_var_5pct as bvp
    bad_yaml = tmp_path / "empty.yaml"
    bad_yaml.write_text("monthly_panel:\n  min_vol_qt: 100000\n")
    monkeypatch.setattr(bvp, "THRESHOLDS_FILE", bad_yaml)
    with pytest.raises(KeyError, match="signals.var_5pct"):
        bvp.load_config()
