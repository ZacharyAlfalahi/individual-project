"""
Unit tests for the overlapping-holding wrapper.

Covers:
  - H=1 reduces exactly to the engine's monthly_returns (regression
    invariant the spec demands).
  - H=2 hand-computed differential: averaged-cohort series cannot collapse
    to the engine's single-month output.
  - Ramp-up: n_cohorts_alive ascends through H, then descends at the tail.
  - Bond exits mid-hold: cohort weights renormalise to the survivors.
  - v1 control rejection: passing `control` raises NotImplementedError so
    a future enable is intentional.
"""

import numpy as np
import pandas as pd
import pytest

from agents.quant.library.characteristic_sort import run_characteristic_sort
from agents.quant.library.overlap import run_with_holding_period


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _me(s: str) -> pd.Timestamp:
    return pd.Timestamp(s) + pd.offsets.MonthEnd(0)


def _panel(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"]) + pd.offsets.MonthEnd(0)
    return df


# ---------------------------------------------------------------------------
# Regression invariant — H=1 reduces to engine
# ---------------------------------------------------------------------------

def test_h1_reduces_to_engine_strictly():
    """H=1 produces a monthly_returns frame identical (to 1e-12) to the
    engine's. The wrapper appends an n_cohorts_alive=1 column; the rest
    must match element-wise."""
    rng = np.random.default_rng(123)
    months = pd.date_range("2010-01-31", periods=24, freq="ME")
    rows = []
    for cid in [f"C{i:02d}" for i in range(1, 13)]:
        base = rng.uniform(0, 1)
        for d in months:
            rows.append({
                "cusip": cid,
                "date": d,
                "ret": rng.normal(scale=0.02),
                "size": rng.uniform(50, 200),
                "score": base + rng.normal(scale=0.1),
            })
    panel = pd.DataFrame(rows)
    rulebook = {"score": "score", "groups": 5, "weighting": "by_size", "min_bonds": 5}

    engine_mr = run_characteristic_sort(panel, rulebook)["monthly_returns"]
    wrapper_mr = run_with_holding_period(panel, rulebook, holding_period=1)

    # The wrapper carries one extra column. Drop it for the comparison.
    assert "n_cohorts_alive" in wrapper_mr.columns
    assert (wrapper_mr["n_cohorts_alive"] == 1).all()
    pd.testing.assert_frame_equal(
        wrapper_mr.drop(columns=["n_cohorts_alive"]),
        engine_mr,
        check_exact=False,
        atol=1e-12,
        rtol=0,
    )


def test_h1_default_argument_matches_explicit_h1():
    """The default holding_period=1 must not diverge from explicit H=1."""
    rng = np.random.default_rng(7)
    months = pd.date_range("2015-01-31", periods=12, freq="ME")
    rows = []
    for cid in [f"C{i:02d}" for i in range(1, 9)]:
        base = rng.uniform(0, 1)
        for d in months:
            rows.append({
                "cusip": cid,
                "date": d,
                "ret": rng.normal(scale=0.02),
                "size": 100.0,
                "score": base + rng.normal(scale=0.1),
            })
    panel = pd.DataFrame(rows)
    rulebook = {"score": "score", "groups": 4, "weighting": "equal", "min_bonds": 4}

    default_mr = run_with_holding_period(panel, rulebook)
    explicit_mr = run_with_holding_period(panel, rulebook, holding_period=1)
    pd.testing.assert_frame_equal(default_mr, explicit_mr)


# ---------------------------------------------------------------------------
# Hand-computed differential — H=2
# ---------------------------------------------------------------------------

def test_h2_hand_computed_with_cohort_overlap():
    """H=2 with deterministic cohort returns. Two formation months (Jan,
    Feb) produce two cohorts; the holding-period mechanic creates an Apr
    realisation row that a buggy `form-and-hold-one-month` implementation
    cannot produce, and the Mar realisation row averages two cohorts.

    Setup:
      - 4 bonds A,B (long), C,D (short) -- scores 4,3,2,1, groups=2
      - Equal weighting, min_bonds=4
      - Returns:
          Feb: A,B = +5%   C,D = -5%   (cohort1 h=1)
          Mar: A,B = +1%   C,D = -1%   (cohort1 h=2 + cohort2 h=1)
          Apr: A,B = +3%   C,D = -3%   (cohort2 h=2)

    Expected H=2 series:
      Feb: 1 cohort, spread = 0.05 - (-0.05) = 0.10
      Mar: 2 cohorts (both 0.02), mean = 0.02
      Apr: 1 cohort, spread = 0.03 - (-0.03) = 0.06

    A buggy form-and-hold-one-month (engine at H=1) produces only
    Feb=0.10, Mar=0.02 -- no Apr row at all.
    """
    score_map = {"A": 4.0, "B": 3.0, "C": 2.0, "D": 1.0}
    # One row per (cusip, date). The engine's validator rejects duplicates,
    # so a single row carries BOTH the row's own ret (used by holding-period
    # lookups at this month) AND the score that makes the month a formation
    # candidate (or NaN to suppress formation at that month).
    panel_rows = [
        # Jan 2010 — formation t1 only. Ret unused by any cohort.
        ("A", "2010-01", 0.0, 4.0), ("B", "2010-01", 0.0, 3.0),
        ("C", "2010-01", 0.0, 2.0), ("D", "2010-01", 0.0, 1.0),
        # Feb 2010 — formation t2 AND realisation for cohort1 (h=1).
        ("A", "2010-02",  0.05, 4.0), ("B", "2010-02",  0.05, 3.0),
        ("C", "2010-02", -0.05, 2.0), ("D", "2010-02", -0.05, 1.0),
        # Mar 2010 — realisation only. Score=NaN suppresses Mar formation.
        ("A", "2010-03",  0.01, np.nan), ("B", "2010-03",  0.01, np.nan),
        ("C", "2010-03", -0.01, np.nan), ("D", "2010-03", -0.01, np.nan),
        # Apr 2010 — realisation only.
        ("A", "2010-04",  0.03, np.nan), ("B", "2010-04",  0.03, np.nan),
        ("C", "2010-04", -0.03, np.nan), ("D", "2010-04", -0.03, np.nan),
    ]
    rows = [
        {"cusip": c, "date": d, "ret": r, "size": 100.0, "score": s}
        for c, d, r, s in panel_rows
    ]

    rulebook = {
        "score": "score", "groups": 2, "weighting": "equal", "min_bonds": 4,
    }
    out = run_with_holding_period(_panel(rows), rulebook, holding_period=2)

    assert list(out["date"]) == [_me("2010-02"), _me("2010-03"), _me("2010-04")]
    assert out.loc[out["date"] == _me("2010-02"), "strategy_ret"].iloc[0] == pytest.approx(0.10, abs=1e-12)
    assert out.loc[out["date"] == _me("2010-03"), "strategy_ret"].iloc[0] == pytest.approx(0.02, abs=1e-12)
    assert out.loc[out["date"] == _me("2010-04"), "strategy_ret"].iloc[0] == pytest.approx(0.06, abs=1e-12)
    assert list(out["n_cohorts_alive"]) == [1, 2, 1]

    # Differential: H=1 has no Apr row at all.
    h1 = run_with_holding_period(_panel(rows), rulebook, holding_period=1)
    assert _me("2010-04") not in set(h1["date"])


# ---------------------------------------------------------------------------
# Ramp-up — n_cohorts_alive ascends to H, descends at tail
# ---------------------------------------------------------------------------

def test_ramp_up_partial_cohorts_h3():
    """With H=3 and 3 formation months (Jan, Feb, Mar), n_cohorts_alive
    should climb 1 → 2 → 3 (reaches full H at the third realisation), then
    descend at the tail."""
    score_map = {"A": 4.0, "B": 3.0, "C": 2.0, "D": 1.0}
    # Three formations (Jan, Feb, Mar); realisations span Feb-May.
    # Apr and May score=NaN suppresses further formation past Mar.
    panel_rows = []
    for d in ["2010-01", "2010-02", "2010-03"]:
        for bid, sc in score_map.items():
            panel_rows.append((bid, d, 0.01, sc))
    for d in ["2010-04", "2010-05"]:
        for bid in score_map:
            panel_rows.append((bid, d, 0.01, float("nan")))
    rows = [
        {"cusip": c, "date": d, "ret": r, "size": 100.0, "score": s}
        for c, d, r, s in panel_rows
    ]

    rulebook = {
        "score": "score", "groups": 2, "weighting": "equal", "min_bonds": 4,
    }
    out = run_with_holding_period(_panel(rows), rulebook, holding_period=3)
    # Expected sequence: Feb=1 (cohort1 h=1), Mar=2 (cohort1 h=2 + cohort2 h=1),
    # Apr=3 (h=3, h=2, h=1), May=2 (cohort2 h=3 + cohort3 h=2),
    # Jun would be cohort3 h=3 but Mar formation needs Jun realisation -- which
    # depends on whether Jun has rets. We didn't add Jun, so cohort3's h=3 is
    # missing => Jun row absent.
    assert list(out["date"]) == [
        _me("2010-02"), _me("2010-03"), _me("2010-04"), _me("2010-05"),
    ]
    assert list(out["n_cohorts_alive"]) == [1, 2, 3, 2]


# ---------------------------------------------------------------------------
# Bond exits mid-hold — weights renormalise
# ---------------------------------------------------------------------------

def test_bond_exits_mid_hold_renormalises_by_size():
    """One cohort, 3 long bonds (A,B,C) and 3 short bonds (X,Y,Z), by_size
    weighting, H=2. At h=2, bonds C and Z exit. Surviving weights must
    renormalise WITHIN the leg from formation sizes.

    Formation sizes: A=100, B=200, C=300 (long); X=100, Y=200, Z=300 (short).
    Total per leg at formation = 600.

    At h=2 returns: A=+2%, B=+4%, X=-2%, Y=-4%.
    Correct renormalised long return:
      surviving weights: A=100/300, B=200/300
      = 1/3 * 0.02 + 2/3 * 0.04 = 0.00667 + 0.02667 = 0.03333
    Correct short return (symmetric): -0.03333
    Correct spread = 0.06667

    A buggy implementation that uses formation weights without
    renormalisation gives spread = 0.0333 (off by a factor of 2). The
    differential is sharp.
    """
    long_sizes = {"A": 100.0, "B": 200.0, "C": 300.0}
    short_sizes = {"X": 100.0, "Y": 200.0, "Z": 300.0}
    # Scores: long bonds scored higher than short bonds. Top-half = long.
    score_map = {"A": 6.0, "B": 5.0, "C": 4.0, "X": 3.0, "Y": 2.0, "Z": 1.0}
    rows = []
    # Formation (Jan 2010): all 6 bonds with formation sizes
    for cusip, sz in {**long_sizes, **short_sizes}.items():
        rows.append({
            "cusip": cusip, "date": "2010-01",
            "ret": 0.0, "size": sz, "score": score_map[cusip],
        })
    # Realisation h=1 (Feb 2010): all 6 bonds present, deterministic returns.
    # Score=NaN suppresses Feb formation.
    for cusip in {**long_sizes, **short_sizes}:
        rows.append({
            "cusip": cusip, "date": "2010-02",
            "ret": 0.01, "size": 100.0, "score": float("nan"),
        })
    # Realisation h=2 (Mar 2010): C and Z exit (no rows). Others have known
    # returns. Score=NaN suppresses Mar formation.
    h2_rets = {"A": 0.02, "B": 0.04, "X": -0.02, "Y": -0.04}
    for cusip, r in h2_rets.items():
        rows.append({
            "cusip": cusip, "date": "2010-03",
            "ret": r, "size": 100.0, "score": float("nan"),
        })

    rulebook = {
        "score": "score", "groups": 2, "weighting": "by_size", "min_bonds": 6,
    }
    out = run_with_holding_period(_panel(rows), rulebook, holding_period=2)
    mar = out.loc[out["date"] == _me("2010-03")].iloc[0]

    # Correct renormalised result:
    expected_long = (100 * 0.02 + 200 * 0.04) / (100 + 200)
    expected_short = (100 * -0.02 + 200 * -0.04) / (100 + 200)
    expected_spread = expected_long - expected_short
    assert mar["strategy_ret"] == pytest.approx(expected_spread, abs=1e-12)
    assert mar["long_ret"] == pytest.approx(expected_long, abs=1e-12)
    assert mar["short_ret"] == pytest.approx(expected_short, abs=1e-12)

    # Buggy formation-weight-without-renormalise spread would be 0.0333 --
    # half the correct answer. Assert we are NOT that.
    buggy = 2 * (100 * 0.02 + 200 * 0.04) / (100 + 200 + 300)
    assert mar["strategy_ret"] != pytest.approx(buggy, abs=1e-6)


# ---------------------------------------------------------------------------
# v1 limitation — control rejected
# ---------------------------------------------------------------------------

def test_rejects_control_in_v1():
    """Passing `control` in the rulebook must raise NotImplementedError so
    a future enable is intentional rather than silently changing behaviour."""
    rows = [
        {"cusip": "A", "date": "2010-01", "ret": 0.0, "size": 100.0,
         "score": 1.0, "ctrl": 0.5},
        {"cusip": "B", "date": "2010-01", "ret": 0.0, "size": 100.0,
         "score": 2.0, "ctrl": 0.5},
        {"cusip": "A", "date": "2010-02", "ret": 0.01, "size": 100.0,
         "score": 1.0, "ctrl": 0.5},
        {"cusip": "B", "date": "2010-02", "ret": 0.02, "size": 100.0,
         "score": 2.0, "ctrl": 0.5},
    ]
    rulebook = {
        "score": "score", "control": "ctrl",
        "groups": 2, "weighting": "equal", "min_bonds": 2,
    }
    with pytest.raises(NotImplementedError, match="control"):
        run_with_holding_period(_panel(rows), rulebook, holding_period=2)


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------

def test_rejects_non_positive_holding_period():
    rows = [
        {"cusip": "A", "date": "2010-01", "ret": 0.0, "size": 100.0, "score": 1.0},
    ]
    with pytest.raises(ValueError, match="positive integer"):
        run_with_holding_period(_panel(rows), {"score": "score"}, holding_period=0)
    with pytest.raises(ValueError, match="positive integer"):
        run_with_holding_period(_panel(rows), {"score": "score"}, holding_period=-1)
