"""
Unit tests for the characteristic-sort engine.

Covers all six correctness tests from
docs/quant/specs/characteristic_sort_engine_spec.md section 7 plus targeted
tests for internal helpers and edge cases. Synthetic data only --
the engine never touches /data/holdout/ (guarded in tests/conftest.py
regardless).
"""

import math

import numpy as np
import pandas as pd
import pytest

from agents.quant.library.characteristic_sort import (
    _apply_defaults,
    _assign_groups,
    _build_lagged_panel,
    _nw_auto_lags,
    _nw_hac_variance,
    _validate_panel,
    regress_on_benchmark,
    run_characteristic_sort,
    summarize_returns,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _me(s: str) -> pd.Timestamp:
    """Construct a month-end Timestamp from a YYYY-MM string."""
    return pd.Timestamp(s) + pd.offsets.MonthEnd(0)


def _panel(rows: list[dict]) -> pd.DataFrame:
    """Build a panel with month-end-normalized dates."""
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"]) + pd.offsets.MonthEnd(0)
    return df


# ===========================================================================
# Spec section 7.1 -- "Lining up the right return"
# ===========================================================================

def test_uses_next_month_return_not_current() -> None:
    """
    3 bonds x 3 months. Bond A is top, B is bottom in month 1. Month-1
    returns are deliberately large (+5%, -5%) so the WRONG (same-month)
    engine would report +10%. The RIGHT engine uses month-2's returns
    (+1%, -2%) and reports +3%.
    """
    rows = [
        # Month 1 (formation t)
        {"cusip": "A", "date": "2010-01", "ret": 0.05, "size": 100.0, "score": 3.0},
        {"cusip": "B", "date": "2010-01", "ret": -0.05, "size": 100.0, "score": 1.0},
        {"cusip": "C", "date": "2010-01", "ret": 0.0, "size": 100.0, "score": 2.0},
        # Month 2 (realisation of t, formation of t+1)
        {"cusip": "A", "date": "2010-02", "ret": 0.01, "size": 100.0, "score": 3.0},
        {"cusip": "B", "date": "2010-02", "ret": -0.02, "size": 100.0, "score": 1.0},
        {"cusip": "C", "date": "2010-02", "ret": 0.0, "size": 100.0, "score": 2.0},
        # Month 3 (realisation of t+1)
        {"cusip": "A", "date": "2010-03", "ret": 0.0, "size": 100.0, "score": 3.0},
        {"cusip": "B", "date": "2010-03", "ret": 0.0, "size": 100.0, "score": 1.0},
        {"cusip": "C", "date": "2010-03", "ret": 0.0, "size": 100.0, "score": 2.0},
    ]
    result = run_characteristic_sort(
        _panel(rows),
        {"score": "score", "groups": 3, "weighting": "equal", "min_bonds": 3},
    )

    mr = result["monthly_returns"]
    # Two formation months (Jan, Feb) -> two realisation rows (Feb, Mar).
    assert len(mr) == 2
    # The first realisation is Feb 2010, with return earned over Jan->Feb,
    # which IS month-2's return: A=+1%, B=-2% -> spread = +3%.
    feb_row = mr[mr["date"] == _me("2010-02")].iloc[0]
    assert feb_row["strategy_ret"] == pytest.approx(0.03, abs=1e-12)
    # The wrong (same-month) answer would be +10%.
    assert feb_row["strategy_ret"] != pytest.approx(0.10, abs=1e-6)
    # The second realisation (Mar) uses March returns, all 0 -> spread = 0.
    mar_row = mr[mr["date"] == _me("2010-03")].iloc[0]
    assert mar_row["strategy_ret"] == pytest.approx(0.0, abs=1e-12)


# ===========================================================================
# Spec section 7.2 -- "Plain spread, equal weights"
# ===========================================================================

def test_plain_spread_equal_weights() -> None:
    """
    4 bonds, groups=2, equal weighting, 7 months. Top group always earns
    +2%, bottom always -1% in their next-month rows -> strategy returns
    +3% every formation month. 6 formation months produce 6 realisation rows.
    Sharpe is undefined (zero variance) and is NOT asserted.
    """
    rows: list[dict] = []
    months = [f"2010-{m:02d}" for m in range(1, 8)]
    for date in months:
        for bid, score, ret in [
            ("A", 4.0, 0.02),
            ("B", 3.0, 0.02),
            ("C", 2.0, -0.01),
            ("D", 1.0, -0.01),
        ]:
            rows.append(
                {
                    "cusip": bid,
                    "date": date,
                    "ret": ret,
                    "size": 100.0,
                    "score": score,
                }
            )

    result = run_characteristic_sort(
        _panel(rows),
        {"score": "score", "groups": 2, "weighting": "equal", "min_bonds": 2},
    )
    mr = result["monthly_returns"]
    assert len(mr) == 6
    # Realisation months are Feb-Jul 2010 (formations Jan-Jun -> next month).
    expected_dates = [_me(f"2010-{m:02d}") for m in range(2, 8)]
    assert list(mr["date"]) == expected_dates
    assert np.allclose(mr["strategy_ret"].values, 0.03, atol=1e-12)
    assert result["summary"]["average"] == pytest.approx(0.03, abs=1e-12)


# ===========================================================================
# Spec section 7.3 -- "Weight by size, using the right month's size"
# ===========================================================================

def test_size_weight_uses_formation_month_size() -> None:
    """
    4 bonds (2 long, 2 short), groups=2, weighting='by_size', 2 months.

    Formation month (Jan 2010):
      L1 size=100, L2 size=300, S1 size=50, S2 size=50.
    Realisation month (Feb 2010):
      L1 size=500  next_ret=+4%
      L2 size=100  next_ret=+1%
      S1 size=?    next_ret= 0%
      S2 size=?    next_ret= 0%

    Hand-computed long-leg using Jan sizes:
      (100*0.04 + 300*0.01) / 400 = 0.0175
    Hand-computed short-leg: 0.
    Spread = 0.0175.

    Equal-weight would yield (0.04 + 0.01) / 2 - 0 = 0.025.
    Next-month-size (forbidden) would yield (500*0.04 + 100*0.01) / 600 = 0.035.
    Both wrong answers differ clearly from 0.0175.
    """
    rows = [
        # Formation: Jan 2010
        {"cusip": "L1", "date": "2010-01", "ret": 0.0, "size": 100.0, "score": 4.0},
        {"cusip": "L2", "date": "2010-01", "ret": 0.0, "size": 300.0, "score": 3.0},
        {"cusip": "S1", "date": "2010-01", "ret": 0.0, "size": 50.0,  "score": 2.0},
        {"cusip": "S2", "date": "2010-01", "ret": 0.0, "size": 50.0,  "score": 1.0},
        # Realisation: Feb 2010 -- sizes deliberately flipped, returns chosen
        {"cusip": "L1", "date": "2010-02", "ret": 0.04, "size": 500.0, "score": 4.0},
        {"cusip": "L2", "date": "2010-02", "ret": 0.01, "size": 100.0, "score": 3.0},
        {"cusip": "S1", "date": "2010-02", "ret": 0.0,  "size": 200.0, "score": 2.0},
        {"cusip": "S2", "date": "2010-02", "ret": 0.0,  "size": 200.0, "score": 1.0},
    ]
    result = run_characteristic_sort(
        _panel(rows),
        {"score": "score", "groups": 2, "weighting": "by_size", "min_bonds": 4},
    )
    mr = result["monthly_returns"]
    assert len(mr) == 1
    spread = mr["strategy_ret"].iloc[0]
    assert spread == pytest.approx(0.0175, abs=1e-12)
    # Sanity: the two wrong answers really are different from 0.0175.
    assert spread != pytest.approx(0.025, abs=1e-6)
    assert spread != pytest.approx(0.035, abs=1e-6)


# ===========================================================================
# Spec section 7.4 -- "Grouping and thin months"
# ===========================================================================

def test_grouping_assignment_ten_bonds_five_groups() -> None:
    """
    10 bonds with scores 1..10, groups=5 -> exact assignment:
      {1,2}->0, {3,4}->1, {5,6}->2, {7,8}->3, {9,10}->4.
    """
    df = pd.DataFrame(
        {
            "cusip": [f"B{i}" for i in range(1, 11)],
            "score": [float(i) for i in range(1, 11)],
        }
    )
    groups = _assign_groups(df, "score", "cusip", 5)
    assignment = dict(zip(df["cusip"], groups))
    expected = {
        "B1": 0, "B2": 0,
        "B3": 1, "B4": 1,
        "B5": 2, "B6": 2,
        "B7": 3, "B8": 3,
        "B9": 4, "B10": 4,
    }
    assert assignment == expected


def test_thin_month_is_skipped_and_recorded() -> None:
    """
    Month A has 10 eligible bonds, Month B has only 3. With groups=5,
    default min_bonds=5: Month B must be absent from monthly_returns
    and present in bookkeeping['months_skipped'].
    """
    rows: list[dict] = []
    # Fat month A: Jan 2010, 10 bonds with score 1..10 (formation t)
    for i in range(1, 11):
        rows.append(
            {
                "cusip": f"B{i}",
                "date": "2010-01",
                "ret": 0.0,
                "size": 100.0,
                "score": float(i),
            }
        )
    # Realisation rows for Jan formation: Feb 2010, all 10 bonds with
    # deterministic next_ret so the engine has SOMETHING to compute.
    for i in range(1, 11):
        rows.append(
            {
                "cusip": f"B{i}",
                "date": "2010-02",
                "ret": 0.01 if i >= 9 else (-0.01 if i <= 2 else 0.0),
                "size": 100.0,
                "score": float(i),
            }
        )
    # Thin formation month: Feb 2010 already has 10 bonds; we instead
    # need a different month with only 3 bonds. Use Mar 2010 (3 bonds)
    # + Apr 2010 realisation rows. min_bonds=5 will skip Mar formation.
    for i in [1, 2, 3]:
        rows.append(
            {
                "cusip": f"B{i}",
                "date": "2010-03",
                "ret": 0.01,
                "size": 100.0,
                "score": float(i),
            }
        )
    for i in [1, 2, 3]:
        rows.append(
            {
                "cusip": f"B{i}",
                "date": "2010-04",
                "ret": 0.0,
                "size": 100.0,
                "score": float(i),
            }
        )

    result = run_characteristic_sort(
        _panel(rows),
        {"score": "score", "groups": 5, "weighting": "equal"},
    )
    mr = result["monthly_returns"]
    bk = result["bookkeeping"]
    # Jan formation -> Feb realisation row exists.
    assert _me("2010-02") in set(mr["date"])
    # Feb formation: 10 bonds in Feb, all eligible, with NO next-month
    # returns because Mar has only 3 bonds -- so Feb has only 3 bonds
    # with a next_ret -> falls below min_bonds and is skipped.
    assert _me("2010-02") in set(bk["months_skipped"])
    # Mar formation: only 3 bonds anyway -> skipped.
    assert _me("2010-03") in set(bk["months_skipped"])
    # Apr formation: 3 bonds, but no May exists -> all next_ret NaN ->
    # falls below min_bonds -> skipped.
    assert _me("2010-04") in set(bk["months_skipped"])


# ===========================================================================
# Spec section 7.5 -- "The double-sort combine"
# ===========================================================================

def _double_sort_panel() -> pd.DataFrame:
    """
    8 bonds, scores 1..8, control values arranged so the score x control
    cells are deliberately UNEVEN:
      stripe 0 (control_bot): long={B5,B6,B7} (3), short={B1} (1)
      stripe 1 (control_top): long={B8}       (1), short={B2,B3,B4} (3)

    Next-month returns chosen so stripe spreads are 6% and 0%, giving a
    double-sort answer of 3.0% and a single-sort answer of 3.5% (clearly
    different).
    """
    score_map = {"B1": 1, "B2": 2, "B3": 3, "B4": 4,
                 "B5": 5, "B6": 6, "B7": 7, "B8": 8}
    control_map = {"B5": 1, "B6": 2, "B7": 3, "B1": 4,
                   "B4": 5, "B3": 6, "B2": 7, "B8": 8}
    next_ret_map = {
        "B5": 0.05, "B6": 0.05, "B7": 0.05,  # stripe-0 longs
        "B1": -0.01,                          # stripe-0 short
        "B8": 0.01,                           # stripe-1 long
        "B2": 0.01, "B3": 0.01, "B4": 0.01,   # stripe-1 shorts
    }
    rows: list[dict] = []
    # Formation: Jan 2010
    for bid in score_map:
        rows.append(
            {
                "cusip": bid,
                "date": "2010-01",
                "ret": 0.0,
                "size": 100.0,
                "score": float(score_map[bid]),
                "control_value": float(control_map[bid]),
            }
        )
    # Realisation: Feb 2010
    for bid in score_map:
        rows.append(
            {
                "cusip": bid,
                "date": "2010-02",
                "ret": next_ret_map[bid],
                "size": 100.0,
                "score": float(score_map[bid]),
                "control_value": float(control_map[bid]),
            }
        )
    return _panel(rows)


def test_double_sort_averages_stripe_spreads() -> None:
    """
    Double-sort engine returns the plain average of the two stripe spreads:
      stripe 0 spread = mean(0.05,0.05,0.05) - (-0.01) = 0.06
      stripe 1 spread = 0.01 - mean(0.01,0.01,0.01)    = 0.00
      average         = 0.03
    A naive global top-minus-bottom (single sort) would return:
      mean(0.05,0.05,0.05,0.01) - mean(-0.01,0.01,0.01,0.01) = 0.04 - 0.005 = 0.035.
    """
    panel = _double_sort_panel()
    result = run_characteristic_sort(
        panel,
        {
            "score": "score",
            "control": "control_value",
            "groups": 2,
            "control_groups": 2,
            "weighting": "equal",
            "min_bonds": 8,
        },
    )
    mr = result["monthly_returns"]
    assert len(mr) == 1
    assert mr["strategy_ret"].iloc[0] == pytest.approx(0.03, abs=1e-12)
    # Naive answer would be 0.035 -- engine must NOT return that.
    assert mr["strategy_ret"].iloc[0] != pytest.approx(0.035, abs=1e-6)


def test_single_sort_on_same_panel_equals_global_top_minus_bottom() -> None:
    """Run the same data WITHOUT control -> exactly 0.035."""
    panel = _double_sort_panel()
    result = run_characteristic_sort(
        panel,
        {
            "score": "score",
            "groups": 2,
            "weighting": "equal",
            "min_bonds": 8,
        },
    )
    mr = result["monthly_returns"]
    assert len(mr) == 1
    assert mr["strategy_ret"].iloc[0] == pytest.approx(0.035, abs=1e-12)


def _asymmetric_double_sort_panel() -> pd.DataFrame:
    """2 score-groups x 3 control-stripes with UNEVEN cells (Workstream K). Global
    score groups (median split at score 4/5) intersect 3 control stripes:
      stripe 0 (control 1,2,3): hi={s5:0.06,s6:0.04}, lo={s1:0.01} -> 0.05-0.01 = 0.04
      stripe 1 (control 4,5,6): hi={s7:0.03},         lo={s2:0.01,s3:0.03} -> 0.03-0.02 = 0.01
      stripe 2 (control 7,8):   hi={s8:0.10},         lo={s4:0.04} -> 0.10-0.04 = 0.06
      double (mean of stripe spreads) = (0.04+0.01+0.06)/3 = 0.11/3 ~= 0.03667
      single (global hi-mean - lo-mean) = 0.0575 - 0.0225 = 0.035 (deliberately different)
    """
    spec = [  # (score, control, next_ret)
        (5, 1, 0.06), (6, 2, 0.04), (1, 3, 0.01),   # stripe 0
        (7, 4, 0.03), (2, 5, 0.01), (3, 6, 0.03),   # stripe 1
        (8, 7, 0.10), (4, 8, 0.04),                 # stripe 2
    ]
    rows: list[dict] = []
    for i, (score, control, nret) in enumerate(spec):
        for date, ret in (("2010-01", 0.0), ("2010-02", nret)):
            rows.append({"cusip": f"B{i}", "date": date, "ret": ret, "size": 100.0,
                         "score": float(score), "control_value": float(control)})
    return _panel(rows)


def test_asymmetric_double_sort_control_groups_ne_groups() -> None:
    """Workstream K: control_groups != groups (2x3) had no KAT — the engine's
    asymmetric stripe handling was unverified. It must form 3 control stripes (not
    2), average the per-stripe score spreads, and land on 0.11/3 — NOT the
    single-sort 0.035 that a control-ignoring bug returns."""
    panel = _asymmetric_double_sort_panel()
    result = run_characteristic_sort(
        panel,
        {"score": "score", "control": "control_value",
         "groups": 2, "control_groups": 3, "weighting": "equal", "min_bonds": 8},
    )
    mr = result["monthly_returns"]
    assert len(mr) == 1
    assert mr["strategy_ret"].iloc[0] == pytest.approx((0.04 + 0.01 + 0.06) / 3, abs=1e-12)
    assert mr["strategy_ret"].iloc[0] != pytest.approx(0.035, abs=1e-6)


# ===========================================================================
# Spec section 7.6 -- "The statistics"
# ===========================================================================

def test_summarize_returns_known_series() -> None:
    """
    Series [0.04, 0.02] * 6 (T=12).
      mean        = 0.03
      sd (ddof=1) = sqrt(0.0012 / 11) ~= 0.01044465936
      Sharpe      = (0.03 / sd) * sqrt(12) ~= 9.94987...
      t-stat (NW lag=0, HC0) = mean / sqrt(sum(u^2) / T^2)
                             = 0.03 / sqrt(0.0012/144) ~= 10.3923...
    """
    s = pd.Series([0.04, 0.02] * 6)
    out = summarize_returns(s, nw_lags=0, months_per_year=12)
    assert out["average"] == pytest.approx(0.03, abs=1e-12)
    assert out["annualised_average"] == pytest.approx(0.36, abs=1e-12)
    expected_sd = math.sqrt(0.0012 / 11.0)
    assert out["bumpiness"] == pytest.approx(expected_sd, abs=1e-12)
    expected_sharpe = (0.03 / expected_sd) * math.sqrt(12.0)
    assert out["sharpe"] == pytest.approx(expected_sharpe, abs=1e-12)
    expected_t = 0.03 / math.sqrt(0.0012 / 144.0)
    assert out["t_stat"] == pytest.approx(expected_t, abs=1e-12)
    assert out["n_months"] == 12
    assert out["nw_lags_used"] == 0


# ===========================================================================
# Supporting tests: _assign_groups
# ===========================================================================

def test_assign_groups_row_order_invariance() -> None:
    """Shuffling input rows produces identical group assignments."""
    df = pd.DataFrame(
        {
            "cusip": [f"B{i}" for i in range(1, 11)],
            "score": [float(i) for i in range(1, 11)],
        }
    )
    g1 = _assign_groups(df, "score", "cusip", 5)
    df2 = df.sample(frac=1.0, random_state=42).reset_index(drop=True)
    g2 = _assign_groups(df2, "score", "cusip", 5)
    # Align by cusip and compare
    map1 = dict(zip(df["cusip"], g1.values))
    map2 = dict(zip(df2["cusip"], g2.values))
    assert map1 == map2


def test_assign_groups_with_ties_uses_cusip_as_tiebreaker() -> None:
    """
    All scores equal -> rank by cusip ascending. With 4 bonds in 2 groups,
    {B1,B2} -> 0, {B3,B4} -> 1.
    """
    df = pd.DataFrame(
        {
            "cusip": ["B1", "B2", "B3", "B4"],
            "score": [5.0, 5.0, 5.0, 5.0],
        }
    )
    g = _assign_groups(df, "score", "cusip", 2)
    assert dict(zip(df["cusip"], g.values)) == {"B1": 0, "B2": 0, "B3": 1, "B4": 1}


def test_assign_groups_empty_returns_empty() -> None:
    df = pd.DataFrame({"cusip": [], "score": []})
    g = _assign_groups(df, "score", "cusip", 5)
    assert len(g) == 0


# ===========================================================================
# Supporting tests: _build_lagged_panel + calendar-strict policy
# ===========================================================================

def test_calendar_strict_gap_produces_nan_next_ret() -> None:
    """A bond with rows in Jan and Mar (Feb missing) -> Jan's next_ret is NaN."""
    rows = [
        {"cusip": "A", "date": "2010-01", "ret": 0.01, "size": 100.0, "score": 1.0},
        {"cusip": "A", "date": "2010-03", "ret": 0.02, "size": 100.0, "score": 1.0},
    ]
    panel = _panel(rows)
    work = _build_lagged_panel(panel, "score", 0)
    jan = work[work["date"] == _me("2010-01")].iloc[0]
    assert pd.isna(jan["next_ret"])


def test_signal_lag_at_panel_start_drops_month() -> None:
    """signal_lag=1 with 3 bonds in Jan + 3 in Feb -> Jan ranking_score NaN
    (no Dec). The Jan formation month falls below min_bonds and is skipped."""
    rows = []
    for bid, score in [("A", 3.0), ("B", 2.0), ("C", 1.0)]:
        rows.append({"cusip": bid, "date": "2010-01", "ret": 0.0, "size": 100.0, "score": score})
        rows.append({"cusip": bid, "date": "2010-02", "ret": 0.01, "size": 100.0, "score": score})
        rows.append({"cusip": bid, "date": "2010-03", "ret": 0.0,  "size": 100.0, "score": score})
    panel = _panel(rows)
    result = run_characteristic_sort(
        panel,
        {"score": "score", "groups": 3, "weighting": "equal",
         "min_bonds": 3, "signal_lag": 1},
    )
    bk = result["bookkeeping"]
    # Jan formation: signal_lag=1 -> ranking_score from Dec 2009 -> NaN ->
    # 0 eligible -> month skipped.
    assert _me("2010-01") in set(bk["months_skipped"])


def test_signal_lag_ranks_on_prior_month_score() -> None:
    """Workstream K — signal_lag=1 POSITIVE semantic: formation ranks on the PRIOR
    month's score (the existing test only checks the first-month DROP edge). Scores
    swap A<->B between Jan and Feb, so the lagged (Jan) and contemporaneous (Feb)
    rankings give opposite-signed spreads. Under lag=1 only the Feb formation is
    valid (Jan has no Dec score; Mar has no next return), and it must reflect Jan's
    ranking -> long A (Jan score 2), short B (Jan score 1) -> A's Mar 0.03 minus
    B's 0.01 = +0.02. A contemporaneous or wrong-direction lag would give -0.02."""
    rows = []
    for date, a_score, b_score, a_ret, b_ret in [
        ("2010-01", 2.0, 1.0, 0.0, 0.0),
        ("2010-02", 1.0, 2.0, 0.0, 0.0),
        ("2010-03", 1.0, 1.0, 0.03, 0.01),
    ]:
        rows.append({"cusip": "A", "date": date, "ret": a_ret, "size": 100.0, "score": a_score})
        rows.append({"cusip": "B", "date": date, "ret": b_ret, "size": 100.0, "score": b_score})
    panel = _panel(rows)
    cfg = {"score": "score", "groups": 2, "weighting": "equal", "min_bonds": 2}
    lagged = run_characteristic_sort(panel, {**cfg, "signal_lag": 1})["monthly_returns"]
    assert lagged["strategy_ret"].tolist() == pytest.approx([0.02], abs=1e-12)
    contemp = run_characteristic_sort(panel, {**cfg, "signal_lag": 0})["monthly_returns"]
    vals = contemp["strategy_ret"].tolist()
    assert any(v == pytest.approx(-0.02, abs=1e-12) for v in vals)   # contemporaneous flips sign
    assert not any(v == pytest.approx(0.02, abs=1e-12) for v in vals)


def test_nan_score_drops_bond_from_eligibility() -> None:
    """A bond with a NaN score is silently dropped from its month."""
    rows = [
        {"cusip": "A", "date": "2010-01", "ret": 0.0, "size": 100.0, "score": float("nan")},
        {"cusip": "B", "date": "2010-01", "ret": 0.0, "size": 100.0, "score": 2.0},
        {"cusip": "C", "date": "2010-01", "ret": 0.0, "size": 100.0, "score": 1.0},
        {"cusip": "A", "date": "2010-02", "ret": 0.05, "size": 100.0, "score": 3.0},
        {"cusip": "B", "date": "2010-02", "ret": 0.02, "size": 100.0, "score": 2.0},
        {"cusip": "C", "date": "2010-02", "ret": -0.01, "size": 100.0, "score": 1.0},
    ]
    result = run_characteristic_sort(
        _panel(rows),
        {"score": "score", "groups": 2, "weighting": "equal", "min_bonds": 2},
    )
    mr = result["monthly_returns"]
    # Only 2 eligible bonds (B,C). Top = B (score 2), bottom = C (score 1).
    # Spread = ret_B(Feb) - ret_C(Feb) = 0.02 - (-0.01) = 0.03.
    assert len(mr) == 1
    assert mr["strategy_ret"].iloc[0] == pytest.approx(0.03, abs=1e-12)


# ===========================================================================
# Supporting tests: input validation
# ===========================================================================

def test_validate_panel_missing_column_raises() -> None:
    df = pd.DataFrame({"cusip": [], "date": pd.to_datetime([]), "ret": [], "size": []})
    with pytest.raises(ValueError, match="missing required columns"):
        _validate_panel(df, _apply_defaults({"score": "score"}))


def test_validate_panel_non_month_end_date_raises() -> None:
    df = pd.DataFrame(
        {
            "cusip": ["A"],
            "date": [pd.Timestamp("2010-01-15")],
            "ret": [0.0],
            "size": [100.0],
            "score": [1.0],
        }
    )
    with pytest.raises(ValueError, match="month-end"):
        _validate_panel(df, _apply_defaults({"score": "score"}))


def test_validate_panel_tz_aware_date_raises() -> None:
    df = pd.DataFrame(
        {
            "cusip": ["A"],
            "date": pd.to_datetime(["2010-01-31"]).tz_localize("UTC"),
            "ret": [0.0],
            "size": [100.0],
            "score": [1.0],
        }
    )
    with pytest.raises(ValueError, match="tz-naive"):
        _validate_panel(df, _apply_defaults({"score": "score"}))


def test_validate_panel_duplicate_bond_date_raises() -> None:
    rows = [
        {"cusip": "A", "date": "2010-01", "ret": 0.0, "size": 100.0, "score": 1.0},
        {"cusip": "A", "date": "2010-01", "ret": 0.0, "size": 100.0, "score": 1.0},
    ]
    with pytest.raises(ValueError, match="duplicate"):
        _validate_panel(_panel(rows), _apply_defaults({"score": "score"}))


def test_apply_defaults_requires_score() -> None:
    with pytest.raises(ValueError, match="score"):
        _apply_defaults({})


def test_apply_defaults_long_short_must_differ() -> None:
    with pytest.raises(ValueError):
        _apply_defaults({"score": "x", "long_group": 0, "short_group": 0})


# ===========================================================================
# Supporting tests: empty panel
# ===========================================================================

def test_empty_panel_returns_empty_result() -> None:
    """An empty panel produces an empty result without raising."""
    df = pd.DataFrame(
        {
            "cusip": pd.Series([], dtype=object),
            "date": pd.Series([], dtype="datetime64[ns]"),
            "ret": pd.Series([], dtype=float),
            "size": pd.Series([], dtype=float),
            "score": pd.Series([], dtype=float),
        }
    )
    result = run_characteristic_sort(df, {"score": "score"})
    assert len(result["monthly_returns"]) == 0
    assert result["summary"]["n_months"] == 0
    assert math.isnan(result["summary"]["average"])
    assert result["bookkeeping"]["months_skipped"] == []


# ===========================================================================
# Supporting tests: HAC / regression
# ===========================================================================

def test_nw_auto_lags_for_T12_is_2() -> None:
    """floor(4 * (12/100)^(2/9)) = floor(2.497...) = 2."""
    assert _nw_auto_lags(12) == 2


def test_nw_auto_lags_clamped_to_nonnegative_and_T_minus_1() -> None:
    assert _nw_auto_lags(0) == 0
    assert _nw_auto_lags(1) == 0
    assert _nw_auto_lags(2) >= 0


def test_regress_on_benchmark_recovers_betas() -> None:
    """Build y = 0.5*F1 + 0.3*F2 + noise; recover alpha~0, betas~(0.5,0.3)."""
    rng = np.random.default_rng(42)
    T = 200
    dates = pd.date_range("2000-01-31", periods=T, freq="ME")
    f1 = rng.normal(size=T)
    f2 = rng.normal(size=T)
    eps = rng.normal(scale=0.01, size=T)
    y_vals = 0.5 * f1 + 0.3 * f2 + eps
    y = pd.Series(y_vals, index=dates)
    factors = pd.DataFrame({"date": dates, "F1": f1, "F2": f2})

    out = regress_on_benchmark(y, factors, nw_lags=0)
    assert out["alpha"] == pytest.approx(0.0, abs=5e-3)
    assert out["betas"]["F1"] == pytest.approx(0.5, abs=5e-3)
    assert out["betas"]["F2"] == pytest.approx(0.3, abs=5e-3)
    assert out["n_obs"] == T


def test_regress_nw0_equals_hc0() -> None:
    """At nw_lags=0, the NW HAC covariance equals White HC0."""
    rng = np.random.default_rng(7)
    T = 100
    dates = pd.date_range("2000-01-31", periods=T, freq="ME")
    f1 = rng.normal(size=T)
    eps = rng.normal(scale=0.1, size=T) * (1 + 0.5 * f1**2)  # heteroskedastic
    y_vals = 0.4 * f1 + eps
    y = pd.Series(y_vals, index=dates)
    factors = pd.DataFrame({"date": dates, "F1": f1})

    # Engine result
    engine_out = regress_on_benchmark(y, factors, nw_lags=0)

    # Hand-rolled HC0 reference
    X = np.column_stack([np.ones(T), f1])
    XtX_inv = np.linalg.inv(X.T @ X)
    beta_hat = XtX_inv @ X.T @ y_vals
    u = y_vals - X @ beta_hat
    # HC0: Var = (X'X)^-1 * X' diag(u^2) X * (X'X)^-1
    middle = X.T @ (X * (u[:, None] ** 2))
    var_hc0 = XtX_inv @ middle @ XtX_inv
    se_hc0 = np.sqrt(np.diag(var_hc0))
    t_hc0 = beta_hat / se_hc0

    assert engine_out["alpha_t"] == pytest.approx(t_hc0[0], rel=1e-9, abs=1e-9)
    assert engine_out["beta_t"]["F1"] == pytest.approx(t_hc0[1], rel=1e-9, abs=1e-9)


def test_nw_hac_variance_lag0_matches_hc0_constant_only() -> None:
    """For a constant-only regression with lag=0, the formula reduces to
    sum(u^2)/T^2."""
    u = np.array([0.01, -0.01, 0.01, -0.01, 0.01, -0.01])
    T = len(u)
    X = np.ones((T, 1))
    var = _nw_hac_variance(u, X, 0)
    expected = (u @ u) / (T * T)
    assert var[0, 0] == pytest.approx(expected, abs=1e-15)


def test_regress_T_le_k_returns_all_nan() -> None:
    """T <= k -> the OLS cannot be fit; everything NaN."""
    dates = pd.date_range("2000-01-31", periods=2, freq="ME")
    y = pd.Series([0.01, 0.02], index=dates)
    factors = pd.DataFrame({"date": dates, "F1": [1.0, -1.0]})
    out = regress_on_benchmark(y, factors, nw_lags=0)
    assert math.isnan(out["alpha"])
    assert math.isnan(out["betas"]["F1"])
    assert math.isnan(out["alpha_t"])
    assert math.isnan(out["beta_t"]["F1"])


def test_regress_small_T_betas_computable_ses_nan() -> None:
    """T > k but T = k+1 -> OLS fits perfectly (zero residuals); HAC SEs are
    degenerate (Var <= 0 path) -> alpha/betas reported, t-stats NaN."""
    dates = pd.date_range("2000-01-31", periods=3, freq="ME")
    y = pd.Series([0.01, 0.02, 0.03], index=dates)
    factors = pd.DataFrame({"date": dates, "F1": [1.0, 2.0, 3.0]})
    out = regress_on_benchmark(y, factors, nw_lags=0)
    assert not math.isnan(out["alpha"])
    assert not math.isnan(out["betas"]["F1"])
    # With perfect fit residuals are zero -> HAC variance is exactly zero ->
    # SE/t-stat NaN by the diag<=0 guard.
    assert math.isnan(out["alpha_t"])
    assert math.isnan(out["beta_t"]["F1"])
    assert out["n_obs"] == 3


# ===========================================================================
# Integration test: defaults end-to-end
# ===========================================================================

def test_safe_rate_missing_rf_column_raises() -> None:
    """When safe_rate is provided, its shape must include date + rf."""
    rows = [
        {"cusip": "A", "date": "2010-01", "ret": 0.0, "size": 100.0, "score": 1.0},
        {"cusip": "A", "date": "2010-02", "ret": 0.01, "size": 100.0, "score": 1.0},
    ]
    bad = pd.DataFrame({"date": pd.to_datetime(["2010-01-31"])})  # no 'rf'
    with pytest.raises(ValueError, match="rf"):
        run_characteristic_sort(_panel(rows), {"score": "score"}, safe_rate=bad)


def test_safe_rate_valid_shape_does_not_change_spread() -> None:
    """A well-formed safe_rate is accepted and -- by design -- does not
    affect the long-short spread (the safe rate cancels)."""
    rows = [
        {"cusip": "A", "date": "2010-01", "ret": 0.0, "size": 100.0, "score": 2.0},
        {"cusip": "B", "date": "2010-01", "ret": 0.0, "size": 100.0, "score": 1.0},
        {"cusip": "A", "date": "2010-02", "ret": 0.05, "size": 100.0, "score": 2.0},
        {"cusip": "B", "date": "2010-02", "ret": -0.01, "size": 100.0, "score": 1.0},
    ]
    panel = _panel(rows)
    rf = pd.DataFrame(
        {
            "date": pd.to_datetime(["2010-01-31", "2010-02-28"]),
            "rf": [0.001, 0.001],
        }
    )
    r_with = run_characteristic_sort(
        panel, {"score": "score", "groups": 2, "weighting": "equal", "min_bonds": 2},
        safe_rate=rf,
    )
    r_without = run_characteristic_sort(
        panel, {"score": "score", "groups": 2, "weighting": "equal", "min_bonds": 2},
    )
    assert (
        r_with["monthly_returns"]["strategy_ret"].iloc[0]
        == pytest.approx(r_without["monthly_returns"]["strategy_ret"].iloc[0], abs=1e-15)
    )


def test_reserved_column_collision_raises() -> None:
    """A panel that pre-defines any of the engine's reserved internal columns
    must be rejected before any silent overwrite can corrupt the math."""
    rows = [
        {"cusip": "A", "date": "2010-01", "ret": 0.0, "size": 100.0, "score": 1.0},
    ]
    df = _panel(rows)
    df["next_ret"] = 0.0  # collides with engine-internal name
    with pytest.raises(ValueError, match="reserved engine-internal column names"):
        run_characteristic_sort(df, {"score": "score"})


def test_nan_control_drops_bond() -> None:
    """A bond with NaN in the control column is dropped from eligibility
    rather than silently lumped into a stripe by `_assign_groups`'s NaN-sort
    behaviour. Built on top of the section 7.5 panel: adding a bond with a
    NaN control and an extreme next-month return must NOT shift the
    double-sort answer from 0.03 -- it stays out of every stripe entirely.
    """
    panel = _double_sort_panel()
    # Add a bond with NaN control. If the engine silently classified it into
    # any stripe, the +50% return would dominate that stripe's spread and
    # the double-sort answer would visibly change.
    extra = pd.DataFrame(
        [
            {"cusip": "B9", "date": _me("2010-01"), "ret": 0.0, "size": 100.0,
             "score": 4.5, "control_value": float("nan")},
            {"cusip": "B9", "date": _me("2010-02"), "ret": 0.50, "size": 100.0,
             "score": 4.5, "control_value": float("nan")},
        ]
    )
    panel = pd.concat([panel, extra], ignore_index=True)
    result = run_characteristic_sort(
        panel,
        {
            "score": "score",
            "control": "control_value",
            "groups": 2,
            "control_groups": 2,
            "weighting": "equal",
            "min_bonds": 8,
        },
    )
    mr = result["monthly_returns"]
    assert len(mr) == 1
    # Unchanged from the section 7.5 result -- B9 was filtered out before
    # group assignment.
    assert mr["strategy_ret"].iloc[0] == pytest.approx(0.03, abs=1e-12)


def test_datetime_us_resolution_normalized_before_merge() -> None:
    """A panel with datetime64[us] dates must still produce correct calendar-
    strict joins."""
    rows = [
        {"cusip": "A", "date": "2010-01", "ret": 0.0, "size": 100.0, "score": 3.0},
        {"cusip": "B", "date": "2010-01", "ret": 0.0, "size": 100.0, "score": 1.0},
        {"cusip": "A", "date": "2010-02", "ret": 0.05, "size": 100.0, "score": 3.0},
        {"cusip": "B", "date": "2010-02", "ret": -0.02, "size": 100.0, "score": 1.0},
    ]
    df = _panel(rows)
    df["date"] = df["date"].astype("datetime64[us]")
    result = run_characteristic_sort(
        df,
        {"score": "score", "groups": 2, "weighting": "equal", "min_bonds": 2},
    )
    mr = result["monthly_returns"]
    assert len(mr) == 1
    assert mr["strategy_ret"].iloc[0] == pytest.approx(0.07, abs=1e-12)


def test_summarize_returns_non_datetime_index_returns_none_dates() -> None:
    """When the caller passes a Series with a RangeIndex, first_date /
    last_date are None (not integers leaking through)."""
    s = pd.Series([0.04, 0.02, 0.03])
    out = summarize_returns(s, nw_lags=0, months_per_year=12)
    assert out["first_date"] is None
    assert out["last_date"] is None


def test_summarize_returns_datetime_index_returns_timestamps() -> None:
    """When the caller passes a DatetimeIndex, first_date / last_date are
    Timestamps."""
    dates = pd.date_range("2020-01-31", periods=12, freq="ME")
    s = pd.Series([0.04, 0.02] * 6, index=dates)
    out = summarize_returns(s, nw_lags=0, months_per_year=12)
    assert isinstance(out["first_date"], pd.Timestamp)
    assert isinstance(out["last_date"], pd.Timestamp)
    assert out["first_date"] == dates.min()
    assert out["last_date"] == dates.max()


def test_nw_hac_lags_above_T_minus_1_is_clamped() -> None:
    """Requesting nw_lags > T-1 must clamp at T-1 (both the loop AND the
    kernel denominator) so the result equals what `nw_lags = T-1` would give."""
    s = pd.Series([0.01, 0.02, -0.01, 0.03, 0.0])  # T=5
    huge = summarize_returns(s, nw_lags=99, months_per_year=12)
    clamped = summarize_returns(s, nw_lags=4, months_per_year=12)  # T-1 = 4
    assert huge["nw_lags_used"] == 4
    assert huge["t_stat"] == pytest.approx(clamped["t_stat"], abs=1e-12)


def test_empty_result_monthly_returns_has_datetime_dtype() -> None:
    """An empty monthly_returns frame must still report datetime64[ns] for
    `date` so downstream consumers can rely on the dtype invariant."""
    df = pd.DataFrame(
        {
            "cusip": pd.Series([], dtype=object),
            "date": pd.Series([], dtype="datetime64[ns]"),
            "ret": pd.Series([], dtype=float),
            "size": pd.Series([], dtype=float),
            "score": pd.Series([], dtype=float),
        }
    )
    result = run_characteristic_sort(df, {"score": "score"})
    mr = result["monthly_returns"]
    assert pd.api.types.is_datetime64_any_dtype(mr["date"])
    assert pd.api.types.is_float_dtype(mr["strategy_ret"])


@pytest.mark.parametrize(
    "rulebook,match",
    [
        ({"score": "score", "groups": 1}, "integer >= 2"),
        ({"score": "score", "groups": "five"}, "integer >= 2"),
        ({"score": "score", "weighting": "median"}, "by_size"),
        ({"score": "score", "signal_lag": -1}, "non-negative"),
        ({"score": "score", "signal_lag": 1.5}, "non-negative"),
        ({"score": "score", "months_per_year": 0}, "positive"),
        ({"score": "score", "nw_lags": -3}, "None or a non-negative"),
        ({"score": "score", "nw_lags": "auto"}, "None or a non-negative"),
        ({"score": "score", "min_bonds": 0}, "positive"),
        ({"score": "score", "control": "ctrl", "control_groups": 1}, "control_groups"),
        ({"score": "score", "long_group": 7}, "long_group"),
        ({"score": "score", "short_group": -1}, "short_group"),
    ],
)
def test_apply_defaults_rejects_bad_inputs(rulebook, match) -> None:
    with pytest.raises(ValueError, match=match):
        _apply_defaults(rulebook)


def test_apply_defaults_rejects_non_dict_rulebook() -> None:
    with pytest.raises(TypeError, match="rulebook must be a dict"):
        _apply_defaults("score=foo")  # type: ignore[arg-type]


def test_validate_panel_rejects_non_dataframe() -> None:
    with pytest.raises(TypeError, match="DataFrame"):
        _validate_panel([], _apply_defaults({"score": "score"}))  # type: ignore[arg-type]


def test_validate_panel_rejects_non_datetime_date_column() -> None:
    df = pd.DataFrame(
        {
            "cusip": ["A"],
            "date": ["2010-01-31"],  # string, not datetime
            "ret": [0.0],
            "size": [100.0],
            "score": [1.0],
        }
    )
    with pytest.raises(TypeError, match="datetime64"):
        _validate_panel(df, _apply_defaults({"score": "score"}))


def test_safe_rate_rejects_non_dataframe() -> None:
    rows = [
        {"cusip": "A", "date": "2010-01", "ret": 0.0, "size": 100.0, "score": 1.0},
    ]
    with pytest.raises(TypeError, match="safe_rate must be a pandas DataFrame"):
        run_characteristic_sort(
            _panel(rows),
            {"score": "score"},
            safe_rate={"rf": [0.001]},  # type: ignore[arg-type]
        )


def test_regress_rejects_non_series_y() -> None:
    factors = pd.DataFrame({"date": pd.to_datetime(["2010-01-31"]), "F1": [0.0]})
    with pytest.raises(TypeError, match="Series"):
        regress_on_benchmark([0.01], factors, nw_lags=0)  # type: ignore[arg-type]


def test_regress_rejects_non_dataframe_factors() -> None:
    y = pd.Series([0.01], index=pd.to_datetime(["2010-01-31"]))
    with pytest.raises(TypeError, match="DataFrame"):
        regress_on_benchmark(y, {"F1": [0.0]}, nw_lags=0)  # type: ignore[arg-type]


def test_regress_rejects_factors_missing_date_column() -> None:
    y = pd.Series([0.01], index=pd.to_datetime(["2010-01-31"]))
    with pytest.raises(ValueError, match="date"):
        regress_on_benchmark(y, pd.DataFrame({"F1": [0.0]}), nw_lags=0)


def test_regress_rejects_factors_with_no_factor_columns() -> None:
    y = pd.Series([0.01], index=pd.to_datetime(["2010-01-31"]))
    with pytest.raises(ValueError, match="at least one factor"):
        regress_on_benchmark(y, pd.DataFrame({"date": pd.to_datetime(["2010-01-31"])}), nw_lags=0)


def test_regress_empty_intersection_returns_nan_result() -> None:
    """y dates and factor dates don't overlap -> T=0 -> all NaN."""
    y = pd.Series([0.01], index=pd.to_datetime(["2010-01-31"]))
    factors = pd.DataFrame(
        {"date": pd.to_datetime(["2020-12-31"]), "F1": [0.0]}
    )
    out = regress_on_benchmark(y, factors, nw_lags=0)
    assert out["n_obs"] == 0
    assert math.isnan(out["alpha"])
    assert math.isnan(out["betas"]["F1"])


def test_summarize_returns_accepts_list_input() -> None:
    """Non-Series input is coerced internally."""
    out = summarize_returns([0.01, 0.02, 0.03], nw_lags=0, months_per_year=12)
    assert out["n_months"] == 3
    assert out["average"] == pytest.approx(0.02, abs=1e-12)


def test_summarize_returns_all_nan_returns_n_months_zero() -> None:
    """A Series that drops to length 0 after dropna returns an empty-result
    dict without raising."""
    s = pd.Series([float("nan"), float("nan"), float("nan")])
    out = summarize_returns(s, nw_lags=0, months_per_year=12)
    assert out["n_months"] == 0
    assert math.isnan(out["average"])
    assert math.isnan(out["bumpiness"])
    assert math.isnan(out["sharpe"])
    assert math.isnan(out["t_stat"])


def test_form_legs_zero_total_size_skips_stripe() -> None:
    """A leg whose sizes sum to zero under by_size weighting -> stripe
    skipped (NaN spread). Verified end-to-end via the engine."""
    rows = [
        # Formation: Jan 2010 -- one bond in long, one in short, with 0 sizes
        {"cusip": "A", "date": "2010-01", "ret": 0.0, "size": 0.0, "score": 2.0},
        {"cusip": "B", "date": "2010-01", "ret": 0.0, "size": 0.0, "score": 1.0},
        # Realisation: Feb 2010
        {"cusip": "A", "date": "2010-02", "ret": 0.05, "size": 100.0, "score": 2.0},
        {"cusip": "B", "date": "2010-02", "ret": -0.02, "size": 100.0, "score": 1.0},
    ]
    result = run_characteristic_sort(
        _panel(rows),
        {"score": "score", "groups": 2, "weighting": "by_size", "min_bonds": 2},
    )
    # Stripe skipped -> month skipped.
    assert len(result["monthly_returns"]) == 0


def test_end_to_end_with_defaults_runs_and_returns_expected_shape() -> None:
    """Build a 24-month, 12-bond panel and run with all defaults. Sanity-check
    the result dict's keys, shapes, and that 23 realisation rows are produced
    (last formation month has no realisation)."""
    rng = np.random.default_rng(123)
    months = pd.date_range("2010-01-31", periods=24, freq="ME")
    rows = []
    for bid in [f"B{i:02d}" for i in range(1, 13)]:
        base_score = rng.uniform(0, 1)
        for d in months:
            rows.append(
                {
                    "cusip": bid,
                    "date": d,
                    "ret": rng.normal(scale=0.02),
                    "size": rng.uniform(50, 200),
                    "score": base_score + rng.normal(scale=0.1),
                }
            )
    panel = pd.DataFrame(rows)
    result = run_characteristic_sort(panel, {"score": "score"})

    assert set(result.keys()) == {
        "monthly_returns",
        "summary",
        "relationship_to_benchmark",
        "settings_used",
        "bookkeeping",
    }
    mr = result["monthly_returns"]
    assert list(mr.columns) == ["date", "strategy_ret", "long_ret", "short_ret", "n_bonds"]
    # 24 formation months -> 23 realisation rows (last formation has no
    # realisation row to draw next_ret from).
    assert len(mr) == 23
    # monthly_returns sorted ascending by date
    assert mr["date"].is_monotonic_increasing
    # Summary populated
    assert result["summary"]["n_months"] == 23
    assert result["summary"]["months_per_year"] == 12
    # Settings used has all defaults filled
    s = result["settings_used"]
    assert s["groups"] == 5
    assert s["weighting"] == "by_size"
    assert s["signal_lag"] == 0
    assert s["min_bonds"] == 5
    assert s["control"] is None
    assert s["long_group"] == 4
    assert s["short_group"] == 0
    # No benchmark provided -> empty dict
    assert result["relationship_to_benchmark"] == {}


def test_end_to_end_with_benchmark_populates_relationship() -> None:
    """Pass a synthetic benchmark and confirm relationship_to_benchmark is populated."""
    rng = np.random.default_rng(7)
    months = pd.date_range("2010-01-31", periods=36, freq="ME")
    rows = []
    for bid in [f"B{i:02d}" for i in range(1, 11)]:
        base_score = rng.uniform(0, 1)
        for d in months:
            rows.append(
                {
                    "cusip": bid,
                    "date": d,
                    "ret": rng.normal(scale=0.02),
                    "size": 100.0,
                    "score": base_score + rng.normal(scale=0.1),
                }
            )
    panel = pd.DataFrame(rows)
    bench = pd.DataFrame(
        {
            "date": months,
            "MKT": rng.normal(scale=0.02, size=36),
            "SMB": rng.normal(scale=0.02, size=36),
        }
    )
    result = run_characteristic_sort(panel, {"score": "score"}, benchmark=bench)
    rel = result["relationship_to_benchmark"]
    assert "alpha" in rel and "alpha_t" in rel
    assert set(rel["betas"].keys()) == {"MKT", "SMB"}
    assert set(rel["beta_t"].keys()) == {"MKT", "SMB"}
    assert rel["n_obs"] > 0
