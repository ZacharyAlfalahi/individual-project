"""Unit tests for the composable cleaning primitives (FL-D21a). Synthetic frames;
each primitive is a pure function, so no pipeline / real data is touched."""

import numpy as np
import pandas as pd
import pytest

from agents.quant.library.cleaning_primitives import (
    apply_profile_screens,
    asof_keep_mask,
    commission_mask,
    daily_vwap,
    data_entry_mask,
    is_reversal_record,
    min_volume_mask,
    price_range_mask,
    when_issued_mask,
)


def _frame(rows):
    return pd.DataFrame(rows)


# --- BBW screens ---

def test_price_range_observation_level():
    df = _frame({"rptd_pr": [4.99, 5.0, 101.3, 1000.0, 1000.01, np.nan, -0.35]})
    m = price_range_mask(df)
    assert list(m) == [False, True, True, True, False, False, False]


def test_min_volume_face_dollars():
    df = _frame({"entrd_vol_qt": [9999.0, 10000.0, 25000.0, np.nan]})
    assert list(min_volume_mask(df)) == [False, True, True, False]


def test_when_issued_drops_Y():
    df = _frame({"wis_fl": ["Y", "N", "", None]})
    assert list(when_issued_mask(df)) == [False, True, True, True]


def test_locked_in_special_sales_settlement():
    from agents.quant.library.cleaning_primitives import (
        locked_in_mask,
        settlement_mask,
        special_sales_mask,
    )
    df = _frame({
        "lckd_in_ind":     ["Y", None, "",  None, None],
        "spcl_trd_fl":     [None, "Y", "",  None, None],
        "sale_cndtn_cd":   ["@",  "@", "Z", "@",  None],
        "days_to_sttl_ct": ["1",  "1", "1", "5",  None],
    })
    assert list(locked_in_mask(df)) == [False, True, True, True, True]
    # row1 special price flag; row2 special condition code Z
    assert list(special_sales_mask(df)) == [True, False, False, True, True]
    # settlement > 2 dropped; MISSING kept (declared convention)
    assert list(settlement_mask(df)) == [True, True, True, False, True]


# --- Jostova screens ---

def test_commission_drops_Y_keeps_NA():
    # pre-2012 the flag is populated; NA (post-2012) rows are kept (not flagged).
    df = _frame({"cmsn_trd": ["Y", "N", None]})
    assert list(commission_mask(df)) == [False, True, True]


def test_data_entry_drops_nonpositive():
    df = _frame({"rptd_pr": [-0.35, 0.0, 0.75, 101.3]})
    assert list(data_entry_mask(df)) == [False, False, True, True]


# --- Regime-aware reversal classification (G3 Feb-2012 shift) ---

def test_reversal_record_regime_aware():
    df = _frame({
        "trd_exctn_dt": ["2010-06-01", "2010-06-01", "2015-06-01", "2015-06-01", "2015-06-01"],
        "trc_st":       ["W",          "T",          "R",          "X",          "T"],
        "asof_cd":      ["",           "",           "",           "",           "R"],
    })
    # pre-2012: W is a reversal; post-2012: R and X are; asof_cd 'R' always is.
    assert list(is_reversal_record(df)) == [True, False, True, True, True]


def test_pre2012_R_code_not_reversal():
    # trc_st 'R' pre-2012 is NOT the reversal marker (that's 'W'); guards the branch.
    df = _frame({"trd_exctn_dt": ["2010-06-01"], "trc_st": ["R"], "asof_cd": [""]})
    assert list(is_reversal_record(df)) == [False]


# --- P4/P5 as-of (FL-D21h) ---

def test_asof_retain_default_and_drop_side():
    df = _frame({"asof_cd": ["", "A", "R", "D", "X"]})
    # default: blank + A (retain) + R (left for matching); D/X (P5) dropped.
    assert list(asof_keep_mask(df)) == [True, True, True, False, False]
    # drop side of the P4 diagnostic: A no longer retained.
    assert list(asof_keep_mask(df, retain_asof=False)) == [True, False, True, False, False]


# --- Shared VWAP ---

def test_daily_vwap_weights_by_volume():
    df = _frame({
        "cusip_id":     ["A", "A", "A", "B"],
        "trd_exctn_dt": ["2015-06-01"] * 3 + ["2015-06-01"],
        "rptd_pr":      [100.0, 102.0, np.nan, 50.0],   # NaN excluded
        "entrd_vol_qt": [1000.0, 3000.0, 5000.0, 2000.0],
    })
    out = daily_vwap(df).sort_values("cusip_id").reset_index(drop=True)
    # A: (100*1000 + 102*3000)/4000 = 101.5 (NaN-price row excluded)
    a = out[out["cusip_id"] == "A"].iloc[0]
    assert a["price_vwap"] == pytest.approx(101.5)
    assert a["total_vol"] == 4000.0 and a["n_trades"] == 2
    assert a["min_price"] == 100.0 and a["max_price"] == 102.0


# --- Profile screen composition ---

def test_bbw_profile_screens_compose():
    df = _frame({
        "rptd_pr":         [101.3, 4.0,  101.3, 101.3, 101.3],
        "entrd_vol_qt":    [25000.0, 25000.0, 9999.0, 25000.0, 25000.0],
        "wis_fl":          ["",    "",   "",    "Y",   ""],
        "lckd_in_ind":     ["",    "",   "",    "",    ""],
        "spcl_trd_fl":     ["",    "",   "",    "",    ""],
        "sale_cndtn_cd":   ["@",   "@",  "@",   "@",   "Z"],
        "days_to_sttl_ct": ["1",   "1",  "1",   "1",   "1"],
    })
    # row0 passes; row1 fails price; row2 fails volume; row3 fails when-issued;
    # row4 fails special-sales.
    assert list(apply_profile_screens(df, "bbw_2019")) == [True, False, False, False, False]


def test_jostova_profile_has_no_price_range():
    # A $2 price (below BBW's $5) SURVIVES jostova (no price range, FL-D21 D3/R2).
    df = _frame({"rptd_pr": [2.0, -0.35], "cmsn_trd": ["N", "N"]})
    assert list(apply_profile_screens(df, "jostova_2013")) == [True, False]  # only data-entry drops <=0


def test_unknown_profile_raises():
    with pytest.raises(KeyError):
        apply_profile_screens(_frame({"rptd_pr": [1.0]}), "nope")
