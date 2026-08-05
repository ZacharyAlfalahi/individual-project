"""
cleaning_primitives.py — composable transaction-level cleaning primitives for the
per-paper as-published baseline profiles (FL-D21a).

Each primitive is a PURE function on a transaction frame returning a boolean
keep-mask (or a transformed frame), so a profile is a selected, ordered subset of
primitives plus profile-specific steps. This module holds the CHUNK-LOCAL
primitives — the record-classification helpers (P1/P2/P3 record identity, P4/P5
as-of), the profile-specific screens (price range, volume floor, when-issued,
commission, data-entry), and the shared daily VWAP. The CROSS-CHUNK matching
engine (building cancelled_keys across scans; interdealer/agency dedup P6/P7) is
NOT re-implemented here — it lives in the shared preprocess path and is selected
per profile; these primitives are what the profile layers on top of it.

All classifications trace to the FL-D21 L1 table + the gate findings
(docs/data/registers/fl_d21_gate_results.md): rptd_pr is per-$100 face,
entrd_vol_qt is face-value dollars, and the trc_st/asof_cd code vocabulary shifts
at Feb-2012 (reversals: W pre-2012 -> R/X post-2012; cmsn_trd ~100% NA post-2012).

No holdout rows are ever read by any caller of these primitives (dev window only,
FL-D21e); the primitives themselves are date-agnostic pure functions.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# --- Code vocabularies (G2/G3, verified against the full 447M-row tabulation) ---
# trc_st: T = normal; C = correction; R = reversal (post-2012); W = withdrawal
# (pre-2012 reversal marker, retired 2012); X = cancellation (post-2012); Y = rare.
NORMAL_TRC_ST = "T"
CANCELLATION_TRC_ST = frozenset({"C", "X", "W", "Y"})   # records that cancel/withdraw
REVERSAL_TRC_ST_PRE2012 = "W"
REVERSAL_TRC_ST_POST2012 = frozenset({"R", "X"})
REGIME_SPLIT = pd.Timestamp("2012-02-06")               # DN-2014 Feb-2012 reporting change

# asof_cd: blank = on-time; A = as-of/late (P4, retain); R = reversal (P3);
# D = delayed (P5, ~2005 only); X = cancellation (P5).
ASOF_ONTIME = ""
ASOF_ASOF = "A"        # P4 — genuine late-reported executions; retained
ASOF_REVERSAL = "R"    # P3
ASOF_OTHER = frozenset({"D", "X"})   # P5


# ---------------------------------------------------------------------------
# Record-classification helpers (P1/P2/P3 identity; the matching engine consumes these)
# ---------------------------------------------------------------------------

def is_normal_trade(df: pd.DataFrame) -> pd.Series:
    """P1 base: a normal executable trade record (trc_st == 'T'). The
    cancellation/correction/reversal RECORDS (non-T) are the match targets the
    cross-chunk engine pairs to originals."""
    return df["trc_st"].astype("string").fillna("") == NORMAL_TRC_ST


def is_reversal_record(df: pd.DataFrame) -> pd.Series:
    """P3: a reversal record, regime-aware (FL-D21 G3). Pre-2012 reversals carry
    trc_st 'W' (or asof_cd 'R'); post-2012 they carry trc_st 'R'/'X'. Used by the
    matching engine to net out the reversed original in every arm."""
    dt = pd.to_datetime(df["trd_exctn_dt"], errors="coerce")
    pre = dt < REGIME_SPLIT
    trc = df["trc_st"].astype("string").fillna("")
    asof = df["asof_cd"].astype("string").fillna("")
    pre_rev = pre & ((trc == REVERSAL_TRC_ST_PRE2012) | (asof == ASOF_REVERSAL))
    post_rev = (~pre) & (trc.isin(REVERSAL_TRC_ST_POST2012) | (asof == ASOF_REVERSAL))
    return pre_rev | post_rev


# ---------------------------------------------------------------------------
# P4/P5 — as-of handling (FL-D21h)
# ---------------------------------------------------------------------------

def asof_keep_mask(df: pd.DataFrame, *, retain_asof: bool = True) -> pd.Series:
    """P4/P5 as-of handling (FL-D21h). Blank asof_cd is always kept (on-time).
    P4 ('A', as-of/late executions) is RETAINED by default (genuine executions,
    execution-dated; deletion would be an unstated OFF-arm step) — set
    retain_asof=False to run the drop side of the P4 diagnostic. P3 'R'
    (reversal) is handled by the matching engine, not dropped here. P5 ('D'/'X')
    takes the faithful-to-silence default (dropped)."""
    asof = df["asof_cd"].astype("string").fillna(ASOF_ONTIME)
    keep = (asof == ASOF_ONTIME)
    if retain_asof:
        keep = keep | (asof == ASOF_ASOF)
    # 'R' rows are left for the matching engine (kept here so it can see them);
    # 'D'/'X' (P5) are dropped (faithful-to-silence default).
    keep = keep | (asof == ASOF_REVERSAL)
    return keep


# ---------------------------------------------------------------------------
# Profile-specific transaction screens (STATED steps; provenance per FL-D21)
# ---------------------------------------------------------------------------

def price_range_mask(df: pd.DataFrame, lo: float = 5.0, hi: float = 1000.0) -> pd.Series:
    """BBW criterion 4 (FL-D21i): keep observations with rptd_pr in [lo, hi], per
    $100 face (5-1000% of par). Observation-level (INFERRED), not bond-level.
    Non-finite prices fail the screen. jostova_2013 does NOT apply this (no price
    range stated; R2 UNKNOWN-default)."""
    p = pd.to_numeric(df["rptd_pr"], errors="coerce")
    return p.notna() & (p >= lo) & (p <= hi)


def min_volume_mask(df: pd.DataFrame, threshold: float = 10000.0) -> pd.Series:
    """BBW volume floor (fn 12): keep entrd_vol_qt >= threshold (face-value $).
    jostova_2013 does NOT apply this (R2 UNKNOWN-default)."""
    v = pd.to_numeric(df["entrd_vol_qt"], errors="coerce")
    return v.notna() & (v >= threshold)


def when_issued_mask(df: pd.DataFrame) -> pd.Series:
    """BBW when-issued removal (STATED): drop wis_fl == 'Y'. Asymmetric (FL-D21f
    R2): STATED for BBW, UNKNOWN-default for jostova_2013."""
    return df["wis_fl"].astype("string").fillna("") != "Y"


def locked_in_mask(df: pd.DataFrame) -> pd.Series:
    """BBW locked-in removal (STATED §3.1): drop lckd_in_ind == 'Y'. Sample basis
    (fl_d21_gate_results): the column is 'Y' (~1.2%) or NA."""
    return df["lckd_in_ind"].astype("string").fillna("") != "Y"


def special_sales_mask(df: pd.DataFrame) -> pd.Series:
    """BBW special-sales-condition removal (STATED §3.1): keep only regular-way
    trades — sale_cndtn_cd blank/NA or '@'; drop the special codes (Z/R/A/N/W/C,
    ~2.7% of the sample) and any spcl_trd_fl == 'Y' special-price row."""
    cndtn = df["sale_cndtn_cd"].astype("string").fillna("")
    regular = cndtn.isin(["", "@"])
    not_special_px = df["spcl_trd_fl"].astype("string").fillna("") != "Y"
    return regular & not_special_px


def settlement_mask(df: pd.DataFrame, max_days: float = 2.0) -> pd.Series:
    """BBW settlement removal (STATED §3.1): drop settlement > max_days days
    (days_to_sttl_ct). Rows with a MISSING settlement count are KEPT — the field
    is ~59% NA and a missing value cannot be judged (declared convention)."""
    d = pd.to_numeric(df["days_to_sttl_ct"], errors="coerce")
    return d.isna() | (d <= max_days)


def commission_mask(df: pd.DataFrame) -> pd.Series:
    """Jostova commission removal (STATED via BKMX, FL-D21c): drop cmsn_trd == 'Y'.
    NB (G3): the cmsn_trd flag is ~100% NA post-2012, so this executes only
    pre-2012; post-2012 it is a no-op (nothing flagged) and the step is a dated
    UNKNOWN in B_s. Rows with NA cmsn_trd are kept (not flagged as commission)."""
    return df["cmsn_trd"].astype("string").fillna("") != "Y"


def data_entry_mask(df: pd.DataFrame) -> pd.Series:
    """Jostova data-entry screen (STATED): drop negative prices. (The maturity <
    issue/trade-date half needs FISD dates and is applied at the universe layer,
    not here.)"""
    p = pd.to_numeric(df["rptd_pr"], errors="coerce")
    return p.notna() & (p > 0)


# ---------------------------------------------------------------------------
# Shared daily price — VWAP (BBW volume-weighted == Jostova trade-size-weighted;
# FL-D21 records these as one primitive, the wording difference is not a difference)
# ---------------------------------------------------------------------------

def daily_vwap(df: pd.DataFrame, *, id_col: str = "cusip_id",
               date_col: str = "trd_exctn_dt") -> pd.DataFrame:
    """Volume-/trade-size-weighted daily price per (cusip, date): the shared VWAP
    primitive (BKMX p.4225 'trade-weighted price'). Emits the daily-layer schema
    the panel builder expects: price_vwap, total_vol, n_trades, min_price,
    max_price. Rows with non-finite price or volume are excluded from the
    aggregation (they cannot weight a VWAP)."""
    w = df.copy()
    w["_pr"] = pd.to_numeric(w["rptd_pr"], errors="coerce")
    w["_vol"] = pd.to_numeric(w["entrd_vol_qt"], errors="coerce")
    w = w[np.isfinite(w["_pr"]) & np.isfinite(w["_vol"]) & (w["_vol"] > 0)]
    if w.empty:
        return pd.DataFrame(columns=[id_col, date_col, "price_vwap", "total_vol",
                                     "n_trades", "min_price", "max_price"])
    w["_pxv"] = w["_pr"] * w["_vol"]
    g = w.groupby([id_col, date_col], sort=True)
    out = g.agg(
        _pxv_sum=("_pxv", "sum"),
        total_vol=("_vol", "sum"),
        n_trades=("_pr", "size"),
        min_price=("_pr", "min"),
        max_price=("_pr", "max"),
    ).reset_index()
    out["price_vwap"] = out["_pxv_sum"] / out["total_vol"]
    return out[[id_col, date_col, "price_vwap", "total_vol",
                "n_trades", "min_price", "max_price"]]


# ---------------------------------------------------------------------------
# Profile step registry — the ordered transaction-screen subset per profile
# (the L1 matching engine + dedup envelope are selected separately, upstream).
# ---------------------------------------------------------------------------

# Each entry: (step_name, mask_fn, provenance). Applied as an AND of keep-masks in
# order; VWAP is the terminal aggregation (not a mask). Mirrors FL-D21's profile
# step lists; the matching engine (P1/P2/P3) and dedup envelope (P6/P7) are applied
# by the shared path before these.
BBW_2019_SCREENS = (
    ("price_range", price_range_mask, "INFERRED"),      # FL-D21i (obs-level)
    ("min_volume", min_volume_mask, "STATED"),          # fn 12
    ("when_issued", when_issued_mask, "STATED"),        # §3.1
    ("locked_in", locked_in_mask, "STATED"),            # §3.1
    ("special_sales", special_sales_mask, "STATED"),    # §3.1
    ("settlement", settlement_mask, "STATED"),          # §3.1 (> 2 days)
)
JOSTOVA_2013_SCREENS = (
    ("commission", commission_mask, "STATED"),          # via BKMX; pre-2012 only (G3)
    ("data_entry", data_entry_mask, "STATED"),          # §1.1
)

PROFILE_SCREENS = {
    "bbw_2019": BBW_2019_SCREENS,
    "jostova_2013": JOSTOVA_2013_SCREENS,
}


def apply_profile_screens(df: pd.DataFrame, profile_id: str) -> pd.Series:
    """The AND of a profile's transaction-screen keep-masks (order-independent for
    a boolean AND). Returns the keep-mask; the caller applies it and then the
    matching engine / dedup / VWAP. Raises for an unknown profile."""
    if profile_id not in PROFILE_SCREENS:
        raise KeyError(f"unknown profile {profile_id!r}; known: {sorted(PROFILE_SCREENS)}")
    keep = pd.Series(True, index=df.index)
    for _name, fn, _prov in PROFILE_SCREENS[profile_id]:
        keep &= fn(df)
    return keep
