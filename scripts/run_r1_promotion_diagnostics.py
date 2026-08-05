"""
R1 promotion diagnostics (FL-D21e) — dev window only, run BEFORE any family build.

For each UNSTATED primitive of each as-published profile, toggle it ALONE against
the profile's diagnostic baseline and measure, at the bond-month level:

  (a) diffuse materiality  — share of bond-months with |Δr| >= diffuse_delta_bp
  (b) concentrated magnitude — max |Δr|
  (c) selection-concentration lift — P(extreme | affected) / P(extreme | unaffected),
      extreme = top ∪ bottom decile of the return-based registered sorts
      (prior-1m return = the str sort; 6m cumulative return = the mom6 sort),
      computed on the BASELINE panel.

A primitive is PROMOTED to a run-both-ways envelope if ANY criterion fires
(thresholds from the frozen `cleaning_promotion:` block, tag
cleaning-promotion-prereg). Non-promoted primitives keep the faithful-to-silence
default, tagged UNKNOWN with the diagnostic recorded.

Declared diagnostic conventions (data-level; the D16 line sits at portfolio
formation, not returns):
  * month-end price  = the last traded day's volume-weighted price in the month
    (a diagnostic convention shared by baseline and every variant);
  * monthly return   = px[m]/px[m-1] - 1 over CONSECUTIVE calendar months, clean
    price, no accrued interest;
  * the diagnostic baseline is the profile's screened normal-trade set
    (trc_st == 'T', profile asof policy, profile screens). The cross-chunk
    matching engine and interdealer dedup are orthogonal row-pairing steps: the
    E1 dedup |Δr| is measured EXACTLY from the dedup-on vs dedup-off family
    builds instead (FL-D21g), not approximated here.

Dev window only: rows dated >= 2022-01-01 are excluded from every aggregate
(the raw file is streamed, as preprocess_trace does; no holdout STATISTIC is
computed or surfaced).

Usage:
  python scripts/run_r1_promotion_diagnostics.py [--profile bbw_2019|jostova_2013|all]
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

RAW_FILE = REPO_ROOT / "data" / "trace_enhanced_repull.csv.gz"
THRESHOLDS_FILE = REPO_ROOT / "docs" / "thresholds.yaml"
OUT_FILE = REPO_ROOT / "data" / "development" / "r1_promotion_diagnostics.json"
DEV_END = "2022-01-01"          # rows >= this date are holdout-window; excluded

_READ_COLS = [
    "cusip_id", "trd_exctn_dt", "trc_st", "asof_cd", "wis_fl", "cmsn_trd",
    "rptd_pr", "entrd_vol_qt", "lckd_in_ind", "spcl_trd_fl", "sale_cndtn_cd",
    "days_to_sttl_ct",
]
_DTYPES = {c: str for c in _READ_COLS if c not in ("rptd_pr", "entrd_vol_qt")}
_DTYPES.update({"rptd_pr": float, "entrd_vol_qt": float})


def load_promotion_thresholds() -> dict:
    cfg = yaml.safe_load(THRESHOLDS_FILE.read_text())
    block = cfg.get("cleaning_promotion")
    if block is None:
        raise KeyError("cleaning_promotion block missing from thresholds.yaml "
                       "(the FL-D21e pre-registered constants are required)")
    return block


# ---------------------------------------------------------------------------
# Per-profile diagnostic baselines + variant definitions (row predicates).
# Each variant maps to (sign, mask_fn): sign -1 = variant REMOVES the masked
# rows from the baseline; +1 = variant ADDS them (they are outside the baseline).
# ---------------------------------------------------------------------------

def _s(df, col):
    return df[col].astype("string").fillna("")


def _baseline_mask(df: pd.DataFrame, profile: str) -> pd.Series:
    from agents.quant.library.cleaning_primitives import apply_profile_screens
    keep = (_s(df, "trc_st") == "T")
    # profile asof policy (P4 retain): blank or 'A'
    keep &= _s(df, "asof_cd").isin(["", "A"])
    keep &= apply_profile_screens(df, profile)
    # non-positive / non-finite prices and volumes cannot price a VWAP
    keep &= np.isfinite(df["rptd_pr"]) & (df["rptd_pr"] > 0)
    keep &= np.isfinite(df["entrd_vol_qt"]) & (df["entrd_vol_qt"] > 0)
    return keep


def _variants(profile: str) -> dict:
    """UNSTATED primitives for this profile (FL-D21f R2 + FL-D21h P4/P5)."""
    from agents.quant.library.cleaning_primitives import (
        locked_in_mask,
        min_volume_mask,
        price_range_mask,
        settlement_mask,
        special_sales_mask,
        when_issued_mask,
    )
    common = {
        # P4: variant DROPS the retained as-of 'A' rows (retain is the default)
        "p4_asof_drop": (-1, lambda df: _s(df, "asof_cd") == "A"),
        # P5: variant RETAINS the dropped 'D'/'X' rows (drop is the default)
        "p5_asof_retain": (+1, lambda df: _s(df, "asof_cd").isin(["D", "X"])),
    }
    if profile == "jostova_2013":
        return {
            **common,
            "price_range": (-1, lambda df: ~price_range_mask(df)),
            "min_volume": (-1, lambda df: ~min_volume_mask(df)),
            "when_issued": (-1, lambda df: ~when_issued_mask(df)),
            "locked_in": (-1, lambda df: ~locked_in_mask(df)),
            "special_sales": (-1, lambda df: ~special_sales_mask(df)),
            "settlement": (-1, lambda df: ~settlement_mask(df)),
        }
    if profile == "bbw_2019":
        return {
            **common,
            # Jostova's steps, unstated for BBW (commission is pre-2012-only in
            # the data; data-entry <=0 is a subset of the price range, omitted)
            "commission": (-1, lambda df: _s(df, "cmsn_trd") == "Y"),
        }
    raise KeyError(profile)


# ---------------------------------------------------------------------------
# Pure computation core (unit-tested): daily aggregates -> criteria
# ---------------------------------------------------------------------------

def monthly_last_day_price(daily: pd.DataFrame, pxv_col: str, vol_col: str) -> pd.DataFrame:
    """Per (cusip, month): the last traded day's VWAP under the given aggregate
    columns (vol > 0 qualifies a day). Returns columns [cusip, month, px]."""
    d = daily[daily[vol_col] > 0].copy()
    if d.empty:
        return pd.DataFrame(columns=["cusip", "month", "px"])
    d["month"] = d["date"].str.slice(0, 7)
    idx = d.groupby(["cusip", "month"])["date"].idxmax()
    last = d.loc[idx].copy()
    last["px"] = last[pxv_col] / last[vol_col]
    return last[["cusip", "month", "px"]]


def _monthly_returns(px: pd.DataFrame) -> pd.DataFrame:
    """Consecutive-calendar-month clean-price returns from [cusip, month, px]."""
    if px.empty:
        return pd.DataFrame(columns=["cusip", "month", "ret"])
    px = px.sort_values(["cusip", "month"]).reset_index(drop=True)
    month_start = pd.PeriodIndex(px["month"], freq="M").to_timestamp()
    prev = px.groupby("cusip")["px"].shift(1)
    prev_month = (
        month_start.to_series(index=px.index).groupby(px["cusip"]).shift(1)
    )
    consec = np.asarray(
        (month_start - pd.offsets.MonthBegin(1)) == prev_month.values
    )
    ret = px["px"] / prev - 1.0
    out = px[["cusip", "month"]].copy()
    out["ret"] = ret.where(consec & prev.notna().values)
    return out.dropna(subset=["ret"])


def compute_criteria(daily: pd.DataFrame, variant_cols: dict, thresholds: dict) -> dict:
    """From the merged daily frame (cusip, date, base_pxv, base_vol, and per-
    variant delta columns d_pxv_<v>/d_vol_<v>), compute the three FL-D21e
    criteria per variant. Pure; unit-tested on synthetic frames."""
    base_px = monthly_last_day_price(daily, "base_pxv", "base_vol")
    base_ret = _monthly_returns(base_px).rename(columns={"ret": "ret_base"})
    n_base = len(base_ret)

    # Return-based registered sorts on the BASELINE panel (criterion c):
    # prior-1m return (str) and 6m cumulative return (mom6).
    br = base_ret.sort_values(["cusip", "month"]).reset_index(drop=True)
    br["sig_str"] = br.groupby("cusip")["ret_base"].shift(1)
    br["sig_mom6"] = (
        np.log1p(br.groupby("cusip")["ret_base"].shift(1))
        .groupby(br["cusip"]).rolling(6, min_periods=6).sum()
        .reset_index(level=0, drop=True)
    )
    frac = float(thresholds.get("extreme_decile_frac", 0.10))
    extreme = pd.Series(False, index=br.index)
    for sig in ("sig_str", "sig_mom6"):
        s = br[sig]
        if s.notna().sum() == 0:
            continue
        lo, hi = s.quantile(frac), s.quantile(1 - frac)
        extreme |= s.notna() & ((s <= lo) | (s >= hi))
    br["extreme"] = extreme

    out = {"n_baseline_bond_months": int(n_base), "variants": {}}
    for v, (dpxv, dvol) in variant_cols.items():
        var_daily = daily[["cusip", "date"]].copy()
        var_daily["v_pxv"] = daily["base_pxv"] + daily[dpxv]
        var_daily["v_vol"] = daily["base_vol"] + daily[dvol]
        var_px = monthly_last_day_price(var_daily, "v_pxv", "v_vol")
        var_ret = _monthly_returns(var_px).rename(columns={"ret": "ret_var"})
        m = br.merge(var_ret, on=["cusip", "month"], how="left")
        # Δr on baseline bond-months; a variant-missing month counts as affected
        # with Δr = NaN -> treated at the max via the affected flag, magnitude
        # criteria use the defined overlap.
        dr = (m["ret_var"] - m["ret_base"]).abs()
        affected = dr.gt(0).fillna(False) | m["ret_var"].isna()
        share = float(dr.ge(thresholds["diffuse_delta_bp"] / 1e4).sum()) / max(n_base, 1)
        max_dr = float(dr.max()) if dr.notna().any() else 0.0
        p_ext_aff = float(m.loc[affected, "extreme"].mean()) if affected.any() else 0.0
        p_ext_un = float(m.loc[~affected, "extreme"].mean()) if (~affected).any() else 0.0
        lift = (p_ext_aff / p_ext_un) if p_ext_un > 0 else (np.inf if p_ext_aff > 0 else 0.0)
        fired = {
            "a_diffuse": share >= thresholds["diffuse_share_min"],
            "b_magnitude": max_dr >= thresholds["max_abs_delta_r"],
            "c_selection": (affected.any() and lift >= thresholds["selection_lift_min"]),
        }
        out["variants"][v] = {
            "n_affected_bond_months": int(affected.sum()),
            "share_ge_1bp": share,
            "max_abs_delta_r": max_dr,
            "selection_lift": None if not np.isfinite(lift) else float(lift),
            "selection_lift_infinite": bool(np.isinf(lift)),
            "criteria_fired": fired,
            "promoted": bool(any(fired.values())),
        }
    return out


# ---------------------------------------------------------------------------
# Streaming accumulation (one pass per profile)
# ---------------------------------------------------------------------------

def accumulate_daily(profile: str, chunk_size: int = 4_000_000) -> tuple:
    variants = _variants(profile)
    parts = []
    reader = pd.read_csv(RAW_FILE, usecols=_READ_COLS, dtype=_DTYPES,
                         chunksize=chunk_size, on_bad_lines="skip")
    for i, chunk in enumerate(reader):
        chunk = chunk[chunk["trd_exctn_dt"].astype(str) < DEV_END]   # dev only
        if chunk.empty:
            continue
        finite = (
            np.isfinite(chunk["rptd_pr"]) & (chunk["rptd_pr"] > 0)
            & np.isfinite(chunk["entrd_vol_qt"]) & (chunk["entrd_vol_qt"] > 0)
        )
        chunk = chunk[finite]
        base = _baseline_mask(chunk, profile)
        pxv = chunk["rptd_pr"] * chunk["entrd_vol_qt"]
        agg = pd.DataFrame({
            "cusip": chunk["cusip_id"].astype(str),
            "date": chunk["trd_exctn_dt"].astype(str),
            "base_pxv": pxv.where(base, 0.0),
            "base_vol": chunk["entrd_vol_qt"].where(base, 0.0),
        })
        for v, (sign, fn) in variants.items():
            m = fn(chunk)
            if sign < 0:
                m = m & base          # removes rows that are IN the baseline
            else:
                m = m & ~base         # adds rows that are OUT of the baseline
                # the added rows must still be normal priced trades
                m = m & (_s(chunk, "trc_st") == "T")
            agg[f"d_pxv_{v}"] = (pxv * sign).where(m, 0.0)
            agg[f"d_vol_{v}"] = (chunk["entrd_vol_qt"] * sign).where(m, 0.0)
        parts.append(agg.groupby(["cusip", "date"], sort=False).sum().reset_index())
        if len(parts) >= 25:
            parts = [pd.concat(parts).groupby(["cusip", "date"], sort=False).sum().reset_index()]
        if (i + 1) % 10 == 0:
            print(f"  [{profile}] chunk {i + 1} done", flush=True)
    daily = pd.concat(parts).groupby(["cusip", "date"], sort=False).sum().reset_index()
    variant_cols = {v: (f"d_pxv_{v}", f"d_vol_{v}") for v in variants}
    return daily, variant_cols


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--profile", choices=["bbw_2019", "jostova_2013", "all"],
                    default="all")
    args = ap.parse_args()
    thresholds = load_promotion_thresholds()
    profiles = (["bbw_2019", "jostova_2013"] if args.profile == "all"
                else [args.profile])

    results = {
        "run_timestamp": datetime.now(timezone.utc).isoformat(),
        "thresholds_used": thresholds,
        "prereg_tag": "cleaning-promotion-prereg",
        "window": "development only (< 2022-01-01)",
        "conventions": ("month-end = last traded day's VWAP; consecutive-month "
                        "clean-price returns; baseline = screened normal-trade "
                        "set; E1 dedup measured exactly from the on/off family "
                        "builds, not here"),
        "profiles": {},
    }
    for p in profiles:
        print(f"Profile {p}: streaming dev-window daily aggregates...", flush=True)
        daily, variant_cols = accumulate_daily(p)
        print(f"  {len(daily):,} cusip-days; computing criteria...", flush=True)
        results["profiles"][p] = compute_criteria(daily, variant_cols, thresholds)
        del daily

    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    OUT_FILE.write_text(json.dumps(results, indent=2))
    print(f"\nWritten: {OUT_FILE}")
    for p, res in results["profiles"].items():
        print(f"\n[{p}] baseline bond-months: {res['n_baseline_bond_months']:,}")
        for v, r in res["variants"].items():
            print(f"  {v:15} promoted={r['promoted']}  share1bp={r['share_ge_1bp']:.5f} "
                  f"max|Δr|={r['max_abs_delta_r']:.4f} lift={r['selection_lift']}")


if __name__ == "__main__":
    main()
