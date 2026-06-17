"""
Intra-month price windows for the str LIB decomposition.

DRR-2026's month-end vs month-begin reversal decomposition (Table 2 Panel A;
BBW_anchor_implementation_spec.md §5.1, §8) needs two price samplings per
(cusip, month) that the whole-month VWAP panel does not carry:

  * price_end   = volume-weighted price over the LAST  `window_days` business
                  days of the calendar month,
  * price_begin = volume-weighted price over the FIRST `window_days` business
                  days of the calendar month.

These let the holding return be measured either from the prior month-end price
(which the sorting signal also ends on — the shared price that creates the
latent-illiquidity / correlated-errors reversal bias) or from the first-of-month
price (sampled a few business days after the signal's endpoint, breaking the
shared-noise link). The spec's "first/last 5 business days of the following
month" is exactly this windowing.

Pure function; reads the daily VWAP panel, writes nothing. `window_days` is a
caller-supplied parameter (sourced from thresholds.yaml by the script).

CONFIRM-ON-READ (§9, §11): the exact DRR price-window definition (calendar
business days vs the bond's own trade days; window length) is to be confirmed
against the DRR-2026 GitHub construction code. This module uses the literal
"first/last N business days of the calendar month" reading; the decomposition
gate is on direction + proportion, which the CEIV mechanism delivers robustly
regardless of the precise window.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _month_bday_cutoffs(periods, window_days: int) -> pd.DataFrame:
    """For each month Period, the begin/end business-day cutoff dates: trades on
    or before `begin_cut` are in the first-`window_days` window; trades on or
    after `end_cut` are in the last-`window_days` window. Short months (fewer
    than 2*window_days business days) clamp `n` to the available count, so the
    two windows can meet but never invert."""
    rows = []
    for p in periods:
        bdays = pd.bdate_range(p.start_time.normalize(), p.end_time.normalize())
        n = min(window_days, len(bdays))
        rows.append((p, bdays[n - 1], bdays[-n]))
    return pd.DataFrame(rows, columns=["_period", "_begin_cut", "_end_cut"])


def month_window_prices(
    daily: pd.DataFrame,
    window_days: int = 5,
    *,
    id_col: str = "cusip_id",
    date_col: str = "trd_exctn_dt",
    price_col: str = "price_vwap",
    vol_col: str = "total_vol",
) -> pd.DataFrame:
    """Volume-weighted begin-of-month and end-of-month prices per (cusip, month).

    Parameters
    ----------
    daily : daily VWAP panel with id/date/price/volume columns.
    window_days : number of business days in each window (default 5).

    Returns
    -------
    DataFrame[cusip, date, price_begin, price_end] with `date` the month-end
    timestamp. A (cusip, month) with no trades in a window gets NaN for that
    side. price = sum(price*vol)/sum(vol) over the window's trades.
    """
    required = {id_col, date_col, price_col, vol_col}
    missing = required - set(daily.columns)
    if missing:
        raise ValueError(f"daily panel missing required columns: {sorted(missing)}")
    if not isinstance(window_days, int) or window_days < 1:
        raise ValueError("window_days must be a positive integer")

    df = daily[[id_col, date_col, price_col, vol_col]].copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df["_period"] = df[date_col].dt.to_period("M")

    periods = [pd.Period(p, freq="M") for p in df["_period"].unique()]
    cut_df = _month_bday_cutoffs(periods, window_days)
    df = df.merge(cut_df, on="_period", how="left")

    d = df[date_col].dt.normalize()
    in_begin = d <= df["_begin_cut"]
    in_end = d >= df["_end_cut"]
    df["_pxv"] = df[price_col].to_numpy(dtype=float) * df[vol_col].to_numpy(dtype=float)

    def _vwap(mask: pd.Series, name: str) -> pd.Series:
        sub = df[mask]
        g = sub.groupby([id_col, "_period"], sort=True).agg(
            _pxv=("_pxv", "sum"), _v=(vol_col, "sum")
        )
        with np.errstate(invalid="ignore", divide="ignore"):
            price = g["_pxv"] / g["_v"]
        return price.where(g["_v"] > 0).rename(name)

    price_begin = _vwap(in_begin, "price_begin")
    price_end = _vwap(in_end, "price_end")

    out = pd.concat([price_begin, price_end], axis=1).reset_index()
    out["date"] = (
        out["_period"].dt.to_timestamp(how="end").dt.normalize() + pd.offsets.MonthEnd(0)
    )
    out = out.rename(columns={id_col: "cusip"}).drop(columns=["_period"])
    return out[["cusip", "date", "price_begin", "price_end"]].sort_values(
        ["cusip", "date"]
    ).reset_index(drop=True)
