"""
Accrued interest + coupon engine for total-return construction (BBW_anchor_
implementation_spec.md §2.1).

Total monthly holding-period return:
    R = [(P_t + AI_t + C_t) − (P_{t-1} + AI_{t-1})] / (P_{t-1} + AI_{t-1})
where P is the clean price, AI the accrued interest, C the coupon paid during
the month. This module computes AI_t and C_t per (bond, month-end) from the FISD
coupon schedule.

Conventions (scoped against the eligible FISD universe):
  * Day-count: US 30/360 for ALL bonds (99.8% are genuinely 30/360). The ~0.08%
    on ACT/* bases use 30/360 as a flagged approximation — `day_count_fallback`
    marks them so the approximation is queryable at the factor level (did a
    fallback bond land in a leg in a surprising month?), not buried per-bond.
  * Coupon schedule derived BACKWARD from maturity (always populated) by
    12/frequency months, on maturity's day-of-month (clamped to month length).
  * Zero-coupon bonds (coupon == 0, the 57% Z control group) get AI = C = 0, so
    their total return is byte-identical to the clean-price return — the
    invariance regression that guards the accrual path.

Pure, vectorised, no IO.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

THIRTY_360 = "30/360"


def days_30_360(start, end) -> np.ndarray:
    """30E/360 (Eurobond) day count between `start` and `end` (datetime64 arrays):

      D1 = min(start.day, 30);  D2 = min(end.day, 30)   (both 31 → 30, always)
      days = 360*(y2-y1) + 30*(m2-m1) + (D2 - D1)

    30E/360 (not 30/360-US/NASD) because the panel settles on CALENDAR month-ends
    (often the 31st): clamping both endpoints makes AI accrue exactly to the
    coupon at period end and treats every month-end consistently. FISD records
    only the generic "30/360"; the US-vs-E difference is sub-day at 31st
    month-ends (negligible for factor levels) — flagged as a minor convention
    choice if a bond-level check ever cares.
    """
    s = pd.DatetimeIndex(pd.to_datetime(start))
    e = pd.DatetimeIndex(pd.to_datetime(end))
    d1 = np.minimum(s.day.to_numpy(), 30)
    d2 = np.minimum(e.day.to_numpy(), 30)
    return (
        360 * (e.year.to_numpy() - s.year.to_numpy())
        + 30 * (e.month.to_numpy() - s.month.to_numpy())
        + (d2 - d1)
    ).astype(np.int64)


def _month_index(dt: pd.DatetimeIndex) -> np.ndarray:
    return dt.year.to_numpy() * 12 + (dt.month.to_numpy() - 1)


def _build_date(month_index: np.ndarray, day: np.ndarray) -> pd.DatetimeIndex:
    """Construct dates from a month index (year*12 + month-1) and a day-of-month,
    clamping the day to each month's length."""
    year = month_index // 12
    month = month_index % 12 + 1
    first = pd.to_datetime({"year": year, "month": month, "day": np.ones_like(year)})
    dim = (pd.DatetimeIndex(first) + pd.offsets.MonthEnd(0)).day.to_numpy()
    day_clamped = np.minimum(day, dim)
    return pd.DatetimeIndex(pd.to_datetime({"year": year, "month": month, "day": day_clamped}))


def accrued_and_coupon(
    dates,
    maturity,
    coupon,
    frequency,
) -> tuple[np.ndarray, np.ndarray]:
    """Vectorised accrued interest and coupon-paid-this-month, per row.

    Parameters (all array-like, aligned)
    ----------
    dates : month-end dates (the panel grid).
    maturity : bond maturity date.
    coupon : annual coupon rate, in PERCENT of par (price units; e.g. 8.0).
    frequency : coupons per year (1, 2, 4, 12). 0/NaN → treated as no coupon.

    Returns (ai, coupon_paid), both in price units (per 100 par). Zero-coupon
    bonds (coupon == 0) and rows with no valid schedule (NaT maturity / freq not
    in {1,2,4,12}) return AI = C = 0 — their return stays the clean-price return.

    A coupon of `coupon/frequency` is paid in a month iff that month is a coupon
    month (the schedule's day-of-month lands in it); AI accrues 30/360 from the
    most recent coupon date to the month-end, resetting at each coupon.
    """
    t = pd.DatetimeIndex(pd.to_datetime(dates))
    mat = pd.DatetimeIndex(pd.to_datetime(maturity))
    cpn = np.asarray(coupon, dtype=float)
    freq = np.asarray(frequency, dtype=float)

    n = len(t)
    ai = np.zeros(n)
    coupon_paid = np.zeros(n)

    cpn_clean = np.where(np.isnan(cpn), 0.0, cpn)
    valid = (
        (cpn_clean > 0)
        & np.isin(freq, [1.0, 2.0, 4.0, 12.0])
        & ~np.asarray(mat.isna())
    )
    if not valid.any():
        return ai, coupon_paid

    freq_v = freq[valid].astype(int)
    p = 12 // freq_v                          # months per coupon period
    Mt = _month_index(t)[valid]
    Mmat = _month_index(mat)[valid]
    offset = np.mod(Mt - Mmat, p)             # months since the last coupon month (0..p-1)
    last_cpn_month = Mt - offset
    mat_day = mat.day.to_numpy()[valid]
    last_cpn = _build_date(last_cpn_month, mat_day)

    days = days_30_360(last_cpn.values, t[valid].values)
    cpn_v = cpn_clean[valid]
    ai[valid] = cpn_v * days / 360.0          # per 100 par
    coupon_paid[valid] = np.where(offset == 0, cpn_v / freq_v, 0.0)
    return ai, coupon_paid
