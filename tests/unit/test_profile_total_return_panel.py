"""Unit tests for scripts/build_profile_total_return_panel.py.

Synthetic panels with analytically-known totals:
  * a zero-coupon bond: total return == clean return (within 1e-12), the accrual
    control group;
  * a 12%-annual monthly-coupon bond on a flat price: each adjacent month pays
    coupon/frequency = 1.0 per 100 par with zero accrued interest (last coupon is the
    month-end itself), so total return = clean + 1.0/P_{t-1} exactly;
plus xret exactness, schema preservation, finite-clean invariance, and fail-loud rf gaps.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scripts.build_profile_total_return_panel import build_total_return_profiles, PROFILES

MONTH_ENDS = [pd.Timestamp("2019-01-31"), pd.Timestamp("2019-02-28"), pd.Timestamp("2019-03-31")]
RF = 0.001


def _clean_ret(prices):
    """Adjacent-month clean growth for a single cusip's consecutive months."""
    out = [np.nan]
    for i in range(1, len(prices)):
        out.append((prices[i] - prices[i - 1]) / prices[i - 1])
    return out


def _profiles_frame(rows):
    """rows: list of (cusip, price). Builds the full profile schema for BOTH pids,
    giving each pid the same synthetic prices so either family exercises the kernel."""
    recs = []
    prices = [p for _, p in rows]
    clean = _clean_ret(prices)
    for (cusip, price), dt, cr in zip(rows, MONTH_ENDS, clean):
        rec = {"cusip": cusip, "date": dt}
        for pid in PROFILES:
            rec[f"price_eom_{pid}"] = float(price)
            rec[f"ret_{pid}"] = cr
            rec[f"xret_{pid}"] = cr - RF if not np.isnan(cr) else np.nan
            rec[f"n_trades_{pid}"] = 5
            rec[f"total_vol_{pid}"] = 1_000_000.0
            rec[f"last_trade_date_{pid}"] = dt
        recs.append(rec)
    cols = ["cusip", "date"] + [
        f"{b}_{pid}" for pid in PROFILES
        for b in ("last_trade_date", "n_trades", "price_eom", "ret", "total_vol", "xret")
    ]
    return pd.DataFrame(recs)[cols]


def _fisd(coupon, freq, maturity="2030-12-31", cusip="B"):
    return pd.DataFrame([{
        "cusip": cusip, "coupon": coupon, "interest_frequency": freq,
        "day_count_basis": "30/360", "maturity": pd.Timestamp(maturity),
    }])


def _rf(months=("2019-01", "2019-02", "2019-03")):
    return pd.DataFrame({"year_month": list(months), "rf_monthly": [RF] * len(months)})


def test_zero_coupon_invariance():
    """coupon==0 -> total return equal to the input clean return within 1e-12."""
    prof = _profiles_frame([("Z", 100.0), ("Z", 105.0), ("Z", 103.0)])
    fisd = _fisd(coupon=0.0, freq=0, cusip="Z")
    out, diag = build_total_return_profiles(prof, fisd, _rf())
    for pid in PROFILES:
        clean = prof[f"ret_{pid}"].to_numpy()
        tot = out[f"ret_{pid}"].to_numpy()
        both = np.isfinite(clean) & np.isfinite(tot)
        assert np.array_equal(clean[both], tot[both]), f"{pid} Z not invariant"
        assert diag[pid]["z_coupon_invariant"] is True


def test_monthly_coupon_flat_price_analytic():
    """12% annual, monthly coupons, flat price 100: AI=0 and coupon_paid=1.0/month,
    so total return = clean(0) + 1.0/100 = 0.01 on each adjacent month."""
    prof = _profiles_frame([("B", 100.0), ("B", 100.0), ("B", 100.0)])
    fisd = _fisd(coupon=12.0, freq=12, cusip="B")
    out, _ = build_total_return_profiles(prof, fisd, _rf())
    for pid in PROFILES:
        tot = out[f"ret_{pid}"].to_numpy()
        assert np.isnan(tot[0]), "first month has no adjacent predecessor"
        assert tot[1] == pytest.approx(0.01, abs=1e-12)
        assert tot[2] == pytest.approx(0.01, abs=1e-12)


def test_coupon_adds_carry_over_clean():
    """A coupon bond's total return strictly exceeds its clean return (positive carry)."""
    prof = _profiles_frame([("B", 100.0), ("B", 101.0), ("B", 99.0)])
    fisd = _fisd(coupon=12.0, freq=12, cusip="B")
    out, _ = build_total_return_profiles(prof, fisd, _rf())
    for pid in PROFILES:
        clean = prof[f"ret_{pid}"].to_numpy()
        tot = out[f"ret_{pid}"].to_numpy()
        for i in (1, 2):
            assert tot[i] > clean[i], f"{pid} month {i}: total {tot[i]} !> clean {clean[i]}"


def test_xret_equals_total_ret_minus_rf():
    prof = _profiles_frame([("B", 100.0), ("B", 100.0), ("B", 100.0)])
    fisd = _fisd(coupon=12.0, freq=12, cusip="B")
    out, _ = build_total_return_profiles(prof, fisd, _rf())
    for pid in PROFILES:
        ret = out[f"ret_{pid}"].to_numpy()
        xret = out[f"xret_{pid}"].to_numpy()
        fin = np.isfinite(ret)
        assert np.allclose(xret[fin], ret[fin] - RF, atol=1e-15)


def test_schema_and_rowcount_preserved():
    prof = _profiles_frame([("B", 100.0), ("B", 100.0), ("B", 100.0)])
    fisd = _fisd(coupon=6.0, freq=2, cusip="B")
    out, diag = build_total_return_profiles(prof, fisd, _rf())
    assert list(out.columns) == list(prof.columns)
    assert len(out) == len(prof) == diag["rows"]


def test_rf_gap_fails_loud():
    prof = _profiles_frame([("B", 100.0), ("B", 100.0), ("B", 100.0)])
    fisd = _fisd(coupon=0.0, freq=0, cusip="B")
    with pytest.raises(AssertionError, match="rf coverage gap"):
        build_total_return_profiles(prof, fisd, _rf(months=("2019-01", "2019-02")))  # missing 03


def test_missing_price_column_fails_loud():
    prof = _profiles_frame([("B", 100.0), ("B", 100.0), ("B", 100.0)]).drop(columns=["price_eom_bbw_2019"])
    with pytest.raises(KeyError, match="price_eom_bbw_2019"):
        build_total_return_profiles(prof, _fisd(0.0, 0, cusip="B"), _rf())


def test_semiannual_accrual_pure_ai_month():
    """Semi-annual 12% coupons, maturity day 28 -> coupon months are Feb/Aug. March is a
    NON-coupon month whose return is pure accrued interest: last coupon = Feb-28 (AI=0),
    so March-31 AI = 12 * days_30E360(Feb28, Mar31)/360 = 12*32/360 = 1.06667 per 100 par,
    and on a flat 100 price the total return is exactly 1.06667/100 = 0.0106667.
    (February itself IS a coupon month, so it is not the pure-accrual case.)"""
    prof = _profiles_frame([("B", 100.0), ("B", 100.0), ("B", 100.0)])
    fisd = _fisd(coupon=12.0, freq=2, maturity="2030-08-28", cusip="B")
    out, _ = build_total_return_profiles(prof, fisd, _rf())
    for pid in PROFILES:
        tot = out[f"ret_{pid}"].to_numpy()
        assert tot[2] == pytest.approx(0.0106667, abs=1e-5), "March pure-AI carry"
        assert np.isfinite(tot[1]) and tot[1] > 0.0, "February coupon month is also positive"


# ---- flexible multi-cusip / gapped / cross-family cases (coverage the 3-month
# single-cusip fixtures above do not exercise) ----

def _general_frame(rows):
    """rows: list of (cusip, date_str, price_bbw, price_jostova); a NaN price is allowed
    (models a bond-month present in one profile family but not the other, as the
    outer-merge in build_profile_monthly_panel produces). Clean returns are computed per
    family with the same per-cusip adjacency (gap==1) rule as the production build."""
    df = pd.DataFrame(
        [{"cusip": c, "date": pd.Timestamp(d), "price_eom_bbw_2019": pb,
          "price_eom_jostova_2013": pj} for c, d, pb, pj in rows]
    ).sort_values(["cusip", "date"]).reset_index(drop=True)
    df["_ym"] = df["date"].dt.to_period("M").astype("int64")
    for pid, pcol in (("bbw_2019", "price_eom_bbw_2019"), ("jostova_2013", "price_eom_jostova_2013")):
        prev_p = df.groupby("cusip")[pcol].shift(1)
        prev_period = df.groupby("cusip")["_ym"].shift(1)
        gap = df["_ym"] - prev_period
        ret = (df[pcol] - prev_p) / prev_p
        ret = ret.where(gap == 1, np.nan)
        df[f"ret_{pid}"] = ret
        df[f"xret_{pid}"] = ret - RF
        df[f"n_trades_{pid}"] = 5
        df[f"total_vol_{pid}"] = 1_000_000.0
        df[f"last_trade_date_{pid}"] = df["date"]
    cols = ["cusip", "date"] + [
        f"{b}_{pid}" for pid in PROFILES
        for b in ("last_trade_date", "n_trades", "price_eom", "ret", "total_vol", "xret")
    ]
    return df[cols]


def test_cusip_boundary_yields_nan_first_row():
    """The first month of each cusip has no in-cusip predecessor -> NaN (no cross-cusip return)."""
    rows = [("A", "2019-01-31", 100.0, 100.0), ("A", "2019-02-28", 110.0, 110.0),
            ("B", "2019-02-28", 50.0, 50.0), ("B", "2019-03-31", 55.0, 55.0)]
    out, _ = build_total_return_profiles(_general_frame(rows), _fisd(0.0, 0, cusip="Z"), _rf())
    for pid in PROFILES:
        tot = out[f"ret_{pid}"].to_numpy()
        assert np.isnan(tot[0]) and np.isnan(tot[2]), "first row of each cusip is NaN"
        assert np.isfinite(tot[1]) and np.isfinite(tot[3]), "within-cusip adjacent months finite"


def test_month_gap_breaks_total_return():
    """A non-adjacent predecessor (Jan -> Mar, February missing) -> NaN, matching the clean rule."""
    rows = [("A", "2019-01-31", 100.0, 100.0), ("A", "2019-03-31", 110.0, 110.0)]
    out, _ = build_total_return_profiles(_general_frame(rows), _fisd(0.0, 0, cusip="Z"), _rf())
    for pid in PROFILES:
        assert np.isnan(out[f"ret_{pid}"].to_numpy()[1]), "gap>1 must break the return"


def test_cross_family_nan_price_isolated():
    """A bond-month present for one family but NaN-priced for the other: the missing family's
    return is NaN for that month and the next adjacent month (its predecessor price is gone),
    while the present family is unaffected."""
    rows = [("A", "2019-01-31", 100.0, 100.0),
            ("A", "2019-02-28", 100.0, np.nan),   # jostova price missing this month
            ("A", "2019-03-31", 100.0, 100.0)]
    out, _ = build_total_return_profiles(_general_frame(rows), _fisd(0.0, 0, cusip="Z"), _rf())
    bbw = out["ret_bbw_2019"].to_numpy()
    jos = out["ret_jostova_2013"].to_numpy()
    assert np.isfinite(bbw[1]) and np.isfinite(bbw[2]), "present family unaffected"
    assert np.isnan(jos[1]) and np.isnan(jos[2]), "missing-price month and its successor are NaN"
