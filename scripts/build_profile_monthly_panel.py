"""
Profile monthly family columns (FL-D21a) — the meas_err OFF-arm baselines.

Builds a SEPARATE `monthly_panel_profiles.parquet` keyed on (cusip, date) that
carries the per-paper baseline family columns:
  price_eom_<pid>, ret_<pid>, xret_<pid>, n_trades_<pid>, total_vol_<pid>,
  last_trade_date_<pid>   for pid in {bbw_2019, jostova_2013}
where each `<pid>` family is that profile's DEDUP-ON baseline daily panel (the
lattice OFF-arm reference; the dedup-off + OFAT panels are the sensitivity
envelope, not the single differential panel — FL-D21g).

CRITICAL — this is ADDITIVE and does NOT touch `monthly_panel_maximal.parquet`.
The recorded maximal panel (its `*_raw`/`*_corr` columns + frozen Phase-1
checkpoint hash + the whole downstream corr chain) is preserved bit-for-bit; the
auditor merges these profile columns in at load time when PROFILES_BUILT (a
left-join on (cusip, date), so a bond-month absent from a profile is NaN there,
exactly like a family-missing month).

Every family is computed through `build_monthly_panel`'s OWN `aggregate_family`
and the SAME per-family adjacency-return + rf logic, so the profile columns are
constructed identically to raw/corr (no divergent code path).

The E1 dedup |Δr| diagnostic (FL-D21g) is computed here from each profile's
dedup-on vs dedup-off monthly returns and written to
`profile_e1_dedup_delta.json` — the EXACT bond-month return envelope the row
counts only bounded.

dev only (no holdout touched). Usage: python scripts/build_profile_monthly_panel.py
"""

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from build_monthly_panel import RF_FILE, aggregate_family  # noqa: E402

DEV = REPO_ROOT / "data" / "development"
OUT = DEV / "monthly_panel_profiles.parquet"
REPORT = DEV / "monthly_panel_profiles_report.json"
E1_REPORT = DEV / "profile_e1_dedup_delta.json"

# The lattice OFF-arm baseline family per profile = the dedup-ON panel.
PROFILE_FAMILIES = {
    "bbw_2019": DEV / "trace_daily_bbw_2019__dedup_on.parquet",
    "jostova_2013": DEV / "trace_daily_jostova_2013__dedup_on.parquet",
}


def _add_family_returns(panel: pd.DataFrame, family: str, rf: pd.DataFrame) -> pd.DataFrame:
    """Per-family adjacency return + xret, IDENTICAL to build_monthly_panel's
    inline loop (A1.5): a return exists only between calendar-consecutive months
    with a non-NaN price in this family."""
    panel = panel.sort_values(["cusip_id", "year_month"]).reset_index(drop=True)
    ym = pd.PeriodIndex(panel["year_month"], freq="M")
    panel["_ym_pd"] = ym
    price_col = f"price_eom_{family}"
    lag = panel.groupby("cusip_id")[price_col].shift(1)
    prev_ym = panel.groupby("cusip_id")["_ym_pd"].shift(1)
    panel[f"ret_{family}"] = (panel[price_col] - lag) / lag
    gap = (panel["_ym_pd"] - prev_ym).map(lambda x: x.n if pd.notna(x) else float("nan"))
    panel.loc[(gap != 1) | gap.isna(), f"ret_{family}"] = float("nan")
    panel.drop(columns=["_ym_pd"], inplace=True)
    return panel


def _monthly_for(daily_path: Path, family: str, rf: pd.DataFrame) -> pd.DataFrame:
    """One family's daily→monthly with ret + xret, via the shared aggregator."""
    df = aggregate_family(daily_path, family)
    df = _add_family_returns(df, family, rf)
    df = df.merge(rf, on="year_month", how="left", validate="m:1")
    df[f"xret_{family}"] = df[f"ret_{family}"] - df["rf_monthly"]
    return df


def _date_col(year_month: pd.Series) -> pd.Series:
    return (pd.PeriodIndex(year_month, freq="M").to_timestamp(how="end").normalize()
            + pd.offsets.MonthEnd(0))


def build() -> dict:
    if not RF_FILE.exists():
        raise FileNotFoundError(f"RF rate file not found: {RF_FILE}")
    rf = pd.read_parquet(RF_FILE)

    merged: pd.DataFrame | None = None
    counts: dict = {}
    for pid, daily in PROFILE_FAMILIES.items():
        fam = _monthly_for(daily, pid, rf).drop(columns=["rf_monthly"])
        counts[pid] = {"cusip_months": int(len(fam)),
                       "cusips": int(fam["cusip_id"].nunique()),
                       "ret_nonnull": int(fam[f"ret_{pid}"].notna().sum())}
        merged = fam if merged is None else merged.merge(
            fam, on=["cusip_id", "year_month"], how="outer")

    merged["cusip"] = merged["cusip_id"]
    merged["date"] = _date_col(merged["year_month"])
    fam_cols = [c for c in merged.columns
                if any(c.startswith(f"{b}_") for b in
                       ("price_eom", "ret", "xret", "n_trades", "total_vol", "last_trade_date"))]
    out = merged[["cusip", "date", *sorted(fam_cols)]].sort_values(["cusip", "date"])

    OUT.parent.mkdir(parents=True, exist_ok=True)
    tmp = OUT.with_suffix(".parquet.tmp")
    out.to_parquet(tmp, index=False)
    actual = pq.read_metadata(str(tmp)).num_rows
    assert actual == len(out), f"row count {actual} != {len(out)}"
    os.replace(tmp, OUT)
    return {"rows": int(len(out)), "columns": list(out.columns), "per_family": counts}


def e1_dedup_delta(rf: pd.DataFrame) -> dict:
    """The EXACT bond-month E1 dedup |Δr| envelope (FL-D21g): for each profile,
    compare the dedup-ON vs dedup-OFF monthly return on the common bond-months."""
    out: dict = {}
    for pid in PROFILE_FAMILIES:
        on = _monthly_for(DEV / f"trace_daily_{pid}__dedup_on.parquet", "on", rf)
        off = _monthly_for(DEV / f"trace_daily_{pid}__dedup_off.parquet", "off", rf)
        m = on[["cusip_id", "year_month", "ret_on"]].merge(
            off[["cusip_id", "year_month", "ret_off"]],
            on=["cusip_id", "year_month"], how="inner")
        d = (m["ret_on"] - m["ret_off"]).abs().dropna().to_numpy()
        out[pid] = {
            "common_bond_months": int(len(m)),
            "both_ret_present": int(len(d)),
            "share_ge_1bp": float((d >= 1e-4).mean()) if len(d) else 0.0,
            "mean_abs_delta_r": float(d.mean()) if len(d) else 0.0,
            "p99_abs_delta_r": float(np.percentile(d, 99)) if len(d) else 0.0,
            "max_abs_delta_r": float(d.max()) if len(d) else 0.0,
        }
    return out


def main() -> None:
    print("Building profile monthly family columns (bbw_2019, jostova_2013 baselines)...")
    res = build()
    print(f"  {OUT.name}: {res['rows']:,} cusip-month rows, "
          f"{len(res['columns'])} columns")

    print("Computing the E1 dedup |Δr| envelope (on vs off)...")
    rf = pd.read_parquet(RF_FILE)
    e1 = e1_dedup_delta(rf)
    for pid, s in e1.items():
        print(f"  {pid}: share>=1bp {s['share_ge_1bp']:.4f}  "
              f"mean|Δr| {s['mean_abs_delta_r']:.5f}  max|Δr| {s['max_abs_delta_r']:.4f}")

    for path, payload in ((REPORT, {"run_timestamp": datetime.now(timezone.utc).isoformat(),
                                    "stage": "profile_monthly_family_columns",
                                    "additive_to": "monthly_panel_maximal.parquet (NOT modified)",
                                    **res}),
                          (E1_REPORT, {"run_timestamp": datetime.now(timezone.utc).isoformat(),
                                       "stage": "e1_dedup_delta_envelope", "per_profile": e1})):
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload, indent=2, default=str))
        os.replace(tmp, path)
    print(f"  Reports: {REPORT.name}, {E1_REPORT.name}")


if __name__ == "__main__":
    main()
