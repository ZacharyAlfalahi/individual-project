"""
Build the gamma / ILLIQ signal — the one new BBW characteristic (Bao, Pan &
Wang 2011 eq.2; docs/quant/specs/BBW_anchor_implementation_spec.md §1).

gamma = sign_multiplier * Cov(Δp_{d}, Δp_{d+1})

over consecutive within-month daily log-price changes, where the daily price is
the within-day volume-weighted clean price (the daily panel's `price_vwap`) and
Δp_d = log(P_d) − log(P_{d-1}). Higher gamma = more illiquid (sign_multiplier
= −1). One value per bond-month.

Observation screen — DRR-2023 RELAXED (default, a DESIGN choice, §1.3):
  * a daily change is recognised only if its two trade days are <= max_gap_bdays
    (7) business days apart;
  * a PAIR (Δp_d, Δp_{d+1}) needs both adjacent changes valid;
  * the month needs >= min_pairs (5) valid pairs, else gamma = NaN;
  * NO "% of business days traded" requirement.
The BPW-strict screen (>= 10 pairs AND >= 75% of business days traded) is the
rejected pure-BPW alternative, retained behind signals.gamma_illiq.bpw_strict
(off by default).

Family-indexed per A9: `gamma_raw` (from trace_daily_raw) and `gamma_corr` (from
trace_daily_corr_filtered) — the same daily sources the monthly panel aggregates.

Output: data/development/signals/gamma_illiq.parquet with columns
  cusip, date, gamma_raw, gamma_corr.

Usage:
  python scripts/build_gamma_illiq.py

Requires:
  data/development/trace_daily_raw.parquet
  data/development/trace_daily_corr_filtered.parquet
"""

import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
RAW_DAILY = REPO_ROOT / "data" / "development" / "trace_daily_raw.parquet"
CORR_DAILY = REPO_ROOT / "data" / "development" / "trace_daily_corr_filtered.parquet"
OUT_DIR = REPO_ROOT / "data" / "development" / "signals"
OUT_FILE = OUT_DIR / "gamma_illiq.parquet"
REPORT_OUT = OUT_DIR / "gamma_illiq_report.json"
THRESHOLDS_FILE = REPO_ROOT / "docs" / "thresholds.yaml"


def load_config() -> dict:
    with open(THRESHOLDS_FILE) as f:
        cfg = yaml.safe_load(f)
    block = cfg.get("signals", {}).get("gamma_illiq")
    if block is None:
        raise KeyError("thresholds.yaml is missing signals.gamma_illiq block")
    for key in ("min_pairs", "max_gap_bdays", "sign_multiplier", "cov_ddof"):
        if key not in block:
            raise KeyError(f"thresholds.yaml signals.gamma_illiq missing '{key}'")
    return block


def thresholds_sha256() -> str:
    return hashlib.sha256(THRESHOLDS_FILE.read_bytes()).hexdigest()


def git_commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT,
                              capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        return "unknown"


def compute_gamma(
    daily: pd.DataFrame,
    min_pairs: int,
    max_gap_bdays: int,
    sign_multiplier: float = -1.0,
    cov_ddof: int = 1,
    strict: dict | None = None,
    *,
    id_col: str = "cusip_id",
    date_col: str = "trd_exctn_dt",
    price_col: str = "price_vwap",
) -> pd.DataFrame:
    """BPW gamma per (cusip, month). Returns cusip, date (month-end), gamma.

    Changes are formed only between consecutive trade days WITHIN the same
    calendar month (so a change never straddles a month boundary), require a
    business-day gap <= max_gap_bdays, and a valid pair needs two adjacent valid
    changes. gamma is NaN unless >= min_pairs valid pairs (relaxed screen);
    `strict` (BPW) additionally requires its own min_pairs and a >= min_bday_frac
    share of the month's business days traded.
    """
    if not all(c in daily.columns for c in (id_col, date_col, price_col)):
        raise ValueError(f"daily panel must contain: {id_col}, {date_col}, {price_col}")
    if not isinstance(min_pairs, int) or min_pairs < 2:
        raise ValueError("min_pairs must be an integer >= 2")

    df = daily[[id_col, date_col, price_col]].copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values([id_col, date_col], kind="mergesort").reset_index(drop=True)

    cusip = df[id_col].to_numpy()
    # Group key = month-end timestamp (carried directly so there is no
    # period-ordinal round-trip to misparse).
    ym = (
        df[date_col].dt.to_period("M").dt.to_timestamp(how="end").dt.normalize()
        + pd.offsets.MonthEnd(0)
    ).to_numpy()
    price = df[price_col].to_numpy(dtype=float)
    with np.errstate(invalid="ignore", divide="ignore"):
        logp = np.where(price > 0, np.log(price), np.nan)

    n = len(df)
    same_grp_prev = np.zeros(n, dtype=bool)
    same_grp_prev[1:] = (cusip[1:] == cusip[:-1]) & (ym[1:] == ym[:-1])

    # Daily change into row i (from the previous within-month trade day).
    change = np.full(n, np.nan)
    change[1:] = logp[1:] - logp[:-1]
    change[~same_grp_prev] = np.nan

    # Business-day gap between consecutive within-month trade days.
    dates_d = df[date_col].to_numpy().astype("datetime64[D]")
    bd_gap = np.full(n, np.iinfo(np.int32).max, dtype=np.int64)
    if n > 1:
        gap_vals = np.busday_count(dates_d[:-1], dates_d[1:])
        bd_gap[1:] = gap_vals
    change_valid = (
        same_grp_prev & (bd_gap <= max_gap_bdays)
        & ~np.isnan(change) & ~np.isnan(logp) & np.r_[False, ~np.isnan(logp[:-1])]
    )

    # Pair change at row i with the change at row i+1 (same group, both valid).
    same_grp_next = np.r_[same_grp_prev[1:], False]
    pair_valid = np.zeros(n, dtype=bool)
    pair_valid[:-1] = change_valid[:-1] & same_grp_next[:-1] & change_valid[1:]

    x = change[pair_valid]
    y = np.r_[change[1:], np.nan][pair_valid]  # next change aligned to row i
    pdf = pd.DataFrame({
        "cusip": cusip[pair_valid],
        "_ym": ym[pair_valid],
        "x": x, "y": y, "xy": x * y,
    })
    agg = pdf.groupby(["cusip", "_ym"], sort=True).agg(
        n=("x", "size"), sx=("x", "sum"), sy=("y", "sum"), sxy=("xy", "sum")
    )
    nn = agg["n"].to_numpy(dtype=float)
    with np.errstate(invalid="ignore", divide="ignore"):
        cov = (agg["sxy"].to_numpy() - agg["sx"].to_numpy() * agg["sy"].to_numpy() / nn) / (nn - cov_ddof)
    gamma = sign_multiplier * cov
    gamma[nn < min_pairs] = np.nan

    out = agg.reset_index()[["cusip", "_ym"]].copy()  # _ym is the month-end timestamp
    out["gamma"] = gamma

    if strict and strict.get("enabled"):
        # BPW-strict extra gate: >= strict.min_pairs pairs AND traded on
        # >= min_bday_frac of the month's business days.
        s_min_pairs = int(strict["min_pairs"])
        s_frac = float(strict["min_bday_frac"])
        traded_days = (
            df.assign(_ym=ym).groupby([id_col, "_ym"])[date_col]
            .nunique().rename("traded_days").reset_index()
            .rename(columns={id_col: "cusip"})
        )
        bdays = {
            ts: len(pd.bdate_range(pd.Timestamp(ts).to_period("M").start_time,
                                   pd.Timestamp(ts).to_period("M").end_time))
            for ts in pd.unique(out["_ym"])
        }
        out = out.merge(traded_days, on=["cusip", "_ym"], how="left")
        out["_bdays"] = out["_ym"].map(bdays)
        frac = out["traded_days"] / out["_bdays"]
        fail = (agg["n"].to_numpy() < s_min_pairs) | (frac.to_numpy() < s_frac)
        out.loc[fail, "gamma"] = np.nan
        out = out.drop(columns=["traded_days", "_bdays"])

    out = out.rename(columns={"_ym": "date"})
    return out[["cusip", "date", "gamma"]].sort_values(["cusip", "date"]).reset_index(drop=True)


def compute_dual_family(cfg: dict) -> pd.DataFrame:
    strict = cfg.get("bpw_strict")
    kwargs = dict(
        min_pairs=int(cfg["min_pairs"]),
        max_gap_bdays=int(cfg["max_gap_bdays"]),
        sign_multiplier=float(cfg["sign_multiplier"]),
        cov_ddof=int(cfg["cov_ddof"]),
        strict=strict,
    )
    print(f"  raw daily: {RAW_DAILY.name}")
    raw = pd.read_parquet(RAW_DAILY, columns=["cusip_id", "trd_exctn_dt", "price_vwap"])
    g_raw = compute_gamma(raw, **kwargs).rename(columns={"gamma": "gamma_raw"})
    print(f"  corr daily: {CORR_DAILY.name}")
    corr = pd.read_parquet(CORR_DAILY, columns=["cusip_id", "trd_exctn_dt", "price_vwap"])
    g_corr = compute_gamma(corr, **kwargs).rename(columns={"gamma": "gamma_corr"})
    return g_raw.merge(g_corr, on=["cusip", "date"], how="outer")


def write_signal(signal: pd.DataFrame) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pandas(signal, preserve_index=False)
    meta = dict(table.schema.metadata or {})
    meta.update({
        b"signal_name":   b"gamma_illiq",
        b"signal_source": b"BPW_2011_eq2_DRR_relaxed_screen",
        b"primary_key":   b"cusip+date",
        b"families":      b"raw,corr",
        b"family_policy": b"A9_no_cross_family_mixing",
    })
    table = table.replace_schema_metadata(meta)
    tmp = OUT_FILE.with_suffix(".parquet.tmp")
    pq.write_table(table, str(tmp))
    os.replace(tmp, OUT_FILE)
    print(f"  Written: {OUT_FILE}")


def write_report(signal: pd.DataFrame, cfg: dict) -> None:
    report = {
        "run_timestamp": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit(),
        "thresholds_sha256": thresholds_sha256(),
        "thresholds_used": cfg,
        "counts": {
            "rows_total": int(len(signal)),
            "rows_non_null_raw": int(signal["gamma_raw"].notna().sum()),
            "rows_non_null_corr": int(signal["gamma_corr"].notna().sum()),
            "eligible_cusips_corr": int(signal.loc[signal["gamma_corr"].notna(), "cusip"].nunique()),
        },
        "diagnostics_corr": {
            "median": float(signal["gamma_corr"].median()),
            "pct_positive": float((signal["gamma_corr"] > 0).mean() * 100),
        },
        "signal_methodology": (
            "BPW (2011) eq.2 gamma = -Cov(Δp_d, Δp_{d+1}) over consecutive "
            "within-month daily log-price changes; DRR-2023 relaxed screen "
            "(>=5 pairs, <=7 business-day gap). Higher gamma = more illiquid."
        ),
    }
    tmp = REPORT_OUT.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(report, f, indent=2)
    os.replace(tmp, REPORT_OUT)
    print(f"  Report: {REPORT_OUT}")


def main():
    for f in (RAW_DAILY, CORR_DAILY):
        if not f.exists():
            print(f"ERROR: required input not found: {f}", file=sys.stderr)
            sys.exit(1)

    cfg = load_config()
    print(f"Config: min_pairs={cfg['min_pairs']}, max_gap_bdays={cfg['max_gap_bdays']}, "
          f"sign={cfg['sign_multiplier']}, bpw_strict={cfg.get('bpw_strict', {}).get('enabled', False)}")

    print("Computing gamma per family...")
    signal = compute_dual_family(cfg)
    raw_nn = int(signal["gamma_raw"].notna().sum())
    corr_nn = int(signal["gamma_corr"].notna().sum())
    print(f"  {raw_nn:,} raw non-NaN, {corr_nn:,} corr non-NaN")

    write_signal(signal)
    write_report(signal, cfg)

    gc = signal["gamma_corr"].dropna()
    print("\nDone.")
    print(f"  {len(signal):,} cusip-month rows  →  {OUT_FILE}")
    if len(gc):
        print(f"  gamma_corr: median {gc.median():.5f}, {(gc>0).mean()*100:.0f}% positive "
              f"(positive = illiquid, expected majority)")


if __name__ == "__main__":
    main()
