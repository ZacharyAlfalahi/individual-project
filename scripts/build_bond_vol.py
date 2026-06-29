"""
Build the 24-month bond-return volatility signal (KPP TOTAL_VOL, Table A.I #27) as a
family-indexed signals parquet — per A9, emits `bond_vol_raw` (from `xret_raw`) and
`bond_vol_corr` (from `xret_corr`) as parallel columns. This signal plays a DUAL role in the
IPCA Workstream B feed: it is instrument #7 AND the scaler for the VOLScaled010 return-scaling
lane (`scripts/build_ipca_panel.py`).

For each (cusip, date) row of the maximal monthly panel, compute the rolling trailing-`window`
sample standard deviation (ddof=1) of monthly EXCESS returns PER FAMILY. Emitted (non-NaN) only
where the trailing window has at least `min_obs` non-NaN returns. Decimal monthly units.

Window/min_obs come from docs/thresholds.yaml under signals.bond_vol; nothing is hard-coded.
(The vol FLOOR used when scaling lives in agents/quant/library/configs/kpp_ipca.yaml:scaling.vol_floor.)

Output: data/development/signals/bond_vol.parquet with columns
  cusip, date, bond_vol_raw, bond_vol_corr.
Engine-mergeable on (cusip, date) at run time.

Usage:
  python scripts/build_bond_vol.py

Requires: data/development/monthly_panel_maximal.parquet (build_monthly_panel.py)
Source: [CODE JF2_pubdata.m:151 (TOTAL_VOL #27); PAPER fn.32]; docs/characteristic_registry_spec.md §1.
"""

import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
PANEL_FILE = REPO_ROOT / "data" / "development" / "monthly_panel_maximal.parquet"
OUT_DIR = REPO_ROOT / "data" / "development" / "signals"
OUT_FILE = OUT_DIR / "bond_vol.parquet"
REPORT_OUT = OUT_DIR / "bond_vol_report.json"
THRESHOLDS_FILE = REPO_ROOT / "docs" / "thresholds.yaml"

DDOF = 1  # sample standard deviation (DESIGN choice; documented in the report)


def load_config() -> dict:
    """Load bond_vol constants from thresholds.yaml. No defaults — every field must be present."""
    with open(THRESHOLDS_FILE) as f:
        cfg = yaml.safe_load(f)
    block = cfg.get("signals", {}).get("bond_vol")
    if block is None:
        raise KeyError("thresholds.yaml is missing signals.bond_vol block (window / min_obs)")
    for key in ("window", "min_obs"):
        if key not in block:
            raise KeyError(f"thresholds.yaml signals.bond_vol missing '{key}'")
    return block


def thresholds_sha256() -> str:
    return hashlib.sha256(THRESHOLDS_FILE.read_bytes()).hexdigest()


def compute_bond_vol(panel: pd.DataFrame, window: int, min_obs: int) -> pd.DataFrame:
    """Per-cusip rolling trailing-`window` sample std (ddof=1) of `ret`.

    Input panel must contain `cusip`, `date`, `ret`. Window is in panel rows (trailing months
    for that cusip), matching the row-window convention of build_var_5pct.py. Rows whose trailing
    window has < min_obs non-NaN returns get NaN. Output: cusip, date, bond_vol.
    """
    if not all(c in panel.columns for c in ("cusip", "date", "ret")):
        raise ValueError("panel must contain columns: cusip, date, ret")
    if not isinstance(window, int) or window < 2:
        raise ValueError("window must be an integer >= 2")
    if not isinstance(min_obs, int) or min_obs < 2 or min_obs > window:
        raise ValueError("min_obs must be an integer in [2, window]")

    panel = panel[["cusip", "date", "ret"]].sort_values(
        ["cusip", "date"], kind="mergesort"
    ).reset_index(drop=True)
    vol = (
        panel.groupby("cusip", sort=False)["ret"]
        .rolling(window=window, min_periods=min_obs)
        .std(ddof=DDOF)
        .reset_index(level=0, drop=True)
        .sort_index()
    )
    return pd.DataFrame({
        "cusip": panel["cusip"].to_numpy(),
        "date": panel["date"].to_numpy(),
        "bond_vol": vol.to_numpy(),
    })


def compute_dual_family(panel: pd.DataFrame, window: int, min_obs: int) -> pd.DataFrame:
    """Compute bond_vol on each family → cusip, date, bond_vol_raw, bond_vol_corr (A9)."""
    raw_in = panel[["cusip", "date", "xret_raw"]].rename(columns={"xret_raw": "ret"})
    raw_out = compute_bond_vol(raw_in, window, min_obs).rename(columns={"bond_vol": "bond_vol_raw"})
    corr_in = panel[["cusip", "date", "xret_corr"]].rename(columns={"xret_corr": "ret"})
    corr_out = compute_bond_vol(corr_in, window, min_obs).rename(columns={"bond_vol": "bond_vol_corr"})
    return raw_out.merge(corr_out, on=["cusip", "date"], how="outer")


def write_signal(signal: pd.DataFrame) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pandas(signal, preserve_index=False)
    meta = dict(table.schema.metadata or {})
    meta.update({
        b"signal_name": b"bond_vol",
        b"signal_source": b"KPP_TOTAL_VOL_24m_excess_return_std",
        b"primary_key": b"cusip+date",
        b"families": b"raw,corr",
        b"family_policy": b"A9_no_cross_family_mixing",
    })
    table = table.replace_schema_metadata(meta)
    tmp = OUT_FILE.with_suffix(".parquet.tmp")
    pq.write_table(table, str(tmp))
    os.replace(tmp, OUT_FILE)
    print(f"  Written: {OUT_FILE}")


def gap_diagnostic(signal: pd.DataFrame, window: int) -> dict:
    """Transparency for the POSITIONAL (row) window: how often per-cusip calendar gaps mean the
    trailing-`window`-row window spans more than `window` calendar months."""
    s = signal[["cusip", "date"]].copy()
    s["mo"] = s["date"].dt.year * 12 + s["date"].dt.month
    s = s.sort_values(["cusip", "mo"])
    max_gap = s.assign(gap=s.groupby("cusip")["mo"].diff()).groupby("cusip")["gap"].max()
    n = int(len(max_gap))
    return {
        "n_cusips": n,
        "frac_cusips_with_gap_gt_1mo": float((max_gap > 1).mean()) if n else 0.0,
        "frac_cusips_with_gap_ge_window": float((max_gap >= window).mean()) if n else 0.0,
        "max_gap_months": int(np.nanmax(max_gap.to_numpy())) if n else 0,
        "note": (
            "Window is POSITIONAL (trailing rows), matching build_var_5pct.py — a deliberate, "
            "consistent convention. For cusips with calendar gaps the 24-row window spans >24 "
            "calendar months and blends returns across the gap. Strictly trailing (no look-ahead). "
            "bond_vol is dual-role (instrument #7 + VOL-lane scaler), so this affects post-gap rows "
            "of both; acceptable for the interface-validation shakedown (non-comparable). "
            "A calendar-window variant is the refinement if KPP-comparability is ever sought."
        ),
    }


def write_report(signal: pd.DataFrame, cfg: dict) -> None:
    report = {
        "run_timestamp": datetime.now(timezone.utc).isoformat(),
        "thresholds_sha256": thresholds_sha256(),
        "thresholds_used": cfg,
        "ddof": DDOF,
        "input_panel": str(PANEL_FILE.relative_to(REPO_ROOT)),
        "calendar_gap_diagnostic": gap_diagnostic(signal, int(cfg["window"])),
        "counts": {
            "rows_total": int(len(signal)),
            "rows_non_null_raw": int(signal["bond_vol_raw"].notna().sum()),
            "rows_non_null_corr": int(signal["bond_vol_corr"].notna().sum()),
            "eligible_cusips_corr": int(
                signal.loc[signal["bond_vol_corr"].notna(), "cusip"].nunique()
            ),
        },
        "signal_methodology": (
            "KPP TOTAL_VOL (Table A.I #27): per-cusip rolling trailing-window sample std "
            "(ddof=1) of monthly EXCESS returns, PER FAMILY (bond_vol_raw from xret_raw; "
            "bond_vol_corr from xret_corr). Decimal monthly units. Dual role: IPCA instrument #7 "
            "AND the VOLScaled010 lane scaler. Cross-family mixing forbidden (A9)."
        ),
    }
    tmp = REPORT_OUT.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(report, f, indent=2)
    os.replace(tmp, REPORT_OUT)
    print(f"  Report: {REPORT_OUT}")


def main() -> None:
    if not PANEL_FILE.exists():
        print(f"ERROR: Required input not found: {PANEL_FILE}", file=sys.stderr)
        sys.exit(1)
    cfg = load_config()
    print(f"Config: window={cfg['window']}, min_obs={cfg['min_obs']}, ddof={DDOF}")
    print(f"Loading panel: {PANEL_FILE}")
    panel = pd.read_parquet(PANEL_FILE, columns=["cusip", "date", "xret_raw", "xret_corr"])
    print(f"  {len(panel):,} rows, {panel['cusip'].nunique():,} unique cusips")
    print("Computing bond_vol on each family...")
    signal = compute_dual_family(panel, window=int(cfg["window"]), min_obs=int(cfg["min_obs"]))
    raw_nn = int(signal["bond_vol_raw"].notna().sum())
    corr_nn = int(signal["bond_vol_corr"].notna().sum())
    print(f"  {raw_nn:,} raw non-NaN, {corr_nn:,} corr non-NaN")
    write_signal(signal)
    write_report(signal, cfg)
    print(f"\nDone. {len(signal):,} cusip-month rows → {OUT_FILE}")


if __name__ == "__main__":
    main()
