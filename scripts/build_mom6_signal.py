"""
Build the mom6 (6-month momentum) formation signal — the trailing cumulative
return that the mom6_1 anchor sorts on (Jostova et al. 2013;
docs/quant/specs/BBW_anchor_implementation_spec.md §5.2).

Family-indexed per A9: emits `mom6_raw` (from `ret_raw`) and `mom6_corr` (from
`ret_corr`).

For each (cusip, month τ) the signal is the cumulative return over the
`formation_months` CONSECUTIVE months ending at τ:
    mom6[τ] = prod_{j=τ-formation_months+1..τ} (1 + ret[j]) − 1
computed as expm1(sum of log1p(ret)). It is NaN unless the window holds
`min_obs` non-NaN monthly returns AND those months are calendar-contiguous —
a missing month breaks the compounding chain, so we segment each cusip into
runs of consecutive months and never roll a window across a gap.

The Jostova SKIP and the staggered HOLDING are NOT applied here — they are the
factor runner's job (build_mom6.py): signal_lag=skip_months picks the window
ending one month before formation, and overlap holds for holding_months. So
this builder produces only the trailing-window signal; the engine + overlap
turn it into the formation=6 / skip=1 / hold=6 factor.

All constants come from docs/thresholds.yaml under signals.mom6.

Output: data/development/signals/mom6.parquet with columns
  cusip, date, mom6_raw, mom6_corr.

Usage:
  python scripts/build_mom6_signal.py

Requires: data/development/monthly_panel_maximal.parquet (build_monthly_panel.py)
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
PANEL_FILE = REPO_ROOT / "data" / "development" / "monthly_panel_maximal.parquet"
OUT_DIR = REPO_ROOT / "data" / "development" / "signals"
OUT_FILE = OUT_DIR / "mom6.parquet"
REPORT_OUT = OUT_DIR / "mom6_report.json"
THRESHOLDS_FILE = REPO_ROOT / "docs" / "thresholds.yaml"


def load_config() -> dict:
    """Load signals.mom6 from thresholds.yaml. No defaults — every signal field
    must be present."""
    with open(THRESHOLDS_FILE) as f:
        cfg = yaml.safe_load(f)
    block = cfg.get("signals", {}).get("mom6")
    if block is None:
        raise KeyError("thresholds.yaml is missing signals.mom6 block")
    for key in ("formation_months", "min_obs"):
        if key not in block:
            raise KeyError(f"thresholds.yaml signals.mom6 missing '{key}'")
    return block


def thresholds_sha256() -> str:
    return hashlib.sha256(THRESHOLDS_FILE.read_bytes()).hexdigest()


def git_commit() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT, capture_output=True, text=True, check=True,
        ).stdout.strip()
    except Exception:
        return "unknown"


def compute_mom6_signal(
    panel: pd.DataFrame, formation_months: int, min_obs: int
) -> pd.DataFrame:
    """Per-cusip trailing cumulative return over `formation_months` consecutive
    months. Input columns: cusip, date, ret. Output: cusip, date, mom6.

    Calendar-contiguity is enforced by segmenting each cusip into runs of
    consecutive month-periods: a gap (or a cusip change) starts a new run, and
    the rolling window never spans two runs. Within a run, a window of
    `formation_months` rows is exactly that many consecutive months; `min_obs`
    is the minimum non-NaN returns required (NaN result otherwise).
    """
    if not all(c in panel.columns for c in ("cusip", "date", "ret")):
        raise ValueError("panel must contain columns: cusip, date, ret")
    if not isinstance(formation_months, int) or formation_months < 1:
        raise ValueError("formation_months must be a positive integer")
    if not isinstance(min_obs, int) or min_obs < 1 or min_obs > formation_months:
        raise ValueError("min_obs must be a positive integer ≤ formation_months")

    df = panel[["cusip", "date", "ret"]].sort_values(
        ["cusip", "date"], kind="mergesort"
    ).reset_index(drop=True)

    period = df["date"].dt.to_period("M").astype("int64")  # month index, contiguous = diff 1
    cusip = df["cusip"].to_numpy()
    same_cusip = np.empty(len(df), dtype=bool)
    same_cusip[0] = False
    same_cusip[1:] = cusip[1:] == cusip[:-1]
    consecutive = np.empty(len(df), dtype=bool)
    consecutive[0] = False
    consecutive[1:] = (period.to_numpy()[1:] - period.to_numpy()[:-1]) == 1
    # A new run starts whenever the previous row is not the immediately-prior
    # month of the same cusip.
    new_run = ~(same_cusip & consecutive)
    run_id = np.cumsum(new_run)

    df["_run"] = run_id
    ret = df["ret"].to_numpy(dtype=float)
    # A monthly return <= -1 implies a non-positive price — impossible for a
    # long position with positive prices, so it only appears as a raw-family
    # data error (a handful of negative raw prices). Treat it as invalid (NaN)
    # so it cannot enter the compounding chain; strict min_obs then NaNs any
    # window that needed it. The corr family is clean (min ret > -1).
    with np.errstate(invalid="ignore"):
        df["_lr"] = np.log1p(np.where(ret > -1.0, ret, np.nan))
    rsum = (
        df.groupby("_run")["_lr"]
        .rolling(window=formation_months, min_periods=min_obs)
        .sum()
        .reset_index(level=0, drop=True)
    )
    df["mom6"] = np.expm1(rsum)

    return df[["cusip", "date", "mom6"]]


def compute_dual_family(panel: pd.DataFrame, formation_months: int, min_obs: int) -> pd.DataFrame:
    raw_in = panel[["cusip", "date", "ret_raw"]].rename(columns={"ret_raw": "ret"})
    raw_out = compute_mom6_signal(raw_in, formation_months, min_obs).rename(
        columns={"mom6": "mom6_raw"}
    )
    corr_in = panel[["cusip", "date", "ret_corr"]].rename(columns={"ret_corr": "ret"})
    corr_out = compute_mom6_signal(corr_in, formation_months, min_obs).rename(
        columns={"mom6": "mom6_corr"}
    )
    return raw_out.merge(corr_out, on=["cusip", "date"], how="outer")


def write_signal(signal: pd.DataFrame) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pandas(signal, preserve_index=False)
    meta = dict(table.schema.metadata or {})
    meta.update({
        b"signal_name":   b"mom6",
        b"signal_source": b"Jostova_2013",
        b"primary_key":   b"cusip+date",
        b"families":      b"raw,corr",
        b"definition":    b"trailing_cumulative_return_contiguous_window",
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
        "input_panel": str(PANEL_FILE.relative_to(REPO_ROOT)),
        "counts": {
            "rows_total": int(len(signal)),
            "rows_non_null_raw": int(signal["mom6_raw"].notna().sum()),
            "rows_non_null_corr": int(signal["mom6_corr"].notna().sum()),
            "eligible_cusips_corr": int(
                signal.loc[signal["mom6_corr"].notna(), "cusip"].nunique()
            ),
        },
        "signal_methodology": (
            "Trailing cumulative return over `formation_months` calendar-"
            "contiguous months, expm1(sum log1p(ret)). Skip + staggered holding "
            "are applied by the factor runner (build_mom6.py), not here."
        ),
    }
    tmp = REPORT_OUT.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(report, f, indent=2)
    os.replace(tmp, REPORT_OUT)
    print(f"  Report: {REPORT_OUT}")


def main():
    if not PANEL_FILE.exists():
        print(f"ERROR: required input not found: {PANEL_FILE}", file=sys.stderr)
        sys.exit(1)

    cfg = load_config()
    fm, mo = int(cfg["formation_months"]), int(cfg["min_obs"])
    print(f"Config: formation_months={fm}, min_obs={mo}")

    print(f"Loading panel: {PANEL_FILE}")
    panel = pd.read_parquet(PANEL_FILE, columns=["cusip", "date", "ret_raw", "ret_corr"])
    print(f"  {len(panel):,} rows, {panel['cusip'].nunique():,} cusips")

    print("Computing mom6 trailing cumulative return per family...")
    signal = compute_dual_family(panel, fm, mo)
    raw_nn = int(signal["mom6_raw"].notna().sum())
    corr_nn = int(signal["mom6_corr"].notna().sum())
    print(f"  {raw_nn:,} raw non-NaN ({raw_nn/len(signal):.1%}), "
          f"{corr_nn:,} corr non-NaN ({corr_nn/len(signal):.1%})")

    write_signal(signal)
    write_report(signal, cfg)

    print("\nDone.")
    print(f"  {len(signal):,} cusip-month rows  →  {OUT_FILE}")


if __name__ == "__main__":
    main()
