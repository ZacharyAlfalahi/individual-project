"""
Build the BBW 2019 5% VaR signal as a separate signals parquet — per A9,
family-indexed: emits `var_5pct_raw` (from `ret_raw`) and `var_5pct_corr`
(from `ret_corr`) as parallel columns. A run with meas_err=OFF must sort
on `var_5pct_raw`; meas_err=ON sorts on `var_5pct_corr`. Mixing families
within one configuration is a chimera that exists at no lattice point and
is forbidden by the Phase 2 view layer.

For each (cusip, date) row of the maximal monthly panel, compute the
rolling trailing-`window` 2nd-lowest non-NaN monthly return × `multiplier`
PER FAMILY. Only emitted (non-NaN) when the trailing window has at least
`min_obs` non-NaN returns in that family.

Per the BBW 2019 definition:
  window     = 36 trailing months
  min_obs    = 24 (eligibility gate)
  rank       = 2 (5% × 36 = 1.8 → 2nd-lowest by their rounding)
  multiplier = -1.0 (higher var_5pct = more downside risk)

All four constants come from docs/thresholds.yaml under signals.var_5pct;
nothing is hard-coded here.

Output: data/development/signals/var_5pct.parquet with columns
  cusip, date, var_5pct_raw, var_5pct_corr.
Engine-mergeable on (cusip, date) at run time.

Usage:
  python scripts/build_var_5pct.py

Requires: data/development/monthly_panel_maximal.parquet (build_monthly_panel.py)
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
OUT_FILE = OUT_DIR / "var_5pct.parquet"
REPORT_OUT = OUT_DIR / "var_5pct_report.json"
THRESHOLDS_FILE = REPO_ROOT / "docs" / "thresholds.yaml"


def load_config() -> dict:
    """Load var_5pct constants from thresholds.yaml. No defaults — every
    field must be present or the script refuses to start."""
    with open(THRESHOLDS_FILE) as f:
        cfg = yaml.safe_load(f)
    block = cfg.get("signals", {}).get("var_5pct")
    if block is None:
        raise KeyError(
            "thresholds.yaml is missing signals.var_5pct block; "
            "cannot determine window / min_obs / rank / multiplier"
        )
    for key in ("window", "min_obs", "rank", "multiplier"):
        if key not in block:
            raise KeyError(f"thresholds.yaml signals.var_5pct missing '{key}'")
    return block


def thresholds_sha256() -> str:
    return hashlib.sha256(THRESHOLDS_FILE.read_bytes()).hexdigest()


def compute_var_5pct(
    panel: pd.DataFrame,
    window: int,
    min_obs: int,
    rank: int,
    multiplier: float,
) -> pd.DataFrame:
    """
    Per-cusip rolling-window n-th lowest return × multiplier.

    Input panel must contain columns `cusip`, `date`, `ret`. Output is a
    DataFrame with columns `cusip`, `date`, `var_5pct`. Caller is
    responsible for selecting which family's `ret` column to pass and for
    renaming the output column post-hoc.

    Rows where the trailing window has < min_obs non-NaN returns get NaN.
    """
    if not all(c in panel.columns for c in ("cusip", "date", "ret")):
        raise ValueError("panel must contain columns: cusip, date, ret")
    if not isinstance(window, int) or window < 1:
        raise ValueError("window must be a positive integer")
    if not isinstance(min_obs, int) or min_obs < 1 or min_obs > window:
        raise ValueError("min_obs must be a positive integer ≤ window")
    if not isinstance(rank, int) or rank < 1:
        raise ValueError("rank must be a positive integer")

    panel = panel[["cusip", "date", "ret"]].sort_values(
        ["cusip", "date"], kind="mergesort"
    ).reset_index(drop=True)

    rets = panel["ret"].to_numpy(dtype=float)
    cusips = panel["cusip"].to_numpy()

    var_out = np.full(len(panel), np.nan)

    cusip_change = np.empty(len(panel), dtype=bool)
    cusip_change[0] = True
    cusip_change[1:] = cusips[1:] != cusips[:-1]
    block_starts = np.flatnonzero(cusip_change)
    block_ends = np.append(block_starts[1:], len(panel))

    for s, e in zip(block_starts, block_ends):
        block_rets = rets[s:e]
        n = e - s
        for i in range(n):
            lo = max(0, i - window + 1)
            sl = block_rets[lo:i + 1]
            valid_mask = ~np.isnan(sl)
            n_valid = int(valid_mask.sum())
            if n_valid >= min_obs and n_valid >= rank:
                valid = sl[valid_mask]
                kth = np.partition(valid, rank - 1)[rank - 1]
                var_out[s + i] = multiplier * float(kth)

    return pd.DataFrame({
        "cusip": panel["cusip"].values,
        "date":  panel["date"].values,
        "var_5pct": var_out,
    })


def compute_dual_family(
    panel: pd.DataFrame,
    window: int,
    min_obs: int,
    rank: int,
    multiplier: float,
) -> pd.DataFrame:
    """
    Compute var_5pct on each family and return a single DataFrame with
    columns `cusip`, `date`, `var_5pct_raw`, `var_5pct_corr` per A9.
    """
    # Raw family
    raw_in = panel[["cusip", "date", "ret_raw"]].rename(columns={"ret_raw": "ret"})
    raw_out = compute_var_5pct(raw_in, window, min_obs, rank, multiplier)
    raw_out = raw_out.rename(columns={"var_5pct": "var_5pct_raw"})

    # Corr family
    corr_in = panel[["cusip", "date", "ret_corr"]].rename(columns={"ret_corr": "ret"})
    corr_out = compute_var_5pct(corr_in, window, min_obs, rank, multiplier)
    corr_out = corr_out.rename(columns={"var_5pct": "var_5pct_corr"})

    merged = raw_out.merge(corr_out, on=["cusip", "date"], how="outer")
    return merged


def write_signal(signal: pd.DataFrame) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pandas(signal, preserve_index=False)
    existing_meta = dict(table.schema.metadata or {})
    existing_meta.update({
        b"signal_name":   b"var_5pct",
        b"signal_source": b"BBW_2019",
        b"primary_key":   b"cusip+date",
        b"families":      b"raw,corr",
        b"family_policy": b"A9_no_cross_family_mixing",
    })
    table = table.replace_schema_metadata(existing_meta)
    tmp = OUT_FILE.with_suffix(".parquet.tmp")
    pq.write_table(table, str(tmp))
    os.replace(tmp, OUT_FILE)
    print(f"  Written: {OUT_FILE}")


def write_report(signal: pd.DataFrame, cfg: dict) -> None:
    raw_non_null = int(signal["var_5pct_raw"].notna().sum())
    corr_non_null = int(signal["var_5pct_corr"].notna().sum())
    raw_eligible_cusips = int(
        signal.loc[signal["var_5pct_raw"].notna(), "cusip"].nunique()
    )
    corr_eligible_cusips = int(
        signal.loc[signal["var_5pct_corr"].notna(), "cusip"].nunique()
    )
    report = {
        "run_timestamp": datetime.now(timezone.utc).isoformat(),
        "thresholds_sha256": thresholds_sha256(),
        "thresholds_used": cfg,
        "input_panel": str(PANEL_FILE.relative_to(REPO_ROOT)),
        "counts": {
            "rows_total": int(len(signal)),
            "rows_non_null_raw": raw_non_null,
            "rows_non_null_corr": corr_non_null,
            "eligible_cusips_raw": raw_eligible_cusips,
            "eligible_cusips_corr": corr_eligible_cusips,
        },
        "signal_methodology": (
            "BBW (2019) 5% VaR proxy: per-cusip rolling trailing-window "
            "n-th lowest non-NaN monthly return × multiplier. PER FAMILY. "
            "var_5pct_raw built from ret_raw; var_5pct_corr from ret_corr. "
            "Cross-family mixing is forbidden by A9 — the Phase 2 view layer "
            "enforces this at run time."
        ),
        "registry_role": (
            "Pre-FISD signal-prep artifact. After FISD lands, ILLIQ and other "
            "return-derived signals will follow the same dual-family pattern."
        ),
    }
    tmp = REPORT_OUT.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(report, f, indent=2)
    os.replace(tmp, REPORT_OUT)
    print(f"  Report: {REPORT_OUT}")


def main():
    if not PANEL_FILE.exists():
        print(f"ERROR: Required input not found: {PANEL_FILE}", file=sys.stderr)
        sys.exit(1)

    cfg = load_config()
    print(
        f"Config: window={cfg['window']}, min_obs={cfg['min_obs']}, "
        f"rank={cfg['rank']}, multiplier={cfg['multiplier']}"
    )

    print(f"Loading panel: {PANEL_FILE}")
    panel = pd.read_parquet(PANEL_FILE, columns=["cusip", "date", "ret_raw", "ret_corr"])
    print(f"  {len(panel):,} rows, {panel['cusip'].nunique():,} unique cusips")

    print("Computing var_5pct on each family...")
    signal = compute_dual_family(
        panel,
        window=int(cfg["window"]),
        min_obs=int(cfg["min_obs"]),
        rank=int(cfg["rank"]),
        multiplier=float(cfg["multiplier"]),
    )
    raw_nn = int(signal["var_5pct_raw"].notna().sum())
    corr_nn = int(signal["var_5pct_corr"].notna().sum())
    print(f"  {raw_nn:,} raw non-NaN ({raw_nn / len(signal):.1%}), "
          f"{corr_nn:,} corr non-NaN ({corr_nn / len(signal):.1%})")

    write_signal(signal)
    write_report(signal, cfg)

    print("\nDone.")
    print(f"  {len(signal):,} cusip-month rows")
    print(f"  raw non-NaN: {raw_nn:,}, corr non-NaN: {corr_nn:,}")
    print(f"  → {OUT_FILE}")


if __name__ == "__main__":
    main()
