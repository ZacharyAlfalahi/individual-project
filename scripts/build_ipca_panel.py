"""
Build the IPCA Workstream B characteristic-panel feed (the buildable FISD+TRACE 7-instrument
subset) for the ipca module. Spec: docs/characteristic_registry_spec.md. Instruments + family
policy: agents/quant/library/configs/ipca_instruments.yaml.

Pipeline (family-parameterised, corr default per D6):
  1. Merge the panel + signal parquets on (cusip, date).
  2. Next-return alignment: instruments at month m-1 are paired with the return at month m
     (adjacent months only). Sets `month` = return-month period ordinal and `asof` = month - 1
     so ipca.validate_panel's double-lag guard passes. str_reversal is the prior-month return.
  3. VOL-scaling (VOLScaled010): R = xret_m / max(bond_vol_{m-1}, floor); bond_vol<=0 excluded.
  4. Complete-case: drop any (cusip, return-month) missing any of the 7 instruments or R.
  5. Per return-month: ipca.rank_transform each instrument (cross-sectional); drop months with
     N_m <= L=8. (The constant column is appended at load time by the shakedown.)
  6. Emit data/development/ipca_panel_<family>.parquet (long) + an audit JSON, wall-enforced.

This is INTERFACE-VALIDATION data — NON-COMPARABLE to KPP (7-instrument bond-only, VOL-scaled,
no equity, no DtS). See docs/characteristic_registry_spec.md §7.

Usage:  python scripts/build_ipca_panel.py [--family corr|raw]
Requires: monthly_panel_maximal.parquet; signals/{mom6,var_5pct,gamma_illiq,bond_vol}.parquet
"""

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
from agents.quant.library.ipca import assert_within_wall, rank_transform, validate_panel  # noqa: E402

DEV = REPO_ROOT / "data" / "development"
PANEL_FILE = DEV / "monthly_panel_maximal.parquet"
REGISTRY = REPO_ROOT / "agents" / "quant" / "library" / "configs" / "ipca_instruments.yaml"
THRESHOLDS_FILE = REPO_ROOT / "docs" / "thresholds.yaml"

# Instrument order is fixed (constant is appended LAST at load time, so L = len + 1 = 8).
INSTRUMENTS = ["str_reversal", "mom6", "var_5pct", "gamma_illiq", "rating", "time_to_maturity", "bond_vol"]
L = len(INSTRUMENTS) + 1


def load_registry() -> dict:
    with open(REGISTRY) as f:
        return yaml.safe_load(f)


def thresholds_sha256() -> str:
    return hashlib.sha256(THRESHOLDS_FILE.read_bytes()).hexdigest()


def assemble(family: str, reg: dict) -> tuple[pd.DataFrame, dict]:
    """Read + merge the panel and signal parquets, then align/scale/complete-case (pure step)."""
    xret = reg["return_column"][family]
    vol_col = reg["scaler"]["column"][family]

    panel = pd.read_parquet(PANEL_FILE, columns=["cusip", "date", xret, "rating", "time_to_maturity"])
    mom6 = pd.read_parquet(DEV / "signals" / "mom6.parquet", columns=["cusip", "date", f"mom6_{family}"])
    var5 = pd.read_parquet(DEV / "signals" / "var_5pct.parquet", columns=["cusip", "date", f"var_5pct_{family}"])
    gam = pd.read_parquet(DEV / "signals" / "gamma_illiq.parquet", columns=["cusip", "date", f"gamma_{family}"])
    vol = pd.read_parquet(DEV / "signals" / "bond_vol.parquet", columns=["cusip", "date", vol_col])

    df = (
        panel.merge(mom6, on=["cusip", "date"], how="left")
        .merge(var5, on=["cusip", "date"], how="left")
        .merge(gam, on=["cusip", "date"], how="left")
        .merge(vol, on=["cusip", "date"], how="left")
        .rename(columns={
            xret: "str_reversal", f"mom6_{family}": "mom6", f"var_5pct_{family}": "var_5pct",
            f"gamma_{family}": "gamma_illiq", vol_col: "bond_vol",
        })
    )
    return align_scale(df, reg)


def align_scale(df: pd.DataFrame, reg: dict) -> tuple[pd.DataFrame, dict]:
    """Pure transform: next-return alignment (instruments at m-1, return at m, adjacent only),
    VOL-scaling, and complete-case selection. `df` must carry cusip, date, the 7 instrument
    columns (str_reversal = instrument-time excess return), and bond_vol. Unit-testable."""
    floor = float(reg["scaler"]["floor"])
    counts = {"rows_merged": int(len(df))}

    df = df.sort_values(["cusip", "date"], kind="mergesort").reset_index(drop=True)
    df["period"] = df["date"].dt.to_period("M")
    g = df.groupby("cusip", sort=False)
    df["next_xret"] = g["str_reversal"].shift(-1)   # str_reversal IS xret at instrument-time t
    df["next_period"] = g["period"].shift(-1)
    df["next_date"] = g["date"].shift(-1)

    # next-return convention: keep only adjacent month pairs (return at t+1, instruments at t)
    adj = df["next_period"] == (df["period"] + 1)
    df = df[adj].copy()
    counts["rows_adjacent"] = int(len(df))

    # VOLScaled010: exclude bond_vol<=0, then scale next-month return by the instrument-time vol
    df = df[df["bond_vol"] > 0].copy()
    counts["rows_vol_positive"] = int(len(df))
    df["below_floor"] = df["bond_vol"] <= floor
    df["R"] = df["next_xret"] / np.maximum(df["bond_vol"].to_numpy(), floor)

    df["month"] = df["period"].apply(lambda p: p.ordinal + 1)   # return-month ordinal
    df["asof"] = df["period"].apply(lambda p: p.ordinal)        # instrument month = month - 1
    df["ret_date"] = df["next_date"]

    # complete-case on the 7 instruments + R
    df = df.dropna(subset=[*INSTRUMENTS, "R"]).reset_index(drop=True)
    counts["rows_complete_case"] = int(len(df))
    counts["below_floor_share"] = float(df["below_floor"].mean()) if len(df) else float("nan")
    return df, counts


def rank_and_emit(df: pd.DataFrame, counts: dict) -> pd.DataFrame:
    """Per return-month: rank_transform each instrument, drop months with N_m <= L, emit long frame."""
    blocks: list[pd.DataFrame] = []
    dropped_small: list[int] = []
    n_per_month: dict[int, int] = {}
    for month, grp in df.groupby("month", sort=True):
        if len(grp) <= L:
            dropped_small.append(int(month))
            continue
        block = pd.DataFrame({
            "cusip": grp["cusip"].to_numpy(),
            "month": grp["month"].to_numpy(),
            "asof": grp["asof"].to_numpy(),
            "ret_date": grp["ret_date"].to_numpy(),
            "R": grp["R"].to_numpy(dtype=float),
            "vol_scaler": grp["bond_vol"].to_numpy(dtype=float),   # raw scaler (for VOL-lane diagnostics)
        })
        for col in INSTRUMENTS:
            block[f"z_{col}"] = rank_transform(grp[col].to_numpy(dtype=float))
        blocks.append(block)
        n_per_month[int(month)] = int(len(grp))
    if not blocks:
        raise RuntimeError("no months survived the N_m > L filter — check inputs")
    out = pd.concat(blocks, ignore_index=True)
    counts["months_kept"] = len(n_per_month)
    counts["months_dropped_small"] = len(dropped_small)
    counts["N_m_min"] = int(min(n_per_month.values()))
    counts["N_m_median"] = int(np.median(list(n_per_month.values())))
    counts["N_m_max"] = int(max(n_per_month.values()))
    counts["rows_emitted"] = int(len(out))
    return out


def validate_emitted(out: pd.DataFrame, reg: dict, family: str) -> None:
    """Round-trip: the emitted Z must pass the module's own receipt check + wall.

    The wall train_end is the ordinal of the last in-window RETURN month (inclusive); `month` is
    the return-month period ordinal, so this rejects any return beyond reg.meta.train_end.
    """
    z_cols = [f"z_{c}" for c in INSTRUMENTS]
    Z: list[np.ndarray] = []
    R: list[np.ndarray] = []
    months: list[int] = []
    asof: list[int] = []
    for month, grp in out.groupby("month", sort=True):
        zmat = grp[z_cols].to_numpy(dtype=float)
        zmat = np.column_stack([zmat, np.ones(len(grp))])   # constant appended last
        Z.append(zmat)
        R.append(grp["R"].to_numpy(dtype=float))
        months.append(int(month))
        asof.append(int(grp["asof"].iloc[0]))
    validate_panel(Z, R, np.asarray(months), L=L, family=family,
                   characteristic_asof=np.asarray(asof))
    train_end = pd.Period(reg["meta"]["train_end"], "M").ordinal   # inclusive last return month
    assert_within_wall(np.asarray(months), train_end=train_end)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", choices=["corr", "raw"], default=None)
    args = ap.parse_args()
    reg = load_registry()
    family = args.family or reg["meta"]["default_family"]
    required = (
        PANEL_FILE,
        DEV / "signals" / "mom6.parquet",
        DEV / "signals" / "var_5pct.parquet",
        DEV / "signals" / "gamma_illiq.parquet",
        DEV / "signals" / "bond_vol.parquet",
    )
    for f in required:
        if not f.exists():
            print(f"ERROR: required input not found: {f}", file=sys.stderr)
            sys.exit(1)

    print(f"Assembling IPCA panel (family={family}, L={L})...")
    df, counts = assemble(family, reg)
    print(f"  merged={counts['rows_merged']:,}  adjacent={counts['rows_adjacent']:,}  "
          f"vol>0={counts['rows_vol_positive']:,}  complete-case={counts['rows_complete_case']:,}")
    out = rank_and_emit(df, counts)
    print(f"  months kept={counts['months_kept']} (dropped small={counts['months_dropped_small']}); "
          f"N_m min/median/max={counts['N_m_min']}/{counts['N_m_median']}/{counts['N_m_max']}; "
          f"rows={counts['rows_emitted']:,}")

    print("Validating emitted feed against ipca.validate_panel + wall...")
    validate_emitted(out, reg, family)
    print("  OK — feed passes the module's receipt check and the train_end wall.")

    out_file = DEV / f"ipca_panel_{family}.parquet"
    tmp = out_file.with_suffix(".parquet.tmp")
    out.to_parquet(tmp, index=False)
    os.replace(tmp, out_file)
    report = {
        "run_timestamp": datetime.now(timezone.utc).isoformat(),
        "thresholds_sha256": thresholds_sha256(),
        "family": family,
        "L": L,
        "instruments": INSTRUMENTS,
        "scaling_lane": reg["meta"]["scaling_lane"],
        "train_end": reg["meta"]["train_end"],
        "comparability_label": reg["meta"]["comparability"],
        "selection_counts": counts,
        "note": (
            "Interface-validation feed (NON-COMPARABLE to KPP). Complete-case is on the 7 buildable "
            "instruments — a LARGER, DIFFERENT universe than KPP's complete-on-29; when the equity "
            "side lands and the set grows the universe will SHRINK (expected, not data loss). "
            "See docs/characteristic_registry_spec.md §2."
        ),
    }
    report_file = DEV / f"ipca_panel_{family}_report.json"
    rtmp = report_file.with_suffix(".tmp")
    with open(rtmp, "w") as fh:
        json.dump(report, fh, indent=2)
    os.replace(rtmp, report_file)
    print(f"\nDone. → {out_file}\n       → {report_file}")


if __name__ == "__main__":
    main()
