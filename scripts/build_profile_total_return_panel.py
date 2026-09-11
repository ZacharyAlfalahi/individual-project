"""
Profile TOTAL-RETURN baselines — the total-return analogue of the meas_err OFF-arm
profile columns (companion to build_total_return_panel.py).

`build_profile_monthly_panel.py` emits the CLEAN-price per-paper baseline families
(`ret_<pid>` = price growth) for pid in {bbw_2019, jostova_2013}. The RQ3 audit's
as-published corner for drf/mom6 selects those families, so a total-return audit of
drf/mom6 needs their TOTAL-return analogue. This script re-emits
`monthly_panel_profiles.parquet` with each profile's `ret_<pid>` / `xret_<pid>`
replaced by the total return

    R = [(P_t + AI_t + C_t) - (P_{t-1} + AI_{t-1})] / (P_{t-1} + AI_{t-1})

using the SAME accrual engine (agents/quant/library/accrual.py) and the SAME
per-family total-return kernel (`_total_return_one_family`) as the maximal
total-return panel — so a profile family and a raw/corr family differ ONLY in the
clean price P they carry, never in the total-return construction. AI and coupon are
family-agnostic (a function of date + FISD schedule only), so they are computed once.

Everything that is NOT a return is carried through byte-for-byte from the clean
profiles panel (`price_eom_<pid>`, `last_trade_date_<pid>`, `n_trades_<pid>`,
`total_vol_<pid>`), so `views.view(..., price_family='bbw_2019')` runs on this panel
unmodified — only the compounded return changes.

Guards (fail-loud, the accrual control set):
  * Zero-coupon invariance: for coupon==0 bonds the total return equals the COMMITTED
    clean `ret_<pid>` byte-for-byte (atol 1e-12) — the accrual path must not touch them.
  * Finite-clean ⇒ finite-total: a bond-month with a finite clean return must keep a
    finite total return (denom = P_{t-1}+AI_{t-1} > 0 whenever the clean divisor P_{t-1}
    was > 0), so the total-return panel never drops an economically-valid observation.
  * rf coverage: every row must match a risk-free month (xret is exact).

ADDITIVE and dev-only: does NOT touch monthly_panel_profiles.parquet,
monthly_panel_maximal.parquet, or monthly_panel_total_return.parquet. Holdout never read.

Usage: python scripts/build_profile_total_return_panel.py
"""

from __future__ import annotations

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

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from agents.quant.library.accrual import accrued_and_coupon, THIRTY_360  # noqa: E402
from build_total_return_panel import _total_return_one_family  # noqa: E402  reuse exact TR kernel

DEV = REPO_ROOT / "data" / "development"
PROFILE_IN = DEV / "monthly_panel_profiles.parquet"
FISD_STATIC = DEV / "fisd" / "fisd_reference_static.parquet"
RF_FILE = DEV / "rf_rate.parquet"
OUT_FILE = DEV / "monthly_panel_profiles_total_return.parquet"
REPORT_OUT = DEV / "monthly_panel_profiles_total_return_report.json"

PROFILES = ("bbw_2019", "jostova_2013")


def git_commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT,
                              capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        return "unknown"


def build_total_return_profiles(
    profiles: pd.DataFrame, fisd: pd.DataFrame, rf: pd.DataFrame,
) -> tuple[pd.DataFrame, dict]:
    """Pure transform: clean-price profiles -> total-return profiles. Returns
    (out_panel, diagnostics). Fails loud on any guard breach. `out_panel` carries the
    IDENTICAL columns/order as `profiles`, with ret_<pid>/xret_<pid> replaced by totals."""
    original_cols = list(profiles.columns)
    for pid in PROFILES:
        for base in ("price_eom", "ret", "xret"):
            col = f"{base}_{pid}"
            if col not in profiles.columns:
                raise KeyError(f"clean profiles panel missing required column {col!r}")

    panel = profiles.merge(
        fisd[["cusip", "coupon", "interest_frequency", "day_count_basis", "maturity"]]
        .drop_duplicates("cusip"),
        on="cusip", how="left",
    )

    # Risk-free by month (exact xret), matching the clean profiles' construction.
    panel["year_month"] = panel["date"].dt.strftime("%Y-%m")
    panel = panel.merge(rf, on="year_month", how="left", validate="m:1")

    # Sort AFTER every merge so the (cusip, date) ordering the total-return kernel
    # relies on is established last — the kernel derives P_{t-1}/AI_{t-1} from the row
    # above, so it must not depend on any merge preserving order.
    panel = panel.sort_values(["cusip", "date"], kind="mergesort").reset_index(drop=True)

    # Accrued interest + coupon: family-agnostic, computed once (date + schedule only).
    ai, coupon_paid = accrued_and_coupon(
        panel["date"].values, panel["maturity"].values,
        panel["coupon"].values, panel["interest_frequency"].values,
    )
    panel["ai"] = ai
    panel["coupon_paid"] = coupon_paid
    panel["day_count_fallback"] = (panel["day_count_basis"].fillna("NA") != THIRTY_360)
    n_rf_missing = int(panel["rf_monthly"].isna().sum())
    if n_rf_missing:
        bad_months = sorted(panel.loc[panel["rf_monthly"].isna(), "year_month"].unique())[:8]
        raise AssertionError(
            f"rf coverage gap: {n_rf_missing} rows have no risk-free month "
            f"(e.g. {bad_months}); xret would be undefined")

    zero = (panel["coupon"].fillna(0.0) == 0.0).to_numpy()
    rf_arr = panel["rf_monthly"].to_numpy(dtype=float)
    diagnostics: dict = {}

    for pid in PROFILES:
        clean = panel[f"ret_{pid}"].to_numpy(dtype=float).copy()
        clean_xret = panel[f"xret_{pid}"].to_numpy(dtype=float).copy()
        tot = _total_return_one_family(panel, f"price_eom_{pid}")
        tot_xret = tot - rf_arr

        # Guard 1 — zero-coupon invariance against the COMMITTED clean return AND xret.
        # The xret arm makes the guarantee explicit: it also fails loud if the risk-free
        # series merged here has drifted from the one the clean profiles were built with.
        both = zero & np.isfinite(clean) & np.isfinite(tot)
        z_ret_ok = bool(np.allclose(clean[both], tot[both], atol=1e-12, rtol=0))
        both_x = zero & np.isfinite(clean_xret) & np.isfinite(tot_xret)
        z_xret_ok = bool(np.allclose(clean_xret[both_x], tot_xret[both_x], atol=1e-12, rtol=0))
        if not (z_ret_ok and z_xret_ok):
            bad = both & ~np.isclose(clean, tot, atol=1e-12, rtol=0)
            raise AssertionError(
                f"[{pid}] zero-coupon invariance FAILED "
                f"(ret_ok={z_ret_ok}, xret_ok={z_xret_ok}): {int(bad.sum())} Z-bond-months "
                f"changed under accrual — the accrual path is touching coupon==0 bonds, "
                f"or the risk-free series has drifted from the clean profiles build.")

        # Guard 2 — a finite clean return must remain finite under total return.
        lost = np.isfinite(clean) & ~np.isfinite(tot)
        if int(lost.sum()):
            raise AssertionError(
                f"[{pid}] {int(lost.sum())} bond-months with a finite clean return lost "
                f"their total return (denom P_(t-1)+AI_(t-1) not > 0) — investigate.")

        panel[f"ret_{pid}"] = tot
        panel[f"xret_{pid}"] = tot_xret

        diagnostics[pid] = {
            "z_coupon_invariant": z_ret_ok and z_xret_ok,
            "z_bond_months_checked": int(both.sum()),
            "clean_ret_nonnull": int(np.isfinite(clean).sum()),
            "total_ret_nonnull": int(np.isfinite(tot).sum()),
        }

    out = panel[original_cols].copy()
    diagnostics["rows"] = int(len(out))
    diagnostics["day_count_fallback_bond_months"] = int(panel["day_count_fallback"].sum())
    diagnostics["ai_max"] = float(np.nanmax(ai)) if len(ai) else 0.0
    return out, diagnostics


def main() -> None:
    for f in (PROFILE_IN, FISD_STATIC, RF_FILE):
        if not f.exists():
            print(f"ERROR: required input not found: {f}", file=sys.stderr)
            sys.exit(1)

    print("Loading clean profiles + FISD schedule + risk-free...")
    profiles = pd.read_parquet(PROFILE_IN)
    fisd = pd.read_parquet(
        FISD_STATIC,
        columns=["cusip", "coupon", "interest_frequency", "day_count_basis", "maturity"],
    )
    rf = pd.read_parquet(RF_FILE)

    out, diag = build_total_return_profiles(profiles, fisd, rf)

    if list(out.columns) != list(profiles.columns):
        raise AssertionError("output schema drifted from the clean profiles panel")

    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pandas(out, preserve_index=False)
    meta = dict(table.schema.metadata or {})
    meta.update({
        b"return_basis": b"total_return_price_plus_AI_plus_coupon_30E360",
        b"derived_from": b"monthly_panel_profiles.parquet (clean) + accrual engine",
        b"companion_of": b"monthly_panel_total_return.parquet",
    })
    table = table.replace_schema_metadata(meta)
    tmp = OUT_FILE.with_suffix(".parquet.tmp")
    pq.write_table(table, str(tmp))
    written = pq.read_metadata(str(tmp)).num_rows
    if written != len(out):
        os.remove(tmp)
        raise AssertionError(f"row count {written} != {len(out)}")
    os.replace(tmp, OUT_FILE)

    report = {
        "run_timestamp": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit(),
        "stage": "profile_total_return_baselines",
        "return_basis": "total return = (P + AI + C) growth; 30E/360 accrual (matches maximal TR)",
        "additive_to": "monthly_panel_profiles.parquet + monthly_panel_total_return.parquet (NEITHER modified)",
        **diag,
    }
    tmp_r = REPORT_OUT.with_suffix(".tmp")
    tmp_r.write_text(json.dumps(report, indent=2, default=str))
    os.replace(tmp_r, REPORT_OUT)

    print(f"\nDone. {OUT_FILE.name}: {diag['rows']:,} rows")
    for pid in PROFILES:
        d = diag[pid]
        print(f"  {pid}: z-invariant={d['z_coupon_invariant']} "
              f"(Z checked {d['z_bond_months_checked']:,})  "
              f"total_ret_nonnull={d['total_ret_nonnull']:,} "
              f"(clean {d['clean_ret_nonnull']:,})")
    print(f"  day_count_fallback: {diag['day_count_fallback_bond_months']:,} bond-months")
    print(f"  Report: {REPORT_OUT.name}")


if __name__ == "__main__":
    main()
