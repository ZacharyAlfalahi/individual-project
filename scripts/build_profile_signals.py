"""
Per-profile signal variants (FL-D21a) — the meas_err OFF-arm signals.

The auditor's `load_dev_signals` merges four family-indexed signals into ONE
frame that `view()`'s A9 resolver checks WHOLE — so a profile-family request
(drf→bbw_2019, mom6→jostova_2013) needs EVERY merged signal to carry that
family, not only the one the anchor sorts on. This builds all four signals for
both baseline profiles, reusing each production builder's own `compute_*` core:

  var_5pct_<pid>   from monthly_panel_profiles ret_<pid>   (build_var_5pct)
  bond_vol_<pid>   from monthly_panel_profiles xret_<pid>  (build_bond_vol)
  mom6_<pid>       from monthly_panel_profiles ret_<pid>   (build_mom6_signal)
  gamma_illiq_<pid> from trace_daily_<pid>__dedup_on price_vwap (build_gamma_illiq)

<pid> in {bbw_2019, jostova_2013}; each family = that profile's DEDUP-ON baseline
(the lattice OFF-arm reference). Output: a SEPARATE
`data/development/signals/profiles_signals.parquet` (cusip, date, and the 8
`<signal>_<pid>` columns). ADDITIVE — the committed per-signal parquets are NOT
touched; `load_dev_signals` outer-merges this file so the profile columns join
the dual-family frame the resolver already understands.

dev only (holdout never read). Usage: python scripts/build_profile_signals.py
"""

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import build_bond_vol as bv          # noqa: E402
import build_gamma_illiq as gi       # noqa: E402
import build_mom6_signal as m6       # noqa: E402
import build_var_5pct as v5          # noqa: E402

from shared.licensed_inputs import require_licensed_input  # noqa: E402

DEV = REPO_ROOT / "data" / "development"
PROFILE_MONTHLY = DEV / "monthly_panel_profiles.parquet"
OUT = DEV / "signals" / "profiles_signals.parquet"
REPORT = DEV / "signals" / "profiles_signals_report.json"

PROFILE_IDS = ("bbw_2019", "jostova_2013")


def _profile_daily(pid: str) -> Path:
    return DEV / f"trace_daily_{pid}__dedup_on.parquet"


def build() -> dict:
    monthly = pd.read_parquet(require_licensed_input(PROFILE_MONTHLY, "profile monthly panel"))
    v5cfg, bvcfg, m6cfg, gicfg = (v5.load_config(), bv.load_config(),
                                  m6.load_config(), gi.load_config())

    merged: pd.DataFrame | None = None
    counts: dict = {}
    for pid in PROFILE_IDS:
        ret_col, xret_col = f"ret_{pid}", f"xret_{pid}"
        base = monthly[["cusip", "date", ret_col, xret_col]]

        # var_5pct — from monthly ret
        v = v5.compute_var_5pct(
            base[["cusip", "date", ret_col]].rename(columns={ret_col: "ret"}),
            int(v5cfg["window"]), int(v5cfg["min_obs"]),
            int(v5cfg["rank"]), float(v5cfg["multiplier"]),
        ).rename(columns={"var_5pct": f"var_5pct_{pid}"})

        # bond_vol — from monthly xret (the dual builder feeds xret as `ret`)
        bvol = bv.compute_bond_vol(
            base[["cusip", "date", xret_col]].rename(columns={xret_col: "ret"}),
            int(bvcfg["window"]), int(bvcfg["min_obs"]),
        ).rename(columns={"bond_vol": f"bond_vol_{pid}"})

        # mom6 — from monthly ret
        mm = m6.compute_mom6_signal(
            base[["cusip", "date", ret_col]].rename(columns={ret_col: "ret"}),
            int(m6cfg["formation_months"]), int(m6cfg["min_obs"]),
        ).rename(columns={"mom6": f"mom6_{pid}"})

        # gamma_illiq — from the profile DAILY panel price_vwap
        daily = pd.read_parquet(require_licensed_input(_profile_daily(pid), "profile daily panel"),
                                columns=["cusip_id", "trd_exctn_dt", "price_vwap"])
        g = gi.compute_gamma(
            daily,
            min_pairs=int(gicfg["min_pairs"]), max_gap_bdays=int(gicfg["max_gap_bdays"]),
            sign_multiplier=float(gicfg["sign_multiplier"]), cov_ddof=int(gicfg["cov_ddof"]),
            strict=gicfg.get("bpw_strict"),
        ).rename(columns={"gamma": f"gamma_illiq_{pid}"})

        pf = v
        for other in (bvol, mm, g):
            pf = pf.merge(other, on=["cusip", "date"], how="outer")
        counts[pid] = {c: int(pf[c].notna().sum()) for c in pf.columns
                       if c not in ("cusip", "date")}
        merged = pf if merged is None else merged.merge(pf, on=["cusip", "date"], how="outer")

    out = merged.sort_values(["cusip", "date"]).reset_index(drop=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    tmp = OUT.with_suffix(".parquet.tmp")
    out.to_parquet(tmp, index=False)
    assert pq.read_metadata(str(tmp)).num_rows == len(out)
    os.replace(tmp, OUT)
    return {"rows": int(len(out)), "columns": list(out.columns), "non_null": counts}


def main() -> None:
    print("Building per-profile signal variants (var_5pct, bond_vol, mom6, gamma_illiq)...")
    res = build()
    print(f"  {OUT.relative_to(REPO_ROOT)}: {res['rows']:,} rows, {len(res['columns'])} cols")
    for pid, c in res["non_null"].items():
        print(f"  {pid}: " + ", ".join(f"{k.split('_'+pid)[0]}={v:,}" for k, v in c.items()))
    REPORT.write_text(json.dumps(
        {"run_timestamp": datetime.now(timezone.utc).isoformat(),
         "stage": "profile_signal_variants",
         "additive_to": "signals/{var_5pct,bond_vol,mom6,gamma_illiq}.parquet (NOT modified)",
         **res}, indent=2, default=str))
    print(f"  Report: {REPORT.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
