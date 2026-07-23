"""
Build the IPCA Workstream B characteristic-panel feed (the buildable FISD+TRACE 7-instrument
subset) for the ipca module. Spec: docs/quant/specs/characteristic_registry_spec.md. Instruments +
family policy: agents/quant/library/configs/ipca_instruments.yaml.

This script is now THIN file-I/O glue over the pure, importable feed builder in
``agents/quant/library/ipca_feed.py`` (the same functions, relocated so a feed can be built from
any in-memory panel state — P_N, P_{N\\b} — for the frozen-loadings IPCA differential). The on-disk
output is unchanged; a bit-for-bit regression is pinned in tests/unit/test_ipca_feed_builder.py.

Pipeline (family-parameterised, corr default per D6):
  1. Merge the panel + signal parquets on (cusip, date)  [I/O, here in ``assemble``].
  2-5. align_scale + rank_and_emit + validate_feed        [pure, in ``ipca_feed``].
  6. Emit data/development/ipca_panel_<family>.parquet (long) + an audit JSON, wall-enforced.

This is INTERFACE-VALIDATION data — NON-COMPARABLE to KPP (7-instrument bond-only, VOL-scaled,
no equity, no DtS). See docs/quant/specs/characteristic_registry_spec.md §7.

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

import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
# The pure feed pipeline now lives in the library so it is importable for the IPCA differential.
# Re-exported here (INSTRUMENTS, L, align_scale, rank_and_emit, validate_emitted) for backward
# compatibility with tests/unit/test_build_ipca_panel.py.
from agents.quant.library.ipca_feed import INSTRUMENTS, L, build_ipca_feed, validate_feed  # noqa: E402

# Re-exported for backward compatibility with tests/unit/test_build_ipca_panel.py, which imports
# the pure stages (and the former script-local `validate_emitted`) from this module.
from agents.quant.library.ipca_feed import align_scale, rank_and_emit  # noqa: E402,F401

validate_emitted = validate_feed  # backward-compatible alias (former script-local name)

DEV = REPO_ROOT / "data" / "development"
PANEL_FILE = DEV / "monthly_panel_maximal.parquet"
REGISTRY = REPO_ROOT / "agents" / "quant" / "library" / "configs" / "ipca_instruments.yaml"
THRESHOLDS_FILE = REPO_ROOT / "docs" / "thresholds.yaml"


def load_registry() -> dict:
    with open(REGISTRY) as f:
        return yaml.safe_load(f)


def thresholds_sha256() -> str:
    return hashlib.sha256(THRESHOLDS_FILE.read_bytes()).hexdigest()


def assemble(family: str, reg: dict) -> pd.DataFrame:
    """Read + merge the panel and signal parquets into the canonical instrument frame (the I/O
    step). Returns the merged frame carrying cusip, date, and the 7 instrument columns; the pure
    align/scale/rank stages are then run by ``ipca_feed.build_ipca_feed``."""
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
    return df


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
    merged = assemble(family, reg)
    out, counts = build_ipca_feed(merged, reg, family, validate=False)
    print(f"  merged={counts['rows_merged']:,}  adjacent={counts['rows_adjacent']:,}  "
          f"vol>0={counts['rows_vol_positive']:,}  complete-case={counts['rows_complete_case']:,}")
    print(f"  months kept={counts['months_kept']} (dropped small={counts['months_dropped_small']}); "
          f"N_m min/median/max={counts['N_m_min']}/{counts['N_m_median']}/{counts['N_m_max']}; "
          f"rows={counts['rows_emitted']:,}")

    print("Validating emitted feed against ipca.validate_panel + wall...")
    validate_feed(out, reg, family)
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
            "See docs/quant/specs/characteristic_registry_spec.md §2."
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
