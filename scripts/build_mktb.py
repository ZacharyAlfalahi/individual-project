"""
Build the BBW MKTB market factor: the value-weighted (par) average EXCESS
return of all eligible bonds each month (spec docs/quant/specs/BBW_anchor_implementation_spec.md
§3.6). No sort — this is the market basket the four BBW long-short factors price
against, and the simplest end-to-end check of the VW-excess plumbing (build
order §10 step 3).

Family-indexed per A9: emits `mktb_raw` (from `xret_raw`) and `mktb_corr` (from
`xret_corr`) as parallel columns; cross-family mixing is forbidden.

Weighting basis: par `offering_amt`, surfaced as the panel `size` column (§2.4),
NOT market value — a deliberate, documented divergence from OSBAP `mcap_e`.

Excess-return basis: the panel's `xret_* = ret_* − rf_monthly`, where rf is FRED
TB3MS (3-month T-bill). Spec §2.2 names the 1-month T-bill; the panel ships a
3-month-bill excess return, so MKTB inherits that divergence (it matters for
MKTB's level, though it cancels in every long-short factor). Switching to a
1-month bill is a panel-layer change, out of scope for this builder.

Output: data/development/factors/mktb.parquet with columns
  date, mktb_raw, mktb_corr, n_bonds_raw, n_bonds_corr.

Usage:
  python scripts/build_mktb.py
  python scripts/build_mktb.py --basis {total_return,clean}
      input = basis_inputs.PANELS[basis][0]; outputs -> results/consistent_basis/<basis>/factors/

Requires: data/development/monthly_panel_maximal.parquet (build_monthly_panel.py)
"""

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))
from scripts import basis_inputs  # noqa: E402
from agents.quant.library.market_factor import compute_market_factor  # noqa: E402

# Anchor-layer headline uses the §2.1 TOTAL-return panel when present (the
# return includes accrued interest and coupon); falls back to the
# clean maximal panel, and is overridable via BBW_ANCHOR_PANEL for comparison.
_TOTAL = REPO_ROOT / "data" / "development" / "monthly_panel_total_return.parquet"
_CLEAN = REPO_ROOT / "data" / "development" / "monthly_panel_maximal.parquet"
PANEL_FILE = Path(os.environ.get("BBW_ANCHOR_PANEL", str(_TOTAL if _TOTAL.exists() else _CLEAN)))
OUT_DIR = REPO_ROOT / "data" / "development" / "factors"
OUT_FILE = OUT_DIR / "mktb.parquet"
REPORT_OUT = OUT_DIR / "mktb_report.json"
THRESHOLDS_FILE = REPO_ROOT / "docs" / "thresholds.yaml"


def thresholds_sha256() -> str:
    return hashlib.sha256(THRESHOLDS_FILE.read_bytes()).hexdigest()


def _rel(path: Path) -> str:
    """Repo-relative path when inside the repo, else the path as given (never raises)."""
    try:
        return str(Path(path).relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def resolve_paths(basis: str | None = None) -> tuple[Path, Path, Path]:
    """(input panel, factor parquet, report json). No basis => the module defaults
    (BBW_ANCHOR_PANEL honoured); a basis => its PANELS entry + the basis factors dir."""
    if basis is None:
        return PANEL_FILE, OUT_FILE, REPORT_OUT
    out_dir = basis_inputs.factors_dir(basis)
    return basis_inputs.PANELS[basis][0], out_dir / OUT_FILE.name, out_dir / REPORT_OUT.name


def basis_provenance(basis: str | None, panel_file: Path) -> dict:
    """Report keys identifying the basis input panel; empty without a basis."""
    if basis is None:
        return {}
    return {"basis": basis, "input_panel": _rel(panel_file),
            "input_panel_sha256": basis_inputs.sha256(panel_file)}


def compute_dual_family(panel: pd.DataFrame) -> pd.DataFrame:
    """Compute MKTB on each family and join on date.

    Par value-weighted (size) average excess return of universe-eligible bonds.
    """
    raw = compute_market_factor(panel, ret_col="xret_raw").rename(
        columns={"mktb": "mktb_raw", "n_bonds": "n_bonds_raw"}
    )
    corr = compute_market_factor(panel, ret_col="xret_corr").rename(
        columns={"mktb": "mktb_corr", "n_bonds": "n_bonds_corr"}
    )
    return raw.merge(corr, on="date", how="outer").sort_values("date").reset_index(drop=True)


def write_factor(factor: pd.DataFrame, out_file: Path | None = None) -> None:
    out_file = OUT_FILE if out_file is None else out_file
    out_file.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pandas(factor, preserve_index=False)
    meta = dict(table.schema.metadata or {})
    meta.update({
        b"factor_name":  b"mktb",
        b"factor_source": b"BBW_2019",
        b"primary_key":  b"date",
        b"families":     b"raw,corr",
        b"weighting":    b"value_weight_par_offering_amt",
        b"return_basis": b"excess_over_TB3MS",
        b"family_policy": b"A9_no_cross_family_mixing",
    })
    table = table.replace_schema_metadata(meta)
    tmp = out_file.with_suffix(".parquet.tmp")
    pq.write_table(table, str(tmp))
    os.replace(tmp, out_file)
    print(f"  Written: {out_file}")


def write_report(factor: pd.DataFrame, report_out: Path | None = None,
                 panel_file: Path | None = None, provenance: dict | None = None) -> None:
    report_out = REPORT_OUT if report_out is None else report_out
    panel_file = PANEL_FILE if panel_file is None else panel_file

    def _stats(col: str) -> dict:
        s = factor[col].dropna()
        return {
            "n_months": int(len(s)),
            "mean_pct_per_month": float(s.mean() * 100) if len(s) else None,
            "sd_pct_per_month": float(s.std(ddof=1) * 100) if len(s) > 1 else None,
            "max_abs_pct_per_month": float(s.abs().max() * 100) if len(s) else None,
            "first_date": str(factor.loc[factor[col].notna(), "date"].min().date())
            if len(s) else None,
            "last_date": str(factor.loc[factor[col].notna(), "date"].max().date())
            if len(s) else None,
        }

    report = {
        "run_timestamp": datetime.now(timezone.utc).isoformat(),
        "thresholds_sha256": thresholds_sha256(),
        "input_panel": _rel(panel_file),
        **(provenance or {}),
        "weighting": "value-weight by par offering_amt (panel `size`); not mcap_e (§2.4)",
        "return_basis": "excess over rf = TB3MS/12 (3-month bill); spec §2.2 names "
                        "the 1-month bill — divergence inherited from the panel, "
                        "cancels in long-short factors but affects MKTB level",
        "eligibility": "universe_eligible == True, finite xret, size > 0",
        "headline_series": "mktb_corr",
        "notes": (
            "HEADLINE = mktb_corr. mktb_raw is the meas_err=OFF (uncorrected) "
            "counterpart and is DOMINATED BY OUTLIERS: the raw family retains "
            "decimal-slip / par-snap price errors that "
            "the decimal-shift + bounce-back + distressed filters remove in the "
            "corr family. A long-only value-weighted mean does not difference "
            "these out (unlike a long-short factor), so mktb_raw's level is not a "
            "usable market return — it is kept only for A9 family completeness. "
            "DRR-2023 reference level: ~0.47%/mo (2004:08–2016:12). On a "
            "clean-price basis (no accrued interest / coupon, spec §2.1) the "
            "level is offset; the §8 differential gates are designed to tolerate "
            "that offset."
        ),
        "mktb_raw": _stats("mktb_raw"),
        "mktb_corr": _stats("mktb_corr"),
    }
    tmp = report_out.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(report, f, indent=2)
    os.replace(tmp, report_out)
    print(f"  Report: {report_out}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Build the BBW MKTB market factor.")
    ap.add_argument("--basis", choices=basis_inputs.BASES, default=None,
                    help="return basis: read basis_inputs.PANELS[basis][0] and write under "
                         "results/consistent_basis/<basis>/factors/ (default: the "
                         "BBW_ANCHOR_PANEL / data/development/factors behaviour)")
    return ap.parse_args(argv or [])


def main(argv: list[str] | None = None):
    args = parse_args(argv)
    panel_file, out_file, report_out = resolve_paths(args.basis)
    if args.basis is not None and "BBW_ANCHOR_PANEL" in os.environ:
        print("WARNING: BBW_ANCHOR_PANEL is ignored when --basis is given", file=sys.stderr)
    if not panel_file.exists():
        print(f"ERROR: required input not found: {panel_file}", file=sys.stderr)
        sys.exit(1)

    print(f"Loading panel: {panel_file}")
    panel = pd.read_parquet(
        panel_file,
        columns=["date", "size", "universe_eligible", "xret_raw", "xret_corr"],
    )
    print(f"  {len(panel):,} rows")

    print("Computing MKTB (par value-weighted excess return) per family...")
    factor = compute_dual_family(panel)

    write_factor(factor, out_file)
    write_report(factor, report_out, panel_file, basis_provenance(args.basis, panel_file))

    raw = factor["mktb_raw"].dropna()
    corr = factor["mktb_corr"].dropna()
    print("\nDone.")
    print(f"  {len(factor):,} months")
    if len(corr):
        print(f"  mktb_corr (HEADLINE): mean {corr.mean()*100:.3f}%/mo, "
              f"sd {corr.std(ddof=1)*100:.3f}%, {len(corr)} months")
    if len(raw):
        print(f"  mktb_raw  (meas_err=OFF): mean {raw.mean()*100:.1f}%/mo "
              f"— uncorrected family (meas_err=OFF); not a usable "
              f"level (see report notes)")
    print(f"  → {out_file}")


if __name__ == "__main__":
    main(sys.argv[1:])
