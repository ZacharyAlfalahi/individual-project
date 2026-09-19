"""
Build the BBW (2019) four-factor harness: DRF, LRF, REV, the three CRF
components, and the CRF composite (docs/quant/specs/BBW_anchor_implementation_spec.md §3, §4).
MKTB is built separately (build_mktb.py); together they are the four BBW model
factors (MKTB, DRF, LRF, CRF) plus the standalone REV.

Per family (A9): view() selects the family's returns and the family-resolved
var_5pct / gamma signals; rating is family-agnostic; rev = the contemporaneous
return (REV sorts on the prior-month return, signal_lag=0). The six factors run
through the audited engine via agents/quant/library/bbw_factors.py; CRF is their
equal-weighted credit-axis composite.

As-published alignment (lib_gap OFF, signal_lag=0, contemporaneous). The lead/lag
injection and lib_gap toggles are Chunk-4 work. Per-factor effective windows
(§6): DRF / CRF start ~2004-06 (VaR5 needs >=24 of trailing 36 months); LRF / REV
from panel start ~2002-08.

Output: data/development/factors/bbw_factors.parquet with columns
  date, {drf,lrf,rev,crf_var,crf_illiq,crf_rev,crf}_{raw,corr}.

Usage:
  python scripts/build_bbw_factors.py
  python scripts/build_bbw_factors.py --basis {total_return,clean}
      input = basis_inputs.PANELS[basis][0]; outputs -> results/consistent_basis/<basis>/factors/

Requires:
  data/development/monthly_panel_maximal.parquet
  data/development/signals/var_5pct.parquet
  data/development/signals/gamma_illiq.parquet
"""

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from functools import reduce
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))
from scripts import basis_inputs  # noqa: E402
from agents.quant.library.bbw_factors import (  # noqa: E402
    BBW_FACTOR_CONFIGS, run_bbw_factor, compose_crf,
)
from agents.quant.library.characteristic_sort import summarize_returns  # noqa: E402
from agents.quant.library.run_config import (  # noqa: E402
    RunConfig, PanelViewConfig, ConstructionConfig, EvaluationConfig,
)
from agents.quant.library.views import view  # noqa: E402

_TOTAL = REPO_ROOT / "data" / "development" / "monthly_panel_total_return.parquet"
_CLEAN = REPO_ROOT / "data" / "development" / "monthly_panel_maximal.parquet"
# Anchor headline uses the §2.1 total-return panel when present (BBW_ANCHOR_PANEL
# overrides; clean maximal fallback).
PANEL_FILE = Path(os.environ.get("BBW_ANCHOR_PANEL", str(_TOTAL if _TOTAL.exists() else _CLEAN)))
VAR_FILE = REPO_ROOT / "data" / "development" / "signals" / "var_5pct.parquet"
GAMMA_FILE = REPO_ROOT / "data" / "development" / "signals" / "gamma_illiq.parquet"
OUT_DIR = REPO_ROOT / "data" / "development" / "factors"
OUT_FILE = OUT_DIR / "bbw_factors.parquet"
REPORT_OUT = OUT_DIR / "bbw_factors_report.json"
THRESHOLDS_FILE = REPO_ROOT / "docs" / "thresholds.yaml"

FACTORS = list(BBW_FACTOR_CONFIGS)  # drf, lrf, rev, crf_var, crf_illiq, crf_rev


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


def run_family(maximal: pd.DataFrame, signals: pd.DataFrame, family: str):
    cfg = RunConfig(
        panel_view=PanelViewConfig(price_family=family, stale_mask=False,
                                   include_terminal_rows=False),
        construction=ConstructionConfig(signal_lag=0, expost_trim="none"),
        evaluation=EvaluationConfig(),
    )
    panel = view(maximal, cfg, signals=signals).drop_duplicates(
        subset=["cusip", "date"]).reset_index(drop=True)
    panel["rev"] = panel["ret"]   # REV factor sorts on the contemporaneous prior-month return (score=rev)
    # CRF_REV controls on the same reversal signal, exposed under `xret`: its gold concept
    # is prior_1m_excess_return, which the frozen D27 table binds to column `xret`, so
    # bbw_factors' crf_rev control reads xret. Alias here (BBW's reversal is
    # the raw prior-month return) so the builder can run the crf_rev leg.
    panel["xret"] = panel["ret"]
    panel = panel[["cusip", "date", "ret", "size", "rating", "var_5pct", "gamma", "rev", "xret"]]

    monthly, summaries, component_mr = {}, {}, {}
    for name in FACTORS:
        res = run_bbw_factor(panel, name)
        mr = res["monthly_returns"]
        monthly[name] = mr[["date", "strategy_ret"]].rename(
            columns={"strategy_ret": f"{name}_{family}"})
        summaries[name] = res["summary"]
        component_mr[name] = mr

    crf = compose_crf(component_mr).rename(columns={"crf": f"crf_{family}"})
    monthly["crf"] = crf
    summaries["crf"] = summarize_returns(
        pd.Series(crf[f"crf_{family}"].values, index=pd.DatetimeIndex(crf["date"])),
        nw_lags=None, months_per_year=12,
    )
    return monthly, summaries


def write_factor(factor: pd.DataFrame, out_file: Path | None = None) -> None:
    out_file = OUT_FILE if out_file is None else out_file
    out_file.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pandas(factor, preserve_index=False)
    meta = dict(table.schema.metadata or {})
    meta.update({
        b"factor_name":  b"bbw_four_factor",
        b"factor_source": b"BBW_2019",
        b"primary_key":  b"date",
        b"families":     b"raw,corr",
        b"weighting":    b"value_weight_par_offering_amt",
        b"alignment":    b"as_published_signal_lag_0",
        b"family_policy": b"A9_no_cross_family_mixing",
    })
    table = table.replace_schema_metadata(meta)
    tmp = out_file.with_suffix(".parquet.tmp")
    pq.write_table(table, str(tmp))
    os.replace(tmp, out_file)
    print(f"  Written: {out_file}")


def write_report(factor: pd.DataFrame, summaries: dict, report_out: Path | None = None,
                 provenance: dict | None = None) -> None:
    report_out = REPORT_OUT if report_out is None else report_out

    def _stats(name: str, fam: str) -> dict:
        col = f"{name}_{fam}"
        s = factor[col].dropna()
        first = factor.loc[factor[col].notna(), "date"]
        return {
            "n_months": int(len(s)),
            "mean_pct_per_month": float(s.mean() * 100) if len(s) else None,
            "sd_pct_per_month": float(s.std(ddof=1) * 100) if len(s) > 1 else None,
            "t_stat": float(summaries[fam][name]["t_stat"]),
            "effective_start": str(first.min().date()) if len(first) else None,
        }

    report = {
        "run_timestamp": datetime.now(timezone.utc).isoformat(),
        "thresholds_sha256": thresholds_sha256(),
        **(provenance or {}),
        "headline_series_family": "corr",
        "alignment": "as-published (signal_lag=0); lead/lag + lib_gap toggles are Chunk 4",
        "leg_directions": {k: {kk: vv for kk, vv in v.items()} for k, v in BBW_FACTOR_CONFIGS.items()},
        "crf_composite": "(crf_var + crf_illiq + crf_rev) / 3 (§3.5)",
        "expected_effective_windows": "DRF/CRF ~2004-06 (VaR5 >=24/36); LRF/REV ~2002-08 (§6)",
        "directional_targets_DRR2023_table1": {
            "drf": "DRR-2023 Table 1: +0.673%/mo (expected sign +)",
            "rev": "expected negative by the losers-winners convention (§5.1)",
            "lrf": "DRR-2023 Table 1: +0.361%/mo",
            "crf": "DRR-2023 Table 1: +0.508%/mo (positive credit premium)",
        },
        "clean_price_contingency": (
            "Per §2.4 the credit/liquidity premia live in low-rated / illiquid "
            "bonds whose returns are dominated by coupon carry; a clean-price "
            "basis (no accrued interest / coupon, §2.1) omits it, which can "
            "flatten LRF and flip CRF. The registered 'revisit if CRF off' "
            "return-basis trigger is served by the §2.1 total-return panel."
        ),
        "note": "raw family inherits uncorrected price-error outliers in VW legs "
                "(see MKTB/str reports); corr is the headline. Construction is "
                "verified exact by tests/unit/test_bbw_factors.py; level "
                "differences versus DRR on a clean-price basis are a "
                "return-measurement effect, not construction.",
        "factors": {name: {fam: _stats(name, fam) for fam in ("raw", "corr")}
                    for name in FACTORS + ["crf"]},
    }
    tmp = report_out.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(report, f, indent=2)
    os.replace(tmp, report_out)
    print(f"  Report: {report_out}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Build the BBW (2019) factor harness.")
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
    for f in (panel_file, VAR_FILE, GAMMA_FILE):
        if not f.exists():
            print(f"ERROR: required input not found: {f}", file=sys.stderr)
            sys.exit(1)

    print("Loading panel + signals...")
    maximal = pd.read_parquet(panel_file)
    var5 = pd.read_parquet(VAR_FILE)
    gamma = pd.read_parquet(GAMMA_FILE)
    signals = var5.merge(gamma, on=["cusip", "date"], how="outer")

    all_monthly, summaries = {}, {}
    for fam in ("raw", "corr"):
        print(f"Running BBW factors on {fam} family...")
        monthly, summ = run_family(maximal, signals, fam)
        summaries[fam] = summ
        all_monthly[fam] = monthly

    frames = []
    for fam in ("raw", "corr"):
        frames.extend(all_monthly[fam][name] for name in FACTORS + ["crf"])
    factor = reduce(lambda a, b: a.merge(b, on="date", how="outer"), frames)
    factor = factor.sort_values("date").reset_index(drop=True)

    write_factor(factor, out_file)
    write_report(factor, summaries, report_out, basis_provenance(args.basis, panel_file))

    print("\nDone (corr family headline):")
    for name in FACTORS + ["crf"]:
        s = factor[f"{name}_corr"].dropna()
        if len(s):
            first = factor.loc[factor[f"{name}_corr"].notna(), "date"].min().date()
            print(f"  {name:9s}: mean {s.mean()*100:+.3f}%/mo, sd {s.std(ddof=1)*100:5.2f}%, "
                  f"t {summaries['corr'][name]['t_stat']:+.2f}, {len(s):3d} mo, from {first}")
    print(f"  → {out_file}")


if __name__ == "__main__":
    main(sys.argv[1:])
