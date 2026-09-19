"""
Gross-error check — descriptive, NON-GATING (always exits 0).

Compares the pipeline's monthly long-short factor series against external
published references, development window only:

  1. drf vs the PRINTED DRR-2023 Table 1 Panel B figure (0.673 %/mo, t 3.355,
     2004:08-2016:12) — the printed window lies entirely inside the development
     split, so the figure is matched directly on the pipeline's series. Level
     divergence is EXPECTED (par-weighting, wider universe, clean-price basis —
     the D-Q17 rationale); reported sign-aware, never gated.
  2. str and mom6 vs the authors' published monthly factor series
     (data/development/dickerson_factor_returns.parquet, written by
     scripts/download_dickerson_factors.py — truncated at the development
     boundary at download, holdout rows discarded unread).
  3. Structural certification of the external file against facts its own
     README documents (factor count, LBFI gap-month NaN pattern, decimal
     units, boundary month) — the load/truncate derivation is certified
     against published documentation, not against a printed factor figure:
     the DRR-2023 replicated BBW factor series is absent from the
     public distribution.

Sign conventions (anchor spec §5.1/§9): comparisons are magnitude/correlation/
direction-aware, never signed-level. The external `str*` column is
sign-corrected (-1 x the raw winners-minus-losers series, README-documented);
our str is winners-minus-losers, and its sign on the panel is reported, not
assumed. Sign disagreement in as-published orientation is therefore
not by itself an error; the leg-convention orientation un-flips it.

Policy firewall (docs/thresholds.yaml validation notes / D16): external level
comparisons stay DESCRIPTIVE; no absolute-level criterion applies.
This script always exits 0 (deliberate divergence from the gate scripts'
exit-code convention) and is NEVER wired into run_validation_gates.py.

Output: data/development/headlines/dickerson_gross_error.json

Usage:
  python scripts/run_dickerson_gross_error_check.py [--factors-dir DIR] [--out PATH]
      defaults: data/development/factors, data/development/headlines/dickerson_gross_error.json
"""

import argparse
import hashlib
import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
from agents.quant.library.characteristic_sort import summarize_returns  # noqa: E402

EXTERNAL_FILE = REPO_ROOT / "data" / "development" / "dickerson_factor_returns.parquet"
FACTORS_DIR = REPO_ROOT / "data" / "development" / "factors"
OUT = REPO_ROOT / "data" / "development" / "headlines" / "dickerson_gross_error.json"
THRESHOLDS_FILE = REPO_ROOT / "docs" / "thresholds.yaml"

# Printed reference figures — citations, not tunable thresholds
# (cf. build_str_decomposition.py, which embeds its published targets).
# Dickerson, Mueller & Robotti (2023, JFE), Table 1 Panel B, replicated
# WRDS-based BBW factors, sample 2004:08-2016:12 (149 months), %/month.
DRR2023_T1PB_WINDOW = ("2004-08", "2016-12")
DRR2023_T1PB_N_MONTHS = 149
DRR2023_T1PB_DRF_MEAN_PCT = 0.673
DRR2023_T1PB_DRF_TSTAT = 3.355

# External column resolution: exact candidates in preference order. Per the
# distribution README (single_sort_exc_all.csv): `str*` (sign-corrected),
# `mom6_1` (not sign-corrected). Candidates retained defensively in case the
# distribution's naming shifts on a future re-pull.
EXTERNAL_COLUMNS = {
    "str": ["str*", "str"],
    "mom6": ["mom6_1", "mom6_1*"],
}
SIGN_CORRECTED_SUFFIX = "*"  # README: trailing * = series multiplied by -1

# Pipeline series: (parquet name, headline column, secondary column). Decimal units.
OUR_SERIES = {
    "str": ("str.parquet", "str_corr", "str_raw"),
    "mom6": ("mom6.parquet", "mom6_corr", "mom6_raw"),
    "drf": ("bbw_factors.parquet", "drf_corr", "drf_raw"),
}

# README-documented LBFI gaps (factor returns NaN in these months) — a
# known-answer pattern entirely outside both windows of interest.
LBFI_GAP_MONTHS = [
    "1975-08", "1975-09", "1975-10", "1975-11",
    "1984-12", "1985-01", "1985-02", "1985-03",
]
LBFI_PROBE_COLUMN = "cs"  # credit spread — README's example full-span factor
N_FACTOR_COLUMNS_DOCUMENTED = 108
MIN_OVERLAP_MONTHS = 24


def thresholds_sha256() -> str:
    return hashlib.sha256(THRESHOLDS_FILE.read_bytes()).hexdigest()


def _dev_boundary_period() -> pd.Period:
    import yaml
    with open(THRESHOLDS_FILE) as f:
        hsy = int(yaml.safe_load(f)["trace_cleaning"]["holdout_start_year"])
    return pd.Period(f"{hsy - 1}-12", freq="M")


def _f(x) -> float | None:
    """JSON-safe float: None for NaN/inf so the report never carries bare NaN."""
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def find_external_column(columns: list[str], candidates: list[str]) -> tuple[str | None, str]:
    """Resolve a factor's external column. Never guesses: exact candidates in
    order; else a single contains-match on the first candidate's base name;
    multiple contains-matches → ('ambiguous', comparison skipped)."""
    for cand in candidates:
        if cand in columns:
            return cand, "matched"
    base = candidates[0].rstrip(SIGN_CORRECTED_SUFFIX)
    contains = [c for c in columns if base in c]
    if len(contains) == 1:
        return contains[0], "matched"
    if len(contains) > 1:
        return None, "ambiguous"
    return None, "missing"


def detect_external_scale(s: pd.Series) -> tuple[float, str]:
    """Decimal vs percent units. Monthly long-short factor returns have
    |median| well under 0.2 in decimal units and well over it in percent."""
    if pd.api.types.is_numeric_dtype(s):
        vals = pd.to_numeric(s, errors="coerce").dropna()
    else:
        vals = pd.to_numeric(s.astype(str).str.rstrip("%"), errors="coerce").dropna()
    if vals.empty:
        return 1.0, "empty"
    return (0.01, "percent") if vals.abs().median() > 0.2 else (1.0, "decimal")


def _summaries_pct(series_pct: pd.Series) -> dict:
    """summarize_returns under auto-NW and plain (nw_lags=0) errors, %/month."""
    out = {}
    for label, lags in (("nw_auto", None), ("plain", 0)):
        s = summarize_returns(series_pct.reset_index(drop=True), lags, 12)
        out[label] = {
            "mean_pct": _f(s["average"]),
            "t_stat": _f(s["t_stat"]),
            "sharpe": _f(s["sharpe"]),
            "n_months": int(s["n_months"]),
            "nw_lags_used": int(s["nw_lags_used"]),
        }
    return out


def _window_slice(s: pd.Series, window: tuple[str, str]) -> pd.Series:
    periods = pd.PeriodIndex(s.index, freq="M")
    lo, hi = pd.Period(window[0], freq="M"), pd.Period(window[1], freq="M")
    return s[(periods >= lo) & (periods <= hi)]


def drf_printed_window_comparison(our_drf_dec: pd.Series, column: str) -> dict:
    """The pipeline's drf over the printed DRR-2023 window vs the printed figure.
    Sign-aware magnitude comparison; level divergence expected (D-Q17)."""
    window_pct = _window_slice(our_drf_dec, DRR2023_T1PB_WINDOW).dropna() * 100.0
    stats = _summaries_pct(window_pct)
    mean_pct = stats["nw_auto"]["mean_pct"]
    ratio = None
    if mean_pct is not None:
        ratio = _f(abs(mean_pct) / DRR2023_T1PB_DRF_MEAN_PCT)
    return {
        "our_column": column,
        "printed_window": list(DRR2023_T1PB_WINDOW),
        "printed_mean_pct": DRR2023_T1PB_DRF_MEAN_PCT,
        "printed_t_stat": DRR2023_T1PB_DRF_TSTAT,
        "n_months": stats["nw_auto"]["n_months"],
        "window_complete": stats["nw_auto"]["n_months"] == DRR2023_T1PB_N_MONTHS,
        "ours": stats,
        "sign_ours": _f(np.sign(mean_pct)) if mean_pct is not None else None,
        "sign_printed": 1.0,
        "sign_agreement": bool(mean_pct is not None and np.sign(mean_pct) > 0),
        "abs_mean_ratio_ours_over_printed": ratio,
        "note": "printed window lies inside the development split; level "
                "divergence expected (par-weighting, wider universe, "
                "clean-price basis — D-Q17 rationale); descriptive only",
    }


def compare_factor(ours_dec: pd.Series, theirs_dec: pd.Series,
                   factor: str, external_column: str) -> dict:
    """Matched-month descriptive comparison, both series decimal units.
    Correlation and |magnitude| carry the information; means are reported
    sign-aware in both the as-published and leg-convention orientations."""
    joined = pd.concat(
        [ours_dec.rename("ours"), theirs_dec.rename("theirs")], axis=1, join="inner"
    ).dropna()
    n = len(joined)
    sign_corrected = external_column.endswith(SIGN_CORRECTED_SUFFIX)
    out = {
        "external_column": external_column,
        "external_sign_corrected": sign_corrected,
        "n_matched_months": n,
        "overlap_sufficient": n >= MIN_OVERLAP_MONTHS,
        "matched_first_month": str(joined.index.min()) if n else None,
        "matched_last_month": str(joined.index.max()) if n else None,
    }
    if n == 0:
        return out
    mean_ours = float(joined["ours"].mean())
    mean_theirs_pub = float(joined["theirs"].mean())
    # Leg convention: undo the README-documented -1 presentation flip so both
    # series are winners-minus-losers (P10-P1) before signs are compared.
    mean_theirs_leg = -mean_theirs_pub if sign_corrected else mean_theirs_pub
    corr_pub = float(joined["ours"].corr(joined["theirs"])) if n >= 3 else float("nan")
    corr_leg = -corr_pub if sign_corrected else corr_pub
    out.update({
        "mean_ours_pct": _f(mean_ours * 100.0),
        "mean_external_as_published_pct": _f(mean_theirs_pub * 100.0),
        "mean_external_leg_convention_pct": _f(mean_theirs_leg * 100.0),
        "sign_ours": _f(np.sign(mean_ours)),
        "sign_external_leg_convention": _f(np.sign(mean_theirs_leg)),
        "sign_agreement_leg_convention": bool(
            np.sign(mean_ours) == np.sign(mean_theirs_leg) and mean_ours != 0.0
        ),
        "abs_mean_ratio": (
            _f(abs(mean_ours) / abs(mean_theirs_leg))
            if abs(mean_theirs_leg) > 1e-12 else None
        ),
        "corr_signed_as_published": _f(corr_pub),
        "corr_signed_leg_convention": _f(corr_leg),
        "abs_corr": _f(abs(corr_pub)),
    })
    if factor == "str":
        out["note"] = (
            "our str is winners-minus-losers; DRR's raw str is the same leg convention "
            "with a negative (reversal) premium, published sign-corrected as "
            "`str*`. Sign disagreement in leg convention is not by itself "
            "an error; |corr| is the gross-error statistic."
        )
    return out


def structural_certification(ext: pd.DataFrame, matched: dict) -> dict:
    """Certify the load/truncate derivation against facts documented in the
    distribution's own README (window-free or pre-development, never holdout)."""
    factor_cols = [c for c in ext.columns if c != "year_month"]
    checks = {}

    checks["factor_column_count"] = {
        "expected": N_FACTOR_COLUMNS_DOCUMENTED,
        "observed": len(factor_cols),
        "pass": len(factor_cols) == N_FACTOR_COLUMNS_DOCUMENTED,
    }
    checks["required_columns_matched"] = {
        "expected": list(EXTERNAL_COLUMNS),
        "observed": {k: v for k, v in matched.items()},
        "pass": all(v is not None for v in matched.values()),
    }
    boundary = str(_dev_boundary_period())
    max_month = str(ext["year_month"].max())
    checks["boundary_month"] = {
        "expected": boundary,
        "observed": max_month,
        "pass": max_month == boundary,
    }
    if LBFI_PROBE_COLUMN in ext.columns:
        probe = ext.set_index("year_month")[LBFI_PROBE_COLUMN]
        gap_rows = probe.reindex(LBFI_GAP_MONTHS)
        neighbour = probe.get("1976-01")
        checks["lbfi_gap_months_nan"] = {
            "probe_column": LBFI_PROBE_COLUMN,
            "expected": "all NaN in documented gap months; populated 1976-01",
            "observed_gap_nan": int(gap_rows.isna().sum()),
            "observed_gap_total": len(LBFI_GAP_MONTHS),
            "neighbour_1976_01_populated": bool(pd.notna(neighbour)),
            "pass": bool(gap_rows.isna().all() and pd.notna(neighbour)),
        }
    else:
        checks["lbfi_gap_months_nan"] = {
            "probe_column": LBFI_PROBE_COLUMN, "pass": None,
            "note": "probe column absent — check skipped",
        }
    mom6_col = matched.get("mom6")
    if mom6_col is not None:
        _, scale_label = detect_external_scale(ext[mom6_col])
        checks["units_decimal"] = {
            "probe_column": mom6_col,
            "expected": "decimal",
            "observed": scale_label,
            "pass": scale_label == "decimal",
        }
    first_non_nan = {
        k: (str(ext.set_index("year_month")[c].dropna().index.min())
            if c is not None and not ext.set_index("year_month")[c].dropna().empty
            else None)
        for k, c in matched.items()
    }
    passes = [c["pass"] for c in checks.values() if c.get("pass") is not None]
    return {
        "basis": "README_factor_time_series.txt (shipped in the distribution); "
                 "the DRR-2023 replicated BBW factor series is not part of the "
                 "public distribution",
        "checks": checks,
        "first_non_nan_month_observational": first_non_nan,
        "all_pass": bool(passes and all(passes)),
    }


def _load_year_month_series(path: Path, column: str) -> pd.Series | None:
    df = pd.read_parquet(path)
    if column not in df.columns:
        return None
    ym = pd.to_datetime(df["date"]).dt.to_period("M").astype(str)
    return pd.Series(df[column].values, index=ym)


def _write_report(report: dict, out: Path | None = None) -> None:
    out = OUT if out is None else out
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".json.tmp")
    with open(tmp, "w") as f:
        json.dump(report, f, indent=2)
    os.replace(tmp, out)
    print(f"\nReport written: {out}")


def _rel(path: Path) -> str:
    """Repo-relative path when inside the repo, else the path as given (never raises)."""
    try:
        return str(Path(path).relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Defaults are read from the module globals at call time (tests monkeypatch them)."""
    ap = argparse.ArgumentParser(description="Descriptive gross-error check.")
    ap.add_argument("--factors-dir", type=Path, default=FACTORS_DIR,
                    help="directory holding str/mom6/bbw_factors parquets "
                         "(default data/development/factors)")
    ap.add_argument("--out", type=Path, default=OUT,
                    help="report path (default data/development/headlines/dickerson_gross_error.json)")
    return ap.parse_args(argv or [])


def _envelope() -> dict:
    return {
        "run_timestamp": datetime.now(timezone.utc).isoformat(),
        "thresholds_sha256": thresholds_sha256(),
        "family": "corr",
        "gate": {
            "gating": False,
            "criterion": "descriptive gross-error check; external level "
                         "comparisons stay descriptive (D16, "
                         "D-Q17); never wired into run_validation_gates.py; "
                         "exits 0 unconditionally",
        },
    }


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    factors_dir, out = Path(args.factors_dir), Path(args.out)
    if factors_dir != FACTORS_DIR and out == OUT:
        raise SystemExit(f"ERROR: --factors-dir {factors_dir} with the default --out would overwrite the default "
                         f"report {OUT}; pass --out")
    report = _envelope()
    if factors_dir != FACTORS_DIR:
        report["factors_dir"] = _rel(factors_dir)
        report["factors_sha256"] = {v[0]: hashlib.sha256((factors_dir / v[0]).read_bytes()).hexdigest()
                                    for v in OUR_SERIES.values() if (factors_dir / v[0]).is_file()}

    missing = [str(p) for p in
               [EXTERNAL_FILE] + [factors_dir / v[0] for v in OUR_SERIES.values()]
               if not p.exists()]
    if missing:
        report["status"] = "inputs_missing"
        report["missing_files"] = missing
        print("Missing inputs — run scripts/download_dickerson_factors.py and the "
              "factor builders first:")
        for m in missing:
            print(f"  {m}")
        _write_report(report, out)
        sys.exit(0)

    ext = pd.read_parquet(EXTERNAL_FILE)
    boundary = _dev_boundary_period()
    periods = pd.PeriodIndex(ext["year_month"], freq="M")
    n_late = int((periods > boundary).sum())
    if n_late:
        print(f"WARNING: {n_late} post-boundary row(s) in {EXTERNAL_FILE.name} — "
              f"stale or foreign artifact; dropped on load.")
    report["post_boundary_rows_dropped_on_load"] = n_late
    ext = ext[periods <= boundary].reset_index(drop=True)

    ext_cols = list(ext.columns)
    matched, statuses = {}, {}
    for factor, candidates in EXTERNAL_COLUMNS.items():
        name, status = find_external_column(ext_cols, candidates)
        matched[factor], statuses[factor] = name, status
        print(f"external column for {factor}: {name!r} ({status})")

    report["external_column_resolution"] = {
        f: {"column": matched[f], "status": statuses[f],
            "candidates": EXTERNAL_COLUMNS[f]}
        for f in EXTERNAL_COLUMNS
    }
    report["structural_certification"] = structural_certification(ext, matched)

    ext_ym = ext.set_index("year_month")

    # 1. drf vs the printed DRR-2023 figure, on the pipeline's series.
    drf_path = factors_dir / OUR_SERIES["drf"][0]
    our_drf = _load_year_month_series(drf_path, OUR_SERIES["drf"][1])
    if our_drf is not None:
        report["drf_printed_figure"] = drf_printed_window_comparison(
            our_drf, OUR_SERIES["drf"][1]
        )
        d = report["drf_printed_figure"]
        print(f"drf printed-window: ours {d['ours']['nw_auto']['mean_pct']} %/mo "
              f"over {d['n_months']} mo vs printed {DRR2023_T1PB_DRF_MEAN_PCT}")
    else:
        report["drf_printed_figure"] = {
            "status": "our_column_missing", "expected_column": OUR_SERIES["drf"][1]
        }

    # 2. str / mom6 vs the truncated external series, primary + secondary arms.
    comparisons = {}
    for factor in ("str", "mom6"):
        ext_col = matched[factor]
        if ext_col is None:
            comparisons[factor] = {"status": statuses[factor]}
            continue
        scale, scale_label = detect_external_scale(ext_ym[ext_col])
        theirs = pd.to_numeric(ext_ym[ext_col], errors="coerce") * scale
        fname, primary, secondary = OUR_SERIES[factor]
        block = {"external_scale_detected": scale_label, "arms": {}}
        for col in (primary, secondary):
            ours = _load_year_month_series(factors_dir / fname, col)
            if ours is None:
                block["arms"][col] = {"status": "our_column_missing"}
                continue
            block["arms"][col] = compare_factor(ours, theirs, factor, ext_col)
            c = block["arms"][col]
            print(f"{factor} [{col}] vs {ext_col}: n={c.get('n_matched_months')}, "
                  f"|corr|={c.get('abs_corr')}, |ratio|={c.get('abs_mean_ratio')}")
        comparisons[factor] = block
    report["series_comparisons"] = comparisons

    _write_report(report, out)
    sys.exit(0)


if __name__ == "__main__":
    main(sys.argv[1:])
