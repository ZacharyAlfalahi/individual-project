"""
Total-return panel — the §2.1 accrual construction (Chunk 5).

Re-emits the maximal monthly panel with the clean-price returns replaced by TOTAL
returns:
    R = [(P_t + AI_t + C_t) − (P_{t-1} + AI_{t-1})] / (P_{t-1} + AI_{t-1})
using the accrual engine (agents/quant/library/accrual.py) and the FISD coupon
schedule (coupon, interest_frequency, maturity, day_count_basis). AI/C are
family-agnostic; only the clean price P_{raw,corr} differs across families. Adds
`ai`, `coupon_paid`, and `day_count_fallback` columns; everything else (size,
rating, universe_eligible, ...) is carried unchanged so the factor builders run
on this panel unmodified.

Accrual is the only difference from the maximal panel (par weighting is
identical), so any factor movement attributes cleanly to accrual.
Output: monthly_panel_total_return_default_flat.parquet when the FISD schedule carries
``default_date`` (the substrate the basis loader and the holdout inventory read), else
monthly_panel_total_return.parquet. Both live under data/development/.

Validation (necessary + sufficient checks):
  * Zero-coupon invariance (the control-group regression): Z bonds (coupon==0)
    must have total return equal to clean return within 1e-12 absolute — asserted here.
  * day_count_fallback surfaced as a panel column (ACT/* bonds priced on 30/360);
    factor-level exposure is reported by run_accrual_validation.py.
The clean-price maximal panel is not modified (it serves the levels /
differentials comparison).

Usage:
  python scripts/build_total_return_panel.py                      # names the output for the substrate built
  python scripts/build_total_return_panel.py --out <panel.parquet> [--report-out <report.json>]
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

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
from agents.quant.library.accrual import accrued_and_coupon, THIRTY_360  # noqa: E402

PANEL_FILE = REPO_ROOT / "data" / "development" / "monthly_panel_maximal.parquet"
FISD_STATIC = REPO_ROOT / "data" / "development" / "fisd" / "fisd_reference_static.parquet"
OUT_FILE = REPO_ROOT / "data" / "development" / "monthly_panel_total_return.parquet"
#: Written when the FISD schedule carries ``default_date`` (defaulted bonds trade flat). The basis
#: loader and the holdout inventory read THIS substrate, so the default output name states which
#: one was produced rather than leaving the caller to remember a flag.
OUT_FILE_DEFAULT_FLAT = REPO_ROOT / "data" / "development" / "monthly_panel_total_return_default_flat.parquet"
THRESHOLDS_FILE = REPO_ROOT / "docs" / "thresholds.yaml"


def thresholds_sha256() -> str:
    return hashlib.sha256(THRESHOLDS_FILE.read_bytes()).hexdigest()


def _total_return_one_family(panel: pd.DataFrame, price_col: str) -> np.ndarray:
    """Total return per row for one family, with the same calendar-adjacency rule
    as the clean-price build (prior month must be the immediately-preceding
    month, else NaN). AI/C columns are family-agnostic and already on `panel`."""
    P = panel[price_col].to_numpy(dtype=float)
    AI = panel["ai"].to_numpy(dtype=float)
    C = panel["coupon_paid"].to_numpy(dtype=float)
    cusip = panel["cusip"].to_numpy()
    per = panel["date"].dt.to_period("M").astype("int64").to_numpy()

    prev_P = np.r_[np.nan, P[:-1]]
    prev_AI = np.r_[np.nan, AI[:-1]]
    adj = np.zeros(len(panel), dtype=bool)
    adj[1:] = (cusip[1:] == cusip[:-1]) & ((per[1:] - per[:-1]) == 1)

    denom = prev_P + prev_AI
    with np.errstate(invalid="ignore", divide="ignore"):
        ret = ((P + AI + C) - (prev_P + prev_AI)) / denom
    ret = np.where(adj & (denom > 0), ret, np.nan)
    return ret


def to_total_return(maximal: pd.DataFrame, fisd_static: pd.DataFrame) -> pd.DataFrame:
    """Pure, in-memory clean→total-return transform for a BASE maximal-format panel (raw + corr).

    The in-memory analogue of ``main()``: converts the raw and corr return legs from clean price
    growth to total return ``R = [(P+AI+C) − (P_{-1}+AI_{-1})] / (P_{-1}+AI_{-1})`` via the SAME
    accrual engine and the SAME per-family kernel (``_total_return_one_family``). Everything that is
    not a return (price_eom_*, size, rating, universe_eligible, …) is carried through unchanged, so
    ``views.view`` runs on the result unmodified. Returns the frame with ``ret_{raw,corr}`` /
    ``xret_{raw,corr}`` replaced by totals, plus ``ai``/``coupon_paid``/``day_count_fallback``; the
    three merged FISD schedule fields are dropped.

    Per-paper PROFILE families (bbw_2019 / jostova_2013) are NOT handled here — they live on the
    standalone profiles panel and are converted by ``build_profile_total_return_panel``'s
    ``build_total_return_profiles`` (the function that builds
    ``monthly_panel_profiles_total_return[_default_flat].parquet``), then merged, exactly as the dev-side
    total-return track merges its two base panels. Applying this transform to an already
    profile-merged frame would recompute profile returns on the wrong (maximal) grid.

    Faithfulness: applied to the dev ``monthly_panel_maximal`` with the FISD static schedule
    (including ``default_date``) this reproduces the dev default-flat total-return panel
    (``monthly_panel_total_return_default_flat``, raw/corr) to within the self-check tolerance
    (max abs diff <= 1e-9) — asserted by the holdout inventory's dev self-check. This is the transform the gated real run applies to the SEEDED base
    maximal, so the
    real substrate is constructed identically to the dev artefact the rehearsal reads directly.

    Fails loud on any zero-coupon-invariance breach (accrual touching a coupon==0 bond).
    """
    required = ["cusip", "date", "maturity", "rf_monthly",
                "price_eom_raw", "price_eom_corr", "ret_raw", "ret_corr"]
    missing = [c for c in required if c not in maximal.columns]
    if missing:
        raise KeyError(f"to_total_return: maximal panel missing required columns {missing}")

    sched_cols = ["cusip", "coupon", "interest_frequency", "day_count_basis"]
    # `default_date` licenses the trade-flat-on-default cutoff (AI=C=0 from the default
    # month). Optional: absent on synthetic fixtures, present on the real FISD static.
    if "default_date" in fisd_static.columns:
        sched_cols.append("default_date")
    sched = fisd_static[sched_cols].drop_duplicates("cusip")
    panel = maximal.merge(sched, on="cusip", how="left")
    panel = panel.sort_values(["cusip", "date"], kind="mergesort").reset_index(drop=True)

    default_arg = panel["default_date"].values if "default_date" in panel.columns else None
    ai, coupon_paid = accrued_and_coupon(
        panel["date"].values, panel["maturity"].values,
        panel["coupon"].values, panel["interest_frequency"].values,
        default=default_arg,
    )
    panel["ai"] = ai
    panel["coupon_paid"] = coupon_paid
    panel["day_count_fallback"] = (panel["day_count_basis"].fillna("NA") != THIRTY_360)

    zero = (panel["coupon"].fillna(0.0) == 0.0).to_numpy()
    rf_arr = panel["rf_monthly"].to_numpy(dtype=float)
    for fam in ("raw", "corr"):
        clean = panel[f"ret_{fam}"].to_numpy(dtype=float).copy()
        tot = _total_return_one_family(panel, f"price_eom_{fam}")
        both = zero & np.isfinite(clean) & np.isfinite(tot)
        if not bool(np.allclose(clean[both], tot[both], atol=1e-12, rtol=0)):
            bad = both & ~np.isclose(clean, tot, atol=1e-12, rtol=0)
            raise AssertionError(
                f"[{fam}] zero-coupon invariance FAILED: {int(bad.sum())} Z-bond-months changed "
                f"under accrual — the accrual path is touching coupon==0 bonds.")
        # NB: a finite clean return can go NaN under total return when the prior month's (P+AI)
        # denominator is not > 0. main() tolerates this for raw/corr (the row is simply NaN in the
        # built panel), so this transform must NOT hard-fail on it either.
        panel[f"ret_{fam}"] = tot
        panel[f"xret_{fam}"] = tot - rf_arr

    drop_cols = ["coupon", "interest_frequency", "day_count_basis"]
    if "default_date" in panel.columns:
        drop_cols.append("default_date")
    return panel.drop(columns=drop_cols)


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--out", type=Path, default=None,
                    help=f"output panel path (default: {OUT_FILE_DEFAULT_FLAT.name} when the FISD "
                         f"schedule carries default_date, else {OUT_FILE.name}); pass a distinct path "
                         "to build a sensitivity substrate without overwriting either")
    ap.add_argument("--report-out", type=Path, default=None,
                    help="report path (default: <out>_report.json beside --out)")
    args = ap.parse_args(argv)
    for f in (PANEL_FILE, FISD_STATIC):
        if not f.exists():
            print(f"ERROR: required input not found: {f}", file=sys.stderr)
            sys.exit(1)

    print("Loading panel + FISD coupon schedule...")
    panel = pd.read_parquet(PANEL_FILE)
    # `maturity` is already on the maximal panel (from its own FISD merge); pull
    # only the coupon-schedule fields (+ default_date for the trade-flat cutoff) to
    # avoid a column collision.
    fisd_cols = ["cusip", "coupon", "interest_frequency", "day_count_basis"]
    avail = set(pq.read_schema(str(FISD_STATIC)).names)
    # Name the output for the substrate actually built: with default_date the cutoff applies, so this
    # is the default-flat panel the basis loader reads; without it, it is the plain total-return panel.
    out_file = args.out or (OUT_FILE_DEFAULT_FLAT if "default_date" in avail else OUT_FILE)
    report_out = args.report_out or out_file.with_name(out_file.stem + "_report.json")
    if "default_date" in avail:
        fisd_cols.append("default_date")
    fisd = pd.read_parquet(FISD_STATIC, columns=fisd_cols).drop_duplicates("cusip")
    if "maturity" not in panel.columns:
        raise KeyError("maximal panel missing 'maturity' (expected from its FISD merge)")
    panel = panel.merge(fisd, on="cusip", how="left")
    panel = panel.sort_values(["cusip", "date"], kind="mergesort").reset_index(drop=True)

    print("Computing accrued interest + coupon (30/360; Z-coupon → 0; defaulted → flat)...")
    default_arg = panel["default_date"].values if "default_date" in panel.columns else None
    ai, coupon_paid = accrued_and_coupon(
        panel["date"].values, panel["maturity"].values,
        panel["coupon"].values, panel["interest_frequency"].values,
        default=default_arg,
    )
    panel["ai"] = ai
    panel["coupon_paid"] = coupon_paid
    panel["day_count_fallback"] = (panel["day_count_basis"].fillna("NA") != THIRTY_360)

    # Keep the clean returns to validate Z-invariance, then overwrite with totals.
    clean_raw = panel["ret_raw"].to_numpy(dtype=float)
    clean_corr = panel["ret_corr"].to_numpy(dtype=float)
    tot_raw = _total_return_one_family(panel, "price_eom_raw")
    tot_corr = _total_return_one_family(panel, "price_eom_corr")

    # --- Zero-coupon invariance regression (the control group) ---
    zero = (panel["coupon"].fillna(0.0) == 0.0).to_numpy()
    def _invariant(clean, tot, mask):
        # The gate is the tolerance; the max |diff| is REPORTED so the stronger claim (bit-exact
        # equality) is measured rather than assumed. `clean` is the stored ret_* column and `tot` is
        # recomputed here, so exactness is a property of the data, not guaranteed by construction.
        both = mask & ~np.isnan(clean) & ~np.isnan(tot)
        diff = np.abs(clean[both] - tot[both])
        max_abs = float(diff.max()) if diff.size else 0.0
        return bool(np.allclose(clean[both], tot[both], atol=1e-12, rtol=0)), int(both.sum()), max_abs
    inv_raw, n_raw, max_raw = _invariant(clean_raw, tot_raw, zero)
    inv_corr, n_corr, max_corr = _invariant(clean_corr, tot_corr, zero)
    if not (inv_raw and inv_corr):
        # Invariant breach — the accrual path touched a zero-coupon bond.
        bad = (~np.isclose(clean_corr, tot_corr, atol=1e-12)) & zero & ~np.isnan(clean_corr) & ~np.isnan(tot_corr)
        raise AssertionError(
            f"Zero-coupon invariance FAILED: {int(bad.sum())} Z-bond-months changed "
            f"(raw_ok={inv_raw}, corr_ok={inv_corr}). The accrual path is touching "
            f"bonds it must not — check Z detection (coupon==0)."
        )
    _exact = " (exact)" if max_raw == 0.0 and max_corr == 0.0 else f" (max |diff| {max(max_raw, max_corr):.2e})"
    print(f"  zero-coupon invariance OK: {n_corr:,} Z bond-months within 1e-12 (corr){_exact}")

    panel["ret_raw"] = tot_raw
    panel["ret_corr"] = tot_corr
    panel["xret_raw"] = tot_raw - panel["rf_monthly"]
    panel["xret_corr"] = tot_corr - panel["rf_monthly"]

    # Trade-flat-on-default diagnostic (verifiability): bond-months in/after the default
    # month on a coupon-bearing, not-yet-matured bond — the rows the cutoff zeroes.
    default_flat_months = 0
    defaulted_cusips = 0
    if "default_date" in panel.columns:
        dd = pd.to_datetime(panel["default_date"])
        has_cpn = panel["coupon"].fillna(0.0) > 0.0
        flat = dd.notna() & has_cpn & (panel["date"] >= dd) & (panel["date"] <= panel["maturity"])
        default_flat_months = int(flat.sum())
        defaulted_cusips = int(panel.loc[dd.notna(), "cusip"].nunique())

    # Drop only the schedule fields merged in here; keep the panel's own
    # `maturity` (an original column) and the computed ai/coupon_paid/fallback.
    drop_cols = ["coupon", "interest_frequency", "day_count_basis"]
    if "default_date" in panel.columns:
        drop_cols.append("default_date")
    panel = panel.drop(columns=drop_cols)

    out_file.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pandas(panel, preserve_index=False)
    meta = dict(table.schema.metadata or {})
    meta.update({
        b"return_basis": b"total_return_price_plus_AI_plus_coupon_30E360",
        b"accrual": (b"BBW_anchor_spec_2.1_default_flat" if "default_date" in avail
                     else b"BBW_anchor_spec_2.1"),
        b"par_weighting": b"unchanged_isolated",
    })
    table = table.replace_schema_metadata(meta)
    tmp = out_file.with_suffix(".parquet.tmp")
    pq.write_table(table, str(tmp))
    os.replace(tmp, out_file)

    fb = panel["day_count_fallback"]
    report = {
        "run_timestamp": datetime.now(timezone.utc).isoformat(),
        "thresholds_sha256": thresholds_sha256(),
        "return_basis": ("total return = (P + AI + C) growth (§2.1); 30E/360 accrual"
                         + ("; defaulted bonds trade flat (AI=C=0 from the default month)"
                            if "default_date" in avail else "; no default_date in the FISD schedule, so no trade-flat cutoff")),
        "zero_coupon_invariance": {"raw_ok": inv_raw, "corr_ok": inv_corr,
                                   "z_bond_months_checked": n_corr,
                                   "max_abs_diff_raw": max_raw, "max_abs_diff_corr": max_corr,
                                   "exact_match": bool(max_raw == 0.0 and max_corr == 0.0)},
        "default_flat": {
            "defaulted_cusips": defaulted_cusips,
            "default_flat_bond_months": default_flat_months,
            "applied": "default_date" in avail,
        },
        "day_count_fallback": {
            "fallback_bond_months": int(fb.sum()),
            "fallback_pct": float(fb.mean() * 100),
        },
        "ai_diagnostics": {
            "ai_max": float(panel["ai"].max()), "ai_mean_nonzero":
            float(panel.loc[panel["ai"] > 0, "ai"].mean()),
            "coupon_paid_months": int((panel["coupon_paid"] > 0).sum()),
        },
    }
    with open(report_out, "w") as f:
        json.dump(report, f, indent=2)

    print("\nDone.")
    print(f"  total-return panel: {len(panel):,} rows  →  {out_file}")
    print(f"  default-flat: {default_flat_months:,} bond-months zeroed across {defaulted_cusips:,} defaulted cusips")
    print(f"  day_count_fallback: {int(fb.sum()):,} bond-months ({fb.mean()*100:.2f}%)")
    print(f"  AI: max {panel['ai'].max():.3f}, mean(nonzero) {panel.loc[panel['ai']>0,'ai'].mean():.3f}, "
          f"coupon-payment months {int((panel['coupon_paid']>0).sum()):,}")
    print(f"  Report: {report_out}")


if __name__ == "__main__":
    main()
