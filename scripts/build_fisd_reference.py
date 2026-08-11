"""
Preprocess raw WRDS Mergent FISD reference tables into a clean, CUSIP-keyed
reference + a monthly as-of credit-rating panel, with a coverage/quality
report.

This is a PREPROCESSING-only step. It does NOT touch the monthly panel build
or the FISD→panel merge — those land in a later Phase 1. Its purpose is to
de-risk the contested decisions (universe rules, rating map, amount-outstanding
quality) and verify coverage BEFORE any panel re-emit is spent.

Inputs (data/fisd/, untracked licensed data):
  reference_fisd_mergedissue.parquet    one row per issue (CUSIP-level facts)
  reference_fisd_ratings.parquet        rating events keyed on issue_id
  reference_fisd_mergedissuer.parquet   issuer-level attributes (issuer_id)
  reference_fisd_mergedredemption.parquet  call/redemption schedule (issue_id)

  data/development/monthly_panel_maximal.parquet  the dev (cusip, date) grid
  the monthly ratings panel is built on this grid only — dev window, no holdout.

Outputs (data/development/fisd/):
  fisd_reference_static.parquet   one row per cusip: universe_eligible + reason
                                  flags, offering_amt, amount_outstanding (raw,
                                  flagged), maturity, coupon, coupon_type,
                                  day_count_basis, bond_type, perpetual,
                                  callable, issue_id, issuer_id, sic_code,
                                  country_domicile
  fisd_ratings_monthly.parquet    (cusip, date) on the dev grid: rating_numeric,
                                  rating_agency_used, investment_grade, is_rated
  fisd_reference_report.json      coverage & quality stats
  fisd_reference_report.md        short human-readable summary

All numerical/categorical thresholds come from docs/thresholds.yaml under the
`fisd:` block; nothing is hard-coded here.

Usage:
  python scripts/build_fisd_reference.py
"""

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
# The FISD reference-building workers now live in the library so agents/ code
# (the one-shot holdout dev-pseudo panel builder) can import them without reaching into
# scripts/. Re-exported here (input-file constants + workers) for main and for
# tests/unit/test_build_fisd_reference.py.
from agents.quant.library.fisd_reference import FISD_DIR, ISSUE_FILE, RATINGS_FILE, ISSUER_FILE, REDEMPTION_FILE, THRESHOLDS_FILE, _ISSUE_COLS, _REASON_FLAGS, load_config, thresholds_sha256, apply_universe_rules, rating_to_numeric, map_rating_events, asof_monthly_rating, _load_issue, _default_dates_by_issue, build_static, build_ratings_monthly  # noqa: E402,F401

PANEL_FILE = REPO_ROOT / "data" / "development" / "monthly_panel_maximal.parquet"

OUT_DIR = REPO_ROOT / "data" / "development" / "fisd"
STATIC_OUT = OUT_DIR / "fisd_reference_static.parquet"
RATINGS_OUT = OUT_DIR / "fisd_ratings_monthly.parquet"
REPORT_JSON = OUT_DIR / "fisd_reference_report.json"
REPORT_MD = OUT_DIR / "fisd_reference_report.md"


# ---------------------------------------------------------------------------
# Writers
# ---------------------------------------------------------------------------

def _write_parquet(df: pd.DataFrame, path: Path, extra_meta: dict) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pandas(df, preserve_index=False)
    meta = dict(table.schema.metadata or {})
    meta.update({k.encode(): v.encode() for k, v in extra_meta.items()})
    table = table.replace_schema_metadata(meta)
    tmp = path.with_suffix(path.suffix + ".tmp")
    pq.write_table(table, str(tmp))
    os.replace(tmp, path)
    print(f"  Written: {path.relative_to(REPO_ROOT)}  ({len(df):,} rows)")


def _write_report(report: dict) -> None:
    tmp = REPORT_JSON.with_suffix(".json.tmp")
    with open(tmp, "w") as f:
        json.dump(report, f, indent=2)
    os.replace(tmp, REPORT_JSON)
    print(f"  Report: {REPORT_JSON.relative_to(REPO_ROOT)}")

    c = report["cusip_coverage"]
    u = report["universe"]
    a = report["amount_outstanding"]
    r = report["ratings"]
    s = report["survivorship"]
    lines = [
        "# FISD reference build — coverage & quality report",
        "",
        f"_Generated {report['run_timestamp']}; thresholds {report['thresholds_sha256'][:12]}…_",
        "",
        "## CUSIP coverage",
        f"- panel distinct cusips: **{c['panel_cusips']:,}**",
        f"- matched in FISD issue table: **{c['matched_in_fisd']:,}** ({c['match_pct']:.1f}%)",
        "",
        "## Universe restriction (FISD issue table)",
        f"- total FISD cusips: {u['total_fisd_cusips']:,}",
        f"- universe-eligible: **{u['eligible']:,}**",
        f"- eligible ∩ panel: **{u['eligible_in_panel']:,}**",
        f"- eligible bond-months on dev grid: **{u['eligible_bond_months']:,}** "
        f"(panel total {u['panel_bond_months']:,}; OSBAP reference ≈1.46M)",
        "- exclusions by rule (counts over panel-matched cusips):",
        *[f"    - `{k}`: {v:,}" for k, v in u["exclusions_in_panel"].items()],
        "",
        "## amount_outstanding usability (eligible ∩ panel)",
        f"- zero: {a['pct_zero']:.1f}% · negative: {a['pct_negative']:.1f}% · "
        f"null: {a['pct_null']:.1f}% · == offering_amt: {a['pct_equals_offering']:.1f}%",
        f"- offering_amt null: {a['offering_amt_pct_null']:.1f}%  → size proxy = "
        f"`{report['size_proxy']}`",
        "",
        "## Ratings (monthly as-of, dev grid)",
        f"- grid rows: {r['grid_rows']:,} · rated: **{r['rated_rows']:,}** ({r['rated_pct']:.1f}%)",
        "- agency used: " + ", ".join(f"{k}={v:,}" for k, v in r["agency_used"].items()),
        f"- investment grade / high yield (rated rows): {r['ig_rows']:,} / {r['hy_rows']:,}",
        f"- rating events dropped by date hygiene: {r['dropped_rating_dates']:,}",
        f"- **future-date leakage: {r['future_date_leakage']}** (must be 0)",
        "",
        "## Survivorship exit dates (eligible ∩ panel)",
        f"- with maturity: {s['with_maturity']:,} · defaulted: {s['defaulted']:,} · "
        f"defeased: {s['defeased']:,} (of {s['eligible_cusips']:,} eligible cusips)",
        "- calls: **undateable in FISD** — called bonds not flagged terminal (documented)",
        "",
        "## Decisions this report tees up (confirm before Phase 1 merge)",
        "- size proxy: `offering_amt` vs reconstructed `amount_outstanding`",
        "- zero-coupon (`Z`) & 144A inclusion; exact `bond_type` keep-list (DRR match)",
        "- rating agency averaging set (S&P + Moody's; Fitch/DBRS excluded)",
    ]
    tmp_md = REPORT_MD.with_suffix(".md.tmp")
    tmp_md.write_text("\n".join(lines) + "\n")
    os.replace(tmp_md, REPORT_MD)
    print(f"  Summary: {REPORT_MD.relative_to(REPO_ROOT)}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    for f in (ISSUE_FILE, RATINGS_FILE, ISSUER_FILE, REDEMPTION_FILE, PANEL_FILE):
        if not f.exists():
            print(f"ERROR: required input not found: {f}", file=sys.stderr)
            sys.exit(1)

    cfg = load_config()
    print(f"Config: average_agencies={cfg['rating']['average_agencies']}, "
          f"size_proxy={cfg['amount_outstanding']['size_proxy']}")

    print("Building static reference (universe + carried facts)...")
    static = build_static(cfg)
    issue_to_cusip = (
        static.dropna(subset=["issue_id"])
        .set_index("issue_id")["cusip"]
    )
    issue_to_cusip = issue_to_cusip[~issue_to_cusip.index.duplicated(keep="first")]

    print(f"Loading dev grid: {PANEL_FILE.relative_to(REPO_ROOT)}")
    grid = pd.read_parquet(PANEL_FILE, columns=["cusip", "date"])
    grid["cusip"] = grid["cusip"].astype("string")

    print("Building monthly as-of rating panel...")
    ratings_monthly = build_ratings_monthly(cfg, grid, issue_to_cusip)

    # ---- report ----
    panel_cusips = set(grid["cusip"].unique())
    fisd_cusips = set(static["cusip"].unique())
    matched = panel_cusips & fisd_cusips
    in_panel = static[static["cusip"].isin(panel_cusips)]
    eligible_in_panel_cusips = set(
        in_panel.loc[in_panel["universe_eligible"], "cusip"].unique()
    )
    grid_eligible = grid["cusip"].isin(eligible_in_panel_cusips)

    ao = in_panel.loc[in_panel["universe_eligible"], "amount_outstanding"]
    off = in_panel.loc[in_panel["universe_eligible"], "offering_amt"]
    n_ao = max(1, len(ao))
    equal_off = (
        in_panel.loc[in_panel["universe_eligible"]]
        .pipe(lambda d: (d["amount_outstanding"] == d["offering_amt"]).mean()) * 100
    )

    leakage = int(
        (
            ratings_monthly["_sel_rating_date"].notna()
            & (ratings_monthly["_sel_rating_date"] > ratings_monthly["date"])
        ).sum()
    )
    agency_used = (
        ratings_monthly.loc[ratings_monthly["is_rated"], "rating_agency_used"]
        .value_counts().to_dict()
    )
    n_rated = int(ratings_monthly["is_rated"].sum())
    ig_rows = int((ratings_monthly["investment_grade"] == True).sum())  # noqa: E712

    elig = in_panel[in_panel["universe_eligible"]]
    survivorship = {
        "eligible_cusips": int(len(elig)),
        "with_maturity": int(elig["maturity"].notna().sum()),
        "defaulted": int(elig["default_date"].notna().sum()),
        "defeased": int(elig["defeased_date"].notna().sum()),
        "calls_undateable": True,
    }

    report = {
        "run_timestamp": datetime.now(timezone.utc).isoformat(),
        "thresholds_sha256": thresholds_sha256(),
        "fisd_config": cfg,
        "size_proxy": cfg["amount_outstanding"]["size_proxy"],
        "inputs": {
            "issue": str(ISSUE_FILE.relative_to(REPO_ROOT)),
            "ratings": str(RATINGS_FILE.relative_to(REPO_ROOT)),
            "panel_grid": str(PANEL_FILE.relative_to(REPO_ROOT)),
        },
        "cusip_coverage": {
            "panel_cusips": len(panel_cusips),
            "matched_in_fisd": len(matched),
            "match_pct": 100 * len(matched) / max(1, len(panel_cusips)),
        },
        "universe": {
            "total_fisd_cusips": int(len(static)),
            "eligible": int(static["universe_eligible"].sum()),
            "eligible_in_panel": len(eligible_in_panel_cusips),
            "panel_bond_months": int(len(grid)),
            "eligible_bond_months": int(grid_eligible.sum()),
            "exclusions_in_panel": {
                flag: int(in_panel[flag].sum()) for flag in _REASON_FLAGS
            },
        },
        "amount_outstanding": {
            "pct_zero": float((ao == 0).sum() / n_ao * 100),
            "pct_negative": float((ao < 0).sum() / n_ao * 100),
            "pct_null": float(ao.isna().sum() / n_ao * 100),
            "pct_equals_offering": float(equal_off),
            "offering_amt_pct_null": float(off.isna().sum() / max(1, len(off)) * 100),
        },
        "ratings": {
            "grid_rows": int(len(ratings_monthly)),
            "rated_rows": n_rated,
            "rated_pct": 100 * n_rated / max(1, len(ratings_monthly)),
            "agency_used": {str(k): int(v) for k, v in agency_used.items()},
            "ig_rows": ig_rows,
            "hy_rows": n_rated - ig_rows,
            "dropped_rating_dates": int(ratings_monthly.attrs.get("n_dropped_rating_dates", 0)),
            "future_date_leakage": leakage,
        },
        "survivorship": survivorship,
    }

    # ---- write outputs ----
    _write_parquet(
        static, STATIC_OUT,
        {"artifact": "fisd_reference_static", "primary_key": "cusip",
         "source": "WRDS_Mergent_FISD", "step": "preprocessing_only"},
    )
    ratings_final = ratings_monthly.drop(columns=["_sel_rating_date"])
    _write_parquet(
        ratings_final, RATINGS_OUT,
        {"artifact": "fisd_ratings_monthly", "primary_key": "cusip+date",
         "grid": "dev_monthly_panel_maximal", "asof": "backward_no_lookahead"},
    )
    _write_report(report)

    print("\nDone.")
    print(f"  cusip match: {report['cusip_coverage']['match_pct']:.1f}%  "
          f"eligible∩panel: {report['universe']['eligible_in_panel']:,} cusips  "
          f"eligible bond-months: {report['universe']['eligible_bond_months']:,}")
    print(f"  ratings: {report['ratings']['rated_pct']:.1f}% rated  "
          f"future-date leakage: {leakage}")
    if leakage != 0:
        print("  ERROR: future-date leakage detected — as-of join is broken.", file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()
