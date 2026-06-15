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
FISD_DIR = REPO_ROOT / "data" / "fisd"
ISSUE_FILE = FISD_DIR / "reference_fisd_mergedissue.parquet"
RATINGS_FILE = FISD_DIR / "reference_fisd_ratings.parquet"
ISSUER_FILE = FISD_DIR / "reference_fisd_mergedissuer.parquet"
REDEMPTION_FILE = FISD_DIR / "reference_fisd_mergedredemption.parquet"

PANEL_FILE = REPO_ROOT / "data" / "development" / "monthly_panel_maximal.parquet"

OUT_DIR = REPO_ROOT / "data" / "development" / "fisd"
STATIC_OUT = OUT_DIR / "fisd_reference_static.parquet"
RATINGS_OUT = OUT_DIR / "fisd_ratings_monthly.parquet"
REPORT_JSON = OUT_DIR / "fisd_reference_report.json"
REPORT_MD = OUT_DIR / "fisd_reference_report.md"

THRESHOLDS_FILE = REPO_ROOT / "docs" / "thresholds.yaml"

# Columns pulled from the (224-column) issue table. Keep this list tight.
_ISSUE_COLS = [
    "complete_cusip", "issue_id", "issuer_id",
    "currency", "convertible", "asset_backed", "rule_144a", "perpetual",
    "coupon_type", "bond_type", "preferred_security",
    "coupon", "day_count_basis", "maturity",
    "offering_amt", "amount_outstanding",
    "defeased_date",
]

# One boolean reason flag per universe rule. True = this rule EXCLUDED the bond.
_REASON_FLAGS = [
    "excl_currency", "excl_convertible", "excl_asset_backed", "excl_144a",
    "excl_perpetual", "excl_coupon_type", "excl_bond_type",
]


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def load_config() -> dict:
    """Load the `fisd:` block from thresholds.yaml. No defaults — every
    required sub-block must be present or the script refuses to start."""
    with open(THRESHOLDS_FILE) as f:
        cfg = yaml.safe_load(f)
    block = cfg.get("fisd")
    if block is None:
        raise KeyError("thresholds.yaml is missing the `fisd:` block")
    for key in ("universe", "rating", "rating_numeric_map", "amount_outstanding"):
        if key not in block:
            raise KeyError(f"thresholds.yaml fisd block missing '{key}'")
    for key in ("agency_priority", "not_rated_tokens", "withdrawn_status",
                "date_min", "ig_threshold"):
        if key not in block["rating"]:
            raise KeyError(f"thresholds.yaml fisd.rating missing '{key}'")
    if not {"sp", "moody"} <= set(block["rating_numeric_map"]):
        raise KeyError("thresholds.yaml fisd.rating_numeric_map needs sp + moody")
    return block


def thresholds_sha256() -> str:
    return hashlib.sha256(THRESHOLDS_FILE.read_bytes()).hexdigest()


# ---------------------------------------------------------------------------
# Universe restriction (pure, testable)
# ---------------------------------------------------------------------------

def apply_universe_rules(issue: pd.DataFrame, rules: dict) -> pd.DataFrame:
    """Add one boolean reason flag per rule + `universe_eligible` to `issue`.

    Each `excl_*` flag is True when that rule excludes the bond. A bond is
    eligible iff no rule excludes it. Missing/NaN categorical fields fail the
    relevant inclusion rule (conservative — we never assert an unknown is
    corporate/fixed/USD).
    """
    out = issue.copy()

    def col(name):
        return out[name] if name in out.columns else pd.Series(pd.NA, index=out.index)

    currency = col("currency")
    allow = set(rules["currency_allow"])
    cur_ok = currency.isin(allow) | (currency.isna() & bool(rules["allow_null_currency"]))
    out["excl_currency"] = ~cur_ok

    out["excl_convertible"] = bool(rules["exclude_convertible"]) & (col("convertible") == "Y")
    out["excl_asset_backed"] = bool(rules["exclude_asset_backed"]) & (col("asset_backed") == "Y")
    out["excl_144a"] = bool(rules["exclude_144a"]) & (col("rule_144a") == "Y")
    out["excl_perpetual"] = bool(rules["exclude_perpetual"]) & (col("perpetual") == "Y")
    out["excl_coupon_type"] = ~col("coupon_type").isin(set(rules["coupon_type_allow"]))
    out["excl_bond_type"] = ~col("bond_type").isin(set(rules["bond_type_keep"]))

    out["universe_eligible"] = ~out[_REASON_FLAGS].any(axis=1)
    return out


# ---------------------------------------------------------------------------
# Rating letter → numeric (pure, testable)
# ---------------------------------------------------------------------------

def rating_to_numeric(rating, agency: str, numeric_map: dict, not_rated_tokens) -> float:
    """Map a single rating string to the unified 1–22 numeric ladder.

    Moody's (agency 'MR') uses the `moody` ladder; everyone else (S&P 'SPR',
    Fitch 'FR') uses the S&P-style `sp` ladder. NR/blank/unmapped → NaN.
    """
    if rating is None or (isinstance(rating, float) and np.isnan(rating)):
        return np.nan
    if rating in set(not_rated_tokens):
        return np.nan
    table = numeric_map["moody"] if agency == "MR" else numeric_map["sp"]
    val = table.get(rating)
    return float(val) if val is not None else np.nan


def map_rating_events(ratings: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Vectorised numeric mapping over a rating-events frame.

    Expects columns: rating, rating_type, rating_status. Adds `rating_numeric`
    (NaN for NR, withdrawn, or unmapped). Returns a copy with the new column.
    """
    out = ratings.copy()
    numeric_map = cfg["rating_numeric_map"]
    not_rated = set(cfg["rating"]["not_rated_tokens"])
    withdrawn = set(cfg["rating"]["withdrawn_status"])

    numeric = [
        rating_to_numeric(r, a, numeric_map, not_rated)
        for r, a in zip(out["rating"].tolist(), out["rating_type"].tolist())
    ]
    out["rating_numeric"] = numeric
    # A withdrawal event ends the prior rating: keep the event (so the as-of
    # join sees it) but with a null numeric.
    out.loc[out["rating_status"].isin(withdrawn), "rating_numeric"] = np.nan
    return out


# ---------------------------------------------------------------------------
# As-of monthly rating (pure, testable)
# ---------------------------------------------------------------------------

def asof_monthly_rating(
    events: pd.DataFrame,
    grid: pd.DataFrame,
    priority: list,
    ig_threshold: int,
) -> pd.DataFrame:
    """Coalesce per-agency as-of ratings onto a (cusip, date) grid.

    `events` columns: cusip, rating_date (datetime), rating_numeric (float,
    NaN for withdrawn/NR), agency (str). For each (cusip, month-end) the latest
    event with rating_date <= month-end per agency is taken (backward as-of, no
    look-ahead); agencies are coalesced in `priority` order, first non-null
    wins. Returns grid + rating_numeric, rating_agency_used, investment_grade,
    is_rated, _sel_rating_date (the winning event date, for leakage checks).
    """
    grid = grid[["cusip", "date"]].copy()
    # merge_asof requires both date keys to share a datetime RESOLUTION. The
    # FISD parquet parses rating_date to datetime64[us] while the panel grid is
    # datetime64[ns]; normalise both to [ns] (the [us]/[ns] self-merge trap).
    grid["date"] = grid["date"].astype("datetime64[ns]")
    grid = grid.sort_values("date", kind="mergesort").reset_index(drop=True)
    result = grid.copy()
    result["rating_numeric"] = np.nan
    result["rating_agency_used"] = pd.Series([None] * len(result), dtype="object")
    result["_sel_rating_date"] = pd.Series(pd.NaT, index=result.index, dtype="datetime64[ns]")

    for ag in priority:
        ev = events[events["agency"] == ag][["cusip", "rating_date", "rating_numeric"]].copy()
        ev["rating_date"] = ev["rating_date"].astype("datetime64[ns]")
        ev = ev.dropna(subset=["rating_date"]).sort_values("rating_date", kind="mergesort")
        if ev.empty:
            continue
        merged = pd.merge_asof(
            grid, ev, left_on="date", right_on="rating_date",
            by="cusip", direction="backward",
        )
        take = result["rating_numeric"].isna() & merged["rating_numeric"].notna()
        result.loc[take, "rating_numeric"] = merged.loc[take, "rating_numeric"].to_numpy()
        result.loc[take, "rating_agency_used"] = ag
        result.loc[take, "_sel_rating_date"] = merged.loc[take, "rating_date"].to_numpy()

    result["is_rated"] = result["rating_numeric"].notna()
    ig = result["rating_numeric"] <= float(ig_threshold)
    result["investment_grade"] = ig.where(result["is_rated"], other=pd.NA).astype("boolean")
    return result


# ---------------------------------------------------------------------------
# Loaders / assembly
# ---------------------------------------------------------------------------

def _load_issue() -> pd.DataFrame:
    df = pd.read_parquet(ISSUE_FILE, columns=_ISSUE_COLS)
    # Force the join key to string and preserve leading zeros (CUSIPs like
    # 000361AB1 must not be coerced to int) — same care as preprocess_trace.py.
    df["complete_cusip"] = df["complete_cusip"].astype("string")
    df = df.dropna(subset=["complete_cusip"])
    df = df.drop_duplicates(subset="complete_cusip", keep="first").reset_index(drop=True)
    df["issue_id"] = df["issue_id"].astype("Int64")
    # Normalise to [ns] so a later merge/compare against the [ns] panel never
    # hits the datetime-resolution mismatch that breaks merge_asof.
    df["maturity"] = pd.to_datetime(df["maturity"], errors="coerce").astype("datetime64[ns]")
    df["defeased_date"] = pd.to_datetime(df["defeased_date"], errors="coerce").astype("datetime64[ns]")
    return df


def _default_dates_by_issue(cfg: dict) -> pd.Series:
    """Earliest rating event in the default-token bucket, per issue_id.

    FISD has no default-date field; the rating ladder's default bucket (D/SD/…)
    dates a default by the first time the bond is rated into it. Returns a
    Series indexed by issue_id (Int64) → first default rating_date (datetime).
    """
    tokens = cfg.get("survivorship", {}).get("default_tokens")
    if not tokens:
        raise KeyError("thresholds.yaml fisd.survivorship.default_tokens is required")
    r = pd.read_parquet(RATINGS_FILE, columns=["issue_id", "rating", "rating_date"])
    r = r[r["rating"].isin(set(tokens))].copy()
    r["issue_id"] = r["issue_id"].astype("Int64")
    r["rating_date"] = pd.to_datetime(r["rating_date"], errors="coerce").astype("datetime64[ns]")
    r = r.dropna(subset=["rating_date"])
    return r.groupby("issue_id")["rating_date"].min()


def build_static(cfg: dict) -> pd.DataFrame:
    """One row per cusip: universe eligibility + reason flags + carried facts."""
    issue = _load_issue()
    issue = apply_universe_rules(issue, cfg["universe"])

    # callable flag from the redemption schedule (issue_id level).
    redemption = pd.read_parquet(REDEMPTION_FILE, columns=["issue_id", "callable"])
    redemption["issue_id"] = redemption["issue_id"].astype("Int64")
    callable_by_issue = (
        redemption.assign(_c=redemption["callable"].eq("Y"))
        .groupby("issue_id")["_c"].any()
    )
    issue["callable"] = issue["issue_id"].map(callable_by_issue).astype("boolean")

    # issuer attributes (issuer_id level).
    issuer = pd.read_parquet(ISSUER_FILE, columns=["issuer_id", "sic_code", "country_domicile"])
    issuer["issuer_id"] = issuer["issuer_id"].astype("Int64")
    issuer = issuer.drop_duplicates(subset="issuer_id", keep="first")
    issue["issuer_id"] = issue["issuer_id"].astype("Int64")
    issue = issue.merge(issuer, on="issuer_id", how="left")

    # Survivorship exit dates: default (from the ratings default bucket) +
    # defeased (carried). maturity is already on `issue`. Calls are undateable
    # in FISD (see thresholds fisd.survivorship note).
    default_dates = _default_dates_by_issue(cfg)
    issue["default_date"] = issue["issue_id"].map(default_dates).astype("datetime64[ns]")

    keep = [
        "complete_cusip", "issue_id", "issuer_id",
        "universe_eligible", *_REASON_FLAGS,
        "bond_type", "coupon_type", "convertible", "asset_backed",
        "rule_144a", "perpetual",
        "offering_amt", "amount_outstanding",
        "maturity", "default_date", "defeased_date", "coupon", "day_count_basis",
        "callable", "sic_code", "country_domicile",
    ]
    static = issue[keep].rename(columns={"complete_cusip": "cusip"})
    static["cusip"] = static["cusip"].astype("string")
    return static


def build_ratings_monthly(cfg: dict, grid: pd.DataFrame, issue_to_cusip: pd.Series) -> pd.DataFrame:
    """Build the (cusip, date) monthly as-of rating panel on the dev grid."""
    priority = list(cfg["rating"]["agency_priority"])
    date_min = pd.Timestamp(cfg["rating"]["date_min"])
    grid_max = pd.Timestamp(grid["date"].max())

    ratings = pd.read_parquet(
        RATINGS_FILE, columns=["issue_id", "rating", "rating_type", "rating_date", "rating_status"]
    )
    ratings = ratings[ratings["rating_type"].isin(priority)].copy()
    ratings["issue_id"] = ratings["issue_id"].astype("Int64")
    ratings["cusip"] = ratings["issue_id"].map(issue_to_cusip).astype("string")
    ratings = ratings.dropna(subset=["cusip"])
    # Restrict to cusips that appear on the grid (bounds size; nothing else can match).
    grid_cusips = set(grid["cusip"].astype("string").unique())
    ratings = ratings[ratings["cusip"].isin(grid_cusips)]

    # rating_date hygiene: parse, drop unparseable, clamp to [date_min, grid_max].
    # Events after grid_max are future relative to every dev month and are
    # dropped — this also removes the implausible-future garbage (e.g. 2091).
    ratings["rating_date"] = pd.to_datetime(ratings["rating_date"], errors="coerce")
    n_pre = len(ratings)
    ratings = ratings.dropna(subset=["rating_date"])
    ratings = ratings[(ratings["rating_date"] >= date_min) & (ratings["rating_date"] <= grid_max)]
    n_dropped_dates = n_pre - len(ratings)

    ratings = map_rating_events(ratings, cfg)
    ratings = ratings.rename(columns={"rating_type": "agency"})

    out = asof_monthly_rating(
        ratings[["cusip", "rating_date", "rating_numeric", "agency"]],
        grid, priority, int(cfg["rating"]["ig_threshold"]),
    )
    out.attrs["n_dropped_rating_dates"] = int(n_dropped_dates)
    return out


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
        f"- agency used: " + ", ".join(f"{k}={v:,}" for k, v in r["agency_used"].items()),
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
        "- rating agency priority",
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
    print(f"Config: agency_priority={cfg['rating']['agency_priority']}, "
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
