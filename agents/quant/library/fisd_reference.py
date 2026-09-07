"""
FISD reference-building workers, lifted verbatim from
``scripts/build_fisd_reference.py`` so ``agents/`` code (the one-shot holdout dev-pseudo panel
builder) can import them without importing a script. The thin script keeps only
the CLI, the parquet/report writers, and ``main``.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from shared.licensed_inputs import require_licensed_input

REPO_ROOT = Path(__file__).resolve().parents[3]
FISD_DIR = REPO_ROOT / "data" / "fisd"
ISSUE_FILE = FISD_DIR / "reference_fisd_mergedissue.parquet"
RATINGS_FILE = FISD_DIR / "reference_fisd_ratings.parquet"
ISSUER_FILE = FISD_DIR / "reference_fisd_mergedissuer.parquet"
REDEMPTION_FILE = FISD_DIR / "reference_fisd_mergedredemption.parquet"
THRESHOLDS_FILE = REPO_ROOT / "docs" / "thresholds.yaml"

_ISSUE_COLS = [
    "complete_cusip", "issue_id", "issuer_id",
    "currency", "convertible", "asset_backed", "rule_144a", "perpetual",
    "coupon_type", "bond_type", "preferred_security",
    "coupon", "day_count_basis", "maturity", "interest_frequency",
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
    for key in ("average_agencies", "not_rated_tokens", "withdrawn_status",
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

    Moody's (agency 'MR') uses the `moody` ladder; any other code uses the
    S&P-style `sp` ladder. In this pipeline only S&P ('SPR') ever takes the
    non-Moody's branch — Fitch ('FR') and DBRS ('DPR') events are filtered out
    at load (`_build_ratings`, the `average_agencies` `.isin` filter) and never
    reach this function, so the `else` branch is a generic fallback, not a live
    Fitch/DBRS path. NR/blank/unmapped → NaN.
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
    agencies: list,
    ig_threshold: int,
) -> pd.DataFrame:
    """Average per-agency as-of ratings onto a (cusip, date) grid.

    BBW/DRR convention (docs/quant/specs/BBW_anchor_implementation_spec.md §2.3): rating_numeric
    is the numeric MEAN of the available agency ratings in `agencies` (S&P
    'SPR' and Moody's 'MR'), with single-agency fallback when only one is
    available and NaN when neither is. Any agency not in `agencies` (Fitch
    'FR', DBRS 'DPR') is excluded entirely. This deliberately MATCHES BBW
    rather than the prior first-non-null priority pick (audit trail: spec §2.3).

    `events` columns: cusip, rating_date (datetime), rating_numeric (float,
    NaN for withdrawn/NR), agency (str). For each (cusip, month-end) the latest
    event with rating_date <= month-end per agency is taken (backward as-of, no
    look-ahead), then averaged across agencies. A withdrawal event carries a
    non-null rating_date but NaN numeric, so it correctly advances the as-of
    pointer while contributing nothing to the average ("currently not rated by
    this agency").

    Notch alignment: the sp/moody ladders coincide notch-for-notch 1..21; only
    S&P's D=22 lacks a Moody's equivalent. The naive mean therefore diverges
    from OSBAP `comp_rating` ONLY at single-agency defaults (S&P-only D → 22 and
    Moody's-only C → 21 here, vs comp_rating's composite-fill 21.5; both-rated
    D/C coincides at 21.5). This is a CONDITIONAL assertion: harmless ONLY
    because the universe excludes defaulted bonds, so these rows never enter a
    sort. It is enforced by `test_asof_bottom_notch_default_behavior` — if the
    universe filter is ever changed to admit defaults, that tripwire fails and
    the bottom notch must be reconciled against DRR's factor-construction
    appendix (spec §11.1). Everything above C is already aligned.

    Returns grid + rating_numeric, rating_agency_used ('SPR+MR' | 'SPR' | 'MR'
    | None), investment_grade, is_rated, _sel_rating_date (the LATEST
    contributing event date, the most stringent value for the leakage check).
    """
    grid = grid[["cusip", "date"]].copy()
    # merge_asof requires both date keys to share a datetime RESOLUTION. The
    # FISD parquet parses rating_date to datetime64[us] while the panel grid is
    # datetime64[ns]; normalise both to [ns] (the [us]/[ns] self-merge trap).
    grid["date"] = grid["date"].astype("datetime64[ns]")
    grid = grid.sort_values("date", kind="mergesort").reset_index(drop=True)
    result = grid.copy()

    # Per-agency as-of numeric + selected event date, as parallel columns
    # aligned to result's row order.
    num_cols: dict[str, np.ndarray] = {}
    date_cols: dict[str, np.ndarray] = {}
    for ag in agencies:
        ev = events[events["agency"] == ag][["cusip", "rating_date", "rating_numeric"]].copy()
        ev["rating_date"] = ev["rating_date"].astype("datetime64[ns]")
        ev = ev.dropna(subset=["rating_date"]).sort_values("rating_date", kind="mergesort")
        if ev.empty:
            num_cols[ag] = np.full(len(result), np.nan)
            date_cols[ag] = np.full(len(result), np.datetime64("NaT", "ns"))
            continue
        merged = pd.merge_asof(
            grid, ev, left_on="date", right_on="rating_date",
            by="cusip", direction="backward",
        )
        num_cols[ag] = merged["rating_numeric"].to_numpy(dtype=float)
        date_cols[ag] = merged["rating_date"].to_numpy()

    num_df = pd.DataFrame(num_cols, index=result.index)
    contrib = num_df.notna()  # which agencies gave a live (non-withdrawn) rating

    # Row-wise mean of the available agency ratings; all-NaN row → NaN.
    result["rating_numeric"] = num_df.mean(axis=1, skipna=True)

    # agency_used label, vectorised via a binary code over the agency columns
    # (e.g. SPR=bit0, MR=bit1) so there is no per-row Python apply on the full
    # grid. code 0 → None (unrated).
    weights = (1 << np.arange(len(agencies)))
    codes = contrib.to_numpy().astype(np.int64) @ weights
    label_map: dict[int, str | None] = {0: None}
    for code in range(1, 1 << len(agencies)):
        used = [agencies[i] for i in range(len(agencies)) if (code >> i) & 1]
        label_map[code] = "+".join(used)
    result["rating_agency_used"] = pd.Series(codes, index=result.index).map(label_map)

    # _sel_rating_date = latest CONTRIBUTING event date (dates of withdrawn /
    # absent agencies are masked out). All contributing dates are <= month-end
    # by the backward as-of, so the max is still <= month-end — the leakage
    # check stays exact.
    date_df = pd.DataFrame(date_cols, index=result.index).where(contrib)
    result["_sel_rating_date"] = date_df.max(axis=1)

    result["is_rated"] = result["rating_numeric"].notna()
    ig = result["rating_numeric"] <= float(ig_threshold)
    result["investment_grade"] = ig.where(result["is_rated"], other=pd.NA).astype("boolean")
    return result


# ---------------------------------------------------------------------------
# Loaders / assembly
# ---------------------------------------------------------------------------

def _load_issue() -> pd.DataFrame:
    df = pd.read_parquet(require_licensed_input(ISSUE_FILE, "FISD issue table"), columns=_ISSUE_COLS)
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
    date_min = pd.Timestamp(cfg["rating"]["date_min"])
    r = pd.read_parquet(require_licensed_input(RATINGS_FILE, "FISD ratings table"), columns=["issue_id", "rating", "rating_date"])
    r = r[r["rating"].isin(set(tokens))].copy()
    r["issue_id"] = r["issue_id"].astype("Int64")
    r["rating_date"] = pd.to_datetime(r["rating_date"], errors="coerce").astype("datetime64[ns]")
    r = r.dropna(subset=["rating_date"])
    # Exit-date hygiene: drop implausibly early default events (same date_min
    # floor the ratings panel uses) so a garbage date can't make a bond's whole
    # history terminal in the survivorship toggle.
    r = r[r["rating_date"] >= date_min]
    return r.groupby("issue_id")["rating_date"].min()


def build_static(cfg: dict) -> pd.DataFrame:
    """One row per cusip: universe eligibility + reason flags + carried facts."""
    issue = _load_issue()
    # Exit-date hygiene: floor maturity / defeased_date at date_min so a
    # parseable-but-implausibly-early date can't mislabel a bond's exit_reason.
    date_min = pd.Timestamp(cfg["rating"]["date_min"])
    for col in ("maturity", "defeased_date"):
        issue.loc[issue[col] < date_min, col] = pd.NaT
    issue = apply_universe_rules(issue, cfg["universe"])

    # callable flag from the redemption schedule (issue_id level).
    redemption = pd.read_parquet(require_licensed_input(REDEMPTION_FILE, "FISD redemption table"), columns=["issue_id", "callable"])
    redemption["issue_id"] = redemption["issue_id"].astype("Int64")
    callable_by_issue = (
        redemption.assign(_c=redemption["callable"].eq("Y"))
        .groupby("issue_id")["_c"].any()
    )
    issue["callable"] = issue["issue_id"].map(callable_by_issue).astype("boolean")

    # issuer attributes (issuer_id level).
    issuer = pd.read_parquet(require_licensed_input(ISSUER_FILE, "FISD issuer table"), columns=["issuer_id", "sic_code", "country_domicile"])
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
        "interest_frequency", "callable", "sic_code", "country_domicile",
    ]
    static = issue[keep].rename(columns={"complete_cusip": "cusip"})
    static["cusip"] = static["cusip"].astype("string")
    return static


def build_ratings_monthly(cfg: dict, grid: pd.DataFrame, issue_to_cusip: pd.Series) -> pd.DataFrame:
    """Build the (cusip, date) monthly as-of rating panel on the dev grid."""
    agencies = list(cfg["rating"]["average_agencies"])
    date_min = pd.Timestamp(cfg["rating"]["date_min"])
    grid_max = pd.Timestamp(grid["date"].max())

    ratings = pd.read_parquet(
        require_licensed_input(RATINGS_FILE, "FISD ratings table"),
        columns=["issue_id", "rating", "rating_type", "rating_date", "rating_status"],
    )
    ratings = ratings[ratings["rating_type"].isin(agencies)].copy()
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
        grid, agencies, int(cfg["rating"]["ig_threshold"]),
    )
    out.attrs["n_dropped_rating_dates"] = int(n_dropped_dates)
    return out
