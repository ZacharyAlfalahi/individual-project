"""
Unit tests for the FISD reference preprocessing build.

Covers:
  - Universe rules: a base-eligible bond passes; flipping each field on its
    own excludes it via the matching reason flag.
  - Rating map: S&P and Moody's notches map to the same numeric ladder; NR
    and unmapped strings → NaN; withdrawn events → NaN numeric.
  - As-of monthly rating: backward-only (no look-ahead), S&P+Moody's averaging
    with single-agency fallback, Fitch excluded, withdrawal ends a rating,
    IG/HY threshold.
  - CUSIP integrity: a leading-zero cusip survives the loader as a string.
  - Config validation: missing fisd block / required keys raise KeyError.
  - Real-data smoke (skipped if data/fisd/ absent): sane universe count and
    zero future-date leakage on a sampled dev grid.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import build_fisd_reference as bfr  # noqa: E402

# The input-file / thresholds constants and the workers now live in the library
# (agents/quant/library/fisd_reference.py); the script re-exports them. Tests
# that monkeypatch those module-level constants must patch them where the moved
# functions READ them — the library module — not the thin script wrapper.
import agents.quant.library.fisd_reference as fref  # noqa: E402


@pytest.fixture(scope="module")
def cfg():
    return bfr.load_config()


# ---------------------------------------------------------------------------
# Universe rules
# ---------------------------------------------------------------------------

def _base_issue_row():
    """A bond that should pass every committed universe rule."""
    return {
        "currency": "USD", "convertible": "N", "asset_backed": "N",
        "rule_144a": "N", "perpetual": "N", "coupon_type": "F",
        "bond_type": "CDEB",
    }


def test_base_bond_is_eligible(cfg):
    df = pd.DataFrame([_base_issue_row()])
    out = bfr.apply_universe_rules(df, cfg["universe"])
    assert bool(out["universe_eligible"].iloc[0]) is True
    assert not out[bfr._REASON_FLAGS].iloc[0].any()


def test_null_currency_is_allowed(cfg):
    row = _base_issue_row(); row["currency"] = None
    out = bfr.apply_universe_rules(pd.DataFrame([row]), cfg["universe"])
    assert bool(out["universe_eligible"].iloc[0]) is True
    assert not bool(out["excl_currency"].iloc[0])


@pytest.mark.parametrize("field,value,flag", [
    ("currency", "EUR", "excl_currency"),
    ("convertible", "Y", "excl_convertible"),
    ("asset_backed", "Y", "excl_asset_backed"),
    ("rule_144a", "Y", "excl_144a"),
    ("perpetual", "Y", "excl_perpetual"),
    ("coupon_type", "V", "excl_coupon_type"),   # floating
    ("bond_type", "USBN", "excl_bond_type"),    # agency/government
])
def test_each_rule_excludes_independently(cfg, field, value, flag):
    row = _base_issue_row(); row[field] = value
    out = bfr.apply_universe_rules(pd.DataFrame([row]), cfg["universe"])
    assert bool(out[flag].iloc[0]) is True, f"{field}={value} should set {flag}"
    assert bool(out["universe_eligible"].iloc[0]) is False


def test_missing_categorical_fails_inclusion(cfg):
    """A NaN bond_type / coupon_type fails its inclusion rule (conservative)."""
    row = _base_issue_row(); row["bond_type"] = None; row["coupon_type"] = None
    out = bfr.apply_universe_rules(pd.DataFrame([row]), cfg["universe"])
    assert bool(out["excl_bond_type"].iloc[0]) is True
    assert bool(out["excl_coupon_type"].iloc[0]) is True
    assert bool(out["universe_eligible"].iloc[0]) is False


# ---------------------------------------------------------------------------
# Rating letter → numeric
# ---------------------------------------------------------------------------

def test_rating_map_sp_and_moody_parity(cfg):
    nmap = cfg["rating_numeric_map"]
    nr = cfg["rating"]["not_rated_tokens"]
    # AAA (S&P) and Aaa (Moody's) are the same notch → 1.
    assert bfr.rating_to_numeric("AAA", "SPR", nmap, nr) == 1.0
    assert bfr.rating_to_numeric("Aaa", "MR", nmap, nr) == 1.0
    # BBB- (S&P) and Baa3 (Moody's) are the IG boundary → same numeric.
    assert bfr.rating_to_numeric("BBB-", "SPR", nmap, nr) == \
        bfr.rating_to_numeric("Baa3", "MR", nmap, nr)
    # rating_to_numeric is a pure map: a non-Moody's code (here 'FR') resolves
    # on the S&P ladder. NOTE this path is unused in the build — Fitch/DBRS
    # events are filtered out at load (_build_ratings) and never reach the
    # average; the assertion just documents the fallback, it is not a live path.
    assert bfr.rating_to_numeric("AA+", "FR", nmap, nr) == \
        bfr.rating_to_numeric("AA+", "SPR", nmap, nr)


def test_rating_map_nr_and_unmapped_are_nan(cfg):
    nmap = cfg["rating_numeric_map"]
    nr = cfg["rating"]["not_rated_tokens"]
    assert np.isnan(bfr.rating_to_numeric("NR", "SPR", nmap, nr))
    assert np.isnan(bfr.rating_to_numeric("WHATEVER", "SPR", nmap, nr))
    assert np.isnan(bfr.rating_to_numeric(np.nan, "MR", nmap, nr))


def test_withdrawn_event_maps_to_nan_numeric(cfg):
    """A withdrawn-status event keeps a row but with NaN numeric, so the
    as-of join treats the bond as unrated from that date."""
    withdrawn = cfg["rating"]["withdrawn_status"][0]
    events = pd.DataFrame([
        {"rating": "AAA", "rating_type": "SPR", "rating_status": None},
        {"rating": "AAA", "rating_type": "SPR", "rating_status": withdrawn},
    ])
    out = bfr.map_rating_events(events, cfg)
    assert out["rating_numeric"].iloc[0] == 1.0
    assert np.isnan(out["rating_numeric"].iloc[1])


# ---------------------------------------------------------------------------
# As-of monthly rating
# ---------------------------------------------------------------------------

def _grid(cusip, months):
    return pd.DataFrame({
        "cusip": cusip,
        "date": [pd.Timestamp(m) + pd.offsets.MonthEnd(0) for m in months],
    })


def test_asof_is_backward_only_no_lookahead():
    events = pd.DataFrame([
        {"cusip": "X", "rating_date": pd.Timestamp("2010-01-15"), "rating_numeric": 6.0, "agency": "SPR"},
        {"cusip": "X", "rating_date": pd.Timestamp("2010-06-15"), "rating_numeric": 9.0, "agency": "SPR"},
        {"cusip": "X", "rating_date": pd.Timestamp("2011-01-15"), "rating_numeric": 3.0, "agency": "SPR"},
    ])
    grid = _grid("X", ["2009-12", "2010-01", "2010-03", "2010-06", "2010-12"])
    out = bfr.asof_monthly_rating(events, grid, ["SPR"], ig_threshold=10).set_index("date")

    assert pd.isna(out.loc[pd.Timestamp("2009-12-31"), "rating_numeric"])   # before any event
    assert out.loc[pd.Timestamp("2010-01-31"), "rating_numeric"] == 6.0     # Jan-15 event in scope
    assert out.loc[pd.Timestamp("2010-03-31"), "rating_numeric"] == 6.0     # carries forward
    assert out.loc[pd.Timestamp("2010-06-30"), "rating_numeric"] == 9.0     # Jun-15 event in scope
    # The 2011-01 upgrade must NOT leak into Dec-2010.
    assert out.loc[pd.Timestamp("2010-12-31"), "rating_numeric"] == 9.0


def test_asof_averages_sp_and_moody_with_single_agency_fallback():
    """BBW/DRR convention (spec §2.3): average S&P + Moody's when both live;
    fall back to the single live agency when the other has withdrawn."""
    events = pd.DataFrame([
        {"cusip": "X", "rating_date": pd.Timestamp("2010-01-15"), "rating_numeric": 6.0, "agency": "SPR"},
        {"cusip": "X", "rating_date": pd.Timestamp("2010-02-15"), "rating_numeric": 7.0, "agency": "MR"},
        {"cusip": "X", "rating_date": pd.Timestamp("2010-06-15"), "rating_numeric": np.nan, "agency": "SPR"},
    ])
    grid = _grid("X", ["2010-03", "2010-12"])
    out = bfr.asof_monthly_rating(events, grid, ["SPR", "MR"], ig_threshold=10).set_index("date")
    # March: both live → average of 6.0 (S&P) and 7.0 (Moody's) = 6.5.
    assert out.loc[pd.Timestamp("2010-03-31"), "rating_numeric"] == 6.5
    assert out.loc[pd.Timestamp("2010-03-31"), "rating_agency_used"] == "SPR+MR"
    # December: S&P latest event is a withdrawal (NaN) → only Moody's contributes.
    assert out.loc[pd.Timestamp("2010-12-31"), "rating_numeric"] == 7.0
    assert out.loc[pd.Timestamp("2010-12-31"), "rating_agency_used"] == "MR"


def test_asof_split_rating_averages_to_half_notch():
    """The headline spec §2.3 example: a BBB (S&P=9) / Baa1 (Moody's=8) bond
    maps to 8.5, not 9 — split ratings straddle a quintile breakpoint."""
    events = pd.DataFrame([
        {"cusip": "S", "rating_date": pd.Timestamp("2010-01-15"), "rating_numeric": 9.0, "agency": "SPR"},
        {"cusip": "S", "rating_date": pd.Timestamp("2010-01-15"), "rating_numeric": 8.0, "agency": "MR"},
    ])
    grid = _grid("S", ["2010-06"])
    out = bfr.asof_monthly_rating(events, grid, ["SPR", "MR"], ig_threshold=10).set_index("date")
    assert out.loc[pd.Timestamp("2010-06-30"), "rating_numeric"] == 8.5
    assert out.loc[pd.Timestamp("2010-06-30"), "rating_agency_used"] == "SPR+MR"


def test_asof_excludes_fitch_from_the_average():
    """Fitch (FR) is dropped: a live FR event must not enter the average even
    when present in the events frame, because it is not in `agencies`."""
    events = pd.DataFrame([
        {"cusip": "F", "rating_date": pd.Timestamp("2010-01-15"), "rating_numeric": 6.0, "agency": "SPR"},
        {"cusip": "F", "rating_date": pd.Timestamp("2010-01-15"), "rating_numeric": 8.0, "agency": "MR"},
        {"cusip": "F", "rating_date": pd.Timestamp("2010-01-15"), "rating_numeric": 2.0, "agency": "FR"},
    ])
    grid = _grid("F", ["2010-06"])
    out = bfr.asof_monthly_rating(events, grid, ["SPR", "MR"], ig_threshold=10).set_index("date")
    # (6 + 8) / 2 = 7.0; the FR=2.0 event is ignored (would give 5.33 if used).
    assert out.loc[pd.Timestamp("2010-06-30"), "rating_numeric"] == 7.0
    assert out.loc[pd.Timestamp("2010-06-30"), "rating_agency_used"] == "SPR+MR"


def test_asof_bottom_notch_default_behavior():
    """TRIPWIRE — pins the one place the S&P (1..22) and Moody's (1..21) ladders
    diverge: the bottom default notch. The build naive-averages the two scales
    (CONFIRM-ON-READ in asof_monthly_rating / thresholds fisd.rating), which is
    harmless ONLY because the universe excludes defaulted bonds, so these values
    never enter a sort. This guard must fail loudly if the averaging logic drifts
    OR if the universe filter is ever changed to admit defaults — at which point
    22/21 here vs OSBAP comp_rating's composite-fill 21.5 starts to matter and
    must be reconciled against the DRR factor-construction appendix. Guard, not
    a target.
    """
    events = pd.DataFrame([
        # Both agencies at their lowest notch: S&P D=22, Moody's C=21.
        {"cusip": "BOTH", "rating_date": pd.Timestamp("2010-01-15"), "rating_numeric": 22.0, "agency": "SPR"},
        {"cusip": "BOTH", "rating_date": pd.Timestamp("2010-01-15"), "rating_numeric": 21.0, "agency": "MR"},
        # S&P-only default.
        {"cusip": "SPONLY", "rating_date": pd.Timestamp("2010-01-15"), "rating_numeric": 22.0, "agency": "SPR"},
        # Moody's-only lowest notch.
        {"cusip": "MRONLY", "rating_date": pd.Timestamp("2010-01-15"), "rating_numeric": 21.0, "agency": "MR"},
    ])
    grid = pd.concat([
        _grid("BOTH", ["2010-06"]),
        _grid("SPONLY", ["2010-06"]),
        _grid("MRONLY", ["2010-06"]),
    ], ignore_index=True)
    out = bfr.asof_monthly_rating(events, grid, ["SPR", "MR"], ig_threshold=10).set_index("cusip")
    # Both-rated D/C → naive mean 21.5 (coincides with OSBAP comp_rating here).
    assert out.loc["BOTH", "rating_numeric"] == 21.5
    # Single-agency default → that agency's own bottom notch (BBW-literal),
    # NOT OSBAP comp_rating's composite-fill 21.5. This is the documented divergence.
    assert out.loc["SPONLY", "rating_numeric"] == 22.0
    assert out.loc["MRONLY", "rating_numeric"] == 21.0


def test_asof_investment_grade_threshold():
    events = pd.DataFrame([
        {"cusip": "IG", "rating_date": pd.Timestamp("2010-01-15"), "rating_numeric": 6.0, "agency": "SPR"},
        {"cusip": "HY", "rating_date": pd.Timestamp("2010-01-15"), "rating_numeric": 12.0, "agency": "SPR"},
    ])
    grid = pd.concat([_grid("IG", ["2010-06"]), _grid("HY", ["2010-06"])], ignore_index=True)
    out = bfr.asof_monthly_rating(events, grid, ["SPR"], ig_threshold=10).set_index("cusip")
    assert bool(out.loc["IG", "investment_grade"]) is True
    assert bool(out.loc["HY", "investment_grade"]) is False
    assert bool(out.loc["IG", "is_rated"]) and bool(out.loc["HY", "is_rated"])


# ---------------------------------------------------------------------------
# CUSIP integrity through the loader
# ---------------------------------------------------------------------------

def test_loader_preserves_leading_zero_cusip(tmp_path, monkeypatch):
    """A CUSIP like 000361AB1 must not be coerced to int (losing leading
    zeros). Writes a minimal issue parquet and reads it through _load_issue."""
    row = {c: None for c in bfr._ISSUE_COLS}
    row.update({
        "complete_cusip": "000361AB1", "issue_id": 42.0, "issuer_id": 7.0,
        "currency": "USD", "convertible": "N", "asset_backed": "N",
        "rule_144a": "N", "perpetual": "N", "coupon_type": "F",
        "bond_type": "CDEB", "coupon": 5.0, "day_count_basis": "30/360",
        "maturity": "2030-01-01", "offering_amt": 10000.0,
        "amount_outstanding": 10000.0, "preferred_security": "N",
    })
    p = tmp_path / "reference_fisd_mergedissue.parquet"
    pd.DataFrame([row]).to_parquet(p)
    monkeypatch.setattr(fref, "ISSUE_FILE", p)

    loaded = bfr._load_issue()
    assert loaded["complete_cusip"].iloc[0] == "000361AB1"
    assert str(loaded["complete_cusip"].dtype) in ("string", "object")
    assert pd.api.types.is_datetime64_any_dtype(loaded["maturity"])


# ---------------------------------------------------------------------------
# Survivorship: default-date derivation
# ---------------------------------------------------------------------------

def test_default_date_is_earliest_default_event(tmp_path, monkeypatch, cfg):
    r = pd.DataFrame([
        {"issue_id": 1.0, "rating": "A",   "rating_date": "2010-01-01"},
        {"issue_id": 1.0, "rating": "D",   "rating_date": "2012-06-01"},
        {"issue_id": 1.0, "rating": "SD",  "rating_date": "2011-03-01"},  # earliest default
        {"issue_id": 2.0, "rating": "BBB", "rating_date": "2010-01-01"},  # never defaults
    ])
    p = tmp_path / "reference_fisd_ratings.parquet"
    r.to_parquet(p)
    monkeypatch.setattr(fref, "RATINGS_FILE", p)
    dd = bfr._default_dates_by_issue(cfg)
    assert dd.loc[1] == pd.Timestamp("2011-03-01")    # earliest of D / SD
    assert 2 not in dd.index                           # never defaulted → absent


def test_default_date_excludes_pre_date_min_events(tmp_path, monkeypatch, cfg):
    """Exit-date hygiene: a default rating event before date_min is dropped, so
    a garbage-early date can't make a bond's whole history terminal."""
    r = pd.DataFrame([
        {"issue_id": 1.0, "rating": "D", "rating_date": "1975-01-01"},  # < date_min
        {"issue_id": 1.0, "rating": "D", "rating_date": "2010-06-01"},  # valid
    ])
    p = tmp_path / "reference_fisd_ratings.parquet"
    r.to_parquet(p)
    monkeypatch.setattr(fref, "RATINGS_FILE", p)
    dd = bfr._default_dates_by_issue(cfg)
    assert dd.loc[1] == pd.Timestamp("2010-06-01")    # 1975 garbage excluded


# ---------------------------------------------------------------------------
# Config validation
# ---------------------------------------------------------------------------

def test_load_config_missing_block_raises(tmp_path, monkeypatch):
    bad = tmp_path / "no_fisd.yaml"
    bad.write_text("monthly_panel:\n  min_vol_qt: 100000\n")
    monkeypatch.setattr(fref, "THRESHOLDS_FILE", bad)
    with pytest.raises(KeyError, match="fisd"):
        bfr.load_config()


def test_load_config_missing_rating_key_raises(tmp_path, monkeypatch):
    bad = tmp_path / "partial.yaml"
    bad.write_text(
        "fisd:\n"
        "  universe: {currency_allow: [USD], allow_null_currency: true,\n"
        "    exclude_convertible: true, exclude_asset_backed: true,\n"
        "    exclude_144a: true, exclude_perpetual: true,\n"
        "    coupon_type_allow: [F], bond_type_keep: [CDEB]}\n"
        "  rating: {average_agencies: [SPR, MR]}\n"     # missing not_rated_tokens etc.
        "  rating_numeric_map: {sp: {AAA: 1}, moody: {Aaa: 1}}\n"
        "  amount_outstanding: {size_proxy: offering_amt}\n"
    )
    monkeypatch.setattr(fref, "THRESHOLDS_FILE", bad)
    with pytest.raises(KeyError, match="not_rated_tokens"):
        bfr.load_config()


# ---------------------------------------------------------------------------
# Real-data smoke (skipped on a clean checkout without licensed FISD data)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(
    not (bfr.ISSUE_FILE.exists() and bfr.PANEL_FILE.exists()),
    reason="licensed FISD data / dev panel not present",
)
def test_real_data_smoke_universe_and_no_leakage():
    cfg = bfr.load_config()
    static = bfr.build_static(cfg)
    # Sane universe: some bonds eligible, but restriction actually bites.
    n_total = len(static)
    n_elig = int(static["universe_eligible"].sum())
    assert 0 < n_elig < n_total, "universe restriction should keep some, drop some"
    # Leading-zero cusips really exist and survived as strings.
    assert static["cusip"].astype(str).str.match(r"^[0-9A-Z]{9}$").mean() > 0.9

    # Build ratings on a small sampled grid (history preserved per cusip) and
    # assert structural no-look-ahead.
    grid = pd.read_parquet(bfr.PANEL_FILE, columns=["cusip", "date"])
    grid["cusip"] = grid["cusip"].astype("string")
    sample = (
        static.loc[static["universe_eligible"], "cusip"]
        .head(300).tolist()
    )
    sub = grid[grid["cusip"].isin(sample)]
    issue_to_cusip = static.dropna(subset=["issue_id"]).set_index("issue_id")["cusip"]
    issue_to_cusip = issue_to_cusip[~issue_to_cusip.index.duplicated(keep="first")]
    rm = bfr.build_ratings_monthly(cfg, sub, issue_to_cusip)
    leakage = int(
        (rm["_sel_rating_date"].notna() & (rm["_sel_rating_date"] > rm["date"])).sum()
    )
    assert leakage == 0, "as-of join leaked future rating dates"
    assert bool(rm["is_rated"].any()), "expected some rated bond-months in the sample"
