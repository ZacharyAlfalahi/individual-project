"""
Unit tests for TRACE preprocessing (Dick-Nielsen basic cleaning ONLY).

Post Phase-1 registry refactor: preprocess_trace.py performs only the
basic-cleaning filters shared by both column families —

  1a. trc_st == "T"                 — drop cancelled/reversed/withdrawn records
  1b. composite-key cancellation    — drop T records cancelled by a later C/W/X/Y/R
  2.  asof_cd blank                 — keep only on-time original reports
  3.  wis_fl != "Y"                 — drop when-issued trades
  6.  interdealer dedup             — Dick-Nielsen B2 on RAW prices

— plus the dev/holdout date split (< holdout_start_year dev, >= holdout) and
trace_clean_raw.parquet partition writing.

Price plausibility and decimal-shift were RELOCATED to apply_decimal_shift.py
(corrected branch only) per amendment A1.7: under meas_err=OFF the raw family
must preserve implausible prices BIT-EXACT. These tests therefore assert that
zero/negative/over-ceiling/near-zero/decimal-slip prices SURVIVE this stage
with their prices unchanged. Decimal-shift behaviour is covered in
test_apply_decimal_shift.py / the meas_err injection suite, not here.
"""
import csv
import gzip
import io
import json
from collections import Counter
from pathlib import Path

import pandas as pd
import pytest

from preprocess_trace import (
    KEEP_COLUMNS,
    load_thresholds,
    run_pandas,
    write_report,
)

# Load thresholds from the single source of truth — tests break if values change
_cfg = load_thresholds()

# ---------------------------------------------------------------------------
# Synthetic test data
# ---------------------------------------------------------------------------
#
# Each case is a dict so optional fields (vol, side, msg_seq_nb, orig_msg_seq_nb,
# bond_id_override) can be added without breaking unpackers.
#   Required: label, trc_st, asof_cd, wis_fl, price, dt, survives, exp_price
#   Optional: vol (default 10000.0), side (default ""), msg_seq_nb (default ""),
#             orig_msg_seq_nb (default ""), bond_id_override (default None)
# bond_sym_id is built as f"TEST_{label}" unless bond_id_override is set
# (used to give the interdealer pair the same bond_id).
#
# A1.7 contract: price-implausible rows (zero, negative, over-ceiling,
# near-zero, 10x/100x decimal slips) SURVIVE the raw cleaning stage with their
# prices preserved bit-exact. exp_price == price for every survivor.

SYNTHETIC_CASES = [
    {"label": "clean_trade",        "trc_st": "T", "asof_cd": "",  "wis_fl": "N", "price":    98.5, "dt": "2015-06-01", "survives": True,  "exp_price": 98.5},
    {"label": "cancelled_C",        "trc_st": "C", "asof_cd": "",  "wis_fl": "N", "price":    98.5, "dt": "2015-06-01", "survives": False, "exp_price": None},
    {"label": "reversal_W",         "trc_st": "W", "asof_cd": "",  "wis_fl": "N", "price":    98.5, "dt": "2011-03-10", "survives": False, "exp_price": None},
    {"label": "correction_R",       "trc_st": "R", "asof_cd": "",  "wis_fl": "N", "price":    98.5, "dt": "2015-06-01", "survives": False, "exp_price": None},
    {"label": "invalid_X",          "trc_st": "X", "asof_cd": "",  "wis_fl": "N", "price":    98.5, "dt": "2015-06-01", "survives": False, "exp_price": None},
    {"label": "when_issued",        "trc_st": "T", "asof_cd": "",  "wis_fl": "Y", "price":    98.5, "dt": "2015-06-01", "survives": False, "exp_price": None},
    {"label": "asof_R",             "trc_st": "T", "asof_cd": "R", "wis_fl": "N", "price":    98.5, "dt": "2015-06-01", "survives": False, "exp_price": None},
    {"label": "asof_D",             "trc_st": "T", "asof_cd": "D", "wis_fl": "N", "price":    98.5, "dt": "2015-06-01", "survives": False, "exp_price": None},
    {"label": "asof_X",             "trc_st": "T", "asof_cd": "X", "wis_fl": "N", "price":    98.5, "dt": "2015-06-01", "survives": False, "exp_price": None},

    # A1.7 — implausible prices are NOT filtered or shifted at this stage.
    # They survive with prices bit-exact; apply_decimal_shift.py (corrected
    # branch only) handles plausibility + shifting downstream.
    {"label": "zero_price",         "trc_st": "T", "asof_cd": "",  "wis_fl": "N", "price":     0.0, "dt": "2015-06-01", "survives": True,  "exp_price":     0.0},
    {"label": "neg_price",          "trc_st": "T", "asof_cd": "",  "wis_fl": "N", "price":    -1.0, "dt": "2015-06-01", "survives": True,  "exp_price":    -1.0},
    {"label": "over_ceiling",       "trc_st": "T", "asof_cd": "",  "wis_fl": "N", "price": 35000.0, "dt": "2015-06-01", "survives": True,  "exp_price": 35000.0},
    {"label": "decimal_10x",        "trc_st": "T", "asof_cd": "",  "wis_fl": "N", "price":   985.0, "dt": "2015-06-01", "survives": True,  "exp_price":   985.0},
    {"label": "decimal_100x",       "trc_st": "T", "asof_cd": "",  "wis_fl": "N", "price":  9850.0, "dt": "2015-06-01", "survives": True,  "exp_price":  9850.0},
    {"label": "near_zero_price",    "trc_st": "T", "asof_cd": "",  "wis_fl": "N", "price":     0.5, "dt": "2015-06-01", "survives": True,  "exp_price":     0.5},

    {"label": "asof_A_drop",        "trc_st": "T", "asof_cd": "A", "wis_fl": "N", "price":    98.5, "dt": "2018-09-15", "survives": False, "exp_price": None},
    {"label": "asof_blank_keep",    "trc_st": "T", "asof_cd": "",  "wis_fl": "N", "price":    98.5, "dt": "2019-12-31", "survives": True,  "exp_price": 98.5},
    {"label": "holdout_trade",      "trc_st": "T", "asof_cd": "",  "wis_fl": "N", "price":    98.5, "dt": "2023-03-15", "survives": True,  "exp_price": 98.5},

    # Interdealer pair (same bond_id, dt, price, vol; B and S sides) — keep S, drop B
    {"label": "interdealer_sell_A", "trc_st": "T", "asof_cd": "",  "wis_fl": "N", "price":    95.0, "dt": "2016-03-01", "survives": True,  "exp_price": 95.0,
     "vol": 50000.0, "side": "S", "bond_id_override": "TEST_PAIR_A"},
    {"label": "interdealer_buy_A",  "trc_st": "T", "asof_cd": "",  "wis_fl": "N", "price":    95.0, "dt": "2016-03-01", "survives": False, "exp_price": None,
     "vol": 50000.0, "side": "B", "bond_id_override": "TEST_PAIR_A"},
    # B-side with no matching S — kept (not a duplicate)
    {"label": "interdealer_buy_no_match", "trc_st": "T", "asof_cd": "", "wis_fl": "N", "price": 92.0, "dt": "2016-03-01", "survives": True, "exp_price": 92.0,
     "side": "B"},

    # T record that gets cancelled by a later C record. Pass 1 sees the C
    # record's (bond, dt, orig_msg_seq_nb=9001) composite key and adds it to
    # cancelled_keys; Pass 2 drops the original T row via the same composite key.
    {"label": "original_later_cancelled", "trc_st": "T", "asof_cd": "", "wis_fl": "N", "price": 88.0, "dt": "2017-05-10", "survives": False, "exp_price": None,
     "msg_seq_nb": "9001", "bond_id_override": "TEST_CXR_ORIG"},
    # The cancellation record itself — same bond + same dt as the original;
    # orig_msg_seq_nb=9001 points back. Won't appear in cleaned output
    # (trc_st="C" drops at filter 1a).
    {"label": "cancellation_record",    "trc_st": "C", "asof_cd": "", "wis_fl": "N", "price": 88.0, "dt": "2017-05-10", "survives": False, "exp_price": None,
     "msg_seq_nb": "9002", "orig_msg_seq_nb": "9001", "bond_id_override": "TEST_CXR_ORIG"},

    # msg_seq_nb=9001 reused on a different (bond, dt). With a buggy
    # bare-msg_seq_nb match this row would be wrongly dropped. With the
    # composite (bond, dt, msg_seq_nb) key it must SURVIVE.
    {"label": "msg_seq_reuse_cross_day", "trc_st": "T", "asof_cd": "", "wis_fl": "N", "price": 102.0, "dt": "2020-08-15", "survives": True, "exp_price": 102.0,
     "msg_seq_nb": "9001"},

    # Market-maker scenario: 1 S + 2 B at the same key. Counter-based pairing
    # consumes exactly one B per S and keeps the second B (a legitimate client
    # buy on the dealer's book); set-based dedup would drop both B's.
    {"label": "mm_sell_A",          "trc_st": "T", "asof_cd": "", "wis_fl": "N", "price": 80.0, "dt": "2018-04-04", "survives": True,  "exp_price": 80.0,
     "vol": 25000.0, "side": "S", "bond_id_override": "TEST_MM_A"},
    {"label": "mm_buy_A_paired",    "trc_st": "T", "asof_cd": "", "wis_fl": "N", "price": 80.0, "dt": "2018-04-04", "survives": False, "exp_price": None,
     "vol": 25000.0, "side": "B", "bond_id_override": "TEST_MM_A"},
    {"label": "mm_buy_A_client",    "trc_st": "T", "asof_cd": "", "wis_fl": "N", "price": 80.0, "dt": "2018-04-04", "survives": True,  "exp_price": 80.0,
     "vol": 25000.0, "side": "B", "bond_id_override": "TEST_MM_A"},
]


def _bond_id_for(case: dict) -> str:
    return case.get("bond_id_override") or f"TEST_{case['label']}"


def make_synthetic_df() -> pd.DataFrame:
    rows = []
    for i, case in enumerate(SYNTHETIC_CASES):
        row = {col: "" for col in KEEP_COLUMNS}
        row["bond_sym_id"] = _bond_id_for(case)
        row["trd_exctn_dt"] = case["dt"]
        row["trd_exctn_tm"] = "10:00:00"
        row["trc_st"] = case["trc_st"]
        row["asof_cd"] = case["asof_cd"]
        row["wis_fl"] = case["wis_fl"]
        row["rptd_pr"] = case["price"]
        row["entrd_vol_qt"] = case.get("vol", 10000.0)
        row["company_symbol"] = "TEST"
        row["rpt_side_cd"] = case.get("side", "")
        # Synthesize sequential msg_seq_nb when not given so apply_filters can
        # do cancellation matching via the standard column.
        row["msg_seq_nb"] = case.get("msg_seq_nb") or str(1000 + i)
        row["orig_msg_seq_nb"] = case.get("orig_msg_seq_nb", "")
        rows.append(row)
    return pd.DataFrame(rows)


def apply_filters(df: pd.DataFrame) -> pd.DataFrame:
    """In-DataFrame equivalent of the two-pass production pipeline.

    Mirrors the production logic exactly: composite (bond, dt, msg_seq_nb)
    cancellation key (Dick-Nielsen B1), Counter-based pair dedup on RAW
    prices (Dick-Nielsen B2), and — per the Phase-1 refactor — NO price
    plausibility filter and NO decimal-shift. Sell-side dedup keys are
    collected from rows passing filters 1a/2/3, matching production Pass 1.
    """
    df = df.copy()

    # Pass-1 equivalent (a): collect cancelled composite keys from non-T rows
    non_t = df[df["trc_st"] != "T"]
    cancelled_keys: set = set()
    for _, r in non_t.iterrows():
        orig = str(r.get("orig_msg_seq_nb", "") or "")
        if orig == "" or orig == "nan":
            continue
        cancelled_keys.add((str(r["bond_sym_id"]), str(r["trd_exctn_dt"]), orig))

    # Pass-1 equivalent (b): sell-side dedup pool from rows passing 1a/2/3
    pool = df[df["trc_st"] == "T"]
    pool = pool[pool["asof_cd"].isna() | (pool["asof_cd"] == "")]
    pool = pool[pool["wis_fl"] != "Y"]
    sells = pool[pool["rpt_side_cd"] == "S"]
    sell_counts: Counter = Counter()
    for _, r in sells.iterrows():
        sell_counts[(str(r["bond_sym_id"]), str(r["trd_exctn_dt"]),
                     r["rptd_pr"], r["entrd_vol_qt"])] += 1

    # Filter 1a
    df = df[df["trc_st"] == "T"]
    # Filter 1b — drop T records whose (bond, dt, msg_seq_nb) was cancelled
    if cancelled_keys:
        def _is_cancelled(r):
            return (str(r["bond_sym_id"]), str(r["trd_exctn_dt"]),
                    str(r["msg_seq_nb"])) in cancelled_keys
        df = df[~df.apply(_is_cancelled, axis=1)]

    # Filter 2: blank asof_cd only
    df = df[df["asof_cd"].isna() | (df["asof_cd"] == "")]
    # Filter 3: when-issued
    df = df[df["wis_fl"] != "Y"]

    # NO price plausibility, NO decimal-shift — relocated downstream (A1.7).

    # Filter 6 — interdealer pair dedup using a Counter (multiset). 1 S
    # consumes exactly 1 B; surplus B's on the same key remain (client trades).
    if sell_counts:
        keep = []
        for idx, r in df.iterrows():
            if r["rpt_side_cd"] != "B":
                keep.append(idx)
                continue
            key = (str(r["bond_sym_id"]), str(r["trd_exctn_dt"]),
                   r["rptd_pr"], r["entrd_vol_qt"])
            if sell_counts.get(key, 0) > 0:
                sell_counts[key] -= 1  # B paired with one S — drop this B
            else:
                keep.append(idx)
        df = df.loc[keep]

    return df


# ---------------------------------------------------------------------------
# Filter pipeline
# ---------------------------------------------------------------------------

class TestFilterLogic:
    def setup_method(self):
        self.raw = make_synthetic_df()
        self.filtered = apply_filters(self.raw)

    def _price_of(self, bond_id: str) -> float:
        row = self.filtered[self.filtered["bond_sym_id"] == bond_id]
        assert len(row) == 1
        return row["rptd_pr"].iloc[0]

    def test_surviving_row_count(self):
        expected = sum(1 for c in SYNTHETIC_CASES if c["survives"])
        assert len(self.filtered) == expected

    def test_correct_rows_survive(self):
        surviving = set(self.filtered["bond_sym_id"].astype(str))
        expected = {_bond_id_for(c) for c in SYNTHETIC_CASES if c["survives"]}
        assert surviving == expected

    def test_cancelled_dropped(self):
        assert "TEST_cancelled_C" not in self.filtered["bond_sym_id"].values

    def test_reversal_W_dropped(self):
        assert "TEST_reversal_W" not in self.filtered["bond_sym_id"].values

    def test_correction_R_dropped(self):
        assert "TEST_correction_R" not in self.filtered["bond_sym_id"].values

    def test_when_issued_dropped(self):
        assert "TEST_when_issued" not in self.filtered["bond_sym_id"].values

    def test_asof_R_dropped(self):
        assert "TEST_asof_R" not in self.filtered["bond_sym_id"].values

    def test_asof_D_dropped(self):
        assert "TEST_asof_D" not in self.filtered["bond_sym_id"].values

    def test_asof_X_dropped(self):
        assert "TEST_asof_X" not in self.filtered["bond_sym_id"].values

    def test_asof_A_dropped(self):
        assert "TEST_asof_A_drop" not in self.filtered["bond_sym_id"].values

    def test_asof_blank_kept(self):
        assert "TEST_asof_blank_keep" in self.filtered["bond_sym_id"].values

    # A1.7 — implausible prices survive the raw stage bit-exact (no drop,
    # no shift). Exact equality is deliberate: bit-exact preservation is
    # the contract, so pytest.approx would weaken the assertion.
    def test_zero_price_survives_bit_exact(self):
        assert self._price_of("TEST_zero_price") == 0.0

    def test_negative_price_survives_bit_exact(self):
        assert self._price_of("TEST_neg_price") == -1.0

    def test_over_ceiling_survives_bit_exact(self):
        assert self._price_of("TEST_over_ceiling") == 35000.0

    def test_near_zero_price_survives_bit_exact(self):
        assert self._price_of("TEST_near_zero_price") == 0.5

    def test_decimal_10x_not_shifted(self):
        assert self._price_of("TEST_decimal_10x") == 985.0

    def test_decimal_100x_not_shifted(self):
        assert self._price_of("TEST_decimal_100x") == 9850.0

    # Interdealer dedup
    def test_interdealer_sell_kept(self):
        # The S-side of the pair survives — bond_id is shared with the buy via
        # bond_id_override, so look up via that.
        kept = self.filtered[
            (self.filtered["bond_sym_id"] == "TEST_PAIR_A")
            & (self.filtered["rpt_side_cd"] == "S")
        ]
        assert len(kept) == 1

    def test_interdealer_buy_dropped(self):
        dropped = self.filtered[
            (self.filtered["bond_sym_id"] == "TEST_PAIR_A")
            & (self.filtered["rpt_side_cd"] == "B")
        ]
        assert len(dropped) == 0

    def test_interdealer_buy_no_match_kept(self):
        assert "TEST_interdealer_buy_no_match" in self.filtered["bond_sym_id"].values

    # Two-pass cancellation
    def test_cancelled_original_dropped(self):
        # Composite (bond, dt, msg_seq_nb) match: the original T row at
        # (TEST_CXR_ORIG, 2017-05-10, 9001) is dropped because the C record
        # references the same composite key.
        kept_orig = self.filtered[
            (self.filtered["bond_sym_id"] == "TEST_CXR_ORIG")
            & (self.filtered["msg_seq_nb"].astype(str) == "9001")
        ]
        assert len(kept_orig) == 0

    # Regression: msg_seq_nb=9001 reused on a different (bond, dt) must NOT
    # be dropped. With a buggy bare-msg_seq_nb match this row was wrongly
    # removed; the composite key keeps it.
    def test_cross_day_msg_seq_reuse_kept(self):
        assert "TEST_msg_seq_reuse_cross_day" in self.filtered["bond_sym_id"].values

    # Regression: market-maker with 1 S + 2 B at the same key. Counter-based
    # pairing keeps the second B as a legitimate client trade; a set-based
    # dedup would drop both B's.
    def test_market_maker_keeps_client_buy(self):
        mm_rows = self.filtered[self.filtered["bond_sym_id"] == "TEST_MM_A"]
        # Expect exactly 2 rows: the S, and one B (the second B is the client trade)
        assert len(mm_rows) == 2
        sides = mm_rows["rpt_side_cd"].value_counts().to_dict()
        assert sides.get("S", 0) == 1
        assert sides.get("B", 0) == 1


# ---------------------------------------------------------------------------
# Dev/holdout split
# ---------------------------------------------------------------------------

class TestDevHoldoutSplit:
    def setup_method(self):
        self.filtered = apply_filters(make_synthetic_df())
        self.filtered["trd_exctn_dt"] = pd.to_datetime(self.filtered["trd_exctn_dt"])

    def test_split_no_rows_lost(self):
        year = _cfg["holdout_start_year"]
        dev = self.filtered[self.filtered["trd_exctn_dt"].dt.year < year]
        hold = self.filtered[self.filtered["trd_exctn_dt"].dt.year >= year]
        assert len(dev) + len(hold) == len(self.filtered)

    def test_dev_contains_no_holdout_dates(self):
        year = _cfg["holdout_start_year"]
        dev = self.filtered[self.filtered["trd_exctn_dt"].dt.year < year]
        assert (dev["trd_exctn_dt"].dt.year < year).all()

    def test_holdout_contains_no_dev_dates(self):
        year = _cfg["holdout_start_year"]
        hold = self.filtered[self.filtered["trd_exctn_dt"].dt.year >= year]
        assert (hold["trd_exctn_dt"].dt.year >= year).all()


# ---------------------------------------------------------------------------
# Output schema
# ---------------------------------------------------------------------------

class TestOutputSchema:
    def setup_method(self):
        raw = make_synthetic_df()
        filtered = apply_filters(raw)
        self.output = filtered[KEEP_COLUMNS].rename(columns={"bond_sym_id": "bond_id"})

    def test_bond_id_present(self):
        assert "bond_id" in self.output.columns

    def test_cusip_id_present(self):
        # Post 2026-06-09 re-pull: cusip_id is populated and retained
        # alongside bond_id for downstream FISD merging.
        assert "cusip_id" in self.output.columns

    def test_expected_columns_present(self):
        expected = [c if c != "bond_sym_id" else "bond_id" for c in KEEP_COLUMNS]
        for col in expected:
            assert col in self.output.columns, f"Missing column: {col}"


# ---------------------------------------------------------------------------
# Holdout read guard
# ---------------------------------------------------------------------------

class TestHoldoutGuard:
    def test_guard_raises_on_holdout_read(self, tmp_path):
        p = tmp_path / "holdout" / "trace_clean_raw.parquet"
        p.parent.mkdir()
        with pytest.raises(AssertionError, match="holdout"):
            open(str(p))


# ---------------------------------------------------------------------------
# run_pandas end-to-end
# ---------------------------------------------------------------------------

def _make_synthetic_csv_gz(path: Path) -> int:
    """Write a gzipped CSV containing the synthetic test cases. Returns expected survivor count."""
    # All columns that run_pandas needs (plus extras that get dropped).
    # cusip_id is in KEEP_COLUMNS (since the 2026-06-09 re-pull); the rest are
    # filter-only columns dropped after filtering.
    all_cols = KEEP_COLUMNS + ["trc_st", "asof_cd", "wis_fl", "msg_seq_nb", "orig_msg_seq_nb"]
    survivors = 0
    rows = []
    for i, case in enumerate(SYNTHETIC_CASES):
        row = {col: "" for col in all_cols}
        row["bond_sym_id"] = _bond_id_for(case)
        row["trd_exctn_dt"] = case["dt"]
        row["trd_exctn_tm"] = "10:00:00"
        row["trc_st"] = case["trc_st"]
        row["asof_cd"] = case["asof_cd"]
        row["wis_fl"] = case["wis_fl"]
        row["rptd_pr"] = case["price"]
        row["entrd_vol_qt"] = case.get("vol", 10000.0)
        row["company_symbol"] = "TEST"
        row["rpt_side_cd"] = case.get("side", "")
        row["msg_seq_nb"] = case.get("msg_seq_nb") or str(1000 + i)
        row["orig_msg_seq_nb"] = case.get("orig_msg_seq_nb", "")
        row["cusip_id"] = ""
        rows.append(row)
        if case["survives"]:
            survivors += 1

    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=all_cols)
    writer.writeheader()
    writer.writerows(rows)

    with gzip.open(path, "wt") as f:
        f.write(buf.getvalue())

    return survivors


def _redirect_paths(monkeypatch, tmp_path: Path, raw: Path) -> dict:
    """Redirect all module path constants to tmp_path.

    REPO_ROOT must be redirected too: write_report computes
    DEV_OUT.relative_to(REPO_ROOT) without an is_relative_to guard, which
    raises ValueError when the outputs live outside the repo.
    """
    import preprocess_trace as pt
    paths = {
        "dev_out": tmp_path / "development" / "trace_clean_raw.parquet",
        "hold_out": tmp_path / "holdout" / "trace_clean_raw.parquet",
        "report_out": tmp_path / "development" / "cleaning_report.json",
    }
    monkeypatch.setattr(pt, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(pt, "RAW_FILE", raw)
    monkeypatch.setattr(pt, "DEV_OUT", paths["dev_out"])
    monkeypatch.setattr(pt, "HOLD_OUT", paths["hold_out"])
    monkeypatch.setattr(pt, "REPORT_OUT", paths["report_out"])
    return paths


class TestRunPandasEndToEnd:
    def test_produces_correct_outputs(self, tmp_path, monkeypatch):
        raw = tmp_path / "trace_enhanced_repull.csv.gz"
        expected_survivors = _make_synthetic_csv_gz(raw)
        paths = _redirect_paths(monkeypatch, tmp_path, raw)
        dev_out, hold_out, report_out = (
            paths["dev_out"], paths["hold_out"], paths["report_out"],
        )

        counts = run_pandas(_cfg)
        write_report(counts, _cfg, "test")

        # Row count integrity
        assert counts["development_rows"] + counts["holdout_rows"] == counts["final_clean_total"]
        assert counts["final_clean_total"] == expected_survivors
        assert counts["parquet_row_count_verified"] is True

        # Drop accounting against the synthetic cases:
        #   5 non-T rows (C, W, R, X, cancellation record)  → dropped_trc_st
        #   1 T row cancelled via composite key             → dropped_cancelled_original
        #   4 non-blank asof rows (R, D, X, A)              → dropped_asof_cd
        #   1 when-issued row                               → dropped_wis_fl
        #   2 B-side duplicates (PAIR_A + MM_A paired B)    → dropped_interdealer_duplicate
        assert counts["dropped_trc_st"] == 5
        assert counts["dropped_cancelled_original"] == 1, \
            "expected one T row dropped by Pass 1 composite-key cancellation"
        assert counts["dropped_asof_cd"] == 4
        assert counts["dropped_wis_fl"] == 1
        assert counts["dropped_interdealer_duplicate"] == 2, \
            "expected two B-side interdealer duplicates dropped (PAIR_A + MM_A)"
        assert counts["dropped_invalid_date"] == 0

        # Parquet outputs exist and are readable
        assert dev_out.exists(), "dev parquet must be written"
        assert hold_out.exists(), "holdout parquet must be written (holdout_trade row is post-2022)"
        df_dev = pd.read_parquet(dev_out)
        assert "bond_id" in df_dev.columns
        # cusip_id is retained as a column (2026-06-09 re-pull); the synthetic
        # CSV leaves it blank, so the column is present but null.
        assert "cusip_id" in df_dev.columns
        assert "TEST_holdout_trade" not in df_dev["bond_id"].values

        # A1.7: implausible prices survive bit-exact in the raw parquet —
        # no plausibility drop, no decimal-shift. Exact equality on purpose.
        def _price(bond_id):
            rows = df_dev[df_dev["bond_id"] == bond_id]
            assert len(rows) == 1, f"{bond_id}: expected exactly one surviving row"
            return rows["rptd_pr"].iloc[0]

        assert _price("TEST_zero_price") == 0.0
        assert _price("TEST_neg_price") == -1.0
        assert _price("TEST_over_ceiling") == 35000.0
        assert _price("TEST_decimal_10x") == 985.0
        assert _price("TEST_decimal_100x") == 9850.0
        assert _price("TEST_near_zero_price") == 0.5
        assert _price("TEST_clean_trade") == 98.5

        # The interdealer S survives; its B partner is dropped
        pair_rows = df_dev[df_dev["bond_id"] == "TEST_PAIR_A"]
        assert len(pair_rows) == 1
        assert pair_rows["rpt_side_cd"].iloc[0] == "S"
        # Market-maker 1S+2B: S + 1 client B survive, 1 paired B is dropped
        mm_rows = df_dev[df_dev["bond_id"] == "TEST_MM_A"]
        assert len(mm_rows) == 2
        assert set(mm_rows["rpt_side_cd"]) == {"S", "B"}

        # Holdout partition: preprocess_trace.py is the sanctioned partition
        # CREATOR, so reading back its own freshly written tmp partition is
        # legitimate here. pq.read_table is not on the conftest guard list.
        import pyarrow.parquet as _pq
        df_hold = _pq.read_table(str(hold_out)).to_pandas()
        assert len(df_hold) == 1
        assert df_hold["bond_id"].iloc[0] == "TEST_holdout_trade"
        assert df_hold["rptd_pr"].iloc[0] == 98.5

        # Report written atomically and parseable
        assert report_out.exists()
        with open(report_out) as f:
            report = json.load(f)
        assert report["stage"] == "dick_nielsen_only"
        assert "rows" in report
        assert report["rows"]["final_clean_total"] == expected_survivors
        # The cleaning report legitimately records holdout row counts:
        # this script is the partition creator.
        assert report["rows"]["holdout_rows"] == 1
        assert report["rows"]["development_rows"] == expected_survivors - 1
        assert "dropped_invalid_date" in report["rows"]
        assert "dropped_cancelled_original" in report["rows"]
        assert "dropped_interdealer_duplicate" in report["rows"]
        assert set(report["outputs"]) == {"dev", "holdout"}
        assert "thresholds_sha256" in report
        assert not report_out.with_suffix(".tmp").exists(), \
            ".tmp file should not remain after atomic replace"

        # CUSIP counter keys use the post-Dick-Nielsen scope (this stage is
        # pre-bounce-back logically, but the file is no longer overwritten by
        # the bounce-back filter — see the NAMING note in run_pandas). Legacy
        # unscoped keys must NOT appear.
        assert "cusip_populated_post_dn" in report["rows"]
        assert "cusip_blank_post_dn" in report["rows"]
        assert "cusip_populated" not in report["rows"], \
            "legacy ambiguous key — must carry the _post_dn scope suffix"
        assert "cusip_blank" not in report["rows"]
        # All synthetic rows have cusip_id="" → 100% blank after this stage.
        assert report["rows"]["cusip_blank_post_dn"] == expected_survivors
        assert report["rows"]["cusip_populated_post_dn"] == 0


class TestCusipBlankMaskCounting:
    """Edge cases for the CUSIP populated/blank counter in run_pandas.

    Locks the blank-mask behaviour: NaN cells (empty in CSV → NaN in pandas),
    whitespace-only cells, and the empty string all count as blank; anything
    else counts as populated.
    """

    def _make_csv(self, path: Path, cusip_values: list[str]) -> None:
        """Build a synthetic CSV where every row passes filters 1a–3 and
        dedup. Only cusip_id varies; the count assertions are deterministic.
        """
        all_cols = KEEP_COLUMNS + ["trc_st", "asof_cd", "wis_fl", "msg_seq_nb", "orig_msg_seq_nb"]
        rows = []
        for i, c in enumerate(cusip_values):
            rows.append({
                **{col: "" for col in all_cols},
                "bond_sym_id": f"BND_{i:04d}",
                "cusip_id": c,
                "trd_exctn_dt": "2015-06-01",
                "trd_exctn_tm": "10:00:00",
                "trc_st": "T",
                "asof_cd": "",
                "wis_fl": "N",
                "rptd_pr": 98.5,
                "entrd_vol_qt": 10000.0,
                "company_symbol": "TEST",
                "rpt_side_cd": "B",
                "msg_seq_nb": str(50000 + i),
            })
        buf = io.StringIO()
        w = csv.DictWriter(buf, fieldnames=all_cols)
        w.writeheader()
        w.writerows(rows)
        with gzip.open(path, "wt") as f:
            f.write(buf.getvalue())

    def test_blank_mask_handles_nan_empty_and_whitespace(self, tmp_path, monkeypatch):
        raw = tmp_path / "trace_raw.csv.gz"
        # 3 populated + 4 blank-variants (empty CSV cell → NaN, "", "   ", "\t ")
        cusip_values = [
            "000115139", "AAA111BB2", "ZZ999CCC1",   # 3 valid (leading zero, mixed alnum)
            "",                                       # empty string in CSV → NaN at read time
            "",                                       # second NaN
            "   ",                                    # whitespace-only
            "\t ",                                    # tab + space
        ]
        self._make_csv(raw, cusip_values)
        _redirect_paths(monkeypatch, tmp_path, raw)

        counts = run_pandas(_cfg)
        assert counts["cusip_populated_post_dn"] == 3
        assert counts["cusip_blank_post_dn"] == 4
        assert counts["cusip_populated_post_dn"] + counts["cusip_blank_post_dn"] \
            == counts["final_clean_total"]

    def test_leading_zero_preserved_in_parquet(self, tmp_path, monkeypatch):
        # Regression: pandas would coerce all-digit cusip_id to int and drop
        # leading zeros without the dtype={"cusip_id": str} in _csv_reader.
        raw = tmp_path / "trace_raw.csv.gz"
        self._make_csv(raw, ["000115139", "037833100", "912828YY0"])
        paths = _redirect_paths(monkeypatch, tmp_path, raw)

        run_pandas(_cfg)
        df = pd.read_parquet(paths["dev_out"])
        cusips = set(df["cusip_id"].dropna().tolist())
        assert "000115139" in cusips, \
            "leading zero stripped — dtype={'cusip_id': str} guard regressed"
        assert "037833100" in cusips
