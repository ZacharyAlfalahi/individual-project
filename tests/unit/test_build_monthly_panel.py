"""
Unit tests for the dual-family maximal monthly panel build
(scripts/build_monthly_panel.py).

The build reads two daily-layer parquets (raw + corrected families, both
produced by build_daily_panel.py — the corr one post-distressed-filter),
outer-joins them on (cusip_id, year_month), computes per-family returns
under the month-adjacency rule, merges the risk-free rate, and emits the
maximal monthly panel with Arrow metadata panel_kind = maximal.

Methodology pins:
  - monthly price = Σ(daily_vwap × daily_vol) / Σ(daily_vol), per family
  - adjacency rule per family: ret is NaN unless the prior panel month is
    immediately adjacent AND that family has a price in it
  - divergent NaN patterns across families are never "repaired"
  - xret_<fam> = ret_<fam> − rf_monthly
  - last_trade_date_<fam> = max(trd_exctn_dt) on that family's daily input
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import build_monthly_panel as bmp
from build_monthly_panel import SIZE_PLACEHOLDER, load_config

# All thresholds come from docs/thresholds.yaml via the production loader —
# never duplicated as literals here.
_cfg = load_config()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

DAILY_SCHEMA = pa.schema([
    pa.field("cusip_id",      pa.string()),
    pa.field("trd_exctn_dt",  pa.timestamp("us")),
    pa.field("price_vwap",    pa.float64()),
    pa.field("total_vol",     pa.float64()),
    pa.field("n_trades",      pa.int64()),
    pa.field("min_price",     pa.float64()),
    pa.field("max_price",     pa.float64()),
])


def _daily_row(cusip, date, vwap, vol, n_trades=1):
    return {
        "cusip_id": cusip,
        "trd_exctn_dt": date,
        "price_vwap": vwap,
        "total_vol": vol,
        "n_trades": n_trades,
        "min_price": vwap,
        "max_price": vwap,
    }


def _make_daily_parquet(path: Path, rows: list[dict]) -> None:
    df = pd.DataFrame(rows)
    df["trd_exctn_dt"] = pd.to_datetime(df["trd_exctn_dt"])
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(
        pa.Table.from_pandas(df, schema=DAILY_SCHEMA, preserve_index=False),
        str(path),
    )


def _make_rf_parquet(path: Path, rf_pairs: list[tuple[str, float]]) -> None:
    df = pd.DataFrame({
        "year_month": [m for m, _ in rf_pairs],
        "rf_monthly": [r for _, r in rf_pairs],
    })
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pandas(df, preserve_index=False), str(path))


def _run(tmp_path, monkeypatch, raw_rows, corr_rows, rf_pairs):
    """Write synthetic daily-layer + rf inputs, patch the module path
    constants to tmp_path, run build_panel. Returns (out_path, counts)."""
    raw_path = tmp_path / "trace_daily_raw.parquet"
    corr_path = tmp_path / "trace_daily_corr_filtered.parquet"
    rf_path = tmp_path / "rf_rate.parquet"
    out_path = tmp_path / "monthly_panel_maximal.parquet"

    _make_daily_parquet(raw_path, raw_rows)
    _make_daily_parquet(corr_path, corr_rows)
    _make_rf_parquet(rf_path, rf_pairs)

    monkeypatch.setattr(bmp, "RAW_DAILY", raw_path)
    monkeypatch.setattr(bmp, "CORR_DAILY", corr_path)
    monkeypatch.setattr(bmp, "RF_FILE", rf_path)
    monkeypatch.setattr(bmp, "OUT_FILE", out_path)
    monkeypatch.setattr(bmp, "REPORT_OUT", tmp_path / "report.json")

    counts = bmp.build_panel(_cfg)
    return out_path, counts


def _me(s: str) -> pd.Timestamp:
    """Month-end Timestamp from a YYYY-MM string."""
    return pd.Timestamp(s) + pd.offsets.MonthEnd(0)


def _row(df: pd.DataFrame, cusip: str, ym: str) -> pd.Series:
    cell = df[(df["cusip"] == cusip) & (df["date"] == _me(ym))]
    assert len(cell) == 1, f"expected exactly one row for {cusip}@{ym}, got {len(cell)}"
    return cell.iloc[0]


C1 = "CUSIP0001"
C2 = "CUSIP0002"


# ---------------------------------------------------------------------------
# Monthly price aggregation — Σ(daily_vwap × daily_vol) / Σ(daily_vol)
# ---------------------------------------------------------------------------

class TestMonthlyPriceAggregation:
    def test_volume_weighted_monthly_price_per_family(self, tmp_path, monkeypatch):
        raw_rows = [
            _daily_row(C1, "2015-06-10", 100.0, 100_000, n_trades=2),
            _daily_row(C1, "2015-06-20", 200.0, 300_000, n_trades=3),
        ]
        # corr drops the second day (e.g. distressed-filtered out)
        corr_rows = [
            _daily_row(C1, "2015-06-10", 100.0, 100_000, n_trades=2),
        ]
        out, _ = _run(tmp_path, monkeypatch, raw_rows, corr_rows,
                      [("2015-06", 0.001)])
        df = pd.read_parquet(out)
        row = _row(df, C1, "2015-06")
        # raw: (100×100k + 200×300k) / 400k = 175.0 — NOT the simple
        # average (150.0) and not any single day's vwap.
        assert row["price_eom_raw"] == pytest.approx(175.0, abs=1e-9)
        # corr: single surviving day → 100.0
        assert row["price_eom_corr"] == pytest.approx(100.0, abs=1e-9)
        # Daily aggregates sum per family.
        assert row["n_trades_raw"] == 5
        assert row["n_trades_corr"] == 2
        assert row["total_vol_raw"] == pytest.approx(400_000.0)
        assert row["total_vol_corr"] == pytest.approx(100_000.0)

    def test_single_day_month_price_is_day_vwap(self, tmp_path, monkeypatch):
        rows = [_daily_row(C1, "2015-06-15", 101.5, 200_000)]
        out, _ = _run(tmp_path, monkeypatch, rows, rows, [("2015-06", 0.001)])
        df = pd.read_parquet(out)
        row = _row(df, C1, "2015-06")
        assert row["price_eom_raw"] == pytest.approx(101.5)
        assert row["price_eom_corr"] == pytest.approx(101.5)


# ---------------------------------------------------------------------------
# Per-family returns under the month-adjacency rule
# ---------------------------------------------------------------------------

class TestAdjacencyReturns:
    def test_consecutive_months_produce_exact_return(self, tmp_path, monkeypatch):
        rows = [
            _daily_row(C1, "2015-05-15", 100.0, 200_000),
            _daily_row(C1, "2015-06-15", 105.0, 200_000),
        ]
        out, _ = _run(tmp_path, monkeypatch, rows, rows,
                      [("2015-05", 0.001), ("2015-06", 0.001)])
        df = pd.read_parquet(out)
        jun = _row(df, C1, "2015-06")
        # (105 − 100) / 100 = 0.05, both families
        assert jun["ret_raw"] == pytest.approx(0.05)
        assert jun["ret_corr"] == pytest.approx(0.05)

    def test_first_observed_month_ret_is_nan(self, tmp_path, monkeypatch):
        rows = [
            _daily_row(C1, "2015-05-15", 100.0, 200_000),
            _daily_row(C1, "2015-06-15", 105.0, 200_000),
        ]
        out, _ = _run(tmp_path, monkeypatch, rows, rows,
                      [("2015-05", 0.001), ("2015-06", 0.001)])
        df = pd.read_parquet(out)
        may = _row(df, C1, "2015-05")
        assert pd.isna(may["ret_raw"]) and pd.isna(may["ret_corr"])

    def test_gap_month_yields_nan_ret_in_month_after_gap(self, tmp_path, monkeypatch):
        # March then June — a 3-month gap. Without the adjacency rule the
        # June row would silently carry a one-month-style (105−100)/100.
        rows = [
            _daily_row(C1, "2015-03-15", 100.0, 200_000),
            _daily_row(C1, "2015-06-15", 105.0, 200_000),
        ]
        out, _ = _run(tmp_path, monkeypatch, rows, rows,
                      [("2015-03", 0.001), ("2015-06", 0.001)])
        df = pd.read_parquet(out)
        jun = _row(df, C1, "2015-06")
        assert pd.isna(jun["ret_raw"]), "return across a gap must be NaN (raw)"
        assert pd.isna(jun["ret_corr"]), "return across a gap must be NaN (corr)"
        assert pd.isna(jun["xret_raw"]) and pd.isna(jun["xret_corr"])

    def test_adjacency_is_per_cusip(self, tmp_path, monkeypatch):
        # C2's first month must not chain off C1's series.
        rows = [
            _daily_row(C1, "2015-05-15", 100.0, 200_000),
            _daily_row(C2, "2015-06-15", 105.0, 200_000),
        ]
        out, _ = _run(tmp_path, monkeypatch, rows, rows,
                      [("2015-05", 0.001), ("2015-06", 0.001)])
        df = pd.read_parquet(out)
        c2 = _row(df, C2, "2015-06")
        assert pd.isna(c2["ret_raw"]) and pd.isna(c2["ret_corr"])


# ---------------------------------------------------------------------------
# Divergent NaN patterns across families are NOT repaired
# ---------------------------------------------------------------------------

class TestDivergentNaNPatterns:
    def _build(self, tmp_path, monkeypatch):
        """raw trades May/June/July; corr is missing June entirely (e.g.
        the distressed filters dropped every June day from corr)."""
        raw_rows = [
            _daily_row(C1, "2015-05-15", 100.0, 200_000),
            _daily_row(C1, "2015-06-15", 110.0, 200_000),
            _daily_row(C1, "2015-07-15", 121.0, 200_000),
        ]
        corr_rows = [
            _daily_row(C1, "2015-05-15", 100.0, 200_000),
            _daily_row(C1, "2015-07-15", 130.0, 200_000),
        ]
        out, _ = _run(tmp_path, monkeypatch, raw_rows, corr_rows,
                      [("2015-05", 0.001), ("2015-06", 0.001), ("2015-07", 0.001)])
        return pd.read_parquet(out)

    def test_month_missing_in_corr_has_real_raw_ret_and_nan_corr_ret(
        self, tmp_path, monkeypatch
    ):
        df = self._build(tmp_path, monkeypatch)
        jun = _row(df, C1, "2015-06")
        assert jun["ret_raw"] == pytest.approx(0.10)       # 110/100 − 1
        assert pd.isna(jun["price_eom_corr"])
        assert pd.isna(jun["ret_corr"]), (
            "a month missing in corr must yield NaN ret_corr — the raw "
            "family's presence must not be used to repair it"
        )

    def test_corr_gap_not_bridged_even_when_panel_rows_are_adjacent(
        self, tmp_path, monkeypatch
    ):
        """July's panel row IS calendar-adjacent to June's (the row exists
        thanks to raw), but corr has no June price — corr's July return must
        be NaN, NOT the gap-bridging (130−100)/100 = 0.30 a 'repairing'
        implementation would produce by chaining over its own NaN."""
        df = self._build(tmp_path, monkeypatch)
        jul = _row(df, C1, "2015-07")
        assert jul["ret_raw"] == pytest.approx(0.10)       # 121/110 − 1
        assert pd.isna(jul["ret_corr"]), (
            "corr return must not bridge over corr's own missing June"
        )

    def test_month_present_in_only_one_family_still_emits_row(
        self, tmp_path, monkeypatch
    ):
        # The outer join keeps the raw-only June row (maximal panel).
        df = self._build(tmp_path, monkeypatch)
        jun = _row(df, C1, "2015-06")
        assert jun["price_eom_raw"] == pytest.approx(110.0)
        assert pd.isna(jun["price_eom_corr"])


# ---------------------------------------------------------------------------
# Excess returns
# ---------------------------------------------------------------------------

class TestExcessReturns:
    def test_xret_equals_ret_minus_rf_per_family(self, tmp_path, monkeypatch):
        raw_rows = [
            _daily_row(C1, "2015-05-15", 100.0, 200_000),
            _daily_row(C1, "2015-06-15", 105.0, 200_000),
        ]
        corr_rows = [
            _daily_row(C1, "2015-05-15", 100.0, 200_000),
            _daily_row(C1, "2015-06-15", 102.0, 200_000),
        ]
        rf_jun = 0.002
        out, _ = _run(tmp_path, monkeypatch, raw_rows, corr_rows,
                      [("2015-05", 0.001), ("2015-06", rf_jun)])
        df = pd.read_parquet(out)
        jun = _row(df, C1, "2015-06")
        assert jun["rf_monthly"] == pytest.approx(rf_jun)
        assert jun["xret_raw"] == pytest.approx(0.05 - rf_jun)
        assert jun["xret_corr"] == pytest.approx(0.02 - rf_jun)


# ---------------------------------------------------------------------------
# last_trade_date per family
# ---------------------------------------------------------------------------

class TestLastTradeDate:
    def test_last_trade_date_comes_from_each_familys_own_daily_input(
        self, tmp_path, monkeypatch
    ):
        # raw trades through 06-25; corr's last surviving day is 06-18
        # (post-distressed-filter, per A5 — must NOT inherit raw's 06-25).
        raw_rows = [
            _daily_row(C1, "2015-06-10", 100.0, 200_000),
            _daily_row(C1, "2015-06-25", 101.0, 200_000),
        ]
        corr_rows = [
            _daily_row(C1, "2015-06-10", 100.0, 200_000),
            _daily_row(C1, "2015-06-18", 100.5, 200_000),
        ]
        out, _ = _run(tmp_path, monkeypatch, raw_rows, corr_rows,
                      [("2015-06", 0.001)])
        df = pd.read_parquet(out)
        row = _row(df, C1, "2015-06")
        assert pd.Timestamp(row["last_trade_date_raw"]) == pd.Timestamp("2015-06-25")
        assert pd.Timestamp(row["last_trade_date_corr"]) == pd.Timestamp("2015-06-18")


# ---------------------------------------------------------------------------
# Engine-contract shape: date, size, exit_reason, cusip, metadata
# ---------------------------------------------------------------------------

class TestOutputContract:
    def _build(self, tmp_path, monkeypatch):
        rows = [
            _daily_row(C1, "2015-06-10", 100.0, 200_000),
            _daily_row(C2, "2015-06-15", 50.0, 300_000),
        ]
        return _run(tmp_path, monkeypatch, rows, rows, [("2015-06", 0.001)])

    def test_required_columns_present(self, tmp_path, monkeypatch):
        out, _ = self._build(tmp_path, monkeypatch)
        df = pd.read_parquet(out)
        for col in [
            "cusip", "date", "size",
            "price_eom_raw", "price_eom_corr",
            "ret_raw", "ret_corr",
            "xret_raw", "xret_corr",
            "n_trades_raw", "n_trades_corr",
            "total_vol_raw", "total_vol_corr",
            "last_trade_date_raw", "last_trade_date_corr",
            "rf_monthly", "exit_reason",
        ]:
            assert col in df.columns, f"missing column: {col}"

    def test_date_is_normalised_month_end(self, tmp_path, monkeypatch):
        out, _ = self._build(tmp_path, monkeypatch)
        df = pd.read_parquet(out)
        d = df["date"].iloc[0]
        assert d == _me("2015-06")
        assert d.tz is None
        assert d == d.normalize()                      # midnight, not 23:59:59
        assert d == d + pd.offsets.MonthEnd(0)         # already month-end

    def test_size_is_placeholder_constant(self, tmp_path, monkeypatch):
        out, _ = self._build(tmp_path, monkeypatch)
        df = pd.read_parquet(out)
        assert (df["size"] == SIZE_PLACEHOLDER).all()

    def test_exit_reason_is_all_nan_placeholder(self, tmp_path, monkeypatch):
        out, _ = self._build(tmp_path, monkeypatch)
        df = pd.read_parquet(out)
        assert df["exit_reason"].isna().all()

    def test_cusip_carried_through_from_daily_layer(self, tmp_path, monkeypatch):
        out, counts = self._build(tmp_path, monkeypatch)
        df = pd.read_parquet(out)
        assert sorted(df["cusip"].unique().tolist()) == sorted([C1, C2])
        assert counts["unique_cusips"] == 2
        assert counts["cusip_month_observations"] == 2

    def test_parquet_metadata_records_maximal_panel_kind(self, tmp_path, monkeypatch):
        out, _ = self._build(tmp_path, monkeypatch)
        meta = pq.read_schema(str(out)).metadata
        assert meta is not None
        assert meta.get(b"panel_kind") == b"maximal"
        assert meta.get(b"primary_key") == b"cusip"
        assert meta.get(b"families") == b"raw,corr"
        assert meta.get(b"size_policy") == (
            f"placeholder_const_{SIZE_PLACEHOLDER}".encode("utf-8")
        )
        assert meta.get(b"survivorship_policy") == b"exit_reason_nan_pre_FISD"

    def test_no_tmp_leftover_after_success(self, tmp_path, monkeypatch):
        self._build(tmp_path, monkeypatch)
        assert not list(tmp_path.glob("*.tmp"))
