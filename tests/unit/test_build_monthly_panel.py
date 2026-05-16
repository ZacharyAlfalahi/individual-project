"""
Unit tests for monthly bond return panel construction.

Covers:
  - VWAP computation (weighted vs simple average)
  - Holding-period return calculation
  - Gap detection: non-consecutive months produce NaN return (not stale carry-forward)
  - Excess return: xret = ret - rf_monthly
  - Institutional volume filter applied before VWAP
  - bond-months with only sub-threshold trades produce no row (filtered out before agg)
  - Output schema matches spec
  - NaN ret/xret for first month of each bond (no prior price)
"""

import gzip
import io
import json
import sys
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "scripts"))
from build_monthly_panel import build_panel, load_config, write_report, OUT_FILE, REPORT_OUT

_cfg = load_config()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_trace_parquet(path: Path, rows: list[dict]) -> None:
    df = pd.DataFrame(rows)
    df["trd_exctn_dt"] = pd.to_datetime(df["trd_exctn_dt"])
    schema = pa.schema([
        pa.field("bond_id",              pa.string()),
        pa.field("trd_exctn_dt",         pa.timestamp("us")),
        pa.field("rptd_pr",              pa.float64()),
        pa.field("entrd_vol_qt",         pa.float64()),
        pa.field("sub_prdct",            pa.string()),
        pa.field("company_symbol",       pa.string()),
        pa.field("trd_exctn_tm",         pa.string()),
        pa.field("rpt_side_cd",          pa.string()),
        pa.field("trdg_mkt_cd",          pa.string()),
        pa.field("trd_mod_3",            pa.string()),
        pa.field("bloomberg_identifier", pa.string()),
        pa.field("scrty_type_cd",        pa.string()),
    ])
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pandas(df, schema=schema, preserve_index=False), str(path))


def _make_rf_parquet(path: Path, months: list[str], rates: list[float]) -> None:
    df = pd.DataFrame({"year_month": months, "rf_monthly": rates})
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pandas(df, preserve_index=False), str(path))


def _base_row(bond_id, date, price, vol, sub_prdct="CORP"):
    return {
        "bond_id": bond_id,
        "trd_exctn_dt": date,
        "rptd_pr": price,
        "entrd_vol_qt": vol,
        "sub_prdct": sub_prdct,
        "company_symbol": "TEST",
        "trd_exctn_tm": "10:00:00",
        "rpt_side_cd": "S",
        "trdg_mkt_cd": "S1",
        "trd_mod_3": "",
        "bloomberg_identifier": "",
        "scrty_type_cd": "",
    }


# ---------------------------------------------------------------------------
# TestVWAP
# ---------------------------------------------------------------------------

class TestVWAP:
    def test_single_trade_price_is_trade_price(self, tmp_path, monkeypatch):
        rows = [_base_row("B1", "2015-06-15", 100.0, 200_000)]
        rf = [("2015-06", 0.001)]
        trace = tmp_path / "trace.parquet"
        rf_path = tmp_path / "rf.parquet"
        out = tmp_path / "panel.parquet"
        _make_trace_parquet(trace, rows)
        _make_rf_parquet(rf_path, [r[0] for r in rf], [r[1] for r in rf])

        import build_monthly_panel as bmp
        monkeypatch.setattr(bmp, "TRACE_FILE", trace)
        monkeypatch.setattr(bmp, "RF_FILE", rf_path)
        monkeypatch.setattr(bmp, "OUT_FILE", out)
        monkeypatch.setattr(bmp, "REPORT_OUT", tmp_path / "report.json")

        counts = build_panel(_cfg)
        df = pd.read_parquet(out)
        row = df[df["bond_id"] == "B1"]
        assert len(row) == 1
        assert row["price_eom"].iloc[0] == pytest.approx(100.0)

    def test_vwap_weights_by_volume(self, tmp_path, monkeypatch):
        rows = [
            _base_row("B1", "2015-06-10", 100.0, 100_000),
            _base_row("B1", "2015-06-20", 200.0, 300_000),
        ]
        trace = tmp_path / "trace.parquet"
        rf_path = tmp_path / "rf.parquet"
        out = tmp_path / "panel.parquet"
        _make_trace_parquet(trace, rows)
        _make_rf_parquet(rf_path, ["2015-06"], [0.001])

        import build_monthly_panel as bmp
        monkeypatch.setattr(bmp, "TRACE_FILE", trace)
        monkeypatch.setattr(bmp, "RF_FILE", rf_path)
        monkeypatch.setattr(bmp, "OUT_FILE", out)
        monkeypatch.setattr(bmp, "REPORT_OUT", tmp_path / "report.json")

        build_panel(_cfg)
        df = pd.read_parquet(out)
        row = df[df["bond_id"] == "B1"]
        # VWAP: (100*100k + 200*300k) / 400k = 175.0
        assert row["price_eom"].iloc[0] == pytest.approx(175.0)

    def test_vwap_differs_from_simple_average(self, tmp_path, monkeypatch):
        rows = [
            _base_row("B1", "2015-06-10", 100.0, 100_000),
            _base_row("B1", "2015-06-20", 200.0, 300_000),
        ]
        trace = tmp_path / "trace.parquet"
        rf_path = tmp_path / "rf.parquet"
        out = tmp_path / "panel.parquet"
        _make_trace_parquet(trace, rows)
        _make_rf_parquet(rf_path, ["2015-06"], [0.001])

        import build_monthly_panel as bmp
        monkeypatch.setattr(bmp, "TRACE_FILE", trace)
        monkeypatch.setattr(bmp, "RF_FILE", rf_path)
        monkeypatch.setattr(bmp, "OUT_FILE", out)
        monkeypatch.setattr(bmp, "REPORT_OUT", tmp_path / "report.json")

        build_panel(_cfg)
        df = pd.read_parquet(out)
        vwap = df[df["bond_id"] == "B1"]["price_eom"].iloc[0]
        simple_avg = 150.0
        assert vwap != pytest.approx(simple_avg)


# ---------------------------------------------------------------------------
# TestReturns
# ---------------------------------------------------------------------------

class TestReturns:
    def _two_month_panel(self, tmp_path, monkeypatch, p0, p1, rf=0.001):
        rows = [
            _base_row("B1", "2015-05-15", p0, 200_000),
            _base_row("B1", "2015-06-15", p1, 200_000),
        ]
        trace = tmp_path / "trace.parquet"
        rf_path = tmp_path / "rf.parquet"
        out = tmp_path / "panel.parquet"
        _make_trace_parquet(trace, rows)
        _make_rf_parquet(rf_path, ["2015-05", "2015-06"], [rf, rf])

        import build_monthly_panel as bmp
        monkeypatch.setattr(bmp, "TRACE_FILE", trace)
        monkeypatch.setattr(bmp, "RF_FILE", rf_path)
        monkeypatch.setattr(bmp, "OUT_FILE", out)
        monkeypatch.setattr(bmp, "REPORT_OUT", tmp_path / "report.json")

        build_panel(_cfg)
        return pd.read_parquet(out).sort_values("year_month")

    def test_first_month_ret_is_nan(self, tmp_path, monkeypatch):
        df = self._two_month_panel(tmp_path, monkeypatch, 100.0, 105.0)
        first = df[df["year_month"] == "2015-05"]
        assert first["ret"].isna().all()

    def test_second_month_ret_correct(self, tmp_path, monkeypatch):
        df = self._two_month_panel(tmp_path, monkeypatch, 100.0, 105.0)
        second = df[df["year_month"] == "2015-06"]
        assert second["ret"].iloc[0] == pytest.approx(0.05)

    def test_xret_equals_ret_minus_rf(self, tmp_path, monkeypatch):
        rf = 0.002
        df = self._two_month_panel(tmp_path, monkeypatch, 100.0, 105.0, rf=rf)
        second = df[df["year_month"] == "2015-06"]
        assert second["xret"].iloc[0] == pytest.approx(0.05 - rf)

    def test_gap_in_months_produces_nan_return(self, tmp_path, monkeypatch):
        rows = [
            _base_row("B1", "2015-03-15", 100.0, 200_000),
            _base_row("B1", "2015-06-15", 105.0, 200_000),  # 3-month gap
        ]
        trace = tmp_path / "trace.parquet"
        rf_path = tmp_path / "rf.parquet"
        out = tmp_path / "panel.parquet"
        _make_trace_parquet(trace, rows)
        _make_rf_parquet(rf_path, ["2015-03", "2015-06"], [0.001, 0.001])

        import build_monthly_panel as bmp
        monkeypatch.setattr(bmp, "TRACE_FILE", trace)
        monkeypatch.setattr(bmp, "RF_FILE", rf_path)
        monkeypatch.setattr(bmp, "OUT_FILE", out)
        monkeypatch.setattr(bmp, "REPORT_OUT", tmp_path / "report.json")

        build_panel(_cfg)
        df = pd.read_parquet(out).sort_values("year_month")
        jun = df[df["year_month"] == "2015-06"]
        # Gap of 3 months — return must be NaN, not (105-100)/100
        assert jun["ret"].isna().all(), "Return across a gap must be NaN"
        assert jun["xret"].isna().all(), "xret across a gap must be NaN"

    def test_consecutive_months_do_not_gap(self, tmp_path, monkeypatch):
        rows = [
            _base_row("B1", "2015-05-15", 100.0, 200_000),
            _base_row("B1", "2015-06-15", 105.0, 200_000),
        ]
        trace = tmp_path / "trace.parquet"
        rf_path = tmp_path / "rf.parquet"
        out = tmp_path / "panel.parquet"
        _make_trace_parquet(trace, rows)
        _make_rf_parquet(rf_path, ["2015-05", "2015-06"], [0.001, 0.001])

        import build_monthly_panel as bmp
        monkeypatch.setattr(bmp, "TRACE_FILE", trace)
        monkeypatch.setattr(bmp, "RF_FILE", rf_path)
        monkeypatch.setattr(bmp, "OUT_FILE", out)
        monkeypatch.setattr(bmp, "REPORT_OUT", tmp_path / "report.json")

        build_panel(_cfg)
        df = pd.read_parquet(out).sort_values("year_month")
        jun = df[df["year_month"] == "2015-06"]
        assert jun["ret"].notna().all()


# ---------------------------------------------------------------------------
# TestVolumeFilter
# ---------------------------------------------------------------------------

class TestVolumeFilter:
    def test_sub_threshold_trades_excluded_from_vwap(self, tmp_path, monkeypatch):
        rows = [
            _base_row("B1", "2015-06-10", 90.0, 50_000),   # below min_vol_qt — excluded
            _base_row("B1", "2015-06-20", 100.0, 200_000),  # above — included
        ]
        trace = tmp_path / "trace.parquet"
        rf_path = tmp_path / "rf.parquet"
        out = tmp_path / "panel.parquet"
        _make_trace_parquet(trace, rows)
        _make_rf_parquet(rf_path, ["2015-06"], [0.001])

        import build_monthly_panel as bmp
        monkeypatch.setattr(bmp, "TRACE_FILE", trace)
        monkeypatch.setattr(bmp, "RF_FILE", rf_path)
        monkeypatch.setattr(bmp, "OUT_FILE", out)
        monkeypatch.setattr(bmp, "REPORT_OUT", tmp_path / "report.json")

        build_panel(_cfg)
        df = pd.read_parquet(out)
        row = df[df["bond_id"] == "B1"]
        # VWAP should be 100.0 (only the passing trade counts)
        assert row["price_eom"].iloc[0] == pytest.approx(100.0)

    def test_all_sub_threshold_bond_month_produces_no_row(self, tmp_path, monkeypatch):
        rows = [_base_row("B1", "2015-06-10", 100.0, 50_000)]  # all below threshold
        trace = tmp_path / "trace.parquet"
        rf_path = tmp_path / "rf.parquet"
        out = tmp_path / "panel.parquet"
        _make_trace_parquet(trace, rows)
        _make_rf_parquet(rf_path, ["2015-06"], [0.001])

        import build_monthly_panel as bmp
        monkeypatch.setattr(bmp, "TRACE_FILE", trace)
        monkeypatch.setattr(bmp, "RF_FILE", rf_path)
        monkeypatch.setattr(bmp, "OUT_FILE", out)
        monkeypatch.setattr(bmp, "REPORT_OUT", tmp_path / "report.json")

        build_panel(_cfg)
        df = pd.read_parquet(out)
        assert len(df) == 0


# ---------------------------------------------------------------------------
# TestSubPrdctFilter
# ---------------------------------------------------------------------------

class TestSubPrdctFilter:
    def _run(self, tmp_path, monkeypatch, rows):
        trace = tmp_path / "trace.parquet"
        rf_path = tmp_path / "rf.parquet"
        out = tmp_path / "panel.parquet"
        _make_trace_parquet(trace, rows)
        _make_rf_parquet(rf_path, ["2015-06"], [0.001])

        import build_monthly_panel as bmp
        monkeypatch.setattr(bmp, "TRACE_FILE", trace)
        monkeypatch.setattr(bmp, "RF_FILE", rf_path)
        monkeypatch.setattr(bmp, "OUT_FILE", out)
        monkeypatch.setattr(bmp, "REPORT_OUT", tmp_path / "report.json")

        build_panel(_cfg)
        return pd.read_parquet(out)

    def test_corp_included(self, tmp_path, monkeypatch):
        rows = [_base_row("B1", "2015-06-15", 100.0, 200_000, sub_prdct="CORP")]
        df = self._run(tmp_path, monkeypatch, rows)
        assert "B1" in df["bond_id"].values

    def test_chrc_excluded(self, tmp_path, monkeypatch):
        rows = [_base_row("B1", "2015-06-15", 100.0, 200_000, sub_prdct="CHRC")]
        df = self._run(tmp_path, monkeypatch, rows)
        assert "B1" not in df["bond_id"].values

    def test_eln_excluded(self, tmp_path, monkeypatch):
        rows = [_base_row("B1", "2015-06-15", 100.0, 200_000, sub_prdct="ELN")]
        df = self._run(tmp_path, monkeypatch, rows)
        assert "B1" not in df["bond_id"].values

    def test_null_subprdct_included(self, tmp_path, monkeypatch):
        row = _base_row("B1", "2015-06-15", 100.0, 200_000)
        row["sub_prdct"] = None  # pre-2012 record
        df = self._run(tmp_path, monkeypatch, [row])
        assert "B1" in df["bond_id"].values


# ---------------------------------------------------------------------------
# TestOutputSchema
# ---------------------------------------------------------------------------

class TestOutputSchema:
    def test_required_columns_present(self, tmp_path, monkeypatch):
        rows = [
            _base_row("B1", "2015-05-15", 100.0, 200_000),
            _base_row("B1", "2015-06-15", 105.0, 200_000),
        ]
        trace = tmp_path / "trace.parquet"
        rf_path = tmp_path / "rf.parquet"
        out = tmp_path / "panel.parquet"
        _make_trace_parquet(trace, rows)
        _make_rf_parquet(rf_path, ["2015-05", "2015-06"], [0.001, 0.001])

        import build_monthly_panel as bmp
        monkeypatch.setattr(bmp, "TRACE_FILE", trace)
        monkeypatch.setattr(bmp, "RF_FILE", rf_path)
        monkeypatch.setattr(bmp, "OUT_FILE", out)
        monkeypatch.setattr(bmp, "REPORT_OUT", tmp_path / "report.json")

        build_panel(_cfg)
        df = pd.read_parquet(out)
        for col in ["bond_id", "year_month", "price_eom", "ret", "xret",
                    "n_trades", "total_vol", "rf_monthly"]:
            assert col in df.columns, f"Missing column: {col}"

    def test_n_trades_correct(self, tmp_path, monkeypatch):
        rows = [
            _base_row("B1", "2015-06-10", 100.0, 200_000),
            _base_row("B1", "2015-06-20", 105.0, 200_000),
        ]
        trace = tmp_path / "trace.parquet"
        rf_path = tmp_path / "rf.parquet"
        out = tmp_path / "panel.parquet"
        _make_trace_parquet(trace, rows)
        _make_rf_parquet(rf_path, ["2015-06"], [0.001])

        import build_monthly_panel as bmp
        monkeypatch.setattr(bmp, "TRACE_FILE", trace)
        monkeypatch.setattr(bmp, "RF_FILE", rf_path)
        monkeypatch.setattr(bmp, "OUT_FILE", out)
        monkeypatch.setattr(bmp, "REPORT_OUT", tmp_path / "report.json")

        build_panel(_cfg)
        df = pd.read_parquet(out)
        assert df[df["bond_id"] == "B1"]["n_trades"].iloc[0] == 2
