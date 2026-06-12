"""
Unit tests for the daily aggregation layer (scripts/build_daily_panel.py).

aggregate_daily() collapses the trade-level parquet to (cusip_id,
trd_exctn_dt) rows with VWAP, min/max price, n_trades and total_vol,
after applying the three trade-level filters that mirror the monthly
build's pre-aggregation pool:

  - cusip_id not null / not blank
  - entrd_vol_qt ≥ min_vol_qt
  - sub_prdct null OR in sub_prdct_keep   (CORP + "" kept; CHRC/ELN dropped)

All filter parameters come from docs/thresholds.yaml via the script's own
load_config() — never duplicated as literals here.
"""

from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from build_daily_panel import aggregate_daily, load_config

_cfg = load_config()
_MIN_VOL = float(_cfg["min_vol_qt"])
# A volume comfortably above / below the institutional-size threshold,
# derived from the loader so the tests track the yaml.
VOL_OK = _MIN_VOL * 2
VOL_LOW = _MIN_VOL / 2


# ---------------------------------------------------------------------------
# Helpers — synthetic trade-level parquet (preprocess_trace OUTPUT_SCHEMA)
# ---------------------------------------------------------------------------

TRACE_SCHEMA = pa.schema([
    pa.field("bond_id",              pa.string()),
    pa.field("cusip_id",             pa.string()),
    pa.field("company_symbol",       pa.string()),
    pa.field("trd_exctn_dt",         pa.timestamp("us")),
    pa.field("trd_exctn_tm",         pa.string()),
    pa.field("rptd_pr",              pa.float64()),
    pa.field("entrd_vol_qt",         pa.float64()),
    pa.field("sub_prdct",            pa.string()),
    pa.field("rpt_side_cd",          pa.string()),
    pa.field("trdg_mkt_cd",          pa.string()),
    pa.field("trd_mod_3",            pa.string()),
    pa.field("bloomberg_identifier", pa.string()),
    pa.field("scrty_type_cd",        pa.string()),
])


def _trade(cusip, date, price, vol, sub_prdct="CORP"):
    return {
        "bond_id": "SYM1",
        "cusip_id": cusip,
        "company_symbol": "TEST",
        "trd_exctn_dt": date,
        "trd_exctn_tm": "10:00:00",
        "rptd_pr": price,
        "entrd_vol_qt": vol,
        "sub_prdct": sub_prdct,
        "rpt_side_cd": "S",
        "trdg_mkt_cd": "S1",
        "trd_mod_3": "",
        "bloomberg_identifier": "",
        "scrty_type_cd": "",
    }


def _run(tmp_path: Path, rows: list[dict]):
    """Write a synthetic trade parquet, aggregate, return (df, counts)."""
    inp = tmp_path / "trace_clean.parquet"
    out = tmp_path / "trace_daily.parquet"
    df = pd.DataFrame(rows)
    df["trd_exctn_dt"] = pd.to_datetime(df["trd_exctn_dt"])
    pq.write_table(
        pa.Table.from_pandas(df, schema=TRACE_SCHEMA, preserve_index=False),
        str(inp),
    )
    counts = aggregate_daily(inp, out, _cfg)
    return pd.read_parquet(out), counts, out


C1 = "CUSIP0001"
C2 = "CUSIP0002"


# ---------------------------------------------------------------------------
# Aggregation arithmetic
# ---------------------------------------------------------------------------

class TestDailyAggregation:
    def test_vwap_min_max_ntrades_totalvol_hand_computed(self, tmp_path):
        rows = [
            _trade(C1, "2015-06-10", 100.0, VOL_OK),        # vol 200k
            _trade(C1, "2015-06-10", 110.0, VOL_OK * 3),    # vol 600k
            _trade(C1, "2015-06-10", 95.0,  VOL_OK),        # vol 200k
        ]
        df, _, _ = _run(tmp_path, rows)
        assert len(df) == 1
        row = df.iloc[0]
        # VWAP = (100×200k + 110×600k + 95×200k) / 1,000k = 105.0
        expected_vwap = (
            100.0 * VOL_OK + 110.0 * VOL_OK * 3 + 95.0 * VOL_OK
        ) / (VOL_OK * 5)
        assert row["price_vwap"] == pytest.approx(expected_vwap, abs=1e-9)
        assert row["price_vwap"] == pytest.approx(105.0)
        assert row["min_price"] == pytest.approx(95.0)
        assert row["max_price"] == pytest.approx(110.0)
        assert row["n_trades"] == 3
        assert row["total_vol"] == pytest.approx(VOL_OK * 5)

    def test_vwap_differs_from_simple_average(self, tmp_path):
        rows = [
            _trade(C1, "2015-06-10", 100.0, VOL_OK),
            _trade(C1, "2015-06-10", 200.0, VOL_OK * 3),
        ]
        df, _, _ = _run(tmp_path, rows)
        # (100×1 + 200×3) / 4 = 175 ≠ simple average 150
        assert df.iloc[0]["price_vwap"] == pytest.approx(175.0)

    def test_grouping_is_per_cusip_per_day(self, tmp_path):
        rows = [
            _trade(C1, "2015-06-10", 100.0, VOL_OK),
            _trade(C1, "2015-06-11", 101.0, VOL_OK),
            _trade(C2, "2015-06-10", 50.0, VOL_OK),
        ]
        df, _, _ = _run(tmp_path, rows)
        assert len(df) == 3
        keys = set(zip(df["cusip_id"], pd.to_datetime(df["trd_exctn_dt"]).dt.strftime("%Y-%m-%d")))
        assert keys == {
            (C1, "2015-06-10"), (C1, "2015-06-11"), (C2, "2015-06-10"),
        }


# ---------------------------------------------------------------------------
# Trade-level filters (params from load_config / thresholds.yaml)
# ---------------------------------------------------------------------------

class TestMinVolFilter:
    def test_sub_threshold_trade_excluded_from_aggregates(self, tmp_path):
        rows = [
            _trade(C1, "2015-06-10", 90.0, VOL_LOW),   # below min_vol_qt
            _trade(C1, "2015-06-10", 100.0, VOL_OK),
        ]
        df, _, _ = _run(tmp_path, rows)
        row = df.iloc[0]
        # The retail-size trade contributes to nothing: vwap, min/max,
        # n_trades and total_vol all reflect only the institutional trade.
        assert row["price_vwap"] == pytest.approx(100.0)
        assert row["min_price"] == pytest.approx(100.0)
        assert row["n_trades"] == 1
        assert row["total_vol"] == pytest.approx(VOL_OK)

    def test_threshold_is_inclusive(self, tmp_path):
        # Source filter is entrd_vol_qt >= min_vol — a trade AT the
        # threshold is kept.
        rows = [_trade(C1, "2015-06-10", 100.0, _MIN_VOL)]
        df, counts, _ = _run(tmp_path, rows)
        assert counts["rows"] == 1
        assert df.iloc[0]["n_trades"] == 1

    def test_all_sub_threshold_day_produces_no_row(self, tmp_path):
        rows = [_trade(C1, "2015-06-10", 100.0, VOL_LOW)]
        df, counts, _ = _run(tmp_path, rows)
        assert len(df) == 0
        assert counts["rows"] == 0


class TestSubPrdctFilter:
    def test_corp_kept(self, tmp_path):
        df, _, _ = _run(tmp_path, [_trade(C1, "2015-06-10", 100.0, VOL_OK, sub_prdct="CORP")])
        assert C1 in df["cusip_id"].values

    def test_chrc_dropped(self, tmp_path):
        rows = [
            _trade(C1, "2015-06-10", 100.0, VOL_OK, sub_prdct="CHRC"),
            _trade(C2, "2015-06-10", 100.0, VOL_OK, sub_prdct="CORP"),
        ]
        df, _, _ = _run(tmp_path, rows)
        assert C1 not in df["cusip_id"].values
        assert C2 in df["cusip_id"].values

    def test_eln_dropped(self, tmp_path):
        df, _, _ = _run(tmp_path, [_trade(C1, "2015-06-10", 100.0, VOL_OK, sub_prdct="ELN")])
        assert len(df) == 0

    def test_null_sub_prdct_kept(self, tmp_path):
        # Pre-2012 records carry NULL sub_prdct; the source filter is
        # is_null() OR is_in(keep) — nulls are explicitly KEPT.
        row = _trade(C1, "2015-06-10", 100.0, VOL_OK)
        row["sub_prdct"] = None
        df, _, _ = _run(tmp_path, [row])
        assert C1 in df["cusip_id"].values

    def test_empty_string_sub_prdct_kept(self, tmp_path):
        # "" is in sub_prdct_keep per thresholds.yaml.
        assert "" in _cfg["sub_prdct_keep"]
        df, _, _ = _run(tmp_path, [_trade(C1, "2015-06-10", 100.0, VOL_OK, sub_prdct="")])
        assert C1 in df["cusip_id"].values


class TestCusipHandling:
    def test_blank_cusip_dropped(self, tmp_path):
        rows = [
            _trade("", "2015-06-10", 100.0, VOL_OK),
            _trade(C2, "2015-06-10", 105.0, VOL_OK),
        ]
        df, counts, _ = _run(tmp_path, rows)
        assert list(df["cusip_id"].unique()) == [C2]
        assert counts["rows"] == 1

    def test_null_cusip_dropped(self, tmp_path):
        row = _trade(C1, "2015-06-10", 100.0, VOL_OK)
        row["cusip_id"] = None
        rows = [row, _trade(C2, "2015-06-10", 105.0, VOL_OK)]
        df, counts, _ = _run(tmp_path, rows)
        assert list(df["cusip_id"].unique()) == [C2]
        assert counts["rows"] == 1

    def test_cusip_id_values_carried_through(self, tmp_path):
        rows = [
            _trade(C1, "2015-06-10", 100.0, VOL_OK),
            _trade(C2, "2015-06-10", 50.0, VOL_OK),
        ]
        df, _, _ = _run(tmp_path, rows)
        assert sorted(df["cusip_id"].unique().tolist()) == sorted([C1, C2])


# ---------------------------------------------------------------------------
# Output integrity
# ---------------------------------------------------------------------------

class TestOutputIntegrity:
    def test_counts_dict_matches_written_parquet(self, tmp_path):
        rows = [
            _trade(C1, "2015-06-10", 100.0, VOL_OK),
            _trade(C1, "2015-06-11", 101.0, VOL_OK),
            _trade(C2, "2015-06-10", 50.0, VOL_OK),
            _trade(C2, "2015-06-10", 51.0, VOL_OK),   # same day → same row
        ]
        df, counts, out = _run(tmp_path, rows)
        assert counts["rows"] == 3
        assert counts["unique_cusips"] == 2
        assert counts["rows"] == len(df)
        assert counts["rows"] == pq.read_metadata(str(out)).num_rows
        assert counts["unique_cusips"] == df["cusip_id"].nunique()

    def test_output_columns_and_no_tmp_leftover(self, tmp_path):
        rows = [_trade(C1, "2015-06-10", 100.0, VOL_OK)]
        df, _, out = _run(tmp_path, rows)
        assert list(df.columns) == [
            "cusip_id", "trd_exctn_dt", "price_vwap", "total_vol",
            "n_trades", "min_price", "max_price",
        ]
        assert not list(out.parent.glob("*.tmp"))

    def test_missing_input_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            aggregate_daily(tmp_path / "nope.parquet", tmp_path / "out.parquet", _cfg)
