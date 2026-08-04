"""
Unit tests for the partition-level distressed-filter orchestration
(scripts/apply_distressed_filters.py :: process_partition).

The per-filter primitives (filter_anomaly / filter_spike / filter_plateau /
filter_intraday) are covered by tests/unit/test_meas_err_injection.py; this
module pins the orchestration on the corr DAILY panel:

  - row conservation: kept_rows + dropped_total == input_rows
  - the dropped companion parquet exists, conforms to DROPPED_SCHEMA, and
    its per-filter boolean flag columns mark exactly the right rows
  - the clean cusip passes through bit-identical
  - no *.tmp leftovers after a successful run

One cusip per filter case, reusing the injection-series shapes from
test_meas_err_injection.py, plus one clean cusip. All parameters come from
docs/thresholds.yaml via the script's own load_config().
"""

from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from apply_distressed_filters import (
    DROPPED_SCHEMA,
    load_config,
    process_partition,
)

PARAMS = load_config()


# ---------------------------------------------------------------------------
# Synthetic corr daily panel — one cusip per filter case + one clean cusip
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

ANOM = "CUSANOM01"
SPIK = "CUSSPIK01"
PLAT = "CUSPLAT01"
INTR = "CUSINTR01"
CLEAN = "CUSCLEAN1"

# Injection-series shapes (DRR ratio/ultra-low semantics, prices in % of par):
#   anomaly : isolated ultra-low print (0.05 < ultra_low_threshold) whose
#             above-neighbour median / price >> min_normal_price_ratio
#   spike   : distressed ~2 series with a 10 print (> high_spike_threshold level
#             gate); 10/median_pre(2)=5 >= min_spike_ratio, recovers next day
#   plateau : 3-day run at 0.50 (a suspicious round number) between ~1.2
#             neighbours — round runs are always suspicious
#   intraday: low day (min 5 < intraday_price_threshold) with (max-min)/mean > 0.75
# min/max are set equal to vwap everywhere except the intraday case so Filter 4
# cannot fire on the non-intraday cusips (zero range). Candidate levels are
# chosen so only the intended filter fires (see the flagged_*_only assertions).
ANOM_PRICES = [10.0, 10.5, 9.5, 10.2, 0.05, 10.1, 10.3, 9.8, 10.0, 10.2]
ANOM_FLAGGED = [4]

SPIK_PRICES = [2.0, 2.0, 2.0, 2.0, 2.0, 10.0, 2.0, 2.0]
SPIK_FLAGGED = [5]

PLAT_PRICES = [1.2, 1.3, 1.1, 0.50, 0.50, 0.50, 1.3, 1.2]
PLAT_FLAGGED = [3, 4, 5]

INTR_VWAP = [100.0, 100.0, 10.0, 100.0]
INTR_MIN = [100.0, 100.0, 5.0, 100.0]
INTR_MAX = [100.0, 100.0, 15.0, 100.0]
INTR_FLAGGED = [2]

CLEAN_PRICES = [101.3, 102.1, 100.7, 101.9, 100.2]

N_INPUT = (
    len(ANOM_PRICES) + len(SPIK_PRICES) + len(PLAT_PRICES)
    + len(INTR_VWAP) + len(CLEAN_PRICES)
)  # 35
N_DROPPED = len(ANOM_FLAGGED) + len(SPIK_FLAGGED) + len(PLAT_FLAGGED) + len(INTR_FLAGGED)  # 6


def _day(i: int) -> pd.Timestamp:
    return pd.Timestamp("2015-06-01") + pd.Timedelta(days=i)


def _series_rows(cusip, vwaps, mins=None, maxs=None):
    mins = mins if mins is not None else vwaps
    maxs = maxs if maxs is not None else vwaps
    return [
        {
            "cusip_id": cusip,
            "trd_exctn_dt": _day(i),
            "price_vwap": float(v),
            "total_vol": 200_000.0,
            "n_trades": 2,
            "min_price": float(lo),
            "max_price": float(hi),
        }
        for i, (v, lo, hi) in enumerate(zip(vwaps, mins, maxs))
    ]


def _write_input(path: Path) -> pd.DataFrame:
    rows = (
        _series_rows(ANOM, ANOM_PRICES)
        + _series_rows(SPIK, SPIK_PRICES)
        + _series_rows(PLAT, PLAT_PRICES)
        + _series_rows(INTR, INTR_VWAP, INTR_MIN, INTR_MAX)
        + _series_rows(CLEAN, CLEAN_PRICES)
    )
    df = pd.DataFrame(rows)
    pq.write_table(
        pa.Table.from_pandas(df, schema=DAILY_SCHEMA, preserve_index=False),
        str(path),
    )
    return df


@pytest.fixture
def run(tmp_path):
    """Run process_partition once on the synthetic panel.
    Returns (input_df, kept_df, dropped_df, counts, paths)."""
    inp = tmp_path / "trace_daily_corr.parquet"
    out = tmp_path / "trace_daily_corr_filtered.parquet"
    dropped = tmp_path / "distressed_dropped_dev.parquet"
    input_df = _write_input(inp)
    counts = process_partition(inp, out, dropped, PARAMS)
    kept_df = pd.read_parquet(out)
    dropped_df = pd.read_parquet(dropped)
    return input_df, kept_df, dropped_df, counts, (inp, out, dropped, tmp_path)


def _flags_of(dropped_df, cusip, day_idx) -> pd.Series:
    cell = dropped_df[
        (dropped_df["cusip_id"] == cusip)
        & (pd.to_datetime(dropped_df["trd_exctn_dt"]) == _day(day_idx))
    ]
    assert len(cell) == 1, f"expected one dropped row for {cusip}@day{day_idx}"
    return cell.iloc[0]


# ---------------------------------------------------------------------------
# Parameter-drift guards — if thresholds.yaml changes such that the
# synthetic shapes no longer exercise the intended filters, fail loudly
# here rather than producing confusing downstream assertions.
# ---------------------------------------------------------------------------

class TestFixtureValidUnderCurrentParams:
    def test_anomaly_case_valid(self):
        assert ANOM_PRICES[4] < float(PARAMS["ultra_low_threshold"])

    def test_spike_case_valid(self):
        assert SPIK_PRICES[SPIK_FLAGGED[0]] > float(PARAMS["high_spike_threshold"])

    def test_plateau_level_is_round(self):
        rounds = [float(r) for r in PARAMS["suspicious_round_numbers"]]
        tol = float(PARAMS["round_tolerance"])
        level = PLAT_PRICES[3]
        assert any(abs(level - r) < tol for r in rounds)
        assert len(PLAT_FLAGGED) >= int(PARAMS["min_plateau_days"])

    def test_intraday_case_valid(self):
        i = INTR_FLAGGED[0]
        assert INTR_MIN[i] < float(PARAMS["intraday_price_threshold"])
        mean = (INTR_MIN[i] + INTR_MAX[i]) / 2.0
        assert (INTR_MAX[i] - INTR_MIN[i]) / mean > float(PARAMS["intraday_range_threshold"])


# ---------------------------------------------------------------------------
# Row conservation + counts
# ---------------------------------------------------------------------------

class TestRowConservation:
    def test_kept_plus_dropped_equals_input(self, run):
        _, kept_df, dropped_df, counts, _ = run
        assert counts["input_rows"] == N_INPUT
        assert counts["kept_rows"] + counts["dropped_total"] == counts["input_rows"]
        assert len(kept_df) == counts["kept_rows"]
        assert len(dropped_df) == counts["dropped_total"]

    def test_per_filter_drop_counts(self, run):
        _, _, _, counts, _ = run
        assert counts["dropped_anomaly"] == len(ANOM_FLAGGED)
        assert counts["dropped_spike"] == len(SPIK_FLAGGED)
        assert counts["dropped_plateau"] == len(PLAT_FLAGGED)
        assert counts["dropped_intraday"] == len(INTR_FLAGGED)
        assert counts["dropped_total"] == N_DROPPED

    def test_no_row_is_in_both_outputs(self, run):
        _, kept_df, dropped_df, _, _ = run
        key = ["cusip_id", "trd_exctn_dt"]
        kept_keys = set(map(tuple, kept_df[key].itertuples(index=False)))
        dropped_keys = set(map(tuple, dropped_df[key].itertuples(index=False)))
        assert kept_keys.isdisjoint(dropped_keys)


# ---------------------------------------------------------------------------
# Dropped companion artefact
# ---------------------------------------------------------------------------

class TestDroppedCompanion:
    def test_companion_exists_and_conforms_to_dropped_schema(self, run):
        *_, (inp, out, dropped, tmp) = run
        assert dropped.exists()
        actual = pq.read_schema(str(dropped))
        assert actual.names == DROPPED_SCHEMA.names
        for field in DROPPED_SCHEMA:
            assert actual.field(field.name).type == field.type, (
                f"dropped column {field.name}: {actual.field(field.name).type} "
                f"!= {field.type}"
            )

    def test_anomaly_row_flagged_anomaly_only(self, run):
        _, _, dropped_df, _, _ = run
        row = _flags_of(dropped_df, ANOM, ANOM_FLAGGED[0])
        assert row["flagged_anomaly"]
        assert not row["flagged_spike"]
        assert not row["flagged_plateau"]
        assert not row["flagged_intraday"]
        assert row["price_vwap"] == pytest.approx(ANOM_PRICES[ANOM_FLAGGED[0]])

    def test_spike_row_flagged_spike_only(self, run):
        _, _, dropped_df, _, _ = run
        row = _flags_of(dropped_df, SPIK, SPIK_FLAGGED[0])
        assert row["flagged_spike"]
        assert not row["flagged_anomaly"]
        assert not row["flagged_plateau"]
        assert not row["flagged_intraday"]

    def test_plateau_run_flagged_plateau_only(self, run):
        _, _, dropped_df, _, _ = run
        for i in PLAT_FLAGGED:
            row = _flags_of(dropped_df, PLAT, i)
            assert row["flagged_plateau"], f"day {i} of plateau run not flagged"
            assert not row["flagged_anomaly"]
            assert not row["flagged_spike"]
            assert not row["flagged_intraday"]

    def test_intraday_row_flagged_intraday_only(self, run):
        _, _, dropped_df, _, _ = run
        row = _flags_of(dropped_df, INTR, INTR_FLAGGED[0])
        assert row["flagged_intraday"]
        assert not row["flagged_anomaly"]
        assert not row["flagged_spike"]
        assert not row["flagged_plateau"]

    def test_every_dropped_row_has_at_least_one_flag(self, run):
        _, _, dropped_df, _, _ = run
        any_flag = (
            dropped_df["flagged_anomaly"] | dropped_df["flagged_spike"]
            | dropped_df["flagged_plateau"] | dropped_df["flagged_intraday"]
        )
        assert any_flag.all()

    def test_clean_cusip_never_appears_in_dropped(self, run):
        _, _, dropped_df, _, _ = run
        assert CLEAN not in dropped_df["cusip_id"].values


# ---------------------------------------------------------------------------
# Clean pass-through + filtered output
# ---------------------------------------------------------------------------

class TestKeptOutput:
    def test_clean_cusip_passes_through_untouched(self, run):
        input_df, kept_df, _, _, _ = run
        cols = ["cusip_id", "trd_exctn_dt", "price_vwap", "total_vol",
                "n_trades", "min_price", "max_price"]
        got = (
            kept_df[kept_df["cusip_id"] == CLEAN][cols]
            .sort_values("trd_exctn_dt").reset_index(drop=True)
        )
        want = (
            input_df[input_df["cusip_id"] == CLEAN][cols]
            .sort_values("trd_exctn_dt").reset_index(drop=True)
        )
        pd.testing.assert_frame_equal(got, want, check_dtype=False)

    def test_flagged_days_absent_from_kept_output(self, run):
        _, kept_df, _, _, _ = run
        dts = pd.to_datetime(kept_df["trd_exctn_dt"])
        for cusip, flagged in [
            (ANOM, ANOM_FLAGGED), (SPIK, SPIK_FLAGGED),
            (PLAT, PLAT_FLAGGED), (INTR, INTR_FLAGGED),
        ]:
            for i in flagged:
                hit = kept_df[(kept_df["cusip_id"] == cusip) & (dts == _day(i))]
                assert hit.empty, f"{cusip} day {i} should have been dropped"

    def test_unflagged_days_retained_per_cusip(self, run):
        _, kept_df, _, _, _ = run
        expect = {
            ANOM: len(ANOM_PRICES) - len(ANOM_FLAGGED),
            SPIK: len(SPIK_PRICES) - len(SPIK_FLAGGED),
            PLAT: len(PLAT_PRICES) - len(PLAT_FLAGGED),
            INTR: len(INTR_VWAP) - len(INTR_FLAGGED),
            CLEAN: len(CLEAN_PRICES),
        }
        got = kept_df.groupby("cusip_id").size().to_dict()
        assert got == expect


# ---------------------------------------------------------------------------
# Hygiene
# ---------------------------------------------------------------------------

class TestHygiene:
    def test_no_tmp_leftovers_after_success(self, run):
        *_, (inp, out, dropped, tmp) = run
        assert not list(tmp.glob("*.tmp")), "stale .tmp artefacts left behind"

    def test_missing_input_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            process_partition(
                tmp_path / "nope.parquet",
                tmp_path / "out.parquet",
                tmp_path / "dropped.parquet",
                PARAMS,
            )
