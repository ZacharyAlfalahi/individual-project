"""
Characterization test — distressed-filter windows are OBSERVATION-based, not
calendar-based.

The four filter primitives in scripts/apply_distressed_filters.py operate on
numpy price arrays with NO date argument: `L` and plateau "runs" are counted in
ROW POSITIONS, not calendar days. process_partition sorts by trd_exctn_dt and
then hands positional slices to those primitives, so calendar spacing has zero
effect beyond ordering.

This test PINS that behaviour; it does NOT assert the behaviour is correct. The
"days vs observations" question is a source-fidelity item tracked in
docs/data/registers/citations_verified.md §1c/I1 against DRR 2026 Appendix A.3, and the current
implementation is described in docs/data/specs/distressed_filters_spec.md §3.

Once A.3 is verified, exactly one of:
  - the spec wording is corrected "days" -> "observations" (these tests stay
    green); or
  - the windows are made date-aware (these tests are updated to the new
    semantics).

All fixtures use min_price == max_price == price_vwap so Filter 4 (intraday) can
never fire; each series is hand-traced so only the intended filter drops rows.
"""

from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from apply_distressed_filters import load_config, process_partition

PARAMS = load_config()
L = int(PARAMS["L"])

CUSIP = "CUSWIN001"
BASE = pd.Timestamp("2015-06-01")

DAILY_SCHEMA = pa.schema([
    pa.field("cusip_id",      pa.string()),
    pa.field("trd_exctn_dt",  pa.timestamp("us")),
    pa.field("price_vwap",    pa.float64()),
    pa.field("total_vol",     pa.float64()),
    pa.field("n_trades",      pa.int64()),
    pa.field("min_price",     pa.float64()),
    pa.field("max_price",     pa.float64()),
])


def _day(offset: int) -> pd.Timestamp:
    return BASE + pd.Timedelta(days=offset)


def _dropped_positions(work_dir: Path, day_offsets, vwaps) -> set:
    """Build a single-cusip daily panel from (day_offset, vwap) pairs, run
    process_partition, and return the set of POSITIONAL indices (into the
    input order) that were dropped.

    min_price == max_price == price_vwap so Filter 4 never fires. Input is
    supplied in strictly-increasing time order (unique offsets), matching
    process_partition's (cusip, date) sort, so input position == sorted
    position and each dropped date maps back to a unique position.
    """
    assert len(day_offsets) == len(vwaps)
    work_dir.mkdir(parents=True, exist_ok=True)
    rows = [
        {
            "cusip_id": CUSIP,
            "trd_exctn_dt": _day(off),
            "price_vwap": float(v),
            "total_vol": 200_000.0,
            "n_trades": 2,
            "min_price": float(v),
            "max_price": float(v),
        }
        for off, v in zip(day_offsets, vwaps)
    ]
    inp = work_dir / "trace_daily_corr.parquet"
    out = work_dir / "trace_daily_corr_filtered.parquet"
    dropped = work_dir / "distressed_dropped_dev.parquet"
    pq.write_table(
        pa.Table.from_pandas(pd.DataFrame(rows), schema=DAILY_SCHEMA,
                             preserve_index=False),
        str(inp),
    )
    process_partition(inp, out, dropped, PARAMS)

    if not dropped.exists():
        return set()
    offset_to_pos = {off: i for i, off in enumerate(day_offsets)}
    dropped_df = pd.read_parquet(dropped)
    dropped_offsets = (pd.to_datetime(dropped_df["trd_exctn_dt"]) - BASE).dt.days
    return {offset_to_pos[int(d)] for d in dropped_offsets}


class TestObservationVsCalendarWindowing:
    def test_spike_recovery_uses_observation_count_not_calendar_gap(self, tmp_path):
        # Spike at position 5 (+5 above a flat 100 level) recovering at position 6.
        vwaps = [100.0, 100.0, 100.0, 100.0, 100.0, 105.0, 100.0]
        dense = list(range(7))               # consecutive calendar days 0..6
        sparse = [0, 1, 2, 3, 4, 5, 100]     # recovery observation 95 cal-days out

        # The sparse recovery observation is far beyond an L-calendar-day window,
        # so a date-aware "recover within L days" rule would NOT flag it.
        assert (sparse[6] - sparse[5]) > L

        dense_dropped = _dropped_positions(tmp_path / "dense", dense, vwaps)
        sparse_dropped = _dropped_positions(tmp_path / "sparse", sparse, vwaps)

        # Current behaviour: recovery is judged by observation adjacency, so the
        # spike is flagged in BOTH spacings.
        assert dense_dropped == {5}
        assert sparse_dropped == {5}

    def test_plateau_run_is_observation_contiguous_not_calendar_contiguous(self, tmp_path):
        # Three-day run at 100 (a round-number level) displaced from ~130 on both
        # sides — positions 3, 4, 5. (Same shape as the spec's plateau fixture.)
        vwaps = [130.0, 131.0, 129.0, 100.0, 100.0, 100.0, 131.0, 130.0]
        dense = list(range(8))
        sparse = [0, 1, 2, 3, 50, 90, 120, 121]   # the three 100s spread over months

        # The three plateau observations span far beyond an L-calendar-day window.
        assert (sparse[5] - sparse[3]) > L

        dense_dropped = _dropped_positions(tmp_path / "dense", dense, vwaps)
        sparse_dropped = _dropped_positions(tmp_path / "sparse", sparse, vwaps)

        # Current behaviour: a "run" is observation-contiguous, so the same three
        # positions are flagged regardless of the calendar gaps between them.
        assert dense_dropped == {3, 4, 5}
        assert sparse_dropped == {3, 4, 5}

    def test_drops_identical_regardless_of_date_spacing(self, tmp_path):
        # Mixed series: isolated ultra-low anomaly (position 2) + spike-and-recover
        # (position 6). Documents the general invariant.
        vwaps = [10.0, 10.0, 0.05, 10.0, 10.0, 10.0, 13.5, 10.0]
        dense = list(range(8))
        sparse = [0, 30, 60, 90, 120, 150, 180, 210]

        dense_dropped = _dropped_positions(tmp_path / "dense", dense, vwaps)
        sparse_dropped = _dropped_positions(tmp_path / "sparse", sparse, vwaps)

        # Calendar spacing is irrelevant to the current pipeline: identical prices
        # produce identical drops (anomaly at 2, spike at 6) under any spacing.
        assert dense_dropped == {2, 6}
        assert sparse_dropped == {2, 6}
