"""
Unit tests for scripts/bounce_back_filter.py (DRR Table A.2 bounce-back filter,
stage 2 of the meas_err corrected branch).

Coverage:
  1. The pure per-bond loop `_apply_bounce_back_loop` (spec Section 3):
     spike removal at par / above par-band / premium / discount, sustained
     distressed blocks kept whole, warm-up inactivity, par-snap flagging vs
     median recovery decoupling, par-only cooldown scope, and lookahead edge
     cases.
  2. `process_partition` orchestration on synthetic parquet inputs:
     end-to-end stats and row arithmetic, kept-output schema preservation
     (14-field OUTPUT_SCHEMA incl. decimal_shift_applied), dropped-companion
     artifact correctness (created only when drops occur), atomic .tmp +
     os.replace hygiene, missing-input and non-finite-price hard failures,
     and CROSS-BATCH invariance: results must be identical whether a bond's
     trades arrive in one read batch or are split across batch boundaries
     (including a boundary that coincides exactly with a bond_id change),
     and must agree with `_apply_bounce_back_loop` called directly on the
     time-sorted price list.

All parameters come from load_bounce_back_config() (docs/thresholds.yaml) so
tests break when production threshold values change. No literal param dicts.
"""
from datetime import datetime
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import pytest

from bounce_back_filter import (
    OUTPUT_SCHEMA,
    _apply_bounce_back_loop,
    load_bounce_back_config,
    process_partition,
)


# Load real production params so tests break when threshold values change.
_PARAMS = load_bounce_back_config()


# ---------------------------------------------------------------------------
# Bounce-back loop (spec Section 3)
# ---------------------------------------------------------------------------

def test_bounce_back_removes_isolated_spike_at_par():
    # Bond at par (~100): warm-up, then typo at 1.87, then recovery to 100.
    # par_snap activates (last 3 trades within par_band of par_level), anchor=100.
    # |1.87-100|=98 > 35 → flagged; recovery to 100 within 1.25 → dropped.
    prices = [100.0, 100.5, 99.5, 100.0, 100.5, 1.87, 100.0, 100.0]
    mask, dropped = _apply_bounce_back_loop(prices, _PARAMS)
    assert mask[5] is False, "the 1.87 spike must be removed"
    assert mask[:5] == [True] * 5
    assert mask[6:] == [True, True]
    assert dropped == 1


def test_bounce_back_removes_isolated_spike_above_par_band():
    # Bond well above par_band (=15) so par_snap does NOT activate. The anchor
    # is the trailing median (~120). A typo at 1.87 recovers back to ~120.
    prices = [120.0, 120.5, 121.0, 120.0, 120.5, 1.87, 120.0, 120.0]
    mask, dropped = _apply_bounce_back_loop(prices, _PARAMS)
    assert mask[5] is False, "the 1.87 spike must be removed (median-anchor path)"
    assert dropped == 1


def test_bounce_back_keeps_sustained_price_drop_gm_2009():
    # Bond trades at 100 then collapses to 24 and stays there — genuine
    # distressed pricing, must NOT be filtered out.
    prices = [100.0, 100.0, 100.5, 100.0, 100.0] + [24.0] * 10
    mask, dropped = _apply_bounce_back_loop(prices, _PARAMS)
    assert dropped == 0
    assert all(mask)


def test_sustained_distressed_block_thirty_trades_kept_whole():
    # Stronger displaced-block invariant. With max_span removed and cooldown
    # narrowed to par-only, the only protection against over-flagging a
    # sustained distressed block is the lookahead recovery condition: each
    # trade in the block has 5 subsequent trades at the same low level, so
    # none recovers to within recovery_tol of the (still-100) median, so none
    # is dropped. Once the trailing window has rolled enough 24s in to push
    # the median itself down, |24 - new_median| <= threshold and no further
    # flagging occurs. The block must survive end-to-end.
    prices = [100.0, 100.5, 99.5, 100.0, 100.5] + [24.0] * 30
    mask, dropped = _apply_bounce_back_loop(prices, _PARAMS)
    assert dropped == 0
    assert all(mask), f"distressed block partially flagged: dropped indices {[i for i, k in enumerate(mask) if not k]}"


def test_bounce_back_inactive_until_two_unique_anchor_prices():
    # i=0: p=100, trailing=[100], len<2 → keep (deque has 1 entry)
    # i=1: p=5, trailing=[100], len<2 → keep, then push → trailing=[100, 5]
    # i=2: p=100, trailing=[100, 5], anchor=52.5, |100-52.5|=47.5 > 35 → candidate.
    #            Lookahead prices[3]=100, |100-52.5|=47.5 > recovery_tol=1.25
    #            → no recovery → keep, push. trailing=[100, 5, 100].
    # i=3: p=100, trailing=[100, 5, 100], _push_unique skips (last=100). anchor=100.
    #            |100-100|=0 <= 35 → keep.
    prices = [100.0, 5.0, 100.0, 100.0]
    mask, dropped = _apply_bounce_back_loop(prices, _PARAMS)
    assert mask == [True, True, True, True], f"all kept; got {mask}"
    assert dropped == 0


def test_par_snap_used_for_flagging_decision_only():
    # Bond at ~88 (inside par_band=15 of par=100). With par-snap ON the
    # flag anchor snaps to 100 → |60-100|=40 > 35 → flagged. With par-snap
    # OFF the flag anchor is the median (~88) → |60-88|=28 ≤ 35 → NOT
    # flagged. Recovery (decoupled) always uses the median, so recovery to
    # 88 succeeds with |88-88|=0 only when the trade was actually flagged.
    prices = [88.0, 89.0, 88.0, 89.0, 88.0, 60.0, 88.0, 88.0]

    mask_on, dropped_on = _apply_bounce_back_loop(prices, _PARAMS)
    assert mask_on[5] is False, "par-snap flags (anchor=100); median recovers (anchor=88)"
    assert dropped_on == 1

    p_off = dict(_PARAMS)
    p_off["par_spike_heuristic"] = False
    mask_off, dropped_off = _apply_bounce_back_loop(prices, p_off)
    assert mask_off[5] is True, "without par-snap, |60-88|=28 ≤ 35 → not flagged"
    assert dropped_off == 0


def test_bounce_back_removes_flm_gm_style_spike_at_premium_bond():
    # The motivating example from spec §1: a bond trading at ~106 has a
    # single erroneous trade at ~1.87 followed by recovery to ~106. Before
    # the par-snap decoupling fix, the algorithm pinned the anchor to 100
    # (par_band=15 catches 106) and the recovery to 106 never satisfied
    # recovery_tol=1.25, so the spike was kept. After the fix the recovery
    # anchor is the median (~106), recovery succeeds, spike is dropped.
    prices = [106.0, 106.5, 107.0, 106.0, 106.5, 1.87, 106.0, 106.0, 106.0]
    mask, dropped = _apply_bounce_back_loop(prices, _PARAMS)
    assert mask[5] is False
    assert dropped == 1


def test_bounce_back_removes_spike_at_discount_bond_in_par_band():
    # A bond at ~88 (inside par_band) with a spike at ~1.5 and recovery to ~88.
    # Same dead-zone family as the FLM.GM case but on the discount side.
    prices = [88.0, 88.5, 87.5, 88.0, 88.5, 1.5, 88.0, 88.0, 88.0]
    mask, dropped = _apply_bounce_back_loop(prices, _PARAMS)
    assert mask[5] is False
    assert dropped == 1


def test_par_block_cooldown_keeps_post_drop_trades_that_would_otherwise_be_flagged():
    # Par-block drop → cooldown fires (DRR Table A.2 clause iii).
    # Post-drop trades at 50 are themselves bounce-back candidates
    # (|50-100|=50 > 35) with valid recovery to 100 in the lookahead.
    # Under par-only cooldown=2 the algorithm keeps them unconditionally;
    # with cooldown disabled they get dropped too.
    prices = [100.0, 100.5, 99.5, 100.0, 100.5, 1.87, 50.0, 50.0, 100.0, 100.0]

    mask_with, dropped_with = _apply_bounce_back_loop(prices, _PARAMS)
    assert mask_with[5] is False, "spike at 1.87 dropped"
    assert mask_with[6] is True, "first post-spike 50 kept by par-block cooldown"
    assert mask_with[7] is True, "second post-spike 50 kept by par-block cooldown"
    assert dropped_with == 1

    p_off = dict(_PARAMS)
    p_off["par_cooldown_after_flag"] = 0
    _, dropped_off = _apply_bounce_back_loop(prices, p_off)
    assert dropped_off == 3, "without cooldown, the two 50s are also dropped"


def test_non_par_drop_does_not_trigger_cooldown():
    # Cooldown scope narrowing test. Bond at ~120 (outside par_band=15 of par=100)
    # has two adjacent recoverable spikes. par_snap never fires → both drops
    # are via the median-anchor path → cooldown must NOT fire → both spikes
    # are independently evaluated and both drop.
    #
    # Under the old broad-cooldown semantics, the first drop would have armed
    # cooldown_remaining=2 and the second spike would have been kept; this
    # test pins the DRR-faithful behavior.
    prices = [120.0, 120.5, 121.0, 120.0, 120.5, 1.5, 50.0, 120.0, 120.0, 120.0]
    mask, dropped = _apply_bounce_back_loop(prices, _PARAMS)
    assert mask[5] is False, "non-par spike 1.5 drops"
    assert mask[6] is False, "second non-par spike 50 drops (no cooldown protection)"
    assert dropped == 2


def test_lookahead_at_end_of_sequence_keeps_trade():
    # No future trades → no recovery evidence → keep.
    prices = [100.0, 100.5, 101.0, 100.0, 100.0, 1.87]
    mask, dropped = _apply_bounce_back_loop(prices, _PARAMS)
    assert mask[5] is True
    assert dropped == 0


def test_no_recovery_within_lookahead_keeps_trade():
    # Spike at index 5; no future trade within recovery_tol of anchor → keep.
    prices = [100.0, 100.5, 101.0, 100.0, 100.0, 50.0, 49.0, 51.0, 50.0, 50.0]
    mask, dropped = _apply_bounce_back_loop(prices, _PARAMS)
    assert mask[5] is True
    assert dropped == 0


# ---------------------------------------------------------------------------
# Synthetic parquet builders (14-field OUTPUT_SCHEMA)
# ---------------------------------------------------------------------------

# Spike sequence used across the process_partition tests: warm-up at par,
# recoverable spike at index 5 (200.0), recovery to par. Under the production
# params the par-snap anchor flags |200-100|=100 > 35 and the lookahead
# recovery to 100 within recovery_tol drops exactly index 5.
_SPIKE_PRICES = [99.0, 101.0, 98.0, 102.0, 100.0, 200.0, 100.0, 100.0]
_SPIKE_INDEX = 5

# Clean bond: tiny oscillation around par, nothing flaggable.
_CLEAN_PRICES = [100.0, 100.1, 100.2, 100.1, 100.0, 100.1]


def _row(bond: str, dt: datetime, tm: str, price: float) -> dict:
    """One synthetic trade matching OUTPUT_SCHEMA exactly (14 fields)."""
    return {
        "bond_id": bond,
        "cusip_id": "0000" + bond[:5].ljust(5, "X"),
        "company_symbol": "SYN",
        "trd_exctn_dt": dt,
        "trd_exctn_tm": tm,
        "rptd_pr": float(price),
        "entrd_vol_qt": 10000.0,
        "sub_prdct": "CORP",
        "rpt_side_cd": "D",
        "trdg_mkt_cd": "P1",
        "trd_mod_3": "",
        "bloomberg_identifier": "",
        "scrty_type_cd": "",
        "decimal_shift_applied": False,
    }


def _row_ci(cusip: str, bond: str, dt: datetime, tm: str, price: float) -> dict:
    """Like _row but with an INDEPENDENT cusip_id (for CUSIP-vs-symbol tests)."""
    r = _row(bond, dt, tm, price)
    r["cusip_id"] = cusip
    return r


def _bond_rows(bond: str, prices, *, base_day: int = 1, month: int = 6) -> list:
    """Rows for one bond, one trade per day so DuckDB date sort preserves
    the intended price order. Days are zero-padded by datetime itself."""
    return [
        _row(bond, datetime(2020, month, base_day + i), "10:00:00", p)
        for i, p in enumerate(prices)
    ]


def _bond_rows_intraday(bond: str, prices) -> list:
    """Rows for one bond all on the SAME date, ordered by zero-padded
    'HH:MM:SS' execution times so the lexical string sort on trd_exctn_tm
    reproduces the intended sequence."""
    return [
        _row(bond, datetime(2020, 6, 1), f"09:{i:02d}:00", p)
        for i, p in enumerate(prices)
    ]


def _write_trace(path: Path, rows: list) -> int:
    """Write synthetic rows as a parquet matching OUTPUT_SCHEMA. Returns n."""
    table = pa.Table.from_pylist(rows, schema=OUTPUT_SCHEMA)
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, str(path))
    return len(rows)


def _run(tmp_path: Path, input_path: Path, tag: str, **kwargs) -> tuple:
    """process_partition with distinct output/dropped paths; returns
    (stats, output_path, dropped_path)."""
    output_path = tmp_path / f"trace_clean_corr_{tag}.parquet"
    dropped_path = tmp_path / f"bounceback_dropped_{tag}.parquet"
    stats = process_partition(input_path, output_path, dropped_path,
                              _PARAMS, output_schema=OUTPUT_SCHEMA, **kwargs)
    return stats, output_path, dropped_path


# ---------------------------------------------------------------------------
# process_partition end-to-end on synthetic parquet
# ---------------------------------------------------------------------------

def test_process_partition_end_to_end_stats_and_schema(tmp_path):
    input_path = tmp_path / "trace_clean_decimal_shifted.parquet"
    rows = _bond_rows("AAA", _CLEAN_PRICES, month=6) + _bond_rows("BBB", _SPIKE_PRICES, month=7)
    n_in = _write_trace(input_path, rows)

    stats, output_path, dropped_path = _run(tmp_path, input_path, "e2e")

    # Row arithmetic: input = kept + dropped, exactly one drop (BBB's spike)
    assert stats["input_rows"] == n_in
    assert stats["dropped_bounce_back"] == 1
    assert stats["kept_rows"] == n_in - 1
    assert stats["input_rows"] == stats["kept_rows"] + stats["dropped_bounce_back"]

    # Kept output: schema preserved (14 fields, same order, same types)
    written = pq.read_schema(str(output_path))
    assert written.equals(OUTPUT_SCHEMA), (
        f"Schema drift detected:\n  expected: {OUTPUT_SCHEMA}\n  got: {written}"
    )
    assert pq.read_metadata(str(output_path)).num_rows == stats["kept_rows"]

    # The spike row is absent from the kept output
    kept = pq.read_table(str(output_path))
    kept_bbb_prices = kept.filter(
        pc.equal(kept.column("bond_id"), "BBB")
    ).column("rptd_pr").to_pylist()
    assert 200.0 not in kept_bbb_prices


def test_process_partition_dropped_companion_contains_exactly_the_spike(tmp_path):
    input_path = tmp_path / "trace_clean_decimal_shifted.parquet"
    rows = _bond_rows("AAA", _CLEAN_PRICES, month=6) + _bond_rows("BBB", _SPIKE_PRICES, month=7)
    _write_trace(input_path, rows)

    stats, _, dropped_path = _run(tmp_path, input_path, "dropped")

    assert dropped_path.exists(), "drops occurred → companion artifact must exist"
    dropped = pq.read_table(str(dropped_path))
    assert dropped.schema.equals(OUTPUT_SCHEMA)
    assert dropped.num_rows == stats["dropped_bounce_back"] == 1
    assert dropped.column("bond_id").to_pylist() == ["BBB"]
    assert dropped.column("rptd_pr").to_pylist() == [_SPIKE_PRICES[_SPIKE_INDEX]]


def test_process_partition_no_drops_means_no_dropped_file(tmp_path):
    input_path = tmp_path / "trace_clean_decimal_shifted.parquet"
    n_in = _write_trace(input_path, _bond_rows("AAA", _CLEAN_PRICES))

    stats, output_path, dropped_path = _run(tmp_path, input_path, "clean")

    assert stats["dropped_bounce_back"] == 0
    assert stats["kept_rows"] == stats["input_rows"] == n_in
    assert output_path.exists()
    assert not dropped_path.exists(), "no drops → companion artifact must NOT be created"


def test_segments_by_cusip_across_symbol_change(tmp_path):
    # One bond (CUSIP) reported under two TRACE symbols over its life. The
    # warm-up that establishes the trailing-median anchor is under the OLD
    # symbol; the recoverable spike is under the NEW symbol. Segmenting by
    # CUSIP keeps the price history continuous so the spike is flagged and
    # dropped. Segmenting by bond_id (the pre-fix behaviour) would reset the
    # anchor at the symbol boundary — the spike's segment would start with
    # < 2 trailing prices and the spike would survive.
    cusip = "SAMECUSIP"
    warmup = _SPIKE_PRICES[:_SPIKE_INDEX]      # [99, 101, 98, 102, 100]
    spike_tail = _SPIKE_PRICES[_SPIKE_INDEX:]  # [200, 100, 100]
    rows = (
        [_row_ci(cusip, "SYM_OLD", datetime(2020, 6, 1 + i), "10:00:00", p)
         for i, p in enumerate(warmup)]
        + [_row_ci(cusip, "SYM_NEW", datetime(2020, 6, 6 + i), "10:00:00", p)
           for i, p in enumerate(spike_tail)]
    )
    input_path = tmp_path / "trace_clean_decimal_shifted.parquet"
    _write_trace(input_path, rows)

    stats, output_path, _ = _run(tmp_path, input_path, "cusip_cont")

    assert stats["dropped_bounce_back"] == 1, \
        "spike across the symbol boundary must be flagged (CUSIP-continuous)"
    kept = pq.read_table(str(output_path))
    assert 200.0 not in kept.column("rptd_pr").to_pylist()


def test_blank_cusip_falls_back_to_bond_id(tmp_path):
    # Rows with a blank CUSIP must segment by bond_id, NOT collapse into one
    # giant blank-keyed segment (which would let one bond's prices corrupt
    # another's anchor). Two distinct blank-CUSIP bonds, processed
    # independently: the spiky one drops its spike; the clean one is kept whole.
    rows = (
        [_row_ci("", "BLANK_SPIKE", datetime(2020, 6, 1 + i), "10:00:00", p)
         for i, p in enumerate(_SPIKE_PRICES)]
        + [_row_ci("", "BLANK_CLEAN", datetime(2020, 7, 1 + i), "10:00:00", p)
           for i, p in enumerate(_CLEAN_PRICES)]
    )
    input_path = tmp_path / "trace_clean_decimal_shifted.parquet"
    _write_trace(input_path, rows)

    stats, output_path, _ = _run(tmp_path, input_path, "blank_cusip")

    assert stats["dropped_bounce_back"] == 1, "only BLANK_SPIKE's spike drops"
    kept = pq.read_table(str(output_path))
    clean_kept = kept.filter(pc.equal(kept.column("bond_id"), "BLANK_CLEAN"))
    assert clean_kept.num_rows == len(_CLEAN_PRICES), \
        "clean blank-CUSIP bond must be untouched (not merged with the spiky one)"
    spike_kept = kept.filter(pc.equal(kept.column("bond_id"), "BLANK_SPIKE"))
    assert 200.0 not in spike_kept.column("rptd_pr").to_pylist()


def test_process_partition_leaves_no_tmp_files(tmp_path):
    input_path = tmp_path / "trace_clean_decimal_shifted.parquet"
    rows = _bond_rows("AAA", _CLEAN_PRICES, month=6) + _bond_rows("BBB", _SPIKE_PRICES, month=7)
    _write_trace(input_path, rows)

    _run(tmp_path, input_path, "atomic")

    leftovers = [p.name for p in tmp_path.iterdir() if p.name.endswith(".tmp")]
    assert leftovers == [], f"atomic write left temp files behind: {leftovers}"


def test_process_partition_raises_on_missing_input(tmp_path):
    missing = tmp_path / "does_not_exist.parquet"
    output_path = tmp_path / "out.parquet"
    dropped_path = tmp_path / "dropped.parquet"
    with pytest.raises(FileNotFoundError):
        process_partition(missing, output_path, dropped_path, _PARAMS)


def test_process_partition_rejects_non_finite_prices(tmp_path):
    # Stage 2 must hard-fail at the partition boundary if any rptd_pr is NaN.
    # IEEE-754 NaN comparisons would otherwise silently disable the bounce-back
    # filter for the affected bond.
    input_path = tmp_path / "trace_clean_decimal_shifted.parquet"
    prices = list(_SPIKE_PRICES)
    prices[3] = float("nan")
    _write_trace(input_path, _bond_rows("BBB", prices))

    output_path = tmp_path / "out.parquet"
    dropped_path = tmp_path / "dropped.parquet"
    with pytest.raises(AssertionError, match="Non-finite price"):
        process_partition(input_path, output_path, dropped_path, _PARAMS)


# ---------------------------------------------------------------------------
# Cross-batch invariance (bond segments accumulate across batch boundaries)
# ---------------------------------------------------------------------------

def test_cross_batch_split_within_one_bond_matches_single_batch_run(tmp_path):
    # A single bond with 11 intraday trades (zero-padded HH:MM:SS times so
    # the lexical sort on trd_exctn_tm preserves the intended order). At
    # batch_size=3 the read batches are [0:3], [3:6], [6:9], [9:11]: the
    # spike (index 5) lands in the second batch while its recovery trades
    # land in the third — the drop decision REQUIRES segment accumulation
    # across batch boundaries.
    prices = [100.0, 100.5, 99.5, 100.0, 100.5, 200.0,
              100.0, 100.0, 100.0, 100.0, 100.0]
    input_path = tmp_path / "trace_clean_decimal_shifted.parquet"
    n_in = _write_trace(input_path, _bond_rows_intraday("CCC", prices))

    stats_small, out_small, dropped_small = _run(
        tmp_path, input_path, "small", batch_size=3)
    stats_big, out_big, dropped_big = _run(
        tmp_path, input_path, "big", batch_size=10_000)

    # Identical stats regardless of batch size
    assert stats_small == stats_big
    assert stats_small["input_rows"] == n_in

    # Consistent with the pure loop on the time-sorted price list
    mask, n_dropped = _apply_bounce_back_loop(prices, _PARAMS)
    assert stats_small["dropped_bounce_back"] == n_dropped == 1
    assert stats_small["kept_rows"] == sum(mask)

    # Kept outputs byte-identical in content
    assert pq.read_table(str(out_small)).equals(pq.read_table(str(out_big)))

    # Dropped row identity identical across batch sizes and matches the loop
    d_small = pq.read_table(str(dropped_small))
    d_big = pq.read_table(str(dropped_big))
    assert d_small.equals(d_big)
    assert d_small.column("bond_id").to_pylist() == ["CCC"]
    assert d_small.column("rptd_pr").to_pylist() == [
        prices[mask.index(False)]
    ] == [200.0]


def test_cross_batch_boundary_coinciding_with_bond_change(tmp_path):
    # Two bonds whose row counts are exact multiples of batch_size=3, so a
    # batch boundary falls precisely on the bond_id change (row 6): bond AAA
    # has 6 clean trades, bond BBB has 9 trades with a recoverable spike at
    # index 5. The flush-on-bond-change path must fire correctly when the
    # new bond starts exactly at a batch boundary.
    aaa_prices = _CLEAN_PRICES                      # 6 rows, no drops
    bbb_prices = _SPIKE_PRICES + [100.0]            # 9 rows, drop index 5
    rows = (_bond_rows("AAA", aaa_prices, month=6)
            + _bond_rows("BBB", bbb_prices, month=7))
    input_path = tmp_path / "trace_clean_decimal_shifted.parquet"
    n_in = _write_trace(input_path, rows)
    assert n_in % 3 == 0 and len(aaa_prices) % 3 == 0  # boundary alignment

    stats_small, out_small, dropped_small = _run(
        tmp_path, input_path, "small", batch_size=3)
    stats_big, out_big, dropped_big = _run(
        tmp_path, input_path, "big", batch_size=10_000)

    assert stats_small == stats_big
    assert stats_small["input_rows"] == n_in
    assert stats_small["dropped_bounce_back"] == 1
    assert stats_small["kept_rows"] == n_in - 1

    # Consistent with the pure loop applied per bond
    mask_aaa, n_aaa = _apply_bounce_back_loop(aaa_prices, _PARAMS)
    mask_bbb, n_bbb = _apply_bounce_back_loop(bbb_prices, _PARAMS)
    assert n_aaa == 0 and all(mask_aaa)
    assert n_bbb == 1 and mask_bbb[_SPIKE_INDEX] is False
    assert stats_small["kept_rows"] == sum(mask_aaa) + sum(mask_bbb)

    assert pq.read_table(str(out_small)).equals(pq.read_table(str(out_big)))

    d_small = pq.read_table(str(dropped_small))
    assert d_small.equals(pq.read_table(str(dropped_big)))
    assert d_small.column("bond_id").to_pylist() == ["BBB"]
    assert d_small.column("rptd_pr").to_pylist() == [bbb_prices[_SPIKE_INDEX]]
