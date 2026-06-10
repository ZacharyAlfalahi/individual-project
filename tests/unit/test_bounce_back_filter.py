"""
Unit tests for the bounce-back filter.

Covers:
  - Bounce-back loop — spike-with-recovery, sustained drop (GM 2009),
    par-snap, par-only cooldown narrowing, end-of-sequence lookahead,
    30-trade sustained-distress block stays whole
  - process_partition end-to-end on synthetic parquet — schema preservation,
    atomic write (.tmp cleanup), row-count correctness, NaN-price rejection
  - update_cleaning_report — preserves existing fields, asserts additivity,
    hard-fails on legacy init_price_error keys
"""
import json
import math
import os
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "scripts"))
from bounce_back_filter import (
    OUTPUT_SCHEMA,
    _apply_bounce_back_loop,
    _assert_additivity,
    load_bounce_back_config,
    process_partition,
    update_cleaning_report,
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
# process_partition end-to-end on synthetic parquet
# ---------------------------------------------------------------------------

def _make_synthetic_trace(path: Path, *, inject_nan: bool = False) -> int:
    """Build a small parquet matching OUTPUT_SCHEMA. Returns row count written."""
    import pandas as pd

    rows = []
    # Bond A — clean, 6 trades, no drops expected
    for i in range(6):
        rows.append(_row("AAA", f"2020-06-{i+1:02d}", 100.0 + i * 0.1))
    # Bond B — warm-up then a recoverable spike (should drop the spike)
    bbb_prices = [100.0, 100.5, 101.0, 100.0, 100.0, 1.87, 100.0, 100.0]
    if inject_nan:
        bbb_prices[3] = float("nan")
    for i, p in enumerate(bbb_prices):
        rows.append(_row("BBB", f"2020-07-{i+1:02d}", p))

    df = pd.DataFrame(rows)
    df["trd_exctn_dt"] = pd.to_datetime(df["trd_exctn_dt"])
    table = pa.Table.from_pandas(df, schema=OUTPUT_SCHEMA, preserve_index=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, str(path))
    return len(rows)


def _row(bond: str, dt: str, price: float) -> dict:
    return {
        "bond_id": bond,
        "cusip_id": "0000" + bond[:5].ljust(5, "X"),
        "company_symbol": "SYN",
        "trd_exctn_dt": dt,
        "trd_exctn_tm": "10:00:00",
        "rptd_pr": float(price),
        "entrd_vol_qt": 10000.0,
        "sub_prdct": "CORP",
        "rpt_side_cd": "D",
        "trdg_mkt_cd": "P1",
        "trd_mod_3": "",
        "bloomberg_identifier": "",
        "scrty_type_cd": "",
    }


def test_process_partition_preserves_schema_and_writes_atomically(tmp_path):
    input_path = tmp_path / "trace_clean.parquet"
    n_in = _make_synthetic_trace(input_path)

    stats = process_partition(input_path, input_path, _PARAMS, OUTPUT_SCHEMA)

    # Row arithmetic
    assert stats["input_rows"] == n_in
    assert stats["dropped_bounce_back"] == 1  # bond BBB's spike
    assert stats["kept_rows"] == n_in - 1

    # Schema preservation (13 fields, same order, same types)
    written = pq.read_schema(str(input_path))
    assert written.equals(OUTPUT_SCHEMA), (
        f"Schema drift detected:\n  expected: {OUTPUT_SCHEMA}\n  got: {written}"
    )

    # Atomic write left no .tmp behind
    assert not input_path.with_suffix(".parquet.tmp").exists()

    # Parquet row count matches stats
    assert pq.read_metadata(str(input_path)).num_rows == stats["kept_rows"]


def test_process_partition_rejects_non_finite_prices(tmp_path):
    # Stage 2 must hard-fail at the partition boundary if any rptd_pr is NaN.
    # IEEE-754 NaN comparisons would otherwise silently disable the bounce-back
    # filter for the affected bond.
    input_path = tmp_path / "trace_clean.parquet"
    _make_synthetic_trace(input_path, inject_nan=True)

    with pytest.raises(AssertionError, match="Non-finite price"):
        process_partition(input_path, input_path, _PARAMS, OUTPUT_SCHEMA)


def test_process_partition_raises_on_missing_input(tmp_path):
    missing = tmp_path / "does_not_exist.parquet"
    with pytest.raises(FileNotFoundError):
        process_partition(missing, missing, _PARAMS, OUTPUT_SCHEMA)


# ---------------------------------------------------------------------------
# update_cleaning_report — additivity assertion, legacy-key hard fail
# ---------------------------------------------------------------------------

_BASE_ROWS = {
    "raw_total": 100,
    "after_trc_st_T": 95,
    "dropped_cancelled_original": 2,
    "after_asof_cd_blank": 90,
    "after_wis_fl": 89,
    "after_price_plausibility": 88,
    "after_decimal_shift_correction": 88,
    "dropped_interdealer_duplicate": 8,
    "after_interdealer_dedup": 80,
    "dropped_invalid_date": 0,
    "development_rows": 50,
    "holdout_rows": 30,
    "final_clean_total": 80,
    "dropped_trc_st": 3,
    "dropped_asof_cd": 5,
    "dropped_wis_fl": 1,
    "dropped_price_plausibility": 1,
    "decimal_shift_unresolvable_dropped": 0,
    "parquet_row_count_verified": True,
}
# Additivity of the BASE (pre-bounce-back) row equation:
# 3+2+5+1+1+0+8+0+50+30 = 100 ✓


def _write_base_report(path: Path) -> None:
    payload = {
        "run_timestamp": "2020-01-01T00:00:00+00:00",
        "thresholds_sha256": "abc",
        "rows": dict(_BASE_ROWS),
        "output_columns": ["bond_id"],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f)


def test_update_cleaning_report_preserves_fields_and_asserts_additivity(tmp_path):
    report_path = tmp_path / "cleaning_report.json"
    _write_base_report(report_path)

    # Drop 3 from dev, 2 from hold. New kept: dev 50-3=47, hold 30-2=28.
    # Additivity: 3+2+5+1+1+0+8+0 + 3+2 + 47+28 = 100 ✓
    dev_stats = {"kept_rows": 47, "dropped_bounce_back": 3}
    hold_stats = {"kept_rows": 28, "dropped_bounce_back": 2}

    update_cleaning_report(report_path, dev_stats, hold_stats, "sha-test", dev_parquet=None)

    with open(report_path) as f:
        new = json.load(f)

    # New top-level fields
    assert new["bounce_back_filter_applied"] is True
    assert "bounce_back_run_timestamp" in new
    assert new["bounce_back_params_sha256"] == "sha-test"

    rows = new["rows"]
    assert rows["dropped_bounce_back_dev"] == 3
    assert rows["dropped_bounce_back_hold"] == 2
    assert rows["development_rows"] == 47
    assert rows["holdout_rows"] == 28
    assert rows["final_clean_total"] == 75
    assert rows["parquet_row_count_verified"] is True

    # Existing pre-bounce-back fields preserved unchanged
    assert rows["dropped_wis_fl"] == _BASE_ROWS["dropped_wis_fl"]
    assert rows["dropped_interdealer_duplicate"] == _BASE_ROWS["dropped_interdealer_duplicate"]

    # Atomic write left no .tmp
    assert not report_path.with_suffix(".tmp").exists()


def test_update_cleaning_report_raises_on_broken_additivity(tmp_path):
    report_path = tmp_path / "cleaning_report.json"
    _write_base_report(report_path)

    # Reduce dev kept_rows by 10 but only claim 1 drop → 9 rows unaccounted for
    bad_dev = {"kept_rows": 40, "dropped_bounce_back": 1}
    bad_hold = {"kept_rows": 30, "dropped_bounce_back": 0}

    with pytest.raises(AssertionError, match="additivity broken"):
        update_cleaning_report(report_path, bad_dev, bad_hold, "sha-test", dev_parquet=None)


def test_update_cleaning_report_hard_fails_on_legacy_init_price_keys(tmp_path):
    # A report from a prior pre-strict-fidelity run carries the removed
    # init_price_error fields. update_cleaning_report must refuse to silently
    # migrate; the user is required to re-run preprocess_trace.py.
    report_path = tmp_path / "cleaning_report.json"
    rows = dict(_BASE_ROWS)
    rows["dropped_init_price_error_dev"] = 0
    rows["dropped_init_price_error_hold"] = 0
    payload = {
        "run_timestamp": "2020-01-01T00:00:00+00:00",
        "thresholds_sha256": "abc",
        "rows": rows,
        "output_columns": ["bond_id"],
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w") as f:
        json.dump(payload, f)

    dev_stats = {"kept_rows": 50, "dropped_bounce_back": 0}
    hold_stats = {"kept_rows": 30, "dropped_bounce_back": 0}

    with pytest.raises(AssertionError, match="Legacy init_price_error fields"):
        update_cleaning_report(report_path, dev_stats, hold_stats, "sha-test", dev_parquet=None)


def test_assert_additivity_pure_function():
    rows = dict(_BASE_ROWS)
    rows.update({
        "dropped_bounce_back_dev": 3,
        "dropped_bounce_back_hold": 2,
        "development_rows": 47,
        "holdout_rows": 28,
        "final_clean_total": 75,
    })
    _assert_additivity(rows)  # should not raise

    rows["dropped_bounce_back_dev"] = 99
    with pytest.raises(AssertionError):
        _assert_additivity(rows)


def test_assert_additivity_catches_final_clean_total_drift():
    # Stale final_clean_total that no longer matches dev+hold must raise.
    rows = dict(_BASE_ROWS)
    rows.update({
        "dropped_bounce_back_dev": 3,
        "dropped_bounce_back_hold": 2,
        "development_rows": 47,
        "holdout_rows": 28,
        "final_clean_total": 80,  # stale: should be 75
    })
    with pytest.raises(AssertionError, match="final_clean_total"):
        _assert_additivity(rows)
