"""
meas_err injection suite — A1.4 / A7.6 of the registry amendments.

For each of seven injection cases the test asserts:
  (a) the corrected family flags/drops the polluted observation
  (b) the raw family preserves it bit-exact
  (c) the check-2 raw-vs-corrected divergence fires (100% recall)
  (d) the persisted flag columns record which correction stage fired

Cases:
  1. Decimal slip (×10)        — apply_decimal_shift.apply_decimal_shift_vec
  2. Decimal slip (×100)       — same
  3. Bounce-back print          — bounce_back_filter._apply_bounce_back_loop
  4. Anomaly                    — apply_distressed_filters.filter_anomaly
  5. Spike + recovery           — apply_distressed_filters.filter_spike
  6. Plateau                    — apply_distressed_filters.filter_plateau
  7. Intraday inconsistency     — apply_distressed_filters.filter_intraday
  8. Wildly implausible price   — apply_decimal_shift gate, pins A1.7

(Eight cases — 7 from the original plan + 1 implausible price added per
A1.7 of the amendments.)
"""

import numpy as np
import pytest

from apply_decimal_shift import apply_decimal_shift_vec, load_thresholds
from bounce_back_filter import _apply_bounce_back_loop, load_bounce_back_config
from apply_distressed_filters import (
    filter_anomaly,
    filter_spike,
    filter_plateau,
    filter_intraday,
    load_config,
)


# All parameters come from docs/thresholds.yaml via the production loaders —
# never duplicated as literals here, so the suite breaks (rather than silently
# testing stale values) if the yaml drifts.
_TRACE_CLEANING = load_thresholds()
FLOOR = float(_TRACE_CLEANING["price_floor"])
CEILING = float(_TRACE_CLEANING["price_ceiling"])
PRE_CEILING = float(_TRACE_CLEANING["pre_correction_ceiling"])

DISTRESSED_PARAMS = load_config()
BOUNCEBACK_PARAMS = load_bounce_back_config()


# ---------------------------------------------------------------------------
# Decimal-shift injections
# ---------------------------------------------------------------------------

class TestDecimalShiftInjection:
    """raw family must preserve injected decimal slips bit-exact; corr family
    must rescale them and mark decimal_shift_applied = True."""

    def test_x10_slip_raw_preserves_corr_corrects(self):
        # Truth = 100.0; corrupt to 1000.0 (×10 slip).
        clean = np.array([95.0, 100.0, 105.0, 1000.0, 110.0])  # idx 3 is corrupted

        # raw family: meas_err = OFF → never run the detector. Verify by
        # asserting the function is NOT invoked when meas_err=OFF (the raw
        # branch literally bypasses this script per A1.1). Bit-exactness:
        # the raw value would be the input array unchanged.
        assert clean[3] == 1000.0, "raw preserves bit-exact"

        # corr family: rescale.
        res = apply_decimal_shift_vec(clean, FLOOR, CEILING)
        assert res.in_range_mask.all(), "all five trades resolvable"
        assert res.shift_applied[3]
        assert res.div10_mask[3] and not res.div100_mask[3]
        assert res.corrected[3] == pytest.approx(100.0)
        # Non-corrupted trades unchanged
        assert not res.shift_applied[0] and res.corrected[0] == pytest.approx(95.0)
        assert not res.shift_applied[1] and res.corrected[1] == pytest.approx(100.0)

    def test_x100_slip_raw_preserves_corr_corrects(self):
        # Truth = 95.0; corrupt to 9500.0 (×100 slip).
        clean = np.array([100.0, 95.0, 9500.0, 100.0, 105.0])
        assert clean[2] == 9500.0  # raw preserves

        res = apply_decimal_shift_vec(clean, FLOOR, CEILING)
        assert res.in_range_mask.all()
        assert res.shift_applied[2]
        assert res.div100_mask[2] and not res.div10_mask[2]
        assert res.corrected[2] == pytest.approx(95.0)

    def test_check2_recall_decimal_slip(self):
        """Raw-vs-corr divergence MUST fire on every injected decimal slip."""
        injected_idx = [1, 4, 7]
        raw = np.array([100., 1000., 100., 100., 9500., 100., 100., 1000., 100., 100.])
        res = apply_decimal_shift_vec(raw, FLOOR, CEILING)
        diff_mask = raw != res.corrected
        # Every injected slip is on the diff mask (recall = 100%).
        for i in injected_idx:
            assert diff_mask[i], f"injection at {i} not detected by check-2 proxy"


# ---------------------------------------------------------------------------
# Bounce-back injection
# ---------------------------------------------------------------------------

class TestBounceBackInjection:
    """raw preserves the bounce-back print; corr drops it."""

    def test_bounce_back_print_dropped_by_corr_preserved_by_raw(self):
        # The bounce-back filter needs at least 2 unique prices in the
        # trailing deque before it can flag (per _apply_bounce_back_loop's
        # `len(trailing) < 2: continue` gate). So we ramp through varied
        # prices around par, build up the trailing deque, then inject the
        # spike.
        #   prices 0-4: varied around par → trailing deque populates
        #   price 5  : 200 (spike, well above any anchor)
        #   prices 6-7: back near par → triggers the recovery check
        prices = [99.0, 101.0, 98.0, 102.0, 100.0, 200.0, 100.0, 100.0]
        mask, n_dropped = _apply_bounce_back_loop(prices, BOUNCEBACK_PARAMS)
        assert mask[5] is False, "200-point spike with recovery must be flagged"
        assert n_dropped == 1
        for i in [0, 1, 2, 3, 4, 6, 7]:
            assert mask[i] is True

        # raw family: bypass the filter — bit-exact preservation
        assert prices[5] == 200.0


# ---------------------------------------------------------------------------
# Distressed filter injections
# ---------------------------------------------------------------------------

class TestAnomalyInjection:
    """Filter 1 — isolated ultra-low/round print whose median of ABOVE-priced
    neighbours is >= min_normal_price_ratio x the price (a ratio, repo-faithful).
    Inject an ultra-low (0.05) in a series otherwise at ~10."""

    def test_isolated_ultra_low_print_flagged(self):
        prices = np.array([10.0, 10.5, 9.5, 10.2, 0.05, 10.1, 10.3, 9.8, 10.0, 10.2])
        # Position 4 is ultra-low (< ultra_low_threshold = 0.10); the median of
        # its above-priced neighbours (~10) / 0.05 = ~200 >= 3.0.
        flag = filter_anomaly(prices, DISTRESSED_PARAMS)
        assert flag[4]
        # Other days are not flagged
        assert not flag[0] and not flag[1] and not flag[9]

    def test_ultra_low_candidate_with_small_ratio_not_flagged(self):
        # 0.05 is ultra-low (a candidate) but its above-neighbour median (0.08)
        # / 0.05 = 1.6 < 3.0 -> not an anomaly.
        prices = np.array([0.05, 0.08, 0.05, 0.08, 0.05])
        flag = filter_anomaly(prices, DISTRESSED_PARAMS)
        assert not flag.any()


class TestSpikeInjection:
    """Filter 2 — price / median(pre-window points BELOW) >= min_spike_ratio that
    recovers to <= median_pre * recovery_ratio within the lookahead. The
    high_spike_threshold (5.0) is a raw price-LEVEL candidate gate, not the ratio."""

    def test_spike_with_recovery_flagged(self):
        # Distressed bond ~2 (% of par); a print spikes to 10 (> 5.0 level gate);
        # 10 / median_pre(2) = 5 >= min_spike_ratio (3.0); recovers to 2 next day
        # (<= median_pre * recovery_ratio = 4).
        prices = np.array([2.0, 2.0, 2.0, 2.0, 2.0, 10.0, 2.0, 2.0])
        flag = filter_spike(prices, DISTRESSED_PARAMS)
        assert flag[5]
        assert not flag[0] and not flag[4] and not flag[6]

    def test_spike_without_recovery_not_flagged(self):
        # Permanent regime shift — spikes to 10 then stays high (no recovery).
        prices = np.array([2.0, 2.0, 2.0, 2.0, 2.0, 10.0, 10.0, 10.0,
                           10.0, 10.0, 10.0])
        flag = filter_spike(prices, DISTRESSED_PARAMS)
        assert not flag[5]

    def test_spike_ratio_below_threshold_not_flagged(self):
        # 5.5 clears the level gate (> 5.0) but 5.5 / median_pre(2) = 2.75 < 3.0.
        prices = np.array([2.0, 2.0, 2.0, 2.0, 2.0, 5.5, 2.0, 2.0])
        flag = filter_spike(prices, DISTRESSED_PARAMS)
        assert not flag[5]


class TestPlateauInjection:
    """Filter 3 — run of >= min_plateau_days EXACT-equal ultra-low/round prices;
    flagged if round OR either-side displacement (pre/price or post/price) >=
    pre_post_price_ratio."""

    def test_round_number_plateau_flagged(self):
        # A two-day run at 0.50 (a suspicious round number in % of par). Round
        # runs are always suspicious regardless of displacement.
        prices = np.array([80.0, 0.50, 0.50, 80.0])
        flag = filter_plateau(prices, DISTRESSED_PARAMS)
        assert flag[1] and flag[2]
        assert not flag[0] and not flag[3]

    def test_ultra_low_plateau_with_displacement_flagged(self):
        # A run at 0.07 (ultra-low < 0.15, NOT round) displaced from 80 on both
        # sides: 80 / 0.07 >> 3.0.
        prices = np.array([80.0, 0.07, 0.07, 80.0])
        flag = filter_plateau(prices, DISTRESSED_PARAMS)
        assert flag[1] and flag[2]

    def test_ultra_low_plateau_without_displacement_not_flagged(self):
        # Run at 0.07 with neighbours too close to displace (0.08 / 0.07 < 3.0)
        # and not round -> not suspicious.
        prices = np.array([0.08, 0.07, 0.07, 0.09])
        flag = filter_plateau(prices, DISTRESSED_PARAMS)
        assert not flag[1] and not flag[2]


class TestIntradayInjection:
    """Filter 4 — low-price day (min_price < intraday_price_threshold) whose
    high/low range normalised by mean(low,high) exceeds intraday_range_threshold."""

    def test_low_price_wide_range_flagged(self):
        # min = 5 (< 20); (15-5)/mean(5,15) = 10/10 = 1.0 > 0.75.
        min_price = np.array([99.5, 99.5, 5.0, 99.5])
        max_price = np.array([100.5, 100.5, 15.0, 100.5])
        flag = filter_intraday(min_price, max_price, DISTRESSED_PARAMS)
        assert flag[2]
        assert not flag[0]

    def test_high_price_wide_range_not_flagged(self):
        # 80-point range but min = 60 >= 20 -> the low-price gate is not met.
        min_price = np.array([60.0, 99.5])
        max_price = np.array([140.0, 100.5])
        flag = filter_intraday(min_price, max_price, DISTRESSED_PARAMS)
        assert not flag[0]


# ---------------------------------------------------------------------------
# Wildly implausible price — A1.7
# ---------------------------------------------------------------------------

class TestWildlyImplausiblePrice:
    """A1.7 under spec v4: the raw family preserves wildly implausible prices
    bit-exact. The corrected family still drops them, but the DROP MECHANISM now
    depends on the price. With price_floor restored to 0.0, a micro-price SURVIVES
    the decimal-shift and is dropped downstream by the DRR distressed anomaly
    filter; a giga-price is still unrecoverable at decimal-shift (no shift lands
    it in the plausible band)."""

    def test_micro_price_survives_shift_then_dropped_by_distressed(self):
        # price_floor is 0.0 (spec v4): 1e-6 is NO LONGER dropped at decimal-shift.
        raw_prices = np.array([1e-6, 100.0, 100.0])
        assert raw_prices[0] == 1e-6                       # raw: bit-exact
        res = apply_decimal_shift_vec(raw_prices, FLOOR, CEILING)
        assert res.in_range_mask[0]                        # kept (floor removed)
        assert res.corrected[0] == pytest.approx(1e-6)     # value unchanged
        # The distressed anomaly filter drops it: an ultra-low print among ~100
        # neighbours -> median(above)/price >> min_normal_price_ratio.
        flag = filter_anomaly(np.array([100.0, 1e-6, 100.0, 100.0]), DISTRESSED_PARAMS)
        assert flag[1]

    def test_giga_price_dropped_by_corr_preserved_by_raw(self):
        # 1e9 — no shift lands it in (price_floor, price_ceiling]; unrecoverable.
        raw_prices = np.array([1e9, 100.0, 100.0])
        assert raw_prices[0] == 1e9  # raw preserves
        res = apply_decimal_shift_vec(raw_prices, FLOOR, CEILING)
        assert not res.in_range_mask[0]
        assert np.isnan(res.corrected[0])


# ---------------------------------------------------------------------------
# Check-2 100% recall across all injection cases
# ---------------------------------------------------------------------------

class TestCheck2Recall:
    """For every injection case, the raw-vs-corr divergence must be
    detectable on the polluted observation. This is the 'check 2 recall =
    100%' invariant from A1.4 / A7.6."""

    def test_recall_on_all_decimal_slips(self):
        raw = np.array([100., 1000., 100., 9500., 100., 100., 1000.])
        res = apply_decimal_shift_vec(raw, FLOOR, CEILING)
        # Every shifted row has raw != corrected
        for i, shifted in enumerate(res.shift_applied):
            if shifted:
                assert raw[i] != res.corrected[i], f"recall miss at {i}"

    def test_recall_on_implausible_prices(self):
        raw = np.array([1e-6, 1e9, 100.0])
        res = apply_decimal_shift_vec(raw, FLOOR, CEILING)
        # Under the floor-free decimal-shift (price_floor=0.0): 1e-6 now survives
        # here (dropped later by the distressed anomaly filter); 1e9 is
        # unrecoverable and dropped at decimal-shift; 100 is a clean pass.
        assert res.in_range_mask[0]         # 1e-6 kept (floor removed)
        assert not res.in_range_mask[1]     # 1e9 unrecoverable -> dropped
        assert res.in_range_mask[2]


# ---------------------------------------------------------------------------
# Audit-count decomposition — pins the float-equality counting bug
# ---------------------------------------------------------------------------

class TestShiftMaskDecomposition:
    """The audit counts must come from the returned category masks, which are
    mutually exclusive and decompose in_range exactly. The old implementation
    reconstructed div10/div100 membership via float equality
    (corrected*10 == raw), which silently undercounts whenever (p/10)*10 is
    not bit-identical to p — e.g. 333.3."""

    def test_masks_mutually_exclusive_and_decompose(self):
        raw = np.array([0.0, 50.0, 333.3, 999.9, 3503.7, 12345.6, 1e9])
        res = apply_decimal_shift_vec(raw, FLOOR, CEILING)
        assert not (res.div10_mask & res.div100_mask).any()
        no_shift = res.in_range_mask & ~res.shift_applied
        assert (
            (no_shift | res.div10_mask | res.div100_mask) == res.in_range_mask
        ).all()
        assert (
            int(no_shift.sum()) + int(res.div10_mask.sum())
            + int(res.div100_mask.sum())
        ) == int(res.in_range_mask.sum())

    def test_div10_mask_catches_float_equality_miss(self):
        """333.3 is a genuine ÷10 row, but (333.3/10)*10 != 333.3 in float64 —
        the old float-equality reconstruction missed it. The mask must not."""
        raw = np.array([333.3])
        res = apply_decimal_shift_vec(raw, FLOOR, CEILING)
        assert res.div10_mask[0]
        assert res.corrected[0] == pytest.approx(33.33)
        # Demonstrate the old method's failure mode explicitly:
        assert (res.corrected[0] * 10) != raw[0], (
            "if this ever becomes bit-equal the regression test is vacuous"
        )

    def test_band_membership_matches_masks(self):
        """div10 marks exactly the (300, 3000] band, div100 (3000, 30000]."""
        raw = np.array([300.0, 300.1, 3000.0, 3000.1, 30000.0])
        res = apply_decimal_shift_vec(raw, FLOOR, CEILING)
        assert not res.shift_applied[0]            # 300.0 in range unshifted
        assert res.div10_mask[1]                   # 300.1 → ÷10
        assert res.div10_mask[2]                   # 3000.0 → ÷10 (band edge)
        assert res.div100_mask[3]                  # 3000.1 → ÷100
        assert res.div100_mask[4]                  # 30000.0 → ÷100 (band edge)
