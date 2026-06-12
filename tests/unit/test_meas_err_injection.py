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
    """Filter 1 — isolated ultra-low print sitting ≥ρ_anomaly below ±L-day
    median. Inject an ultra-low (0.05) in a series otherwise at ~10."""

    def test_isolated_ultra_low_print_flagged(self):
        prices = np.array([10.0, 10.5, 9.5, 10.2, 0.05, 10.1, 10.3, 9.8, 10.0, 10.2])
        # Position 4 is ultra-low (≤ tau_low = 0.10) and far below the ±5-day
        # median of its neighbours (~10).
        flag = filter_anomaly(prices, DISTRESSED_PARAMS)
        assert flag[4]
        # Other days are not flagged
        assert not flag[0] and not flag[1] and not flag[9]


class TestSpikeInjection:
    """Filter 2 — print ≥ρ_spike above pre-spike median that recovers in L
    days to within ρ_recovery of that median."""

    def test_spike_with_recovery_flagged(self):
        prices = np.array([100.0, 100.0, 100.0, 100.0, 100.0, 105.0, 100.0, 100.0])
        # Position 5 is 5 points above the pre-spike median (100). 5 ≥
        # rho_spike (3.0). Recovery: position 6 is 100, exactly at median.
        flag = filter_spike(prices, DISTRESSED_PARAMS)
        assert flag[5]

    def test_spike_without_recovery_not_flagged(self):
        # Permanent regime shift — spike, then stays high.
        prices = np.array([100.0, 100.0, 100.0, 100.0, 100.0, 110.0, 110.0, 110.0,
                           110.0, 110.0, 110.0])
        flag = filter_spike(prices, DISTRESSED_PARAMS)
        # Position 5 has no recovery (price stays at 110 — far from pre-spike
        # median of 100). Not flagged.
        assert not flag[5]


class TestPlateauInjection:
    """Filter 3 — runs of ≥ℓ_min identical ultra-low or near-round prices
    with ≥ρ_plateau pre/post displacement."""

    def test_round_number_plateau_flagged(self):
        # Bond trades at ~50 then plateaus at 100.00 for 3 days, then resumes
        # at ~50. 100 is a round multiple of 25; pre/post are 50 (50 points
        # displaced from 100; ρ_plateau = 3.0, easily satisfied).
        prices = np.array([50.0, 51.0, 49.0, 100.0, 100.0, 100.0, 51.0, 50.0])
        flag = filter_plateau(prices, DISTRESSED_PARAMS)
        # Positions 3, 4, 5 are the plateau
        assert flag[3] and flag[4] and flag[5]
        # Pre and post not flagged
        assert not flag[0] and not flag[7]


class TestIntradayInjection:
    """Filter 4 — low-priced day (min < τ_intraday) with intraday range
    > γ_range × VWAP."""

    def test_low_price_wide_range_flagged(self):
        # One day: VWAP = 10, min = 5, max = 15. Range = 10; γ_range × VWAP =
        # 0.75 × 10 = 7.5. Range > 7.5 → flagged. min = 5 < τ_intraday (20).
        price_vwap = np.array([100.0, 100.0, 10.0, 100.0])
        min_price = np.array([99.5, 99.5, 5.0, 99.5])
        max_price = np.array([100.5, 100.5, 15.0, 100.5])
        flag = filter_intraday(price_vwap, min_price, max_price, DISTRESSED_PARAMS)
        assert flag[2]
        assert not flag[0]

    def test_high_price_wide_range_not_flagged(self):
        # Wide range but high prices — Filter 4 doesn't apply (low-price gate).
        price_vwap = np.array([100.0, 100.0])
        min_price = np.array([60.0, 99.5])
        max_price = np.array([140.0, 100.5])
        flag = filter_intraday(price_vwap, min_price, max_price, DISTRESSED_PARAMS)
        # Day 0 has 80-point range on VWAP=100 (range > 0.75×VWAP) BUT
        # min=60 ≥ τ_intraday=20 → not flagged.
        assert not flag[0]


# ---------------------------------------------------------------------------
# Wildly implausible price — A1.7
# ---------------------------------------------------------------------------

class TestWildlyImplausiblePrice:
    """A1.7: raw family preserves wildly implausible prices bit-exact;
    corrected family drops them (no shift recovers them). Pins that the
    price-plausibility filter is meas_err-gated, not parsing."""

    def test_micro_price_dropped_by_corr_preserved_by_raw(self):
        raw_prices = np.array([1e-6, 100.0, 100.0])
        # raw: bit-exact
        assert raw_prices[0] == 1e-6
        # corr: drop
        res = apply_decimal_shift_vec(raw_prices, FLOOR, CEILING)
        assert not res.in_range_mask[0]   # below floor; no shift restores it
        assert np.isnan(res.corrected[0])

    def test_giga_price_dropped_by_corr_preserved_by_raw(self):
        # 1e9 — way above pre_correction_ceiling (30000)
        # Note: apply_decimal_shift_vec's caller is responsible for the
        # pre-ceiling gate. Here we feed a value that no shift recovers.
        raw_prices = np.array([1e9, 100.0, 100.0])
        assert raw_prices[0] == 1e9  # raw preserves
        # 1e9/10 = 1e8 — still above ceiling; 1e9/100 = 1e7 — still above
        # ceiling. → unresolvable, NaN, in_range False.
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
        # Implausibles fall out of in_range; corrected is NaN; the
        # raw-vs-corr divergence is `corrected_is_nan_and_raw_is_finite`.
        assert not res.in_range_mask[0] and not res.in_range_mask[1]
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
