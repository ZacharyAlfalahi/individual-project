"""
Unit tests for the accrued-interest + coupon engine (agents/quant/library/accrual.py).

These are the POSITIVE magnitude checks the validation protocol requires (not
just differentials-static):
  - 30/360 day counts match hand values.
  - Accrued interest matches a hand-computed value at known dates.
  - A coupon month pays coupon/frequency and AI resets.
  - Coupon-cycle recovery: total interest income over a full coupon period equals
    one period's coupon (no leak, no double-count).
  - Zero-coupon invariance: coupon == 0 → AI = C = 0 (the control group).
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))
from agents.quant.library.accrual import days_30_360, accrued_and_coupon  # noqa: E402


def _d(strs):
    return pd.to_datetime(list(strs)).values


def test_days_30_360_hand_values():
    assert days_30_360(_d(["2009-11-15"]), _d(["2009-12-31"]))[0] == 45  # D2=30 (D1=15≠30→min(31,30))
    assert days_30_360(_d(["2010-01-15"]), _d(["2010-07-15"]))[0] == 180  # exactly a semiannual period
    assert days_30_360(_d(["2010-05-15"]), _d(["2010-05-31"]))[0] == 15   # 30 - 15


def test_accrued_interest_hand_value():
    # 8% semiannual, maturity 2010-05-15 → coupons ...2009-11-15, 2010-05-15.
    # At 2009-12-31: last coupon 2009-11-15, 30/360 days = 45 → AI = 8 * 45/360 = 1.0.
    ai, c = accrued_and_coupon(_d(["2009-12-31"]), _d(["2010-05-15"]),
                               coupon=[8.0], frequency=[2])
    assert ai[0] == pytest.approx(1.0, abs=1e-9)
    assert c[0] == 0.0  # December is not a coupon month


def test_coupon_month_pays_and_resets_ai():
    # At 2010-05-31: coupon 2010-05-15 just paid → C = 8/2 = 4.0; AI accrues only
    # from the 15th → 8 * 15/360 = 0.3333.
    ai, c = accrued_and_coupon(_d(["2010-05-31"]), _d(["2010-05-15"]),
                               coupon=[8.0], frequency=[2])
    assert c[0] == pytest.approx(4.0, abs=1e-9)
    assert ai[0] == pytest.approx(8.0 * 15 / 360, abs=1e-9)


def test_coupon_cycle_recovery():
    # Total interest income over a full coupon period (Nov→May, spanning the May
    # coupon) = (AI_end − AI_start) + sum(coupons) = one semiannual coupon = 4.0.
    months = ["2009-11-30", "2009-12-31", "2010-01-31", "2010-02-28",
              "2010-03-31", "2010-04-30", "2010-05-31"]
    mat = ["2010-05-15"] * len(months)
    ai, c = accrued_and_coupon(_d(months), _d(mat), coupon=[8.0] * 7, frequency=[2] * 7)
    # Interest income from month-end A to B = (AI_B - AI_A) + coupons in (A, B].
    # The start month's own coupon (Nov, paid Nov-15 <= Nov-30) belongs to the
    # PRIOR cycle, so it is excluded via c[1:].
    total_interest = (ai[-1] - ai[0]) + c[1:].sum()
    assert total_interest == pytest.approx(4.0, abs=1e-9)   # = one coupon / frequency


def test_zero_coupon_invariance():
    # coupon == 0 → no AI, no coupon, for every date (the Z control group).
    months = ["2009-11-30", "2010-02-28", "2010-05-31"]
    ai, c = accrued_and_coupon(_d(months), _d(["2010-05-15"] * 3),
                               coupon=[0.0] * 3, frequency=[0] * 3)
    assert np.all(ai == 0.0)
    assert np.all(c == 0.0)


def test_no_accrual_after_maturity_month():
    # A bond is dead after it redeems: rows whose month-end falls AFTER the
    # maturity month must return AI = C = 0. Before the maturity guard, a
    # 2010-05-15 maturity fabricated AI=0.333/C=4.0 at 2010-06-30 and AI=2.333
    # at 2010-09-30 — accrual running months past redemption.
    dates = ["2010-06-30", "2010-09-30", "2011-01-31"]
    ai, c = accrued_and_coupon(_d(dates), _d(["2010-05-15"] * 3),
                               coupon=[8.0] * 3, frequency=[2] * 3)
    assert np.all(ai == 0.0)
    assert np.all(c == 0.0)


def test_maturity_month_still_accrues():
    # The maturity month itself is NOT dead: the final coupon is paid and AI
    # accrues to month-end per the existing month-end convention (guard is
    # date_index <= maturity_index, inclusive of the maturity month).
    ai, c = accrued_and_coupon(_d(["2010-05-31"]), _d(["2010-05-15"]),
                               coupon=[8.0], frequency=[2])
    assert c[0] == pytest.approx(4.0, abs=1e-9)
    assert ai[0] == pytest.approx(8.0 * 15 / 360, abs=1e-9)


def test_quarterly_frequency():
    # 6% quarterly (freq=4), maturity 2010-12-20 → coupons on the 20th every 3
    # months: ...Mar, Jun, Sep, Dec. September IS a coupon month: at 2010-09-30
    # the Sep-20 coupon was just paid → C = 6/4 = 1.5; AI accrues 30/360 from the
    # 20th → 6 * 10/360.
    ai, c = accrued_and_coupon(_d(["2010-09-30"]), _d(["2010-12-20"]),
                               coupon=[6.0], frequency=[4])
    assert ai[0] == pytest.approx(6.0 * 10 / 360, abs=1e-9)
    assert c[0] == pytest.approx(1.5, abs=1e-9)
