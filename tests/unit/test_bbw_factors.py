"""
Unit tests for the BBW factor wiring (agents/quant/library/bbw_factors.py).

The engine is tested elsewhere; here we guard the BBW-specific definitions:
  - each factor's rulebook maps the right axes / legs (the error-prone part),
  - on a clean 5×5 grid the leg directions give the exact expected spreads
    (DRF = high-VaR − low-VaR; CRF_VaR = low-rating − high-rating; REV =
    losers − winners),
  - compose_crf is the equal-weighted mean of the three CRF components on the
    common date set.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))
from agents.quant.library.bbw_factors import (  # noqa: E402
    BBW_FACTOR_CONFIGS, factor_rulebook, run_bbw_factor, compose_crf,
)

M0 = pd.Timestamp("2010-01-31")
M1 = pd.Timestamp("2010-02-28")


def test_rulebook_leg_directions():
    drf = factor_rulebook("drf")
    assert drf["score"] == "var_5pct" and drf["control"] == "rating"
    assert drf["long_group"] == 4 and drf["short_group"] == 0   # high-VaR − low-VaR
    rev = factor_rulebook("rev")
    assert rev["score"] == "rev"
    assert rev["long_group"] == 0 and rev["short_group"] == 4   # losers − winners
    crf = factor_rulebook("crf_var")
    assert crf["score"] == "rating" and crf["control"] == "var_5pct"
    assert crf["long_group"] == 4 and crf["short_group"] == 0   # low-rating − high-rating
    for name in BBW_FACTOR_CONFIGS:
        rb = factor_rulebook(name)
        assert rb["groups"] == 5 and rb["control_groups"] == 5 and rb["weighting"] == "by_size"


def _grid_panel(score_col: str, ret_fn) -> pd.DataFrame:
    """5×5 grid: one bond per (rating group r, score group v) cell. Formation
    month carries score_col=v and rating=r; the realisation month carries the
    held return ret_fn(v, r). size=1 so by_size == equal weight."""
    rows = []
    for r in range(5):
        for v in range(5):
            cusip = f"R{r}V{v}"
            rows.append({"cusip": cusip, "date": M0, "ret": np.nan, "size": 1.0,
                         score_col: float(v), "rating": float(r)})
            rows.append({"cusip": cusip, "date": M1, "ret": float(ret_fn(v, r)), "size": 1.0,
                         score_col: np.nan, "rating": np.nan})
    return pd.DataFrame(rows)


def test_drf_and_crf_var_exact_spreads():
    # next return = 0.01*VaRgroup + 0.02*ratinggroup → DRF = 4*0.01 = 0.04 per
    # rating stripe; CRF_VaR = 4*0.02 = 0.08 per VaR stripe.
    panel = _grid_panel("var_5pct", lambda v, r: 0.01 * v + 0.02 * r)
    drf = run_bbw_factor(panel, "drf")["monthly_returns"].set_index("date")
    crf_var = run_bbw_factor(panel, "crf_var")["monthly_returns"].set_index("date")
    assert drf.loc[M1, "strategy_ret"] == pytest.approx(0.04, abs=1e-12)
    assert crf_var.loc[M1, "strategy_ret"] == pytest.approx(0.08, abs=1e-12)


def test_rev_is_losers_minus_winners():
    # Reversal: next return DECREASES in the prior-return signal (−0.01*rev) and
    # increases in rating group (+0.02*r). REV (long losers=group0) = +0.04;
    # CRF_REV (long low-rating=group4) = +0.08.
    panel = _grid_panel("rev", lambda v, r: -0.01 * v + 0.02 * r)
    rev = run_bbw_factor(panel, "rev")["monthly_returns"].set_index("date")
    crf_rev = run_bbw_factor(panel, "crf_rev")["monthly_returns"].set_index("date")
    assert rev.loc[M1, "strategy_ret"] == pytest.approx(0.04, abs=1e-12)
    assert crf_rev.loc[M1, "strategy_ret"] == pytest.approx(0.08, abs=1e-12)


def test_compose_crf_is_mean_on_common_dates():
    def mr(vals):
        return pd.DataFrame({"date": [M0, M1], "strategy_ret": vals})
    comps = {
        "crf_var":   mr([0.030, 0.060]),
        "crf_illiq": mr([0.000, 0.030]),
        "crf_rev":   mr([0.060, 0.090]),
    }
    out = compose_crf(comps).set_index("date")
    assert out.loc[M0, "crf"] == pytest.approx((0.030 + 0.000 + 0.060) / 3, abs=1e-12)
    assert out.loc[M1, "crf"] == pytest.approx((0.060 + 0.030 + 0.090) / 3, abs=1e-12)


def test_compose_crf_inner_joins_dates():
    # crf_illiq missing M1 → composite only spans the common date M0.
    out = compose_crf({
        "crf_var":   pd.DataFrame({"date": [M0, M1], "strategy_ret": [0.01, 0.02]}),
        "crf_illiq": pd.DataFrame({"date": [M0],     "strategy_ret": [0.03]}),
        "crf_rev":   pd.DataFrame({"date": [M0, M1], "strategy_ret": [0.05, 0.06]}),
    })
    assert list(out["date"]) == [M0]
