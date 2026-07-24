"""
Anchor routing: str/mom6 route through the single-sort engine, drf through
the bivariate BBW factor. All three produce a monthly long-short return series indexed by return-month
period. (The drf path is additionally validated end-to-end on the real dev panel by the smoke; this
test covers str/mom6 cheaply on synthetic data.)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from agents.auditor.ipca_differential.differential import _STR_RULEBOOK, _anchor_series, _mom6_config


def _sortable_panel(n_bonds: int = 120, n_months: int = 18, seed: int = 0) -> pd.DataFrame:
    """A view()-shaped panel with the columns the sort engine needs: cusip, date, ret, size, and the
    anchor score/control columns (xret, mom6, var_5pct, rating)."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2005-01-31", periods=n_months, freq="ME")
    rows = []
    for c in range(n_bonds):
        for d in dates:
            rows.append({
                "cusip": f"B{c:04d}", "date": d,
                "ret": float(rng.normal(0, 0.02)), "size": float(rng.uniform(1.0, 10.0)),
                "xret": float(rng.normal(0, 0.02)), "mom6": float(rng.normal()),
                "var_5pct": float(rng.normal()), "rating": float(rng.integers(1, 22)),
            })
    return pd.DataFrame(rows)


@pytest.mark.parametrize("anchor", ["str", "mom6", "drf"])
def test_anchor_routes_to_a_monthly_series(anchor):
    s = _anchor_series(_sortable_panel(), anchor)
    assert isinstance(s, pd.Series)
    assert isinstance(s.index, pd.PeriodIndex)
    assert len(s) >= 1
    assert np.isfinite(s.to_numpy()).all()


def test_str_rulebook_is_a_value_weighted_decile_sort():
    assert _STR_RULEBOOK["score"] == "xret" and _STR_RULEBOOK["groups"] == 10
    assert _STR_RULEBOOK["weighting"] == "by_size"
    assert _STR_RULEBOOK["long_group"] == 9 and _STR_RULEBOOK["short_group"] == 0   # P10/P1 explicit


def test_mom6_anchor_uses_jostova_skip_and_staggered_holding():
    """H1 regression: mom6 must be the canonical skip+staggered factor, not a plain 1-month sort."""
    c = _mom6_config()
    assert c["skip_months"] == 1 and c["holding_months"] == 6
    assert c["n_groups"] == 10 and c["weighting"] == "equal"


def test_unknown_anchor_raises():
    with pytest.raises(ValueError, match="unknown anchor"):
        _anchor_series(_sortable_panel(), "not_an_anchor")
