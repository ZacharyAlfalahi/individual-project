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

from agents.auditor.ipca_differential.differential import _ANCHOR_RULEBOOKS, _anchor_series


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


def test_str_and_mom6_rulebooks_are_decile_sorts():
    assert _ANCHOR_RULEBOOKS["str"]["groups"] == 10 and _ANCHOR_RULEBOOKS["str"]["score"] == "xret"
    assert _ANCHOR_RULEBOOKS["mom6"]["groups"] == 10 and _ANCHOR_RULEBOOKS["mom6"]["weighting"] == "equal"


def test_unknown_anchor_raises():
    with pytest.raises(ValueError, match="unknown anchor"):
        _anchor_series(_sortable_panel(), "not_an_anchor")
