"""
Per-panel-state signal recompute (decision b). Recomputes var_5pct/bond_vol/mom6 from a
view panel's own returns via the canonical build functions; leaves gamma_illiq (daily-sourced) and
everything else untouched. Deterministic.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from agents.auditor.ipca_differential.signal_recompute import RECOMPUTED_SIGNALS, recompute_signals


def _view_panel(n_bonds: int = 30, n_months: int = 48, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2005-01-31", periods=n_months, freq="ME")
    rows = []
    for c in range(n_bonds):
        for d in dates:
            rows.append({
                "cusip": f"B{c:03d}", "date": d,
                "ret": float(rng.normal(0, 0.02)), "xret": float(rng.normal(0, 0.02)),
                "var_5pct": 999.0, "bond_vol": 999.0, "mom6": 999.0,   # sentinels (must be overwritten)
                "gamma_illiq": 7.0, "rating": 3.0,
            })
    return pd.DataFrame(rows)


def test_recompute_overwrites_return_history_signals_and_keeps_gamma():
    pv = _view_panel()
    out = recompute_signals(pv)
    for col in RECOMPUTED_SIGNALS:                      # var_5pct, bond_vol, mom6
        assert (out[col].dropna() != 999.0).all()       # sentinel replaced by real recomputed values
        assert out[col].notna().any()                   # some non-NaN once the window fills
    assert (out["gamma_illiq"] == 7.0).all()            # daily-sourced → NOT recomputed
    assert set(out["cusip"]) == set(pv["cusip"]) and len(out) == len(pv)


def test_recompute_is_deterministic():
    pv = _view_panel(seed=3)
    a, b = recompute_signals(pv), recompute_signals(pv)
    for col in RECOMPUTED_SIGNALS:
        pd.testing.assert_series_equal(a[col], b[col])


def test_masking_returns_changes_the_recomputed_vol():
    """Propagation: masking the returns bond_vol consumes changes its recomputed value (the channel
    that would fire if stale_price masked anything on real data)."""
    pv = _view_panel(seed=5)
    masked = pv.copy()
    masked.loc[masked["cusip"] == "B000", "xret"] = np.nan
    a = recompute_signals(pv).set_index(["cusip", "date"])["bond_vol"]
    b = recompute_signals(masked).set_index(["cusip", "date"])["bond_vol"]
    assert not a.fillna(-1.0).equals(b.fillna(-1.0))
