"""IPCA feed with instruments from one return basis and the return leg from another (synthetic, no data).

Pins `panels._feed_with_return_leg` / `build_cell_feed(return_panel=...)`: identical panels reproduce
the default feed byte-for-byte; a different return leg changes only R (by construction of the synthetic
panel: doubled xret ⇒ doubled R) while every instrument column is unchanged; mismatched rows are refused.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from agents.auditor.ipca_differential.panels import build_cell_feed
from agents.auditor.ipca_differential.runner import load_registry


def _view_panel(seed: int = 0, n_bonds: int = 14, n_months: int = 8) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    months = pd.date_range("2015-01-31", periods=n_months, freq="ME")
    rows = []
    for i in range(n_bonds):
        for d in months:
            rows.append({
                "cusip": f"C{i:02d}", "date": d,
                "xret": float(rng.normal(scale=0.02)),
                "rating": float(rng.integers(1, 20)),
                "time_to_maturity": float(rng.uniform(1, 20)),
                "mom6": float(rng.normal()), "var_5pct": float(rng.normal()),
                "gamma_illiq": float(rng.normal()), "bond_vol": float(rng.uniform(0.005, 0.05)),
            })
    return pd.DataFrame(rows).sort_values(["cusip", "date"]).reset_index(drop=True)


def _same(a, b) -> bool:
    return (len(a.Z) == len(b.Z) and all(np.array_equal(x, y) for x, y in zip(a.Z, b.Z))
            and all(np.array_equal(x, y) for x, y in zip(a.R, b.R))
            and np.array_equal(a.months, b.months))


def test_identical_return_panel_reproduces_the_default_feed():
    reg, v = load_registry(), _view_panel()
    default = build_cell_feed(v, reg, "corr")
    same_leg = build_cell_feed(v, reg, "corr", return_panel=v.copy())
    assert len(default.Z) > 0
    assert _same(default, same_leg)


def test_other_return_leg_changes_only_r():
    reg, v = load_registry(), _view_panel(seed=3)
    doubled = v.copy()
    doubled["xret"] = 2.0 * doubled["xret"]
    base = build_cell_feed(v, reg, "corr")
    other = build_cell_feed(v, reg, "corr", return_panel=doubled)
    assert np.array_equal(base.months, other.months)
    for zb, zo, rb, ro in zip(base.Z, other.Z, base.R, other.R):
        np.testing.assert_array_equal(zb, zo)                      # instruments (incl. str_reversal) unchanged
        np.testing.assert_allclose(ro, 2.0 * rb, rtol=0, atol=1e-15)


def test_return_leg_uses_return_panel_instrument_basis_not_swapped():
    # Instruments must come from the characteristics panel even when the return panel's xret differs:
    # the z_str_reversal column equals the default feed's (ranks of the characteristics basis's xret).
    reg, v = load_registry(), _view_panel(seed=5)
    shuffled = v.copy()
    rng = np.random.default_rng(9)
    shuffled["xret"] = rng.permutation(shuffled["xret"].to_numpy())
    base = build_cell_feed(v, reg, "corr")
    other = build_cell_feed(v, reg, "corr", return_panel=shuffled)
    for zb, zo in zip(base.Z, other.Z):
        np.testing.assert_array_equal(zb[:, 0], zo[:, 0])          # column 0 = z_str_reversal


def test_mismatched_rows_refused():
    reg, v = load_registry(), _view_panel(seed=1)
    with pytest.raises(ValueError, match="rows differ"):
        build_cell_feed(v, reg, "corr", return_panel=v.iloc[:-1].copy())
    with pytest.raises(KeyError, match="xret"):
        build_cell_feed(v, reg, "corr", return_panel=v.drop(columns=["xret"]))


def test_return_leg_is_joined_by_key_not_row_position():
    # A shuffled return panel gives the same feed as an ordered one, and a doubled xret still doubles R.
    reg, v = load_registry(), _view_panel(seed=7)
    doubled = v.copy()
    doubled["xret"] = 2.0 * doubled["xret"]
    ordered = build_cell_feed(v, reg, "corr", return_panel=doubled)
    shuffled = build_cell_feed(v, reg, "corr", return_panel=doubled.sample(frac=1.0, random_state=11))
    assert _same(ordered, shuffled)
    base = build_cell_feed(v, reg, "corr")
    for rb, rs in zip(base.R, shuffled.R):
        np.testing.assert_allclose(rs, 2.0 * rb, rtol=0, atol=1e-15)


def test_duplicate_return_key_refused():
    reg, v = load_registry(), _view_panel(seed=2)
    dup = pd.concat([v.iloc[:-1], v.iloc[[0]]], ignore_index=True)       # same length: one key twice, one missing
    with pytest.raises(ValueError, match="rows differ"):
        build_cell_feed(v, reg, "corr", return_panel=dup)


def test_missing_return_value_drops_only_the_affected_cells():
    reg, v = load_registry(), _view_panel(seed=4)
    holed = v.copy()
    holed.loc[(holed["cusip"] == "C03") & (holed["date"] == sorted(holed["date"].unique())[4]), "xret"] = np.nan
    base = build_cell_feed(v, reg, "corr", return_panel=v.copy())
    other = build_cell_feed(v, reg, "corr", return_panel=holed)
    n_base, n_other = sum(len(r) for r in base.R), sum(len(r) for r in other.R)
    assert 1 <= n_base - n_other <= 2      # the prior-month return cell, and at most the instrument-month row
    assert all(np.isfinite(r).all() for r in other.R)


def test_run_pair_full_builds_instruments_from_characteristics_and_returns_from_maximal(monkeypatch):
    import agents.auditor.ipca_differential.runner as RN

    calls = {"panel_states": [], "feeds": []}

    def fake_panel_states(bias, maximal, signals):
        calls["panel_states"].append(maximal)
        return (f"{maximal}_N", f"{maximal}_B")

    def fake_build_cell_feed(view_panel, reg, family, **kw):
        calls["feeds"].append((view_panel, kw.get("return_panel")))
        return object()

    class _Val:
        value = 0.0

    class _Res:
        interaction_bracket_raw = _Val()

    monkeypatch.setattr(RN, "panel_states", fake_panel_states)
    monkeypatch.setattr(RN, "build_cell_feed", fake_build_cell_feed)
    monkeypatch.setattr(RN, "_anchor_series", lambda p, a, **k: ("anchor_from", p))
    monkeypatch.setattr(RN, "differential_from_feeds", lambda *a, **k: _Res())
    monkeypatch.setattr(RN, "stability_diagnostic", lambda *a, **k: "stab")
    for name in ("load_ipca_lambda", "load_ipca_projection_gate", "load_ipca_bootstrap_config",
                 "load_ipca_stability_config"):
        monkeypatch.setattr(RN, name, lambda *a, **k: None)
    monkeypatch.setattr(RN, "load_ipca_reporting",
                        lambda *a, **k: type("R", (), {"focal_pairs": {}})())

    RN.run_pair_full("survivorship", "mom6", "TR", "SIG", {}, characteristics_maximal="CLEAN")
    assert calls["panel_states"] == ["TR", "CLEAN"]
    assert calls["feeds"] == [("CLEAN_N", "TR_N"), ("CLEAN_B", "TR_B")]

    calls["panel_states"].clear()
    calls["feeds"].clear()
    RN.run_pair_full("survivorship", "mom6", "TR", "SIG", {})
    assert calls["panel_states"] == ["TR"]
    assert calls["feeds"] == [("TR_N", None), ("TR_B", None)]
