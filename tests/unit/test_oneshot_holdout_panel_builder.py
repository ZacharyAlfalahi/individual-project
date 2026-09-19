"""one-shot holdout panel_builder: the inspection-verifiable parts (leakage predicate, slicing, gating, registry).

The dev-pseudo builder's end-to-end assembly is validated by the ``--rehearsal`` run against
the real development parquets (not exercised here); these tests pin the pure logic on synthetic data.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from agents.scientist.experimentalist.oneshot_holdout.panel_builder import (
    BUILDERS,
    OneshotHoldoutBuilderGated,
    _slice_months,
    dev_pseudo_builder,
    holdout_builder,
    zero_leakage_check,
)
from agents.scientist.experimentalist.oneshot_holdout.windows import Window


def _grid(start, end):
    months = pd.period_range(start, end, freq="M").to_timestamp("M")
    return pd.DataFrame({"date": months, "x": np.arange(len(months), dtype=float)})


def test_slice_months_is_inclusive():
    df = _grid("2017-01", "2022-06")
    sl = _slice_months(df, "2018-01", "2021-09")
    assert pd.to_datetime(sl["date"]).min() == pd.Timestamp("2018-01-31")
    assert pd.to_datetime(sl["date"]).max() == pd.Timestamp("2021-09-30")
    assert len(sl) == 45


def test_zero_leakage_check_passes_on_backward_dated_ratings():
    g = _grid("2022-01", "2022-06")
    g["_sel_rating_date"] = pd.to_datetime(g["date"]) - pd.Timedelta(days=40)   # all in the past
    zero_leakage_check(g, Window("2022-01", "2022-06", 6))                       # no raise


def test_zero_leakage_check_raises_on_future_rating():
    g = _grid("2022-01", "2022-06")
    sel = pd.to_datetime(g["date"]) - pd.Timedelta(days=40)
    sel.iloc[3] = pd.to_datetime(g["date"]).iloc[3] + pd.Timedelta(days=5)       # one future-dated
    g["_sel_rating_date"] = sel
    with pytest.raises(AssertionError):
        zero_leakage_check(g, Window("2022-01", "2022-06", 6))


def test_zero_leakage_check_requires_marker_columns():
    with pytest.raises(ValueError):
        zero_leakage_check(_grid("2022-01", "2022-03"), Window("2022-01", "2022-03", 3))


def test_holdout_builder_is_gated_and_refuses_in_development():
    with pytest.raises(OneshotHoldoutBuilderGated):
        holdout_builder()


def test_builders_registry_exposes_both():
    assert set(BUILDERS) == {"dev_pseudo", "holdout"}
    assert callable(dev_pseudo_builder())        # dev-pseudo factory returns a PanelBuilder callable


def test_seeded_feed_wall_is_the_window_end_not_the_development_end():
    """Behind the gate the seeded IPCA feed carries post-development months by design: the relocated wall
    accepts them up to the registered window end, refuses anything past it, and the frozen validator (whose
    wall is the development end) still refuses the same feed — so the wall's location is the only difference."""
    from agents.quant.library.ipca import ContractViolation
    from agents.quant.library.ipca_feed import build_ipca_feed, validate_feed
    from agents.scientist.experimentalist.oneshot_holdout.panel_builder import validate_seeded_feed
    from scripts.build_ipca_panel import load_registry

    reg = load_registry()
    rng = np.random.default_rng(0)
    months = pd.date_range("2019-01-31", "2025-09-30", freq="ME")
    rows = [{"cusip": f"C{i:02d}", "date": d, "str_reversal": float(rng.normal(scale=0.02)),
             "rating": float(rng.integers(1, 20)), "time_to_maturity": float(rng.uniform(1, 20)),
             "mom6": float(rng.normal()), "var_5pct": float(rng.normal()), "gamma_illiq": float(rng.normal()),
             "bond_vol": float(rng.uniform(0.005, 0.05))} for i in range(14) for d in months]
    merged = pd.DataFrame(rows)
    feed, _ = build_ipca_feed(merged, reg, "corr", validate=False)
    assert int(feed["month"].max()) > pd.Period(reg["meta"]["train_end"], "M").ordinal   # post-development months present
    validate_seeded_feed(feed, reg, "corr", last_month="2025-09")                          # accepted up to the window end
    with pytest.raises(ContractViolation):
        validate_seeded_feed(feed, reg, "corr", last_month="2024-12")                      # refused past it
    with pytest.raises(ContractViolation):
        validate_feed(feed, reg, "corr")                                                   # the frozen wall still refuses
