"""
Unit tests for run_strategy (D28, ledger item 40): strategy-level execution +
leg combination on top of run_from_config.

  * single_leg is a verbatim pass-through of the one leg's envelope.
  * equal_average is the by-date arithmetic mean of the per-leg strategy_ret
    series, with the ADAPTIVE divisor -- a month where a leg is absent divides
    by the count of legs present that month, NOT the nominal leg count. The
    adaptive-divisor pin is the item-40 regression.
  * a refused strategy is returned UNRUN; the engine is never touched.
"""

import numpy as np
import pandas as pd
import pytest

from agents.librarian.adapter.result import AdaptResult, CombinerInstruction, LegCall
from agents.quant.config import (
    Binding,
    ConfigRefusal,
    Evidence,
    Inherited,
    Locator,
    RefusalCode,
    StrategyResult,
    build_quant_config,
    run_from_config,
    run_strategy,
)

_LOC = Locator(1, 0, 1)  # placeholder span for synthetic STATED fixtures (D7 needs a locator)


# --- synthetic panel + two h=1 leg configs (mirrors test_runner.py) ---------

def _panel(n_bonds=60, n_months=24, seed=0):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2010-01-31", periods=n_months, freq="ME")
    rows = []
    for b in range(n_bonds):
        cusip = f"B{b:05d}"
        base = float(rng.normal())
        for d in dates:
            rows.append(
                {
                    "cusip": cusip,
                    "date": d,
                    "ret": float(rng.normal(0.001, 0.02)),
                    "size": float(rng.uniform(1e6, 1e8)),
                    "score": base + float(rng.normal(0, 0.5)),
                    "mom6": base + float(rng.normal(0, 0.5)),
                }
            )
    return pd.DataFrame(rows)


def _str_cfg():
    return build_quant_config(
        "str",
        Binding("score", "BOUND", Evidence(column="score", note="prior-month return")),
        groups=Inherited(5, "STATED", Evidence(locator=_LOC, quote="quintiles")),
        weighting=Inherited("size", "DESIGN", Evidence(note="VW par")),
        long_group=Inherited(0, "STATED", Evidence(locator=_LOC, quote="losers")),
        short_group=Inherited(4, "STATED", Evidence(locator=_LOC, quote="winners")),
    )


def _mom6_h1_cfg():
    return build_quant_config(
        "mom6h1",
        Binding("mom6", "BOUND", Evidence(column="mom6", note="6m momentum")),
        groups=Inherited(5, "STATED", Evidence(locator=_LOC, quote="quintiles")),
        weighting=Inherited("equal", "STATED", Evidence(locator=_LOC, quote="EW")),
        long_group=Inherited(4, "STATED", Evidence(locator=_LOC, quote="winners")),
        short_group=Inherited(0, "STATED", Evidence(locator=_LOC, quote="losers")),
    )


def _adapt(configs, kind, *, label="strat", variant=False):
    combiner = CombinerInstruction(kind, "available" if kind == "equal_average" else None)
    legs = tuple(
        LegCall(f"{label}_leg{i + 1}", {}, cfg) for i, cfg in enumerate(configs)
    )
    return AdaptResult(
        strategy_label=label, leg_calls=legs, combiner=combiner, variant=variant
    )


def _series(mr):
    """Reconstruct a date-indexed strategy_ret series from a monthly_returns frame."""
    return pd.Series(
        mr["strategy_ret"].values, index=pd.DatetimeIndex(mr["date"].values)
    )


def _envelope(dates, rets, *, months_per_year=12, nw_lags=None, avg_bonds=10.0):
    """A minimal five-key envelope for isolating the combine logic."""
    mr = pd.DataFrame({"date": pd.DatetimeIndex(dates), "strategy_ret": list(rets)})
    return {
        "monthly_returns": mr,
        "summary": {"avg_bonds_per_month": avg_bonds},
        "relationship_to_benchmark": {},
        "settings_used": {"months_per_year": months_per_year, "nw_lags": nw_lags},
        "bookkeeping": {},
    }


# --- single_leg is a verbatim pass-through ----------------------------------

def test_single_leg_passthrough_matches_run_from_config():
    panel = _panel()
    cfg = _str_cfg()
    direct = run_from_config(cfg, panel)
    res = run_strategy(_adapt([cfg], "single_leg"), panel)

    assert isinstance(res, StrategyResult)
    assert res.strategy_label == "strat"
    assert res.n_legs == 1
    assert res.combiner == {"kind": "single_leg"}
    pd.testing.assert_frame_equal(res.monthly_returns, direct["monthly_returns"])
    assert res.relationship_to_benchmark == direct["relationship_to_benchmark"]
    assert res.settings_used == direct["settings_used"]
    assert res.bookkeeping == direct["bookkeeping"]
    # NaN-aware summary equality (deterministic engine -> identical values).
    assert res.summary.keys() == direct["summary"].keys()
    for k, a in res.summary.items():
        b = direct["summary"][k]
        if isinstance(a, float) and pd.isna(a):
            assert pd.isna(b)
        else:
            assert a == b


def test_single_leg_benchmark_flows_through():
    panel = _panel()
    bench = pd.DataFrame(
        {"date": pd.date_range("2010-01-31", periods=24, freq="ME"),
         "MKT": np.linspace(-0.01, 0.01, 24)}
    )
    res = run_strategy(_adapt([_str_cfg()], "single_leg"), panel, benchmark=bench)
    # single_leg h=1 has a benchmark path -> relationship is populated (not {}).
    assert res.relationship_to_benchmark != {}


# --- equal_average is the by-date mean --------------------------------------

def test_equal_average_is_by_date_mean():
    panel = _panel()
    cfg_a, cfg_b = _str_cfg(), _mom6_h1_cfg()
    env_a, env_b = run_from_config(cfg_a, panel), run_from_config(cfg_b, panel)
    expected = pd.concat(
        [_series(env_a["monthly_returns"]), _series(env_b["monthly_returns"])], axis=1
    ).mean(axis=1).sort_index()

    res = run_strategy(_adapt([cfg_a, cfg_b], "equal_average"), panel)

    assert isinstance(res, StrategyResult)
    assert res.n_legs == 2
    assert res.combiner == {"kind": "equal_average", "divisor": "available"}
    assert res.relationship_to_benchmark == {}
    assert res.bookkeeping == {}
    got = _series(res.monthly_returns).sort_index()
    pd.testing.assert_series_equal(got, expected, check_names=False)


def test_adaptive_divisor_pin(monkeypatch):
    """Item-40 pin: a month where one leg is absent divides by the PRESENT count
    (1), not the nominal leg count (2) -- that month equals the other leg alone."""
    dates = pd.date_range("2010-01-31", periods=3, freq="ME")
    env_a = _envelope(dates, [0.02, 0.04, 0.06])       # present all three months
    env_b = _envelope(dates[[0, 2]], [0.10, 0.30])      # absent the middle month
    envs = iter([env_a, env_b])
    monkeypatch.setattr(
        "agents.quant.config.runner.run_from_config", lambda *a, **k: next(envs)
    )

    res = run_strategy(_adapt([_str_cfg(), _str_cfg()], "equal_average"), _panel())
    got = _series(res.monthly_returns)

    assert got.loc[dates[0]] == pytest.approx((0.02 + 0.10) / 2)  # both present
    assert got.loc[dates[1]] == pytest.approx(0.04)               # A alone -- NOT halved
    assert got.loc[dates[2]] == pytest.approx((0.06 + 0.30) / 2)  # both present


def test_equal_average_all_empty_is_nan_not_error(monkeypatch):
    empty = _envelope(pd.DatetimeIndex([]), [])
    envs = iter([empty, empty])
    monkeypatch.setattr(
        "agents.quant.config.runner.run_from_config", lambda *a, **k: next(envs)
    )
    res = run_strategy(_adapt([_str_cfg(), _str_cfg()], "equal_average"), _panel())
    assert len(res.monthly_returns) == 0
    assert res.summary["n_months"] == 0
    assert pd.isna(res.summary["average"])


def test_equal_average_avg_bonds_is_sum_of_legs(monkeypatch):
    """avg_bonds_per_month for equal_average is the SUM of per-leg averages (total
    bonds the strategy deploys per month across legs), NOT their mean. A leg reporting
    NaN drops from the sum (pandas sum skips NaN)."""
    dates = pd.date_range("2010-01-31", periods=3, freq="ME")
    env_a = _envelope(dates, [0.01, 0.02, 0.03], avg_bonds=12.0)
    env_b = _envelope(dates, [0.04, 0.05, 0.06], avg_bonds=8.0)
    envs = iter([env_a, env_b])
    monkeypatch.setattr(
        "agents.quant.config.runner.run_from_config", lambda *a, **k: next(envs)
    )
    res = run_strategy(_adapt([_str_cfg(), _str_cfg()], "equal_average"), _panel())
    assert res.summary["avg_bonds_per_month"] == pytest.approx(20.0)  # 12 + 8, not mean 10


def test_equal_average_avg_bonds_nan_when_all_legs_report_nan(monkeypatch):
    """The notna-any else-branch: if EVERY leg reports NaN avg_bonds, the combined
    value is NaN (not 0.0 from an empty sum)."""
    dates = pd.date_range("2010-01-31", periods=2, freq="ME")
    env_a = _envelope(dates, [0.01, 0.02], avg_bonds=float("nan"))
    env_b = _envelope(dates, [0.03, 0.04], avg_bonds=float("nan"))
    envs = iter([env_a, env_b])
    monkeypatch.setattr(
        "agents.quant.config.runner.run_from_config", lambda *a, **k: next(envs)
    )
    res = run_strategy(_adapt([_str_cfg(), _str_cfg()], "equal_average"), _panel())
    assert pd.isna(res.summary["avg_bonds_per_month"])


# --- a refused strategy is returned UNRUN -----------------------------------

def test_refused_strategy_returned_unrun(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("the engine must not run on a refused strategy")

    monkeypatch.setattr("agents.quant.config.runner.run_from_config", boom)
    monkeypatch.setattr("agents.quant.config.runner.run_characteristic_sort", boom)
    monkeypatch.setattr("agents.quant.config.runner.run_with_holding_period", boom)

    refusal = ConfigRefusal("strat_leg1", RefusalCode.MISSING_BINDING, "score", "no col", None)
    ar = AdaptResult(
        strategy_label="strat",
        leg_calls=(LegCall("strat_leg1", {}, refusal),),
        combiner=CombinerInstruction("single_leg", None),
    )
    assert ar.refused
    out = run_strategy(ar, _panel())
    assert out is ar  # the AdaptResult unchanged, carrying its refusal provenance


# --- metadata + edge rules --------------------------------------------------

def test_variant_propagates():
    res = run_strategy(_adapt([_str_cfg()], "single_leg", variant=True), _panel())
    assert res.variant is True


def test_benchmark_with_equal_average_raises():
    bench = pd.DataFrame(
        {"date": pd.date_range("2010-01-31", periods=3, freq="ME"),
         "MKT": [0.01, -0.01, 0.0]}
    )
    with pytest.raises(ValueError):
        run_strategy(
            _adapt([_str_cfg(), _mom6_h1_cfg()], "equal_average"), _panel(), benchmark=bench
        )
