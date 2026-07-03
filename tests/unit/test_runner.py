"""
Unit tests for run_from_config: refusal short-circuit, and F3 -- both the base
engine (h=1) and overlap (h>1) paths return ONE normalized envelope shape.
"""

import numpy as np
import pandas as pd
import pytest

from agents.quant.config import (
    Binding,
    ConfigRefusal,
    Evidence,
    Inherited,
    RefusalCode,
    build_quant_config,
    run_from_config,
)

_ENVELOPE_KEYS = {
    "monthly_returns",
    "summary",
    "relationship_to_benchmark",
    "settings_used",
    "bookkeeping",
}


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
        groups=Inherited(5, "STATED", Evidence(quote="quintiles")),
        weighting=Inherited("size", "DESIGN", Evidence(note="VW par")),
        long_group=Inherited(0, "STATED", Evidence(quote="losers")),
        short_group=Inherited(4, "STATED", Evidence(quote="winners")),
    )


def _mom6_cfg():
    return build_quant_config(
        "mom6",
        Binding("mom6", "BOUND", Evidence(column="mom6", note="6m momentum")),
        groups=Inherited(10, "STATED", Evidence(quote="deciles")),
        weighting=Inherited("equal", "STATED", Evidence(quote="EW")),
        long_group=Inherited(9, "STATED", Evidence(quote="winners")),
        short_group=Inherited(0, "STATED", Evidence(quote="losers")),
        signal_lag=Inherited(1, "STATED", Evidence(quote="skip")),
        holding_period=Inherited(6, "STATED", Evidence(quote="H=6")),
    )


# --- a refusal short-circuits the engine entirely --------------------------

def test_refusal_short_circuits_engine(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("engine must not be called on a refusal")

    monkeypatch.setattr("agents.quant.config.runner.run_characteristic_sort", boom)
    monkeypatch.setattr("agents.quant.config.runner.run_with_holding_period", boom)

    refusal = ConfigRefusal("x", RefusalCode.MISSING_BINDING, "score", "no col", None)
    assert run_from_config(refusal, _panel()) is refusal


# --- F3: one envelope shape across both holding paths ----------------------

def test_base_engine_envelope():
    res = run_from_config(_str_cfg(), _panel())
    assert set(res) == _ENVELOPE_KEYS
    assert isinstance(res["monthly_returns"], pd.DataFrame)
    assert len(res["monthly_returns"]) > 0


def test_overlap_envelope_matches_base_shape():
    res = run_from_config(_mom6_cfg(), _panel())
    assert set(res) == _ENVELOPE_KEYS
    mr = res["monthly_returns"]
    assert isinstance(mr, pd.DataFrame)
    assert "n_cohorts_alive" in mr.columns  # overlap-specific column preserved
    assert len(mr) > 0
    assert "sharpe" in res["summary"]  # normalized summary present


def test_benchmark_with_multimonth_hold_raises():
    bench = pd.DataFrame({"date": pd.date_range("2010-01-31", periods=3, freq="ME"),
                          "MKT": [0.01, -0.01, 0.0]})
    with pytest.raises(ValueError):
        run_from_config(_mom6_cfg(), _panel(), benchmark=bench)


def test_safe_rate_with_multimonth_hold_raises():
    rf = pd.DataFrame({"date": pd.date_range("2010-01-31", periods=3, freq="ME"),
                       "rf": [0.001, 0.001, 0.001]})
    with pytest.raises(ValueError):
        run_from_config(_mom6_cfg(), _panel(), safe_rate=rf)


# --- T9: the two paths agree on the summary key set, not just the envelope --

def test_summary_key_parity_across_paths():
    panel = _panel()
    base = run_from_config(_str_cfg(), panel)      # h=1 -> base engine
    overlap = run_from_config(_mom6_cfg(), panel)  # h>1 -> normalized overlap
    assert set(base["summary"]) == set(overlap["summary"])
