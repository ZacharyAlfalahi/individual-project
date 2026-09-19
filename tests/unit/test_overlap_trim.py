"""
The published return trim on the OVERLAPPING-HOLD path (holding_period > 1).

The contract these tests pin: on the overlapping-hold path a percentile trim is resolved before use
(a percentile LEVEL is never applied as an absolute return bound) and applies to every held-month
return, not only at formation.

Contract (spec E1/E2a; gold ``gold_mom6_jnps_2013.md`` fn.16 "eliminated return observations
above the 99.5th percentile" — sample-wide):
  * the percentile is resolved ONCE per call over the full-sample eligible next_ret — the SAME
    series and the SAME resolver as ``run_characteristic_sort``;
  * the resolved bounds apply at formation AND to every held-month return
    (truncate -> the observation is dropped and the leg renormalised; winsorise -> clipped);
  * an unresolved percentile can never reach ``_apply_trim_rule`` (it refuses).
Hand-computed answers throughout.
"""

import numpy as np
import pandas as pd
import pytest

from agents.quant.library.characteristic_sort import (
    _apply_trim_rule,
    extract_monthly_selections,
    resolve_trim_rule,
    run_characteristic_sort,
)
from agents.quant.library.overlap import run_with_holding_period


def _panel(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"]) + pd.offsets.MonthEnd(0)
    return df


def _held_outlier_panel() -> pd.DataFrame:
    """One formation month (Jan: only Jan rows carry a score), held for H=2 (Feb, Mar).
    Long leg = A, B (scores 4, 3); short leg = C, D (scores 2, 1). A's Mar return (+5.00)
    is an outlier in a HELD month — it is not A's formation-month next_ret (Feb, +0.01)."""
    nan = float("nan")
    rows = [
        {"cusip": "A", "date": "2010-01", "ret": 0.00, "size": 1.0, "score": 4.0},
        {"cusip": "B", "date": "2010-01", "ret": 0.00, "size": 1.0, "score": 3.0},
        {"cusip": "C", "date": "2010-01", "ret": 0.00, "size": 1.0, "score": 2.0},
        {"cusip": "D", "date": "2010-01", "ret": 0.00, "size": 1.0, "score": 1.0},
        {"cusip": "A", "date": "2010-02", "ret": 0.01, "size": 1.0, "score": nan},
        {"cusip": "B", "date": "2010-02", "ret": 0.03, "size": 1.0, "score": nan},
        {"cusip": "C", "date": "2010-02", "ret": 0.00, "size": 1.0, "score": nan},
        {"cusip": "D", "date": "2010-02", "ret": -0.02, "size": 1.0, "score": nan},
        {"cusip": "A", "date": "2010-03", "ret": 5.00, "size": 1.0, "score": nan},
        {"cusip": "B", "date": "2010-03", "ret": 0.02, "size": 1.0, "score": nan},
        {"cusip": "C", "date": "2010-03", "ret": 0.01, "size": 1.0, "score": nan},
        {"cusip": "D", "date": "2010-03", "ret": 0.01, "size": 1.0, "score": nan},
    ]
    return _panel(rows)


def _rulebook(trim_rule: dict | None = None) -> dict:
    rb = {"score": "score", "groups": 2, "weighting": "equal", "min_bonds": 4}
    if trim_rule is not None:
        rb["trim_rule"] = trim_rule
    return rb


def _abs_trim(method: str, hi: float) -> dict:
    return {"target": "return", "method": method,
            "bounds": {"type": "absolute", "hi": hi}, "sample": "full_sample"}


def _pct_trim(method: str, hi: float) -> dict:
    return {"target": "return", "method": method,
            "bounds": {"type": "percentile", "hi": hi, "percentile_method": "linear"},
            "sample": "full_sample"}


def _ret_on(mr: pd.DataFrame, month: str) -> float:
    return float(mr.set_index("date").loc[pd.Timestamp(month) + pd.offsets.MonthEnd(0),
                                          "strategy_ret"])


# ---------------------------------------------------------------------------
# Held-month application
# ---------------------------------------------------------------------------

def test_no_trim_carries_the_held_month_outlier():
    """Control: without a trim the held-month outlier enters the long leg.
    Feb: long (0.01+0.03)/2=0.02, short (0.00-0.02)/2=-0.01 -> 0.03.
    Mar: long (5.00+0.02)/2=2.51, short (0.01+0.01)/2=0.01 -> 2.50."""
    mr = run_with_holding_period(_held_outlier_panel(), _rulebook(), holding_period=2)
    assert _ret_on(mr, "2010-02") == pytest.approx(0.03, abs=1e-12)
    assert _ret_on(mr, "2010-03") == pytest.approx(2.50, abs=1e-12)


def test_truncate_drops_held_month_outlier_and_renormalises():
    """truncate hi=1.0: A's Mar +5.00 is eliminated; the long leg renormalises to B alone.
    Feb unchanged 0.03. Mar: long 0.02, short 0.01 -> 0.01 (2.50 untrimmed)."""
    mr = run_with_holding_period(_held_outlier_panel(), _rulebook(_abs_trim("truncate", 1.0)),
                                 holding_period=2)
    assert _ret_on(mr, "2010-02") == pytest.approx(0.03, abs=1e-12)
    assert _ret_on(mr, "2010-03") == pytest.approx(0.01, abs=1e-12)
    mar = mr.set_index("date").loc[pd.Timestamp("2010-03-31")]
    assert int(mar["n_bonds"]) == 3      # A's Mar observation dropped, B/C/D survive


def test_winsorise_clips_held_month_outlier():
    """winsorise hi=1.0: A's Mar +5.00 clipped to 1.00.
    Mar: long (1.00+0.02)/2=0.51, short 0.01 -> 0.50. Feb unchanged 0.03."""
    mr = run_with_holding_period(_held_outlier_panel(), _rulebook(_abs_trim("winsorise", 1.0)),
                                 holding_period=2)
    assert _ret_on(mr, "2010-02") == pytest.approx(0.03, abs=1e-12)
    assert _ret_on(mr, "2010-03") == pytest.approx(0.50, abs=1e-12)


# ---------------------------------------------------------------------------
# Percentile resolution on the overlap path
# ---------------------------------------------------------------------------

def test_percentile_resolves_over_eligible_next_ret_not_as_absolute_level():
    """The eligible full-sample series is Jan formation's next_ret [0.01, 0.03, 0.00, -0.02]
    (Feb/Mar rows carry no score). hi=0.75 linear -> 0.015, the same realised-threshold record the
    H=1 engine keeps.
    (Reading the level 0.75 as an absolute bound is the failure mode; its held-month consequence is
    pinned by
    test_percentile_cutoff_reaches_held_month_returns, not here.)"""
    rb = _rulebook(_pct_trim("truncate", 0.75))
    trim_abs, realised = resolve_trim_rule(_held_outlier_panel(), rb)
    expected = float(np.quantile([0.01, 0.03, 0.00, -0.02], 0.75, method="linear"))
    assert expected == pytest.approx(0.015)
    assert realised["n_obs"] == 4
    assert realised["hi"]["threshold"] == pytest.approx(expected, abs=1e-15)
    assert trim_abs["bounds"] == {"type": "absolute", "hi": realised["hi"]["threshold"]}

    # Same series, same threshold as the H=1 engine resolves (spec E condition 1).
    h1 = run_characteristic_sort(_held_outlier_panel(), rb)
    assert h1["bookkeeping"]["realised_trim_threshold"] == realised


@pytest.mark.parametrize("method", ["truncate", "winsorise"])
def test_percentile_equals_absolute_at_realised_threshold_h2(method):
    """Condition-5 analogue on the overlap path: a percentile trim is byte-identical to the
    absolute trim at its realised threshold, at formation AND across held months."""
    panel = _synth_panel(seed=11)
    pct = _rulebook(_pct_trim(method, 0.9))
    trim_abs, realised = resolve_trim_rule(panel, pct)
    assert realised is not None
    out_pct = run_with_holding_period(panel, pct, holding_period=3)
    out_abs = run_with_holding_period(panel, _rulebook(trim_abs), holding_period=3)
    assert len(out_pct) > 0
    pd.testing.assert_frame_equal(out_pct, out_abs)


def test_percentile_trim_moves_the_h3_series_vs_no_trim():
    """Non-vacuity guard for the synthetic panel used above (the trim bites at all). It does NOT by
    itself prove held months are trimmed — that contract is pinned by
    test_percentile_cutoff_reaches_held_month_returns."""
    panel = _synth_panel(seed=11)
    trimmed = run_with_holding_period(panel, _rulebook(_pct_trim("truncate", 0.9)), holding_period=3)
    untrimmed = run_with_holding_period(panel, _rulebook(), holding_period=3)
    assert not np.allclose(trimmed["strategy_ret"].to_numpy(), untrimmed["strategy_ret"].to_numpy())


def test_extract_monthly_selections_resolves_percentile_standalone():
    """Called directly (not via run_with_holding_period), a percentile rule must resolve — the
    selections equal those under the absolute trim at the realised threshold."""
    panel = _synth_panel(seed=5)
    pct = _rulebook(_pct_trim("truncate", 0.9))
    trim_abs, _ = resolve_trim_rule(panel, pct)
    sel_pct = extract_monthly_selections(panel, pct)
    sel_abs = extract_monthly_selections(panel, _rulebook(trim_abs))
    assert sel_pct.keys() == sel_abs.keys() and len(sel_pct) > 0
    for t in sel_pct:
        for leg in ("long", "short"):
            pd.testing.assert_frame_equal(sel_pct[t][leg], sel_abs[t][leg])


# ---------------------------------------------------------------------------
# Tripwire + regression invariants
# ---------------------------------------------------------------------------

def test_apply_trim_rule_refuses_unresolved_percentile():
    """A percentile LEVEL must never be read as an absolute bound."""
    df = pd.DataFrame({"next_ret": [0.5, 0.01]})
    with pytest.raises(ValueError, match="unresolved percentile"):
        _apply_trim_rule(df, _pct_trim("truncate", 0.995))


@pytest.mark.parametrize("method", ["truncate", "winsorise"])
def test_h1_with_percentile_trim_is_the_engine(method):
    """H=1 short-circuits to run_characteristic_sort: byte-identical with a percentile trim."""
    panel = _synth_panel(seed=3)
    rb = _rulebook(_pct_trim(method, 0.9))
    engine = run_characteristic_sort(panel, rb)["monthly_returns"]
    wrapper = run_with_holding_period(panel, rb, holding_period=1)
    pd.testing.assert_frame_equal(wrapper.drop(columns=["n_cohorts_alive"]), engine)


def test_resolve_trim_rule_passes_none_and_absolute_through():
    panel = _synth_panel(seed=3)
    assert resolve_trim_rule(panel, _rulebook()) == ({"method": "none"}, None)
    trim_abs = _abs_trim("truncate", 0.05)
    assert resolve_trim_rule(panel, _rulebook(trim_abs)) == (trim_abs, None)


def test_percentile_cutoff_reaches_held_month_returns():
    """The held-month pin: a RESOLVED percentile cutoff must trim a held-month return.
    Fails if the level 0.75 is read as an absolute bound (Feb 0.004167, Mar 2.488667)
    AND if the cutoff is resolved but held months stay untrimmed (Mar 2.492).

    Jan formation (only Jan rows carry a score), H=2. Eligible next_ret (Feb returns) =
    [A 0.01, B 0.03, C 0.00, D -0.02, E 0.005]; linear 75th percentile = sorted[3] = 0.01 exactly.
    truncate keeps r <= 0.01: B dropped at formation, A kept (equal to the bound).
    Groups of 2 over A(4), C(2), D(1), E(5): long {A, E}, short {C, D}.
    Feb: long (0.01+0.005)/2 = 0.0075, short (0.00-0.02)/2 = -0.01 -> 0.0175.
    Mar: A's held +5.00 > 0.01 is eliminated; E 0.004; C, D 0.01 kept (equal to the bound)
         -> long 0.004, short 0.01 -> -0.006."""
    nan = float("nan")
    jan = [("A", 4.0), ("B", 3.0), ("C", 2.0), ("D", 1.0), ("E", 5.0)]
    feb = {"A": 0.01, "B": 0.03, "C": 0.00, "D": -0.02, "E": 0.005}
    mar = {"A": 5.00, "B": 0.02, "C": 0.01, "D": 0.01, "E": 0.004}
    rows = [{"cusip": k, "date": "2010-01", "ret": 0.0, "size": 1.0, "score": s} for k, s in jan]
    rows += [{"cusip": k, "date": "2010-02", "ret": v, "size": 1.0, "score": nan} for k, v in feb.items()]
    rows += [{"cusip": k, "date": "2010-03", "ret": v, "size": 1.0, "score": nan} for k, v in mar.items()]
    panel = _panel(rows)
    rb = _rulebook(_pct_trim("truncate", 0.75))

    _trim_abs, realised = resolve_trim_rule(panel, rb)
    assert realised["hi"]["threshold"] == 0.01
    mr = run_with_holding_period(panel, rb, holding_period=2)
    assert _ret_on(mr, "2010-02") == pytest.approx(0.0175, abs=1e-12)
    assert _ret_on(mr, "2010-03") == pytest.approx(-0.006, abs=1e-12)


def test_trimmed_returns_bounds_inclusive_one_and_two_sided_nan_kept():
    from agents.quant.library.overlap import _trimmed_returns
    r = np.array([-0.5, -0.1, 0.0, 0.1, 0.5, np.nan])

    def trim(method, **b):
        return _trimmed_returns(r.copy(), {"method": method, "bounds": {"type": "absolute", **b}})
    nan = np.nan
    np.testing.assert_array_equal(trim("truncate", hi=0.1), [-0.5, -0.1, 0.0, 0.1, nan, nan])
    np.testing.assert_array_equal(trim("truncate", lo=-0.1), [nan, -0.1, 0.0, 0.1, 0.5, nan])
    np.testing.assert_array_equal(trim("truncate", lo=-0.1, hi=0.1), [nan, -0.1, 0.0, 0.1, nan, nan])
    np.testing.assert_array_equal(trim("winsorise", hi=0.1), [-0.5, -0.1, 0.0, 0.1, 0.1, nan])
    np.testing.assert_array_equal(trim("winsorise", lo=-0.1), [-0.1, -0.1, 0.0, 0.1, 0.5, nan])
    np.testing.assert_array_equal(_trimmed_returns(r.copy(), {"method": "none"}), r)
    with pytest.raises(ValueError, match="unresolved percentile"):
        _trimmed_returns(r.copy(), _pct_trim("truncate", 0.9))


def test_overlap_cohort_path_at_h1_equals_engine_under_winsorise_percentile():
    """The overlap cohort path (selections + trimmed lookup + leg returns), driven at H=1 without the
    short-circuit, reproduces the engine under a winsorise percentile trim. An unclipped held-month
    lookup would diverge from the engine here."""
    from agents.quant.library.overlap import _leg_return_at, _trimmed_returns
    panel = _synth_panel(seed=21)
    rb = _rulebook(_pct_trim("winsorise", 0.9))
    trim_abs, realised = resolve_trim_rule(panel, rb)
    assert realised is not None
    selections = extract_monthly_selections(panel, _rulebook(trim_abs))
    lookup = dict(zip(zip(panel["cusip"].values, panel["date"].astype("datetime64[ns]").values),
                      _trimmed_returns(panel["ret"].to_numpy(dtype=float, copy=True), trim_abs)))
    got = {}
    for t, sel in selections.items():
        m = t + pd.offsets.MonthEnd(1)
        long_r, _ = _leg_return_at(sel["long"], m, lookup, "equal")
        short_r, _ = _leg_return_at(sel["short"], m, lookup, "equal")
        if long_r is not None and short_r is not None:
            got[m] = long_r - short_r
    engine = run_characteristic_sort(panel, rb)["monthly_returns"].set_index("date")["strategy_ret"]
    got_s = pd.Series(got).sort_index()
    assert len(got_s) == len(engine) > 0
    np.testing.assert_allclose(got_s.to_numpy(), engine.reindex(got_s.index).to_numpy(), rtol=0, atol=1e-12)


def _synth_panel(seed: int, n_bonds: int = 20, n_months: int = 18) -> pd.DataFrame:
    """Fat-tailed returns (t, 3 df) so a 90th-percentile trim bites in held months."""
    rng = np.random.default_rng(seed)
    months = pd.date_range("2012-01-31", periods=n_months, freq="ME")
    rows = []
    for i in range(n_bonds):
        base = rng.uniform(0, 1)
        for d in months:
            rows.append({"cusip": f"B{i:02d}", "date": d,
                         "ret": float(0.02 * rng.standard_t(3)), "size": 1.0,
                         "score": base + float(rng.normal(scale=0.1))})
    return pd.DataFrame(rows)
