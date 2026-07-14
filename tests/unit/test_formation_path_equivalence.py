"""
Pin the engine's TWO formation paths against silent divergence.

The engine writes the eligibility mask + min_bonds gate INLINE in two functions:
  * ``_month_step``                (MAIN path,    via ``run_characteristic_sort``)
  * ``extract_monthly_selections`` (OVERLAP path, via ``overlap.run_with_holding_period`` at H>1)

The existing ``test_overlap.py::test_h1_reduces_to_engine_strictly`` exercises overlap's
**H=1 short-circuit**, which calls ``run_characteristic_sort`` directly and therefore does
**not** touch ``extract_monthly_selections`` at all. So nothing currently catches a future
edit to one inline region that does not reach the other.

These tests close that guard-rail gap. ``_overlap_formation_h1`` drives the OVERLAP formation
path (``extract_monthly_selections`` + ``_leg_return_at``) at holding_period=1 **without** the
short-circuit, and asserts it reproduces the MAIN path element-wise — so divergence between the
two inline regions fails here.

Context: ``docs/quant/registers/assumptions_ledger_v2.md`` §3 (two-path divergence) + the anchor-validation
report. The one KNOWN current divergence — the ``by_size`` non-positive-size guard, present in
``_form_legs`` (MAIN) but not at formation in ``extract_monthly_selections`` (OVERLAP) — is
pinned by ``test_bysize_zero_formation_size_divergence_is_pinned``. It is DORMANT: par (``size``)
is > 0 on all development panels (empirically min=1.0, zero non-positive), so it cannot fire on
real data; it could only bite a future ``by_size`` multi-month-hold strategy on a pathological
panel. The engine is frozen — these are additive tests, no engine change.
"""

import numpy as np
import pandas as pd
import pytest

from agents.quant.library.characteristic_sort import (
    _apply_defaults,
    extract_monthly_selections,
    run_characteristic_sort,
)
from agents.quant.library.overlap import _leg_return_at


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _overlap_formation_h1(panel: pd.DataFrame, rulebook: dict) -> pd.DataFrame:
    """Run the OVERLAP formation path at holding_period=1 WITHOUT overlap's H=1
    short-circuit, so the comparison actually exercises ``extract_monthly_selections``.

    Mirrors ``overlap.run_with_holding_period``'s H>1 cohort loop with H=1: form
    selections, then realise each cohort at t+1 via ``_leg_return_at``. At H=1 every
    formation-leg member survives (eligibility required ``next_ret`` present), so this
    must reproduce ``run_characteristic_sort`` exactly.
    """
    settings = _apply_defaults(rulebook)
    weighting = settings["weighting"]
    selections = extract_monthly_selections(panel, rulebook)
    lookup = dict(
        zip(zip(panel["cusip"].values, panel["date"].values), panel["ret"].values)
    )
    rows = []
    for formation_t, sel in selections.items():
        m = formation_t + pd.offsets.MonthEnd(1)
        long_r, n_long = _leg_return_at(sel["long"], m, lookup, weighting)
        short_r, n_short = _leg_return_at(sel["short"], m, lookup, weighting)
        if long_r is None or short_r is None:
            continue
        rows.append(
            {
                "date": m,
                "strategy_ret": long_r - short_r,
                "long_ret": long_r,
                "short_ret": short_r,
                "n_bonds": n_long + n_short,
            }
        )
    if not rows:
        return pd.DataFrame(
            {
                "date": pd.Series(dtype="datetime64[ns]"),
                "strategy_ret": pd.Series(dtype=float),
                "long_ret": pd.Series(dtype=float),
                "short_ret": pd.Series(dtype=float),
                "n_bonds": pd.Series(dtype=int),
            }
        )
    return pd.DataFrame(rows).sort_values("date").reset_index(drop=True)


def _synth_panel(seed: int, n_bonds: int = 12, n_months: int = 24) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    months = pd.date_range("2010-01-31", periods=n_months, freq="ME")
    rows = []
    for i in range(n_bonds):
        base = rng.uniform(0, 1)
        for d in months:
            rows.append(
                {
                    "cusip": f"C{i:02d}",
                    "date": d,
                    "ret": float(rng.normal(scale=0.02)),
                    "size": float(rng.uniform(50, 200)),
                    "score": base + float(rng.normal(scale=0.1)),
                }
            )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# The pin: OVERLAP formation path reduces to the MAIN path at H=1
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("weighting", ["equal", "by_size"])
def test_overlap_formation_path_reduces_to_engine(weighting):
    """extract_monthly_selections + _leg_return_at at H=1 == run_characteristic_sort,
    element-wise. Pins the OVERLAP formation path against the MAIN path. Unlike the
    existing short-circuit test, this genuinely exercises extract_monthly_selections,
    so an edit to either inline eligibility/min_bonds region that does not reach the
    other will break this test."""
    panel = _synth_panel(seed=123)
    rulebook = {"score": "score", "groups": 5, "weighting": weighting, "min_bonds": 5}

    engine_mr = run_characteristic_sort(panel, rulebook)["monthly_returns"]
    overlap_mr = _overlap_formation_h1(panel, rulebook)

    # Not vacuous: both paths must actually produce rows.
    assert len(engine_mr) > 0
    pd.testing.assert_frame_equal(
        overlap_mr[list(engine_mr.columns)],
        engine_mr,
        check_exact=False,
        atol=1e-12,
        rtol=0,
        check_dtype=False,
    )


def test_extract_monthly_selections_is_actually_exercised():
    """Guard against a vacuous pin: confirm the equivalence test's path really calls
    extract_monthly_selections and gets a non-empty selection set."""
    panel = _synth_panel(seed=123)
    rulebook = {"score": "score", "groups": 5, "weighting": "equal", "min_bonds": 5}
    sels = extract_monthly_selections(panel, rulebook)
    assert len(sels) > 0
    any_key = next(iter(sels))
    assert set(sels[any_key].keys()) == {"long", "short"}
    assert list(sels[any_key]["long"].columns) == ["cusip", "size"]


# ---------------------------------------------------------------------------
# Pin the ONE known current divergence (by_size non-positive formation size)
# ---------------------------------------------------------------------------

def test_bysize_zero_formation_size_divergence_is_pinned():
    """DOCUMENTED, DORMANT divergence. Under ``by_size``, when a formation leg's
    ``sum(size) <= 0``:
      * MAIN (`_form_legs`, cs:309-313) skips the stripe -> the month is dropped.
      * OVERLAP (`extract_monthly_selections`) has NO formation-stage size guard,
        so it still emits the selection (the guard lives later, differently, in
        `_leg_return_at` over survivors).
    This asserts that current behaviour so any future change is caught. It is DORMANT
    on real data: par (`size`) is > 0 everywhere in the development panels, so this
    never fires in production — it could only reach a future by_size holding-period
    strategy on a pathological panel.
    """
    months = pd.date_range("2010-01-31", periods=2, freq="ME")
    form, real = months[0], months[1]
    rows = []
    for i in range(10):
        cusip, score = f"C{i:02d}", float(i)  # C09 highest score
        # Long group (groups=5 → top quintile = 2 highest scores, C08/C09) gets size 0.
        form_size = 0.0 if i >= 8 else 100.0
        rows.append({"cusip": cusip, "date": form, "ret": 0.0, "size": form_size, "score": score})
        # Realisation month: rets present (so next_ret exists at formation); score NaN
        # suppresses a second formation.
        rows.append({"cusip": cusip, "date": real, "ret": 0.01, "size": 100.0, "score": np.nan})
    panel = pd.DataFrame(rows)
    rb = {"score": "score", "groups": 5, "weighting": "by_size", "min_bonds": 5,
          "long_group": 4, "short_group": 0}

    engine_mr = run_characteristic_sort(panel, rb)["monthly_returns"]
    sels = extract_monthly_selections(panel, rb)

    # MAIN drops the zero-size-leg month; OVERLAP formation still emits it.
    assert len(engine_mr) == 0, "MAIN path should skip the zero-size by_size leg"
    assert set(sels.keys()) == {form}, "OVERLAP formation emits the leg MAIN skipped"

    # Contrast: with positive sizes the two paths AGREE (the divergence is specific to
    # size<=0), confirming it is an edge case, not a general disagreement.
    rows_pos = [{**r, "size": 100.0} for r in rows]
    panel_pos = pd.DataFrame(rows_pos)
    engine_pos = run_characteristic_sort(panel_pos, rb)["monthly_returns"]
    sels_pos = extract_monthly_selections(panel_pos, rb)
    assert len(engine_pos) == 1 and set(sels_pos.keys()) == {form}
