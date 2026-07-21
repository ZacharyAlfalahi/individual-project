"""Stage 5 — common support, the minimum-support gate, and metric vectors (§4.2)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from agents.auditor.checks.lattice import run_lattice
from agents.auditor.checks.metrics import metric_set_on
from agents.auditor.checks.shapley import shapley_result
from agents.auditor.checks.support import (
    common_support,
    native_metric_vector,
    primary_metric_vector,
    support_info,
    valid_months,
)
from agents.auditor.checks.algebra import saturated_basis
from agents.auditor.data.synthetic_panel import SyntheticSpec, make_clean_maximal_panel
from agents.auditor.hashing import hash_metrics, hash_series
from agents.auditor.schemas.lattice_types import METRIC_NAMES, CellReturns
from agents.auditor.schemas.toggle import TOGGLE_IDS
from agents.auditor.thresholds import SupportGate

from _auditor_fixtures import make_score_strategy


def _cell(on_set, index, values):
    idx = pd.DatetimeIndex(index)
    returns = pd.Series(values, index=idx, dtype=float)
    n_bonds = pd.Series([10] * len(idx), index=idx, dtype=float)
    ms = metric_set_on(returns)
    return CellReturns(
        on_set=frozenset(on_set),
        run_config=None,
        returns=returns,
        n_bonds=n_bonds,
        metrics_native=ms,
        run_config_hash="rc",
        panel_view_hash="pv",
        return_hash=hash_series(returns),
        n_bonds_hash=hash_series(n_bonds),
        metric_hash=hash_metrics(ms.as_metric_dict(), METRIC_NAMES),
    )


def _months(start, n):
    return pd.date_range(start, periods=n, freq="ME")


# --------------------------------------------------------------------------
# valid_months + common_support
# --------------------------------------------------------------------------

def test_valid_months_excludes_nan():
    idx = _months("2005-01-31", 4)
    s = pd.Series([0.1, np.nan, 0.2, np.nan], index=idx)
    assert list(valid_months(s)) == [idx[0], idx[2]]


def test_common_support_is_the_intersection():
    m = _months("2005-01-31", 5)
    # cell A valid months 0-3; cell B valid months 1-4 => common 1-3
    a = _cell(set(), m[:4], [0.1, 0.1, 0.1, 0.1])
    b = _cell({"meas_err"}, m[1:5], [0.2, 0.2, 0.2, 0.2])
    common = common_support([a, b])
    assert list(common) == list(m[1:4])


# --------------------------------------------------------------------------
# support_info + gate
# --------------------------------------------------------------------------

def test_gate_passes_with_ample_support():
    m = _months("2005-01-31", 60)
    cells = [_cell(set(), m, [0.01] * 60), _cell({"meas_err"}, m, [0.02] * 60)]
    info = support_info(cells, SupportGate(min_common_months=24, min_common_fraction_of_reference=0.5))
    assert info.t_common == 60
    assert info.gate_passed and not info.downgraded
    assert info.common_fraction == 1.0


def test_gate_fails_and_downgrades_when_support_short():
    m = _months("2005-01-31", 60)
    a = _cell(set(), m, [0.01] * 60)
    # b valid only in last 10 months => common support = 10 months
    vals = [np.nan] * 50 + [0.02] * 10
    b = _cell({"meas_err"}, m, vals)
    info = support_info([a, b], SupportGate(min_common_months=24, min_common_fraction_of_reference=0.5))
    assert info.t_common == 10
    assert not info.gate_passed and info.downgraded


def test_fraction_uses_longest_native_as_reference():
    m = _months("2005-01-31", 40)
    a = _cell(set(), m, [0.01] * 40)                        # native 40
    b = _cell({"meas_err"}, m, [np.nan] * 20 + [0.02] * 20)  # native 20, common 20
    info = support_info([a, b], SupportGate(min_common_months=10, min_common_fraction_of_reference=0.4))
    assert info.reference_native_months == 40
    assert info.t_common == 20
    assert info.common_fraction == 0.5


# --------------------------------------------------------------------------
# metric vectors + end-to-end efficiency
# --------------------------------------------------------------------------

def test_primary_and_native_metric_vectors_keyed_by_on_set():
    m = _months("2005-01-31", 30)
    a = _cell(set(), m, [0.01] * 30)
    b = _cell({"meas_err"}, m, [0.03] * 30)
    common = common_support([a, b])
    prim = primary_metric_vector([a, b], common, "average")
    nat = native_metric_vector([a, b], "average")
    assert set(prim) == {frozenset(), frozenset({"meas_err"})}
    assert np.isclose(prim[frozenset()], 0.01)
    assert np.isclose(prim[frozenset({"meas_err"})], 0.03)
    assert set(nat) == set(prim)


def test_end_to_end_common_support_decomposition_is_efficient():
    # Run a real 32-cell lattice, decompose the common-support metric vector, and
    # confirm Shapley efficiency holds on real engine output.
    panel, signals = make_clean_maximal_panel(SyntheticSpec(n_bonds=40, n_months=48, seed=3))
    lat = run_lattice(make_score_strategy(), TOGGLE_IDS, {}, panel, signals=signals)
    common = common_support(lat.cells)
    assert len(common) > 0
    Y = primary_metric_vector(lat.cells, common, "average")
    assert len(Y) == 32
    # shapley_result asserts efficiency internally; a low denominator threshold
    # keeps shares defined even though the clean-panel gap is ~0.
    res = shapley_result(Y, TOGGLE_IDS, percentage_denominator_min=0.0)
    assert abs(res.efficiency_residual) < 1e-9
    # clean panel => every DOE effect ~ 0 (no toggle does anything)
    basis = saturated_basis(Y, TOGGLE_IDS)
    for T, e in basis.doe.items():
        if T:
            assert abs(e) < 1e-9
