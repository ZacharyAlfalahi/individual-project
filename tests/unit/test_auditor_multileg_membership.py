"""Multi-leg (equal_average) invariance membership proxy.

The equal_average combiner used to drop n_bonds, making the invariance no-op
membership check silently vacuous for multi-leg strategies (e.g. BBW). The runner
now carries a combined n_bonds, and the auditor refuses to certify a no-op when the
proxy is unavailable.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from agents.auditor.checks.cell_runner import run_cell
from agents.auditor.checks.invariance import invariance_test
from agents.auditor.checks.lattice import run_lattice
from agents.auditor.data.synthetic_panel import SyntheticSpec, make_clean_maximal_panel, score_strategy
from agents.auditor.hashing import hash_metrics, hash_series
from agents.auditor.schemas.lattice_types import METRIC_NAMES, CellReturns, LatticeResult, MetricSet
from agents.auditor.schemas.toggle import TOGGLE_IDS
from agents.quant.config import Binding, Evidence, Inherited, build_quant_config
from agents.quant.library.run_config import uncorrected
from agents.quant.library.views import view
from agents.librarian.adapter.result import AdaptResult, CombinerInstruction, LegCall


def _two_leg_equal_average(label="ea2") -> AdaptResult:
    score = Binding("score", "BOUND", Evidence(column="score"))

    def cfg(sid):
        return build_quant_config(
            sid, score,
            groups=Inherited(5, "DESIGN", Evidence(note="g")),
            weighting=Inherited("equal", "DESIGN", Evidence(note="w")),
            signal_lag=Inherited(0, "DESIGN", Evidence(note="l")),
            long_group=Inherited(4, "DESIGN", Evidence(note="lg")),
            short_group=Inherited(0, "DESIGN", Evidence(note="sg")),
            holding_period=Inherited(1, "DESIGN", Evidence(note="h")),
        )

    legs = (
        LegCall(f"{label}::0", {}, cfg(f"{label}::0")),
        LegCall(f"{label}::1", {}, cfg(f"{label}::1")),
    )
    return AdaptResult(
        strategy_label=label, leg_calls=legs,
        combiner=CombinerInstruction("equal_average", "available"),
    )


def _clean():
    return make_clean_maximal_panel(SyntheticSpec(n_bonds=40, n_months=36, seed=1))


# --------------------------------------------------------------------------
# The combiner now carries a real n_bonds
# --------------------------------------------------------------------------

def test_equal_average_cell_carries_nonempty_n_bonds():
    panel, signals = _clean()
    lat = run_lattice(_two_leg_equal_average(), TOGGLE_IDS, {}, panel, signals=signals)
    cell = lat.cell_for(frozenset())
    assert len(cell.returns) > 0
    assert len(cell.n_bonds) == len(cell.returns)   # proxy present, not empty
    assert cell.n_bonds.notna().any()


def test_equal_average_n_bonds_is_sum_of_legs():
    panel, signals = _clean()
    vp = view(panel, uncorrected(), signals=signals)
    single = run_cell(score_strategy(), uncorrected(), frozenset(), vp)
    two = run_cell(_two_leg_equal_average(), uncorrected(), frozenset(), vp)
    # identical legs => combined n_bonds is twice the single-leg count each month
    aligned_single = single.n_bonds.reindex(two.n_bonds.index)
    assert np.allclose(two.n_bonds.to_numpy(), 2.0 * aligned_single.to_numpy(), equal_nan=True)


def test_multileg_invariance_membership_is_verified():
    panel, signals = _clean()
    lat = run_lattice(_two_leg_equal_average(), TOGGLE_IDS, {}, panel, signals=signals)
    res = invariance_test(lat, "meas_err")  # clean panel => inert
    assert res.membership_verified          # proxy available AND identical
    assert res.is_no_op


# --------------------------------------------------------------------------
# The guard: an unavailable proxy conservatively blocks no-op certification
# --------------------------------------------------------------------------

def _cell(on_set, index, returns, n_bonds):
    idx = pd.DatetimeIndex(index)
    r = pd.Series(returns, index=idx, dtype=float)
    nb = pd.Series(n_bonds, index=pd.DatetimeIndex([]) if len(n_bonds) == 0 else idx, dtype=float)
    ms = MetricSet.from_summary(
        {"n_months": len(idx), "months_per_year": 12, "nw_lags_used": 0,
         "average": float(np.nanmean(returns)) if len(returns) else float("nan"),
         "annualised_average": 0.0, "bumpiness": 0.0, "sharpe": 0.0, "t_stat": 0.0,
         "first_date": None, "last_date": None}
    )
    return CellReturns(
        on_set=frozenset(on_set), run_config=None, returns=r, n_bonds=nb,
        metrics_native=ms, run_config_hash=("OFF" if not on_set else "ON"),
        panel_view_hash="pv", return_hash=hash_series(r),
        n_bonds_hash=hash_series(nb), metric_hash=hash_metrics(ms.as_metric_dict(), METRIC_NAMES),
    )


def test_no_op_not_certified_when_membership_proxy_missing():
    idx = pd.date_range("2005-01-31", periods=12, freq="ME")
    rets = [0.01] * 12
    # both cells: identical returns/metrics, DIFFERENT config, but EMPTY n_bonds
    off = _cell(set(), idx, rets, [])
    on = _cell({"meas_err"}, idx, rets, [])
    lat = LatticeResult("s", ("meas_err",), {}, (off, on))
    res = invariance_test(lat, "meas_err")
    assert res.returns_identical and res.metrics_identical and res.config_hashes_differ
    assert not res.membership_verified
    assert not res.is_no_op
    assert "membership proxy" in res.note
