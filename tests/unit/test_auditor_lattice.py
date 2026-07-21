"""Stage 4 — cell runner (SEAM 1) and the 2^k lattice (§3.8-3.9)."""

from __future__ import annotations

import dataclasses

import pytest

from agents.auditor.checks.cell_runner import CellRunError, run_cell
from agents.auditor.checks.lattice import (
    LatticeError,
    build_lattice_configs,
    build_run_config,
    run_lattice,
)
from agents.auditor.checks.preflight import derive_scope
from agents.auditor.data.synthetic_panel import SyntheticSpec, make_clean_maximal_panel
from agents.auditor.schemas.toggle import TOGGLE_IDS
from agents.quant.library.run_config import corrected, uncorrected
from agents.quant.library.views import view

from _auditor_fixtures import all_runnable_facts, make_score_strategy


# --------------------------------------------------------------------------
# build_run_config / config enumeration
# --------------------------------------------------------------------------

def test_all_off_config_equals_uncorrected_endpoint():
    states = {t: "OFF" for t in TOGGLE_IDS}
    assert build_run_config(states) == uncorrected()


def test_all_on_config_equals_corrected_endpoint():
    states = {t: "ON" for t in TOGGLE_IDS}
    assert build_run_config(states) == corrected()


def test_lattice_has_2_to_the_k_configs():
    configs = build_lattice_configs(TOGGLE_IDS, {})
    assert len(configs) == 2 ** len(TOGGLE_IDS) == 32
    on_sets = {oc[0] for oc in configs}
    assert len(on_sets) == 32  # all distinct


def test_partial_lattice_holds_non_runnable_at_fixed_state():
    runnable = ("meas_err", "stale_price", "lib_gap", "lab_trim")
    configs = build_lattice_configs(runnable, {"survivorship": "OFF"})
    assert len(configs) == 2 ** 4
    # survivorship held OFF => include_terminal_rows False in every cell
    assert all(rc.panel_view.include_terminal_rows is False for _, rc in configs)


def test_refused_toggle_without_fixed_state_raises():
    with pytest.raises(LatticeError, match="REFUSED"):
        build_lattice_configs(("meas_err",), {})  # survivorship etc. unheld


# --------------------------------------------------------------------------
# Panel-view reuse (§3.9)
# --------------------------------------------------------------------------

def test_distinct_panel_views_fewer_than_cells():
    # 5 toggles => 32 cells, but only the 3 panel toggles change the panel view,
    # so at most 2^3 = 8 distinct panel_view_hashes.
    configs = build_lattice_configs(TOGGLE_IDS, {})
    distinct = {rc.panel_view_hash() for _, rc in configs}
    assert len(distinct) == 8


# --------------------------------------------------------------------------
# run_lattice end-to-end on the clean synthetic panel
# --------------------------------------------------------------------------

def _clean():
    return make_clean_maximal_panel(SyntheticSpec(n_bonds=40, n_months=48, seed=1))


def test_run_lattice_produces_32_cells():
    panel, signals = _clean()
    strat = make_score_strategy()
    facts = all_runnable_facts()
    pf = derive_scope("synthetic", facts)
    lat = run_lattice(strat, pf.runnable_toggles, pf.fixed_states, panel, signals=signals)
    assert lat.k == 5
    assert len(lat.cells) == 32
    # the empty ON-set cell is the as-published endpoint; the full set is corrected
    assert lat.cell_for(frozenset()).run_config == uncorrected()
    assert lat.cell_for(frozenset(TOGGLE_IDS)).run_config == corrected()


def test_cells_have_returns_and_metrics():
    panel, signals = _clean()
    lat = run_lattice(
        make_score_strategy(), TOGGLE_IDS, {}, panel, signals=signals
    )
    cell = lat.cell_for(frozenset())
    assert len(cell.returns) > 0
    # planted alpha>0 with score=q => long-short mean return is positive
    assert cell.metrics_native.average > 0


# --------------------------------------------------------------------------
# Purity + toggle threading
# --------------------------------------------------------------------------

def test_run_cell_is_pure():
    panel, signals = _clean()
    strat = make_score_strategy()
    rc = uncorrected()
    view_panel = view(panel, rc, signals=signals)
    c1 = run_cell(strat, rc, frozenset(), view_panel)
    c2 = run_cell(strat, rc, frozenset(), view_panel)
    assert c1.return_hash == c2.return_hash
    assert c1.metric_hash == c2.metric_hash
    assert c1.n_bonds_hash == c2.n_bonds_hash


def test_meas_err_toggle_changes_output_when_families_differ():
    # Perturb the corr family so meas_err ON (corr) differs from OFF (raw).
    panel, signals = _clean()
    panel = panel.copy()
    panel["ret_corr"] = panel["ret_corr"] + 0.02  # corrected family differs
    strat = make_score_strategy()

    off = view(panel, uncorrected(), signals=signals)
    on_cfg = dataclasses.replace(
        uncorrected(),
        panel_view=dataclasses.replace(uncorrected().panel_view, price_family="corr"),
    )
    on = view(panel, on_cfg, signals=signals)
    c_off = run_cell(strat, uncorrected(), frozenset(), off)
    c_on = run_cell(strat, on_cfg, frozenset({"meas_err"}), on)
    assert c_off.return_hash != c_on.return_hash


def test_lib_gap_toggle_threads_into_the_rulebook():
    # A time-varying score makes ranking_score at lag 0 differ from lag 1, so the
    # constructed portfolios — and thus the return series — differ. Proves the
    # construction toggle actually reaches the engine rulebook (SEAM 1).
    panel, signals = make_clean_maximal_panel(
        SyntheticSpec(n_bonds=40, n_months=48, seed=2, time_varying_score=True)
    )
    strat = make_score_strategy()
    view_panel = view(panel, uncorrected(), signals=signals)  # panel identical for both lags

    off_cfg = uncorrected()  # signal_lag 0
    on_cfg = dataclasses.replace(
        uncorrected(),
        construction=dataclasses.replace(uncorrected().construction, signal_lag=1),
    )
    c_off = run_cell(strat, off_cfg, frozenset(), view_panel)
    c_on = run_cell(strat, on_cfg, frozenset({"lib_gap"}), view_panel)
    assert c_off.return_hash != c_on.return_hash


# --------------------------------------------------------------------------
# Refusal short-circuit
# --------------------------------------------------------------------------

def test_refused_strategy_is_not_run():
    panel, signals = _clean()
    strat = make_score_strategy()
    refused = dataclasses.replace(strat, refusals=("dummy",))  # force refused=True
    assert refused.refused
    with pytest.raises(LatticeError, match="refused"):
        run_lattice(refused, TOGGLE_IDS, {}, panel, signals=signals)


def test_cell_runner_rejects_refused_strategy_directly():
    panel, signals = _clean()
    strat = make_score_strategy()
    refused = dataclasses.replace(strat, refusals=("dummy",))
    view_panel = view(panel, uncorrected(), signals=signals)
    with pytest.raises(CellRunError, match="refused"):
        run_cell(refused, uncorrected(), frozenset(), view_panel)
