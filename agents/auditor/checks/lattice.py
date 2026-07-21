"""
lattice.py — build and run the 2^k lattice (§3.8-3.9).

Given the pre-flight verdict (which toggles are runnable, and the fixed states of
any held non-runnable toggles), this module enumerates the 2^k combinations over
the runnable toggles, builds a `RunConfig` for each, and runs each as a cell.

"One panel, not 32" (§3.9): cells that differ only in construction toggles share
one materialised panel, so `view()` is called once per distinct `panel_view_hash`
and the result is reused. The RunConfig->field mapping is the single source of
truth in `schemas.toggle.TOGGLE_AXES`; only lib_gap's ON/OFF lag is parameterised
here (the mom6-skip subtlety, §3.5 / O-A2), defaulting to OFF=0, ON=1.
"""

from __future__ import annotations

from itertools import combinations
from typing import Mapping, Sequence

import pandas as pd

from agents.quant.library.run_config import (
    ConstructionConfig,
    EvaluationConfig,
    PanelViewConfig,
    RunConfig,
)
from agents.quant.library.views import view

from ..schemas.lattice_types import CellReturns, LatticeResult
from ..schemas.toggle import TOGGLE_AXES, TOGGLE_IDS, ToggleId, ToggleState
from .cell_runner import run_cell


class LatticeError(RuntimeError):
    """The lattice cannot be built or run (refused strategy, or a toggle that is
    neither runnable nor held at a fixed state)."""


def build_run_config(
    states: Mapping[ToggleId, ToggleState],
    *,
    lib_gap_lags: tuple[int, int] = (0, 1),
) -> RunConfig:
    """Assemble a RunConfig from a full OFF/ON state for all five toggles, via the
    canonical TOGGLE_AXES mapping. `lib_gap_lags` = (off_lag, on_lag) sets the
    signal_lag values for lib_gap OFF/ON."""
    missing = set(TOGGLE_IDS) - set(states)
    if missing:
        raise LatticeError(f"build_run_config needs all five toggle states; missing {sorted(missing)}")

    panel_vals: dict = {}
    constr_vals: dict = {}
    for t in TOGGLE_IDS:
        state = states[t]
        if state not in ("OFF", "ON"):
            raise LatticeError(f"toggle {t} has invalid state {state!r}")
        axis = TOGGLE_AXES[t]
        if t == "lib_gap":
            value: object = lib_gap_lags[1] if state == "ON" else lib_gap_lags[0]
        else:
            value = axis.on if state == "ON" else axis.off
        if axis.block == "panel_view":
            panel_vals[axis.field] = value
        else:
            constr_vals[axis.field] = value

    return RunConfig(
        panel_view=PanelViewConfig(**panel_vals),
        construction=ConstructionConfig(**constr_vals),
        evaluation=EvaluationConfig(),
    )


def build_lattice_configs(
    runnable_toggles: Sequence[ToggleId],
    fixed_states: Mapping[ToggleId, ToggleState],
    *,
    lib_gap_lags: tuple[int, int] = (0, 1),
) -> list[tuple[frozenset, RunConfig]]:
    """Enumerate the 2^k (on_set, RunConfig) pairs over the runnable toggles.
    Non-runnable toggles are held at their fixed_state; a toggle that is neither
    runnable nor held is a REFUSED audit and raises."""
    runnable = tuple(t for t in TOGGLE_IDS if t in set(runnable_toggles))
    non_runnable = [t for t in TOGGLE_IDS if t not in set(runnable)]
    for t in non_runnable:
        if t not in fixed_states:
            raise LatticeError(
                f"toggle {t!r} is neither runnable nor held at a fixed_state — "
                "this is a REFUSED audit and has no lattice (§3.4)"
            )

    out: list[tuple[frozenset, RunConfig]] = []
    k = len(runnable)
    for r in range(k + 1):
        for on_tuple in combinations(runnable, r):
            on_set = frozenset(on_tuple)
            states: dict[ToggleId, ToggleState] = {}
            for t in TOGGLE_IDS:
                if t in runnable:
                    states[t] = "ON" if t in on_set else "OFF"
                else:
                    states[t] = fixed_states[t]
            out.append((on_set, build_run_config(states, lib_gap_lags=lib_gap_lags)))
    return out


def run_lattice(
    strategy,
    runnable_toggles: Sequence[ToggleId],
    fixed_states: Mapping[ToggleId, ToggleState],
    maximal_panel: pd.DataFrame,
    *,
    signals: pd.DataFrame | None = None,
    lib_gap_lags: tuple[int, int] = (0, 1),
    safe_rate: pd.DataFrame | None = None,
    benchmark: pd.DataFrame | None = None,
) -> LatticeResult:
    """Run every cell of the lattice, reusing materialised panels by
    panel_view_hash (§3.9). Returns a LatticeResult."""
    if getattr(strategy, "refused", False):
        raise LatticeError(
            f"strategy {getattr(strategy, 'strategy_label', '?')!r} is refused; "
            "no lattice is run for a refused strategy"
        )

    configs = build_lattice_configs(
        runnable_toggles, fixed_states, lib_gap_lags=lib_gap_lags
    )

    view_cache: dict[str, pd.DataFrame] = {}
    cells: list[CellReturns] = []
    for on_set, rc in configs:
        pvh = rc.panel_view_hash()
        if pvh not in view_cache:
            view_cache[pvh] = view(maximal_panel, rc, signals=signals)
        panel = view_cache[pvh]
        cells.append(
            run_cell(
                strategy, rc, on_set, panel,
                safe_rate=safe_rate, benchmark=benchmark,
            )
        )

    runnable = tuple(t for t in TOGGLE_IDS if t in set(runnable_toggles))
    return LatticeResult(
        strategy_label=getattr(strategy, "strategy_label", "?"),
        runnable_toggles=runnable,
        fixed_states=dict(fixed_states),
        cells=tuple(cells),
    )
