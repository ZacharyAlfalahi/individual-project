"""
cell_runner.py — run ONE lattice cell (SEAM 1).

A cell is one strategy executed under one `RunConfig`. The subtlety this module
exists to centralise: `views.view()` absorbs only the THREE panel toggles
(meas_err -> price_family, stale_price -> stale_mask, survivorship ->
include_terminal_rows). The TWO construction toggles are applied at the engine
rulebook level, not the panel:

  * lib_gap  -> signal_lag  (QuantConfig.signal_lag -> to_rulebook -> _build_lagged_panel)
  * lab_trim -> trim_rule   (QuantConfig.trim_rule  -> to_rulebook -> _apply_trim_rule)

So a cell run is: view() the panel from `run_config.panel_view`, then OVERRIDE the
construction toggles onto every leg's `QuantConfig` (before `run_strategy` strips
it to a rulebook), then run. The lab_trim polarity trap (§3.3 condition 2) is
handled here: lab_trim OFF is the paper's PUBLISHED trim (whatever the base config
carries), NOT "none" — only lab_trim ON forces `TrimRule(method="none")`.

The existing anchor build scripts duplicate signal_lag into both the RunConfig and
a hand-written rulebook with nothing keeping them in sync; this module derives the
rulebook toggle FROM the RunConfig, so the two can never diverge.
"""

from __future__ import annotations

import dataclasses

import pandas as pd

from agents.quant.config import (
    Evidence,
    Inherited,
    QuantConfig,
    TrimRule,
    run_strategy,
)
from agents.quant.config.runner import StrategyResult
from agents.quant.library.run_config import RunConfig

from ..hashing import hash_metrics, hash_series
from ..schemas.lattice_types import METRIC_NAMES, CellReturns, MetricSet
from ..schemas.toggle import ToggleId

# Import type only for annotations; the runtime object arrives from the caller.
try:  # pragma: no cover - import guard
    from agents.librarian.adapter.result import AdaptResult
except Exception:  # pragma: no cover
    AdaptResult = object  # type: ignore


class CellRunError(RuntimeError):
    """A cell could not be executed (e.g. a refused strategy reached the runner)."""


def _override_construction(result: "AdaptResult", run_config: RunConfig) -> "AdaptResult":
    """Return a copy of `result` with every leg's QuantConfig carrying the cell's
    construction-toggle states. signal_lag is set absolutely from the RunConfig;
    trim_rule is forced to `none` only for lab_trim ON — lab_trim OFF keeps the
    paper's published trim."""
    lag = run_config.construction.signal_lag
    expost_trim = run_config.construction.expost_trim

    new_legs = []
    for lc in result.leg_calls:
        cfg = lc.result
        if not isinstance(cfg, QuantConfig):
            raise CellRunError(
                f"leg {lc.strategy_id!r} carries a non-runnable result "
                f"({type(cfg).__name__}); a refused strategy must not reach the cell runner"
            )
        new_lag = Inherited(
            lag, "DESIGN",
            Evidence(note=f"auditor lattice: lib_gap -> signal_lag={lag}"),
        )
        if expost_trim == "none":
            new_trim: Inherited = Inherited(
                TrimRule(method="none"), "DESIGN",
                Evidence(note="auditor lattice: lab_trim ON (no ex-post trim)"),
            )
        else:
            # lab_trim OFF: the paper's PUBLISHED trim — keep the base config's
            # trim_rule unchanged. NEVER silently substitute 'none' (§3.3 cond. 2).
            new_trim = cfg.trim_rule
        new_cfg = dataclasses.replace(cfg, signal_lag=new_lag, trim_rule=new_trim)
        new_legs.append(dataclasses.replace(lc, result=new_cfg))

    return dataclasses.replace(result, leg_calls=tuple(new_legs))


def _series_from_monthly(monthly: pd.DataFrame, column: str) -> pd.Series:
    """A date-indexed series from a monthly_returns column, or an empty series if
    the column is absent (e.g. n_bonds on a combined multi-leg envelope)."""
    if column not in monthly.columns or len(monthly) == 0:
        return pd.Series([], dtype=float, index=pd.DatetimeIndex([]))
    return pd.Series(
        monthly[column].to_numpy(),
        index=pd.DatetimeIndex(monthly["date"].to_numpy()),
    )


def run_cell(
    strategy: "AdaptResult",
    run_config: RunConfig,
    on_set: frozenset[ToggleId],
    panel: pd.DataFrame,
    *,
    safe_rate: pd.DataFrame | None = None,
    benchmark: pd.DataFrame | None = None,
) -> CellReturns:
    """Execute one cell. `panel` is the ALREADY-view()'d engine-shape panel for
    `run_config.panel_view` (the lattice materialises + caches it by
    panel_view_hash). `on_set` is the set of runnable toggles held ON — the cell's
    coordinate in the lattice."""
    if getattr(strategy, "refused", False):
        raise CellRunError(
            f"strategy {getattr(strategy, 'strategy_label', '?')!r} is refused; "
            "the lattice must not be run for a refused strategy"
        )

    overridden = _override_construction(strategy, run_config)
    result = run_strategy(overridden, panel, safe_rate=safe_rate, benchmark=benchmark)
    if not isinstance(result, StrategyResult):
        raise CellRunError(
            f"run_strategy returned {type(result).__name__} (refused) for a cell; "
            "construction overrides must not introduce a refusal"
        )

    returns = _series_from_monthly(result.monthly_returns, "strategy_ret")
    n_bonds = _series_from_monthly(result.monthly_returns, "n_bonds")
    metrics = MetricSet.from_summary(result.summary)

    return CellReturns(
        on_set=frozenset(on_set),
        run_config=run_config,
        returns=returns,
        n_bonds=n_bonds,
        metrics_native=metrics,
        run_config_hash=run_config.hash(),
        panel_view_hash=run_config.panel_view_hash(),
        return_hash=hash_series(returns),
        n_bonds_hash=hash_series(n_bonds),
        metric_hash=hash_metrics(result.summary, METRIC_NAMES),
    )
