"""
run_from_config -- execute a ``QuantConfig`` against a panel (or short-circuit a
``ConfigRefusal``), returning ONE normalized result envelope regardless of the
holding-period path.

The base engine (``run_characteristic_sort``) returns a rich dict; the overlap
wrapper (``run_with_holding_period``) returns a bare DataFrame. This module
reconciles them: both paths yield the same five-key envelope
(``monthly_returns`` / ``summary`` / ``relationship_to_benchmark`` /
``settings_used`` / ``bookkeeping``) so callers get a consistent shape. It
authors no algorithmic code -- the overlap path's summary reuses the engine's
public ``summarize_returns``.
"""

from __future__ import annotations

import pandas as pd

from ..library.characteristic_sort import (
    _apply_defaults,
    run_characteristic_sort,
    summarize_returns,
)
from ..library.overlap import run_with_holding_period
from .quant_config import QuantConfig, QuantConfigError, to_rulebook
from .refusal import ConfigRefusal


def run_from_config(
    config: QuantConfig | ConfigRefusal,
    panel: pd.DataFrame,
    *,
    safe_rate: pd.DataFrame | None = None,
    benchmark: pd.DataFrame | None = None,
) -> dict | ConfigRefusal:
    """Run one strategy. A ``ConfigRefusal`` is returned unchanged and the engine
    is never touched. Otherwise the config is stripped to a rulebook and
    dispatched: ``holding_period == 1`` -> base engine; ``> 1`` -> overlap
    (normalized to the same envelope)."""
    if isinstance(config, ConfigRefusal):
        return config  # record + terminate; never call the engine

    rulebook = to_rulebook(config)
    holding_period = config.holding_period.value
    assert isinstance(holding_period, int)  # invariant: QuantConfig.__post_init__ guarantees it

    control_bound = config.control is not None and config.control.is_usable
    # Unreachable via build_quant_config (it refuses control + holding>1); this
    # tripwire only guards a hand-built config that bypassed the factory, so the
    # overlap module's raw NotImplementedError can never leak. An explicit raise
    # (not an assert) so it survives `python -O`.
    if control_bound and holding_period > 1:
        raise QuantConfigError(
            "control + holding_period>1 is unsupported and must have been refused "
            "by build_quant_config (overlap cannot double-sort)"
        )

    if holding_period == 1:
        return run_characteristic_sort(
            panel, rulebook, safe_rate=safe_rate, benchmark=benchmark
        )

    # The overlap path takes neither benchmark nor safe_rate; reject both
    # explicitly rather than dropping one silently (symmetry).
    if benchmark is not None:
        raise ValueError(
            "benchmark regression is not available for holding_period>1 "
            "(overlap.run_with_holding_period has no benchmark path)"
        )
    if safe_rate is not None:
        raise ValueError(
            "safe_rate is not accepted for holding_period>1 "
            "(overlap.run_with_holding_period has no safe_rate path)"
        )

    monthly = run_with_holding_period(panel, rulebook, holding_period=holding_period)
    return _normalize_overlap(monthly, rulebook)


def _normalize_overlap(monthly: pd.DataFrame, rulebook: dict) -> dict:
    """Wrap the overlap DataFrame into the base engine's result envelope."""
    settings = _apply_defaults(rulebook)

    if len(monthly) > 0:
        ret_series = pd.Series(
            monthly["strategy_ret"].values,
            index=pd.DatetimeIndex(monthly["date"].values),
        )
        avg_bonds = float(monthly["n_bonds"].mean())
    else:
        ret_series = pd.Series([], dtype=float, index=pd.DatetimeIndex([]))
        avg_bonds = float("nan")

    summary = summarize_returns(
        ret_series, settings["nw_lags"], settings["months_per_year"]
    )
    summary["avg_bonds_per_month"] = avg_bonds

    return {
        "monthly_returns": monthly,
        "summary": summary,
        # overlap has no benchmark path; kept empty for envelope parity.
        "relationship_to_benchmark": {},
        "settings_used": settings,
        # overlap does not expose the engine's per-month bookkeeping.
        "bookkeeping": {},
    }
