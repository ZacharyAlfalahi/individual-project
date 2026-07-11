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

``run_strategy`` sits one level up (D28, ledger item 40): it takes an adapter
``AdaptResult`` (a whole multi-leg strategy), runs each leg through
``run_from_config``, and combines the per-leg return series per the strategy's
combiner (``single_leg`` pass-through or ``equal_average`` by-date mean with an
adaptive divisor). It still authors no algorithmic code -- the combine is a
by-date arithmetic mean and the summary reuses ``summarize_returns``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import pandas as pd

from ..library.characteristic_sort import (
    _apply_defaults,
    run_characteristic_sort,
    summarize_returns,
)
from ..library.overlap import run_with_holding_period
from .quant_config import QuantConfig, QuantConfigError, to_rulebook
from .refusal import ConfigRefusal

if TYPE_CHECKING:  # runtime import would cycle: adapter.result -> quant.config -> runner
    from agents.librarian.adapter.result import AdaptResult


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


# ---------------------------------------------------------------------------
# Strategy-level execution: run every leg + combine (D28, ledger item 40).
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class StrategyResult:
    """One whole strategy's result. Mirrors the five-key ``run_from_config``
    envelope (``monthly_returns`` / ``summary`` / ``relationship_to_benchmark`` /
    ``settings_used`` / ``bookkeeping``) so downstream (RQ1) sees ONE shape,
    plus strategy-level metadata (``strategy_label`` / ``variant`` / ``n_legs`` /
    ``combiner``). ``single_leg`` passes the leg envelope through verbatim;
    ``equal_average`` combines and leaves ``relationship_to_benchmark`` /
    ``bookkeeping`` empty (envelope parity, as ``_normalize_overlap`` does)."""

    strategy_label: str
    monthly_returns: pd.DataFrame
    summary: dict
    relationship_to_benchmark: dict
    settings_used: dict
    bookkeeping: dict
    variant: bool
    n_legs: int
    combiner: dict


def run_strategy(
    result: "AdaptResult",
    panel: pd.DataFrame,
    *,
    safe_rate: pd.DataFrame | None = None,
    benchmark: pd.DataFrame | None = None,
) -> "StrategyResult | AdaptResult":
    """Execute a whole strategy: run each leg's ``QuantConfig`` via
    ``run_from_config``, then combine per ``result.combiner``.

    A refused strategy (``result.refused``) is returned UNRUN -- the engine is
    never touched and no partial averaging happens (a refused leg would silently
    change the construction, D28). ``single_leg`` is a pass-through of the one
    leg. ``equal_average`` takes the by-date arithmetic mean of the per-leg
    ``strategy_ret`` series over the legs PRESENT that month (adaptive divisor =
    count of legs with a value that date -- ledger item 40)."""
    if result.refused:
        # Return the AdaptResult unchanged: it carries the full refusal
        # provenance (result.refusals) for the RQ2 coverage table.
        return result

    combiner = result.combiner
    if combiner is None:
        raise ValueError(
            f"run_strategy: strategy {result.strategy_label!r} has no combiner; "
            "a non-refused AdaptResult must carry one (D28)"
        )

    if combiner.kind == "single_leg":
        env = _run_leg(result.leg_calls[0], panel, safe_rate=safe_rate, benchmark=benchmark)
        return _strategy_result_from_envelope(result, env)

    if combiner.kind == "equal_average":
        # A benchmark regression on a *combined* multi-leg series is undefined
        # (run_from_config only benchmarks the single-leg h=1 path); reject it
        # explicitly rather than silently dropping (mirrors the overlap path).
        if benchmark is not None:
            raise ValueError(
                "benchmark regression is not available for an equal_average "
                "strategy (the combined series has no single-leg benchmark path)"
            )
        return _combine_equal_average(result, panel, safe_rate=safe_rate)

    raise ValueError(
        f"run_strategy: unsupported combiner kind {combiner.kind!r} "
        "(expected 'single_leg' or 'equal_average')"
    )


def _run_leg(
    leg_call,
    panel: pd.DataFrame,
    *,
    safe_rate: pd.DataFrame | None,
    benchmark: pd.DataFrame | None,
) -> dict:
    """Run one leg's config, returning its five-key envelope. Guards the
    invariant that a non-refused strategy's legs are all runnable QuantConfigs
    (a refused/None leg must have short-circuited in run_strategy already)."""
    cfg = leg_call.result
    if not isinstance(cfg, QuantConfig):
        raise ValueError(
            f"run_strategy reached leg {leg_call.strategy_id!r} with a "
            f"non-runnable result ({type(cfg).__name__}); a refused strategy "
            "must short-circuit before any leg runs"
        )
    env = run_from_config(cfg, panel, safe_rate=safe_rate, benchmark=benchmark)
    # run_from_config only returns a ConfigRefusal for a ConfigRefusal input,
    # excluded above -- so env is always the dict envelope here.
    assert isinstance(env, dict)
    return env


def _strategy_result_from_envelope(result: "AdaptResult", env: dict) -> StrategyResult:
    """Wrap a single leg's envelope as the strategy result (pass-through)."""
    return StrategyResult(
        strategy_label=result.strategy_label,
        monthly_returns=env["monthly_returns"],
        summary=env["summary"],
        relationship_to_benchmark=env["relationship_to_benchmark"],
        settings_used=env["settings_used"],
        bookkeeping=env["bookkeeping"],
        variant=result.variant,
        n_legs=len(result.leg_calls),
        combiner=result.combiner.to_dict(),
    )


def _leg_return_series(env: dict) -> pd.Series:
    """The leg's monthly ``strategy_ret`` as a date-indexed Series, reconstructed
    exactly as ``_normalize_overlap`` does (``.values`` + explicit DatetimeIndex
    to avoid dtype/tz drift on the index)."""
    monthly = env["monthly_returns"]
    if len(monthly) == 0:
        return pd.Series([], dtype=float, index=pd.DatetimeIndex([]))
    return pd.Series(
        monthly["strategy_ret"].values,
        index=pd.DatetimeIndex(monthly["date"].values),
    )


def _combine_equal_average(
    result: "AdaptResult", panel: pd.DataFrame, *, safe_rate: pd.DataFrame | None
) -> StrategyResult:
    """By-date arithmetic mean of the per-leg return series, with the adaptive
    divisor (ledger item 40): a month where a leg is absent divides by the count
    of legs PRESENT that month, not the nominal leg count."""
    envs = [
        _run_leg(lc, panel, safe_rate=safe_rate, benchmark=None) for lc in result.leg_calls
    ]

    # DataFrame.mean(axis=1, skipna=True) divides each month by the count of
    # non-NaN legs that month -- exactly the adaptive divisor. A leg absent that
    # month (min_bonds skipped it -> NaN / no row) drops from the divisor.
    wide = pd.concat([_leg_return_series(e) for e in envs], axis=1)
    combined = wide.mean(axis=1, skipna=True).sort_index()

    # nw_lags / months_per_year come from the common fields, identical across
    # legs by construction; take leg 0's and assert agreement (loud on drift).
    settings = envs[0]["settings_used"]
    mpy = settings["months_per_year"]
    for e in envs[1:]:
        if e["settings_used"]["months_per_year"] != mpy:
            raise ValueError(
                "equal_average legs disagree on months_per_year "
                f"({mpy} vs {e['settings_used']['months_per_year']}); the adapter "
                "emitted inconsistent per-leg settings"
            )

    summary = summarize_returns(combined, settings["nw_lags"], settings["months_per_year"])
    # Parity with the base/overlap envelope summary, which carries this key.
    # Total average bonds the strategy deploys per month = sum over legs of each
    # leg's own avg_bonds_per_month (NaN only if every leg is empty).
    leg_avgs = pd.Series([e["summary"].get("avg_bonds_per_month", float("nan")) for e in envs])
    summary["avg_bonds_per_month"] = (
        float(leg_avgs.sum()) if bool(leg_avgs.notna().any()) else float("nan")
    )

    combined_mr = pd.DataFrame(
        {"date": pd.DatetimeIndex(combined.index), "strategy_ret": combined.to_numpy()}
    )

    return StrategyResult(
        strategy_label=result.strategy_label,
        monthly_returns=combined_mr,
        summary=summary,
        # A combined multi-leg series has no single benchmark / bookkeeping path;
        # empty for envelope parity (mirrors _normalize_overlap).
        relationship_to_benchmark={},
        settings_used=settings,
        bookkeeping={},
        variant=result.variant,
        n_legs=len(result.leg_calls),
        combiner=result.combiner.to_dict(),
    )
