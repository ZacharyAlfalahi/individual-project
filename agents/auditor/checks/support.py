"""
support.py — common support and the minimum-support gate (§4.2).

Two toggles alter which months have a computable return: `stale_price = ON` masks
month-end prices (invalidating the touching returns) and `survivorship = ON` adds
terminal rows — so cells can differ in which months are valid. If point estimates
were computed on native samples and intervals on a shared one, the interval would
not be an interval for the reported estimate (D-A16). So EVERY cross-cell quantity
is computed on the COMMON support: the intersection of months valid in all cells.

A lattice can be mathematically complete and statistically unusable (§4.2, D-A40),
so a pre-registered minimum-support gate decides whether cross-cell nonlinear
inference proceeds. Native-support metrics are reported alongside, descriptively.
"""

from __future__ import annotations

from typing import Iterable, Mapping, Sequence

import pandas as pd

from ..schemas.audit_core import SupportInfo
from ..schemas.lattice_types import CellReturns
from ..thresholds import SupportGate
from .metrics import metric_set_on


def valid_months(series: pd.Series) -> pd.DatetimeIndex:
    """The months with a computable (non-NaN) return, sorted."""
    return pd.DatetimeIndex(series.dropna().index).sort_values()


def return_matrix(
    cells: Sequence[CellReturns], months: pd.DatetimeIndex
) -> tuple[list, "object"]:
    """The (T x 2^k) return matrix on the common support: column j is cell j's
    returns restricted to `months` (order follows `cells`). Returns (on_set keys,
    ndarray). The single source of the cell return vector process {r_t} consumed
    by the bootstrap (§6) and the inference layer (§7)."""
    import numpy as np

    idx = pd.DatetimeIndex(months).sort_values()
    keys = [cell.on_set for cell in cells]
    cols = [cell.returns.reindex(idx).to_numpy(dtype=float) for cell in cells]
    R = np.column_stack(cols) if cols else np.empty((len(idx), 0))
    return keys, R


def common_support(cells: Sequence[CellReturns]) -> pd.DatetimeIndex:
    """The intersection of months valid in ALL cells (§4.2), sorted."""
    if not cells:
        return pd.DatetimeIndex([])
    common: set | None = None
    for cell in cells:
        months = set(valid_months(cell.returns))
        common = months if common is None else (common & months)
    return pd.DatetimeIndex(sorted(common or set()))


def _native_month_counts(cells: Iterable[CellReturns]) -> list[int]:
    return [len(valid_months(c.returns)) for c in cells]


def support_info(cells: Sequence[CellReturns], gate: SupportGate) -> SupportInfo:
    """Compute the §4.2 support diagnostics and the gate verdict. `downgraded` is
    True when the common support is too short for cross-cell nonlinear inference —
    the caller must then treat the decomposition as descriptive, not inferential."""
    months = common_support(cells)
    t_common = len(months)
    native_counts = _native_month_counts(cells) or [0]
    reference = max(native_counts)
    fraction = (t_common / reference) if reference > 0 else 0.0

    gate_passed = (
        t_common >= gate.min_common_months
        and fraction >= gate.min_common_fraction_of_reference
    )
    return SupportInfo(
        t_common=t_common,
        reference_native_months=reference,
        common_fraction=fraction,
        min_common_months=gate.min_common_months,
        min_common_fraction_of_reference=gate.min_common_fraction_of_reference,
        native_min_months=min(native_counts),
        native_max_months=max(native_counts),
        gate_passed=gate_passed,
        downgraded=not gate_passed,
    )


def primary_metric_vector(
    cells: Sequence[CellReturns],
    months: pd.DatetimeIndex,
    metric_name: str,
    *,
    months_per_year: int = 12,
    nw_lags: int | None = None,
) -> dict[frozenset, float]:
    """Y^common(S) — the primary metric of every cell computed on the SAME common
    support, keyed by the cell's ON-set. This is the exact 2^k input vector the
    §5 transforms and §6 bootstrap consume."""
    out: dict[frozenset, float] = {}
    for cell in cells:
        ms = metric_set_on(
            cell.returns, months, months_per_year=months_per_year, nw_lags=nw_lags
        )
        out[cell.on_set] = ms.value(metric_name)
    return out


def native_metric_vector(
    cells: Sequence[CellReturns], metric_name: str
) -> dict[frozenset, float]:
    """The same metric on each cell's OWN sample — secondary, descriptive (§4.2).
    Never an input to a cross-cell attribution."""
    return {cell.on_set: cell.metrics_native.value(metric_name) for cell in cells}


def common_support_metrics(
    cells: Sequence[CellReturns],
    months: pd.DatetimeIndex,
    *,
    months_per_year: int = 12,
    nw_lags: int | None = None,
) -> Mapping[frozenset, object]:
    """The full MetricSet of every cell on the common support (for reporting all
    metrics, not just the primary one)."""
    return {
        cell.on_set: metric_set_on(
            cell.returns, months, months_per_year=months_per_year, nw_lags=nw_lags
        )
        for cell in cells
    }
