"""
metrics.py — turn a return series into a typed MetricSet on a chosen month support
(§9.1).

Metrics are functionals of the return series (§4.1), and the same functional is
applied on two supports (§4.2): the cell's own (native) sample, and the common
intersection shared by every cell. This module is the single place the engine's
`summarize_returns` is called from the Auditor — the shared statistics contract
(§9.1: "defined once in the shared statistics module; never reimplemented inside
the Auditor").
"""

from __future__ import annotations

import pandas as pd

from agents.quant.library.characteristic_sort import summarize_returns

from ..schemas.lattice_types import MetricSet


def metric_set_on(
    returns: pd.Series,
    months: pd.DatetimeIndex | None = None,
    *,
    months_per_year: int = 12,
    nw_lags: int | None = None,
) -> MetricSet:
    """MetricSet for `returns` restricted to `months` (or the full series when
    `months` is None). On the common support every cell is non-NaN by
    construction, so the restricted series has exactly `len(months)` observations."""
    if months is not None:
        idx = pd.DatetimeIndex(months).sort_values()
        restricted = returns.reindex(idx)
        # Self-enforce the §4.2 "same support for every cell" contract: a requested
        # month where this cell is NaN would be silently dropped by summarize_returns'
        # internal dropna, computing the metric on a DIFFERENT support than requested
        # (exactly the estimand mismatch common support exists to prevent). On the
        # common support every cell is valid by construction, so this never fires there.
        if int(restricted.notna().sum()) != len(idx):
            raise ValueError(
                "metric_set_on: the requested months include months where this cell "
                "has no return; the common-support invariant (§4.2) is violated"
            )
    else:
        restricted = returns
    summary = summarize_returns(restricted, nw_lags, months_per_year)
    return MetricSet.from_summary(summary)
