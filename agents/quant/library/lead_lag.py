"""
Lead/lag (look-ahead) error injector — the BBW factor lead/lag toggle
(docs/quant/specs/BBW_anchor_implementation_spec.md §7 toggle 1).

DRR-2023 document that the as-published BBW factor series carry a date-alignment
error over specific windows: DRF and CRF for month t are actually month t+1's
value (a LEAD error) over 2004-08…2014-12, and LRF for month t is month t-1's
value (a LAG error) over 2015-01…2016-12. OFF reproduces the defect; ON is the
correct contemporaneous alignment.

The signature (§7): injecting the defect collapses the correlation between the
as-published and correctly-aligned series to roughly the factor's lag-1
autocorrelation (~0.26 DRF, ~0.44 CRF — monthly factor returns are nearly
serially uncorrelated, so a one-month misalignment scrambles them); correcting
the alignment restores correlation to >0.90.

Pure function on a monthly factor series. The injector is the only "bias on"
operation; correctness is the un-injected series itself.
"""

from __future__ import annotations

import pandas as pd


def inject_lead_lag(
    series: pd.DataFrame,
    shift_months: int,
    window: tuple[str, str],
    *,
    date_col: str = "date",
    value_col: str = "strategy_ret",
) -> pd.DataFrame:
    """Return a copy of `series` with the as-published lead/lag error injected.

    Within the inclusive month window, value[t] is replaced by value[t + shift]:
      shift_months = +1  → LEAD error (t carries t+1's value), DRF/CRF.
      shift_months = -1  → LAG error  (t carries t-1's value), LRF.
    Outside the window the value is unchanged. Where the shifted source falls
    outside the series (e.g. a lead at the final month), the injected value is
    NaN (the defect is undefined there).
    """
    if not isinstance(shift_months, int) or shift_months == 0:
        raise ValueError("shift_months must be a non-zero integer (+1 lead, -1 lag)")

    df = series.sort_values(date_col, kind="mergesort").reset_index(drop=True).copy()
    start = pd.Timestamp(window[0]) + pd.offsets.MonthEnd(0)
    end = pd.Timestamp(window[1]) + pd.offsets.MonthEnd(0)

    # value[t] := value[t + shift]  ⇒  shift the series up by `shift_months`.
    shifted = df[value_col].shift(-shift_months)
    in_window = (df[date_col] >= start) & (df[date_col] <= end)
    df.loc[in_window, value_col] = shifted[in_window].to_numpy()
    return df
