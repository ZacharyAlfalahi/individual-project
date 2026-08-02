"""Panel transforms (finding F8) — how T1/T2/T3 are realised WITHOUT touching the audited engine.

A MONTH filter (T1/T2) partitions the corrected panel by the lagged macro regime; a ROW filter
(T3) restricts it to an ex-ante segment. The unmodified `characteristic_sort` engine then runs on
the transformed panel. The audit-critical rule (F8 / INVARIANT 3): the regime threshold is an
EXPANDING past-only median, NEVER a full-sample median — a full-sample "above the 2002-2021
median" uses the future at every prior point. Row-filter segments are observed at formation.
"""

from __future__ import annotations

import pandas as pd


def expanding_regime_mask(
    macro: pd.Series, *, lag: int, high: bool = True, min_history: int = 60
) -> pd.Series:
    """Per-month regime classification with NO look-ahead. At month t, compare the lagged regime
    value regime(t-lag) to the EXPANDING median of the lagged series up to t (past-only). Returns
    a bool Series indexed like `macro`: True where the lagged regime is above (high) / below the
    expanding median. The first `min_history` observations are False — an expanding median needs
    history, and there is deliberately never a full-sample fallback."""
    macro = macro.sort_index()
    lagged = macro.shift(lag)                                   # regime observed `lag` months earlier
    hist_median = lagged.expanding(min_periods=min_history).median()
    mask = (lagged > hist_median) if high else (lagged < hist_median)
    return mask.reindex(macro.index).fillna(False).astype(bool)


def apply_month_filter(
    panel: pd.DataFrame, macro: pd.Series, *, lag: int, form: str, min_history: int = 60
) -> pd.DataFrame:
    """Subset the panel to the months selected by the lagged-regime mask (`macro` indexed by the
    same month-end date as panel['date']). `form` chooses the regime side."""
    high = "above" in form or "top" in form
    mask = expanding_regime_mask(macro, lag=lag, high=high, min_history=min_history)
    keep = set(mask.index[mask])
    return panel[panel["date"].isin(keep)].copy()


def apply_row_filter(panel: pd.DataFrame, *, variable: str, form: str) -> pd.DataFrame:
    """Restrict to an EX-ANTE segment (observed at formation). Rating segments use the direct
    membership column; a liquidity tercile uses the cross-sectional rank of `variable` WITHIN each
    formation month (ex-ante — no future, no realised return)."""
    if form == "restrict_investment_grade":
        return panel[panel["investment_grade"].astype(bool)].copy()
    if form == "restrict_high_yield":
        return panel[~panel["investment_grade"].astype(bool)].copy()
    if form in ("restrict_top_liquidity_tercile", "restrict_bottom_liquidity_tercile"):
        top = "top" in form

        def _seg(g: pd.DataFrame) -> pd.DataFrame:
            pct = g[variable].rank(pct=True, method="average")
            return g[pct > 2 / 3] if top else g[pct <= 1 / 3]

        return panel.groupby("date", group_keys=False)[panel.columns.tolist()].apply(_seg).copy()
    raise ValueError(f"unknown row-filter form: {form!r}")
