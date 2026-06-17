"""
Market-factor (MKTB) builder.

The bond-market excess return the BBW factors price against: the value-weighted
average excess return of all eligible bonds in a month. No sort, no long-short
(spec BBW_anchor_implementation_spec.md §3.6). Par-weighted by default
(weight = FISD `offering_amt`, surfaced as the panel `size` column, §2.4).

Kept correction-agnostic and family-agnostic: the caller passes the excess-return
column for the family it wants (`xret_raw` or `xret_corr`); this module never
mixes families. Pure function — IO and dual-family assembly live in
scripts/build_mktb.py, mirroring the engine / signal-script split.
"""

from __future__ import annotations

import pandas as pd


def compute_market_factor(
    panel: pd.DataFrame,
    ret_col: str,
    weight_col: str = "size",
    eligible_col: str | None = "universe_eligible",
) -> pd.DataFrame:
    """Value-weighted market excess-return series (BBW MKTB).

    For each month, MKTB = sum_i w_i * ret_i with w_i = weight_i / sum_j weight_j,
    taken over the rows that simultaneously

      * are eligible — `eligible_col` is True, when that column is supplied and
        present (None or absent skips the eligibility gate),
      * have a finite `ret_col`, and
      * have a strictly positive `weight_col`.

    No sort and no long-short: this is the market basket, not a spread. The
    safe rate is NOT subtracted here — `ret_col` is already an excess return
    (the panel's `xret_*`). Par-weighted when `weight_col='size'`.

    Parameters
    ----------
    panel : DataFrame with at least `date`, `ret_col`, `weight_col`.
    ret_col : excess-return column for one family (e.g. 'xret_raw').
    weight_col : value-weight column (default 'size' = par offering_amt).
    eligible_col : boolean eligibility column, or None to weight all rows.

    Returns
    -------
    DataFrame[date, mktb, n_bonds] sorted by date. Months with no eligible bond
    (or a non-positive total weight) are omitted.
    """
    base = {"date", ret_col, weight_col}
    missing = base - set(panel.columns)
    if missing:
        raise ValueError(f"panel missing required columns: {sorted(missing)}")

    cols = ["date", ret_col, weight_col]
    use_eligible = bool(eligible_col) and eligible_col in panel.columns
    if use_eligible:
        cols.append(eligible_col)
    df = panel[cols].copy()

    mask = df[ret_col].notna() & df[weight_col].notna() & (df[weight_col] > 0)
    if use_eligible:
        mask &= df[eligible_col] == True  # noqa: E712
    df = df[mask]

    empty = pd.DataFrame(
        {
            "date": pd.Series(dtype="datetime64[ns]"),
            "mktb": pd.Series(dtype=float),
            "n_bonds": pd.Series(dtype=int),
        }
    )
    if df.empty:
        return empty

    df = df.copy()
    df["_wr"] = df[weight_col].to_numpy(dtype=float) * df[ret_col].to_numpy(dtype=float)
    g = df.groupby("date", sort=True)
    agg = g.agg(
        _wsum=(weight_col, "sum"),
        _wrsum=("_wr", "sum"),
        n_bonds=(ret_col, "size"),
    )
    # A month survives the mask but could in principle net a non-positive weight
    # sum only if all weights were filtered; the (weight > 0) mask already rules
    # that out, so _wsum > 0 here. Guard anyway rather than emit inf/NaN.
    agg = agg[agg["_wsum"] > 0]
    agg["mktb"] = agg["_wrsum"] / agg["_wsum"]
    out = agg.reset_index()[["date", "mktb", "n_bonds"]]
    out["n_bonds"] = out["n_bonds"].astype(int)
    return out.sort_values("date").reset_index(drop=True)
