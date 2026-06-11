"""
Overlapping-holding wrapper around the single-month characteristic-sort engine.

Lets a factor be held for H months via staggered cohorts:

  * Each formation month t produces a long/short cohort using the engine's
    selection logic at t (frozen at formation: bonds + formation-month sizes).
  * The cohort earns its weighted return in each of months t+1 ... t+H.
    Bonds that exit mid-hold (no panel row at the realisation month, or NaN
    ret) are dropped; surviving weights are renormalised within the leg.
  * The factor's return at calendar month m = simple average across all
    cohorts still alive at m (up to H cohorts, formed in m-1, m-2, ..., m-H).

H=1 short-circuits to `run_characteristic_sort` -- regression invariant the
spec demands.

v1 limitation: single-sort only. Passing `control` in the rulebook raises
NotImplementedError so any future enable is intentional.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from characteristic_sort import (  # noqa: E402
    _apply_defaults,
    extract_monthly_selections,
    run_characteristic_sort,
)


def run_with_holding_period(
    panel: pd.DataFrame,
    rulebook: dict,
    holding_period: int = 1,
) -> pd.DataFrame:
    """
    Run the characteristic-sort engine with a multi-month holding period.

    Parameters
    ----------
    panel : DataFrame
        Same contract as `run_characteristic_sort`: `cusip`, `date`
        (month-end, tz-naive), `ret`, `size`, plus the score column named
        in `rulebook["score"]`.
    rulebook : dict
        Same as `run_characteristic_sort`. `control` is not supported in
        v1 -- passing it raises NotImplementedError.
    holding_period : int, default 1
        Number of months a cohort is held. H=1 reduces exactly to
        `run_characteristic_sort` (regression invariant).

    Returns
    -------
    pd.DataFrame with columns
        date              : realisation month (calendar month m, month-end)
        strategy_ret      : simple mean across cohorts of long_ret - short_ret
        long_ret          : simple mean of cohort long-leg returns at m
        short_ret         : simple mean of cohort short-leg returns at m
        n_bonds           : total bonds across all cohorts alive at m
        n_cohorts_alive   : number of contributing cohorts (1..H)

    At H=1 the underlying engine's `monthly_returns` frame is returned with
    `n_cohorts_alive` appended (always 1), so the schema is identical at
    H=1 and H>1.
    """
    if not isinstance(holding_period, int) or holding_period < 1:
        raise ValueError("holding_period must be a positive integer")

    if rulebook.get("control") is not None:
        raise NotImplementedError(
            "overlap.run_with_holding_period does not support control "
            "(double-sort) in v1. Use run_characteristic_sort directly "
            "or omit 'control' from the rulebook."
        )

    if holding_period == 1:
        # Strict short-circuit: identical to the engine's single-month
        # output. Append n_cohorts_alive=1 so the schema matches the H>1
        # path; the rest of the frame is the engine's verbatim output and
        # must compare element-wise equal for the regression invariant.
        engine_out = run_characteristic_sort(panel, rulebook)
        mr = engine_out["monthly_returns"].copy()
        if len(mr) > 0:
            mr["n_cohorts_alive"] = 1
        else:
            mr["n_cohorts_alive"] = pd.Series(dtype=int)
        return mr

    settings = _apply_defaults(rulebook)
    weighting = settings["weighting"]

    selections = extract_monthly_selections(panel, rulebook)

    # Fast (cusip, date) -> ret lookup. Built once per call.
    panel_lookup = dict(
        zip(
            zip(panel["cusip"].values, panel["date"].values),
            panel["ret"].values,
        )
    )

    # Each cohort contributes one (long_ret, short_ret, n_bonds) triple per
    # realisation month it's alive at. Skip the cohort-month if either leg
    # is empty after exits.
    contributions: dict[pd.Timestamp, list[tuple[float, float, int]]] = {}

    for formation_t, sel in selections.items():
        for h in range(1, holding_period + 1):
            m = formation_t + pd.offsets.MonthEnd(h)

            long_r, n_long = _leg_return_at(
                sel["long"], m, panel_lookup, weighting
            )
            short_r, n_short = _leg_return_at(
                sel["short"], m, panel_lookup, weighting
            )

            if long_r is None or short_r is None:
                continue

            contributions.setdefault(m, []).append(
                (long_r, short_r, n_long + n_short)
            )

    if not contributions:
        return pd.DataFrame(
            {
                "date": pd.Series(dtype="datetime64[ns]"),
                "strategy_ret": pd.Series(dtype=float),
                "long_ret": pd.Series(dtype=float),
                "short_ret": pd.Series(dtype=float),
                "n_bonds": pd.Series(dtype=int),
                "n_cohorts_alive": pd.Series(dtype=int),
            }
        )

    rows = []
    for m in sorted(contributions.keys()):
        triples = contributions[m]
        longs = np.array([t[0] for t in triples])
        shorts = np.array([t[1] for t in triples])
        # By linearity of mean: mean(long_i - short_i) = mean(long_i) -
        # mean(short_i) since every cohort contributes to both sides.
        long_mean = float(longs.mean())
        short_mean = float(shorts.mean())
        rows.append(
            {
                "date": m,
                "strategy_ret": long_mean - short_mean,
                "long_ret": long_mean,
                "short_ret": short_mean,
                "n_bonds": int(sum(t[2] for t in triples)),
                "n_cohorts_alive": len(triples),
            }
        )
    return pd.DataFrame(rows).sort_values("date").reset_index(drop=True)


def _leg_return_at(
    leg: pd.DataFrame,
    m: pd.Timestamp,
    panel_lookup: dict,
    weighting: str,
) -> tuple[float | None, int]:
    """
    Compute one leg's weighted return at realisation month m.

    Drops cohort bonds with no panel row at m (or NaN ret); surviving
    weights are renormalised within the leg.

    Returns (leg_ret, n_surviving). (None, 0) if no bonds survive or, under
    by_size weighting, all surviving sizes sum to <= 0.
    """
    cusips = leg["cusip"].values
    sizes = leg["size"].values

    rets_list: list[float] = []
    surviving_sizes: list[float] = []
    # numpy.datetime64('us') from the panel can mismatch a pandas Timestamp
    # at lookup time; normalise m to the same dtype the panel_lookup keys
    # were built with.
    m64 = np.datetime64(m, "ns")
    for c, s in zip(cusips, sizes):
        r = panel_lookup.get((c, m64))
        if r is None or pd.isna(r):
            continue
        rets_list.append(float(r))
        surviving_sizes.append(float(s))

    if not rets_list:
        return None, 0

    rets_arr = np.array(rets_list)
    sz_arr = np.array(surviving_sizes)

    if weighting == "equal":
        return float(rets_arr.mean()), len(rets_arr)
    elif weighting == "by_size":
        w = float(sz_arr.sum())
        if w <= 0:
            return None, 0
        return float((sz_arr * rets_arr).sum() / w), len(rets_arr)
    else:
        raise ValueError(f"Unknown weighting: {weighting!r}")
