"""
Naive plain-Python reference implementation of the characteristic-sort engine.

Used by tests/unit/test_characteristic_sort_invariants.py as a differential
oracle. This is deliberately written with NO pandas merges, NO groupby, and
NO vectorised numpy -- just dict lookups, sorted(), and explicit Python
loops over per-bond, per-month dictionaries.

If the production engine and this naive reference disagree on a synthetic
panel to 1e-12, ONE of them has a bug. Since the two implementations are
maximally different in style (vectorised pandas vs. plain Python), an
agreement to floating-point precision is strong evidence that both
correctly implement the spec.

This is NOT a production module -- it would be unusably slow on real data.
The leading underscore in the filename keeps pytest from collecting it as
a test module.

Schema returned:
  {"monthly_returns": [
       {date, strategy_ret, long_ret, short_ret, n_bonds},
       ...
   ]}

Supports the subset of rulebook keys needed for differential testing:
  score (required), groups, weighting, control, control_groups,
  long_group, short_group, min_bonds, signal_lag.
"""

from __future__ import annotations

import math
from collections import defaultdict

import pandas as pd


def _is_nan(x) -> bool:
    if x is None:
        return True
    try:
        return math.isnan(float(x))
    except (TypeError, ValueError):
        return False


def naive_run_characteristic_sort(panel: pd.DataFrame, rulebook: dict) -> dict:
    score = rulebook["score"]
    groups = rulebook.get("groups", 5)
    weighting = rulebook.get("weighting", "by_size")
    control = rulebook.get("control", None)
    control_groups = rulebook.get("control_groups", groups)
    long_group = rulebook.get("long_group", groups - 1)
    short_group = rulebook.get("short_group", 0)
    min_bonds = rulebook.get("min_bonds", groups)
    signal_lag = rulebook.get("signal_lag", 0)

    # Pre-group rows by formation date and build a (bond_id, date) lookup
    # for next_ret / signal_lag score lookups.
    records = panel.to_dict("records")
    by_date: dict = defaultdict(list)
    by_key: dict = {}
    for r in records:
        d = pd.Timestamp(r["date"])
        by_date[d].append(r)
        by_key[(r["bond_id"], d)] = r

    monthly_rows = []
    for t in sorted(by_date.keys()):
        # Eligible bonds at formation date t.
        eligible = []
        for row_t in by_date[t]:
            bond_id = row_t["bond_id"]

            # Ranking score: from the row signal_lag months earlier.
            score_obs_date = t - pd.offsets.MonthEnd(signal_lag)
            score_row = by_key.get((bond_id, score_obs_date))
            if score_row is None:
                continue
            ranking_score = score_row[score]
            if _is_nan(ranking_score):
                continue

            # Size at formation.
            size = row_t["size"]
            if _is_nan(size):
                continue

            # Next-month return: from the row at t + 1 month.
            realisation_date = t + pd.offsets.MonthEnd(1)
            next_row = by_key.get((bond_id, realisation_date))
            if next_row is None:
                continue
            next_ret = next_row["ret"]
            if _is_nan(next_ret):
                continue

            # Control value (if set).
            if control is not None:
                ctrl_val = row_t[control]
                if _is_nan(ctrl_val):
                    continue
            else:
                ctrl_val = None

            eligible.append(
                {
                    "bond_id": bond_id,
                    "ranking_score": float(ranking_score),
                    "size": float(size),
                    "next_ret": float(next_ret),
                    "ctrl": float(ctrl_val) if ctrl_val is not None else None,
                }
            )

        if len(eligible) < min_bonds:
            continue

        n = len(eligible)

        # Score groups: stable sort by (score asc, bond_id asc), then
        # group = ((rank-1) * G) // n. Rank is 1-indexed; here we use the
        # 0-indexed position i directly, which is equivalent.
        for i, b in enumerate(
            sorted(eligible, key=lambda x: (x["ranking_score"], x["bond_id"]))
        ):
            b["_score_group"] = (i * groups) // n

        # Control groups: independent over the same eligible set.
        if control is not None:
            for i, b in enumerate(
                sorted(eligible, key=lambda x: (x["ctrl"], x["bond_id"]))
            ):
                b["_control_group"] = (i * control_groups) // n
            stripe_keys = list(range(control_groups))
        else:
            for b in eligible:
                b["_control_group"] = 0
            stripe_keys = [0]

        stripe_spreads = []
        stripe_longs = []
        stripe_shorts = []
        stripe_n = []
        for s in stripe_keys:
            stripe_bonds = [b for b in eligible if b["_control_group"] == s]
            long_bonds = [b for b in stripe_bonds if b["_score_group"] == long_group]
            short_bonds = [b for b in stripe_bonds if b["_score_group"] == short_group]
            if not long_bonds or not short_bonds:
                continue

            if weighting == "equal":
                long_ret = sum(b["next_ret"] for b in long_bonds) / len(long_bonds)
                short_ret = sum(b["next_ret"] for b in short_bonds) / len(short_bonds)
            elif weighting == "by_size":
                lw = sum(b["size"] for b in long_bonds)
                sw = sum(b["size"] for b in short_bonds)
                if lw <= 0 or sw <= 0:
                    continue
                long_ret = sum(b["size"] * b["next_ret"] for b in long_bonds) / lw
                short_ret = sum(b["size"] * b["next_ret"] for b in short_bonds) / sw
            else:
                raise ValueError(f"Unknown weighting: {weighting!r}")

            stripe_spreads.append(long_ret - short_ret)
            stripe_longs.append(long_ret)
            stripe_shorts.append(short_ret)
            stripe_n.append(len(long_bonds) + len(short_bonds))

        if not stripe_spreads:
            continue

        monthly_rows.append(
            {
                "date": t + pd.offsets.MonthEnd(1),
                "strategy_ret": sum(stripe_spreads) / len(stripe_spreads),
                "long_ret": sum(stripe_longs) / len(stripe_longs),
                "short_ret": sum(stripe_shorts) / len(stripe_shorts),
                "n_bonds": sum(stripe_n),
            }
        )

    monthly_rows.sort(key=lambda r: r["date"])
    return {"monthly_returns": monthly_rows}
