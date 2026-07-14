"""
Ex-post vs ex-ante return winsorization — the mom6 look-ahead (LAB) toggle
(docs/quant/specs/BBW_anchor_implementation_spec.md §7.1).

This is the bias DRR-2026 identify: the +0.30%/mo momentum premium is entirely
an artefact of asymmetric EX-POST winsorization — a one-sided right-tail clip
whose 99.5th-percentile threshold is computed on the FULL sample, embedding
future information. The repair is EX-ANTE: recompute the threshold each month on
strictly past data (date < t), so it cannot know future return spikes.

The audited characteristic-sort engine's `trim_rule` deliberately supports only
absolute bounds on a full sample (it raises NotImplementedError for percentile /
by-month samples), so this percentile+expanding logic lives here, as a panel-
level transform applied to the realised holding return BEFORE the sort. Per
§7.1, clipping does not change which bonds are ranked or held — the leak is
purely the capped realised return — so only the return column is touched, not
the ranking signal.

`loc='right'` is one-sided (upper tail only) — asymmetric, per DRR Table 3 (the
momentum factors clip the right tail; left-tail 0.5th is a different factor's
knob). Winsorize = CLIP (`adj='wins'`), NOT truncate/drop (`adj='trim'`).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _apply_clip(ret: np.ndarray, lower, upper, loc: str) -> np.ndarray:
    out = ret.copy()
    if loc in ("right", "both"):
        out = np.where(out > upper, upper, out)
    if loc in ("left", "both"):
        out = np.where(out < lower, lower, out)
    return out


def winsorize_returns(
    returns: pd.Series,
    dates: pd.Series | None = None,
    level: float = 99.5,
    loc: str = "right",
    mode: str = "ex_post",
) -> pd.Series:
    """Winsorize (clip) a return series; returns a Series on the same index.

    Parameters
    ----------
    returns : the return series to clip.
    dates : month-end dates aligned to `returns`; REQUIRED for mode='ex_ante'.
    level : the upper percentile (e.g. 99.5); the lower tail uses 100 - level.
    loc : 'right' (upper tail only, default), 'left', or 'both'.
    mode : 'ex_post' — one static full-sample percentile threshold (biased; embeds
           future info); 'ex_ante' — per-month threshold from strictly past data
           (date < t), expanding window (the repair). Months with no prior data
           are left unclipped.

    NaNs are preserved (never clipped, never count toward a percentile).
    """
    if loc not in ("right", "left", "both"):
        raise ValueError("loc must be 'right', 'left', or 'both'")
    if mode not in ("ex_post", "ex_ante"):
        raise ValueError("mode must be 'ex_post' or 'ex_ante'")
    if not 0 < level < 100:
        raise ValueError("level must be in (0, 100)")

    r = returns.to_numpy(dtype=float)
    out = r.copy()

    if mode == "ex_post":
        finite = r[~np.isnan(r)]
        if finite.size == 0:
            return pd.Series(out, index=returns.index)
        upper = np.percentile(finite, level)
        lower = np.percentile(finite, 100.0 - level)
        clipped = _apply_clip(r, lower, upper, loc)
        # Preserve NaNs (np.where on a NaN keeps NaN since NaN > upper is False).
        out = clipped
        return pd.Series(out, index=returns.index)

    # ex_ante: expanding past-only percentile per month.
    if dates is None:
        raise ValueError("dates is required for mode='ex_ante'")
    d = pd.to_datetime(dates).to_numpy()
    uniq = np.sort(pd.unique(d))
    for t in uniq:
        past = r[(d < t) & ~np.isnan(r)]
        cur = d == t
        if past.size == 0:
            continue  # no history → leave this month unclipped
        upper = np.percentile(past, level)
        lower = np.percentile(past, 100.0 - level)
        out[cur] = _apply_clip(r[cur], lower, upper, loc)

    return pd.Series(out, index=returns.index)
