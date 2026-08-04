"""
alignment.py — paired-sample bookkeeping (shared/evaluation, spec §5).

Every as-published/corrected comparison the pipeline emits must report BOTH the
endpoint estimate on each endpoint's natural sample AND the paired estimate on the
common months (rule 1): if a correction drops months, the raw endpoint gap mixes the
correction's effect with a change in sample composition, which destroys the
differential identification the Auditor's design rests on. So months are PAIRED.

Bond membership differences WITHIN a month are recorded per month but never forced to
match (D-E5): a correction frequently acts by changing bond membership (a survivorship
correction adds defaulted bonds; a stale-price screen removes bonds), so forcing common
bond membership would subtract the treatment being measured.

Pure, I/O-free. Operates on month-end `datetime64[ns]`-indexed `pd.Series` — the
de-facto return-series primitive across the codebase (`summarize_returns`, crowding).
"""

from __future__ import annotations

from typing import Mapping

import numpy as np
import pandas as pd

from .contracts import PairedDifference, RefusalCode, SampleAlignment


def _normalise(series: pd.Series) -> pd.Series:
    """Normalise the index to datetime64[ns] and drop NaN observations — a NaN month is
    not covered. Mirrors the engine's own dropna so coverage here matches the T a
    regression would actually fit on."""
    if not isinstance(series, pd.Series):
        raise TypeError("series must be a pandas Series indexed by date")
    idx = pd.to_datetime(series.index).astype("datetime64[ns]")
    s = pd.Series(series.to_numpy(), index=idx)
    s = s[s.notna()]
    # Defensive: a malformed upstream join can duplicate a month-end label; collapse to
    # the last observation per month so the paired reindex never raises on duplicate
    # labels (invariant #1: degenerate shape -> typed result, never a crash).
    return s[~s.index.duplicated(keep="last")]


def _month_keys(index: pd.Index) -> list[str]:
    """Canonical month-end date key ('YYYY-MM-DD') for each index label."""
    return [pd.Timestamp(t).strftime("%Y-%m-%d") for t in index]


def _normalise_count_keys(mapping: Mapping) -> dict[str, int]:
    """Bond-count mappings are keyed by the same month labels as the series index
    (Timestamps or month-end date strings); normalise both to the canonical key."""
    out: dict[str, int] = {}
    for k, v in mapping.items():
        out[pd.Timestamp(k).strftime("%Y-%m-%d")] = int(v)
    return out


def align_monthly(
    a: pd.Series,
    b: pd.Series,
    *,
    a_label: str,
    b_label: str,
    bond_counts_a: Mapping | None = None,
    bond_counts_b: Mapping | None = None,
) -> SampleAlignment:
    """Month-level alignment of two return series. `a` is treated as the as-published
    endpoint, `b` as the corrected endpoint (the `SampleAlignment` fields carry that
    naming positionally). `a_label`/`b_label` name the endpoints for the paired
    difference and diagnostics. Bond counts, if supplied, are RECORDED per month, never
    used to force common membership."""
    sa = _normalise(a)
    sb = _normalise(b)
    keys_a = _month_keys(sa.index)
    keys_b = _month_keys(sb.index)
    set_a, set_b = set(keys_a), set(keys_b)
    common = sorted(set_a & set_b)

    def _counts(mapping: Mapping | None, keys: list[str]) -> tuple[tuple[str, int], ...]:
        if mapping is None:
            return tuple()
        norm = _normalise_count_keys(mapping)
        return tuple((k, norm[k]) for k in sorted(set(keys)) if k in norm)

    return SampleAlignment(
        as_published_n_months=len(set_a),
        corrected_n_months=len(set_b),
        common_n_months=len(common),
        as_published_only_months=tuple(sorted(set_a - set_b)),
        corrected_only_months=tuple(sorted(set_b - set_a)),
        bond_count_by_month_as_published=_counts(bond_counts_a, keys_a),
        bond_count_by_month_corrected=_counts(bond_counts_b, keys_b),
        common_month_index=tuple(common),
    )


def paired_difference(
    a: pd.Series,
    b: pd.Series,
    alignment: SampleAlignment,
    *,
    a_label: str = "a",
    b_label: str = "b",
) -> PairedDifference:
    """Endpoint means on each natural sample AND the paired mean(a - b) over the common
    months. `common_n_months == 0` -> NO_COMMON_SAMPLE, no paired estimate (rule 2).
    `a_label`/`b_label` name the endpoints (pass the same labels used in `align_monthly`)."""
    sa = _normalise(a)
    sb = _normalise(b)
    mean_a_natural = float(sa.mean()) if len(sa) else None
    mean_b_natural = float(sb.mean()) if len(sb) else None

    if alignment.common_n_months == 0:
        return PairedDifference(
            estimable=False,
            refusal_code=RefusalCode.NO_COMMON_SAMPLE,
            a_label=a_label,
            b_label=b_label,
            mean_a_natural=mean_a_natural,
            mean_b_natural=mean_b_natural,
            n_a_natural=len(sa),
            n_b_natural=len(sb),
            mean_a_common=None,
            mean_b_common=None,
            paired_mean_difference=None,
            n_common=0,
            alignment=alignment,
        )

    sa_k = pd.Series(sa.to_numpy(), index=_month_keys(sa.index))
    sb_k = pd.Series(sb.to_numpy(), index=_month_keys(sb.index))
    common = list(alignment.common_month_index)
    a_c = sa_k.reindex(common).to_numpy()
    b_c = sb_k.reindex(common).to_numpy()

    return PairedDifference(
        estimable=True,
        refusal_code=None,
        a_label=a_label,
        b_label=b_label,
        mean_a_natural=mean_a_natural,
        mean_b_natural=mean_b_natural,
        n_a_natural=len(sa),
        n_b_natural=len(sb),
        mean_a_common=float(np.mean(a_c)),
        mean_b_common=float(np.mean(b_c)),
        paired_mean_difference=float(np.mean(a_c - b_c)),
        n_common=len(common),
        alignment=alignment,
    )
