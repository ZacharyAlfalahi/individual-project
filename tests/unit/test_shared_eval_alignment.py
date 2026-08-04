"""
test_shared_eval_alignment.py — month-level pairing mandatory, bond membership recorded
not forced (D-E5); NO_COMMON_SAMPLE on no overlap (rule 2); the alignment record travels
with the paired result (rule 3).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from shared.evaluation.alignment import align_monthly, paired_difference
from shared.evaluation.contracts import RefusalCode


def _series(start: str, n: int, val: float = 0.01) -> pd.Series:
    idx = pd.date_range(start, periods=n, freq="ME")
    return pd.Series(np.full(n, val), index=idx)


def test_identical_endpoints_common_equals_both() -> None:
    a = _series("2002-01-31", 12, 0.01)
    b = _series("2002-01-31", 12, 0.02)
    al = align_monthly(a, b, a_label="pub", b_label="corr")
    assert al.common_n_months == 12 == al.as_published_n_months == al.corrected_n_months
    pd_ = paired_difference(a, b, al, a_label="pub", b_label="corr")
    assert pd_.estimable and pd_.n_common == 12
    assert pd_.paired_mean_difference == pytest.approx(-0.01)  # 0.01 - 0.02


def test_one_endpoint_loses_months_paired_differs_from_natural() -> None:
    # corrected drops the first 3 months -> the paired estimate is on the 9 common months.
    a = _series("2002-01-31", 12, 0.01)
    b = pd.Series(
        np.arange(9, dtype=float) / 1000.0,
        index=pd.date_range("2002-04-30", periods=9, freq="ME"),
    )
    al = align_monthly(a, b, a_label="pub", b_label="corr")
    assert al.common_n_months == 9
    assert len(al.as_published_only_months) == 3  # the dropped months recorded
    pd_ = paired_difference(a, b, al, a_label="pub", b_label="corr")
    # natural-sample mean of a is over 12 months; paired mean of a is over 9 -> differ here.
    assert pd_.mean_a_natural != pd_.mean_a_common


def test_bond_membership_recorded_not_forced() -> None:
    a = _series("2002-01-31", 6, 0.01)
    b = _series("2002-01-31", 6, 0.02)
    counts_a = {ts: 100 for ts in a.index}
    counts_b = {ts: 137 for ts in b.index}  # different membership, same months
    al = align_monthly(a, b, a_label="pub", b_label="corr", bond_counts_a=counts_a, bond_counts_b=counts_b)
    # months still fully paired despite different bond counts (never forced to match).
    assert al.common_n_months == 6
    assert dict(al.bond_count_by_month_as_published)[a.index[0].strftime("%Y-%m-%d")] == 100
    assert dict(al.bond_count_by_month_corrected)[b.index[0].strftime("%Y-%m-%d")] == 137


def test_duplicate_month_labels_do_not_crash() -> None:
    # MAJOR-3 regression: a malformed input with a doubled month-end must not raise on
    # the paired reindex — it collapses defensively (keep-last) to a typed result.
    dup_idx = pd.to_datetime(["2002-01-31", "2002-01-31", "2002-02-28", "2002-03-31"])
    a = pd.Series([0.01, 0.02, 0.03, 0.04], index=dup_idx)
    b = _series("2002-01-31", 3, 0.005)
    al = align_monthly(a, b, a_label="pub", b_label="corr")
    pd_ = paired_difference(a, b, al, a_label="pub", b_label="corr")  # must not raise
    assert pd_.estimable and pd_.n_common == 3


def test_no_overlap_refuses() -> None:
    a = _series("2002-01-31", 6)
    b = _series("2050-01-31", 6)
    al = align_monthly(a, b, a_label="pub", b_label="corr")
    assert al.common_n_months == 0
    pd_ = paired_difference(a, b, al, a_label="pub", b_label="corr")
    assert not pd_.estimable and pd_.refusal_code is RefusalCode.NO_COMMON_SAMPLE
    assert pd_.paired_mean_difference is None
    assert pd_.alignment is al  # record attached, not discarded (rule 3)
