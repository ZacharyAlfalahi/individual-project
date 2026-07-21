"""hash_series/hash_metrics hardening (invariance surface)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from agents.auditor.hashing import hash_metrics, hash_series


def _dt(n):
    return pd.date_range("2005-01-31", periods=n, freq="ME")


# --------------------------------------------------------------------------
# Fail loud on a non-DatetimeIndex (silent-wrong-hash prevention)
# --------------------------------------------------------------------------

def test_hash_series_rejects_range_index():
    s = pd.Series([0.1, 0.2, 0.3])  # RangeIndex — would coerce to ns timestamps
    with pytest.raises(TypeError, match="DatetimeIndex"):
        hash_series(s)


def test_hash_series_rejects_period_index():
    s = pd.Series([0.1, 0.2], index=pd.period_range("2005-01", periods=2, freq="M"))
    with pytest.raises(TypeError, match="DatetimeIndex"):
        hash_series(s)


def test_hash_series_accepts_empty_datetime_index():
    s = pd.Series([], dtype=float, index=pd.DatetimeIndex([]))
    assert isinstance(hash_series(s), str)  # empty is a valid cell (no returns)


# --------------------------------------------------------------------------
# Signed-zero normalisation (removes a latent spurious "not a no-op")
# --------------------------------------------------------------------------

def test_hash_series_treats_negative_zero_as_zero():
    idx = _dt(3)
    pos = pd.Series([0.0, 0.1, 0.0], index=idx)
    neg = pd.Series([-0.0, 0.1, -0.0], index=idx)
    assert hash_series(pos) == hash_series(neg)


def test_hash_metrics_treats_negative_zero_as_zero():
    keys = ("a", "b")
    assert hash_metrics({"a": -0.0, "b": 1.0}, keys) == hash_metrics({"a": 0.0, "b": 1.0}, keys)


# --------------------------------------------------------------------------
# NaN handling unchanged: stable, and distinct from None
# --------------------------------------------------------------------------

def test_hash_series_nan_stable_and_detects_change():
    idx = _dt(3)
    a = pd.Series([0.1, np.nan, 0.3], index=idx)
    b = pd.Series([0.1, np.nan, 0.3], index=idx)
    c = pd.Series([0.1, np.nan, 0.3000001], index=idx)
    assert hash_series(a) == hash_series(b)
    assert hash_series(a) != hash_series(c)


def test_hash_metrics_nan_distinct_from_none():
    assert hash_metrics({"x": float("nan")}, ("x",)) != hash_metrics({"x": None}, ("x",))
