"""
hashing.py — bit-exact output hashing for the invariance gate (§3.5).

The invariance test proves "different config, identical behaviour": it asserts the
RunConfig hashes DIFFER while the OUTPUT hashes are IDENTICAL. For that to mean
"bit-exact behaviour" the output hash must encode the raw float bits, not a
rounded or repr-based form — a no-op that is inert only after rounding is not a
proven no-op. So series are hashed over their int64-nanosecond index bytes and
float64 value bytes.

Shared by `checks.cell_runner` (which computes the hashes) and `checks.invariance`
(which compares them). Kept separate from the schemas so the data types stay pure.
"""

from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd


def hash_series(series: pd.Series) -> str:
    """SHA-256 over a date-indexed numeric series, exact to the float bits.

    The series is sorted by index; the index is encoded as int64 nanoseconds and
    the values as float64, both little-endian. NaN is preserved (it has a fixed
    bit pattern), so two series that are NaN in the same places and equal
    elsewhere hash identically, and any real difference changes the digest.
    """
    if not isinstance(series, pd.Series):
        raise TypeError(f"hash_series expects a pd.Series; got {type(series).__name__}")
    s = series.sort_index()
    idx = pd.DatetimeIndex(s.index)
    # int64 ns since epoch — deterministic, tz-naive (the pipeline is tz-naive).
    idx_bytes = idx.view("int64").astype("<i8").tobytes()
    val_bytes = np.asarray(s.to_numpy(), dtype="<f8").tobytes()
    h = hashlib.sha256()
    h.update(b"idx")
    h.update(idx_bytes)
    h.update(b"val")
    h.update(val_bytes)
    return h.hexdigest()


def hash_metrics(metrics: dict, keys: tuple[str, ...]) -> str:
    """SHA-256 over selected metric values in a fixed key order, exact to the
    float bits. `keys` pins which metrics enter the digest (so adding a
    descriptive field later cannot change an invariance verdict). A NaN metric
    hashes stably; a None (e.g. absent date) is encoded distinctly from NaN."""
    h = hashlib.sha256()
    for key in keys:
        if key not in metrics:
            raise KeyError(f"hash_metrics: metric {key!r} absent from {sorted(metrics)}")
        val = metrics[key]
        h.update(key.encode("utf-8"))
        if val is None:
            h.update(b"\x00none")
        elif isinstance(val, (int, float)) and not isinstance(val, bool):
            h.update(b"\x01")
            h.update(np.asarray([val], dtype="<f8").tobytes())
        else:
            # Fallback for non-numeric metrics (e.g. an int count already covered
            # above); encode the repr so a change is still detected.
            h.update(b"\x02")
            h.update(repr(val).encode("utf-8"))
    return h.hexdigest()
