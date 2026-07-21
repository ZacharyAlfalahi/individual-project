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
    # The invariance surface must never hash a mis-typed index: a plain integer index
    # would be silently coerced to nanosecond timestamps (a silent wrong hash) and a
    # PeriodIndex would crash opaquely. Fail loud instead (an empty DatetimeIndex is fine).
    if not isinstance(s.index, pd.DatetimeIndex):
        raise TypeError(
            f"hash_series requires a DatetimeIndex; got {type(s.index).__name__}. "
            "Convert a PeriodIndex via .to_timestamp() upstream."
        )
    # int64 ns since epoch — deterministic, tz-naive (the pipeline is tz-naive).
    idx_bytes = s.index.view("int64").astype("<i8").tobytes()
    # `+ 0.0` normalises -0.0 to 0.0 so two numerically identical runs that differ only
    # in the sign of a zero are not judged non-invariant (NaN is unaffected).
    val_bytes = (np.asarray(s.to_numpy(), dtype="<f8") + 0.0).tobytes()
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
            # `+ 0.0` normalises -0.0 to 0.0 (int/float both hash via float64, so 240
            # and 240.0 unify, which is the desired invariance behaviour).
            h.update((np.asarray([val], dtype="<f8") + 0.0).tobytes())
        else:
            # Fallback for a non-numeric, non-None metric (e.g. a string label);
            # encode the repr so a change is still detected.
            h.update(b"\x02")
            h.update(repr(val).encode("utf-8"))
    return h.hexdigest()
