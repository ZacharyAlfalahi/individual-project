"""canonical.py — the Reporter's single canonical-JSON contract (docs/reporter/reporter_spec_v0.2.md §10.2).

Every hashed or persisted Reporter artefact (notes, ledgers, registry rows, manifests)
is serialised through `canonical_json` / `canonical_hash` so that "same artefacts + same
Reporter code -> byte-identical output" (INV-8) holds. The contract:

  * UTF-8, NFC-normalised strings, lexicographic key order, `(",", ":")` separators.
  * ``allow_nan=False`` — a bare ``NaN`` token is invalid, non-portable JSON.
  * NaN is a deliberate first-class upstream value (near-zero-variance Sharpe, deflated
    Sharpe with n<2, zero-variance HAC coordinates). It is therefore ENCODED as the
    sentinel ``{"__nan__": true, "reason": <str|null>}``, never rejected (INV / §10.2).
  * ``inf`` is rejected — no upstream path produces it.
  * ``-0.0`` is normalised to ``0.0``.

This reuses the repo's canonical-JSON idiom (`agents/scientist/schemas/proposal_set.py::
_canonical_hash`) but adds the mandatory `allow_nan=False` and the NaN-sentinel pass that
that helper lacks. It does NOT reuse `agents/auditor/hashing.py` (raw-float-bit invariance
hashing — a different, rounding-intolerant purpose).
"""

from __future__ import annotations

import hashlib
import json
import math
import unicodedata
from dataclasses import dataclass


class CanonicalisationError(ValueError):
    """A value cannot be placed into the canonical JSON contract (e.g. an infinity, a
    non-string mapping key, or an unsupported type). Raised, never silently coerced."""


@dataclass(frozen=True)
class NanValue:
    """A first-class encoding of an upstream NaN with an optional reason string. Distinct
    from ``None`` (absence): a NaN metric is a produced-but-degenerate value, not a missing
    one. Serialises to the ``{"__nan__": true, "reason": ...}`` sentinel."""

    reason: str | None = None

    def __post_init__(self) -> None:
        if self.reason is not None and not isinstance(self.reason, str):
            raise TypeError(f"NanValue.reason must be str|None; got {self.reason!r}")


def _nfc(text: str) -> str:
    return unicodedata.normalize("NFC", text)


def _canonicalise(obj: object) -> object:
    """Recursively convert `obj` into a JSON-primitive tree obeying the contract. After
    this pass there are no NaN/inf floats and every string is NFC, so `json.dumps` with
    ``allow_nan=False`` is a pure guard."""
    if isinstance(obj, NanValue):
        reason = _nfc(obj.reason) if obj.reason is not None else None
        return {"__nan__": True, "reason": reason}
    # bool is an int subclass — test it first so True/False stay booleans.
    if isinstance(obj, bool):
        return obj
    if isinstance(obj, int):
        return obj
    if isinstance(obj, float):
        if math.isnan(obj):
            return {"__nan__": True, "reason": None}
        if math.isinf(obj):
            raise CanonicalisationError(
                "infinity is not encodable in the canonical contract "
                "(no upstream path produces it)"
            )
        # `-0.0 == 0.0` is True, so this normalises -0.0 -> 0.0.
        return 0.0 if obj == 0.0 else obj
    if isinstance(obj, str):
        return _nfc(obj)
    if obj is None:
        return None
    if isinstance(obj, dict):
        out: dict[str, object] = {}
        for key, value in obj.items():
            if not isinstance(key, str):
                raise CanonicalisationError(
                    f"non-string mapping key {key!r} — canonicalise the serialised "
                    "(to_dict) shape, whose keys are already strings"
                )
            key_nfc = _nfc(key)
            if key_nfc in out:
                raise CanonicalisationError(
                    f"two keys collapse to the same NFC form {key_nfc!r}; "
                    "one would be silently dropped"
                )
            out[key_nfc] = _canonicalise(value)
        return out
    if isinstance(obj, (list, tuple)):
        return [_canonicalise(value) for value in obj]
    raise CanonicalisationError(
        f"type {type(obj).__name__} is not canonicalisable"
    )


def canonical_json(obj: object) -> str:
    """The canonical, byte-stable JSON encoding of `obj` (no trailing newline)."""
    return json.dumps(
        _canonicalise(obj),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def canonical_hash(obj: object) -> str:
    """sha256 hex digest over `canonical_json(obj)`."""
    return hashlib.sha256(canonical_json(obj).encode("utf-8")).hexdigest()
