"""words.py — direction and verdict words as total functions (INV-6).

Every direction word ("higher"/"lower"), verdict word ("passed"/"failed"/"refused") and
scope word is RETURNED by a total function keyed on a stored sign, flag or enum, with an
explicit failure branch. No renderer template string contains a direction word — a swapped
sign therefore cannot silently produce the wrong sentence; it either selects the right word
or raises. A NaN has no defined direction and raises rather than guessing.
"""

from __future__ import annotations

import math

from shared.reporting.canonical import NanValue

_METRIC_DIRECTION: dict[int, str] = {1: "higher", 0: "unchanged", -1: "lower"}
_EFFECT_DIRECTION: dict[int, str] = {1: "raises", 0: "leaves unchanged", -1: "lowers"}
_SIGN_WORD: dict[int, str] = {1: "positive", 0: "zero", -1: "negative"}

_SCOPE_WORDS: dict[str, str] = {
    "COMPLETE": "complete",
    "PARTIAL": "partial",
    "REFUSED": "refused",
}
_VERDICT_WORDS: dict[str, str] = {
    "PASS": "passed",
    "FAIL": "failed",
    "REFUSED": "refused",
}


def _sign(value: object) -> int:
    """The sign of a real number as -1/0/+1. A NaN has no direction and raises (INV-6)."""
    if isinstance(value, NanValue) or (
        isinstance(value, float) and math.isnan(value)
    ):
        raise ValueError("no direction word is defined for NaN")
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"_sign expects a real number; got {value!r}")
    if value > 0:
        return 1
    if value < 0:
        return -1
    return 0


def metric_direction_word(signed_value: object) -> str:
    """"higher" / "unchanged" / "lower" for a signed level change."""
    return _METRIC_DIRECTION[_sign(signed_value)]


def effect_direction_word(signed_value: object) -> str:
    """"raises" / "leaves unchanged" / "lowers" for a signed effect on a metric."""
    return _EFFECT_DIRECTION[_sign(signed_value)]


def sign_word(signed_value: object) -> str:
    """"positive" / "zero" / "negative"."""
    return _SIGN_WORD[_sign(signed_value)]


def audit_scope_word(scope: str) -> str:
    """The lowercase adjective for an `AuditScope` literal. Unknown scope raises (INV-6)."""
    try:
        return _SCOPE_WORDS[scope]
    except KeyError:
        raise ValueError(f"unknown audit scope {scope!r}") from None


def verdict_word(verdict: str) -> str:
    """The past-tense word for a PASS/FAIL/REFUSED verdict. Unknown verdict raises."""
    try:
        return _VERDICT_WORDS[verdict]
    except KeyError:
        raise ValueError(f"unknown verdict {verdict!r}") from None
