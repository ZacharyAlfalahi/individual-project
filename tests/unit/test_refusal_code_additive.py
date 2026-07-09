"""
Regression guard for the D29 additive RefusalCode edit (the ONE frozen-code change
landing with the adapter).

The edit adds four members (ASSUMPTION_MISMATCH promoted + REVIEW_REQUIRED,
REFUSED_ON_SILENCE, UNSUPPORTED_COMBINER) to ``RefusalCode``. This proves the edit
is purely additive:

  * the four original members keep their exact ``.value`` strings;
  * every existing ``ConfigRefusal`` serialises byte-identically (the new members
    are emitted only by the ADAPTER, never by the factory);
  * ``build_quant_config`` still emits ONLY the four original codes.
"""

from __future__ import annotations

from agents.quant.config import (
    Binding,
    Evidence,
    Inherited,
    Locator,
    RefusalCode,
    build_quant_config,
)
from agents.quant.config.refusal import ConfigRefusal

_ORIGINAL = {
    "MISSING_BINDING",
    "OUT_OF_ENUM_WEIGHTING",
    "UNSUPPORTED_TRIM_VARIANT",
    "UNSUPPORTED_COMBINATION",
}
_ADDED = {
    "ASSUMPTION_MISMATCH",
    "REVIEW_REQUIRED",
    "REFUSED_ON_SILENCE",
    "UNSUPPORTED_COMBINER",
}


def _loc():
    return Locator(1, 0, 1)


def _bound(col):
    return Binding(col, "BOUND", Evidence(column=col))


def test_original_members_unchanged():
    # The four originals keep their exact value strings (a rename would break every
    # recorded RQ2 refusal).
    for name in _ORIGINAL:
        assert RefusalCode[name].value == name


def test_added_members_present():
    for name in _ADDED:
        assert RefusalCode[name].value == name


def test_no_stray_members():
    # Exactly the original four + the four added -- nothing else crept in.
    assert {c.value for c in RefusalCode} == _ORIGINAL | _ADDED


def test_existing_refusal_serialises_byte_identically():
    # A canonical existing refusal (a MISSING score binding) serialises exactly as
    # before the edit -- the new members are never emitted here.
    r = ConfigRefusal(
        "strat_x",
        RefusalCode.MISSING_BINDING,
        "score",
        "the strategy signal has no corresponding column in the panel",
        Evidence(note="searched panel columns"),
    )
    assert r.to_dict() == {
        "strategy_id": "strat_x",
        "code": "MISSING_BINDING",
        "field": "score",
        "detail": "the strategy signal has no corresponding column in the panel",
        "evidence": {"note": "searched panel columns"},
    }


def test_factory_emits_only_original_codes():
    # Drive build_quant_config through its four refusal paths; every code it can
    # emit is an ORIGINAL member (the four new codes are adapter-only).
    good = Inherited(5, "STATED", Evidence(quote="q", locator=_loc()))
    seen: set[str] = set()

    # MISSING_BINDING: a MISSING score.
    r1 = build_quant_config("s", Binding(None, "MISSING", Evidence(note="none")))
    seen.add(r1.code.value)

    # OUT_OF_ENUM_WEIGHTING: an unrepresentable weighting.
    r2 = build_quant_config(
        "s", _bound("mom6"),
        weighting=Inherited("market_value", "INFERRED", Evidence(rule_id="w")),
    )
    seen.add(r2.code.value)

    # UNSUPPORTED_TRIM_VARIANT: a percentile trim.
    r3 = build_quant_config(
        "s", _bound("mom6"),
        trim=Inherited({"method": "truncate", "bounds": {"type": "percentile"}}, "STATED",
                       Evidence(quote="q", locator=_loc())),
    )
    seen.add(r3.code.value)

    # UNSUPPORTED_COMBINATION: control + holding_period > 1.
    r4 = build_quant_config(
        "s", _bound("mom6"), control=_bound("rating"), holding_period=good,
    )
    # holding_period=5 to trip the combination.
    r4 = build_quant_config(
        "s", _bound("mom6"), control=_bound("rating"),
        holding_period=Inherited(5, "STATED", Evidence(quote="q", locator=_loc())),
    )
    seen.add(r4.code.value)

    assert seen <= _ORIGINAL, f"factory emitted a non-original code: {seen - _ORIGINAL}"
