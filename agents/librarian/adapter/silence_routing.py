"""
Silence routing (D26): what a field's *provenance* means at adapter intake.

Given a field's ``FieldPolicy`` (from the silence-policy table) and its
``Inherited`` from the spec, decide one of four outcomes:

  * ``Proceed(inherited)`` -- a real, non-silent value; run the transform.
  * ``OmitField()``        -- silent under ``tag_and_proceed``: omit the kwarg so
                              the FACTORY fills + tags its documented default (P5 --
                              the adapter never learns the default value).
  * ``RefuseField(code, detail)`` -- a HARD refusal (strategy refuses):
        - ``REVIEW_REQUIRED``    -- an UNKNOWN whose reason is a review reason
                                    (disagreement / quote_match_failure /
                                    single_response) -> the manual-review lane (D26:
                                    never a default; the paper likely states it).
        - ``REFUSED_ON_SILENCE`` -- silent under a ``refuse`` policy (load-bearing).
        - ``ASSUMPTION_MISMATCH``-- a STATED value in ``refuse_on_stated``: the
                                    engine cannot represent it (a LIMIT conflict,
                                    D23 refuse-on-conflict; e.g. strategy_side=long_only).
  * ``FlagField(detail)``  -- a SOFT flag (strategy proceeds, review-flagged): a
                              STATED value in ``flag_on_stated`` (e.g.
                              transaction_cost=net_of_costs).

Conditional policies (v1's only one: ``sort_kind``) are resolved on control
presence before routing.

This module is data-only: it reads ``Inherited.tag`` /
``Inherited.evidence.unknown_reason`` and the ``FieldPolicy``; it never touches a
transform or the engine.
"""

from __future__ import annotations

from dataclasses import dataclass

from agents.quant.config import Inherited, RefusalCode

from ..registries.silence_policy import FieldPolicy

# The UNKNOWN reasons that route to the manual-review lane, never a default (D26).
# not_stated / input_unknown are *silence* (routed by the field's policy);
# these three signal extraction uncertainty and always go to review.
REVIEW_REASONS: frozenset[str] = frozenset(
    ("disagreement", "quote_match_failure", "single_response")
)


@dataclass(frozen=True)
class Proceed:
    """A real value; run the field's transform."""

    inherited: Inherited


@dataclass(frozen=True)
class OmitField:
    """Omit the kwarg; the factory fills + tags its documented default (P5)."""


@dataclass(frozen=True)
class RefuseField:
    """A hard refusal -- the strategy refuses (D28)."""

    code: RefusalCode
    detail: str


@dataclass(frozen=True)
class FlagField:
    """A soft review flag -- the strategy proceeds but is flagged for review."""

    detail: str


Routed = Proceed | OmitField | RefuseField | FlagField


def route_field(
    policy: FieldPolicy, inherited: Inherited, *, control_present: bool
) -> Routed:
    """Route one field's ``Inherited`` through its silence policy (D26)."""
    # Resolve a conditional policy on structural context first (v1: sort_kind).
    if policy.is_conditional:
        context = "when_control_present" if control_present else "when_no_control"
        policy = policy.resolve(context)

    if inherited.tag == "UNKNOWN":
        reason = inherited.evidence.unknown_reason
        if reason in REVIEW_REASONS:
            return RefuseField(
                RefusalCode.REVIEW_REQUIRED,
                f"{policy.field}: extraction needs review (unknown_reason={reason!r})",
            )
        # not_stated / input_unknown / unset -> the paper is silent; the policy decides.
        if policy.policy == "refuse":
            return RefuseField(
                RefusalCode.REFUSED_ON_SILENCE,
                policy.reason or f"{policy.field}: paper silent on a load-bearing field",
            )
        # tag_and_proceed / flag: proceed on silence -> omit (factory default).
        return OmitField()

    # A real (non-silent) value: honour the D23 refuse-on-conflict overrides.
    value = inherited.value
    if value in policy.refuse_on_stated:
        return RefuseField(
            RefusalCode.ASSUMPTION_MISMATCH,
            f"{policy.field}: STATED {value!r} is not engine-representable "
            f"({policy.note or policy.reason or 'engine LIMIT'})",
        )
    if value in policy.flag_on_stated:
        return FlagField(
            f"{policy.field}: STATED {value!r} flagged "
            f"({policy.note or 'the engine cannot honour it'})"
        )
    return Proceed(inherited)
