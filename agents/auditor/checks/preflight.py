"""
preflight.py — executability gate and derived audit scope (§3.3-3.4).

Two jobs, both deterministic and engine-free:

1. `is_runnable` encodes the FOUR conditions for `runnable: true` (§3.3). ALL four
   must hold. In-sample exposure is deliberately NOT among them — v1.5's fifth
   condition was withdrawn (D-A30): "the required data exist but no event occurs
   in the sample" is a MEASURED no-op (runnable: true, expect_no_op: true), not an
   unmeasurable toggle. Re-adding a fifth key is rejected, so the deleted
   condition cannot creep back.

2. `derive_scope` turns the per-toggle `ToggleFacts` into the STRATEGY-level
   derived `audit_scope` (§3.4) — COMPLETE / PARTIAL / REFUSED — plus the
   conditioning signature and statement that must travel with every number a
   PARTIAL audit produces. Deriving it (rather than declaring it per toggle)
   makes an inconsistent scope unrepresentable (D-A35).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

from ..schemas.audit_core import AuditScope, conditioning_statement_for
from ..schemas.toggle import TOGGLE_IDS, ToggleFacts, ToggleId, ToggleState

# The four conditions for runnable: true (§3.3). All must hold.
FOUR_RUNNABLE_CONDITIONS: tuple[str, ...] = (
    "both_states_materialisable",     # 1. both states materialisable from available data
    "off_state_known",                # 2. the as-published (OFF) state is known
    "both_states_engine_supported",   # 3. both states supported by the engine
    "differential_interpretable",     # 4. the differential is interpretable from provenance
)


def is_runnable(condition_results: Mapping[str, bool]) -> bool:
    """True iff all four §3.3 conditions hold. Raises if a condition is missing or
    an unknown one is supplied — the latter guards against re-introducing the
    withdrawn 'in-sample exposure' fifth condition (D-A30)."""
    keys = set(condition_results)
    expected = set(FOUR_RUNNABLE_CONDITIONS)
    if keys != expected:
        extra = keys - expected
        missing = expected - keys
        raise ValueError(
            "runnable conditions must be exactly the four §3.3 predicates "
            f"{FOUR_RUNNABLE_CONDITIONS}; "
            + (f"unknown: {sorted(extra)} " if extra else "")
            + (f"missing: {sorted(missing)}" if missing else "")
        )
    return all(bool(condition_results[c]) for c in FOUR_RUNNABLE_CONDITIONS)


@dataclass(frozen=True)
class PreflightResult:
    """The strategy-level pre-flight verdict."""

    strategy_label: str
    audit_scope: AuditScope
    runnable_toggles: tuple[ToggleId, ...]
    fixed_states: Mapping[ToggleId, ToggleState]      # held non-runnable toggles
    conditioning_signature: tuple[tuple[ToggleId, ToggleState], ...]
    conditioning_statement: str | None
    facts: Mapping[ToggleId, ToggleFacts]
    refused_toggles: tuple[ToggleId, ...]             # non-runnable with no fixed_state

    @property
    def is_refused(self) -> bool:
        return self.audit_scope == "REFUSED"


def derive_scope(
    strategy_label: str, facts: Sequence[ToggleFacts]
) -> PreflightResult:
    """Derive `audit_scope` and the conditioning from the five toggle facts.

        COMPLETE  iff every registered toggle is runnable
        PARTIAL   iff every non-runnable toggle has a defensible fixed_state
        REFUSED   iff any non-runnable toggle lacks a defensible fixed_state

    Requires exactly one fact per registered toggle (§3.1's five). The
    conditioning signature is the sorted (canonical-order) list of held
    non-runnable toggles and their fixed states — empty for COMPLETE.
    """
    by_id: dict[ToggleId, ToggleFacts] = {}
    for f in facts:
        if f.toggle_id in by_id:
            raise ValueError(f"duplicate ToggleFacts for {f.toggle_id!r}")
        by_id[f.toggle_id] = f
    if set(by_id) != set(TOGGLE_IDS):
        missing = set(TOGGLE_IDS) - set(by_id)
        extra = set(by_id) - set(TOGGLE_IDS)
        raise ValueError(
            "derive_scope needs exactly one fact per registered toggle "
            f"{TOGGLE_IDS}; missing={sorted(missing)} extra={sorted(extra)}"
        )

    runnable = tuple(t for t in TOGGLE_IDS if by_id[t].runnable)
    non_runnable = [t for t in TOGGLE_IDS if not by_id[t].runnable]

    refused = tuple(t for t in non_runnable if by_id[t].fixed_state is None)
    held = {t: by_id[t].fixed_state for t in non_runnable if by_id[t].fixed_state is not None}

    if refused:
        scope: AuditScope = "REFUSED"
    elif non_runnable:
        scope = "PARTIAL"
    else:
        scope = "COMPLETE"

    signature = tuple(
        (t, by_id[t].fixed_state)  # type: ignore[misc]
        for t in TOGGLE_IDS
        if t in held
    )
    statement = conditioning_statement_for(signature)

    return PreflightResult(
        strategy_label=strategy_label,
        audit_scope=scope,
        runnable_toggles=runnable,
        fixed_states=held,  # type: ignore[arg-type]
        conditioning_signature=signature,
        conditioning_statement=statement,
        facts=by_id,
        refused_toggles=refused,
    )
