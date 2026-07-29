"""
outcome.py — the DERIVED terminal-label function (spec Appendix B; INVARIANT 4 / D5).

`derive_outcome` is the single authority for a proposal's terminal `Outcome`. It is
deterministic and FIRST-MATCH-WINS: the earliest unsatisfied precondition names the failure,
so an earlier gate's failure always dominates a later one (e.g. a proposal that neither
compiled nor was audit-clean is an EXECUTION_FAILURE, never an AUDIT_FAILURE). The outcome is
a pure function of the recorded booleans and nothing else — never assigned by a human or a
model. This is what makes the outcome taxonomy a strengthening (D5) rather than a place a
degree of freedom can hide.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..schemas.outcomes import Outcome

if TYPE_CHECKING:  # import for typing only — keeps this module free of a runtime dependency on
    from ..schemas.evaluation import Booleans  # evaluation.py (which imports THIS module).


def derive_outcome(b: "Booleans") -> Outcome:
    """Appendix B, verbatim. First matching condition wins."""
    if not (
        b.schema_valid
        and b.mechanism_authorised
        and b.template_supported
        and b.toggles_preserved
        and b.inputs_available
        and b.not_duplicate
    ):
        return Outcome.INVALID_PROPOSAL
    if not (b.compiled and b.execution_verified):
        return Outcome.EXECUTION_FAILURE
    if not b.audit_clean:
        return Outcome.AUDIT_FAILURE
    if not b.bh_survived:
        return Outcome.NO_DEVELOPMENT_EVIDENCE
    if not b.cpcv_qualified:
        return Outcome.DEVELOPMENT_SURVIVOR_NOT_ADVANCED
    return Outcome.HOLDOUT_EVALUATED
