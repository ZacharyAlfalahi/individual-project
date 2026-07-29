"""
outcomes.py — the Scientist's terminal vocabularies (spec §5.4, Appendix A/B).

Three closed vocabularies, each with a documented deterministic consumer (INVARIANT 4 —
no enum without a named consumer):

  Verdict     — the entry-rule verdict per applicable toggle. Consumer: the handoff seam
                (`shared/handoff/scientist_case.py::apply_entry_rule`) and
                `ScientistCase.failed_check_verdicts`.
  Outcome     — the DERIVED terminal label for one proposal (Appendix B). Consumer:
                `agents/scientist/experimentalist/outcome.py::derive_outcome` and the
                economic funnel. Never assigned by a human or a model (D5).
  RefusalCode — the 13 gate-stack refusal codes (Appendix A). Consumer: each halts a
                proposal, is counted in the agent-quality funnel, and suppresses all
                downstream computation (`EvaluationRecord.refusal_code`).

These are labels only — there is no magnitude anywhere in this module (INVARIANT 1).
"""

from __future__ import annotations

from enum import Enum
from typing import Literal

# The per-toggle entry-rule verdict (D14 / R3). Kept a Literal (not an Enum) so it serialises
# as a bare string in the magnitude-free ScientistCase and reads identically to the allow-list
# in the spec (§3 "PASS / FAIL / REFUSED").
Verdict = Literal["PASS", "FAIL", "REFUSED"]
VERDICTS: tuple[str, ...] = ("PASS", "FAIL", "REFUSED")


class Outcome(str, Enum):
    """The six terminal outcomes (Appendix B). DERIVED from the booleans; declared in the
    first-match-wins precedence order of `derive_outcome` — the earliest branch is the earliest
    point of failure, and it always dominates a later one. `HOLDOUT_EVALUATED` is deliberately
    NOT a success label (a 36-month holdout does not license a binary verdict)."""

    INVALID_PROPOSAL = "INVALID_PROPOSAL"                                    # G0 integrity failed
    EXECUTION_FAILURE = "EXECUTION_FAILURE"                                  # G1a/G1b compile/execute failed
    AUDIT_FAILURE = "AUDIT_FAILURE"                                          # G2 not audit-clean
    NO_DEVELOPMENT_EVIDENCE = "NO_DEVELOPMENT_EVIDENCE"                      # G3 BH-FDR non-survivor
    DEVELOPMENT_SURVIVOR_NOT_ADVANCED = "DEVELOPMENT_SURVIVOR_NOT_ADVANCED"  # G4 CPCV not qualified
    HOLDOUT_EVALUATED = "HOLDOUT_EVALUATED"                                  # reached G6 (NOT "success")


class RefusalCode(str, Enum):
    """The 13 Appendix A refusal codes, exact names, grouped by the gate that emits them (§9).
    Each halts its proposal, is counted, and suppresses downstream computation."""

    # G0 — proposal integrity
    INVALID_SCHEMA = "INVALID_SCHEMA"
    UNKNOWN_MECHANISM = "UNKNOWN_MECHANISM"
    UNSUPPORTED_TEMPLATE = "UNSUPPORTED_TEMPLATE"
    FIELD_OUT_OF_DOMAIN = "FIELD_OUT_OF_DOMAIN"
    FORBIDDEN_CHANGE = "FORBIDDEN_CHANGE"
    TOGGLE_REVERSAL = "TOGGLE_REVERSAL"
    MISSING_INPUT = "MISSING_INPUT"
    DUPLICATE_PROPOSAL = "DUPLICATE_PROPOSAL"
    # G1a / G1b — compilation & execution fidelity
    COMPILATION_FAILED = "COMPILATION_FAILED"
    EXECUTION_MISMATCH = "EXECUTION_MISMATCH"
    # G2 — audit-clean (lag / parameter-provenance)
    NEW_TIMING_VIOLATION = "NEW_TIMING_VIOLATION"
    PARAMETER_PROVENANCE_VIOLATION = "PARAMETER_PROVENANCE_VIOLATION"
    # G6 — holdout
    HOLDOUT_VIOLATION = "HOLDOUT_VIOLATION"


REFUSAL_CODES: tuple[str, ...] = tuple(c.value for c in RefusalCode)
