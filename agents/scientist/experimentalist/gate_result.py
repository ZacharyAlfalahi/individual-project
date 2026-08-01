"""`GateOutcome` — the structured result of one Experimentalist gate (spec §9).

Gates run in order; each returns a GateOutcome carrying the boolean fields it set (a subset of the
twelve §5.4 booleans), whether it `passed`, and — on failure — the FIRST failing check's
`RefusalCode`. A gate that fails stops the proposal and no performance is computed (prohibition 9).
The orchestrator assembles the full `Booleans` + `EvaluationRecord` from the per-gate outcomes.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..schemas.outcomes import RefusalCode


@dataclass(frozen=True)
class GateOutcome:
    gate: str                                  # "G0" | "G1a" | ...
    passed: bool
    booleans: dict = field(default_factory=dict)   # {boolean_field_name: bool} this gate set
    refusal_code: RefusalCode | None = None    # the FIRST failing check's code (None if passed)
