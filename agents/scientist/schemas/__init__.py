"""
agents.scientist.schemas — the Scientist data contracts (spec §5).

Frozen dataclasses everywhere, EXCEPT Pydantic v2 at the single `ExtensionProposal` decode
boundary (R6). None of these modules imports a magnitude-bearing Auditor schema (the wall,
R2) — the only Auditor reuse is toggle *identities* (`ToggleId`) in `case.py`.
"""

from __future__ import annotations

from .case import DevelopmentWindow, HoldoutStatus, ScientistCase
from .evaluation import (
    BOOLEAN_FIELDS,
    Booleans,
    EvaluationRecord,
    GrossMeasurements,
    Measurements,
    NetMeasurements,
)
from .outcomes import REFUSAL_CODES, VERDICTS, Outcome, RefusalCode, Verdict
from .proposal import (
    ConfigDelta,
    ExtensionProposal,
    Generation,
    ProposalDecode,
    ProposalSource,
    decode_proposal,
)
from .proposal_set import ProposalSet

__all__ = [
    # outcomes
    "Verdict",
    "VERDICTS",
    "Outcome",
    "RefusalCode",
    "REFUSAL_CODES",
    # case
    "ScientistCase",
    "DevelopmentWindow",
    "HoldoutStatus",
    # proposal (decode boundary)
    "ExtensionProposal",
    "ConfigDelta",
    "Generation",
    "ProposalSource",
    "ProposalDecode",
    "decode_proposal",
    # proposal set
    "ProposalSet",
    # evaluation
    "Booleans",
    "BOOLEAN_FIELDS",
    "Measurements",
    "GrossMeasurements",
    "NetMeasurements",
    "EvaluationRecord",
]
