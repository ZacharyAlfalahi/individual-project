"""WS-A (P3 / SC-SCI-11) — gate invariance.

The posterior layer is PRESENTATION ONLY. Two properties prove it structurally:
(1) no Scientist gate module imports the posterior code (or the Reporter), so
no gate decision can depend on it; (2) `derive_outcome` — the single authority
for terminal outcomes — still maps the canonical boolean patterns to exactly
the pre-amendment outcomes (Appendix B, first-match-wins), pinned here."""

from __future__ import annotations

import dataclasses
from pathlib import Path

from agents.scientist.experimentalist.outcome import derive_outcome
from agents.scientist.schemas.evaluation import BOOLEAN_FIELDS, Booleans
from agents.scientist.schemas.outcomes import Outcome

_GATE_DIR = Path(__file__).resolve().parents[2] / "agents" / "scientist" / "experimentalist"


def _all_true() -> Booleans:
    return Booleans(**{f: True for f in BOOLEAN_FIELDS})


def test_no_gate_module_imports_the_posterior_layer():
    for path in sorted(_GATE_DIR.glob("*.py")):
        src = path.read_text(encoding="utf-8")
        assert "shared.stats.posterior" not in src, f"{path.name} imports the posterior layer"
        assert "posterior_summary" not in src, f"{path.name} references posterior_summary"
        assert "agents.reporter" not in src, f"{path.name} imports the Reporter"


def test_posterior_layer_does_not_import_the_scientist():
    src = (
        Path(__file__).resolve().parents[2] / "shared" / "stats" / "posterior.py"
    ).read_text(encoding="utf-8")
    assert "agents.scientist" not in src


def test_derive_outcome_decisions_pinned():
    """Appendix B verbatim, one flip per stage — byte-for-byte the pre-SC-SCI-11 map."""
    assert derive_outcome(_all_true()) is Outcome.HOLDOUT_EVALUATED

    expected = {
        "schema_valid": Outcome.INVALID_PROPOSAL,
        "mechanism_authorised": Outcome.INVALID_PROPOSAL,
        "template_supported": Outcome.INVALID_PROPOSAL,
        "toggles_preserved": Outcome.INVALID_PROPOSAL,
        "inputs_available": Outcome.INVALID_PROPOSAL,
        "not_duplicate": Outcome.INVALID_PROPOSAL,
        "compiled": Outcome.EXECUTION_FAILURE,
        "execution_verified": Outcome.EXECUTION_FAILURE,
        "audit_clean": Outcome.AUDIT_FAILURE,
        "bh_survived": Outcome.NO_DEVELOPMENT_EVIDENCE,
        "cpcv_qualified": Outcome.DEVELOPMENT_SURVIVOR_NOT_ADVANCED,
    }
    for field, outcome in expected.items():
        flipped = dataclasses.replace(_all_true(), **{field: False})
        assert derive_outcome(flipped) is outcome, field
