"""Shared Reporter test fixtures (not collected — leading underscore).

Builds a `ReportBundle` directly from serialised docs, so renderer/verifier/registry tests do
not have to round-trip through the loader. Reuses recorded artefacts where available.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from agents.reporter.bundle import (  # noqa: E402
    STAGES,
    ArtefactRef,
    CodeVersion,
    GitStamp,
    HoldoutView,
    ProposalReport,
    ReportBundle,
    StageEvidence,
    StageRecord,
    StageStatus,
    UpstreamStamps,
    derive_reportability,
    load_phase_stacks,
)
from shared.reporting.canonical import canonical_hash  # noqa: E402
from shared.reporting.claims import ArtefactType  # noqa: E402

_PHASE_F_IDS, _PHASE_D_IDS = load_phase_stacks()


def _fixture_model_ids(phase: str) -> str:
    """The recorded model-id string a fixture stamps: the configured stack matching its phase, so
    a phase-F fixture derives REPORTABLE and a phase-D fixture NON_REPORTABLE_PHASE_D."""
    return ",".join(sorted(_PHASE_F_IDS if phase == "F" else _PHASE_D_IDS))

DRF_CORE = _REPO / "results/auditor/drf/drf_core.json"
DRF_QUANT = _REPO / "results/quant/drf/drf.json"

# A synthetic StrategySpec.to_dict()-shaped doc with paper_facts (real specs are not always in
# a checkout). Shape matches the pointers the renderer authors.
SPEC_DOC = {
    "header": {"paper_id": "BBW", "strategy_label": {"value": "drf", "tag": "STATED"}},
    "part1": {
        "asset_class": {"value": "corporate bonds", "tag": "STATED"},
        "formation_structure": {"value": "long-short decile", "tag": "STATED"},
    },
    "paper_facts": {
        "claimed_headline_metric": {
            "value": {"mean": 0.0031, "t_stat": 3.2, "unit": "decimal_monthly"},
            "tag": "STATED",
        }
    },
}


def load_real(path: Path) -> dict:
    if not path.exists():
        import pytest
        pytest.skip(f"results fixture not shipped with the repository: {path}")
    return json.loads(path.read_text())


def _artefact_ref(artefact_type: ArtefactType, doc: dict) -> ArtefactRef:
    return ArtefactRef(
        key=artefact_type.value,
        path=f"fixture/{artefact_type.value}.json",
        sha256=canonical_hash(doc),
        schema_version="test",
        artefact_type=artefact_type,
        producing_component="fixture",
    )


def make_bundle(
    *,
    docs: dict,
    stage_status: dict[str, StageStatus] | None = None,
    phase: str = "D",
    join: str = "asserted by test; no mechanical join exists",
    run_id: str = "bbw_drf",
    paper: str = "BBW",
    strategy: str = "drf",
    proposals: tuple = (),
) -> ReportBundle:
    stage_status = stage_status or {}
    stages = {}
    for stage in STAGES:
        status = stage_status.get(stage, StageStatus.UNOBSERVED)
        stages[stage] = StageRecord(
            stage=stage,
            status=status,
            evidence=StageEvidence(kind="fixture", detail=f"{stage} {status.value}"),
        )
    artefacts = tuple(_artefact_ref(t, d) for t, d in docs.items())
    return ReportBundle(
        reporter_run_id=run_id,
        paper_id=paper,
        strategy_id=strategy,
        phase=phase,
        docs=docs,
        stages=stages,
        reportability=derive_reportability(
            phase,
            _fixture_model_ids(phase),
            phase_f_ids=_PHASE_F_IDS,
            phase_d_ids=_PHASE_D_IDS,
        ),
        artefacts=artefacts,
        upstream_stamps=UpstreamStamps(
            quant=GitStamp(full="a" * 40, short="aaaaaaa", source_form="full+short"),
            audit=GitStamp(full=None, short="1111111", source_form="short"),
            auditor_prereg_tag="auditor-prereg",
        ),
        code_version=CodeVersion(full="reporterfullsha", short="reporter"),
        join_rationale=join,
        proposals=proposals,
    )


def make_proposal(
    *,
    proposal_id: str = "p1",
    outcome: str = "HOLDOUT_EVALUATED",
    refusal: str | None = None,
    mechanism_ref: str = "mech_008",
    gate: str = "G6",
    diagnostics: dict | None = None,
    holdout: HoldoutView | None = None,
) -> ProposalReport:
    return ProposalReport(
        record={
            "proposal_id": proposal_id,
            "final_outcome": outcome,
            "refusal_code": refusal,
        },
        gate_outcomes=({"gate": gate, "passed": True},),
        proposal={"proposal_id": proposal_id, "mechanism_ref": mechanism_ref},
        diagnostics=diagnostics,
        holdout=holdout,
    )


def extension_bundle(**kw) -> ReportBundle:
    """A bundle for the EXTENSION_PATH disposition (scientist SUCCEEDED) with one proposal that
    reaches holdout, crowding diagnostics, and a real mechanism ref."""
    docs = {
        ArtefactType.STRATEGY_SPEC: SPEC_DOC,
        ArtefactType.STRATEGY_RESULT: load_real(DRF_QUANT),
        ArtefactType.AUDIT_REPORT: load_real(DRF_CORE),
    }
    stage_status = {
        "extraction": StageStatus.SUCCEEDED,
        "compilation": StageStatus.SUCCEEDED,
        "execution": StageStatus.SUCCEEDED,
        "audit": StageStatus.SUCCEEDED,
        "scientist": StageStatus.SUCCEEDED,
    }
    proposals = (
        make_proposal(
            diagnostics={"alpha": 0.0012, "alpha_t": 2.4},
            holdout=HoldoutView(
                sharpe_sign=1,
                sharpe_ci_low=0.10,
                sharpe_ci_high=0.60,
                paired_difference=0.02,
            ),
        ),
    )
    return make_bundle(
        docs=docs, stage_status=stage_status, proposals=proposals, **kw
    )


def audited_replication_bundle(**kw) -> ReportBundle:
    """A bundle for the AUDIT_COMPLETE_NO_EXTENSION disposition from real quant + audit cores."""
    docs = {
        ArtefactType.STRATEGY_SPEC: SPEC_DOC,
        ArtefactType.STRATEGY_RESULT: load_real(DRF_QUANT),
        ArtefactType.AUDIT_REPORT: load_real(DRF_CORE),
    }
    stage_status = {
        "extraction": StageStatus.SUCCEEDED,
        "compilation": StageStatus.SUCCEEDED,
        "execution": StageStatus.SUCCEEDED,
        "audit": StageStatus.SUCCEEDED,
    }
    return make_bundle(docs=docs, stage_status=stage_status, **kw)
