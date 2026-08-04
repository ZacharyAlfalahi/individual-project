"""bundle.py — R2 ReportBundle, stage records, loader (docs/reporter/reporter_spec_v0.2.md §5, as amended by C9).

The loader copies, hashes and validates; it performs no arithmetic (INV-2). A stage status is
assigned ONLY from present evidence — a valid output artefact, a typed refusal, an explicit
failure record — never inferred from silence (§5.2). Everything else is `UNOBSERVED`.

DESIGN (C9, verified 2026-08-03): the pipeline persists only serialised artefacts, and no type
except the Pydantic `ExtensionProposal` is reconstructable from JSON. The bundle therefore holds
the PARSED SERIALISED DICTS (each equal to the producing type's `to_dict()` output), not
reconstructed dataclasses. Pointers resolve against these dicts. The persisted per-anchor audit
artefact is `AuditCore.to_dict()` only (no fdr/economic/bootstrap layers), so the audit section
binds from the AuditCore keys; `render_report`/`build_scientist_case` are fixture-only paths
(see routing.py). From disk the Scientist stage is `UNOBSERVED` unless the pointer declares it.
"""

from __future__ import annotations

import json
import re
import subprocess
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Mapping

from shared.reporting.claims import ArtefactType

from .manifest import REPO_ROOT, ArtefactRef, ManifestError, RunManifest

# The nine pipeline stages the Reporter records, in narrative order.
STAGES: tuple[str, ...] = (
    "extraction",
    "compilation",
    "execution",
    "audit",
    "scientist",
    "diagnostics_crowding",
    "diagnostics_capacity",
    "diagnostics_regime",
    "holdout",
)

_HEX40 = re.compile(r"[0-9a-f]{40}")
_HEX_SHORT = re.compile(r"[0-9a-f]{7,12}")


class StageStatus(str, Enum):
    """The five distinguishable, distinctly-rendered stage states (INV-11). `None` never
    stands in for any of them — the stage record says which and why."""

    NOT_APPLICABLE = "not_applicable"
    UNOBSERVED = "unobserved"
    SUCCEEDED = "succeeded"
    REFUSED = "refused"
    FAILED = "failed"


@dataclass(frozen=True)
class StageEvidence:
    """What licensed a stage status (INV-2). `kind` is the license category; `detail` is a
    short human string; `source` optionally names the artefact key or function."""

    kind: str
    detail: str
    source: str | None = None

    def to_dict(self) -> dict:
        return {"kind": self.kind, "detail": self.detail, "source": self.source}


@dataclass(frozen=True)
class StageRecord:
    """One stage's status with the evidence that licensed it (§5.2)."""

    stage: str
    status: StageStatus
    evidence: StageEvidence
    artefact_ref: ArtefactRef | None = None
    refusal_code: str | None = None
    failure_code: str | None = None
    detail: str | None = None

    def to_dict(self) -> dict:
        return {
            "stage": self.stage,
            "status": self.status.value,
            "evidence": self.evidence.to_dict(),
            "artefact_key": self.artefact_ref.key if self.artefact_ref else None,
            "refusal_code": self.refusal_code,
            "failure_code": self.failure_code,
            "detail": self.detail,
        }


class ReportabilityStatus(str, Enum):
    """Whether a run's figures may be reported. Orthogonal to disposition; fails closed to
    non-reportable (INV-7)."""

    REPORTABLE = "reportable"
    NON_REPORTABLE_PHASE_D = "non_reportable_phase_d"
    NON_REPORTABLE_OTHER = "non_reportable_other"


def derive_reportability(phase: str) -> ReportabilityStatus:
    """Phase F -> reportable; phase D -> non-reportable (free dev pair); anything else fails
    closed to NON_REPORTABLE_OTHER (never reportable)."""
    if phase == "F":
        return ReportabilityStatus.REPORTABLE
    if phase == "D":
        return ReportabilityStatus.NON_REPORTABLE_PHASE_D
    return ReportabilityStatus.NON_REPORTABLE_OTHER


@dataclass(frozen=True)
class GitStamp:
    """A normalised git identity from an upstream run log. `full` is a 40-hex SHA (or None if
    the source carried only a short form — never fabricated); `short` is derived from full
    when only full is present. `source_form` records which form the log actually carried."""

    full: str | None
    short: str | None
    source_form: str  # "full", "short", "full+short", or "none"

    def to_dict(self) -> dict:
        return {"full": self.full, "short": self.short, "source_form": self.source_form}


def _stamp_from_log(log: dict | None) -> GitStamp | None:
    """Normalise `git_commit` / `git_short` from a run log. Quant writes the 40-char SHA under
    `git_commit` plus a separate `git_short`; the Auditor writes the SHORT form under the same
    `git_commit` key. Never fabricate a full SHA from a short one (C5 / §5.4)."""
    if not isinstance(log, dict):
        return None
    raw_commit = log.get("git_commit")
    raw_short = log.get("git_short")
    commit = raw_commit if isinstance(raw_commit, str) else None
    # Only accept a `git_short` that is actually a short hex SHA — an empty string or non-hex
    # garbage is treated as absent, so it never masquerades as a verified short identity.
    short = (
        raw_short
        if isinstance(raw_short, str) and _HEX_SHORT.fullmatch(raw_short)
        else None
    )
    carried_short = short is not None

    full: str | None = None
    if commit is not None and _HEX40.fullmatch(commit):
        full = commit
    elif commit is not None and _HEX_SHORT.fullmatch(commit) and short is None:
        short = commit  # Auditor-style short-only under git_commit
        carried_short = True

    if short is None and full is not None:
        short = full[:7]  # derive short from full (the only safe direction)

    if full is not None and carried_short:
        form = "full+short"
    elif full is not None:
        form = "full"
    elif short is not None:
        form = "short"
    else:
        form = "none"
    return GitStamp(full=full, short=short, source_form=form)


@dataclass(frozen=True)
class UpstreamStamps:
    """Git identities and pre-registration tags copied from upstream run logs."""

    quant: GitStamp | None = None
    audit: GitStamp | None = None
    auditor_prereg_tag: str | None = None

    def to_dict(self) -> dict:
        return {
            "quant": self.quant.to_dict() if self.quant else None,
            "audit": self.audit.to_dict() if self.audit else None,
            "auditor_prereg_tag": self.auditor_prereg_tag,
        }


@dataclass(frozen=True)
class CodeVersion:
    """The Reporter's own code version, stamped into every manifest for INV-8."""

    full: str
    short: str

    def to_dict(self) -> dict:
        return {"full": self.full, "short": self.short}


def _git(*args: str) -> str:
    try:
        return (
            subprocess.check_output(["git", *args], cwd=REPO_ROOT).decode().strip()
        )
    except Exception:
        return "unknown"


def current_code_version() -> CodeVersion:
    """The Reporter's git identity now (mirrors the driver idiom; 'unknown' if unavailable)."""
    full = _git("rev-parse", "HEAD")
    short = _git("rev-parse", "--short", "HEAD")
    return CodeVersion(full=full, short=short)


@dataclass(frozen=True)
class HoldoutView:
    """Reporter-owned holdout consumer type (C5). No `HoldoutEvaluation` exists upstream, so
    this carries the fields the note renders — a sign, a Sharpe CI and a paired difference —
    assembled in fixtures from `Measurements` (extension path is fixture-only, D1)."""

    sharpe_sign: int
    sharpe_ci_low: float
    sharpe_ci_high: float
    paired_difference: float | None = None

    def to_dict(self) -> dict:
        return {
            "sharpe_sign": self.sharpe_sign,
            "sharpe_ci_low": self.sharpe_ci_low,
            "sharpe_ci_high": self.sharpe_ci_high,
            "paired_difference": self.paired_difference,
        }


@dataclass(frozen=True)
class ProposalReport:
    """One proposal's Reporter view (extension path, fixture-only). `record` and `proposal`
    are the parsed serialised dicts; `diagnostics` is a flat crowding map; `holdout` is a
    Reporter-owned view. Capacity/regime have no upstream module and stay absent."""

    record: dict
    gate_outcomes: tuple[dict, ...] = ()
    proposal: dict | None = None
    diagnostics: Mapping[str, float] | None = None
    holdout: HoldoutView | None = None


@dataclass(frozen=True)
class ReportBundle:
    """The assembled, validated inputs for one reported run. `docs` maps each present artefact
    type to its parsed serialised dict (the `to_dict` shape) — the pointer-resolution surface."""

    reporter_run_id: str
    paper_id: str
    strategy_id: str
    phase: str
    docs: Mapping[ArtefactType, object]
    stages: Mapping[str, StageRecord]
    reportability: ReportabilityStatus
    artefacts: tuple[ArtefactRef, ...]
    upstream_stamps: UpstreamStamps
    code_version: CodeVersion
    join_rationale: str
    proposals: tuple[ProposalReport, ...] = ()

    def doc(self, artefact_type: ArtefactType) -> object | None:
        return self.docs.get(artefact_type)

    def has(self, artefact_type: ArtefactType) -> bool:
        return artefact_type in self.docs

    def ref(self, artefact_type: ArtefactType) -> ArtefactRef | None:
        for artefact in self.artefacts:
            if artefact.artefact_type == artefact_type:
                return artefact
        return None

    def audit_scope(self) -> str | None:
        audit = self.docs.get(ArtefactType.AUDIT_REPORT)
        if isinstance(audit, dict):
            scope = audit.get("audit_scope")
            return scope if isinstance(scope, str) else None
        return None


# ---------------------------------------------------------------------------------------------
# Stage-record derivation (INV-2): a status only from present evidence.
# ---------------------------------------------------------------------------------------------

def _present(ref_by_type: dict, artefact_type: ArtefactType) -> ArtefactRef | None:
    return ref_by_type.get(artefact_type)


def _derive_stage_records(
    manifest: RunManifest,
    docs: Mapping[ArtefactType, object],
    ref_by_type: dict[ArtefactType, ArtefactRef],
) -> dict[str, StageRecord]:
    records: dict[str, StageRecord] = {}

    def unobserved(stage: str, why: str) -> StageRecord:
        return StageRecord(
            stage=stage,
            status=StageStatus.UNOBSERVED,
            evidence=StageEvidence(kind="absent", detail=why),
        )

    # extraction — spec is required, so its presence licenses SUCCEEDED.
    spec_ref = _present(ref_by_type, ArtefactType.STRATEGY_SPEC)
    records["extraction"] = StageRecord(
        stage="extraction",
        status=StageStatus.SUCCEEDED,
        evidence=StageEvidence(kind="output_artefact", detail="spec present", source="spec"),
        artefact_ref=spec_ref,
    )

    # compilation — a typed refusal dominates; else an adapt/config output; else unobserved.
    refusal_ref = _present(ref_by_type, ArtefactType.CONFIG_REFUSAL)
    if refusal_ref is not None:
        refusal_doc = docs.get(ArtefactType.CONFIG_REFUSAL)
        code = refusal_doc.get("code") if isinstance(refusal_doc, dict) else None
        records["compilation"] = StageRecord(
            stage="compilation",
            status=StageStatus.REFUSED,
            evidence=StageEvidence(
                kind="typed_refusal", detail="ConfigRefusal present", source="config_refusal"
            ),
            artefact_ref=refusal_ref,
            refusal_code=code,
        )
    elif (
        _present(ref_by_type, ArtefactType.ADAPT_RESULT)
        or _present(ref_by_type, ArtefactType.QUANT_CONFIG)
        or _present(ref_by_type, ArtefactType.STRATEGY_RESULT)
    ):
        out_ref = (
            _present(ref_by_type, ArtefactType.QUANT_CONFIG)
            or _present(ref_by_type, ArtefactType.ADAPT_RESULT)
        )
        records["compilation"] = StageRecord(
            stage="compilation",
            status=StageStatus.SUCCEEDED,
            evidence=StageEvidence(
                kind="output_artefact",
                detail="compiled config or downstream result present",
                source=out_ref.key if out_ref else "quant_result",
            ),
            artefact_ref=out_ref,
        )
    else:
        records["compilation"] = unobserved(
            "compilation", "no compiled config, refusal or downstream result on disk"
        )

    # execution — a strategy result licenses SUCCEEDED.
    result_ref = _present(ref_by_type, ArtefactType.STRATEGY_RESULT)
    if result_ref is not None:
        records["execution"] = StageRecord(
            stage="execution",
            status=StageStatus.SUCCEEDED,
            evidence=StageEvidence(
                kind="output_artefact", detail="strategy result present", source="quant_result"
            ),
            artefact_ref=result_ref,
        )
    else:
        records["execution"] = unobserved("execution", "no strategy result on disk")

    # audit — the persisted AuditCore carries audit_scope; REFUSED scope is a refusal.
    audit_ref = _present(ref_by_type, ArtefactType.AUDIT_REPORT)
    audit_doc = docs.get(ArtefactType.AUDIT_REPORT)
    if audit_ref is not None and isinstance(audit_doc, dict):
        scope = audit_doc.get("audit_scope")
        if scope is None:
            # Present but missing the scope key — fail closed rather than assume success.
            records["audit"] = StageRecord(
                stage="audit",
                status=StageStatus.UNOBSERVED,
                evidence=StageEvidence(
                    kind="malformed",
                    detail="audit core present but audit_scope key absent",
                    source="audit_report",
                ),
                artefact_ref=audit_ref,
            )
        elif scope == "REFUSED":
            records["audit"] = StageRecord(
                stage="audit",
                status=StageStatus.REFUSED,
                evidence=StageEvidence(
                    kind="typed_refusal", detail="audit_scope REFUSED", source="audit_report"
                ),
                artefact_ref=audit_ref,
                refusal_code="AUDIT_SCOPE_REFUSED",
            )
        else:
            records["audit"] = StageRecord(
                stage="audit",
                status=StageStatus.SUCCEEDED,
                evidence=StageEvidence(
                    kind="output_artefact",
                    detail=f"audit core present (scope {scope})",
                    source="audit_report",
                ),
                artefact_ref=audit_ref,
            )
    else:
        records["audit"] = unobserved("audit", "no audit core on disk")

    # scientist + diagnostics + holdout — not derivable from persisted artefacts (C9). Default
    # UNOBSERVED; a pointer-declared stage overrides below. Capacity/regime have no module.
    records["scientist"] = unobserved(
        "scientist", "Scientist persists nothing; entry rule not evaluable from disk (C9)"
    )
    records["diagnostics_crowding"] = unobserved(
        "diagnostics_crowding", "no per-proposal crowding diagnostic on disk"
    )
    records["diagnostics_capacity"] = StageRecord(
        stage="diagnostics_capacity",
        status=StageStatus.UNOBSERVED,
        evidence=StageEvidence(
            kind="absent", detail="capacity has no diagnostic module in the codebase"
        ),
    )
    records["diagnostics_regime"] = StageRecord(
        stage="diagnostics_regime",
        status=StageStatus.UNOBSERVED,
        evidence=StageEvidence(
            kind="absent", detail="regime has no diagnostic module in the codebase"
        ),
    )
    records["holdout"] = unobserved("holdout", "holdout is a gated one-time access")

    # Pointer-declared stage records override the derived ones (e.g. scientist not_applicable).
    # An unknown stage name is an author error, not silently ignored (a typo must not vanish).
    for stage, declared in manifest.declared_stages.items():
        if stage not in STAGES:
            raise ManifestError(
                f"pointer-declared stage {stage!r} is not a known stage {STAGES}"
            )
        records[stage] = StageRecord(
            stage=stage,
            status=StageStatus(declared.status),
            evidence=StageEvidence(
                kind="declared_in_pointer", detail=declared.evidence, source="pointer_file"
            ),
        )
    return records


def load_bundle(
    manifest: RunManifest,
    *,
    repo_root: Path = REPO_ROOT,
    code_version: CodeVersion | None = None,
) -> ReportBundle:
    """Assemble a `ReportBundle` from a validated `RunManifest`. Reads and parses each declared
    artefact (its hash was verified at manifest load), derives stage records from what is
    present, and stamps reportability, upstream git identities and the Reporter code version."""
    docs: dict[ArtefactType, object] = {}
    ref_by_type: dict[ArtefactType, ArtefactRef] = {}
    for ref in manifest.artefacts.values():
        parsed = json.loads((repo_root / ref.path).read_text())
        docs[ref.artefact_type] = parsed
        ref_by_type[ref.artefact_type] = ref

    stages = _derive_stage_records(manifest, docs, ref_by_type)
    upstream = UpstreamStamps(
        quant=_stamp_from_log(docs.get(ArtefactType.QUANT_RUN_LOG)),
        audit=_stamp_from_log(docs.get(ArtefactType.AUDIT_RUN_LOG)),
        auditor_prereg_tag=_prereg_tag(docs.get(ArtefactType.AUDIT_RUN_LOG)),
    )
    return ReportBundle(
        reporter_run_id=manifest.reporter_run_id,
        paper_id=manifest.paper_id,
        strategy_id=manifest.strategy_id,
        phase=manifest.phase,
        docs=docs,
        stages=stages,
        reportability=derive_reportability(manifest.phase),
        artefacts=tuple(manifest.artefacts.values()),
        upstream_stamps=upstream,
        code_version=code_version or current_code_version(),
        join_rationale=manifest.join_rationale,
    )


def _prereg_tag(audit_log: object) -> str | None:
    if isinstance(audit_log, dict):
        tag = audit_log.get("auditor_prereg_tag")
        return tag if isinstance(tag, str) else None
    return None
