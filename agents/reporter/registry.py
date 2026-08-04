"""registry.py — R7 canonical JSONL registry + atomic publication (docs/reporter/reporter_spec_v0.2.md §10, §11).

Aggregates the reported runs into six append-only JSONL files plus a `manifest.json` written
LAST (its presence is the publication commit marker). Everything is serialised through the
canonical contract (`shared/reporting/canonical.py`): sorted keys, tight separators,
`allow_nan=False` with the NaN sentinel, NFC, LF, one trailing newline — so a rebuild from the
same inputs is byte-identical (INV-8). NaN is a first-class upstream value and is ENCODED, never
rejected (§10.2). Publication builds everything in a staging directory, verifies every note, and
only then swaps onto the published directory, so a failure leaves the last valid publication
untouched (INV-12). No SQLite, no process locks (§1.2).
"""

from __future__ import annotations

import hashlib
import os
import shutil
from dataclasses import dataclass
from pathlib import Path

from agents.auditor.schemas.toggle import TOGGLE_BIAS_CLASS, TOGGLE_IDS
from shared.reporting.canonical import canonical_hash, canonical_json
from shared.reporting.claims import ArtefactType

from .manifest import REPO_ROOT
from .renderer import RenderedDocument, render_note
from .verify import assert_verified

# The canonical toggle order (single source of truth; the TOGGLE_BIAS_CLASS
# import-time guard already pins class coverage).
_TOGGLES = TOGGLE_IDS


def _dummy_reason(toggle: str, audit: dict) -> str | None:
    """Derive the per-(toggle) dummy_reason from the persisted AuditCore (ADR §5.4):

      not_applicable       — toggle is in `not_applicable_toggles` (declared: no estimand).
      input_unavailable    — toggle is non-runnable but not not_applicable (construct failure).
      no_treatment_support — runnable but MEASURED as a no-op (invariance.is_no_op).
      None                 — runnable with a real effect / empirical null (NOT a dummy).
    """
    if toggle in set(audit.get("not_applicable_toggles") or []):
        return "not_applicable"
    if toggle not in set(audit.get("runnable_toggles") or []):
        return "input_unavailable"
    for inv in audit.get("invariance") or []:
        if inv.get("toggle_id") == toggle and inv.get("is_no_op"):
            return "no_treatment_support"
    return None
_REGISTRY_SCHEMA_VERSION = 1
_JSONL_FILES = (
    "runs",
    "stages",
    "audit_toggles",
    "proposals",
    "artefacts",
    "claims_index",
)
_PUBLISHED_ROOT = REPO_ROOT / "artifacts" / "reporter"


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _claim_index(doc: RenderedDocument) -> dict:
    return {c.claim_id: c for c in doc.claims}


def _raw(doc: RenderedDocument, claim_id: str):
    claim = _claim_index(doc).get(claim_id)
    return claim.raw_value if claim is not None else None


# --- row builders ----------------------------------------------------------------------------

def _runs_row(bundle, doc: RenderedDocument) -> dict:
    stamps = bundle.upstream_stamps
    return {
        "reporter_run_id": bundle.reporter_run_id,
        "paper_id": bundle.paper_id,
        "strategy_id": bundle.strategy_id,
        "phase": bundle.phase,
        "reportability": bundle.reportability.value,
        "pipeline_disposition": doc.disposition.value,
        "audit_scope": bundle.audit_scope(),
        "primary_metric_sharpe": _raw(doc, "replication.sharpe"),
        "primary_metric_tstat": _raw(doc, "replication.tstat"),
        "primary_metric_avg_monthly": _raw(doc, "replication.avg"),
        "claimed_headline_mean": _raw(doc, "extraction.paper_mean"),
        "claimed_headline_tstat": _raw(doc, "extraction.paper_tstat"),
        "n_proposals": len(bundle.proposals),
        "note_sha256": _sha256_text(doc.note_markdown),
        "claim_ledger_sha256": canonical_hash([c.to_dict() for c in doc.claims]),
        "reporter_schema_version": _REGISTRY_SCHEMA_VERSION,
        "reporter_code_version": bundle.code_version.short,
        "reporter_code_version_full": bundle.code_version.full,
        "quant_git_short": stamps.quant.short if stamps.quant else None,
        "audit_git_short": stamps.audit.short if stamps.audit else None,
        "auditor_prereg_tag": stamps.auditor_prereg_tag,
        "legal_state_hash": doc.legal_state_hash,
    }


def _stage_rows(bundle) -> list[dict]:
    rows = []
    for stage, record in bundle.stages.items():
        rows.append(
            {
                "reporter_run_id": bundle.reporter_run_id,
                "stage": stage,
                "status": record.status.value,
                "evidence_kind": record.evidence.kind,
                "artefact_key": record.artefact_ref.key if record.artefact_ref else None,
                "refusal_code": record.refusal_code,
                "failure_code": record.failure_code,
                "detail": record.evidence.detail,
            }
        )
    return rows


def _audit_toggle_rows(bundle) -> list[dict]:
    audit = bundle.doc(ArtefactType.AUDIT_REPORT)
    if not isinstance(audit, dict):
        return []
    runnable = set(audit.get("runnable_toggles") or [])
    shapley = audit.get("shapley") or {}
    shares = shapley.get("shapley_share_of_registered_endpoint_gap") or {}
    doe = (audit.get("saturated_bases") or {}).get("doe_effects") or {}
    rows = []
    for toggle in _TOGGLES:
        rows.append(
            {
                "reporter_run_id": bundle.reporter_run_id,
                "toggle": toggle,
                # bias_class is a GLOBAL property of the intervention (ADR §5.1),
                # from the registry (single source of truth); dummy_reason is the
                # per-strategy applicability record (ADR §5.4).
                "bias_class": TOGGLE_BIAS_CLASS[toggle],
                "dummy_reason": _dummy_reason(toggle, audit),
                "runnable": toggle in runnable,
                "shapley_share": shares.get(toggle),
                "doe_main_effect": doe.get(toggle),
                # Bootstrap CI / FDR flag are full-report layers, not persisted (C9).
                "bootstrap_ci_low": None,
                "bootstrap_ci_high": None,
                "fdr_flag": None,
                "refusal_reason": None if toggle in runnable else "not_runnable",
            }
        )
    return rows


def _proposal_rows(bundle) -> list[dict]:
    rows = []
    for i, pr in enumerate(bundle.proposals):
        pid = pr.record.get("proposal_id", f"proposal_{i}")
        gate = pr.gate_outcomes[-1].get("gate") if pr.gate_outcomes else None
        holdout = pr.holdout
        rows.append(
            {
                "reporter_run_id": bundle.reporter_run_id,
                "proposal_id": pid,
                "outcome": pr.record.get("final_outcome"),
                "gate_reached": gate,
                "refusal_code": pr.record.get("refusal_code"),
                "mechanism_ref": (pr.proposal or {}).get("mechanism_ref"),
                "holdout_sharpe_sign": holdout.sharpe_sign if holdout else None,
                "holdout_sharpe_ci_low": holdout.sharpe_ci_low if holdout else None,
                "holdout_sharpe_ci_high": holdout.sharpe_ci_high if holdout else None,
                "holdout_paired_difference": (
                    holdout.paired_difference if holdout else None
                ),
            }
        )
    return rows


def _artefact_rows(bundle) -> list[dict]:
    return [
        {
            "reporter_run_id": bundle.reporter_run_id,
            "artefact_key": a.key,
            "artefact_type": a.artefact_type.value,
            "schema_version": a.schema_version,
            "path": a.path,
            "sha256": a.sha256,
            "producing_component": a.producing_component,
        }
        for a in bundle.artefacts
    ]


def _claims_index_rows(bundle, doc: RenderedDocument) -> list[dict]:
    return [
        {
            "reporter_run_id": bundle.reporter_run_id,
            "claim_id": c.claim_id,
            "slot_id": c.slot_id,
            "source_artifact": c.source_artifact.value,
            "source_artifact_sha256": c.source_artifact_sha256,
            "pointer": c.source_locator.pointer,
        }
        for c in doc.claims
    ]


@dataclass(frozen=True)
class Registry:
    rows: dict[str, list[dict]]

    def counts(self) -> dict[str, int]:
        return {name: len(self.rows[name]) for name in _JSONL_FILES}


def build_registry(pairs: list[tuple]) -> Registry:
    """Aggregate (bundle, RenderedDocument) pairs into the six JSONL row-sets, then check
    referential integrity: every non-runs row references a declared run, and every run appears
    exactly once (INV-9)."""
    rows: dict[str, list[dict]] = {name: [] for name in _JSONL_FILES}
    for bundle, doc in pairs:
        rows["runs"].append(_runs_row(bundle, doc))
        rows["stages"].extend(_stage_rows(bundle))
        rows["audit_toggles"].extend(_audit_toggle_rows(bundle))
        rows["proposals"].extend(_proposal_rows(bundle))
        rows["artefacts"].extend(_artefact_rows(bundle))
        rows["claims_index"].extend(_claims_index_rows(bundle, doc))

    run_ids = [r["reporter_run_id"] for r in rows["runs"]]
    if len(run_ids) != len(set(run_ids)):
        raise ValueError("registry: duplicate reporter_run_id in runs.jsonl (INV-9)")
    known = set(run_ids)
    for name in _JSONL_FILES:
        for row in rows[name]:
            if row["reporter_run_id"] not in known:
                raise ValueError(
                    f"registry: {name} row references unknown run {row['reporter_run_id']!r}"
                )
    return Registry(rows=rows)


# --- serialisation ---------------------------------------------------------------------------

def _jsonl(rows: list[dict]) -> str:
    """One canonical JSON object per line, LF-terminated (JSONL). Empty set -> empty string."""
    return "".join(canonical_json(row) + "\n" for row in rows)


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")


# --- publication -----------------------------------------------------------------------------

def _write_run_outputs(staging: Path, bundle, doc: RenderedDocument) -> None:
    run_dir = staging / "runs" / bundle.reporter_run_id
    _write_text(run_dir / "note.md", doc.note_markdown)
    claims_payload = {
        "reporter_run_id": bundle.reporter_run_id,
        "claims": [c.to_dict() for c in doc.claims],
        "evidence": [e.to_dict() for e in doc.evidence],
        "reporter_code_version": bundle.code_version.full,
        "legal_state_hash": doc.legal_state_hash,
    }
    _write_text(run_dir / "claims.json", canonical_json(claims_payload) + "\n")
    per_run_manifest = {
        "reporter_run_id": bundle.reporter_run_id,
        "disposition": doc.disposition.value,
        "reportability": bundle.reportability.value,
        "note_sha256": _sha256_text(doc.note_markdown),
        "claim_ledger_sha256": canonical_hash([c.to_dict() for c in doc.claims]),
        "reporter_code_version": bundle.code_version.full,
        "legal_state_hash": doc.legal_state_hash,
        "artefacts": [a.to_dict() for a in bundle.artefacts],
    }
    _write_text(run_dir / "manifest.json", canonical_json(per_run_manifest) + "\n")


def _write_registry(staging: Path, registry: Registry, code_version_full: str) -> None:
    reg_dir = staging / "registry"
    file_hashes: dict[str, str] = {}
    for name in _JSONL_FILES:
        text = _jsonl(registry.rows[name])
        _write_text(reg_dir / f"{name}.jsonl", text)
        file_hashes[f"{name}.jsonl"] = _sha256_text(text)
    manifest = {
        "registry_schema_version": _REGISTRY_SCHEMA_VERSION,
        "row_counts": registry.counts(),
        "file_sha256": file_hashes,
        "reporter_code_version": code_version_full,
        "legal_state_hash": _legal_state_hash(),
    }
    # manifest.json is written LAST — its presence marks a complete publication.
    _write_text(reg_dir / "manifest.json", canonical_json(manifest) + "\n")


def _legal_state_hash() -> str:
    from .legal_states import load_table

    return load_table().content_hash


def _atomic_swap(staging: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        backup = target.with_name(target.name + ".prev")
        if backup.exists():
            shutil.rmtree(backup)
        os.replace(target, backup)
        try:
            os.replace(staging, target)
        except Exception:
            os.replace(backup, target)  # roll back to the last valid publication
            raise
        shutil.rmtree(backup, ignore_errors=True)
    else:
        os.replace(staging, target)


@dataclass(frozen=True)
class PublishResult:
    out_root: Path
    n_runs: int
    row_counts: dict[str, int]


def publish(
    bundles: list,
    *,
    out_root: Path = _PUBLISHED_ROOT,
    staging_root: Path | None = None,
    verify: bool = True,
) -> PublishResult:
    """Render + verify every bundle, build the registry, and atomically swap the published
    directory. A failure at any step before the swap leaves `out_root` untouched (INV-12)."""
    if not bundles:
        # Never swap an empty publication over a prior good one.
        raise ValueError("publish: no bundles to publish")
    out_root = Path(out_root)
    staging = Path(staging_root) if staging_root is not None else out_root.with_name(
        out_root.name + ".staging"
    )
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)

    pairs = []
    for bundle in bundles:
        doc = render_note(bundle)
        if verify:
            assert_verified(doc, bundle)
        _write_run_outputs(staging, bundle, doc)
        pairs.append((bundle, doc))

    registry = build_registry(pairs)
    code_version = bundles[0].code_version.full if bundles else "unknown"
    _write_registry(staging, registry, code_version)

    _atomic_swap(staging, out_root)
    return PublishResult(
        out_root=out_root, n_runs=len(pairs), row_counts=registry.counts()
    )
