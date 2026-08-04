"""emit.py — claim/evidence emission and RenderedFragment (docs/reporter/reporter_spec_v0.2.md §8.1).

The renderer has NO API to interpolate a raw numeric value into prose. A section builds a
`ClaimSpec`, calls `emit_claim`, and gets back the displayed token plus a `ClaimRecord` — the
number is resolved from its typed source, formatted centrally through `fmt`, and recorded with
full provenance (INV-1). A section returns a `RenderedFragment`, never bare text, so a stray
number cannot escape the ledger. Verbatim upstream prose is carried by `emit_evidence` as an
`EvidenceBlock` (INV-14).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from shared.reporting.canonical import NanValue
from shared.reporting.claims import (
    ArtefactType,
    ClaimRecord,
    ClaimSpec,
    EvidenceBlock,
    EvidenceVerification,
    SourceLocator,
    Unit,
)
from shared.reporting.resolve import PointerResolutionError, resolve_locator

from .format import fmt, scale_for_unit
from .projections import resolve_projection

# Named display precisions (decimal places). `raw` -> shortest round-trip.
FORMATTER_PRECISION: dict[str, int | None] = {
    "raw": None,
    "int": 0,
    "1dp": 1,
    "2dp": 2,
    "3dp": 3,
    "4dp": 4,
}


class ClaimResolutionError(ValueError):
    """A claim could not be resolved or is not a display number (fail-closed, INV-13)."""


class SuppressedValueError(ValueError):
    """A magnitude was requested for a value that must not be emitted — a refused toggle, a
    REFUSED audit scope, or a proposal figure before G3. The renderer RAISES rather than quietly
    omitting it (INV-4)."""


def assert_not_suppressed(suppressed: bool, detail: str) -> None:
    """Guard an emission site: raise `SuppressedValueError` if the value is suppressed."""
    if suppressed:
        raise SuppressedValueError(detail)


@dataclass(frozen=True)
class RenderedFragment:
    """A section's output: prose plus every claim and evidence block it emitted. A section
    cannot return bare text — this is the type constraint behind INV-1/INV-10."""

    text: str
    claims: tuple[ClaimRecord, ...] = ()
    evidence: tuple[EvidenceBlock, ...] = ()


def _precision(formatter_id: str) -> int | None:
    if formatter_id not in FORMATTER_PRECISION:
        raise ClaimResolutionError(f"unknown formatter {formatter_id!r}")
    return FORMATTER_PRECISION[formatter_id]


def _fetch_doc(spec: ClaimSpec, bundle) -> object:
    doc = bundle.doc(spec.source_artifact)
    if doc is None:
        raise ClaimResolutionError(
            f"claim {spec.claim_id!r} sources {spec.source_artifact.value!r}, "
            "which is not present in the bundle"
        )
    if spec.source_locator.kind == "projection_pointer":
        return resolve_projection(spec.source_locator.projection_id, doc)
    return doc


def _record_value(raw: object) -> object:
    """Coerce a resolved value to a valid `ClaimRecord.raw_value`. A NaN becomes a `NanValue`;
    a bool / None / non-numeric resolution is an error (a quantitative claim needs a number)."""
    if isinstance(raw, bool):
        raise ClaimResolutionError(f"claim resolved to a bool ({raw!r}), not a number")
    if raw is None:
        raise ClaimResolutionError("claim resolved to null; a quantitative claim needs a value")
    if isinstance(raw, float) and math.isnan(raw):
        return NanValue(reason=None)
    if not isinstance(raw, (int, float)):
        raise ClaimResolutionError(f"claim resolved to {type(raw).__name__}, not a number")
    return raw


def _resolve_conditioning(spec: ClaimSpec, bundle) -> str | None:
    """Resolve the conditioning statement bound to this claim (INV-5), against the claim's own
    source artefact. Returns None if the pointer is absent or resolves to an empty value."""
    if spec.conditioning_pointer is None:
        return None
    doc = bundle.doc(spec.source_artifact)
    if doc is None:
        return None
    try:
        value = resolve_locator(spec.conditioning_pointer, doc)
    except PointerResolutionError:
        # Genuine absence only. A malformed pointer / other error is not silently dropped.
        return None
    if isinstance(value, str) and value.strip():
        return value
    return None


def emit_claim(spec: ClaimSpec, bundle) -> tuple[str, ClaimRecord]:
    """Resolve, format and record one claim. Returns the displayed token (for the prose) and
    the `ClaimRecord` (for the ledger). Raises on any unresolved/typed failure (fail-closed)."""
    doc = _fetch_doc(spec, bundle)
    raw = resolve_locator(spec.source_locator, doc)
    raw_record = _record_value(raw)
    precision = _precision(spec.formatter_id)
    scale = scale_for_unit(spec.unit)
    displayed = fmt(
        raw_record if isinstance(raw_record, NanValue) else raw,
        precision=precision,
        scale=scale,
    )
    ref = bundle.ref(spec.source_artifact)
    record = ClaimRecord(
        claim_id=spec.claim_id,
        slot_id=spec.slot_id,
        source_artifact=spec.source_artifact,
        source_artifact_sha256=ref.sha256 if ref else "",
        source_schema_version=(
            ref.schema_version if ref and ref.schema_version else "unknown"
        ),
        source_locator=spec.source_locator,
        raw_value=raw_record,
        displayed_value=displayed,
        unit=spec.unit.value,
        precision=precision,
        conditioning_text=_resolve_conditioning(spec, bundle),
    )
    return displayed, record


def emit_from_mapping(
    *,
    claim_id: str,
    slot_id: str,
    source_artifact: ArtefactType,
    pointer: str,
    formatter_id: str,
    unit: Unit,
    mapping: dict,
    sha256: str,
    schema_version: str = "unknown",
) -> tuple[str, ClaimRecord]:
    """Emit a claim resolved against an arbitrary in-memory `mapping` rather than a bundle
    artefact — for per-proposal diagnostics / holdout views, which live on the ProposalReport,
    not in `bundle.docs`. Same discipline: resolve, format centrally, record with provenance."""
    locator = SourceLocator(kind="json_pointer", pointer=pointer)
    raw = resolve_locator(locator, mapping)
    raw_record = _record_value(raw)
    precision = _precision(formatter_id)
    scale = scale_for_unit(unit)
    displayed = fmt(
        raw_record if isinstance(raw_record, NanValue) else raw,
        precision=precision,
        scale=scale,
    )
    record = ClaimRecord(
        claim_id=claim_id,
        slot_id=slot_id,
        source_artifact=source_artifact,
        source_artifact_sha256=sha256,
        source_schema_version=schema_version,
        source_locator=locator,
        raw_value=raw_record,
        displayed_value=displayed,
        unit=unit.value,
        precision=precision,
        conditioning_text=None,
    )
    return displayed, record


def emit_evidence(
    *,
    evidence_id: str,
    source_artifact: ArtefactType,
    locator: SourceLocator,
    bundle,
    verbatim_text: str | None = None,
    verification: EvidenceVerification | None = None,
) -> EvidenceBlock:
    """Build an `EvidenceBlock` carrying verbatim upstream text (INV-14). If `verbatim_text`
    is None the text is resolved from `locator` against the source artefact."""
    ref = bundle.ref(source_artifact)
    if verbatim_text is None:
        doc = bundle.doc(source_artifact)
        if doc is None:
            raise ClaimResolutionError(
                f"evidence {evidence_id!r} sources {source_artifact.value!r}, which is absent"
            )
        resolved = resolve_locator(locator, doc)
        if not isinstance(resolved, str):
            raise ClaimResolutionError(
                f"evidence {evidence_id!r} resolved to {type(resolved).__name__}, not text"
            )
        verbatim_text = resolved
    return EvidenceBlock(
        evidence_id=evidence_id,
        source_artifact=source_artifact,
        source_artifact_sha256=ref.sha256 if ref else "",
        source_locator=locator,
        verbatim_text=verbatim_text,
        verification=verification,
    )
