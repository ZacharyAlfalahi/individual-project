"""
Spec + trace emission and the run orchestrator (build brief §5.5, D8/D12).

``emit_spec_and_trace`` assembles a ``StrategySpec`` from its already-filled
parts, stamps the ``SpecHeader`` with full run provenance (registry
version/hash, silence-table version/hash, canonical-text hash, prompt-template
hashes, model ids, run id, timestamp, AND the trace's sha256), then runs
``validate_librarian_spec`` **fail-closed**: any policy violation (e.g. a DESIGN
tag surviving into Librarian output, D8) raises ``LibrarianEmissionError`` -- a
non-clean spec is never emitted.

``run_paper`` is the top-level ORCHESTRATOR. It calls
``canonical_text.require_frozen()`` FIRST, so a stub canonical text can never
reach a real extraction (the §5.1 status gate): every pipeline component runs
happily on a stub via ``.locate()``, but the full run refuses. It then
enumerates constructions, and for the agreed strategy list drives the
form-filler over Part 1 + the committed Part 2 fields, assembles each spec, and
emits (spec, trace) pairs -- collecting failure/review events on the run record
rather than raising them (P1: run-to-completion, visible-or-bounded).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..config.canonical_text import CanonicalText
from ..errors import LibrarianSchemaError, LibrarianValidationError
from ..schema.strategy_spec import Part1, Part2, SpecHeader, StrategySpec
from ..validators.spec_validators import (
    SignalRegistryLike,
    validate_librarian_spec,
)
from ..validators.tag_reason import TagReasonRegistry
from .failures import EnumerationDisagreement
from .lister import Construction, EnumerationResult
from .trace import ExtractionTrace, TraceRunHeader


class LibrarianEmissionError(LibrarianSchemaError):
    """``emit_spec_and_trace`` refused to emit a spec: ``validate_librarian_spec``
    returned a non-empty violation list (fail-closed, D8). Carries the collected
    violations so the caller can report every problem in one pass."""

    def __init__(self, violations: list[LibrarianValidationError]) -> None:
        self.violations = list(violations)
        summary = "; ".join(f"{v.field}: {v.reason}" for v in self.violations)
        super().__init__(f"spec failed Librarian-output validation and was not emitted: {summary}")


@dataclass(frozen=True)
class RunProvenance:
    """The run/registry/model stamps threaded into every emitted header + trace
    (build brief §2). Plain scalars -- computed once per run, passed in."""

    paper_id: str
    registry_version: str
    registry_hash: str
    silence_table_version: str
    canonical_text_hash: str
    model_a_id: str
    model_b_id: str
    prompt_template_hashes: str | None = None
    run_id: str | None = None
    timestamp: str | None = None

    @property
    def model_ids(self) -> str:
        return f"{self.model_a_id},{self.model_b_id}"


def build_trace_header(
    prov: RunProvenance, strategy_label: str
) -> TraceRunHeader:
    """Assemble the trace run header (build brief §2) for one strategy."""
    return TraceRunHeader(
        paper_id=prov.paper_id,
        strategy_label=strategy_label,
        registry_version=prov.registry_version,
        registry_hash=prov.registry_hash,
        silence_table_version=prov.silence_table_version,
        canonical_text_hash=prov.canonical_text_hash,
        model_a_id=prov.model_a_id,
        model_b_id=prov.model_b_id,
        run_id=prov.run_id,
        timestamp=prov.timestamp,
        prompt_template_hashes=prov.prompt_template_hashes,
    )


def emit_spec_and_trace(
    part1: Part1,
    part2: Part2,
    strategy_label,
    trace: ExtractionTrace,
    prov: RunProvenance,
    registry: SignalRegistryLike | None = None,
    tag_reason_registry: TagReasonRegistry | None = None,
) -> tuple[StrategySpec, ExtractionTrace]:
    """Assemble + stamp + validate one spec, fail-closed.

    ``strategy_label`` is the ``Inherited[str]`` name of the strategy (RQ1 scoring
    key, D20). The header is stamped with the full provenance set INCLUDING
    ``trace.sha256()`` so the spec and its trace cannot silently desync (D12).
    ``validate_librarian_spec`` runs last: a non-empty result raises
    ``LibrarianEmissionError`` (D8 fail-closed) -- a DESIGN-tagged field, an
    unknown concept_id, a STATED without a locator all block emission."""
    header = SpecHeader(
        paper_id=prov.paper_id,
        strategy_label=strategy_label,
        registry_version=prov.registry_version,
        registry_hash=prov.registry_hash,
        silence_table_version=prov.silence_table_version,
        canonical_text_hash=prov.canonical_text_hash,
        prompt_template_hashes=prov.prompt_template_hashes,
        model_ids=prov.model_ids,
        run_id=prov.run_id,
        timestamp=prov.timestamp,
        trace_sha256=trace.sha256(),
    )
    spec = StrategySpec(header=header, part1=part1, part2=part2)

    violations = validate_librarian_spec(
        spec, registry=registry, tag_reason_registry=tag_reason_registry
    )
    if violations:
        raise LibrarianEmissionError(violations)
    return spec, trace


# ---------------------------------------------------------------------------
# The orchestrator.
# ---------------------------------------------------------------------------

@dataclass
class PaperRunResult:
    """The run-to-completion outcome for one paper (P1: visible-or-bounded). Any
    combination of emitted specs and routed events may be present."""

    paper_id: str
    specs: list[tuple[StrategySpec, ExtractionTrace]] = field(default_factory=list)
    events: list[object] = field(default_factory=list)

    @property
    def review_events(self) -> list[object]:
        return [e for e in self.events if getattr(e, "routing", None) == "review"]

    @property
    def paper_failed_events(self) -> list[object]:
        return [e for e in self.events if getattr(e, "routing", None) == "paper_failed"]


# A strategy assembler is any callable that, given one strategy Construction +
# the frozen canonical text + provenance, produces (Part1, Part2,
# strategy_label, ExtractionTrace). Injected so the orchestrator does not hardcode
# the per-field form-filling loop (kept testable + swappable).
def run_paper(
    canonical_text: CanonicalText,
    enumeration: EnumerationResult,
    assemble_strategy,
    prov: RunProvenance,
    registry: SignalRegistryLike | None = None,
    tag_reason_registry: TagReasonRegistry | None = None,
) -> PaperRunResult:
    """Top-level orchestrator. Calls ``canonical_text.require_frozen()`` FIRST --
    a stub canonical text raises ``CanonicalTextNotFrozenError`` before any
    extraction (the §5.1 status gate). Then:

      * if enumeration disagreed -> record its ``EnumerationDisagreement`` event
        (routes to review) and return (no specs);
      * else for each STRATEGY construction, call ``assemble_strategy`` to fill
        its parts + trace, and ``emit_spec_and_trace`` to stamp + validate +
        emit. Emission failures are collected on the run record, never raised
        past the orchestrator (run-to-completion).

    ``assemble_strategy(construction, canonical_text, prov) -> (Part1, Part2,
    strategy_label, ExtractionTrace)`` is injected -- the orchestrator owns the
    gate + the loop, not the per-field extraction detail."""
    # STATUS GATE (§5.1): refuse a non-frozen (stub) text before any extraction.
    canonical_text.require_frozen()

    result = PaperRunResult(paper_id=enumeration.paper_id)

    if not enumeration.agreed:
        result.events.append(enumeration.disagreement)
        return result

    for construction in enumeration.strategies:
        if not isinstance(construction, Construction):  # defensive
            raise LibrarianSchemaError("enumeration.strategies must yield Constructions")
        part1, part2, strategy_label, trace = assemble_strategy(
            construction, canonical_text, prov
        )
        try:
            spec_and_trace = emit_spec_and_trace(
                part1=part1,
                part2=part2,
                strategy_label=strategy_label,
                trace=trace,
                prov=prov,
                registry=registry,
                tag_reason_registry=tag_reason_registry,
            )
        except LibrarianEmissionError as exc:
            result.events.append(exc)
            continue
        result.specs.append(spec_and_trace)

    return result


__all__ = [
    "LibrarianEmissionError",
    "RunProvenance",
    "PaperRunResult",
    "build_trace_header",
    "emit_spec_and_trace",
    "run_paper",
    "EnumerationDisagreement",
]
