"""
Unit tests for the trace + emission layer (build brief §5.5, D8/D12) and the
run_paper orchestrator status gate (§5.1).

  * ExtractionTrace schema + sha256 (reproducible, hashes to_dict);
  * emitted SpecHeader carries the trace sha256;
  * emit_spec_and_trace is fail-closed (a DESIGN-tagged field -> refusal);
  * run_paper on the STUB raises CanonicalTextNotFrozenError (frozen gate first);
  * run_paper on a frozen text runs the assembler + emits.
"""

import pytest

from agents.quant.config import Evidence, Inherited, Locator

from agents.librarian.errors import CanonicalTextNotFrozenError
from agents.librarian.pipeline.emission import (
    LibrarianEmissionError,
    emit_spec_and_trace,
    run_paper,
)
from agents.librarian.pipeline.lister import EnumerationResult
from agents.librarian.pipeline.trace import (
    ExtractionTrace,
    FieldTraceRecord,
    ModelTrace,
    TraceRunHeader,
)

from _librarian_fixtures import build_part1, build_part2, stated
from _librarian_pipeline_fixtures import (
    Q_RANKED,
    construction,
    frozen_stub,
    load_stub,
    provenance,
)


def _trace(paper_id="SYNTH-0001", label="Synthetic Momentum"):
    header = TraceRunHeader(
        paper_id=paper_id,
        strategy_label=label,
        registry_version="sig-v1",
        registry_hash="deadbeef",
        silence_table_version="v1.1",
        canonical_text_hash="cafef00d",
        model_a_id="fake-a",
        model_b_id="fake-b",
    )
    rec = FieldTraceRecord(
        field="n_groups",
        model_a=ModelTrace(answered=True, raw=5, quote="q", locate_result=Locator(0, 0, 1)),
        model_b=ModelTrace(answered=True, raw=5, quote="q2", locate_result=Locator(0, 5, 7)),
        normalised_a=5,
        normalised_b=5,
        agreement=True,
        final_tag="STATED",
        final_reason="quoted",
        ship_choice="model_a",
    )
    return ExtractionTrace(header=header, records=(rec,))


# --- ExtractionTrace schema + sha256 ----------------------------------------

def test_trace_to_dict_and_sha256_reproducible():
    t1 = _trace()
    t2 = _trace()
    assert t1.to_dict() == t2.to_dict()
    # sha256 is a hex digest, reproducible across identical traces.
    h = t1.sha256()
    assert isinstance(h, str) and len(h) == 64
    assert t1.sha256() == t2.sha256()


def test_trace_sha256_changes_with_content():
    t1 = _trace()
    t2 = _trace(label="Different Label")
    assert t1.sha256() != t2.sha256()


def test_trace_record_lookup():
    t = _trace()
    assert t.record_for("n_groups") is not None
    assert t.record_for("absent") is None


# --- emitted header carries the trace sha256 --------------------------------

def test_emitted_header_carries_trace_sha256():
    trace = _trace()
    spec, out_trace = emit_spec_and_trace(
        part1=build_part1(),
        part2=build_part2(),
        strategy_label=stated("Synthetic Momentum"),
        trace=trace,
        prov=provenance(),
    )
    assert spec.header.trace_sha256 == trace.sha256()
    assert out_trace is trace
    # provenance stamps threaded through (build brief §2).
    assert spec.header.registry_version == "sig-v1"
    assert spec.header.model_ids == "fake-a,fake-b"


# --- fail-closed: a DESIGN-tagged field is refused (D8) ---------------------

def test_emit_is_fail_closed_on_design():
    design_field = Inherited(
        "par", "DESIGN", Evidence(note="par-weighting substitution (config decision)")
    )
    part2 = build_part2(weighting_base=design_field)
    with pytest.raises(LibrarianEmissionError) as exc:
        emit_spec_and_trace(
            part1=build_part1(),
            part2=part2,
            strategy_label=stated("Synthetic Momentum"),
            trace=_trace(),
            prov=provenance(),
        )
    # the collected violation names the DESIGN field.
    assert any("DESIGN" in v.reason for v in exc.value.violations)


# --- run_paper status gate (§5.1) -------------------------------------------

def _assembler(construction_arg, canonical_text, prov):
    """A trivial assembler: returns well-formed parts + a trace. Should never be
    reached on the stub path (the gate raises first)."""
    return build_part1(), build_part2(), stated("Synthetic Momentum"), _trace()


def test_run_paper_on_stub_raises_not_frozen():
    stub = load_stub()  # status: stub
    enum = EnumerationResult(paper_id="SYNTH-0001", constructions=(construction(),))
    with pytest.raises(CanonicalTextNotFrozenError):
        run_paper(stub, enum, _assembler, provenance())


def test_run_paper_on_frozen_emits_specs():
    ct = frozen_stub()
    enum = EnumerationResult(
        paper_id="SYNTH-0001",
        constructions=(construction(name="Momentum", quote=Q_RANKED, cls="strategy"),),
    )
    result = run_paper(ct, enum, _assembler, provenance())
    assert len(result.specs) == 1
    spec, trace = result.specs[0]
    assert spec.header.trace_sha256 == trace.sha256()
    assert result.review_events == [] and result.paper_failed_events == []


def test_run_paper_records_enumeration_disagreement_and_emits_nothing():
    from agents.librarian.pipeline.failures import EnumerationDisagreement

    ct = frozen_stub()
    enum = EnumerationResult(
        paper_id="P",
        disagreement=EnumerationDisagreement(paper_id="P", detail="d", only_model_a=("X",)),
    )
    result = run_paper(ct, enum, _assembler, provenance())
    assert result.specs == []
    assert len(result.review_events) == 1


def test_run_paper_collects_emission_failure_run_to_completion():
    # an assembler that emits a DESIGN-tagged field -> emission refuses; the
    # orchestrator collects it rather than raising (run-to-completion, P1).
    def bad_assembler(construction_arg, canonical_text, prov):
        design = Inherited("par", "DESIGN", Evidence(note="config decision"))
        return build_part1(), build_part2(weighting_base=design), stated("S"), _trace()

    ct = frozen_stub()
    enum = EnumerationResult(paper_id="P", constructions=(construction(),))
    result = run_paper(ct, enum, bad_assembler, provenance())
    assert result.specs == []
    assert any(isinstance(e, LibrarianEmissionError) for e in result.events)
