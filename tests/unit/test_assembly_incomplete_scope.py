"""
CI-9 (2026-09-06) tripwires: an ``AssemblyIncomplete`` (unresolved sort signal)
is a PER-CONSTRUCTION review outcome, not a whole-paper abort.

Before CI-9 the exception escaped ``run_paper`` uncaught, so the first failing
construction zeroed every remaining construction on the paper (the DFPS 0/27
cascade). These tests pin the repaired scope:

  * one failing construction -> its siblings still emit; a typed event with
    ``kind="assembly_incomplete"`` and ``routing="review"`` is recorded;
  * all constructions failing -> zero specs, one typed event each (never a
    silent drop, never an uncaught escape);
  * the routing table + ``route_of`` know the new kind;
  * driver exits: zero specs + assembly events -> exit 2 (review, matching the
    pre-CI-9 semantics for that condition) with ``events.json`` persisted;
    zero specs + no events -> the established exit 3.
"""

import json

from agents.librarian.pipeline import FAILURE_ROUTING, AssemblyIncomplete, route_of
from agents.librarian.pipeline.emission import PaperRunResult, run_paper
from agents.librarian.pipeline.lister import EnumerationResult

from _librarian_fixtures import build_part1, build_part2, stated
from _librarian_pipeline_fixtures import Q_RANKED, construction, frozen_stub, provenance
from test_trace_emission import _trace


def _failing_for(bad_names):
    """An assembler that raises AssemblyIncomplete for the named constructions
    and returns well-formed parts for every other one."""

    def assemble(construction_arg, canonical_text, prov):
        if construction_arg.name in bad_names:
            raise AssemblyIncomplete(
                f"{construction_arg.name!r}: sort_signal did not resolve",
                paper_id=prov.paper_id,
                construction_name=construction_arg.name,
            )
        return build_part1(), build_part2(), stated(construction_arg.name), _trace()

    return assemble


def _enum(*names):
    return EnumerationResult(
        paper_id="SYNTH-0001",
        constructions=tuple(
            construction(name=n, quote=Q_RANKED, cls="strategy") for n in names
        ),
    )


def test_assembly_failure_is_scoped_to_the_one_construction():
    result = run_paper(
        frozen_stub(), _enum("VaR", "Rating", "Momentum"),
        _failing_for({"VaR"}), provenance(),
    )
    assert len(result.specs) == 2  # the siblings of the failing row still emit
    emitted = {spec.header.strategy_label.value for spec, _ in result.specs}
    assert emitted == {"Rating", "Momentum"}
    events = [e for e in result.events if isinstance(e, AssemblyIncomplete)]
    assert len(events) == 1
    assert events[0].construction_name == "VaR"
    assert events[0].routing == "review"
    assert result.review_events == events  # counted as a review event


def test_all_constructions_failing_assembly_emit_nothing_but_typed_events():
    result = run_paper(
        frozen_stub(), _enum("A", "B"), _failing_for({"A", "B"}), provenance(),
    )
    assert result.specs == []
    assert [e.construction_name for e in result.events] == ["A", "B"]
    assert all(e.routing == "review" for e in result.events)


def test_assembly_incomplete_is_typed_and_routable():
    assert FAILURE_ROUTING["assembly_incomplete"] == "review"
    ev = AssemblyIncomplete("detail", paper_id="P", construction_name="C")
    assert route_of(ev) == "review"
    d = ev.to_dict()
    assert d == {"kind": "assembly_incomplete", "paper_id": "P",
                 "construction_name": "C", "detail": "detail",
                 "routing": "review"}
    # Back-compat: the single-positional-message form still constructs.
    assert AssemblyIncomplete("just a message").construction_name is None


def test_driver_exits_review_when_all_constructions_fall_to_assembly(monkeypatch, tmp_path):
    from scripts import run_librarian

    fabricated = PaperRunResult(paper_id="bbw")
    fabricated.events.append(
        AssemblyIncomplete("sort signal unresolved", paper_id="bbw",
                           construction_name="VaR")
    )
    monkeypatch.setattr(run_librarian, "run_paper", lambda **kw: fabricated)
    out = tmp_path / "out_assembly"
    rc = run_librarian.main(["--paper", "bbw", "--phase", "fake", "--out", str(out)])
    assert rc == 2  # review, as before CI-9 for the nothing-emitted condition
    events = json.loads((out / "events.json").read_text(encoding="utf-8"))
    assert events == [{"kind": "assembly_incomplete", "paper_id": "bbw",
                       "construction_name": "VaR",
                       "detail": "sort signal unresolved", "routing": "review"}]
    # The FULL manifest lands (completion path), not the early zero-spec stub.
    manifest = json.loads((out / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["manifest_schema"] == 2


def test_driver_keeps_exit_3_for_zero_specs_without_events(monkeypatch, tmp_path):
    from scripts import run_librarian

    monkeypatch.setattr(
        run_librarian, "run_paper", lambda **kw: PaperRunResult(paper_id="bbw")
    )
    out = tmp_path / "out_empty"
    rc = run_librarian.main(["--paper", "bbw", "--phase", "fake", "--out", str(out)])
    assert rc == 3
    assert not (out / "events.json").exists()
