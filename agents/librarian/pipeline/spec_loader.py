"""Typed StrategySpec deserialization from emitted JSON.

Reconstructs an emitted StrategySpec JSON object as the corresponding typed,
validated schema objects for downstream adaptation.

Design rules:
  * Lives outside ``agents/librarian/schema/`` to preserve that package as the
    frozen data contract. This module imports its types and mirrors each
    ``to_dict`` representation exactly.
  * STRICT: every dict's key set must be exactly what the paired ``to_dict``
    emits (modulo its documented conditional omissions). An unknown key is a
    loud ``SpecDeserialisationError`` naming the path — schema drift must break
    HERE, at the seam, never as a silently-dropped field.
  * The round-trip law is ``to_dict(spec_from_dict(d)) == d`` for every emitted
    spec dict (pinned over the real on-disk runs by tests/unit/test_spec_loader).
    NOTE the law is DICT-level: JSON has no tuples, so a tuple-valued
    ``Inherited.value`` (none exist in the sort schema today) would come back as
    a list — values pass through untouched in both directions.
  * Construction happens through the schema constructors, so every schema
    ``__post_init__`` shape guard (and the provenance layer's tag/evidence
    invariants) re-validates the loaded artefact for free.
"""

from __future__ import annotations

from typing import Any, Mapping

from agents.quant.config import Evidence, Inherited
from agents.quant.config.provenance import Locator

from ..errors import LibrarianSchemaError
from ..schema.estimation_fields import ESTIMATION_FIELDS, INSTRUMENT_INHERITED_FIELDS
from ..schema.signal_ref import DescribedSignal, LocatedQuote, SignalRef
from ..schema.strategy_spec import (
    _COMMON_INHERITED_FIELDS,
    _LEG_INHERITED_FIELDS,
    _PAPER_FACTS_INHERITED_FIELDS,
    Combiner,
    EstimationBlock,
    InstrumentRef,
    InstrumentSet,
    Leg,
    MethodSummary,
    Part1,
    Part2,
    PaperFacts,
    SpecHeader,
    StrategySpec,
)


class SpecDeserialisationError(LibrarianSchemaError):
    """An emitted-spec dict does not match the schema's serialisation surface."""


def _require_mapping(d: object, path: str) -> Mapping:
    if not isinstance(d, Mapping):
        raise SpecDeserialisationError(f"{path}: expected an object, got {type(d).__name__}")
    return d


def _check_keys(d: Mapping, path: str, required: tuple[str, ...],
                optional: tuple[str, ...] = ()) -> None:
    keys = set(d)
    missing = [k for k in required if k not in keys]
    unknown = sorted(keys - set(required) - set(optional))
    if missing:
        raise SpecDeserialisationError(f"{path}: missing key(s) {missing}")
    if unknown:
        raise SpecDeserialisationError(
            f"{path}: unknown key(s) {unknown} — the schema's to_dict never emits "
            "these; update spec_loader alongside any schema change")


_EVIDENCE_KEYS = ("quote", "rule_id", "rule_text", "note", "candidates",
                  "chosen", "column", "locator", "unknown_reason")


def _locator_from(d: object, path: str) -> Locator:
    d = _require_mapping(d, path)
    _check_keys(d, path, ("page", "char_start", "char_end"), ("end_page",))
    return Locator(page=d["page"], char_start=d["char_start"],
                   char_end=d["char_end"], end_page=d.get("end_page"))


def _evidence_from(d: object, path: str) -> Evidence:
    d = _require_mapping(d, path)
    _check_keys(d, path, (), _EVIDENCE_KEYS)   # Evidence.to_dict drops every None
    kwargs: dict[str, Any] = {k: d[k] for k in _EVIDENCE_KEYS
                              if k in d and k != "locator"}
    if "candidates" in kwargs and kwargs["candidates"] is not None:
        kwargs["candidates"] = tuple(kwargs["candidates"])
    if "locator" in d:
        kwargs["locator"] = _locator_from(d["locator"], f"{path}.locator")
    return Evidence(**kwargs)


def _inherited_from(d: object, path: str) -> Inherited:
    d = _require_mapping(d, path)
    _check_keys(d, path, ("value", "tag", "evidence"))
    return Inherited(value=d["value"], tag=d["tag"],
                     evidence=_evidence_from(d["evidence"], f"{path}.evidence"))


def _located_quote_from(d: object, path: str) -> LocatedQuote:
    d = _require_mapping(d, path)
    _check_keys(d, path, ("text", "page", "char_start", "char_end"))
    return LocatedQuote(text=d["text"], page=d["page"],
                        char_start=d["char_start"], char_end=d["char_end"])


def _described_from(d: object, path: str) -> DescribedSignal:
    d = _require_mapping(d, path)
    _check_keys(d, path, ("label", "quotes"))
    return DescribedSignal(
        label=d["label"],
        quotes=tuple(_located_quote_from(q, f"{path}.quotes[{i}]")
                     for i, q in enumerate(d["quotes"])),
    )


def _signal_ref_from(d: object, path: str) -> SignalRef:
    d = _require_mapping(d, path)
    _check_keys(d, path, ("concept_id", "parameters", "as_described"))
    params = _require_mapping(d["parameters"], f"{path}.parameters")
    return SignalRef(
        concept_id=_inherited_from(d["concept_id"], f"{path}.concept_id"),
        as_described=_described_from(d["as_described"], f"{path}.as_described"),
        parameters={name: _inherited_from(v, f"{path}.parameters[{name!r}]")
                    for name, v in params.items()},
    )


def _leg_from(d: object, path: str) -> Leg:
    d = _require_mapping(d, path)
    _check_keys(d, path, ("sort_signal", "control_axis") + _LEG_INHERITED_FIELDS)
    control = d["control_axis"]
    return Leg(
        sort_signal=_signal_ref_from(d["sort_signal"], f"{path}.sort_signal"),
        control_axis=(_signal_ref_from(control, f"{path}.control_axis")
                      if control is not None else None),
        **{name: _inherited_from(d[name], f"{path}.{name}")
           for name in _LEG_INHERITED_FIELDS},
    )


def _method_summary_from(d: object, path: str) -> MethodSummary:
    d = _require_mapping(d, path)
    _check_keys(d, path, ("summary", "quotes"))
    return MethodSummary(
        summary=_inherited_from(d["summary"], f"{path}.summary"),
        quotes=tuple(_located_quote_from(q, f"{path}.quotes[{i}]")
                     for i, q in enumerate(d["quotes"])),
    )


def _part1_from(d: object, path: str) -> Part1:
    d = _require_mapping(d, path)
    _check_keys(d, path, ("formation_structure", "asset_class", "method_summary"))
    return Part1(
        formation_structure=_inherited_from(d["formation_structure"],
                                            f"{path}.formation_structure"),
        asset_class=_inherited_from(d["asset_class"], f"{path}.asset_class"),
        method_summary=_method_summary_from(d["method_summary"], f"{path}.method_summary"),
    )


def _part2_from(d: object, path: str) -> Part2:
    d = _require_mapping(d, path)
    _check_keys(d, path, _COMMON_INHERITED_FIELDS + ("legs", "combiner"))
    combiner = _require_mapping(d["combiner"], f"{path}.combiner")
    _check_keys(combiner, f"{path}.combiner", ("kind",))
    return Part2(
        **{name: _inherited_from(d[name], f"{path}.{name}")
           for name in _COMMON_INHERITED_FIELDS},
        legs=tuple(_leg_from(leg, f"{path}.legs[{i}]")
                   for i, leg in enumerate(d["legs"])),
        combiner=Combiner(kind=_inherited_from(combiner["kind"], f"{path}.combiner.kind")),
    )


def _paper_facts_from(d: object, path: str) -> PaperFacts:
    d = _require_mapping(d, path)
    _check_keys(d, path, _PAPER_FACTS_INHERITED_FIELDS)
    return PaperFacts(**{name: _inherited_from(d[name], f"{path}.{name}")
                         for name in _PAPER_FACTS_INHERITED_FIELDS})


_HEADER_REQUIRED = ("paper_id", "strategy_label", "registry_version", "registry_hash",
                    "silence_table_version", "canonical_text_hash",
                    "prompt_template_hashes", "model_ids", "run_id", "timestamp",
                    "trace_sha256")
_HEADER_OPTIONAL = ("standing_substitutions_version", "standing_substitutions_hash")


def _header_from(d: object, path: str) -> SpecHeader:
    d = _require_mapping(d, path)
    _check_keys(d, path, _HEADER_REQUIRED, _HEADER_OPTIONAL)
    return SpecHeader(
        paper_id=d["paper_id"],
        strategy_label=_inherited_from(d["strategy_label"], f"{path}.strategy_label"),
        registry_version=d["registry_version"],
        registry_hash=d["registry_hash"],
        silence_table_version=d["silence_table_version"],
        canonical_text_hash=d["canonical_text_hash"],
        prompt_template_hashes=d["prompt_template_hashes"],
        model_ids=d["model_ids"],
        run_id=d["run_id"],
        timestamp=d["timestamp"],
        trace_sha256=d["trace_sha256"],
        standing_substitutions_version=d.get("standing_substitutions_version"),
        standing_substitutions_hash=d.get("standing_substitutions_hash"),
    )


def _estimation_from(d: object, path: str) -> EstimationBlock:
    d = _require_mapping(d, path)
    _check_keys(d, path, tuple(ESTIMATION_FIELDS))
    return EstimationBlock(**{name: _inherited_from(d[name], f"{path}.{name}")
                              for name in ESTIMATION_FIELDS})


def _instrument_from(d: object, path: str) -> InstrumentRef:
    d = _require_mapping(d, path)
    _check_keys(d, path, ("concept_id",) + tuple(INSTRUMENT_INHERITED_FIELDS)
                + ("as_described",))
    return InstrumentRef(
        concept_id=_inherited_from(d["concept_id"], f"{path}.concept_id"),
        as_described=_described_from(d["as_described"], f"{path}.as_described"),
        **{name: _inherited_from(d[name], f"{path}.{name}")
           for name in INSTRUMENT_INHERITED_FIELDS},
    )


def _instrument_set_from(d: object, path: str) -> InstrumentSet:
    d = _require_mapping(d, path)
    _check_keys(d, path, ("instruments",))
    return InstrumentSet(instruments=tuple(
        _instrument_from(ins, f"{path}.instruments[{i}]")
        for i, ins in enumerate(d["instruments"])))


def spec_from_dict(d: object) -> StrategySpec:
    """Reconstruct a typed ``StrategySpec`` from an emitted ``spec_*.json`` dict.

    Strict inverse of ``StrategySpec.to_dict``: exact key sets (modulo the
    documented conditional omissions), every schema shape guard re-runs via the
    constructors, and ``spec_from_dict(d).to_dict() == d`` (the round-trip law
    pinned by tests/unit/test_spec_loader.py over the real emitted runs)."""
    d = _require_mapping(d, "spec")
    _check_keys(d, "spec", ("header", "part1", "part2", "paper_facts"),
                ("estimation", "instruments"))
    paper_facts = d["paper_facts"]
    return StrategySpec(
        header=_header_from(d["header"], "spec.header"),
        part1=_part1_from(d["part1"], "spec.part1"),
        part2=_part2_from(d["part2"], "spec.part2"),
        paper_facts=(_paper_facts_from(paper_facts, "spec.paper_facts")
                     if paper_facts is not None else None),
        estimation=(_estimation_from(d["estimation"], "spec.estimation")
                    if "estimation" in d else None),
        instruments=(_instrument_set_from(d["instruments"], "spec.instruments")
                     if "instruments" in d else None),
    )
