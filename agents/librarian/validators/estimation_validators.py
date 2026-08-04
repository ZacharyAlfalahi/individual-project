"""
Estimation-block validator (schema v1.2) -- the fitted-factor-model analogue of
``spec_validators.validate_librarian_spec``, kept as a SEPARATE pass so the shared
sort-path validator (`spec_validators.py`) stays byte-identical and a bug here can
never touch a sort spec.

``validate_estimation_block`` walks the (optional) ``spec.estimation`` +
``spec.instruments`` siblings -- which ``spec_validators._iter_inherited`` does
NOT reach -- and applies the same D8 negatives (no DESIGN; STATED implies a
locator) plus instrument-registry membership (the fitted-model analogue of the
SignalRef registry check). Returns a list of ``LibrarianValidationError`` (does
not raise on a policy finding); mirrors the sort validator's return contract so
callers report every violation in one pass.

The instrument registry is passed in as a duck-typed ``SignalRegistryLike``
(``has_concept`` + ``parameter_schema``) -- a ``SignalConceptRegistry`` loaded
from ``instrument_concept_registry.yaml`` drops straight in. When ``None``, the
membership check is skipped (shape/D8 checks still run).
"""

from __future__ import annotations

from typing import Iterable

from agents.quant.config import Inherited

from ..errors import LibrarianSchemaError, LibrarianValidationError
from ..schema.estimation_fields import ESTIMATION_FIELDS, INSTRUMENT_INHERITED_FIELDS
from ..schema.signal_ref import UNRECOGNISED
from ..schema.strategy_spec import InstrumentRef, StrategySpec
from .spec_validators import SignalRegistryLike


def _iter_estimation_inherited(spec: StrategySpec) -> Iterable[tuple[str, Inherited]]:
    """Yield (path, Inherited) for every Inherited reachable from the fitted-model
    siblings -- the 11 estimation fields + each instrument's concept_id and its
    source_class / transform / lag. Does NOT touch header / part1 / part2 (the
    sort validator owns those)."""
    est = spec.estimation
    if est is not None:
        for name in ESTIMATION_FIELDS:
            yield f"estimation.{name}", getattr(est, name)
    ins = spec.instruments
    if ins is not None:
        for i, instr in enumerate(ins.instruments):
            yield f"instruments[{i}].concept_id", instr.concept_id
            for name in INSTRUMENT_INHERITED_FIELDS:
                yield f"instruments[{i}].{name}", getattr(instr, name)


def _iter_instrument_refs(spec: StrategySpec) -> Iterable[tuple[str, InstrumentRef]]:
    """Yield (path, InstrumentRef) for every instrument in the set."""
    ins = spec.instruments
    if ins is None:
        return
    for i, instr in enumerate(ins.instruments):
        yield f"instruments[{i}]", instr


def _check_no_design(spec: StrategySpec) -> list[LibrarianValidationError]:
    errors: list[LibrarianValidationError] = []
    for path, inh in _iter_estimation_inherited(spec):
        if inh.tag == "DESIGN":
            errors.append(
                LibrarianValidationError(
                    path, "DESIGN tag is forbidden in Librarian output (D8)"
                )
            )
    return errors


def _check_stated_has_locator(spec: StrategySpec) -> list[LibrarianValidationError]:
    errors: list[LibrarianValidationError] = []
    for path, inh in _iter_estimation_inherited(spec):
        if inh.tag == "STATED" and inh.evidence.locator is None:
            errors.append(
                LibrarianValidationError(path, "STATED value is missing its locator (D7)")
            )
    return errors


def _check_instrument_registry(
    spec: StrategySpec,
    registry: SignalRegistryLike | None,
) -> list[LibrarianValidationError]:
    """Instrument-registry membership -- the fitted-model analogue of
    ``spec_validators._check_signal_registry``. An ``unrecognised`` instrument
    requires non-empty ``as_described.quotes`` (the paper's words that failed to
    match); a known instrument's ``concept_id`` must be in the registry (exact +
    binary, D22). Skipped when no registry is supplied."""
    errors: list[LibrarianValidationError] = []
    for path, instr in _iter_instrument_refs(spec):
        cid = instr.concept_id.value
        if cid == UNRECOGNISED:
            if len(instr.as_described.quotes) == 0:
                errors.append(
                    LibrarianValidationError(
                        f"{path}.concept_id",
                        "unrecognised instrument requires non-empty as_described.quotes (D22)",
                    )
                )
            continue
        if registry is None:
            continue
        if cid is None or not isinstance(cid, str):
            errors.append(
                LibrarianValidationError(
                    f"{path}.concept_id",
                    f"InstrumentRef.concept_id.value must be a str; got {cid!r}",
                )
            )
            continue
        if not registry.has_concept(cid):
            errors.append(
                LibrarianValidationError(
                    f"{path}.concept_id",
                    f"concept_id {cid!r} is not in the Instrument Concept Registry "
                    f"(D22: exact match; the correct answer for an out-of-registry "
                    f"instrument is {UNRECOGNISED!r})",
                )
            )
    return errors


def validate_estimation_block(
    spec: StrategySpec,
    registry: SignalRegistryLike | None = None,
) -> list[LibrarianValidationError]:
    """Validate the fitted-model siblings (``estimation`` + ``instruments``) of a
    ``StrategySpec`` against the D8 negatives + instrument-registry membership.
    Returns the full list of violations (empty = clean); does not raise on a
    policy finding.

    Raises ``LibrarianSchemaError`` only if ``spec`` is not a ``StrategySpec`` (a
    caller-contract bug). A spec with no fitted-model blocks (both siblings None)
    yields an empty list -- there is nothing to validate."""
    if not isinstance(spec, StrategySpec):
        raise LibrarianSchemaError(
            f"validate_estimation_block expects a StrategySpec; got {type(spec).__name__}"
        )
    errors: list[LibrarianValidationError] = []
    errors += _check_no_design(spec)
    errors += _check_stated_has_locator(spec)
    errors += _check_instrument_registry(spec, registry)
    return errors
