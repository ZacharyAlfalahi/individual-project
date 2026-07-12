"""
Librarian-output spec validator (D8, D22, D24).

``validate_librarian_spec`` runs an already-constructed ``StrategySpec`` through
the Librarian-output policy gate and *returns a list of*
``LibrarianValidationError`` (it does not raise) so a caller can report every
violation in one pass.

The four mandatory D8 negatives:

  (1) **No DESIGN anywhere.** The ``Inherited`` type admits DESIGN (D2), but the
      Librarian's output forbids it (D8): DESIGN is a project decision made
      config-side, never a product of extraction. Any ``Inherited.tag ==
      "DESIGN"`` reachable from the spec -> error.

  (2) **STATED implies a locator.** Construction already blocks a STATED without
      a locator (``Inherited.__post_init__``, D7); re-checked here as
      defence-in-depth so the invariant is asserted at the output boundary too.

  (3) **Registry-aware SignalRef.** With a Signal Concept Registry passed in:
      an unknown ``concept_id`` is rejected, parameter names/types are checked
      against the concept's parameter schema, and the ``unrecognised`` escape
      requires a non-empty ``as_described`` (the pure-value SignalRef already
      guards the last one structurally; re-checked here).

  (4) **Adapter-never-AMBIGUOUS invariant.** No ``AMBIGUOUS`` may appear in
      Librarian output. Structurally it cannot -- Librarian output uses
      ``Inherited`` (STATED/INFERRED/DESIGN/UNKNOWN), never ``Binding``
      (BOUND/AMBIGUOUS/MISSING) -- so this is an assertion that the structural
      guarantee holds, backed by a negative test.

Two optional registries are passed in (never imported/loaded here, to keep the
validator a pure function of its inputs):

  * ``registry``           -- the Signal Concept Registry (membership + param
                              schema). A minimal duck-typed protocol
                              (``SignalRegistryLike``) so this module need not
                              import the registry's concrete type (D22: the
                              registry loader is a separate artifact). When
                              ``None``, the registry-aware checks are skipped.
  * ``tag_reason_registry``-- the tag-reason registry (D24). When provided,
                              every reachable ``Inherited`` (tag, reason) pair is
                              asserted to be a registered row.
"""

from __future__ import annotations

from typing import Iterable, Mapping, Protocol, runtime_checkable

from agents.quant.config import Inherited
from agents.quant.config.provenance import Binding

from ..errors import LibrarianSchemaError, LibrarianValidationError
from ..schema.signal_ref import UNRECOGNISED, SignalRef
from ..schema.strategy_spec import StrategySpec
from .tag_reason import TagReasonRegistry


# ---------------------------------------------------------------------------
# The Signal Concept Registry protocol (duck-typed; the concrete loader is a
# separate D22 artifact). A registry need only answer two questions.
# ---------------------------------------------------------------------------

@runtime_checkable
class SignalRegistryLike(Protocol):
    def has_concept(self, concept_id: str) -> bool:
        """Is ``concept_id`` a known registry concept?"""
        ...

    def parameter_schema(self, concept_id: str) -> Mapping[str, type]:
        """The concept's parameter schema: ``{param_name -> python type}``.
        Called only for known concepts."""
        ...


def _reason_of(inh: Inherited) -> str:
    """Best-effort reason code for a value's tag, for the tag-reason check.
    STATED -> 'quoted'; UNKNOWN/INFERRED/DESIGN carry their reason in
    evidence.unknown_reason / rule_id / note; when absent we fall back to a
    conventional reason so the registry lookup is still meaningful."""
    ev = inh.evidence
    if inh.tag == "STATED":
        return "quoted"
    if inh.tag == "UNKNOWN":
        return ev.unknown_reason or "not_stated"
    if inh.tag == "INFERRED":
        # rule_id carries the namespace/reason (e.g. 'adapter/R1'); the reason
        # token is registered per-namespace, so we map by shape.
        rid = ev.rule_id or ""
        if rid.startswith("adapter/") or rid.startswith("adapter"):
            return "adapter"
        if rid.startswith("default/"):
            return "default/<field>"
        return "inference"
    # DESIGN -- reason is one of the three DESIGN reasons; note carries it, but
    # DESIGN is rejected by check (1) regardless, so a coarse fallback is fine.
    return "derived_from_design"


def _iter_signal_refs(spec: StrategySpec) -> Iterable[tuple[str, SignalRef]]:
    """Yield (path, SignalRef) for every SignalRef in the spec."""
    for i, leg in enumerate(spec.part2.legs):
        yield f"part2.legs[{i}].sort_signal", leg.sort_signal
        if leg.control_axis is not None:
            yield f"part2.legs[{i}].control_axis", leg.control_axis


def _iter_inherited(spec: StrategySpec) -> Iterable[tuple[str, Inherited]]:
    """Yield (path, Inherited) for every Inherited reachable from the spec --
    header, Part 1, Part 2 common fields, per-leg fields, SignalRef concept_ids
    and parameters, combiner."""
    h = spec.header
    yield "header.strategy_label", h.strategy_label

    p1 = spec.part1
    yield "part1.formation_structure", p1.formation_structure
    yield "part1.asset_class", p1.asset_class
    yield "part1.method_summary.summary", p1.method_summary.summary

    p2 = spec.part2
    from ..schema.strategy_spec import _COMMON_INHERITED_FIELDS, _LEG_INHERITED_FIELDS

    for name in _COMMON_INHERITED_FIELDS:
        yield f"part2.{name}", getattr(p2, name)
    yield "part2.combiner.kind", p2.combiner.kind

    for i, leg in enumerate(p2.legs):
        for name in _LEG_INHERITED_FIELDS:
            yield f"part2.legs[{i}].{name}", getattr(leg, name)
        for label, sig in (("sort_signal", leg.sort_signal), ("control_axis", leg.control_axis)):
            if sig is None:
                continue
            yield f"part2.legs[{i}].{label}.concept_id", sig.concept_id
            for pname, pval in sig.parameters.items():
                yield f"part2.legs[{i}].{label}.parameters.{pname}", pval


# ---------------------------------------------------------------------------
# The four D8 negatives.
# ---------------------------------------------------------------------------

def _check_no_design(spec: StrategySpec) -> list[LibrarianValidationError]:
    errors: list[LibrarianValidationError] = []
    for path, inh in _iter_inherited(spec):
        if inh.tag == "DESIGN":
            errors.append(
                LibrarianValidationError(
                    path, "DESIGN tag is forbidden in Librarian output (D8)"
                )
            )
    return errors


def _check_stated_has_locator(spec: StrategySpec) -> list[LibrarianValidationError]:
    errors: list[LibrarianValidationError] = []
    for path, inh in _iter_inherited(spec):
        if inh.tag == "STATED" and inh.evidence.locator is None:
            errors.append(
                LibrarianValidationError(
                    path, "STATED value is missing its locator (D7)"
                )
            )
    return errors


def _check_no_ambiguous(spec: StrategySpec) -> list[LibrarianValidationError]:
    """Adapter-never-AMBIGUOUS: Librarian output uses Inherited, never Binding,
    so a Binding (which alone can carry AMBIGUOUS) must not appear. Structurally
    guaranteed; asserted here."""
    errors: list[LibrarianValidationError] = []
    for path, inh in _iter_inherited(spec):
        if isinstance(inh, Binding):  # pragma: no cover -- structurally impossible
            errors.append(
                LibrarianValidationError(
                    path, "AMBIGUOUS/Binding value in Librarian output is forbidden"
                )
            )
    return errors


def _check_signal_registry(
    spec: StrategySpec,
    registry: SignalRegistryLike | None,
) -> list[LibrarianValidationError]:
    errors: list[LibrarianValidationError] = []
    for path, sig in _iter_signal_refs(spec):
        cid = sig.concept_id.value

        # unrecognised: requires non-empty as_described (also guarded in SignalRef).
        if cid == UNRECOGNISED:
            if len(sig.as_described.quotes) == 0:
                errors.append(
                    LibrarianValidationError(
                        path,
                        "unrecognised SignalRef requires non-empty as_described.quotes (D22)",
                    )
                )
            continue  # unrecognised is deliberately not in the registry

        if registry is None:
            continue  # registry-aware checks skipped when no registry supplied

        if cid is None or not isinstance(cid, str):
            errors.append(
                LibrarianValidationError(path, f"SignalRef.concept_id.value must be a str; got {cid!r}")
            )
            continue

        if not registry.has_concept(cid):
            errors.append(
                LibrarianValidationError(
                    path,
                    f"concept_id {cid!r} is not in the Signal Concept Registry "
                    f"(D22: exact match; the correct answer for an out-of-registry "
                    f"signal is {UNRECOGNISED!r})",
                )
            )
            continue

        # parameter-schema enforcement: names + types.
        schema = registry.parameter_schema(cid)
        for pname, pval in sig.parameters.items():
            if pname not in schema:
                errors.append(
                    LibrarianValidationError(
                        f"{path}.parameters.{pname}",
                        f"parameter {pname!r} is not in the schema for concept {cid!r}",
                    )
                )
                continue
            expected = schema[pname]
            val = pval.value
            # UNKNOWN values legitimately carry None -- a paper-silent parameter.
            if val is None:
                continue
            if isinstance(val, bool) and expected is not bool:
                errors.append(
                    LibrarianValidationError(
                        f"{path}.parameters.{pname}",
                        f"parameter {pname!r} must be {expected.__name__}; got bool",
                    )
                )
            elif not isinstance(val, expected):
                errors.append(
                    LibrarianValidationError(
                        f"{path}.parameters.{pname}",
                        f"parameter {pname!r} must be {expected.__name__}; got "
                        f"{type(val).__name__}",
                    )
                )
    return errors


def _check_tag_reason(
    spec: StrategySpec,
    tag_reason_registry: TagReasonRegistry | None,
) -> list[LibrarianValidationError]:
    """Every reachable Inherited's (tag, reason) must be a registered row (D24).
    Skipped when no tag-reason registry is supplied."""
    if tag_reason_registry is None:
        return []
    errors: list[LibrarianValidationError] = []
    for path, inh in _iter_inherited(spec):
        reason = _reason_of(inh)
        if not tag_reason_registry.has(inh.tag, reason):
            errors.append(
                LibrarianValidationError(
                    path,
                    f"tag {inh.tag!r} for reason {reason!r} is not a registered "
                    "tag-reason row (D24)",
                )
            )
    return errors


# The two double-sort kinds: a declared double sort must name its control axis.
_DOUBLE_SORT_KINDS: frozenset[str] = frozenset({"independent", "conditional"})


def _check_sort_structure(spec: StrategySpec) -> list[LibrarianValidationError]:
    """Guard 1 (schema-v1.1 §3): per-leg sort-structure consistency between
    ``sort_kind`` and the presence of a ``control_axis``. A deterministic
    intra-spec check (D18 family); NEVER auto-repaired (D14) -- a violation is a
    typed outcome (two extracted fields contradicting), not a fix-up.

    Enforced in ``control_axis`` terms (the two implications; the brief's XOR
    shorthand is imprecise for ``other``/UNKNOWN, which these handle correctly):
      * ``sort_kind ∈ {independent, conditional}`` ⇒ ``control_axis is not None``
        -- a declared double sort with no named 2nd axis is identity-missing;
      * ``control_axis is not None`` ⇒ ``sort_kind != "single"``
        -- a single sort cannot carry a 2nd axis.
    ``other`` / UNKNOWN ``sort_kind`` are neither ``single`` nor a declared double
    sort, so they trigger neither rule: a *silent* sort_kind is the silence
    policy's job (conditional_refuse when a control axis is present), not this
    structural check, which fires only on a STATED contradiction."""
    errors: list[LibrarianValidationError] = []
    for i, leg in enumerate(spec.part2.legs):
        sort_kind = leg.sort_kind.value
        control_present = leg.control_axis is not None
        if sort_kind in _DOUBLE_SORT_KINDS and not control_present:
            errors.append(
                LibrarianValidationError(
                    f"part2.legs[{i}].control_axis",
                    f"sort_kind={sort_kind!r} is a double sort but control_axis is absent -- "
                    "a declared double sort must name its 2nd axis (Guard 1, schema-v1.1 §3)",
                )
            )
        if control_present and sort_kind == "single":
            errors.append(
                LibrarianValidationError(
                    f"part2.legs[{i}].sort_kind",
                    "sort_kind='single' but a control_axis is present -- a single sort "
                    "cannot carry a 2nd axis (Guard 1, schema-v1.1 §3)",
                )
            )
    return errors


def validate_librarian_spec(
    spec: StrategySpec,
    registry: SignalRegistryLike | None = None,
    tag_reason_registry: TagReasonRegistry | None = None,
) -> list[LibrarianValidationError]:
    """Validate a Librarian-output spec against the D8 negatives + (optionally)
    registry membership and the tag-reason registry. Returns the full list of
    violations (empty = clean); does not raise on policy violations.

    Raises ``LibrarianSchemaError`` only if ``spec`` is not a ``StrategySpec``
    (a caller-contract bug, distinct from a policy finding)."""
    if not isinstance(spec, StrategySpec):
        raise LibrarianSchemaError(
            f"validate_librarian_spec expects a StrategySpec; got {type(spec).__name__}"
        )
    errors: list[LibrarianValidationError] = []
    errors += _check_no_design(spec)                       # (1) D8
    errors += _check_stated_has_locator(spec)              # (2) D7
    errors += _check_signal_registry(spec, registry)       # (3) D22
    errors += _check_no_ambiguous(spec)                    # (4) adapter-never-AMBIGUOUS
    errors += _check_sort_structure(spec)                  # Guard 1 (schema-v1.1 §3)
    errors += _check_tag_reason(spec, tag_reason_registry)  # D24 (optional)
    return errors
