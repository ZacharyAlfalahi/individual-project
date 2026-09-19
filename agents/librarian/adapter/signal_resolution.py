"""
Signal resolution (D27): a ``SignalRef`` -> a ``Binding``.

The adapter's third translation (D25). A concept id (+ its canonical parameter
tuple) is looked up in the Quant-side concept->column table:

  * hit                     -> ``Binding(BOUND, value=column, evidence.column=column)``
  * ``unrecognised`` escape -> ``Binding(MISSING, ...)`` (the paper's signal is not
                               one of ours)
  * no row / unknown concept -> ``Binding(MISSING, ...)``

A MISSING binding rides into ``build_quant_config`` unchanged, which emits the one
``MISSING_BINDING`` refusal (P4: a single emitter). The adapter NEVER emits
``AMBIGUOUS`` -- the table is a function (one column per (concept, params)), so a
resolved bind is always BOUND (the mirror rule; the concept->column loader rejects
duplicate keys, making AMBIGUOUS structurally impossible).

The registry-version handshake (spec.registry_version == table.registry_version)
is asserted ONCE at the adapter top level (``adapt.py``), which holds the spec;
this function assumes it has passed.
"""

from __future__ import annotations

from agents.quant.config import Binding, Evidence
from agents.quant.config.concept_column import ConceptColumnTable

from ..schema.signal_ref import SignalRef


def _param_values(signal_ref: SignalRef) -> dict[str, object]:
    """The concept's parameter dials as ``{name: value}`` (the Inherited values),
    for the concept->column lookup key. Empty for every v1 concept."""
    return {name: inh.value for name, inh in signal_ref.parameters.items()}


def resolve_signal(
    signal_ref: SignalRef,
    concept_table: ConceptColumnTable,
    *,
    override_column: str | None = None,
) -> Binding:
    """Resolve a ``SignalRef`` to a ``Binding`` via the concept->column table.

    ``override_column`` is a human authorisation-record binding substitution (D27):
    a hand-chosen column that bypasses the table lookup. It binds BOUND (the
    ``variant`` flag records the authorisation; Binding has no DESIGN tag) with a
    note marking it authorised -- the caller sets ``variant``."""
    label = signal_ref.as_described.label

    if override_column is not None:
        return Binding(
            override_column,
            "BOUND",
            Evidence(
                column=override_column,
                note=(
                    f"authorised binding substitution (variant, D27): "
                    f"{label!r} -> column {override_column!r}"
                ),
            ),
        )

    # The 'unrecognised' escape: the paper's signal is not one of ours (D22) -> a
    # MISSING binding (there is nothing to bind), which the factory refuses.
    if signal_ref.is_unrecognised:
        return Binding(
            None,
            "MISSING",
            Evidence(note=f"signal unrecognised (not a registry concept): {label!r}"),
        )

    concept_id = signal_ref.concept_id.value
    if not isinstance(concept_id, str) or concept_id.strip() == "":
        # A non-STATED / empty concept id has nothing to bind (defence in depth --
        # a silent sort_signal is caught by the silence router before this).
        return Binding(
            None,
            "MISSING",
            Evidence(note=f"signal concept id absent/empty for {label!r}; nothing to bind"),
        )

    params = _param_values(signal_ref)
    column = concept_table.lookup(concept_id, params)
    if column is None:
        return Binding(
            None,
            "MISSING",
            Evidence(
                note=(
                    f"no concept->column row for concept {concept_id!r} "
                    f"(params={params or '{}'}, registry {concept_table.registry_version}); "
                    "signal has no panel column"
                )
            ),
        )

    return Binding(
        column,
        "BOUND",
        Evidence(
            column=column,
            note=(
                f"bound concept {concept_id!r} -> column {column!r} "
                f"(registry {concept_table.registry_version}; adapter signal_resolution)"
            ),
        ),
    )
