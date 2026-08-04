"""
Instrument Concept Registry loader (schema v1.2).

The fitted-factor-model analogue of the sort ``signal_concept_registry`` (D22):
the menu of INSTRUMENT identities a StrategySpec's ``instruments`` block may
reference (KPP Table A.I, 29 characteristics). Same file shape (id / definition /
aliases / parameter_schema) and the SAME typed model + hashing, so a loaded
instrument registry IS a valid ``SignalRegistryLike`` and drops straight into
``validate_estimation_block``.

**No loader duplication.** This module is a thin path-binding over
``signal_concept_registry.load_signal_concept_registry`` -- it just points the
generic loader at ``data/instrument_concept_registry.yaml``. The sort registry
file and its semantic ``content_hash`` are never read here, so the two registries
are physically independent (a defect in the instrument set cannot perturb the
sort registry hash that flows into every sort spec header).
"""

from __future__ import annotations

from pathlib import Path

from .signal_concept_registry import (
    SignalConcept,
    SignalConceptRegistry,
    load_signal_concept_registry,
)

# Default instrument-registry location (co-located with the librarian data).
_DEFAULT_INSTRUMENT_REGISTRY_PATH = (
    Path(__file__).resolve().parent.parent / "data" / "instrument_concept_registry.yaml"
)


def load_instrument_concept_registry(path: str | Path | None = None) -> SignalConceptRegistry:
    """Load and validate the Instrument Concept Registry into typed concepts.

    Delegates to the generic ``load_signal_concept_registry`` with the instrument
    file path (default ``data/instrument_concept_registry.yaml``); it therefore
    gets an independent ``content_hash`` and the same wall-split / duplicate-id
    guards. The returned ``SignalConceptRegistry`` satisfies ``SignalRegistryLike``
    (``has_concept`` + ``parameter_schema``)."""
    p = Path(path) if path is not None else _DEFAULT_INSTRUMENT_REGISTRY_PATH
    return load_signal_concept_registry(p)


__all__ = [
    "SignalConcept",
    "SignalConceptRegistry",
    "load_instrument_concept_registry",
]
