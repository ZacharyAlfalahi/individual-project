"""
Librarian registries -- the versioned, hashed reference data the pipeline reads
(never re-hardcodes).

  * ``signal_concept_registry`` -- the Signal Concept Registry (D22): the menu of
                                    signal identities a spec may reference. A
                                    ``SignalConceptRegistry`` implements
                                    ``SignalRegistryLike`` and drives
                                    ``validate_librarian_spec`` directly.
                                    Wall-split: ids/definitions/aliases/params
                                    only, no column names.
  * ``silence_policy``          -- the silence-policy table (D26 / D32c): per-field
                                    routing for paper-silent fields at adapter
                                    intake. Versioned + byte-hashed data.
  * ``field_definitions``       -- the frozen per-field definitions (RQ1
                                    close-out): the ``{definition}`` prompt slot,
                                    one gloss per routed field. Versioned +
                                    byte-hashed data.

Both loaders return frozen dataclasses carrying a ``version`` + a reproducible
``content_hash`` that stamps into every spec/config header (build brief §2).
"""

from __future__ import annotations

from .instrument_concept_registry import load_instrument_concept_registry
from .signal_concept_registry import (
    SignalConcept,
    SignalConceptRegistry,
    load_signal_concept_registry,
)
from .silence_policy import (
    POLICIES,
    FieldPolicy,
    SilencePolicyTable,
    load_silence_policy_table,
)
from .field_definitions import (
    FieldDefinitions,
    load_field_definitions,
)

__all__ = [
    # signal concept registry (D22)
    "SignalConcept",
    "SignalConceptRegistry",
    "load_signal_concept_registry",
    # instrument concept registry (v1.2, fitted-model)
    "load_instrument_concept_registry",
    # silence-policy table (D26 / D32c)
    "FieldPolicy",
    "SilencePolicyTable",
    "load_silence_policy_table",
    "POLICIES",
    # per-field definitions (RQ1 close-out)
    "FieldDefinitions",
    "load_field_definitions",
]
