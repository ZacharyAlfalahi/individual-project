"""
Signal Concept Registry loader (D22).

Signal identity is a *menu pick*, not a computation (D22): every characteristic
reference in a spec (the sort signal, the control axis of a double sort) names a
concept from this registry, or takes the first-class escape ``"unrecognised"``.
This module loads the versioned registry file into a typed
``SignalConceptRegistry`` and exposes exactly the two questions a validator asks:

  * ``has_concept(concept_id) -> bool``          -- is it a known concept?
  * ``parameter_schema(concept_id) -> {name: type}`` -- its parameter dials.

so a ``SignalConceptRegistry`` instance IS a valid ``SignalRegistryLike`` (the
duck-typed Protocol in ``validators/spec_validators.py``) and drops straight into
``validate_librarian_spec`` with no adapter.

**Wall-split (D22).** The registry carries id / definition / aliases / parameter
schema only -- *no column names*. The concept->column table lives on the Quant
side of the wall; nothing here imports the engine or a column name.

**Registry parameters (boundary resolutions).** Per the Part 2 field
inventory's registry-parameter appendix, the ledger surfaces exactly two
candidate registry parameters, both still ``boundary_flag`` at v1:

  * ``lag``       -- a *cross-entry* signal-timing dial (v2-6); competes with the
                     common ``signal_lag`` Part 2 field.
  * ``transform`` -- a *per-entry* pre-processing dial (v2-9); competes with the
                     common ``signal_transform`` Part 2 field.

No entry-specific numeric parameter (window / months / min-obs) arises from the
ledger -- signal-construction windows (e.g. "6-month" momentum) are baked into
each concept's *identity*, not a parameter. The v1 seed therefore keeps
parameter schemas minimal and marks the two candidates in the file.

**Versioning + hashing (D22).** The registry declares a ``version`` and computes a
``content_hash`` = sha256 over the *canonically serialised* (sorted-key JSON)
concept contents, so two loads of the same file produce byte-identical hashes and
the version stamps into every spec header. The hash covers the concept semantics
(id / definition / aliases / parameter schema) -- not YAML formatting or comments.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Mapping

import yaml

from ..errors import LibrarianSchemaError

# Default registry location (co-located with the librarian data).
_DEFAULT_REGISTRY_PATH = Path(__file__).resolve().parent.parent / "data" / "signal_concept_registry.yaml"

# The type-token vocabulary: YAML strings -> Python types (D22 parameter schema).
# Deliberately tiny -- the registry describes signal *identity*, not computation.
_TYPE_TOKENS: dict[str, type] = {
    "int": int,
    "float": float,
    "str": str,
    "bool": bool,
}


def _type_of_token(token: str) -> type:
    if token not in _TYPE_TOKENS:
        raise LibrarianSchemaError(
            f"unknown parameter type-token {token!r}; expected one of {sorted(_TYPE_TOKENS)}"
        )
    return _TYPE_TOKENS[token]


@dataclass(frozen=True)
class SignalConcept:
    """One registry concept (D22): a stable id, a human definition, alias strings
    the extractor may see in a paper, and a parameter schema mapping each dial's
    name to a Python type. **No column name** -- the concept->column binding is a
    Quant-side artifact (the wall split)."""

    concept_id: str
    definition: str
    aliases: tuple[str, ...]
    parameter_schema: Mapping[str, type]

    def __post_init__(self) -> None:
        if not isinstance(self.concept_id, str) or self.concept_id.strip() == "":
            raise LibrarianSchemaError("SignalConcept.concept_id must be a non-empty string")
        if not isinstance(self.definition, str) or self.definition.strip() == "":
            raise LibrarianSchemaError(
                f"SignalConcept.definition must be a non-empty string (concept {self.concept_id!r})"
            )
        if isinstance(self.aliases, list):
            object.__setattr__(self, "aliases", tuple(self.aliases))
        if not isinstance(self.aliases, tuple) or any(
            not isinstance(a, str) or a.strip() == "" for a in self.aliases
        ):
            raise LibrarianSchemaError(
                f"SignalConcept.aliases must be a tuple of non-empty strings (concept {self.concept_id!r})"
            )
        if not isinstance(self.parameter_schema, Mapping):
            raise LibrarianSchemaError(
                f"SignalConcept.parameter_schema must be a Mapping (concept {self.concept_id!r})"
            )
        for name, typ in self.parameter_schema.items():
            if not isinstance(name, str) or name.strip() == "":
                raise LibrarianSchemaError(
                    f"parameter names must be non-empty strings (concept {self.concept_id!r})"
                )
            if not isinstance(typ, type):
                raise LibrarianSchemaError(
                    f"parameter {name!r} type must be a Python type (concept {self.concept_id!r})"
                )
        # Freeze the schema mapping so the concept stays genuinely immutable.
        object.__setattr__(self, "parameter_schema", MappingProxyType(dict(self.parameter_schema)))

    def _canonical(self) -> dict:
        """The hash-covered, formatting-independent view of this concept: id,
        definition, sorted aliases, and parameter schema as ``{name: token}``
        (types mapped back to their stable token so the hash never depends on a
        Python ``type`` repr)."""
        token_of = {v: k for k, v in _TYPE_TOKENS.items()}
        return {
            "concept_id": self.concept_id,
            "definition": self.definition,
            "aliases": sorted(self.aliases),
            "parameter_schema": {
                name: token_of[typ] for name, typ in sorted(self.parameter_schema.items())
            },
        }


@dataclass(frozen=True)
class SignalConceptRegistry:
    """The loaded Signal Concept Registry (D22): a ``version`` + the concepts,
    keyed by id. Implements ``SignalRegistryLike`` (``has_concept`` +
    ``parameter_schema``) so an instance drops straight into
    ``validate_librarian_spec``.

    ``content_hash`` = sha256 over the canonically serialised (sorted-key JSON)
    concept contents -- reproducible across loads, independent of YAML formatting
    or comment edits."""

    version: str
    concepts: tuple[SignalConcept, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.version, str) or self.version.strip() == "":
            raise LibrarianSchemaError("SignalConceptRegistry.version must be a non-empty string")
        seen: set[str] = set()
        for c in self.concepts:
            if not isinstance(c, SignalConcept):
                raise LibrarianSchemaError(
                    f"SignalConceptRegistry.concepts entries must be SignalConcept; got {type(c).__name__}"
                )
            if c.concept_id in seen:
                raise LibrarianSchemaError(
                    f"duplicate concept id {c.concept_id!r} -- one entry per concept (D22)"
                )
            seen.add(c.concept_id)
        # index for O(1) lookup; kept private (the public surface is the protocol).
        object.__setattr__(self, "_by_id", {c.concept_id: c for c in self.concepts})

    # --- SignalRegistryLike protocol -------------------------------------------

    def has_concept(self, concept_id: str) -> bool:
        """Is ``concept_id`` a known registry concept? Exact + binary (D22): no
        alias resolution here -- aliases are for the extractor's prompt, the
        registry key is the canonical id."""
        return concept_id in self._by_id  # type: ignore[attr-defined]

    def parameter_schema(self, concept_id: str) -> Mapping[str, type]:
        """The concept's parameter schema ``{name -> python type}``. Raises if the
        concept is unknown (callers gate with ``has_concept`` first, matching the
        protocol contract)."""
        try:
            return self._by_id[concept_id].parameter_schema  # type: ignore[attr-defined]
        except KeyError as exc:
            raise LibrarianSchemaError(
                f"parameter_schema called for unknown concept {concept_id!r}"
            ) from exc

    # --- hashing ----------------------------------------------------------------

    @property
    def content_hash(self) -> str:
        """sha256 over the canonically serialised concept contents (sorted-key
        JSON, concepts sorted by id). Reproducible across loads; independent of
        YAML formatting/comments."""
        canonical = [c._canonical() for c in sorted(self.concepts, key=lambda c: c.concept_id)]
        payload = json.dumps(canonical, sort_keys=True, ensure_ascii=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def get(self, concept_id: str) -> SignalConcept | None:
        return self._by_id.get(concept_id)  # type: ignore[attr-defined]

    def ids(self) -> tuple[str, ...]:
        return tuple(c.concept_id for c in self.concepts)


def load_signal_concept_registry(path: str | Path | None = None) -> SignalConceptRegistry:
    """Load and validate the Signal Concept Registry file into typed concepts.

    The file declares a top-level ``version`` and a ``concepts`` list; each concept
    carries ``id``, ``definition``, ``aliases`` and a ``parameter_schema`` mapping
    param names to type-tokens (``"int"`` / ``"float"`` / ``"str"`` / ``"bool"``).
    A ``column`` (or any column-name key) anywhere is a wall-split violation and
    raises -- the registry may never carry an engine column name (D22)."""
    p = Path(path) if path is not None else _DEFAULT_REGISTRY_PATH
    if not p.exists():
        raise LibrarianSchemaError(f"signal concept registry not found at {p}")
    with p.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    if not isinstance(raw, dict):
        raise LibrarianSchemaError("signal concept registry must be a mapping at top level")

    version = raw.get("version")
    if not isinstance(version, str) or version.strip() == "":
        raise LibrarianSchemaError("signal concept registry must declare a non-empty 'version'")

    raw_concepts = raw.get("concepts")
    if not isinstance(raw_concepts, list) or len(raw_concepts) == 0:
        raise LibrarianSchemaError("signal concept registry 'concepts' must be a non-empty list")

    concepts: list[SignalConcept] = []
    for i, rc in enumerate(raw_concepts):
        if not isinstance(rc, dict):
            raise LibrarianSchemaError(f"signal concept {i} must be a mapping")
        # Wall-split guard (D22): no column names, ever, Librarian-side.
        for banned in ("column", "columns", "column_name", "panel_column"):
            if banned in rc:
                raise LibrarianSchemaError(
                    f"signal concept {i} carries a column key {banned!r} -- the registry is "
                    "wall-split (D22): ids/definitions/aliases/params only, no column names"
                )
        try:
            cid = rc["id"]
            definition = rc["definition"]
        except KeyError as exc:
            raise LibrarianSchemaError(f"signal concept {i} missing required key {exc}") from exc
        aliases = rc.get("aliases", [])
        if not isinstance(aliases, list):
            raise LibrarianSchemaError(f"signal concept {cid!r} aliases must be a list")
        raw_schema = rc.get("parameter_schema", {})
        if not isinstance(raw_schema, dict):
            raise LibrarianSchemaError(f"signal concept {cid!r} parameter_schema must be a mapping")
        parameter_schema = {str(name): _type_of_token(str(token)) for name, token in raw_schema.items()}
        concepts.append(
            SignalConcept(
                concept_id=str(cid),
                definition=str(definition),
                aliases=tuple(str(a) for a in aliases),
                parameter_schema=parameter_schema,
            )
        )
    return SignalConceptRegistry(version=str(version), concepts=tuple(concepts))
