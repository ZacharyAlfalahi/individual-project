"""
Model client boundary (build brief §5.3, D9/D10/D33).

The pipeline talks to its two extraction models through ONE tiny protocol,
``ModelClient``. A model is asked a single ``FieldQuery`` (a field id + the
structured-decoding contract that field uses) against a ``CanonicalText`` and
answers with a ``ModelAnswer`` (a raw typed value + the verbatim quote it claims
supports that value). Nothing here knows about vendors, SDKs, API keys, or the
network -- those live behind a real implementation of the protocol landed with
the model-pair experiment (D33). This module ships only:

  * ``FieldQuery``  -- the immutable request (field id, kind, template/schema
                       hashes, the k=1-primary sampling parameter as a NAMED but
                       unused seam per D10);
  * ``ModelAnswer`` -- the immutable reply (raw value + optional quote, or the
                       explicit "paper is silent" answer, ``answered=False``);
  * ``ModelClient`` -- the ``Protocol`` the pipeline depends on;
  * ``FakeModelClient`` -- a fully offline, deterministic, scripted client used
                       by every unit test (dependency-injected as model_a /
                       model_b). No I/O of any kind.

D9 lives one layer up (the form-filler compares two ``ModelAnswer``s on their
NORMALISED typed values and locates each quote); this module is purely the
request/response value objects + the offline stand-in.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from ..config.canonical_text import CanonicalText
from ..errors import LibrarianSchemaError

# The field kinds a query can carry -- one per prompt-template family (build brief
# §5.3): the enum menus, the ints, a SignalRef pick, the method_summary free
# text, the Part-1 enum menus, and (v1.1) the two paper_facts types -- a
# YYYY-MM date and the composite {mean, t_stat, unit} claimed headline metric.
# Matches data/prompts/manifest.yaml templates.
FIELD_KINDS: frozenset[str] = frozenset(
    ("enum", "int", "signal_ref", "method_summary", "part1_enum", "date", "paper_metric",
     # Scope B (2026-09-04): fitted-model estimation kinds (forked templates;
     # never bound to any sort field).
     "estimation_enum", "int_set",
     # Rubric freeze (2026-09-04): the declared-weaker prose kind (3 KPP fields).
     "prose")
)


@dataclass(frozen=True)
class FieldQuery:
    """One immutable extraction request for a single field.

    ``field`` is the schema field name (e.g. ``"weighting_scheme"``);
    ``kind`` selects the prompt-template family (``FIELD_KINDS``);
    ``template_hash`` / ``schema_hash`` / ``decoding_hash`` pin the exact
    parameterised prompt + JSON schema + decoding config used (stamped into the
    trace + header for reproducibility). ``k`` is the D10 self-consistency
    sampling count -- a NAMED but unused seam: v1 is k=1 (primary), the k=3 /
    2-of-3 ladder is pre-registered, not built."""

    field: str
    kind: str
    template_hash: str | None = None
    schema_hash: str | None = None
    decoding_hash: str | None = None
    k: int = 1

    def __post_init__(self) -> None:
        if not isinstance(self.field, str) or self.field.strip() == "":
            raise LibrarianSchemaError("FieldQuery.field must be a non-empty string")
        if self.kind not in FIELD_KINDS:
            raise LibrarianSchemaError(
                f"FieldQuery.kind must be one of {sorted(FIELD_KINDS)}; got {self.kind!r}"
            )
        for name, val in (
            ("template_hash", self.template_hash),
            ("schema_hash", self.schema_hash),
            ("decoding_hash", self.decoding_hash),
        ):
            if val is not None and (not isinstance(val, str) or val.strip() == ""):
                raise LibrarianSchemaError(f"FieldQuery.{name} must be a non-empty string or None")
        if not isinstance(self.k, int) or isinstance(self.k, bool) or self.k < 1:
            raise LibrarianSchemaError(f"FieldQuery.k must be an int >= 1 (D10); got {self.k!r}")


@dataclass(frozen=True)
class ModelAnswer:
    """One immutable model reply for a ``FieldQuery``.

    ``answered`` is the explicit "did the model find the fact?" bit: ``False``
    means the model reports the paper is SILENT (its raw value + quote are
    ignored). When ``answered`` is ``True``:

      * ``raw`` is the model's typed answer *before* normalisation (the
        form-filler normalises + compares); for a SignalRef field ``raw`` is the
        concept id string (or ``"unrecognised"``);
      * ``quote`` is the verbatim span the model claims supports ``raw`` -- it
        must later locate in the canonical text (the D9 quote gate). ``None`` is
        legal only when ``answered`` is ``False``.

    method_summary answers carry the summary text in ``raw`` and up to three
    supporting quotes in ``quotes`` (the single ``quote`` field is the primary /
    earliest for the 3-slot ship rule)."""

    field: str
    answered: bool
    raw: Any = None
    quote: str | None = None
    quotes: tuple[str, ...] = ()
    model_id: str | None = None
    # B2: True when this silence is a FORMAT/SCHEMA failure (unparseable JSON, or
    # answered:true with an unusable value/quote shape) rather than the model
    # reporting genuine paper silence. Additive, default False; carried into the
    # per-model trace so the §3.6 gate can separate the quote/format bucket from
    # genuine silence. Never True on an answered reply.
    parse_failed: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.field, str) or self.field.strip() == "":
            raise LibrarianSchemaError("ModelAnswer.field must be a non-empty string")
        if not isinstance(self.answered, bool):
            raise LibrarianSchemaError("ModelAnswer.answered must be a bool")
        if not isinstance(self.parse_failed, bool):
            raise LibrarianSchemaError("ModelAnswer.parse_failed must be a bool")
        if self.parse_failed and self.answered:
            raise LibrarianSchemaError(
                "ModelAnswer.parse_failed=True contradicts answered=True -- a format "
                "failure is a kind of silence, never an answer"
            )
        if isinstance(self.quotes, list):
            object.__setattr__(self, "quotes", tuple(self.quotes))
        if not isinstance(self.quotes, tuple) or any(not isinstance(q, str) for q in self.quotes):
            raise LibrarianSchemaError("ModelAnswer.quotes must be a tuple of strings")
        if self.answered:
            if self.quote is None and len(self.quotes) == 0:
                raise LibrarianSchemaError(
                    "an answered ModelAnswer must carry at least one quote (the span it claims "
                    "supports its value) -- the D9 quote gate has nothing to locate otherwise"
                )
        # A silent answer carries no value/quote (defensive: normalise to None).
        else:
            object.__setattr__(self, "raw", None)
            object.__setattr__(self, "quote", None)
            object.__setattr__(self, "quotes", ())

    def to_dict(self) -> dict:
        return {
            "field": self.field,
            "answered": self.answered,
            "raw": self.raw,
            "quote": self.quote,
            "quotes": list(self.quotes),
            "model_id": self.model_id,
            "parse_failed": self.parse_failed,
        }


@runtime_checkable
class ModelClient(Protocol):
    """The pipeline's dependency on an extraction model: answer one field query
    against one canonical text. A real (vendor) client and the offline
    ``FakeModelClient`` both satisfy it; the pipeline is injected two of them
    (model_a, model_b) and never imports a concrete client."""

    model_id: str

    def answer(self, query: FieldQuery, canonical_text: CanonicalText) -> ModelAnswer:
        ...


@dataclass
class FakeModelClient:
    """A deterministic, fully offline ``ModelClient`` for tests.

    Scripted: ``answers`` maps a field id to a ``ModelAnswer`` (or a callable
    ``(query, canonical_text) -> ModelAnswer`` for the rare case a test needs the
    text). A field with no script returns a silent answer (``answered=False``) --
    the paper-is-silent path. No network, no SDK, no keys: it just looks up the
    script. Two instances (distinct ``model_id``s) are passed to the pipeline."""

    model_id: str = "fake-model"
    answers: dict[str, Any] = field(default_factory=dict)
    # Scripted enumeration output (WS-3): the construction tuple this fake "model"
    # returns from extract_enumeration. Empty by default (a paper with no scripted
    # constructions). Not part of the answer() per-field path.
    enumeration: tuple = ()

    def extract_enumeration(self, canonical_text: CanonicalText) -> tuple:
        """Return the scripted construction list (WS-3). Deterministic + offline:
        no locate, no network -- ``enumerate_constructions`` relocates downstream."""
        return tuple(self.enumeration)

    def answer(self, query: FieldQuery, canonical_text: CanonicalText) -> ModelAnswer:
        if not isinstance(query, FieldQuery):
            raise LibrarianSchemaError("FakeModelClient.answer expects a FieldQuery")
        scripted = self.answers.get(query.field)
        if scripted is None:
            # No script for this field -> the model reports the paper is silent.
            return ModelAnswer(field=query.field, answered=False, model_id=self.model_id)
        if callable(scripted):
            result = scripted(query, canonical_text)
        else:
            result = scripted
        if not isinstance(result, ModelAnswer):
            raise LibrarianSchemaError(
                f"FakeModelClient script for {query.field!r} must yield a ModelAnswer; "
                f"got {type(result).__name__}"
            )
        # Stamp this client's id if the script didn't set one (keeps traces honest).
        if result.model_id is None:
            result = ModelAnswer(
                field=result.field,
                answered=result.answered,
                raw=result.raw,
                quote=result.quote,
                quotes=result.quotes,
                model_id=self.model_id,
                parse_failed=result.parse_failed,   # B2: never drop the format signal
            )
        return result
