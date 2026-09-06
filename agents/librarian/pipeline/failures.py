"""
Librarian failure taxonomy (build brief §5.6, D31).

Five typed outcome events, NEVER tags (D24: tags describe values that exist; a
failure describes a run outcome). Each is carried on the run record with a
fixed ``routing``:

  * ``UnparseablePdf``            -> ``paper_failed`` (typed, counted)
  * ``PartialParse``             -> ``paper_failed`` (v1-conservative; degrade
                                    mode explicitly rejected for v1, D31)
  * ``LocatorSystematicFailure`` -> ``review`` (the paper's true-quote match rate
                                    fell below the calibrated bar)
  * ``EnumerationDisagreement``  -> ``review`` (the two models disagree on the
                                    construction list, D20)
  * ``AssemblyIncomplete``       -> ``review`` (one construction's sort signal
                                    failed to resolve; PER-CONSTRUCTION scope --
                                    the CI-9 amendment, 2026-09-06; the first
                                    four are paper-level, this one is not)

``FAILURE_ROUTING`` is the single source of truth for outcome -> route; each event
exposes its own ``routing`` via that table so a caller cannot mis-route one. The
routing vocabulary is closed: ``{"paper_failed", "review"}``.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..errors import LibrarianSchemaError

# The closed routing vocabulary (D31 / build brief §5.6).
PAPER_FAILED: str = "paper_failed"
REVIEW: str = "review"
ROUTES: frozenset[str] = frozenset((PAPER_FAILED, REVIEW))

# Outcome event kind -> route. The one authoritative mapping (build brief §5.6).
# "assembly_incomplete" added by the dated CI-9 amendment (2026-09-06): a
# PER-CONSTRUCTION review outcome (one construction's sort signal failed to
# resolve), scoped to that construction only -- the paper's remaining
# constructions proceed. Before CI-9 the same condition escaped as an uncaught
# RuntimeError and aborted the whole paper, an unregistered cascade.
FAILURE_ROUTING: dict[str, str] = {
    "unparseable_pdf": PAPER_FAILED,
    "partial_parse": PAPER_FAILED,
    "locator_systematic_failure": REVIEW,
    "enumeration_disagreement": REVIEW,
    "assembly_incomplete": REVIEW,
}


def _nonempty(name: str, value: object) -> None:
    if not isinstance(value, str) or value.strip() == "":
        raise LibrarianSchemaError(f"{name} must be a non-empty string")


@dataclass(frozen=True)
class UnparseablePdf:
    """The parser could not produce any canonical text for the paper.
    Routes to ``paper_failed`` (D31)."""

    paper_id: str
    detail: str

    kind: str = "unparseable_pdf"

    def __post_init__(self) -> None:
        _nonempty("UnparseablePdf.paper_id", self.paper_id)
        _nonempty("UnparseablePdf.detail", self.detail)

    @property
    def routing(self) -> str:
        return FAILURE_ROUTING[self.kind]

    def to_dict(self) -> dict:
        return {"kind": self.kind, "paper_id": self.paper_id, "detail": self.detail,
                "routing": self.routing}


@dataclass(frozen=True)
class PartialParse:
    """The parser produced only part of the paper (v1 treats this as a hard
    failure -- degrade mode is explicitly rejected for v1, D31). Routes to
    ``paper_failed``."""

    paper_id: str
    detail: str
    pages_recovered: int | None = None

    kind: str = "partial_parse"

    def __post_init__(self) -> None:
        _nonempty("PartialParse.paper_id", self.paper_id)
        _nonempty("PartialParse.detail", self.detail)
        if self.pages_recovered is not None and (
            not isinstance(self.pages_recovered, int) or isinstance(self.pages_recovered, bool)
        ):
            raise LibrarianSchemaError("PartialParse.pages_recovered must be an int or None")

    @property
    def routing(self) -> str:
        return FAILURE_ROUTING[self.kind]

    def to_dict(self) -> dict:
        return {"kind": self.kind, "paper_id": self.paper_id, "detail": self.detail,
                "pages_recovered": self.pages_recovered, "routing": self.routing}


@dataclass(frozen=True)
class LocatorSystematicFailure:
    """The paper's true-quote match rate fell below the calibrated bar -- the
    parser/normalisation ladder is systematically failing to locate this paper's
    quotes, so its STATED evidence cannot be trusted. Routes to ``review`` (D31)."""

    paper_id: str
    match_rate: float
    bar: float
    detail: str = ""

    kind: str = "locator_systematic_failure"

    def __post_init__(self) -> None:
        _nonempty("LocatorSystematicFailure.paper_id", self.paper_id)
        for name, val in (("match_rate", self.match_rate), ("bar", self.bar)):
            if isinstance(val, bool) or not isinstance(val, (int, float)):
                raise LibrarianSchemaError(f"LocatorSystematicFailure.{name} must be a number")
            if not 0.0 <= float(val) <= 1.0:
                raise LibrarianSchemaError(
                    f"LocatorSystematicFailure.{name} must be a rate in [0, 1]; got {val!r}"
                )

    @property
    def routing(self) -> str:
        return FAILURE_ROUTING[self.kind]

    def to_dict(self) -> dict:
        return {"kind": self.kind, "paper_id": self.paper_id, "match_rate": self.match_rate,
                "bar": self.bar, "detail": self.detail, "routing": self.routing}


@dataclass(frozen=True)
class EnumerationDisagreement:
    """The two models disagree on the paper's construction list (D20): the sets of
    construction names, or a construction's strategy/auxiliary class, differ.
    The paper goes to ``review`` -- never auto-reconciled. Routes to ``review``."""

    paper_id: str
    detail: str
    only_model_a: tuple[str, ...] = ()
    only_model_b: tuple[str, ...] = ()

    kind: str = "enumeration_disagreement"

    def __post_init__(self) -> None:
        _nonempty("EnumerationDisagreement.paper_id", self.paper_id)
        _nonempty("EnumerationDisagreement.detail", self.detail)
        for name in ("only_model_a", "only_model_b"):
            v = getattr(self, name)
            if isinstance(v, list):
                object.__setattr__(self, name, tuple(v))
            v = getattr(self, name)
            if not isinstance(v, tuple) or any(not isinstance(x, str) for x in v):
                raise LibrarianSchemaError(
                    f"EnumerationDisagreement.{name} must be a tuple of strings"
                )

    @property
    def routing(self) -> str:
        return FAILURE_ROUTING[self.kind]

    def to_dict(self) -> dict:
        return {"kind": self.kind, "paper_id": self.paper_id, "detail": self.detail,
                "only_model_a": list(self.only_model_a),
                "only_model_b": list(self.only_model_b), "routing": self.routing}


class AssemblyIncomplete(RuntimeError):
    """A construction could not be assembled into a valid spec (e.g. its sort
    signal did not resolve to a registry concept). Surfaced, never silently
    dropped -- an unresolved sort signal means "cannot build THIS strategy",
    a review outcome scoped to the one construction, not a blank spec and not
    a whole-paper abort (CI-9, 2026-09-06). Both an exception (raised by the
    injected assembler) and a typed run-record event (``run_paper`` catches it
    per construction and appends it to ``result.events``). Routes to
    ``review``."""

    kind: str = "assembly_incomplete"

    def __init__(self, detail: str, *, paper_id: str | None = None,
                 construction_name: str | None = None) -> None:
        super().__init__(detail)
        self.detail = str(detail)
        self.paper_id = paper_id
        self.construction_name = construction_name

    @property
    def routing(self) -> str:
        return FAILURE_ROUTING[self.kind]

    def to_dict(self) -> dict:
        return {"kind": self.kind, "paper_id": self.paper_id,
                "construction_name": self.construction_name,
                "detail": self.detail, "routing": self.routing}


# The union of the failure event types, for typing + isinstance sweeps.
FailureEvent = (
    UnparseablePdf | PartialParse | LocatorSystematicFailure | EnumerationDisagreement
    | AssemblyIncomplete
)


def route_of(event: object) -> str:
    """The route for a failure event. Raises if ``event`` is not a known failure
    event (a build error -- routing is never guessed)."""
    if not isinstance(
        event, (UnparseablePdf, PartialParse, LocatorSystematicFailure,
                EnumerationDisagreement, AssemblyIncomplete)
    ):
        raise LibrarianSchemaError(
            f"route_of expects a failure event; got {type(event).__name__}"
        )
    return event.routing
