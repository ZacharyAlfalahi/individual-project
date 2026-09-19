"""P2 coverage-boundary census (WS-C) — the runtime census router seam.

``agents/librarian/corpus_fate.py`` encodes the D21 corpus walk as a *frozen
specification table* and its docstring explicitly defers the live artefact to
"[P2] ... a separate artifact". This module IS that artefact: the runtime
census router that decides a fresh (scale-layer) paper's fate from its extracted
Librarian spec and the deterministic compiler routing, and packages the result
as a typed ``CensusResult`` — the input the P2 selector, metrics and taxonomy
consume.

Design (mirrors corpus_fate's typed-data discipline, kept build-only):

  * The router does NOT reimplement the compiler. Routing is *injected* as a
    ``RouteFn`` (extracted spec -> ``RoutingDecision``) so this seam is
    deterministic and fixture-testable with ZERO pipeline/engine imports beyond
    the ``REFUSAL_FATES`` enum it types refusals against.
  * Text-acquisition / text-quality failures are *typed COUNTED exclusions*
    (``text_quality_ok=False`` + a non-empty ``exclusion_reason``), never silent
    drops and never routed — you cannot route a spec you could not extract.
  * A refusal's ``refusal_reason`` MUST be one of the two closed vocabularies
    ``REFUSAL_REASONS`` = adjudicated ``corpus_fate.REFUSAL_FATES`` ∪ the live
    router's ``RefusalCode`` values (both imported for typing, neither mutated);
    an unknown reason is a build error.
  * Fail-loud everywhere: a duplicate paper, an unknown paper id at lookup, an
    off-enum refusal reason, or an internally inconsistent member is raised
    (``P2CensusError``), never silently coerced.

The frozen ``corpus_fate.refusals()`` table is the *fixture-test oracle* the
router is checked against (see ``tests/unit/test_p2_census.py``); it is imported
for typing but NEVER mutated.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

# Imported for typing only — the closed refusal enums. NEITHER is mutated.
from agents.librarian.corpus_fate import REFUSAL_FATES
from agents.quant.config.refusal import RefusalCode

#: A refusal reason is typed against one of two closed vocabularies, because the census can be
#: built two ways and both must stay legible:
#:   * ``REFUSAL_FATES`` — the adjudicated D21 corpus fates (e.g. ``refuse_asset_class``), used
#:     when the partition transcribes the pre-registered implement/refuse labels;
#:   * ``ROUTER_REFUSAL_CODES`` — the LIVE deterministic router's own typed codes (e.g.
#:     ``REVIEW_REQUIRED``), used when the partition is the router's own decision, which is
#:     what the contract's arm rule describes.
#: An unknown reason stays a build error either way.
ROUTER_REFUSAL_CODES: frozenset[str] = frozenset(code.value for code in RefusalCode)
REFUSAL_REASONS: frozenset[str] = REFUSAL_FATES | ROUTER_REFUSAL_CODES

#: Refusals are of two KINDS, and the distinction is load-bearing for this experiment.
#:
#:   * ``coverage`` — the audited engine cannot REPRESENT the construction (an adjudicated
#:     corpus fate, a missing signal binding, a weighting/trim/combiner outside the enum).
#:     This is the coverage boundary the experiment is built to measure.
#:   * ``extraction`` — the SPECIFICATION did not certify (a field needs the manual-review
#:     lane; the paper was silent on a load-bearing field). The construction never reaches
#:     the coverage layer, so a refusal here says nothing about representability.
#:
#: An arm filled with ``extraction`` refusals is NOT the coverage boundary, however wide it
#: is; reporting it as one would answer a different question than the arm rule asks.
EXTRACTION_REFUSAL_REASONS: frozenset[str] = frozenset({
    RefusalCode.REVIEW_REQUIRED.value,
    RefusalCode.REFUSED_ON_SILENCE.value,
})
COVERAGE_REFUSAL_REASONS: frozenset[str] = REFUSAL_REASONS - EXTRACTION_REFUSAL_REASONS


def refusal_kind(reason: str | None) -> str | None:
    """``"coverage"`` / ``"extraction"`` for a typed refusal reason, ``None`` for no refusal.
    Fail-loud on an unknown reason — the two kinds partition the closed vocabulary."""
    if reason is None:
        return None
    if reason in EXTRACTION_REFUSAL_REASONS:
        return "extraction"
    if reason in COVERAGE_REFUSAL_REASONS:
        return "coverage"
    raise P2CensusError(
        f"refusal_reason {reason!r} is not in the closed refusal vocabulary, so its kind "
        "cannot be decided"
    )


#: The closed vocabulary of eligibility-exclusion reasons. An eligibility
#: exclusion (``text_quality_ok=False``) MUST carry one of these — an unvalidated
#: free string is how a silent drop gets reintroduced later (the same discipline
#: as ``REFUSAL_FATES`` for refusals). Two families:
#:   * text acquisition / quality failures (the general eligibility bar), and
#:   * extraction-outcome exclusions (a reportable run produced no usable spec for
#:     the member) — ``extraction_review_exit_no_spec`` types a member whose Phase-F
#:     extraction exited to review with zero specs (the schema forbids partial
#:     emission, so the member has no spec to generate from and is a COUNTED
#:     exclusion, never a silent drop).
ELIGIBILITY_EXCLUSION_REASONS = frozenset({
    "text_acquisition_failed",         # text could not be acquired
    "text_quality_below_bar",          # text acquired but below the quality bar
    "extraction_review_exit_no_spec",  # a reportable run exited to review with zero specs
    "extraction_not_attempted",        # no reportable extraction run exists for the member
    "spec_unmatched_to_member",        # a spec exists but no gold construction matches byte-exactly
})


class P2CensusError(ValueError):
    """A census input or router decision is malformed — a build error surfaced
    loudly (never a silent drop or a coerced fate)."""


# ---------------------------------------------------------------------------
# Router I/O types.
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CensusInput:
    """One paper handed to the router: its stable id, the Librarian-extracted
    spec (``None`` when nothing was extractable), whether text acquisition /
    quality cleared the eligibility bar, and — iff it did NOT — a non-empty
    typed ``exclusion_reason`` so the exclusion is COUNTED, never silent."""

    paper_id: str
    extracted_spec: object | None
    text_quality_ok: bool
    exclusion_reason: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.paper_id, str) or self.paper_id.strip() == "":
            raise P2CensusError("CensusInput.paper_id must be a non-empty string")
        if self.text_quality_ok:
            if self.exclusion_reason is not None:
                raise P2CensusError(
                    f"{self.paper_id!r}: text_quality_ok is True but an exclusion_reason "
                    f"{self.exclusion_reason!r} was given — an eligible paper is not excluded"
                )
        else:
            if not isinstance(self.exclusion_reason, str) or self.exclusion_reason.strip() == "":
                raise P2CensusError(
                    f"{self.paper_id!r}: text_quality_ok is False, so a non-empty typed "
                    "exclusion_reason is required — eligibility exclusions are COUNTED, never silent"
                )
            if self.exclusion_reason not in ELIGIBILITY_EXCLUSION_REASONS:
                raise P2CensusError(
                    f"{self.paper_id!r}: exclusion_reason {self.exclusion_reason!r} is not one of "
                    f"the closed ELIGIBILITY_EXCLUSION_REASONS {sorted(ELIGIBILITY_EXCLUSION_REASONS)}"
                )


@dataclass(frozen=True)
class RoutingDecision:
    """The deterministic compiler routing of one extracted spec: does it compile
    to an audited family, is it a typed refusal, and — iff refused — the refusal
    reason (one of ``REFUSAL_FATES``). ``compilable`` and ``refused`` are
    mutually exclusive."""

    compilable: bool
    refused: bool
    refusal_reason: str | None = None

    def __post_init__(self) -> None:
        if self.compilable and self.refused:
            raise P2CensusError("a routing decision cannot be BOTH compilable and refused")
        if self.refused:
            if self.refusal_reason not in REFUSAL_REASONS:
                raise P2CensusError(
                    f"refusal_reason {self.refusal_reason!r} is not one of the closed "
                    f"refusal vocabularies: adjudicated fates {sorted(REFUSAL_FATES)} or "
                    f"router codes {sorted(ROUTER_REFUSAL_CODES)}"
                )
        elif self.refusal_reason is not None:
            raise P2CensusError(
                f"a non-refused decision must carry refusal_reason=None; got "
                f"{self.refusal_reason!r}"
            )


#: The injected deterministic-compiler routing: extracted spec -> decision.
RouteFn = Callable[[CensusInput], RoutingDecision]


# ---------------------------------------------------------------------------
# Census member + result.
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CensusMember:
    """One paper's decided census disposition. Exactly one of three fates holds:
    an eligibility exclusion (``not text_quality_ok``), a refusal
    (``refused``), or compilable (``compilable``) — enforced in
    ``__post_init__`` so a member can never be internally inconsistent."""

    paper_id: str
    compilable: bool
    refused: bool
    refusal_reason: str | None
    text_quality_ok: bool
    extracted_spec: object | None
    exclusion_reason: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.paper_id, str) or self.paper_id.strip() == "":
            raise P2CensusError("CensusMember.paper_id must be a non-empty string")
        if self.compilable and self.refused:
            raise P2CensusError(
                f"{self.paper_id!r}: a member cannot be BOTH compilable and refused"
            )
        if not self.text_quality_ok:
            # Eligibility exclusion: never routed, always typed + counted.
            if self.compilable or self.refused or self.refusal_reason is not None:
                raise P2CensusError(
                    f"{self.paper_id!r}: an eligibility-excluded member (text_quality_ok=False) "
                    "cannot also be compilable/refused — it was never routed"
                )
            if not isinstance(self.exclusion_reason, str) or self.exclusion_reason.strip() == "":
                raise P2CensusError(
                    f"{self.paper_id!r}: an eligibility exclusion requires a non-empty typed reason"
                )
            if self.exclusion_reason not in ELIGIBILITY_EXCLUSION_REASONS:
                raise P2CensusError(
                    f"{self.paper_id!r}: exclusion_reason {self.exclusion_reason!r} is not one of "
                    f"the closed ELIGIBILITY_EXCLUSION_REASONS {sorted(ELIGIBILITY_EXCLUSION_REASONS)}"
                )
            return
        if self.exclusion_reason is not None:
            raise P2CensusError(
                f"{self.paper_id!r}: an eligible (routed) member carries no exclusion_reason"
            )
        if self.refused:
            if self.refusal_reason not in REFUSAL_REASONS:
                raise P2CensusError(
                    f"{self.paper_id!r}: refusal_reason {self.refusal_reason!r} is not one of "
                    f"the closed refusal vocabularies: adjudicated fates "
                    f"{sorted(REFUSAL_FATES)} or router codes {sorted(ROUTER_REFUSAL_CODES)}"
                )
        elif self.refusal_reason is not None:
            raise P2CensusError(
                f"{self.paper_id!r}: a non-refused member must carry refusal_reason=None"
            )

    @property
    def is_eligibility_exclusion(self) -> bool:
        return not self.text_quality_ok

    @property
    def refusal_kind(self) -> str | None:
        """``"coverage"`` / ``"extraction"`` for a refused member (see the module constants);
        ``None`` for any member that was not refused."""
        return refusal_kind(self.refusal_reason) if self.refused else None

    @property
    def disposition(self) -> str:
        """A single stable label for the census fate table: one of
        ``eligibility_excluded`` / ``refused`` / ``compilable``."""
        if not self.text_quality_ok:
            return "eligibility_excluded"
        return "refused" if self.refused else "compilable"

    def to_dict(self) -> dict:
        return {
            "paper_id": self.paper_id,
            "disposition": self.disposition,
            "compilable": self.compilable,
            "refused": self.refused,
            "refusal_reason": self.refusal_reason,
            "refusal_kind": self.refusal_kind,
            "text_quality_ok": self.text_quality_ok,
            "exclusion_reason": self.exclusion_reason,
        }


@dataclass(frozen=True)
class CensusResult:
    """The typed output of the census router: every member in input order, plus
    the three partitions P2 consumes (refusal set, compilable set, eligibility
    exclusions). Membership is a genuine mapping (one disposition per paper —
    duplicate ids are a build error caught at construction)."""

    members: tuple[CensusMember, ...]

    def __post_init__(self) -> None:
        seen: set[str] = set()
        for m in self.members:
            if m.paper_id in seen:
                raise P2CensusError(
                    f"duplicate paper {m.paper_id!r} in the census — one disposition per paper"
                )
            seen.add(m.paper_id)

    def member(self, paper_id: str) -> CensusMember:
        """The census disposition of ``paper_id``. Fail-loud on an unknown paper
        — the census is a closed record of the papers it walked, not an open
        router; asking about an unlisted paper is a caller-contract error."""
        for m in self.members:
            if m.paper_id == paper_id:
                return m
        raise P2CensusError(
            f"{paper_id!r} is not a paper in this census; "
            f"known papers: {sorted(mm.paper_id for mm in self.members)}"
        )

    def refusal_set(self) -> tuple[CensusMember, ...]:
        """Every refused member, in input order — Arm A (no selection discretion)."""
        return tuple(m for m in self.members if m.refused)

    def compilable_set(self) -> tuple[CensusMember, ...]:
        """Every compilable member, in input order — the Arm-B candidate pool."""
        return tuple(m for m in self.members if m.compilable)

    def refusals_by_kind(self) -> dict[str, tuple[CensusMember, ...]]:
        """The refusal set split into ``coverage`` and ``extraction`` refusals. An Arm A of a
        given size means different things depending on this split, so every artefact that
        quotes an arm size should quote this beside it."""
        refused = self.refusal_set()
        return {
            "coverage": tuple(m for m in refused if m.refusal_kind == "coverage"),
            "extraction": tuple(m for m in refused if m.refusal_kind == "extraction"),
        }

    def eligibility_exclusions(self) -> tuple[CensusMember, ...]:
        """Every eligibility-excluded member (text acquisition/quality failure),
        in input order — typed COUNTED exclusions, never silent drops."""
        return tuple(m for m in self.members if m.is_eligibility_exclusion)

    def fate_table(self) -> tuple[dict, ...]:
        """The census fate table (one row per member) — always emittable, even
        when agreement/divergence numbers are suppressed below floor."""
        return tuple(m.to_dict() for m in self.members)


# ---------------------------------------------------------------------------
# The runtime census router.
# ---------------------------------------------------------------------------

def run_census(inputs: Sequence[CensusInput], route: RouteFn) -> CensusResult:
    """Route every input paper into its census disposition.

    An eligibility-excluded input (``text_quality_ok=False``) is recorded as a
    typed COUNTED exclusion and is NEVER passed to ``route`` (you cannot route a
    spec you could not extract). Every other input is routed through the injected
    deterministic-compiler ``route``; its ``RoutingDecision`` becomes the
    member's compilable/refused disposition. Duplicate paper ids are a build
    error (caught by ``CensusResult``)."""
    members: list[CensusMember] = []
    for inp in inputs:
        if not inp.text_quality_ok:
            members.append(
                CensusMember(
                    paper_id=inp.paper_id,
                    compilable=False,
                    refused=False,
                    refusal_reason=None,
                    text_quality_ok=False,
                    extracted_spec=inp.extracted_spec,
                    exclusion_reason=inp.exclusion_reason,
                )
            )
            continue
        decision = route(inp)
        if not isinstance(decision, RoutingDecision):
            raise P2CensusError(
                f"{inp.paper_id!r}: route() must return a RoutingDecision, got "
                f"{type(decision).__name__}"
            )
        members.append(
            CensusMember(
                paper_id=inp.paper_id,
                compilable=decision.compilable,
                refused=decision.refused,
                refusal_reason=decision.refusal_reason,
                text_quality_ok=True,
                extracted_spec=inp.extracted_spec,
                exclusion_reason=None,
            )
        )
    return CensusResult(tuple(members))
