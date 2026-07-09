"""
Signal-ref filler -- dual-model extraction of a ``SignalRef`` (D22, D9/D11).

A ``SignalRef`` names the paper's sort signal (or the control axis of a double
sort) as a *menu pick* from the Signal Concept Registry (D22), or takes the
first-class ``unrecognised`` escape. This module assembles one from two models'
answers, reusing the SAME D9 merge the ordinary form-filler uses -- there is one
merge and one set of reason codes in the Librarian, not two.

The extraction has three parts, each merged independently through the shared
merge:

  1. **concept_id** -- both models name a registry id (or ``"unrecognised"``);
     ``fill_field`` (kind ``"signal_ref"``) runs the D9 merge on the normalised
     concept-id token -> an ``Inherited[str]``. STATED with a located quote on
     agreement; UNKNOWN with the right reason otherwise (``not_stated`` /
     ``quote_match_failure`` / ``single_response`` / ``disagreement``).
  2. **as_described** -- the paper's own words behind the concept: the model's
     label + its located quotes, assembled ONLY when concept_id is STATED. This
     is audit-only (D22) for a known concept, and load-bearing for the escape
     invariant when the concept is ``unrecognised`` (a SignalRef whose
     concept_id == ``"unrecognised"`` REQUIRES a non-empty ``as_described``).
  3. **parameters** -- for a known STATED concept, each name in the concept's
     registry ``parameter_schema`` is filled through the SAME merge (its own
     ``FieldQuery``); a paper-silent param -> UNKNOWN(``not_stated``). v1 registry
     concepts have EMPTY parameter schemas, so the loop is usually
     empty -- but it is implemented generally.

When concept_id is UNKNOWN (any reason), the ``SignalRef`` is ``None``: a
non-agreement never fabricates a signal. The caller routes to review/omit per
policy. The concept_id trace (and any per-param traces) always travel back so
RQ1 keeps the full dual-model record.

The assembled ``SignalRef`` passes ``validate_librarian_spec``'s registry-aware
checks against the same registry (a known concept validates; ``unrecognised``
with ``as_described`` validates).
"""

from __future__ import annotations

from dataclasses import dataclass

from ..schema.signal_ref import UNRECOGNISED, DescribedSignal, LocatedQuote, SignalRef
from ..validators.spec_validators import SignalRegistryLike
from .form_filler import fill_field
from .model_client import FieldQuery, ModelAnswer, ModelClient
from .trace import FieldTraceRecord

# The most quotes carried in as_described (audit-only; keep the payload small).
MAX_DESCRIBED_QUOTES: int = 3


@dataclass(frozen=True)
class SignalRefOutcome:
    """The merged result of filling one ``SignalRef`` field.

      * ``signal_ref``  -- the assembled ``SignalRef`` on a STATED concept_id, else
                           ``None`` (a non-agreement never fabricates a signal).
      * ``concept_id``  -- the concept-id ``Inherited`` (STATED on agreement, else
                           UNKNOWN carrying the D9/D11 reason).
      * ``trace``       -- the concept-id decision's ``FieldTraceRecord`` (both
                           models to the trace, D12).
      * ``param_traces``-- one ``FieldTraceRecord`` per registry parameter filled
                           (empty for the v1 concepts, whose schemas are empty).
    """

    field: str
    signal_ref: SignalRef | None
    concept_id: object  # Inherited[str]
    trace: FieldTraceRecord
    param_traces: tuple[FieldTraceRecord, ...] = ()

    @property
    def tag(self) -> str:
        return self.concept_id.tag


def _label_of(ans_a: ModelAnswer, ans_b: ModelAnswer, ship_choice: str | None) -> str:
    """The as_described label: the raw concept token from the model whose quote
    shipped (falls back to whichever model answered). Audit-only (D22)."""
    chosen = ans_a if ship_choice != "model_b" else ans_b
    if chosen.answered and chosen.raw is not None:
        return str(chosen.raw)
    other = ans_b if chosen is ans_a else ans_a
    return str(other.raw) if other.answered and other.raw is not None else UNRECOGNISED


def _located_quotes(
    answers: tuple[ModelAnswer, ...], canonical_text, cap: int
) -> tuple[LocatedQuote, ...]:
    """The distinct located quotes across the answers, earliest span first, capped.

    Each model's justifying span is located against the canonical text (reusing
    ``.locate``, the same L0 gate the merge uses); non-locating spans are dropped
    (they cannot anchor an audit quote). Deduplicated on text so two models citing
    the same phrase contribute one ``LocatedQuote``."""
    located: list[tuple[LocatedQuote, tuple[int, int]]] = []
    seen: set[str] = set()
    for ans in answers:
        if not ans.answered:
            continue
        for span in _spans_of(ans):
            if span in seen:
                continue
            loc = canonical_text.locate(span)
            if loc is None:
                continue
            seen.add(span)
            located.append(
                (
                    LocatedQuote(
                        text=span,
                        page=loc.page,
                        char_start=loc.char_start,
                        char_end=loc.char_end,
                    ),
                    (loc.page, loc.char_start),
                )
            )
    located.sort(key=lambda pair: pair[1])
    return tuple(lq for lq, _key in located)[:cap]


def _spans_of(ans: ModelAnswer) -> tuple[str, ...]:
    """The verbatim spans a model claims for its concept: its ``quote`` plus any
    ``quotes`` (order preserved, quote first)."""
    spans: list[str] = []
    if ans.quote is not None:
        spans.append(ans.quote)
    spans.extend(ans.quotes)
    return tuple(spans)


def fill_signal_ref(
    field: str,
    model_a: ModelClient,
    model_b: ModelClient,
    canonical_text,
    registry: SignalRegistryLike,
    param_queries: dict[str, FieldQuery] | None = None,
) -> SignalRefOutcome:
    """Fill one ``SignalRef`` field via dual-model extraction + the D9 merge.

    ``field`` is the SignalRef marker's field name (e.g. ``"sort_signal"``);
    ``registry`` is the Signal Concept Registry (its ``has_concept`` /
    ``parameter_schema`` drive which parameters to fill and whether the concept is
    known). ``param_queries`` optionally supplies a per-parameter ``FieldQuery``
    (its own template/schema hashes); a param with no supplied query gets a
    default ``enum`` query keyed by the parameter name.

    Returns a ``SignalRefOutcome``: the assembled ``SignalRef`` (or ``None`` on a
    non-agreement) + the concept_id ``Inherited`` + the trace record(s)."""
    # (1) concept_id: the shared D9 merge on the signal-ref concept token.
    query = FieldQuery(field=field, kind="signal_ref")
    cid_outcome = fill_field(field, model_a, model_b, canonical_text, query=query)
    concept_id = cid_outcome.value
    trace = cid_outcome.trace

    # (5) UNKNOWN concept_id (not_stated / quote_match_failure / single_response /
    # disagreement) -> no SignalRef; the caller routes to review/omit per policy.
    if concept_id.tag != "STATED":
        return SignalRefOutcome(
            field=field, signal_ref=None, concept_id=concept_id, trace=trace
        )

    # Re-ask both models (the FakeModelClient is deterministic; a real client is
    # cached upstream) to assemble as_described from their justifying spans.
    ans_a = model_a.answer(query, canonical_text)
    ans_b = model_b.answer(query, canonical_text)

    cid_value = concept_id.value

    # (4) as_described: the paper's own words. For a known concept this is
    # audit-only; for the unrecognised escape it is load-bearing (the escape
    # invariant needs a non-empty quotes tuple).
    quotes = _located_quotes((ans_a, ans_b), canonical_text, MAX_DESCRIBED_QUOTES)
    label = _label_of(ans_a, ans_b, trace.ship_choice)
    as_described = DescribedSignal(label=label, quotes=quotes)

    # (3) parameters: only for a KNOWN registry concept (unrecognised has no
    # schema). Each param merges through the SAME merge; paper-silent -> UNKNOWN.
    parameters: dict[str, object] = {}
    param_traces: list[FieldTraceRecord] = []
    if cid_value != UNRECOGNISED and registry.has_concept(cid_value):
        schema = registry.parameter_schema(cid_value)
        param_queries = param_queries or {}
        for pname, ptype in schema.items():
            pquery = param_queries.get(pname) or FieldQuery(field=pname, kind="enum")
            # int-typed params fold "6"/"6 months"/6 like the schema int fields do;
            # the param name is not a schema field, so pass the type explicitly.
            value_kind = "int" if ptype is int else None
            p_outcome = fill_field(
                pname, model_a, model_b, canonical_text, query=pquery, value_kind=value_kind
            )
            parameters[pname] = p_outcome.value
            param_traces.append(p_outcome.trace)

    signal_ref = SignalRef(
        concept_id=concept_id,
        as_described=as_described,
        parameters=parameters,
    )
    return SignalRefOutcome(
        field=field,
        signal_ref=signal_ref,
        concept_id=concept_id,
        trace=trace,
        param_traces=tuple(param_traces),
    )
