"""
Form-filler -- per-field dual-model extraction (build brief §5.3, D9/D11/D16).

For one field the form-filler asks both models (via the injected ``ModelClient``s),
NORMALISES each raw answer to its typed value (D9: agreement is judged on
normalised typed values, not surface form), locates each model's quote in the
canonical text (the D9 quote gate), and merges to a single provenance-tagged
``Inherited`` plus a full ``FieldTraceRecord`` (both models to the trace, D12).

The D9 merge, in order:

  1. Neither model answered (both content-silent)   -> UNKNOWN(``not_stated``)
  2. A quote that fails to locate                   -> UNKNOWN(``quote_match_failure``)
  3. Exactly one model answered (the other silent)  -> UNKNOWN(``single_response``)
                                                       (D11 amendment 2026-07-09; review)
  4. Both answered, normalised values disagree      -> UNKNOWN(``disagreement``)
  5. Agree + both quotes locate                     -> STATED, shipping the
                                                       EARLIEST-span quote (D9), both
                                                       quotes recorded in the trace.

Every STATED ``Inherited`` gets a ``Locator`` (from ``canonical_text.locate``) so
it satisfies the frozen provenance guard (D7). Scope = Part 1 + the fields already
final in the committed inventory (``ALREADY_FINAL_PART2``) -- the S3-adjudicated
fields' prompts land post-S3. The k=3 ladder is a NAMED but unused parameter
(default k=1, D10).

``fill_method_summary`` is the D16 3-slot case: free text, per-strategy, 1-3
locating quotes, both models' summaries to the trace, earliest-span ship rule.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass

from agents.quant.config import Evidence, Inherited, Locator

from ..errors import LibrarianSchemaError
from ..schema import fields as F
from ..schema.strategy_spec import MethodSummary
from ..schema.signal_ref import LocatedQuote
from ..validators.domains import Domain, load_domains
from .model_client import FieldQuery, ModelAnswer, ModelClient
from .trace import FieldTraceRecord, ModelTrace

# The UNKNOWN reason codes (D11) plus the STATED reason (D24 tag-reason row).
NOT_STATED: str = "not_stated"
DISAGREEMENT: str = "disagreement"
QUOTE_MATCH_FAILURE: str = "quote_match_failure"
SINGLE_RESPONSE: str = "single_response"  # one model answered, the other content-silent (D11 amendment 2026-07-09)
STATED_REASON: str = "quoted"

# Fields the form-filler is scoped to fill (build brief §5.3 scope note): Part 1
# menu fields + every field already final in the committed Part 2 inventory. The
# free-text method_summary and the SignalRef markers are handled specially.
_ENUM_MENU_BY_FIELD: dict[str, tuple[str, ...]] = {
    F.FORMATION_STRUCTURE: F.FORMATION_STRUCTURE_MENU,
    F.ASSET_CLASS: F.ASSET_CLASS_MENU,
}


def _field_kind(field: str) -> str:
    """The prompt-template family for a field (build brief §5.3 / manifest).

    The manifest is authoritative -- this is the fallback for a query built
    without one. The paper_facts branches matter: the ``enum`` default would send
    them to ``_menu_for``, which fails loud (correctly -- they have no menu), so
    an unbound paper_facts field would abort a live run rather than degrade."""
    if field in (F.FORMATION_STRUCTURE, F.ASSET_CLASS):
        return "part1_enum"
    if field in F.INT_FIELDS:
        return "int"
    if field in (F.SAMPLE_START, F.SAMPLE_END):
        return "date"
    if field == F.CLAIMED_HEADLINE_METRIC:
        return "paper_metric"
    return "enum"


@dataclass(frozen=True)
class FieldOutcome:
    """The merged result of filling one field: the provenance-tagged ``value``
    (an ``Inherited``) plus the full dual-model ``trace`` record (D12). The
    published spec consumes ``value``; the sidecar consumes ``trace``."""

    field: str
    value: Inherited
    trace: FieldTraceRecord

    def __post_init__(self) -> None:
        if not isinstance(self.field, str) or self.field.strip() == "":
            raise LibrarianSchemaError("FieldOutcome.field must be a non-empty string")
        if not isinstance(self.value, Inherited):
            raise LibrarianSchemaError("FieldOutcome.value must be an Inherited")
        if not isinstance(self.trace, FieldTraceRecord):
            raise LibrarianSchemaError("FieldOutcome.trace must be a FieldTraceRecord")

    @property
    def tag(self) -> str:
        return self.value.tag


# ---------------------------------------------------------------------------
# Normalisation (D9): raw model answer -> typed value agreement is judged on.
# ---------------------------------------------------------------------------

def normalise(field: str, raw: object, value_kind: str | None = None) -> object:
    """Normalise a raw model answer to the typed value D9 compares on.

    Driven by the field's kind (fields.py / domains):

      * int fields (``INT_FIELDS``) -> a Python ``int``. Accepts an int, a float
        with no fractional part, or a leading-integer string (``"6"``,
        ``"6 months"``, ``"6-month"``) -- so ``"six months"`` surface forms that
        already decoded to ``6`` and a bare ``6`` agree. A value that is not an
        integer (``"quarterly"``, ``6.5``) -> ``LibrarianSchemaError`` (a decoding
        contract violation, surfaced not silently coerced).
      * enum / signal_ref / other -> a canonical string: trimmed, lower-cased,
        internal whitespace and hyphens collapsed to single underscores. Menu
        tokens are already canonical, so this is identity for well-decoded
        answers and folds surface variants (``"Equal Weighted"`` ->
        ``"equal_weighted"``) together.

    ``value_kind`` optionally overrides the int-vs-token decision (``"int"`` ->
    int, ``"paper_metric"`` -> canonical metric tuple, anything else -> token)
    for a field whose name is NOT in ``INT_FIELDS`` -- used by the signal filler
    for a registry parameter typed ``int`` in its concept's schema (the param
    name is not a schema field, so ``INT_FIELDS`` membership cannot classify it),
    and by the paper_facts filler for the composite headline metric. ``None``
    (the default) keeps the field-name-driven behaviour unchanged.

    ``None`` normalises to ``None`` (a silent / absent value)."""
    if raw is None:
        return None
    # Dispatch on the FIELD NAME as well as value_kind, matching the int rule. The
    # name alone is sufficient to classify these, and a caller that forgets
    # value_kind would otherwise silently fall through to _normalise_token, which
    # destroys both shapes (a date loses its hyphen, a metric folds to its repr).
    if value_kind == "paper_metric" or field == F.CLAIMED_HEADLINE_METRIC:
        return _normalise_paper_metric(field, raw)
    if value_kind == "date" or field in (F.SAMPLE_START, F.SAMPLE_END):
        return _normalise_date(field, raw)
    if value_kind == "int" or field in F.INT_FIELDS:
        return _normalise_int(field, raw)
    if value_kind == "int_set":
        # Scope B (2026-09-04): the K-sweep set -> a sorted de-duplicated tuple of
        # ints (order-invariant equality for the D9 merge; JSON serialises it as a
        # list, and the scorer compares as a set). Falling through to the token
        # branch would stringify the collection -- a decoding-contract violation.
        if isinstance(raw, (list, tuple)) and raw and all(
                isinstance(x, int) and not isinstance(x, bool) for x in raw):
            return tuple(sorted(set(raw)))
        raise LibrarianSchemaError(
            f"{field}: int_set requires a non-empty list of ints; got {raw!r}")
    return _normalise_token(raw)


# paper_facts dates are YYYY-MM, per schemas/date.schema.json.
_DATE_RE = re.compile(r"^[0-9]{4}-(0[1-9]|1[0-2])$")


def _normalise_int(field: str, raw: object) -> int:
    if isinstance(raw, bool):
        raise LibrarianSchemaError(f"{field}: a bool is not a valid int value; got {raw!r}")
    if isinstance(raw, int):
        return raw
    if isinstance(raw, float):
        if raw.is_integer():
            return int(raw)
        raise LibrarianSchemaError(f"{field}: non-integer float {raw!r} is not a valid int value")
    if isinstance(raw, str):
        token = raw.strip()
        # Leading integer: "6", "6 months", "6-month". Reject if no int prefix.
        digits = ""
        for ch in token:
            if ch.isdigit() or (ch == "-" and digits == ""):
                digits += ch
            else:
                break
        if digits not in ("", "-"):
            return int(digits)
        raise LibrarianSchemaError(f"{field}: string {raw!r} has no leading integer to normalise")
    raise LibrarianSchemaError(f"{field}: cannot normalise {type(raw).__name__} to int")


def _normalise_date(field: str, raw: object) -> str:
    """A paper_facts date, normalised to itself.

    YYYY-MM is ALREADY the canonical form, so normalisation here is validation
    plus pass-through -- deliberately NOT ``_normalise_token``, which collapses
    the hyphen to an underscore and would ship ``2004_07`` for a value the prompt
    and schema both demand as ``2004-07``. The shipped value is the normalised
    one (see the STATED branch of ``fill_field``), so a destructive normaliser
    here corrupts the artefact rather than just the comparison key: the field
    would be read correctly and still score as a mismatch against gold.

    The real client already degrades a non-conforming date to silent, so a bad
    one reaching here is a decoding-contract violation, surfaced not coerced."""
    if not isinstance(raw, str):
        raise LibrarianSchemaError(
            f"{field}: date value must be a string; got {type(raw).__name__}"
        )
    token = raw.strip()
    if not _DATE_RE.match(token):
        raise LibrarianSchemaError(
            f"{field}: {raw!r} is not a YYYY-MM date (zero-padded month)"
        )
    return token


def _normalise_paper_metric(field: str, raw: object) -> dict:
    """The composite claimed_headline_metric ({mean, t_stat, unit}) with its
    components coerced: mean / t_stat to float (so 0.7 and "0.70" agree), unit to
    a token.

    Returns a DICT, not a tuple. Python dict equality is already order-independent
    (``{'a':1,'b':2} == {'b':2,'a':1}``), so D9 gets key-order invariance for free
    and there is no reason to fold to a sorted tuple -- doing so would change the
    shipped TYPE, since the STATED branch of ``fill_field`` ships the normalised
    value. Gold holds a mapping (``strategy_spec.PaperFacts`` declares
    ``Inherited[{mean, t_stat, unit}]``), so a tuple could never match it and
    would break every downstream subscript.

    A value that is not a complete triple -> ``LibrarianSchemaError``: the client
    already degrades a partial metric to silent, so a partial one reaching here
    is a decoding-contract violation, surfaced not silently coerced."""
    if not isinstance(raw, Mapping):
        raise LibrarianSchemaError(
            f"{field}: paper_metric value must be a mapping; got {type(raw).__name__}"
        )
    missing = [k for k in ("mean", "t_stat", "unit") if k not in raw]
    if missing:
        raise LibrarianSchemaError(
            f"{field}: paper_metric value is missing {missing} -- a mean without its "
            "t-statistic is not a headline claim"
        )
    out: dict = {}
    for key in sorted(raw):
        value = raw[key]
        if key == "unit":
            out[key] = _normalise_token(value)
            continue
        if isinstance(value, bool) or not isinstance(value, (int, float, str)):
            raise LibrarianSchemaError(
                f"{field}.{key}: cannot normalise {type(value).__name__} to a number"
            )
        try:
            out[key] = float(value)
        except ValueError as exc:
            raise LibrarianSchemaError(
                f"{field}.{key}: string {value!r} is not a number"
            ) from exc
    return out


def _normalise_token(raw: object) -> str:
    if not isinstance(raw, str):
        # A non-string typed value (e.g. a bool from a mis-decoded enum) -- fold
        # to its lower-cased repr so two identical mis-decodes still agree.
        return str(raw).strip().lower()
    token = raw.strip().lower()
    out_chars = []
    prev_us = False
    for ch in token:
        if ch.isspace() or ch == "-":
            if not prev_us:
                out_chars.append("_")
                prev_us = True
        else:
            out_chars.append(ch)
            prev_us = False
    return "".join(out_chars).strip("_")


# ---------------------------------------------------------------------------
# The D9 merge.
# ---------------------------------------------------------------------------

def _unknown(field: str, reason: str, note: str) -> Inherited:
    return Inherited(None, "UNKNOWN", Evidence(note=note, unknown_reason=reason))


def _model_trace(ans: ModelAnswer, locator: Locator | None, normalised: object) -> ModelTrace:
    return ModelTrace(
        answered=ans.answered,
        raw=ans.raw,
        quote=ans.quote,
        locate_result=locator,
        model_id=ans.model_id,
        parse_failed=getattr(ans, "parse_failed", False),  # B2: format-failure trace signal
    )


def _primary_quote(ans: ModelAnswer) -> str | None:
    """The single span a model claims for this field: its ``quote`` if set, else
    the first of its ``quotes`` (method_summary uses ``quotes``)."""
    if ans.quote is not None:
        return ans.quote
    return ans.quotes[0] if ans.quotes else None


def fill_field(
    field: str,
    model_a: ModelClient,
    model_b: ModelClient,
    canonical_text,
    query: FieldQuery | None = None,
    k: int = 1,
    value_kind: str | None = None,
) -> FieldOutcome:
    """Fill one field via dual-model extraction + the D9 merge. Operates on ANY
    ``CanonicalText`` (stub or frozen) through ``.locate()``; the frozen gate is
    the orchestrator's concern.

    ``k`` is the D10 self-consistency count -- NAMED but unused (v1 = k=1
    primary; the k=3 / 2-of-3 ladder is pre-registered, not built). ``k != 1``
    raises so the seam is honest rather than silently ignored.

    ``value_kind`` optionally forces int vs token normalisation (see
    ``normalise``) for a field the ``INT_FIELDS`` set cannot classify -- the
    signal filler passes ``"int"`` for an int-typed registry parameter."""
    if k != 1:
        raise LibrarianSchemaError(
            f"fill_field k={k}: the k=3 self-consistency ladder (D10) is a named-but-unbuilt "
            "seam; v1 runs k=1 primary only"
        )
    if query is None:
        query = FieldQuery(field=field, kind=_field_kind(field))

    ans_a = model_a.answer(query, canonical_text)
    ans_b = model_b.answer(query, canonical_text)

    q_a = _primary_quote(ans_a)
    q_b = _primary_quote(ans_b)
    loc_a = canonical_text.locate(q_a) if (ans_a.answered and q_a) else None
    loc_b = canonical_text.locate(q_b) if (ans_b.answered and q_b) else None

    norm_a = normalise(field, ans_a.raw, value_kind) if ans_a.answered else None
    norm_b = normalise(field, ans_b.raw, value_kind) if ans_b.answered else None

    # (1) Neither model answered -> the paper is silent.
    if not ans_a.answered and not ans_b.answered:
        value = _unknown(
            field, NOT_STATED, f"{field}: neither model found a stated value (paper silent)"
        )
        trace = _make_trace(field, ans_a, ans_b, loc_a, loc_b, norm_a, norm_b, value, None)
        return FieldOutcome(field=field, value=value, trace=trace)

    # A model that answered but whose quote failed to locate -> quote gate fails.
    a_gate_fail = ans_a.answered and loc_a is None
    b_gate_fail = ans_b.answered and loc_b is None
    if a_gate_fail or b_gate_fail:
        value = _unknown(
            field,
            QUOTE_MATCH_FAILURE,
            f"{field}: a model's quote did not locate in the canonical text (quote gate failed)",
        )
        trace = _make_trace(field, ans_a, ans_b, loc_a, loc_b, norm_a, norm_b, value, None)
        return FieldOutcome(field=field, value=value, trace=trace)

    # Only one model answered; the other content-reported silence (answered=False,
    # the explicit not-present selection). D9 requires BOTH models to agree to ship
    # STATED, so a lone answer is not enough -> review. This is single_response, a
    # distinct reason from disagreement (both answered, values conflict) and from
    # not_stated (both silent) -- D11 amendment 2026-07-09; keeps the reason-code
    # column honest for RQ1.
    if ans_a.answered != ans_b.answered:
        value = _unknown(
            field,
            SINGLE_RESPONSE,
            f"{field}: only one model answered, the other content-reported silence "
            "(value-vs-silence) -> review (single_response)",
        )
        trace = _make_trace(field, ans_a, ans_b, loc_a, loc_b, norm_a, norm_b, value, None)
        return FieldOutcome(field=field, value=value, trace=trace)

    # Both answered + both quotes located: compare NORMALISED typed values.
    if norm_a != norm_b:
        value = _unknown(
            field,
            DISAGREEMENT,
            f"{field}: models disagree on the normalised value ({norm_a!r} vs {norm_b!r}) -> review",
        )
        trace = _make_trace(field, ans_a, ans_b, loc_a, loc_b, norm_a, norm_b, value, None)
        return FieldOutcome(field=field, value=value, trace=trace)

    # Agreement + both located -> STATED. Ship the EARLIEST-span quote (D9).
    ship = _earliest(loc_a, q_a, loc_b, q_b)
    value = Inherited(
        norm_a,
        "STATED",
        Evidence(quote=ship.quote, locator=ship.locator),
    )
    trace = _make_trace(field, ans_a, ans_b, loc_a, loc_b, norm_a, norm_b, value, ship.choice)
    return FieldOutcome(field=field, value=value, trace=trace)


@dataclass(frozen=True)
class _Ship:
    quote: str
    locator: Locator
    choice: str


def _earliest(loc_a: Locator, q_a: str, loc_b: Locator, q_b: str) -> _Ship:
    """The earliest-span ship rule (D9): the quote with the earlier (page,
    char_start) wins; ties break to model_a (deterministic)."""
    a_key = (loc_a.page, loc_a.char_start)
    b_key = (loc_b.page, loc_b.char_start)
    if b_key < a_key:
        return _Ship(quote=q_b, locator=loc_b, choice="model_b")
    return _Ship(quote=q_a, locator=loc_a, choice="model_a")


def _make_trace(
    field: str,
    ans_a: ModelAnswer,
    ans_b: ModelAnswer,
    loc_a: Locator | None,
    loc_b: Locator | None,
    norm_a: object,
    norm_b: object,
    value: Inherited,
    ship_choice: str | None,
) -> FieldTraceRecord:
    reason = value.evidence.unknown_reason if value.tag == "UNKNOWN" else STATED_REASON
    return FieldTraceRecord(
        field=field,
        model_a=_model_trace(ans_a, loc_a, norm_a),
        model_b=_model_trace(ans_b, loc_b, norm_b),
        normalised_a=norm_a,
        normalised_b=norm_b,
        agreement=(value.tag == "STATED"),
        final_tag=value.tag,
        final_reason=reason,
        ship_choice=ship_choice,
    )


# ---------------------------------------------------------------------------
# method_summary (D16): 3-slot free text, 1-3 locating quotes.
# ---------------------------------------------------------------------------

def fill_method_summary(
    model_a: ModelClient,
    model_b: ModelClient,
    canonical_text,
    query: FieldQuery | None = None,
    field: str = F.METHOD_SUMMARY,
) -> tuple[MethodSummary | None, FieldTraceRecord]:
    """Fill the D16 method_summary: both models write a 3-slot summary with 1-3
    supporting quotes; every shipped quote must locate (the quote gate). The
    pre-committed ship rule picks the summary whose FIRST located quote has the
    earliest span; both summaries go to the trace.

    Returns ``(MethodSummary | None, FieldTraceRecord)``. ``None`` (a
    ``FieldTraceRecord`` UNKNOWN row) is returned when neither model answered or
    a chosen summary carries no locating quote -- the caller then omits the field
    or routes to review per policy. On success the shipped ``MethodSummary``
    carries up to 3 ``LocatedQuote``s (D16 cap)."""
    # ``field`` defaults to method_summary; the prose kind (rubric freeze,
    # 2026-09-04) reuses this whole no-equality-gate mechanism for a named field.
    if query is None:
        query = FieldQuery(field=field, kind="method_summary")
    ans_a = model_a.answer(query, canonical_text)
    ans_b = model_b.answer(query, canonical_text)

    located_a = _locate_summary_quotes(ans_a, canonical_text)
    located_b = _locate_summary_quotes(ans_b, canonical_text)

    q_a = located_a[0][1] if located_a else None
    q_b = located_b[0][1] if located_b else None

    # Neither model produced a locating-quote summary -> not stated.
    if not (ans_a.answered and located_a) and not (ans_b.answered and located_b):
        value = _unknown(field, NOT_STATED, f"{field}: no model produced a located 3-slot summary")
        trace = _make_trace(field, ans_a, ans_b, q_a, q_b, ans_a.raw, ans_b.raw, value, None)
        return None, trace

    # Ship the summary whose first located quote has the earliest span; ties ->
    # model_a. A model that answered but located nothing is not eligible.
    choice = _pick_summary(q_a, q_b)
    chosen_ans = ans_a if choice == "model_a" else ans_b
    chosen_located = located_a if choice == "model_a" else located_b

    quotes = tuple(lq for lq, _loc in chosen_located)[:3]  # D16: at most 3
    summary = MethodSummary(
        summary=Inherited(
            chosen_ans.raw,
            "STATED",
            Evidence(quote=quotes[0].text, locator=chosen_located[0][1]),
        ),
        quotes=quotes,
    )
    trace = _make_trace(
        field, ans_a, ans_b, q_a, q_b, ans_a.raw, ans_b.raw, summary.summary, choice
    )
    return summary, trace


def _locate_summary_quotes(ans: ModelAnswer, canonical_text) -> list[tuple[LocatedQuote, Locator]]:
    """Locate each of a summary answer's quotes; drop any that fail to locate.
    Returns ``[(LocatedQuote, Locator), ...]`` sorted by span (earliest first)."""
    if not ans.answered:
        return []
    out: list[tuple[LocatedQuote, Locator]] = []
    for q in ans.quotes:
        loc = canonical_text.locate(q)
        if loc is not None:
            out.append(
                (
                    LocatedQuote(text=q, page=loc.page, char_start=loc.char_start,
                                 char_end=loc.char_end),
                    loc,
                )
            )
    out.sort(key=lambda pair: (pair[1].page, pair[1].char_start))
    return out


def _pick_summary(loc_a: Locator | None, loc_b: Locator | None) -> str:
    if loc_a is None:
        return "model_b"
    if loc_b is None:
        return "model_a"
    if (loc_b.page, loc_b.char_start) < (loc_a.page, loc_a.char_start):
        return "model_b"
    return "model_a"


# Re-exported for callers that want the enum menu / domain of a field (thin
# convenience over the loaded domains; not required by the merge).
def field_domain(field: str, domains: dict | None = None) -> Domain | None:
    doms = domains if domains is not None else load_domains()
    return doms.get("part2", {}).get(field)
