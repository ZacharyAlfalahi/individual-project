"""
Gold-spec loader (G2 Piece B): parse a hand-authored anchor gold Markdown file
into a full, validating ``StrategySpec``.

The three anchor golds (``gold_str_drr_2026.md``, ``gold_drf_bbw_2019.md``,
``gold_mom6_jnps_2013.md``) are human-authored field-by-field transcriptions of
the paper's construction, in *paper language* (D3), each field carrying a tag
(STATED / UNKNOWN) plus a verbatim quote + page. This module turns one such file
into the frozen ``StrategySpec`` the rest of the pipeline consumes, wiring every
STATED field to a REAL D6/D7 locator resolved out of
``locator_backfill_report.md`` (page + char span into the frozen canonical text).

Design (mirrors ``tests/unit/_librarian_fixtures.py`` constructors):

  * Every schema field is driven off the schema's own field-name tuples
    (``_COMMON_INHERITED_FIELDS`` / ``_LEG_INHERITED_FIELDS`` /
    ``_PAPER_FACTS_INHERITED_FIELDS``), so a schema change auto-widens coverage:
    a schema field absent from the gold is a HARD ERROR (named), never a silent
    skip.
  * Each fact-bearing field -> ``Inherited(value, tag, Evidence(...))``:
      STATED  -> value + Evidence(quote, locator=Locator(page, cs, ce))
      UNKNOWN -> Inherited(None, "UNKNOWN", Evidence(note, unknown_reason=reason))
  * ``sort_signal`` / ``control_axis`` -> ``SignalRef`` (concept_id STATED off the
    ``as_described`` quote; ``control_axis: none`` -> ``None``).
  * Every STATED locator is resolved by matching ``(spec, page, quote[:70])``
    against the backfill table. No row -> raise; multiple DISTINCT offsets ->
    raise (ambiguous). Duplicate rows (same prefix + same offsets) are fine.

FAIL LOUD everywhere: an unclassifiable field, a STATED field with no resolvable
locator, an ambiguous locator -> a clear ``GoldParseError``. The loader NEVER
emits a wrong STATED value or a fabricated locator.

``is_binding(anchor_id)`` reports whether the anchor's locators index a FROZEN
canonical text (str, drf -> True) or a pending-freeze parse (mom6 -> False).
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from agents.quant.config import Evidence, Inherited, Locator

from agents.librarian.schema import (
    Combiner,
    DescribedSignal,
    Leg,
    LocatedQuote,
    MethodSummary,
    PaperFacts,
    Part1,
    Part2,
    SignalRef,
    SpecHeader,
    StrategySpec,
)
from agents.librarian.schema.strategy_spec import (
    _COMMON_INHERITED_FIELDS,
    _LEG_INHERITED_FIELDS,
    _PAPER_FACTS_INHERITED_FIELDS,
)
from agents.librarian.validators import validate_librarian_spec


class GoldParseError(ValueError):
    """A gold Markdown file or the locator backfill table could not be parsed
    into a well-formed, locator-complete spec -- a build error, surfaced loudly
    (never a silent skip or a fabricated locator)."""


# ---------------------------------------------------------------------------
# Anchor registry: id -> (gold filename, locator-table spec key, binding?).
#   binding = the locators index a FROZEN canonical text (real D6 locators).
#   str / drf are frozen; mom6 is a pending-freeze parse (provisional offsets).
# ---------------------------------------------------------------------------

_HERE = Path(__file__).resolve().parent
_LOCATOR_REPORT = _HERE / "locator_backfill_report.md"

_ANCHORS: dict[str, dict[str, Any]] = {
    "str": {"file": "gold_str_drr_2026.md", "spec_key": "str_drr_2026", "binding": True},
    "drf": {"file": "gold_drf_bbw_2019.md", "spec_key": "drf_bbw_2019", "binding": True},
    "mom6": {"file": "gold_mom6_jnps_2013.md", "spec_key": "mom6_jnps_2013", "binding": False},
}

# Integer-valued schema fields -- parse their STATED value as an int.
_INT_FIELDS: frozenset[str] = frozenset(
    {"n_groups", "control_n_groups", "signal_lag", "min_bonds", "holding_period",
     "realisation_min_survivors", "hac_lags"}
)

# The line markers/glyphs a gold field line may carry as a prefix.
_MARKER_RE = re.compile(r"^\s*(?:\[[A-Z]+\]|◇|⚠)?\s*")


def is_binding(anchor_id: str) -> bool:
    """True iff the anchor's locators index a FROZEN canonical text (binding D6
    locators): str / drf -> True, mom6 -> False (pending JNPS freeze)."""
    try:
        return bool(_ANCHORS[anchor_id]["binding"])
    except KeyError:
        raise GoldParseError(
            f"unknown anchor_id {anchor_id!r}; expected one of {sorted(_ANCHORS)}"
        ) from None


# ---------------------------------------------------------------------------
# Locator index -- parse locator_backfill_report.md's Markdown table.
# key: (spec_key, page, quote_first70_rstripped) -> set of (char_start, char_end)
# ---------------------------------------------------------------------------

_TABLE_ROW_RE = re.compile(
    r"^\|\s*([A-Za-z0-9_]+)\s*\|\s*(\d+)\s*\|\s*(\d+)\s*\|\s*(\d+)\s*\|\s*(.*?)\s*\|\s*$"
)


class _LocatorIndex:
    """The parsed backfill table, resolvable by (spec_key, page, quote)."""

    def __init__(self, report_path: Path) -> None:
        if not report_path.exists():
            raise GoldParseError(f"locator backfill report not found at {report_path}")
        # (spec_key, page, first70) -> set of distinct (char_start, char_end)
        self._index: dict[tuple[str, int, str], set[tuple[int, int]]] = {}
        for line in report_path.read_text(encoding="utf-8").splitlines():
            m = _TABLE_ROW_RE.match(line)
            if m is None:
                continue
            spec_key, page, cs, ce, quote = m.groups()
            # Skip the header separator / non-data rows (spec_key would be e.g. '---').
            if page.strip() == "" or not page.isdigit():
                continue
            key = (spec_key, int(page), quote.rstrip())
            self._index.setdefault(key, set()).add((int(cs), int(ce)))

    def resolve(self, spec_key: str, page: int, quote: str, *, field: str) -> Locator:
        """Resolve one STATED quote to a Locator. The match key is
        ``(spec_key, page, quote[:70])`` with trailing whitespace stripped (the
        table stores a 70-char prefix, trailing space trimmed by Markdown).

        No matching row -> raise (naming field + quote). Multiple DISTINCT offset
        rows -> raise (ambiguous). Duplicate rows (same offsets) resolve cleanly."""
        first70 = quote[:70].rstrip()
        offsets = self._index.get((spec_key, page, first70))
        if not offsets:
            raise GoldParseError(
                f"no locator row for field {field!r}: spec={spec_key!r} page={page} "
                f"quote[:70]={first70!r} -- STATED requires a real locator (D7); refusing "
                "to fabricate one"
            )
        if len(offsets) > 1:
            raise GoldParseError(
                f"ambiguous locator for field {field!r}: spec={spec_key!r} page={page} "
                f"quote[:70]={first70!r} matches multiple DISTINCT offsets {sorted(offsets)}"
            )
        cs, ce = next(iter(offsets))
        return Locator(page=page, char_start=cs, char_end=ce)


# ---------------------------------------------------------------------------
# Field-block parsing.
#
# A gold code block is a sequence of fields. A line begins a NEW field iff, after
# stripping leading markers/whitespace, it starts with ``<known_field>:``. Every
# other non-comment/non-blank line is a continuation of the current field. This
# is what makes the stray ``return (price convention)`` note line (a non-schema
# token with a space) fold harmlessly into the preceding field.
# ---------------------------------------------------------------------------

# A ParsedField collects the raw text of one field: its head (post-colon, first
# line) plus all continuation lines, joined so multi-line quotes/dicts re-form.
class _ParsedField:
    __slots__ = ("name", "text")

    def __init__(self, name: str, text: str) -> None:
        self.name = name
        self.text = text


def _split_fields(block: str, known_names: set[str]) -> dict[str, _ParsedField]:
    """Split a code block into {field_name: _ParsedField}. A field's text is the
    remainder of its first line plus every following continuation line (comments
    and blanks dropped), joined by ' ' so wrapped quotes/dicts re-form."""
    fields: dict[str, _ParsedField] = {}
    cur_name: str | None = None
    cur_parts: list[str] = []

    def flush() -> None:
        if cur_name is not None:
            if cur_name in fields:
                raise GoldParseError(f"duplicate field {cur_name!r} in a single block")
            fields[cur_name] = _ParsedField(cur_name, " ".join(cur_parts).strip())

    for raw in block.splitlines():
        stripped = raw.strip()
        if stripped == "" or stripped.startswith("#"):
            continue  # blank / comment -- never terminates the current field
        head = _MARKER_RE.sub("", raw)  # drop a leading [MARKER]/◇/⚠ + whitespace
        m = re.match(r"^([a-z][a-z0-9_]*)\s*:(.*)$", head)
        if m is not None and m.group(1) in known_names:
            flush()
            cur_name = m.group(1)
            cur_parts = [m.group(2).strip()]
        else:
            if cur_name is None:
                # A stray line before any field -- ignore (author preamble).
                continue
            cur_parts.append(stripped)
    flush()
    return fields


# ---------------------------------------------------------------------------
# Value / tag extraction from a field's joined text.
# ---------------------------------------------------------------------------

_QUOTE_PAGE_RE = re.compile(r'quote:\s*"(.*?)"\s+page:\s*(\d+)')
_SAME_QUOTE_RE = re.compile(r"[Ss]ame quote as ([a-z_]+)")


class _RawField:
    """The classified content of one gold field, pre-locator-resolution."""

    __slots__ = ("name", "tag", "value", "quote", "page", "reason", "note", "same_quote_as")

    def __init__(self, name: str) -> None:
        self.name = name
        self.tag: str | None = None      # "STATED" | "UNKNOWN"
        self.value: Any = None
        self.quote: str | None = None
        self.page: int | None = None
        self.reason: str | None = None   # UNKNOWN reason code
        self.note: str | None = None     # UNKNOWN / searched note
        self.same_quote_as: str | None = None  # "Same quote as <field>" reuse


def _extract_quote_page(text: str) -> tuple[str | None, int | None]:
    m = _QUOTE_PAGE_RE.search(text)
    if m is None:
        return None, None
    return m.group(1), int(m.group(2))


def _coerce_scalar(name: str, raw: str) -> Any:
    """Coerce a STATED scalar value token to its schema type (ints for the int
    fields; the leading composite token for significance_convention; else str)."""
    raw = raw.strip()
    if name in _INT_FIELDS:
        # Integer fields carry a plain int value EXCEPT hac_lags, whose gold value
        # is a formula string ("floor(T^0.25)") -- kept verbatim (schema permits
        # a composite value in the single Inherited, cf. significance_convention).
        if re.fullmatch(r"-?\d+", raw):
            return int(raw)
        return raw
    if name == "significance_convention":
        # Composite: "hac_t_of_mean (Newey-West)" -> the token the menu expects,
        # dropping a trailing " (...)" annotation. "floor(T^0.25)"-style values
        # (no leading space before '(') are left intact by this rule.
        return re.sub(r"\s+\([^)]*\)\s*$", "", raw).strip()
    return raw


def _classify_field(pf: _ParsedField) -> _RawField:
    """Classify a parsed field into STATED / UNKNOWN with its value + evidence
    fragments. Raises on an unclassifiable field (FAIL LOUD)."""
    rf = _RawField(pf.name)
    text = pf.text

    # --- UNKNOWN compact:  UNKNOWN(reason)   note: "..."  -------------------
    m = re.match(r'^\s*UNKNOWN\(([a-z_]+)\)(.*)$', text)
    if m is not None:
        rf.tag = "UNKNOWN"
        rf.reason = m.group(1)
        note_m = re.search(r'note:\s*"(.*)"', m.group(2), re.S)
        rf.note = note_m.group(1).strip() if note_m else f"not stated ({rf.reason})"
        return rf

    # --- verbose forms:  value: X   tag: STATED | reason: R  ----------------
    vm = re.match(r'^\s*value:\s*(.*)$', text)
    if vm is not None:
        rest = vm.group(1)
        if re.search(r'\btag:\s*STATED\b', rest):
            rf.tag = "STATED"
            # value = everything up to the first "  tag:"/"  note:"/"  quote:"/"  bounds:"
            val_token = re.split(
                r'\s+(?:tag:|note:|quote:|bounds:|reason:)', rest, maxsplit=1
            )[0].strip()
            rf.value = _coerce_scalar(pf.name, val_token)
            rf.quote, rf.page = _extract_quote_page(text)
            sm = _SAME_QUOTE_RE.search(text)
            if sm is not None:
                rf.same_quote_as = sm.group(1)
            return rf
        rm = re.search(r'\breason:\s*([a-z_]+)', rest)
        if rm is not None or re.match(r'^\s*UNKNOWN\b', rest):
            rf.tag = "UNKNOWN"
            rf.reason = rm.group(1) if rm else "not_stated"
            note_m = re.search(r'(?:searched_note|note):\s*"(.*)"', text, re.S)
            rf.note = note_m.group(1).strip() if note_m else f"not stated ({rf.reason})"
            return rf
        # Implicit STATED (paper_facts form): value: X with a quote+page but no
        # explicit `tag:` keyword -- sample_start / sample_end / universe_filter.
        q_imp, p_imp = _extract_quote_page(text)
        if q_imp is not None and p_imp is not None:
            rf.tag = "STATED"
            val_token = re.split(
                r'\s+(?:tag:|note:|quote:|bounds:|reason:)', rest, maxsplit=1
            )[0].strip()
            if len(val_token) >= 2 and val_token[0] == '"' and val_token[-1] == '"':
                val_token = val_token[1:-1]  # unwrap a quoted string value (universe_filter)
            rf.value = _coerce_scalar(pf.name, val_token)
            rf.quote, rf.page = q_imp, p_imp
            return rf
        # value: X with no tag and no reason -> a bare value we treat below.

    # --- bare structural value (no tag, no quote): e.g. "single_leg", "none"  -
    # These are the plain structural tokens (combiner, control_axis).
    # Callers that need them (combiner) handle the bare token specially; a field
    # that reaches here with a non-empty single token is returned as a bare value.
    token = text.strip()
    if token != "" and re.fullmatch(r'[A-Za-z0-9_]+', token):
        rf.tag = "BARE"
        rf.value = token
        return rf

    raise GoldParseError(
        f"could not classify gold field {pf.name!r}: text={text!r}"
    )


# ---------------------------------------------------------------------------
# Inherited construction from a classified field (+ locator resolution).
# ---------------------------------------------------------------------------

def _build_inherited(
    rf: _RawField,
    idx: _LocatorIndex,
    spec_key: str,
    resolved: dict[str, tuple[str, Locator]],
) -> Inherited:
    """Turn a classified field into an ``Inherited``, resolving its STATED locator.

    ``resolved`` maps already-built STATED field-name -> (quote, Locator) so a
    field whose gold text says "Same quote as <other>" (e.g. ``hac_lags``) reuses
    the other field's resolved quote + locator instead of carrying its own."""
    if rf.tag == "UNKNOWN":
        return Inherited(None, "UNKNOWN", Evidence(note=rf.note, unknown_reason=rf.reason))

    if rf.tag == "STATED":
        if rf.same_quote_as is not None and (rf.quote is None or rf.page is None):
            # Reuse another field's resolved quote + locator (no own quote line).
            other = rf.same_quote_as
            if other not in resolved:
                raise GoldParseError(
                    f"field {rf.name!r} says 'Same quote as {other}' but {other!r} has "
                    "no resolved quote/locator yet (ordering or naming error)"
                )
            quote, locator = resolved[other]
        else:
            if rf.quote is None or rf.page is None:
                raise GoldParseError(
                    f"STATED field {rf.name!r} has no quote+page (and no 'Same quote as')"
                )
            locator = idx.resolve(spec_key, rf.page, rf.quote, field=rf.name)
            quote = rf.quote
        resolved[rf.name] = (quote, locator)
        return Inherited(rf.value, "STATED", Evidence(quote=quote, locator=locator))

    raise GoldParseError(
        f"field {rf.name!r} has unexpected tag {rf.tag!r} for an Inherited"
    )


# ---------------------------------------------------------------------------
# SignalRef parsing (sort_signal / control_axis).
# ---------------------------------------------------------------------------

_CONCEPT_RE = re.compile(r"concept_id:\s*([A-Za-z0-9_]+)")
_AS_DESC_RE = re.compile(
    r'as_described:\s*\{\s*label:\s*"(.*?)"\s*,\s*quote:\s*"(.*?)"\s*,\s*page:\s*(\d+)\s*\}',
    re.S,
)


def _parse_signal_ref(
    pf: _ParsedField,
    idx: _LocatorIndex,
    spec_key: str,
) -> SignalRef | None:
    """Parse a ``sort_signal`` / ``control_axis`` field into a SignalRef, or None
    for ``control_axis: none``. concept_id is STATED off the as_described quote's
    resolved locator (the paper's own words behind the concept mapping)."""
    text = pf.text.strip()
    if pf.name == "control_axis" and re.fullmatch(r"none", text):
        return None

    cm = _CONCEPT_RE.search(text)
    if cm is None:
        raise GoldParseError(f"{pf.name!r} has no concept_id: text={text!r}")
    concept = cm.group(1)

    dm = _AS_DESC_RE.search(text)
    if dm is None:
        raise GoldParseError(
            f"{pf.name!r} has no parseable as_described {{label, quote, page}}: {text!r}"
        )
    label, quote, page = dm.group(1), dm.group(2), int(dm.group(3))
    locator = idx.resolve(spec_key, page, quote, field=f"{pf.name}.as_described")
    located = LocatedQuote(
        text=quote, page=page, char_start=locator.char_start, char_end=locator.char_end
    )
    concept_inh = Inherited(
        concept, "STATED", Evidence(quote=quote, locator=locator)
    )
    return SignalRef(
        concept_id=concept_inh,
        as_described=DescribedSignal(label=label, quotes=(located,)),
        parameters={},
    )


# ---------------------------------------------------------------------------
# method_summary (multi-line free text ending in "Locating quotes: (1)... (2)...").
# ---------------------------------------------------------------------------

def _build_method_summary(pf: _ParsedField, construction: tuple[str, Locator]) -> MethodSummary:
    """Build the MethodSummary. The prose summary is authored synthesis (D16), not
    a verbatim quote; to satisfy the STATED locator discipline we bind it to the
    construction sentence's resolved quote + locator (the same table-backed quote
    that grounds formation_structure) and carry that as the single supporting
    LocatedQuote. The full authored prose is the Inherited value."""
    prose = pf.text.strip()
    quote, locator = construction
    summary = Inherited(prose, "STATED", Evidence(quote=quote, locator=locator))
    located = LocatedQuote(
        text=quote, page=locator.page, char_start=locator.char_start, char_end=locator.char_end
    )
    return MethodSummary(summary=summary, quotes=(located,))


# ---------------------------------------------------------------------------
# claimed_headline_metric dict parsing.
# ---------------------------------------------------------------------------

_METRIC_RE = re.compile(
    r"value:\s*\{\s*mean:\s*(-?\d+(?:\.\d+)?)\s*,\s*t_stat:\s*(-?\d+(?:\.\d+)?)\s*,\s*"
    r"unit:\s*([A-Za-z0-9_]+)\s*\}"
)


def _parse_claimed_headline_metric(
    pf: _ParsedField, idx: _LocatorIndex, spec_key: str
) -> Inherited:
    m = _METRIC_RE.search(pf.text)
    if m is None:
        raise GoldParseError(
            f"claimed_headline_metric has no parseable value dict: {pf.text!r}"
        )
    value = {"mean": float(m.group(1)), "t_stat": float(m.group(2)), "unit": m.group(3)}
    quote, page = _extract_quote_page(pf.text)
    if quote is None or page is None:
        raise GoldParseError("claimed_headline_metric has no quote+page")
    locator = idx.resolve(spec_key, page, quote, field="claimed_headline_metric")
    return Inherited(value, "STATED", Evidence(quote=quote, locator=locator))


# ---------------------------------------------------------------------------
# Header parsing.
# ---------------------------------------------------------------------------

_HEADER_HASH_RE = re.compile(r"normalise_sha256\s+([0-9a-f]{64})")


def _parse_header(
    fields: dict[str, _ParsedField],
    paper_id: str,
    strategy_label_value: str,
    construction: tuple[str, Locator],
) -> SpecHeader:
    """Build SpecHeader. ``strategy_label`` is a STATED Inherited whose evidence is
    the construction sentence's table-backed quote + locator: the header's own
    ``strategy_quote`` is a display variant NOT present in the backfill table (str,
    mom6) -- its table-backed core is exactly the formation_structure quote, which
    the construction pair carries. Reusing it keeps every STATED locator REAL."""
    registry_version = fields["registry_version"].text.strip()
    hash_field = fields["canonical_text_hash"].text
    hm = _HEADER_HASH_RE.search(hash_field)
    if hm is None:
        raise GoldParseError(
            f"header canonical_text_hash has no normalise_sha256: {hash_field!r}"
        )
    canonical_text_hash = hm.group(1)

    quote, locator = construction
    strategy_label = Inherited(
        strategy_label_value, "STATED", Evidence(quote=quote, locator=locator)
    )
    return SpecHeader(
        paper_id=paper_id,
        strategy_label=strategy_label,
        registry_version=registry_version,
        registry_hash="gold",  # golds do not state a registry hash; placeholder
        silence_table_version="v1.1",
        canonical_text_hash=canonical_text_hash,
    )


# ---------------------------------------------------------------------------
# Block navigation: pull the code blocks out of the gold Markdown.
# ---------------------------------------------------------------------------

_CODE_BLOCK_RE = re.compile(r"```(.*?)```", re.S)
_SECTION_RE = re.compile(r"^#{2,3}\s+(.*)$", re.M)


def _read_code_blocks(md: str) -> list[str]:
    return _CODE_BLOCK_RE.findall(md)


def _all_known_field_names() -> set[str]:
    """Every schema field name that can head a gold line (drives new-field
    detection). Superset is fine -- unknown tokens fold as continuations."""
    names: set[str] = set()
    names |= set(_COMMON_INHERITED_FIELDS)
    names |= set(_LEG_INHERITED_FIELDS)
    names |= set(_PAPER_FACTS_INHERITED_FIELDS)
    names |= {"sort_signal", "control_axis", "combiner"}
    names |= {"formation_structure", "asset_class", "method_summary"}
    names |= {"paper", "strategy_label", "strategy_quote", "registry_version",
              "silence_policy_version", "canonical_text_hash"}
    return names


def _merge_blocks(blocks: list[str], known: set[str]) -> dict[str, _ParsedField]:
    """Parse + merge every code block's fields into one {name: _ParsedField}.
    A field appearing in two blocks is a gold error (raised)."""
    merged: dict[str, _ParsedField] = {}
    for block in blocks:
        for name, pf in _split_fields(block, known).items():
            if name in merged:
                raise GoldParseError(f"field {name!r} appears in more than one block")
            merged[name] = pf
    return merged


# ---------------------------------------------------------------------------
# The public loader.
# ---------------------------------------------------------------------------

def _require(fields: dict[str, _ParsedField], name: str, where: str) -> _ParsedField:
    if name not in fields:
        raise GoldParseError(
            f"gold is missing schema field {name!r} ({where}) -- a schema field absent "
            "from the gold is a hard error, never a silent skip"
        )
    return fields[name]


def load_gold_spec(anchor_id: str) -> StrategySpec:
    """Parse the anchor's hand-authored gold Markdown into a full ``StrategySpec``.

    ``anchor_id`` in {"str", "drf", "mom6"}. Every schema field is parsed (driven
    off the schema field-name tuples); a missing field is a hard error. Every
    STATED field is wired to a real Locator from ``locator_backfill_report.md``.
    The produced spec is validated with ``validate_librarian_spec`` (fail-closed).
    """
    if anchor_id not in _ANCHORS:
        raise GoldParseError(
            f"unknown anchor_id {anchor_id!r}; expected one of {sorted(_ANCHORS)}"
        )
    meta = _ANCHORS[anchor_id]
    gold_path = _HERE / meta["file"]
    spec_key = meta["spec_key"]
    if not gold_path.exists():
        raise GoldParseError(f"gold file not found at {gold_path}")

    idx = _LocatorIndex(_LOCATOR_REPORT)
    known = _all_known_field_names()
    blocks = _read_code_blocks(gold_path.read_text(encoding="utf-8"))
    fields = _merge_blocks(blocks, known)

    # `resolved`: STATED field-name -> (quote, Locator), so 'Same quote as' reuse
    # (hac_lags) can borrow a sibling's resolved locator.
    resolved: dict[str, tuple[str, Locator]] = {}

    # --- the construction pair (grounds strategy_label + method_summary) -------
    # formation_structure's STATED quote is the table-backed construction sentence.
    fs_raw = _classify_field(_require(fields, "formation_structure", "Part 1"))
    if fs_raw.tag != "STATED" or fs_raw.quote is None or fs_raw.page is None:
        raise GoldParseError("formation_structure must be STATED with a quote+page")
    fs_locator = idx.resolve(spec_key, fs_raw.page, fs_raw.quote, field="formation_structure")
    resolved["formation_structure"] = (fs_raw.quote, fs_locator)
    construction = (fs_raw.quote, fs_locator)
    formation_structure = Inherited(
        fs_raw.value, "STATED", Evidence(quote=fs_raw.quote, locator=fs_locator)
    )

    # --- Part 1 ----------------------------------------------------------------
    asset_class = _build_inherited(
        _classify_field(_require(fields, "asset_class", "Part 1")),
        idx, spec_key, resolved,
    )
    method_summary = _build_method_summary(
        _require(fields, "method_summary", "Part 1"), construction
    )
    part1 = Part1(
        formation_structure=formation_structure,
        asset_class=asset_class,
        method_summary=method_summary,
    )

    # --- Leg (sort block) ------------------------------------------------------
    sort_signal = _parse_signal_ref(
        _require(fields, "sort_signal", "sort block"), idx, spec_key
    )
    if sort_signal is None:
        raise GoldParseError("sort_signal cannot be 'none'")
    control_axis = _parse_signal_ref(
        _require(fields, "control_axis", "sort block"), idx, spec_key
    )

    leg_kwargs: dict[str, Inherited] = {}
    for name in _LEG_INHERITED_FIELDS:
        leg_kwargs[name] = _build_inherited(
            _classify_field(_require(fields, name, "sort block leg")),
            idx, spec_key, resolved,
        )
    leg = Leg(sort_signal=sort_signal, control_axis=control_axis, **leg_kwargs)

    # --- combiner (bare structural token; grounded on sort_kind's quote) -------
    combiner_pf = _require(fields, "combiner", "sort block")
    combiner_raw = _classify_field(combiner_pf)
    if combiner_raw.tag != "BARE" or combiner_raw.value is None:
        raise GoldParseError(
            f"combiner expected a bare structural token; got tag={combiner_raw.tag!r}"
        )
    # single_leg is structurally entailed by the single long-short construction the
    # sort_kind quote describes -> reuse sort_kind's resolved quote + locator so the
    # STATED locator is real (no fabrication; keeps the gold's 'zero INFERRED' audit).
    if "sort_kind" not in resolved:
        raise GoldParseError("combiner grounding needs sort_kind resolved (STATED) first")
    ck_quote, ck_locator = resolved["sort_kind"]
    combiner = Combiner(
        kind=Inherited(
            combiner_raw.value, "STATED", Evidence(quote=ck_quote, locator=ck_locator)
        )
    )

    # --- common block (28 fields) ---------------------------------------------
    common_kwargs: dict[str, Inherited] = {}
    for name in _COMMON_INHERITED_FIELDS:
        common_kwargs[name] = _build_inherited(
            _classify_field(_require(fields, name, "common block")),
            idx, spec_key, resolved,
        )

    part2 = Part2(legs=(leg,), combiner=combiner, **common_kwargs)

    # --- paper_facts -----------------------------------------------------------
    pf_kwargs: dict[str, Inherited] = {}
    for name in _PAPER_FACTS_INHERITED_FIELDS:
        pfield = _require(fields, name, "paper_facts")
        if name == "claimed_headline_metric":
            pf_kwargs[name] = _parse_claimed_headline_metric(pfield, idx, spec_key)
        else:
            pf_kwargs[name] = _build_inherited(
                _classify_field(pfield), idx, spec_key, resolved
            )
    paper_facts = PaperFacts(**pf_kwargs)

    # --- header ----------------------------------------------------------------
    paper_id = _require(fields, "paper", "header").text.strip()
    strategy_label_value = _require(fields, "strategy_label", "header").text.strip()
    header = _parse_header(fields, paper_id, strategy_label_value, construction)

    spec = StrategySpec(header=header, part1=part1, part2=part2, paper_facts=paper_facts)

    # --- fail-closed validation (D8 negatives + Guard 1) -----------------------
    violations = validate_librarian_spec(spec)
    if violations:
        rendered = "; ".join(f"{v.path}: {v.message}" for v in violations)
        raise GoldParseError(
            f"gold spec for {anchor_id!r} failed validate_librarian_spec: {rendered}"
        )
    return spec
