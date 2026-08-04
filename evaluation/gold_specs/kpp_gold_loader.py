"""
KPP gold loader (schema v1.2): parse the hand-authored fitted-factor-model gold
(``gold_kpp_ipca.md``) into a full, validating ``StrategySpec``.

The FIRST non-sort gold. It reuses the sort ``gold_loader`` primitives
(``_LocatorIndex``, ``_split_fields``, ``_classify_field``, ``_build_inherited``,
``_build_method_summary``, ``_parse_claimed_headline_metric``, ``_extract_quote_page``,
``_read_code_blocks``, ``_require``, ``_AS_DESC_RE``) but assembles a KPP-shaped
spec: Header + Part 1 + an **estimation block** (11 fields) + an **instrument set**
(29 InstrumentRefs, Table A.I) + paper_facts, plus an in-code all-UNKNOWN **stub
Part2** (the sort block a fitted-model spec does not use, present only to satisfy
``Part2.legs`` non-empty; never scored).

Every STATED quote is wired to a REAL locator resolved from
``locator_backfill_report.md`` (generate the KPP rows first with
``scripts/regenerate_locator_backfill.py kpp``). Fail-loud everywhere, and the
produced spec is validated with BOTH ``validate_librarian_spec`` (registry=None --
the stub carries no real signal) and ``validate_estimation_block`` against the
instrument registry (fail-closed).
"""

from __future__ import annotations

import re
from pathlib import Path

from agents.quant.config import Evidence, Inherited, Locator  # noqa: F401 (Locator via primitives)

from agents.librarian.registries import load_instrument_concept_registry
from agents.librarian.schema import (
    Combiner,
    DescribedSignal,
    EstimationBlock,
    InstrumentRef,
    InstrumentSet,
    Leg,
    LocatedQuote,
    PaperFacts,
    Part1,
    Part2,
    SignalRef,
    SpecHeader,
    StrategySpec,
)
from agents.librarian.schema.estimation_fields import (
    ESTIMATION_FIELDS,
    ESTIMATION_INT_FIELDS,
    ESTIMATION_SET_FIELDS,
)
from agents.librarian.schema.strategy_spec import (
    _COMMON_INHERITED_FIELDS,
    _PAPER_FACTS_INHERITED_FIELDS,
)
from agents.librarian.validators import validate_estimation_block, validate_librarian_spec

from evaluation.gold_specs.gold_loader import (
    _AS_DESC_RE,
    _HERE,
    _LOCATOR_REPORT,
    GoldParseError,
    _LocatorIndex,
    _build_inherited,
    _build_method_summary,
    _classify_field,
    _extract_quote_page,
    _parse_claimed_headline_metric,
    _read_code_blocks,
    _require,
    _split_fields,
)

# KPP anchor metadata (mirrors gold_loader._ANCHORS, kept here too so this loader
# is usable standalone). ``kind: estimation`` routes gold_loader.load_gold_spec here.
_KPP_ANCHOR = {
    "file": "gold_kpp_ipca.md",
    "spec_key": "kpp_2023",
    "binding": True,
    "kind": "estimation",
}

_SECTION_RE = re.compile(r"^##\s+(.*)$", re.M)
_CID_LINE_RE = re.compile(r"^concept_id:\s*([A-Za-z0-9_]+)\s*$", re.M)
_SOURCE_CLASS_RE = re.compile(r"^source_class:\s*([a-z]+)\b")
_HEADER_HASH_RE = re.compile(r"normalise_sha256\s+([0-9a-f]{64})")

_HEADER_FIELDS = {
    "paper", "strategy_label", "registry_version",
    "silence_policy_version", "canonical_text_hash",
}
_PART1_FIELDS = {"formation_structure", "asset_class", "method_summary"}


# ---------------------------------------------------------------------------
# Section navigation.
# ---------------------------------------------------------------------------

def _sections(md: str) -> list[tuple[str, str]]:
    """Split the gold Markdown into (title, body) by ``## `` headers, in order."""
    matches = list(_SECTION_RE.finditer(md))
    out: list[tuple[str, str]] = []
    for i, m in enumerate(matches):
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(md)
        out.append((m.group(1).strip(), md[start:end]))
    return out


def _find_body(sections: list[tuple[str, str]], predicate) -> str:
    for title, body in sections:
        if predicate(title):
            return body
    raise GoldParseError(f"KPP gold is missing a required section (predicate {predicate!r})")


# ---------------------------------------------------------------------------
# Estimation block.
# ---------------------------------------------------------------------------

def _build_estimation(block: str, idx: _LocatorIndex, spec_key: str) -> EstimationBlock:
    fields = _split_fields(block, set(ESTIMATION_FIELDS))
    resolved: dict[str, tuple[str, Locator]] = {}
    kwargs: dict[str, Inherited] = {}
    for name in ESTIMATION_FIELDS:
        rf = _classify_field(_require(fields, name, "estimation block"))
        if rf.tag == "STATED":
            if name in ESTIMATION_INT_FIELDS:
                rf.value = int(str(rf.value).strip())
            elif name in ESTIMATION_SET_FIELDS:
                rf.value = frozenset(int(n) for n in re.findall(r"-?\d+", str(rf.value)))
        kwargs[name] = _build_inherited(rf, idx, spec_key, resolved)
    return EstimationBlock(**kwargs)


# ---------------------------------------------------------------------------
# Instrument set (29 InstrumentRefs).
# ---------------------------------------------------------------------------

def _unknown(note: str) -> Inherited:
    return Inherited(None, "UNKNOWN", Evidence(note=note, unknown_reason="not_stated"))


def _parse_instrument_block(block: str, idx: _LocatorIndex, spec_key: str) -> InstrumentRef:
    cm = _CID_LINE_RE.search(block)
    if cm is None:
        raise GoldParseError(f"instrument block has no concept_id line: {block!r}")
    concept = cm.group(1)

    sc_line = next(
        (ln.strip() for ln in block.splitlines() if ln.strip().startswith("source_class:")),
        None,
    )
    if sc_line is None:
        raise GoldParseError(f"instrument {concept!r} has no source_class line")
    scm = _SOURCE_CLASS_RE.search(sc_line)
    if scm is None:
        raise GoldParseError(f"instrument {concept!r} source_class line unparseable: {sc_line!r}")
    sc_val = scm.group(1)
    sc_quote, sc_page = _extract_quote_page(sc_line)
    if sc_quote is None or sc_page is None:
        raise GoldParseError(f"instrument {concept!r} source_class has no quote+page")
    sc_locator = idx.resolve(spec_key, sc_page, sc_quote, field=f"{concept}.source_class")

    dm = _AS_DESC_RE.search(block)
    if dm is None:
        raise GoldParseError(f"instrument {concept!r} has no parseable as_described")
    label, ad_quote, ad_page = dm.group(1), dm.group(2), int(dm.group(3))
    ad_locator = idx.resolve(spec_key, ad_page, ad_quote, field=f"{concept}.as_described")

    return InstrumentRef(
        concept_id=Inherited(concept, "STATED", Evidence(quote=ad_quote, locator=ad_locator)),
        source_class=Inherited(sc_val, "STATED", Evidence(quote=sc_quote, locator=sc_locator)),
        transform=_unknown(
            "uniform rank-standardization captured in estimation.characteristic_preprocessing; "
            "not stated per instrument"
        ),
        lag=_unknown("no per-instrument reporting/availability lag stated"),
        as_described=DescribedSignal(
            label=label,
            quotes=(
                LocatedQuote(
                    text=ad_quote, page=ad_page,
                    char_start=ad_locator.char_start, char_end=ad_locator.char_end,
                ),
            ),
        ),
    )


def _build_instruments(section_body: str, idx: _LocatorIndex, spec_key: str) -> InstrumentSet:
    blocks = _read_code_blocks(section_body)
    if not blocks:
        raise GoldParseError("Instruments section has no instrument code blocks")
    refs = tuple(_parse_instrument_block(b, idx, spec_key) for b in blocks)
    return InstrumentSet(instruments=refs)


# ---------------------------------------------------------------------------
# Header + stub Part2.
# ---------------------------------------------------------------------------

def _parse_header(header_block: str, construction: tuple[str, Locator]) -> SpecHeader:
    fields = _split_fields(header_block, _HEADER_FIELDS)
    paper_id = _require(fields, "paper", "header").text.strip()
    strategy_label_value = _require(fields, "strategy_label", "header").text.strip()
    registry_version = _require(fields, "registry_version", "header").text.strip()
    hash_field = _require(fields, "canonical_text_hash", "header").text
    hm = _HEADER_HASH_RE.search(hash_field)
    if hm is None:
        raise GoldParseError(f"header canonical_text_hash has no normalise_sha256: {hash_field!r}")
    quote, locator = construction
    return SpecHeader(
        paper_id=paper_id,
        strategy_label=Inherited(strategy_label_value, "STATED", Evidence(quote=quote, locator=locator)),
        registry_version=registry_version,
        registry_hash="gold",  # golds do not state a registry hash; placeholder
        silence_table_version="v1.2",
        canonical_text_hash=hm.group(1),
    )


def _stub_part2() -> Part2:
    """A minimal all-UNKNOWN sort block: the fitted-model spec's Part2, present
    only to satisfy the non-empty ``legs`` guard. Never scored (parallel path)."""
    note = "fitted-model spec: sort block is a schema stub, not construction (never scored)"
    stub_leg = Leg(
        sort_signal=SignalRef(concept_id=_unknown(note), as_described=DescribedSignal(label="schema stub")),
        sort_kind=_unknown(note),
        bucketing_method=_unknown(note),
        n_groups=_unknown(note),
        stripe_aggregation=_unknown(note),
        control_missing_policy=_unknown(note),
        long_leg=_unknown(note),
        signal_transform=_unknown(note),
        control_n_groups=_unknown(note),
    )
    common = {name: _unknown(note) for name in _COMMON_INHERITED_FIELDS}
    return Part2(legs=(stub_leg,), combiner=Combiner(kind=_unknown(note)), **common)


# ---------------------------------------------------------------------------
# The public loader.
# ---------------------------------------------------------------------------

def load_kpp_gold_spec(anchor_id: str = "kpp") -> StrategySpec:
    """Parse ``gold_kpp_ipca.md`` into a full, validating ``StrategySpec``
    (header + Part 1 + estimation block + 29-instrument set + paper_facts +
    stub Part2). Every STATED field is wired to a real Locator from
    ``locator_backfill_report.md``; the spec is validated fail-closed."""
    if anchor_id != "kpp":
        raise GoldParseError(f"load_kpp_gold_spec only handles 'kpp'; got {anchor_id!r}")
    spec_key = _KPP_ANCHOR["spec_key"]
    gold_path = _HERE / _KPP_ANCHOR["file"]
    if not gold_path.exists():
        raise GoldParseError(f"KPP gold file not found at {gold_path}")

    idx = _LocatorIndex(_LOCATOR_REPORT)
    md = gold_path.read_text(encoding="utf-8")
    sections = _sections(md)

    header_block = _read_code_blocks(_find_body(sections, lambda t: t == "Header"))[0]
    part1_block = _read_code_blocks(_find_body(sections, lambda t: t.startswith("Part 1")))[0]
    est_block = _read_code_blocks(_find_body(sections, lambda t: t.startswith("Estimation")))[0]
    instr_body = _find_body(sections, lambda t: t == "Instruments")
    pf_block = _read_code_blocks(_find_body(sections, lambda t: t.startswith("paper_facts")))[0]

    # --- Part 1 (formation_structure grounds strategy_label + method_summary) --
    p1_fields = _split_fields(part1_block, _PART1_FIELDS)
    fs_raw = _classify_field(_require(p1_fields, "formation_structure", "Part 1"))
    if fs_raw.tag != "STATED" or fs_raw.quote is None or fs_raw.page is None:
        raise GoldParseError("formation_structure must be STATED with a quote+page")
    fs_locator = idx.resolve(spec_key, fs_raw.page, fs_raw.quote, field="formation_structure")
    construction = (fs_raw.quote, fs_locator)
    resolved: dict[str, tuple[str, Locator]] = {"formation_structure": construction}
    formation_structure = Inherited(
        fs_raw.value, "STATED", Evidence(quote=fs_raw.quote, locator=fs_locator)
    )
    asset_class = _build_inherited(
        _classify_field(_require(p1_fields, "asset_class", "Part 1")), idx, spec_key, resolved
    )
    method_summary = _build_method_summary(
        _require(p1_fields, "method_summary", "Part 1"), construction
    )
    part1 = Part1(
        formation_structure=formation_structure, asset_class=asset_class, method_summary=method_summary
    )

    # --- header / estimation / instruments / paper_facts -----------------------
    header = _parse_header(header_block, construction)
    estimation = _build_estimation(est_block, idx, spec_key)
    instruments = _build_instruments(instr_body, idx, spec_key)

    pf_fields = _split_fields(pf_block, set(_PAPER_FACTS_INHERITED_FIELDS))
    pf_kwargs: dict[str, Inherited] = {}
    for name in _PAPER_FACTS_INHERITED_FIELDS:
        pfield = _require(pf_fields, name, "paper_facts")
        if name == "claimed_headline_metric":
            pf_kwargs[name] = _parse_claimed_headline_metric(pfield, idx, spec_key)
        else:
            pf_kwargs[name] = _build_inherited(_classify_field(pfield), idx, spec_key, resolved)
    paper_facts = PaperFacts(**pf_kwargs)

    spec = StrategySpec(
        header=header,
        part1=part1,
        part2=_stub_part2(),
        paper_facts=paper_facts,
        estimation=estimation,
        instruments=instruments,
    )

    # --- fail-closed validation -----------------------------------------------
    violations = validate_librarian_spec(spec)  # registry=None: the stub has no real signal
    violations += validate_estimation_block(spec, load_instrument_concept_registry())
    if violations:
        rendered = "; ".join(f"{v.field}: {v.reason}" for v in violations)
        raise GoldParseError(f"KPP gold failed validation: {rendered}")
    return spec
