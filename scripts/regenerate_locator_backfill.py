"""
Regenerate / verify the locator backfill for an anchor against its FROZEN canonical
text. For every STATED, quote-bearing element of the gold spec, re-locate the quote
via ``locate_quote`` at L1 and emit the binding backfill row
(spec, page, char_start, char_end, quote[:70]). Also verifies each located offset
matches the gold's CURRENT locator (resolved from ``locator_backfill_report.md`` by
``gold_loader``), so a freeze that reproduces the pending parse is a no-op on the
offsets -- they simply become binding -- and any drift is surfaced loudly (exit 1).

Usage: ./.venv/bin/python scripts/regenerate_locator_backfill.py mom6
"""

from __future__ import annotations

import sys
from pathlib import Path

import yaml

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import re  # noqa: E402

from agents.librarian.config.locate import locate_quote  # noqa: E402
from agents.librarian.schema.strategy_spec import (  # noqa: E402
    _COMMON_INHERITED_FIELDS,
    _LEG_INHERITED_FIELDS,
    _PAPER_FACTS_INHERITED_FIELDS,
)
from evaluation.gold_specs.gold_loader import _ANCHORS, _AS_DESC_RE, load_gold_spec  # noqa: E402

_FROZEN = {
    "str": "drr_2026.frozen.yaml",
    "drf": "bbw_2019.frozen.yaml",
    "mom6": "jnps_2013.frozen.yaml",
    "kpp": "kpp_2023.frozen.yaml",
    "crf": "bbw_2019.frozen.yaml",  # CRF shares BBW's frozen canonical text with drf
}

# Golds the loader cannot build until their locators exist (multi-block / multi-leg
# constructions), so their backfill is bootstrapped from the RAW gold Markdown
# instead of a loaded spec: KPP's estimation gold and CRF's three-leg sort gold.
_MARKDOWN_BOOTSTRAP = frozenset({"kpp", "crf"})

# Field-style quote (estimation / paper_facts / instrument source_class lines):
# ``quote: "..."   page: N`` (whitespace before ``page:``). The as_described form
# (``..., page: N``) is handled by _AS_DESC_RE, so this deliberately does NOT match it.
_QUOTE_PAGE_RE = re.compile(r'quote:\s*"(.*?)"\s+page:\s*(\d+)')
_LOCATOR_REPORT = _REPO_ROOT / "evaluation" / "gold_specs" / "locator_backfill_report.md"


def _stated_quotes(spec):
    """Yield (label, Inherited) for every STATED, quote-bearing element of the spec."""
    yield "formation_structure", spec.part1.formation_structure
    yield "asset_class", spec.part1.asset_class
    leg = spec.part2.legs[0]
    for f in _LEG_INHERITED_FIELDS:
        yield f"leg.{f}", getattr(leg, f)
    yield "leg.sort_signal", leg.sort_signal.concept_id
    if leg.control_axis is not None:
        yield "leg.control_axis", leg.control_axis.concept_id
    yield "combiner", spec.part2.combiner.kind
    for f in _COMMON_INHERITED_FIELDS:
        yield f"common.{f}", getattr(spec.part2, f)
    for f in _PAPER_FACTS_INHERITED_FIELDS:
        yield f"paper_facts.{f}", getattr(spec.paper_facts, f)
    yield "header.strategy_label", spec.header.strategy_label


def verify(anchor: str):
    frozen_path = _REPO_ROOT / "evaluation" / "canonical_texts" / _FROZEN[anchor]
    frozen = yaml.safe_load(frozen_path.read_text(encoding="utf-8"))
    if frozen.get("status") != "frozen":
        raise SystemExit(f"{frozen_path} is not status: frozen")
    pages = tuple(frozen["pages"])
    spec_key = _ANCHORS[anchor]["spec_key"]
    spec = load_gold_spec(anchor)

    rows, mismatches, seen = [], [], set()
    for label, inh in _stated_quotes(spec):
        if inh.tag != "STATED":
            continue
        q, loc = inh.evidence.quote, inh.evidence.locator
        m = locate_quote(pages, q, "L1")
        if not m.matched:
            mismatches.append((label, "NOT FOUND in the frozen text", q[:70]))
            continue
        page1 = m.page + 1  # locate_quote page is 0-based; report/gold are 1-based
        got, exp = (page1, m.char_start, m.char_end), (loc.page, loc.char_start, loc.char_end)
        key = (page1, m.char_start, m.char_end)
        if key not in seen:
            seen.add(key)
            rows.append((spec_key, page1, m.char_start, m.char_end, q[:70], m.used_cross_page))
        if got != exp:
            mismatches.append((label, f"offset drift: frozen {got} vs report {exp}", q[:70]))
    return rows, mismatches


def _markdown_quotes(md: str):
    """Yield (quote, stated_page) for every STATED, quote-bearing element of a gold,
    extracted from the RAW Markdown (no gold load -- avoids the backfill-before-load
    chicken-and-egg). Field-style quotes + as_described quotes."""
    for m in _QUOTE_PAGE_RE.finditer(md):
        yield m.group(1), int(m.group(2))
    for m in _AS_DESC_RE.finditer(md):
        yield m.group(2), int(m.group(3))  # as_described quote + page


def verify_from_markdown(anchor: str):
    """Bootstrap + verify a gold's backfill from the raw gold Markdown against the
    frozen text (used for the golds in ``_MARKDOWN_BOOTSTRAP`` the loader cannot
    build until their locators exist). Returns (rows, mismatches). Each STATED quote
    must locate at L1 with no cross-page fallback AND its located page must equal the
    gold's stated page (a page mismatch is an authoring error, surfaced loudly)."""
    frozen_path = _REPO_ROOT / "evaluation" / "canonical_texts" / _FROZEN[anchor]
    frozen = yaml.safe_load(frozen_path.read_text(encoding="utf-8"))
    if frozen.get("status") != "frozen":
        raise SystemExit(f"{frozen_path} is not status: frozen")
    pages = tuple(frozen["pages"])
    spec_key = _ANCHORS[anchor]["spec_key"]
    gold_md = (_REPO_ROOT / "evaluation" / "gold_specs" / _ANCHORS[anchor]["file"]).read_text(
        encoding="utf-8"
    )

    rows, mismatches, seen = [], [], set()
    for q, stated_page in _markdown_quotes(gold_md):
        m = locate_quote(pages, q, "L1")
        if not m.matched:
            mismatches.append((anchor, "NOT FOUND in the frozen text", q[:70]))
            continue
        if m.used_cross_page:
            mismatches.append((anchor, "cross-page fallback (not allowed for a gold quote)", q[:70]))
            continue
        page1 = m.page + 1  # locate_quote page is 0-based; report/gold are 1-based
        if page1 != stated_page:
            mismatches.append(
                (anchor, f"page mismatch: gold says {stated_page}, frozen text is {page1}", q[:70])
            )
            continue
        key = (page1, m.char_start, m.char_end)
        if key not in seen:
            seen.add(key)
            rows.append((spec_key, page1, m.char_start, m.char_end, q[:70]))
    return rows, mismatches


def _write_rows(spec_key: str, rows) -> None:
    """Append the ``spec_key`` rows to locator_backfill_report.md, replacing any
    prior rows for that spec_key. Every other anchor's rows are left byte-untouched."""
    text = _LOCATOR_REPORT.read_text(encoding="utf-8")
    prefix = f"| {spec_key} |"
    kept = [ln for ln in text.splitlines() if not ln.startswith(prefix)]
    # trim trailing blank lines, then append the rows.
    while kept and kept[-1].strip() == "":
        kept.pop()
    new_rows = [f"| {r[0]} | {r[1]} | {r[2]} | {r[3]} | {r[4]} |" for r in rows]
    _LOCATOR_REPORT.write_text("\n".join(kept + new_rows) + "\n", encoding="utf-8")


def main() -> None:
    anchor = sys.argv[1] if len(sys.argv) > 1 else "mom6"
    if anchor in _MARKDOWN_BOOTSTRAP:
        spec_key = _ANCHORS[anchor]["spec_key"]
        rows, mismatches = verify_from_markdown(anchor)
        print(f"=== {anchor}: {len(rows)} distinct STATED-quote locators against the frozen text ===")
        for r in rows:
            print(f"| {r[0]} | {r[1]} | {r[2]} | {r[3]} | {r[4]} |")
        if mismatches:
            print(f"\n!!! {len(mismatches)} MISMATCH(es):")
            for label, why, q in mismatches:
                print(f"  - {label}: {why} | {q!r}")
            raise SystemExit(1)
        _write_rows(spec_key, rows)
        print(f"\nWrote {len(rows)} {spec_key} rows to {_LOCATOR_REPORT.name} -> BINDING.")
        return

    rows, mismatches = verify(anchor)
    cross = sum(1 for r in rows if r[5])
    print(f"=== {anchor}: {len(rows)} distinct STATED-quote locators against the frozen text "
          f"({cross} cross-page) ===")
    for r in rows:
        print(f"| {r[0]} | {r[1]} | {r[2]} | {r[3]} | {r[4]} |")
    if mismatches:
        print(f"\n!!! {len(mismatches)} MISMATCH(es) vs the current report:")
        for label, why, q in mismatches:
            print(f"  - {label}: {why} | {q!r}")
        raise SystemExit(1)
    print("\nAll located quotes match the report offsets -> BINDING against the frozen text.")


if __name__ == "__main__":
    main()
