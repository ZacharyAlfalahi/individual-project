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

from agents.librarian.config.locate import locate_quote  # noqa: E402
from agents.librarian.schema.strategy_spec import (  # noqa: E402
    _COMMON_INHERITED_FIELDS,
    _LEG_INHERITED_FIELDS,
    _PAPER_FACTS_INHERITED_FIELDS,
)
from evaluation.gold_specs.gold_loader import _ANCHORS, load_gold_spec  # noqa: E402

_FROZEN = {"str": "drr_2026.frozen.yaml", "drf": "bbw_2019.frozen.yaml", "mom6": "jnps_2013.frozen.yaml"}


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


def main() -> None:
    anchor = sys.argv[1] if len(sys.argv) > 1 else "mom6"
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
