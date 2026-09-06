#!/usr/bin/env python
"""
Entity-masked BBW text + enum (contract §3.5's pre-registered response, fired
2026-09-04 by the closed-book probe exceeding δ_cb: 47.6pp > 15pp).

Deterministic derivation, mechanical only: the frozen bbw_2019 canonical text
and its ratified enum gold pass through the FROZEN mask table below (title,
author names, journal, factor acronyms → neutral tokens, exactly the §3.5
masking set; word-boundary regexes so 'Wen' never touches 'when'). Outputs are
machine-local derived artefacts (the canonical-texts tree is gitignored):

  evaluation/canonical_texts/bbw_2019_masked.frozen.yaml
  evaluation/gold_specs/derived_enum_bbw_2019_masked.local.yaml

The masked run then uses run_librarian's --canonical-text/--gold-enum overrides
-- no registered artefact is touched. The DESCRIPTIVE construction language
("downside risk", "credit rating", VaR definitions) is deliberately NOT masked:
the extraction task must stay intact; only the paper's identity hooks go.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import yaml

_REPO_ROOT = Path(__file__).resolve().parent.parent

# FROZEN mask table (§3.5 set: title, authors, journal, factor names).
MASKS: list[tuple[str, str]] = [
    (r"Common Risk Factors in the Cross-Section of Corporate Bond Returns",
     "[MASKED TITLE]"),
    (r"Common risk factors in the cross-section of corporate bond returns",
     "[masked title]"),
    (r"\bJennie\b", "Author-A"), (r"\bBai\b", "Author-A"),
    (r"\bTuran\b", "Author-B"), (r"\bBali\b", "Author-B"),
    (r"\bQuan\b", "Author-C"), (r"\bWen\b", "Author-C"),
    (r"\bJournal of Financial Economics\b", "[MASKED JOURNAL]"),
    (r"\bDRF\b", "XF1"), (r"\bCRF\b", "XF2"),
    (r"\bLRF\b", "XF3"), (r"\bREV\b", "XF4"),
]


def mask_text(s: str) -> str:
    for pat, repl in MASKS:
        s = re.sub(pat, repl, s)
    return s


def main() -> int:
    src = _REPO_ROOT / "evaluation" / "canonical_texts" / "bbw_2019.frozen.yaml"
    d = yaml.safe_load(src.read_text(encoding="utf-8"))
    n_hits = sum(len(re.findall(pat, p)) for p in d["pages"] for pat, _ in MASKS)
    d["pages"] = [mask_text(p) for p in d["pages"]]
    d["page_canonicalisation"] = d.get("page_canonicalisation", {})
    out_text = _REPO_ROOT / "evaluation" / "canonical_texts" / "bbw_2019_masked.frozen.yaml"
    out_text.write_text(yaml.safe_dump(d, allow_unicode=True, sort_keys=False),
                        encoding="utf-8")

    enum_src = _REPO_ROOT / "evaluation" / "gold_specs" / "enum_bbw_2019.yaml"
    e = yaml.safe_load(enum_src.read_text(encoding="utf-8"))
    for c in e["constructions"]:
        c["name"] = mask_text(c["name"])
        c["quote"] = mask_text(c["quote"])
    e.setdefault("authoring", {})["derived_masked"] = (
        "MECHANICAL derivation 2026-09-04: the ratified enum passed through the "
        "frozen section-3.5 mask table (scripts/build_masked_bbw.py); no new "
        "judgement. Machine-local; never a registered artefact.")
    out_enum = _REPO_ROOT / "evaluation" / "gold_specs" / "derived_enum_bbw_2019_masked.local.yaml"
    out_enum.write_text(yaml.safe_dump(e, allow_unicode=True, sort_keys=False),
                        encoding="utf-8")
    print(f"[mask] {n_hits} entity occurrences masked -> {out_text.name}, {out_enum.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
