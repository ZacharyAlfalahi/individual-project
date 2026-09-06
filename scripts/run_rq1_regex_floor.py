#!/usr/bin/env python
"""
RQ1 regex floor (referents §3, RATIFIED 2026-09-03/04 — see the dated entry in
docs/evaluation/rq1_referents_registration_draft_2026-09-02.md): a no-LLM
reference row for the referents table. Five fields — n_groups,
sample_start, sample_end, holding_period, signal_lag — extracted from the frozen
canonical texts by the PATTERNS FROZEN AT REGISTRATION (reproduced verbatim
below; no post-hoc edits — editing a pattern is fitting the floor), scored
against the ratified golds under the registered rule: first match ships;
gold-STATED + ship → correct/wrong by normalised equality (ints; YEARS for the
sample fields); gold-silent + ship → over-claim; no ship → abstain.

Zero cost, zero LLM, deterministic.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from agents.librarian.config import load_canonical_text  # noqa: E402
from evaluation.gold_specs.gold_loader import load_gold_spec  # noqa: E402

ANCHOR_TEXT = {"drf": "evaluation/canonical_texts/bbw_2019.frozen.yaml",
               "mom6": "evaluation/canonical_texts/jnps_2013.frozen.yaml",
               "str": "evaluation/canonical_texts/drr_2026.frozen.yaml"}

_WORDS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
          "seven": 7, "eight": 8, "nine": 9, "ten": 10}
_GROUP_OF = {"quintile": 5, "decile": 10, "tercile": 3, "tertile": 3, "quartile": 4}
_MONTH = (r"(?:january|february|march|april|may|june|july|august|september|"
          r"october|november|december)")

# The frozen patterns (registration entry, 2026-09-04).
_RE_GROUPS = re.compile(r"\b(quintile|decile|tercile|tertile|quartile)s?\b", re.I)
_RE_HOLDING = re.compile(
    r"\b(one|two|three|four|five|six|seven|eight|nine|ten|\d+)[-\s]months?\s+holding\b"
    r"|\bholding\s+period\s+of\s+(one|two|three|four|five|six|seven|eight|nine|ten|\d+)"
    r"\s+months?\b", re.I)
_RE_RANGE = re.compile(
    r"(?:from|between)\s+(?:" + _MONTH + r"\s+)?(\d{4})\s+"
    r"(?:to|through|until|and|[-–—])\s+(?:" + _MONTH + r"\s+)?(\d{4})", re.I)
_RE_RANGE_BARE = re.compile(r"\b(\d{4})\s*[-–—]\s*(\d{4})\b")
_RE_LAG = re.compile(
    r"\bskip(?:ping)?\s+(?:a|one|1)\s+month\b|\bone[-\s]month\s+(?:gap|lag|skip)\b", re.I)


def _int_of(token: str) -> int:
    return _WORDS.get(token.lower(), 0) or int(token)


def extract_floor(text: str) -> dict[str, object]:
    """The five floor fields from one paper text. First match ships; None = abstain."""
    out: dict[str, object] = {}
    m = _RE_GROUPS.search(text)
    out["n_groups"] = _GROUP_OF[m.group(1).lower()] if m else None
    m = _RE_HOLDING.search(text)
    out["holding_period"] = _int_of(m.group(1) or m.group(2)) if m else None
    m = _RE_RANGE.search(text)
    if m is None:
        for cand in _RE_RANGE_BARE.finditer(text):
            a, b = int(cand.group(1)), int(cand.group(2))
            if 1900 <= a <= 2030 and 1900 <= b <= 2030:
                m = cand
                break
    out["sample_start"] = int(m.group(1)) if m else None
    out["sample_end"] = int(m.group(2)) if m else None
    out["signal_lag"] = 1 if _RE_LAG.search(text) else None
    return out


def _year_of(value: object) -> int | None:
    m = re.search(r"\b(19|20)\d{2}\b", str(value))
    return int(m.group(0)) if m else None


def gold_targets(anchor: str) -> dict[str, tuple[object, str]]:
    """(normalised gold value, tag) per floor field, from the ratified gold."""
    spec = load_gold_spec(anchor)
    leg = spec.part2.legs[0]
    pf = spec.paper_facts
    rows = {
        "n_groups": leg.n_groups,
        "holding_period": spec.part2.holding_period,
        "signal_lag": spec.part2.signal_lag,
        "sample_start": pf.sample_start if pf else None,
        "sample_end": pf.sample_end if pf else None,
    }
    out = {}
    for name, inh in rows.items():
        if inh is None:
            out[name] = (None, "ABSENT")
            continue
        v = inh.value
        if name in ("sample_start", "sample_end") and v is not None:
            v = _year_of(v)                      # registered: YEAR granularity
        elif isinstance(v, str) and v.isdigit():
            v = int(v)                           # "6" and 6 compare equal
        out[name] = (v, inh.tag)
    return out


def score_anchor_floor(anchor: str) -> list[dict]:
    text = "\n\n".join(load_canonical_text(_REPO_ROOT / ANCHOR_TEXT[anchor]).pages)
    shipped = extract_floor(text)
    gold = gold_targets(anchor)
    rows = []
    for name in ("n_groups", "sample_start", "sample_end", "holding_period", "signal_lag"):
        got = shipped[name]
        gv, gtag = gold[name]
        if got is None:
            outcome = "abstain_gold_stated" if gtag == "STATED" else "abstain"
        elif gtag == "STATED":
            outcome = "correct" if got == gv else "wrong"
        else:
            outcome = "overclaim_gold_silent"
        rows.append({"anchor": anchor, "field": name, "regex": got,
                     "gold": gv, "gold_tag": gtag, "outcome": outcome})
    return rows


def main() -> int:
    rows = [r for a in ANCHOR_TEXT for r in score_anchor_floor(a)]
    shipped = [r for r in rows if r["regex"] is not None]
    correct = [r for r in rows if r["outcome"] == "correct"]
    over = [r for r in rows if r["outcome"] == "overclaim_gold_silent"]
    summary = {
        "fields_scored": len(rows),
        "coverage": f"{len(shipped)}/{len(rows)}",
        "selective_accuracy": (f"{len(correct)}/{len(correct) + len([r for r in shipped if r['outcome'] == 'wrong'])}"),
        "correct": len(correct), "wrong": sum(r["outcome"] == "wrong" for r in rows),
        "overclaims": len(over),
        "abstain_gold_stated": sum(r["outcome"] == "abstain_gold_stated" for r in rows),
    }
    out = _REPO_ROOT / "results" / "rq1_regex_floor.json"
    out.write_text(json.dumps({"summary": summary, "rows": rows}, indent=2),
                   encoding="utf-8")
    print(json.dumps(summary, indent=2))
    for r in rows:
        print(f"  {r['anchor']:5s} {r['field']:14s} regex={r['regex']!r:8} "
              f"gold={r['gold']!r:8} ({r['gold_tag']}) -> {r['outcome']}")
    print(f"[regex_floor] -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
