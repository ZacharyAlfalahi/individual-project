#!/usr/bin/env python
"""
Naive single-prompt baseline (referents §2, RATIFIED 2026-09-03 as EXPLICITLY
NON-REPORTABLE-AS-SYSTEM-OUTPUT — the contract prohibits the batch-all
configuration for reported figures; this is a comparison harness only, one row
of the referents table).

Design per the registration: ONE call per (paper, model) — canonical text + the
frozen instruction below + the field list, asking for EVERY field with a
verbatim supporting quote each. Parsed with the production JSON parser; quotes
located with the production locator (the quote gate analogue: a value without a
locating quote does not ship); values scored with the production compare_field
against the ratified golds. One shot, no retries, prompt frozen at registration.

Cost: 3 anchors x 2 models = 6 whole-paper calls.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import yaml

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from agents.librarian.config import load_canonical_text  # noqa: E402
from agents.librarian.pipeline.real_client import (  # noqa: E402
    _parse_json,
    make_backend,
)
from agents.librarian.schema import fields as F  # noqa: E402
from agents.scientist.researcher.cache import ResponseCache  # noqa: E402
from evaluation.harness.compare_policy import Comparability, compare_field  # noqa: E402
from scripts.run_closed_book_probe import _LEG_FIELDS, _gold_map  # noqa: E402
from scripts.run_librarian import _load_dotenv  # noqa: E402

ANCHOR_TEXT = {"drf": "evaluation/canonical_texts/bbw_2019.frozen.yaml",
               "mom6": "evaluation/canonical_texts/jnps_2013.frozen.yaml",
               "str": "evaluation/canonical_texts/drr_2026.frozen.yaml"}

_FIELDS = ((F.FORMATION_STRUCTURE, F.ASSET_CLASS) + tuple(F.COMMON_FIELDS)
           + (F.SORT_SIGNAL,) + _LEG_FIELDS
           + (F.COMBINER, F.SAMPLE_START, F.SAMPLE_END, F.CLAIMED_HEADLINE_METRIC))

# FROZEN at registration (referents §2): no retries, no iteration, no edits.
_INSTRUCTION = (
    "You are given the full text of an academic finance paper describing a "
    "corporate bond trading strategy. Fill in EVERY field listed below for the "
    "paper's headline strategy. Return ONLY one JSON object mapping each field "
    "name to {\"value\": <value>, \"quote\": <verbatim supporting quote copied "
    "character-for-character from the paper>}. If the paper does not state a "
    "field, use {\"value\": null, \"quote\": null}.\n\nFIELDS: ")


def _backends():
    _load_dotenv(_REPO_ROOT / ".env")
    stack = yaml.safe_load((_REPO_ROOT / "docs" / "thresholds.yaml")
                           .read_text(encoding="utf-8"))["librarian"]["model_stack"]
    block = stack["phase_f"]
    return [(block[s]["model_id"], make_backend(
        block[s]["vendor"], block[s]["model_id"],
        os.environ.get(block[s]["api_key_env"], ""),
        temperature=float(stack.get("temperature", 0))))
        for s in ("model_a", "model_b")]


def main() -> int:
    cache = ResponseCache(_REPO_ROOT / "runs" / "librarian_cache")
    backends = _backends()
    per_cell, rows = {}, []

    for anchor, text_path in ANCHOR_TEXT.items():
        ct = load_canonical_text(_REPO_ROOT / text_path)
        paper_text = "\n\n".join(ct.pages)
        gold = _gold_map(anchor)
        prompt = (f"PAPER TEXT:\n<<<\n{paper_text}\n>>>\n\n"
                  + _INSTRUCTION + ", ".join(_FIELDS))
        for model_id, backend in backends:
            hit = cache.get(prompt, f"naive:{model_id}", 0)
            if hit is None:
                raw, _v = backend.generate(prompt, 8000)
                cache.put(prompt, f"naive:{model_id}", 0, raw)
            else:
                raw = hit
            parsed = _parse_json(raw) or {}
            universe = shipped = correct = overclaim = 0
            for name in _FIELDS:
                g = gold.get(name)
                g_val = g.value if hasattr(g, "value") else g
                g_tag = g.tag if hasattr(g, "tag") else "SIGNAL"
                if name == F.SORT_SIGNAL:
                    g_val, g_tag = gold[F.SORT_SIGNAL], "STATED"
                universe += 1
                entry = parsed.get(name) if isinstance(parsed, dict) else None
                value = entry.get("value") if isinstance(entry, dict) else None
                quote = entry.get("quote") if isinstance(entry, dict) else None
                ships = (value is not None and isinstance(quote, str)
                         and ct.locate(quote) is not None)   # the quote-gate analogue
                if not ships:
                    rows.append({"anchor": anchor, "model": model_id, "field": name,
                                 "outcome": "abstained"})
                    continue
                shipped += 1
                if g_tag != "STATED" or g_val is None:
                    overclaim += 1
                    rows.append({"anchor": anchor, "model": model_id, "field": name,
                                 "outcome": "overclaim_gold_silent"})
                    continue
                cmp = compare_field(name, g_val, value)
                ok = bool(cmp.comparability == Comparability.COMPARABLE and cmp.equal)
                correct += int(ok)
                rows.append({"anchor": anchor, "model": model_id, "field": name,
                             "outcome": "correct" if ok else "wrong"})
            per_cell[f"{anchor}/{model_id}"] = {
                "universe": universe, "shipped": shipped, "correct": correct,
                "overclaims": overclaim}
            print(f"[naive] {anchor}/{model_id}: shipped {shipped}/{universe}, "
                  f"correct {correct}/{shipped}, overclaims {overclaim}")

    tot = {k: sum(c[k] for c in per_cell.values())
           for k in ("universe", "shipped", "correct", "overclaims")}
    summary = {
        "label": "NON-REPORTABLE comparison harness (referents §2)",
        "coverage": f"{tot['shipped']}/{tot['universe']}",
        "selective_accuracy": f"{tot['correct']}/{tot['shipped']}",
        "overclaim_rate": f"{tot['overclaims']}/{tot['shipped']}",
        "per_cell": per_cell,
        "frozen_instruction": _INSTRUCTION,
    }
    out = _REPO_ROOT / "results" / "naive_baseline.json"
    out.write_text(json.dumps({"summary": summary, "rows": rows}, indent=2),
                   encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items()
                      if k not in ("frozen_instruction", "per_cell")}, indent=2))
    print(f"[naive] -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
