#!/usr/bin/env python
"""
Relocator bar calibration (DIAGNOSTIC; independent data only).

Applies the pre-registered, target-blind bar-selection rule mechanically —
sweep the grid; the bar is the strictest value with zero false accepts that
meets both recall floors; if zero-FA bars exist but no floor is met, the
loosest zero-FA bar is chosen and every downstream artifact carries the
``recall_floor_unmet`` label; if no zero-FA bar exists the diagnostic does not
run. Concretely:

* POSITIVES — every human-authored gold STATED quote (six golds; kpp via a
  generic estimation-form walker, pre-declared best-effort), each carrying one
  deterministic hash-seeded synthetic edit per registered class. Clean gold quotes
  must pass via the exact short-circuit (asserted, fail-loud). Edits that are
  no-ops, fall below the a-priori length guard, or still locate exactly are
  RECORDED and excluded from recall denominators — never silently dropped.
* NEGATIVES — every gold span's ORIGINAL quote against each of the six
  canonical texts minus its home paper. A false accept is ANY guard-passing
  score at or above the bar. ``bbw_2021`` may genuinely restate ``bbw_2019``
  (same authors); counting such accepts as false accepts is the conservative
  direction (it can only raise the bar), and every negative pair scoring
  >= 0.80 is listed verbatim for inspection.

The 24 target quotes NEVER enter this script. The chosen bar is hand-copied
into ``docs/thresholds.yaml`` (stage B) citing this script's artifact; the
re-scoring loader raises until that happens.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from agents.librarian.config import load_canonical_text                    # noqa: E402
from agents.librarian.config.canonical_text import CanonicalText           # noqa: E402
from agents.librarian.config.normalise import normalise                    # noqa: E402
from evaluation.gold_specs.gold_loader import load_gold_spec               # noqa: E402
from evaluation.harness.field_pairing import iter_gold_fields              # noqa: E402
from evaluation.harness.relocate import score_quote                        # noqa: E402
from evaluation.harness.relocate_thresholds import (                       # noqa: E402
    load_relocator_calibration_config,
)

# Anchor -> (home paper key, canonical text). drf/crf/lrf share bbw_2019 —
# negatives therefore exclude by PAPER, not anchor.
GOLD_ANCHORS: dict[str, tuple[str, str]] = {
    "str": ("drr", "evaluation/canonical_texts/drr_2026.frozen.yaml"),
    "drf": ("bbw", "evaluation/canonical_texts/bbw_2019.frozen.yaml"),
    "mom6": ("jnps", "evaluation/canonical_texts/jnps_2013.frozen.yaml"),
    "crf": ("bbw", "evaluation/canonical_texts/bbw_2019.frozen.yaml"),
    "lrf": ("bbw", "evaluation/canonical_texts/bbw_2019.frozen.yaml"),
    "kpp": ("kpp", "evaluation/canonical_texts/kpp_2023.frozen.yaml"),
}

# The six negative-target papers.
NEGATIVE_TEXTS: dict[str, str] = {
    "bbw": "evaluation/canonical_texts/bbw_2019.frozen.yaml",
    "jnps": "evaluation/canonical_texts/jnps_2013.frozen.yaml",
    "drr": "evaluation/canonical_texts/drr_2026.frozen.yaml",
    "kpp": "evaluation/canonical_texts/kpp_2023.frozen.yaml",
    "bbw2021": "evaluation/canonical_texts/bbw_2021.frozen.yaml",
    "dfps": "evaluation/canonical_texts/dfps_2026.frozen.yaml",
}

LIGHT_CLASSES = ("truncate_head", "truncate_tail", "punctuation_drift", "typo_substitution")
# Corrected-universe truncation classes: a plain truncation of a located quote
# is still an exact substring (it falls out as exact_after_edit, contributing
# nothing to recall), so these truncate THEN apply the hash-seeded typo — the
# edit cannot exact-locate.
LIGHT_CLASSES_QR2 = ("truncate_head_typo", "truncate_tail_typo",
                     "punctuation_drift", "typo_substitution")
MODERATE_CLASSES = ("ellipsis_elision", "word_substitution")
NEAR_ACCEPT_FLOOR = 0.80

# The declared author-overlap matrix (re-declare on any change to the text
# set). The sole same-author pair is {bbw_2019, bbw_2021}.
SAME_AUTHOR_PAIRS: frozenset[frozenset[str]] = frozenset({frozenset({"bbw", "bbw2021"})})
PROTOCOLS = ("qr1", "qr2")

_PUNCT_DROP = set(",;:()")


@dataclass(frozen=True)
class GoldSpan:
    anchor: str
    paper_key: str
    field_path: str
    quote: str            # the original gold quote (verbatim from the gold spec)
    q_l1: str
    gold_page0: int       # 0-based page (gold locators are 1-based)
    gold_span: tuple[int, int]


def _hash_seed(paper_key: str, field_path: str, edit_class: str) -> int:
    return int(hashlib.sha256(f"{paper_key}|{field_path}|{edit_class}".encode()).hexdigest(), 16)


def _next_letter(c: str) -> str:
    if c == "z":
        return "a"
    if c == "Z":
        return "A"
    return chr(ord(c) + 1) if c.isalpha() else c


def apply_edit(q_l1: str, edit_class: str, seed: int) -> str | None:
    """One deterministic registered edit on the L1 quote; None when the class is
    structurally inapplicable (recorded as skipped_short_quote by the caller)."""
    words = q_l1.split(" ")
    n = len(words)
    if edit_class == "truncate_head":
        k = max(1, round(0.15 * n))
        return " ".join(words[k:]) if n - k >= 1 else None
    if edit_class == "truncate_tail":
        k = max(1, round(0.15 * n))
        return " ".join(words[:n - k]) if n - k >= 1 else None
    if edit_class == "punctuation_drift":
        out = "".join(c for c in q_l1 if c not in _PUNCT_DROP)
        while "  " in out:
            out = out.replace("  ", " ")
        return out.strip()
    if edit_class == "typo_substitution":
        alpha = [i for i, c in enumerate(q_l1) if c.isalpha()]
        if not alpha:
            return None
        chars = list(q_l1)
        for pos in {alpha[seed % len(alpha)], alpha[(seed // 7) % len(alpha)]}:
            chars[pos] = _next_letter(chars[pos])
        return "".join(chars)
    if edit_class == "ellipsis_elision":
        m = max(1, round(0.20 * n))
        if n - m - 2 < 1:
            return None
        start = 1 + (seed % (n - m - 2))
        return " ".join(words[:start] + ["..."] + words[start + m:])
    if edit_class == "word_substitution":
        eligible = [i for i in range(1, n - 1) if len(words[i]) >= 4]
        if not eligible:
            return None
        idx = eligible[seed % len(eligible)]
        return " ".join(words[:idx] + ["the"] + words[idx + 1:])
    if edit_class in ("truncate_head_typo", "truncate_tail_typo"):
        base = apply_edit(q_l1, edit_class.removesuffix("_typo"), seed)
        if base is None:
            return None
        return apply_edit(base, "typo_substitution", seed)
    raise ValueError(f"unknown edit class {edit_class!r}")


def _walk_stated_spans(obj, path: str = "") -> list[tuple[str, object]]:
    """Generic recursive walk collecting (dotted_path, Inherited-shaped node)
    for every STATED node carrying evidence — the kpp estimation-form walker
    (pre-declared best-effort)."""
    found: list[tuple[str, object]] = []
    seen: set[int] = set()

    def visit(node, p):
        if node is None or id(node) in seen:
            return
        if hasattr(node, "tag") and hasattr(node, "evidence") and hasattr(node, "value"):
            found.append((p, node))
            return
        if isinstance(node, (str, bytes, int, float, bool)):
            return
        seen.add(id(node))
        if isinstance(node, dict):
            for k, v in node.items():
                visit(v, f"{p}.{k}" if p else str(k))
            return
        if isinstance(node, (list, tuple)):
            for i, v in enumerate(node):
                visit(v, f"{p}[{i}]")
            return
        if hasattr(node, "__dataclass_fields__"):
            for name in node.__dataclass_fields__:
                visit(getattr(node, name), f"{p}.{name}" if p else name)

    visit(obj, path)
    return found


def _span_from_inherited(anchor: str, paper_key: str, ct: CanonicalText,
                         path: str, node) -> GoldSpan | None:
    if getattr(node, "tag", None) != "STATED":
        return None
    ev = getattr(node, "evidence", None)
    quote = getattr(ev, "quote", None)
    loc = getattr(ev, "locator", None)
    if not isinstance(quote, str) or quote.strip() == "" or loc is None:
        return None
    if loc.end_page is not None:
        raise RuntimeError(
            f"{anchor}:{path}: cross-page gold locator — the recovery predicate "
            "(_true_window) slices a single page; extend it before calibrating on a "
            "corpus with seam-straddling gold spans"
        )
    q_l1 = normalise(quote, "L1")
    page0 = loc.page - 1                     # gold locators are 1-based
    if not (0 <= page0 < len(ct.pages)):
        raise RuntimeError(
            f"{anchor}:{path}: gold locator page {loc.page} out of range for "
            f"{paper_key} ({len(ct.pages)} pages)"
        )
    return GoldSpan(anchor=anchor, paper_key=paper_key, field_path=path, quote=quote,
                    q_l1=q_l1, gold_page0=page0, gold_span=(loc.char_start, loc.char_end))


def build_positives(*, limit: int | None = None) -> tuple[list[GoldSpan], dict, bool]:
    """(deduped gold spans, per-anchor accounting, kpp_included). Clean quotes
    must exact-locate in their home text — they are gold; fail loud if not."""
    cts = {pk: load_canonical_text(_REPO_ROOT / path) for pk, path in NEGATIVE_TEXTS.items()}
    spans: list[GoldSpan] = []
    accounting: dict[str, dict] = {}
    kpp_included = True

    for anchor, (paper_key, _) in GOLD_ANCHORS.items():
        ct = cts[paper_key]
        try:
            spec = load_gold_spec(anchor)
            if anchor == "kpp":
                nodes = _walk_stated_spans(spec)
            else:
                nodes = [(p, n) for p, n in iter_gold_fields(spec) if n is not None]
            anchor_spans: list[GoldSpan] = []
            for path, node in sorted(nodes, key=lambda t: t[0]):
                span = _span_from_inherited(anchor, paper_key, ct, path, node)
                if span is None:
                    continue
                if ct.locate(span.quote, level="L1") is None:
                    raise RuntimeError(
                        f"gold quote does not exact-locate in its home text — "
                        f"{anchor}:{span.field_path} in {paper_key}; the gold set is broken"
                    )
                anchor_spans.append(span)
        except Exception as exc:                                   # noqa: BLE001
            if anchor == "kpp":
                # Pre-declared best-effort drop path — recorded, never
                # silent. Covers the generic walker surfacing an estimation-form
                # node whose quote does not L1-locate, not only a loader failure.
                kpp_included = False
                accounting[anchor] = {"spans": 0, "dropped": f"kpp walker failed: {exc}"}
                continue
            raise
        spans.extend(anchor_spans)
        accounting[anchor] = {"spans": len(anchor_spans)}

    # Dedupe identical L1 quotes within one paper (shared header quotes etc.).
    seen: set[tuple[str, str]] = set()
    deduped: list[GoldSpan] = []
    for s in spans:
        key = (s.paper_key, s.q_l1)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(s)
    if limit is not None:
        deduped = deduped[:limit]
    return deduped, accounting, kpp_included


def negative_targets(paper_key: str, protocol: str = "qr1") -> list[str]:
    """The negative-target papers for one span: the six texts minus its HOME
    PAPER (drf/crf/lrf all live in bbw_2019, so exclusion is by
    paper, never by anchor). Under ``qr2`` same-author targets
    are additionally excluded — a same-author restatement is not the
    fabrication hazard the false-accept test models."""
    if paper_key not in NEGATIVE_TEXTS:
        raise ValueError(f"unknown home paper {paper_key!r}")
    if protocol not in PROTOCOLS:
        raise ValueError(f"unknown protocol {protocol!r}")
    out = [t for t in NEGATIVE_TEXTS if t != paper_key]
    if protocol == "qr2":
        out = [t for t in out if frozenset({paper_key, t}) not in SAME_AUTHOR_PAIRS]
    return sorted(out)


def _true_window(ct: CanonicalText, span: GoldSpan) -> str:
    return normalise(ct.pages[span.gold_page0], "L1")[span.gold_span[0]:span.gold_span[1]]


def _location_ok(ct: CanonicalText, span: GoldSpan, cand) -> bool:
    """Recall predicate: page agreement + span overlap, or exact string
    equality of the relocated window with the true gold-page window (handles
    verbatim-duplicated sentences without crediting wrong-location accepts)."""
    if cand.cross_page:
        text_l1 = normalise(ct.pages[cand.page] + "\n" + ct.pages[cand.page + 1], "L1")
    else:
        text_l1 = normalise(ct.pages[cand.page], "L1")
    cand_text = text_l1[cand.l1_start:cand.l1_end]
    if cand_text == _true_window(ct, span):
        return True
    if not cand.cross_page and cand.page == span.gold_page0:
        gs, ge = span.gold_span
        return cand.l1_start < ge and gs < cand.l1_end
    return False


def choose_bar(roc: dict[float, dict], *, floor_light: float, floor_moderate: float
               ) -> tuple[float | None, str]:
    """The registered bar rule, mechanically. roc[bar] = {fa, r_light: [k, n], r_mod: [k, n]}."""
    def ratio(pair):
        k, n = pair
        return (k / n) if n else 0.0

    zero_fa = sorted(b for b, row in roc.items() if row["fa"] == 0)
    if not zero_fa:
        return None, "no_zero_fa_bar"
    feasible = [b for b in zero_fa
                if ratio(roc[b]["r_light"]) >= floor_light
                and ratio(roc[b]["r_mod"]) >= floor_moderate]
    if feasible:
        return max(feasible), "ok"
    return min(zero_fa), "recall_floor_unmet"


def run_calibration(*, limit: int | None = None, protocol: str = "qr1") -> dict:
    if protocol not in PROTOCOLS:
        raise ValueError(f"unknown protocol {protocol!r}")
    light_classes = LIGHT_CLASSES if protocol == "qr1" else LIGHT_CLASSES_QR2
    cfg = load_relocator_calibration_config()
    cts = {pk: load_canonical_text(_REPO_ROOT / path) for pk, path in NEGATIVE_TEXTS.items()}
    spans, accounting, kpp_included = build_positives(limit=limit)

    # --- positives: one edit per class per span, scored bar-free -------------
    tier_of = {**{c: "light" for c in light_classes}, **{c: "moderate" for c in MODERATE_CLASSES}}
    positives: list[dict] = []
    drop_counts: dict[str, int] = {}
    for span in spans:
        ct = cts[span.paper_key]
        for edit_class in light_classes + MODERATE_CLASSES:
            seed = _hash_seed(span.paper_key, span.field_path, edit_class)
            edited = apply_edit(span.q_l1, edit_class, seed)
            row = {"anchor": span.anchor, "paper": span.paper_key,
                   "field": span.field_path, "class": edit_class, "tier": tier_of[edit_class]}
            if edited is None:
                row.update({"status": "skipped_short_quote", "score": None, "located_ok": None})
            elif normalise(edited, "L1") == span.q_l1:
                row.update({"status": "noop", "score": None, "located_ok": None})
            elif len(normalise(edited, "L1")) < cfg.min_quote_chars:
                row.update({"status": "too_short", "score": None, "located_ok": None})
            elif ct.locate(edited, level="L1") is not None:
                row.update({"status": "exact_after_edit", "score": None, "located_ok": None})
            else:
                cand = score_quote(ct, edited, min_anchor_chars=cfg.min_anchor_chars)
                if cand is None:
                    row.update({"status": "ok", "score": 0.0, "located_ok": False})
                else:
                    row.update({"status": "ok", "score": round(cand.score, 6),
                                "located_ok": _location_ok(ct, span, cand)})
            if row["status"] != "ok":
                drop_counts[row["status"]] = drop_counts.get(row["status"], 0) + 1
            positives.append(row)

    # --- negatives: original quotes vs the non-home target papers ------------
    negatives: list[dict] = []
    excluded = {"same_author_pairs": 0, "exact_present_pairs": 0}
    for span in spans:
        if len(span.q_l1) < cfg.min_quote_chars:
            continue                          # guard-failing quotes can never accept
        all_targets = negative_targets(span.paper_key, "qr1")
        kept_targets = negative_targets(span.paper_key, protocol)
        excluded["same_author_pairs"] += len(all_targets) - len(kept_targets)
        for target in kept_targets:
            ct = cts[target]
            if protocol == "qr2" and ct.locate(span.quote, level="L1") is not None:
                # Exact-present exclusion: the production exact gate itself could
                # not reject this pair — recorded, never a false accept.
                excluded["exact_present_pairs"] += 1
                continue
            cand = score_quote(ct, span.quote, min_anchor_chars=cfg.min_anchor_chars)
            negatives.append({
                "anchor": span.anchor, "field": span.field_path, "home": span.paper_key,
                "target": target, "score": round(cand.score, 6) if cand else 0.0,
            })

    # --- sweep + rule --------------------------------------------------------
    usable = [p for p in positives if p["status"] == "ok"]
    roc: dict[float, dict] = {}
    for bar in cfg.bar_grid:
        fa = sum(1 for x in negatives if x["score"] >= bar)
        def recall(classes):
            rows = [p for p in usable if p["class"] in classes]
            k = sum(1 for p in rows if p["score"] >= bar and p["located_ok"])
            return [k, len(rows)]
        r_by_class = {}
        for c in light_classes + MODERATE_CLASSES:
            r_by_class[c] = recall((c,))
        roc[bar] = {"fa": fa, "r_light": recall(light_classes),
                    "r_mod": recall(MODERATE_CLASSES), "r_by_class": r_by_class}

    bar, status = choose_bar(roc, floor_light=cfg.recall_floor_light,
                             floor_moderate=cfg.recall_floor_moderate)
    near_accepts = sorted((x for x in negatives if x["score"] >= NEAR_ACCEPT_FLOOR),
                          key=lambda x: -x["score"])
    return {
        "diagnostic": f"Relocator calibration ({protocol}) — independent data only; "
                      "the target quotes never enter this artifact",
        "protocol": protocol,
        "negatives_excluded": excluded,
        "chosen_bar": bar,
        "status": status,
        "fa_ceiling": max((x["score"] for x in negatives), default=0.0),
        "n_spans": len(spans),
        "spans_per_anchor": accounting,
        "kpp_included": kpp_included,
        "n_positives": len(positives),
        "n_usable_positives": len(usable),
        "positive_drop_counts": dict(sorted(drop_counts.items())),
        "n_negatives": len(negatives),
        "near_accept_negatives": [
            {**x, "quote_head": next(s.q_l1[:80] for s in spans
                                     if s.field_path == x["field"] and s.anchor == x["anchor"])}
            for x in near_accepts
        ],
        "roc": {f"{b:.2f}": row for b, row in sorted(roc.items())},
        "limit": limit,
    }


def _render_md(result: dict) -> str:
    lines = [
        f"# Relocator calibration ({result['protocol']}) "
        "(diagnostic; the registered bar rule applied mechanically)",
        "",
        f"Protocol `{result['protocol']}`; negatives excluded: "
        f"{result['negatives_excluded']}",
        "",
        f"**Chosen bar: {result['chosen_bar']}** — status `{result['status']}` "
        f"(FA ceiling {result['fa_ceiling']:.3f}; {result['n_negatives']} negatives, "
        f"{result['n_usable_positives']}/{result['n_positives']} usable positives over "
        f"{result['n_spans']} gold spans; kpp_included={result['kpp_included']})",
        "",
        "| bar | FA | R_light | R_mod |",
        "|---|---|---|---|",
    ]
    for b, row in result["roc"].items():
        rl, rm = row["r_light"], row["r_mod"]
        lines.append(f"| {b} | {row['fa']} | {rl[0]}/{rl[1]} | {rm[0]}/{rm[1]} |")
    lines += ["", f"Near-accept negatives (score >= {NEAR_ACCEPT_FLOOR}): "
                  f"{len(result['near_accept_negatives'])} (listed verbatim in the JSON).", ""]
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Relocator bar calibration (diagnostic).")
    ap.add_argument("--json", required=True, dest="json_out")
    ap.add_argument("--limit", type=int, default=None,
                    help="smoke flag: cap the number of gold spans")
    ap.add_argument("--protocol", choices=PROTOCOLS, default="qr1",
                    help="qr1 = the strict all-pairs negative universe; qr2 = the corrected universe")
    args = ap.parse_args(argv)

    result = run_calibration(limit=args.limit, protocol=args.protocol)

    out = Path(args.json_out)
    if not out.is_absolute():
        out = _REPO_ROOT / out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    out.with_suffix(".md").write_text(_render_md(result), encoding="utf-8")
    print(f"[calibrate] json -> {out}", file=sys.stderr)
    print(json.dumps({"chosen_bar": result["chosen_bar"], "status": result["status"],
                      "fa_ceiling": result["fa_ceiling"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
