"""
T5 class-7 "deletion" enabler -- a DETERMINISTIC, no-model, no-pipeline scan.

Class-7 of the T5 perturbation family DELETES a field's grounding sentence from a
paper and asks whether the field is still recoverable. A deletion is only a VALID
test target if the paper states that field's value exactly ONCE: if the value is
restated elsewhere, deleting the gold sentence does not remove the information, so
"the model should now answer UNKNOWN" is the wrong expectation. Papers restate
their headline construction constantly ("value-weighted", "each month", "decile"
appear on many pages), so most fields are NOT valid deletion targets. This tool
surfaces, per anchor per STATED field, the value-witnesses that SURVIVE deleting
the gold sentence, so `deletion_valid` can be adjudicated by eye.

WHAT IT DOES (pure in-memory text analysis; zero LLM, zero pipeline, zero KAT):
  1. Load each sort anchor's committed gold (`gold_loader.load_gold_spec`) and its
     FROZEN canonical text (`canonical_text.load_canonical_text`).
  2. Walk every gold field (`field_pairing.iter_gold_fields`); keep the STATED ones
     that carry a verbatim quote + locator.
  3. For each STATED field:
       (a) count exact-L1 copies of the gold quote across all pages;
       (b) delete ALL those copies from the L1-normalised text;
       (c) search the remainder (case-insensitive) for the value's surface forms,
           drawn from the OVER-INCLUSIVE VALUE_WITNESS map below.
     Zero surviving witnesses  -> provisional `likely_deletion_valid`.
     One or more survivors      -> `needs_adjudication`.
     Prose / rubric fields (method_summary, universe_filter) carry authored
     synthesis, not a once-stated scalar, so they get NO witness scan and a
     distinct `rubric_prose` disposition.

  It emits `evaluation/adversarial/deletion_candidates.md` and prints the global
  summary + the deletion-valid tail. Determinism: same inputs -> byte-identical
  report. No randomness. No dates are baked into the output.

Run:  PYTHONPATH=<repo-root> ./.venv/bin/python evaluation/adversarial/deletion_scan.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from agents.librarian.config.canonical_text import load_canonical_text  # noqa: E402
from agents.librarian.config.normalise import normalise  # noqa: E402
from evaluation.gold_specs.gold_loader import load_gold_spec  # noqa: E402
from evaluation.harness.field_pairing import RUBRIC_FIELDS, iter_gold_fields  # noqa: E402

# ---------------------------------------------------------------------------
# VALUE_WITNESS -- OVER-INCLUSIVE by design; the report is an input for HUMAN
# ADJUDICATION of `deletion_valid`, NOT an automatic verdict.
#
# A "value-witness" is a paper SURFACE FORM that counts as the paper re-stating a
# field's value. The map is deliberately broad: a spurious survivor (a false
# `needs_adjudication`) is a safe error; a MISSED survivor (a false
# `likely_deletion_valid`, i.e. flagging a field as safe to delete when
# the value is actually restated) is not. So each rule leans toward MORE surface
# forms, not fewer. All forms are authored in L1-normalised convention (ASCII
# hyphen, straight quotes, single spaces) and matched case-insensitively.
#
# Keyed by the field's leaf name, then by the field's STATED value (as a string).
# Concept-id fields (sort_signal / control_axis) are mapped by concept id in
# SIGNAL_CONCEPT_WITNESS. Dates and the headline-metric dict are DERIVED from the
# value (see _date_witnesses / _metric_witnesses) rather than enumerated. Any
# field/value with no rule falls back to alpha tokens of the value string
# (_fallback) and is flagged `source=fallback` in the report
# the witnesses were auto-derived, not curated.
# ---------------------------------------------------------------------------

VALUE_WITNESS: dict[str, dict[str, list[str]]] = {
    # Part 1
    "formation_structure": {
        "sorted_portfolios": ["sort", "sorted", "sorting", "portfolio"],
    },
    "asset_class": {
        "corporate_bonds": ["corporate bond"],
    },
    # Leg fields
    "sort_kind": {
        "independent": ["independent", "independently sort"],
        "single": ["single-sort", "single sort", "univariate", "single decile"],
    },
    "n_groups": {
        "5": ["quintile", "five ", "5x5", "5 x 5"],
        "10": ["decile", "ten ", "p1 to p10", "p10", "p1"],
    },
    "control_n_groups": {
        "5": ["quintile", "five ", "5x5", "5 x 5"],
    },
    "long_leg": {
        "highest_signal": [
            "highest", "winner", "top decile", "top portfolio", "top quintile",
            "long the top", "long the winner", "long the highest", "long p10",
            "going long p10", "p10",
        ],
    },
    # Common block
    "weighting_scheme": {
        "value": [
            "value-weight", "value weighted", "value-weighted",
            "market capitalization", "market value",
        ],
        "equal": [
            "equal-weight", "equally weight", "equally-weighted",
            "equal weighted", "equally weighted",
        ],
    },
    "weighting_base": {
        "par": ["amount outstanding", "par value", "face value", "offering amount"],
        "market_value": ["market capitalization", "market value"],
    },
    "rebalance_frequency": {
        "monthly": ["each month", "every month", "monthly", "month t"],
    },
    "strategy_side": {
        "long_short": ["long-short", "long short", "long the", "short the"],
    },
    "signal_lag": {
        "1": ["skip one month", "skip a month", "one month between", "one-month skip"],
        "0": ["month-end price", "signals observed at", "no skip", "zero lag"],
    },
    "holding_period": {
        "6": [
            "holding period", "six-month", "six month", "6-month",
            "t +6", "t+6", "held over months",
        ],
    },
    "return_label": {
        "realisation": [
            "realisation", "realization", "month-t return", "month t return",
            "realized month",
        ],
    },
    "overlap_convention": {
        "overlapping": [
            "overlapping", "staggered", "prior month", "six months earlier",
            "overlap",
        ],
    },
    "cohort_weighting": {
        "equal": ["equally weighted average", "equally weighted", "equal weight"],
    },
    "significance_convention": {
        "hac_t_of_mean": ["newey-west", "newey west", "t-statistic", "t-stat", "hac"],
    },
    "hac_lags": {
        "floor(T^0.25)": ["lags =", "lags", "0.25", "t 0.25"],
    },
    "rf_convention": {
        "subtract_rf": [
            "excess return", "risk-free rate", "risk free", "t-bill",
            "treasury bill", "one-month t-bill",
        ],
        "none": ["excess return", "risk-free rate", "raw return", "difference"],
    },
    "benchmark_model": {
        "other": [
            "factor model", "alpha", "abnormal return", "risk-adjusted",
            "regression",
        ],
        "capm": [
            "capm", "single-factor", "market factor", "mktb",
            "capital asset pricing", "alpha",
        ],
    },
    "expost_trim": {
        "truncate": [
            "eliminated return", "eliminate return", "99.5th percentile",
            "30% per month", "outlier", "truncat", "winsoriz",
        ],
    },
    "return_availability_policy": {
        "require_next_month_return": [
            "valid return requires", "excluded for that month", "both months t",
            "trade in either window",
        ],
    },
    # Structural / header
    "combiner": {
        "single_leg": ["single-sort", "single sort", "single", "one leg"],
    },
    "strategy_label": {
        "DRF": ["drf", "downside risk factor"],
        "mom6": ["mom6", "momentum"],
        "str": ["str", "short-term reversal", "reversal"],
    },
}

# Concept ids -> the paper's own surface words for the signal (over-inclusive).
SIGNAL_CONCEPT_WITNESS: dict[str, list[str]] = {
    "var_5pct": [
        "downside risk", "5% var", "value-at-risk", "second lowest monthly return",
    ],
    "credit_rating": ["credit rating", "credit risk", "rating"],
    "past_6m_cumulative_return": [
        "cumulative return", "momentum", "t -6", "formation period", "past six",
    ],
    "prior_1m_excess_return": [
        "short-term reversal", "reversal", "str", "month-end price", "prior return",
    ],
}

_MONTHS = [
    "", "january", "february", "march", "april", "may", "june", "july",
    "august", "september", "october", "november", "december",
]

# Anchors, in fixed order: (anchor_id, canonical-text file, human label).
ANCHORS: list[tuple[str, str, str]] = [
    ("drf", "bbw_2019.frozen.yaml", "BBW 2019 -- downside risk factor (bivariate 5x5 sort)"),
    ("mom6", "jnps_2013.frozen.yaml", "JNPS 2013 -- six-month momentum (decile sort)"),
    ("str", "drr_2026.frozen.yaml", "DRR 2026 -- short-term reversal (decile single-sort)"),
]

_CANON_DIR = _REPO_ROOT / "evaluation" / "canonical_texts"
_OUT_PATH = _REPO_ROOT / "evaluation" / "adversarial" / "deletion_candidates.md"

CTX = 80            # +/- chars of context around a witness span
MAX_SPANS = 25      # spans displayed per field (true total always reported)


# ---------------------------------------------------------------------------
# Witness derivation.
# ---------------------------------------------------------------------------

def _fallback(value: object) -> list[str]:
    """No curated rule: derive witnesses from the value's alpha tokens (len>=3),
    lowercased, order-preserving unique. e.g. 'long_short' -> ['long', 'short']."""
    seen: list[str] = []
    for tok in re.findall(r"[A-Za-z]{3,}", str(value)):
        t = tok.lower()
        if t not in seen:
            seen.append(t)
    return seen


def _date_witnesses(value: object) -> list[str]:
    """A 'YYYY-MM' sample bound -> [month-name year, year]. e.g. '2004-07' ->
    ['july 2004', '2004']. Bare year is intentionally broad (matches any mention
    of the year)."""
    m = re.match(r"(\d{4})-(\d{2})", str(value))
    if not m:
        return _fallback(value)
    year, mm = m.group(1), int(m.group(2))
    name = _MONTHS[mm] if 1 <= mm <= 12 else ""
    out = [year]
    if name:
        out.insert(0, f"{name} {year}")
    return out


def _metric_witnesses(value: object) -> list[str]:
    """Headline-metric dict {mean, t_stat, unit} -> the printed numbers. Uses
    2-decimal magnitude strings (L1 unifies the minus sign, so 'X.YZ' matches a
    signed occurrence) plus basis-point forms of the mean."""
    if not isinstance(value, dict):
        return _fallback(value)
    out: list[str] = []
    mean, ts = value.get("mean"), value.get("t_stat")
    if isinstance(mean, (int, float)):
        out.append(f"{abs(mean):.2f}")
        bps = round(abs(mean) * 100)
        out += [f"{bps} basis points", f"{bps} bps"]
    if isinstance(ts, (int, float)):
        out.append(f"{abs(ts):.2f}")
    return [w.lower() for w in out]


def _leaf_of(path: str) -> tuple[str, bool]:
    """Return (leaf_name, is_concept) for a gold dotted path."""
    if path == "part1.method_summary.summary":
        return "method_summary", False
    if path == "part2.combiner.kind":
        return "combiner", False
    if path.endswith(".concept_id"):
        return path.split(".")[-2], True   # 'sort_signal' | 'control_axis'
    return path.split(".")[-1], False


def resolve_witnesses(leaf: str, value: object, is_concept: bool) -> tuple[list[str], str]:
    """(witnesses_lowercased, source). source in
    {concept, date, metric, map, fallback}."""
    if is_concept:
        ws = SIGNAL_CONCEPT_WITNESS.get(str(value))
        if ws:
            return [w.lower() for w in ws], "concept"
        return _fallback(value), "fallback"
    if leaf in ("sample_start", "sample_end"):
        return [w.lower() for w in _date_witnesses(value)], "date"
    if leaf == "claimed_headline_metric":
        return _metric_witnesses(value), "metric"
    table = VALUE_WITNESS.get(leaf)
    if table is not None:
        ws = table.get(str(value))
        if ws is not None:
            return [w.lower() for w in ws], "map"
    return _fallback(value), "fallback"


# ---------------------------------------------------------------------------
# Core scan.
# ---------------------------------------------------------------------------

class FieldResult:
    __slots__ = (
        "path", "leaf", "short_key", "value", "quote", "page", "copies",
        "witnesses", "source", "spans", "disposition",
    )

    def __init__(self, **kw):
        for k in self.__slots__:
            setattr(self, k, kw.get(k))


def _short_key(path: str, leaf: str, is_concept: bool) -> str:
    if is_concept:
        return f"{leaf}.concept_id"
    return leaf


def scan_field(path, inh, pages_l1) -> FieldResult | None:
    """Scan one STATED gold field. Returns None for non-STATED / no-evidence."""
    if inh is None or getattr(inh, "tag", None) != "STATED":
        return None
    ev = inh.evidence
    quote = ev.quote
    if not quote or ev.locator is None:
        return None
    leaf, is_concept = _leaf_of(path)
    value = inh.value

    q_l1 = normalise(quote, "L1")
    copies = sum(p.count(q_l1) for p in pages_l1) if q_l1 else 0
    remainder = [p.replace(q_l1, "") for p in pages_l1] if q_l1 else list(pages_l1)

    rubric = leaf in RUBRIC_FIELDS
    witnesses: list[str] = []
    source = "n/a (rubric/prose)"
    spans: list[tuple[int, int, int, str]] = []

    if not rubric:
        witnesses, source = resolve_witnesses(leaf, value, is_concept)
        # find every occurrence of every witness in every page's remainder
        by_pos: dict[tuple[int, int], tuple[int, str]] = {}
        for i, rem in enumerate(remainder):
            rem_l = rem.lower()
            for w in witnesses:
                if not w:
                    continue
                pos = 0
                while True:
                    j = rem_l.find(w, pos)
                    if j < 0:
                        break
                    key = (i, j)
                    if key not in by_pos:      # dedup overlapping witnesses at one start
                        by_pos[key] = (j + len(w), w)
                    pos = j + 1
        for (i, j), (end, w) in by_pos.items():
            spans.append((i, j, end, w))
        spans.sort(key=lambda s: (s[0], s[1]))

    if rubric:
        disposition = "rubric_prose"
    elif not spans:
        disposition = "likely_deletion_valid"
    else:
        disposition = "needs_adjudication"

    return FieldResult(
        path=path, leaf=leaf, short_key=_short_key(path, leaf, is_concept),
        value=value, quote=quote, page=ev.locator.page, copies=copies,
        witnesses=witnesses, source=source, spans=spans, disposition=disposition,
    )


def _context(remainder: list[str], span: tuple[int, int, int, str]) -> str:
    i, start, end, w = span
    rem = remainder[i]
    lo = rem.lower()
    src = rem if len(lo) == len(rem) else lo   # slice original case when length-safe
    s = max(0, start - CTX)
    e = min(len(rem), end + CTX)
    ctx = src[s:e].strip()
    pre = "..." if s > 0 else ""
    suf = "..." if e < len(rem) else ""
    return f"p{i + 1}: {pre}{ctx}{suf}"


def _fmt_value(value: object) -> str:
    s = str(value)
    if len(s) > 160:
        s = s[:157] + "..."
    return s


def _fmt_quote(quote: str) -> str:
    q = quote.replace("\n", " ").strip()
    if len(q) > 240:
        q = q[:237] + "..."
    return q


# ---------------------------------------------------------------------------
# Report.
# ---------------------------------------------------------------------------

def build_report() -> tuple[str, list[dict], list[dict]]:
    """Return (markdown, per_anchor_summaries, tail_candidates)."""
    lines: list[str] = []
    summaries: list[dict] = []
    tail: list[dict] = []

    lines.append("# T5 class-7 deletion-target scan -- surviving value-witnesses")
    lines.append("")
    lines.append(
        "_Deterministic, no-model, no-pipeline. Inputs: the committed anchor golds "
        "(`evaluation/gold_specs/`) + the FROZEN canonical texts "
        "(`evaluation/canonical_texts/`). Method: per STATED field, count exact-L1 "
        "copies of the gold quote, DELETE all of them from the L1-normalised text, "
        "then search the remainder (case-insensitive) for OVER-INCLUSIVE value-witness "
        "surface forms (see the `VALUE_WITNESS` block in `deletion_scan.py`). Zero "
        "survivors -> provisional `likely_deletion_valid` (the value appears to be "
        "stated only in the deleted sentence); any survivor -> `needs_adjudication`. "
        "Prose/rubric fields carry authored synthesis, not a once-stated scalar, so "
        "they get no witness scan (`rubric_prose`). This is an input for human "
        "adjudication of `deletion_valid`, NOT a verdict._"
    )
    lines.append("")

    # Per-anchor bodies (accumulated, appended after the global summary table).
    bodies: list[str] = []

    for anchor_id, canon_file, label in ANCHORS:
        spec = load_gold_spec(anchor_id)
        ct = load_canonical_text(str(_CANON_DIR / canon_file))
        pages_l1 = [normalise(p, "L1") for p in ct.pages]

        results: list[FieldResult] = []
        for path, gold in iter_gold_fields(spec):
            r = scan_field(path, gold, pages_l1)
            if r is not None:
                results.append(r)

        n_valid = sum(r.disposition == "likely_deletion_valid" for r in results)
        n_needs = sum(r.disposition == "needs_adjudication" for r in results)
        n_rubric = sum(r.disposition == "rubric_prose" for r in results)
        summaries.append({
            "anchor": anchor_id, "label": label, "stated": len(results),
            "likely_valid": n_valid, "needs_adjudication": n_needs, "rubric": n_rubric,
        })

        body: list[str] = []
        body.append(f"## Anchor: {anchor_id} ({label})")
        body.append("")
        body.append(
            f"STATED fields scanned: **{len(results)}** | "
            f"likely_deletion_valid: **{n_valid}** | "
            f"needs_adjudication: **{n_needs}** | rubric_prose: **{n_rubric}**"
        )
        body.append("")

        # recompute per-anchor remainder for context slicing (cheap; deterministic)
        for r in results:
            q_l1 = normalise(r.quote, "L1")
            remainder = [p.replace(q_l1, "") for p in pages_l1] if q_l1 else list(pages_l1)

            body.append(f"### `{r.short_key}` -- **{r.disposition}**")
            body.append(f"- path: `{r.path}`")
            body.append(f"- gold value: `{_fmt_value(r.value)}`")
            body.append(f'- gold quote (p{r.page}): "{_fmt_quote(r.quote)}"')
            body.append(f"- exact-L1 copies of gold quote: {r.copies}")
            if r.disposition == "rubric_prose":
                body.append(
                    "- witnesses: n/a -- prose/rubric field (authored synthesis, "
                    "not a once-stated scalar); no witness scan performed."
                )
                body.append("")
                continue
            body.append(
                f"- witnesses searched (source={r.source}): "
                + (", ".join(f"`{w}`" for w in r.witnesses) if r.witnesses else "(none derived)")
            )
            body.append(f"- surviving witness spans: {len(r.spans)} total")
            if not r.spans:
                body.append("    - (none surviving -> value appears once-stated)")
            else:
                for span in r.spans[:MAX_SPANS]:
                    body.append(f"    - {_context(remainder, span)}")
                if len(r.spans) > MAX_SPANS:
                    body.append(f"    - (+{len(r.spans) - MAX_SPANS} more spans not shown)")
            body.append("")

            if r.disposition == "likely_deletion_valid":
                tail.append({
                    "anchor": anchor_id, "field": r.short_key, "path": r.path,
                    "value": _fmt_value(r.value), "copies": r.copies,
                    "witnesses": r.witnesses, "source": r.source,
                })

        bodies.append("\n".join(body))

    # Global summary table.
    lines.append("## Global summary")
    lines.append("")
    lines.append("| anchor | #STATED | likely_deletion_valid | needs_adjudication | rubric_prose |")
    lines.append("|---|---|---|---|---|")
    for s in summaries:
        lines.append(
            f"| {s['anchor']} | {s['stated']} | {s['likely_valid']} | "
            f"{s['needs_adjudication']} | {s['rubric']} |"
        )
    tot = {k: sum(s[k] for s in summaries)
           for k in ("stated", "likely_valid", "needs_adjudication", "rubric")}
    lines.append(
        f"| **total** | **{tot['stated']}** | **{tot['likely_valid']}** | "
        f"**{tot['needs_adjudication']}** | **{tot['rubric']}** |"
    )
    lines.append("")

    # Deletion-valid tail.
    lines.append("## Deletion-valid tail (provisional `likely_deletion_valid`)")
    lines.append("")
    lines.append(
        "_The value survives nowhere else under the over-inclusive witness map -- the "
        "strongest candidates for a genuine class-7 deletion target, to be confirmed "
        "each by eye before use._"
    )
    lines.append("")
    if not tail:
        lines.append("(none)")
    else:
        for t in tail:
            lines.append(
                f"- **{t['anchor']}** `{t['field']}` = `{t['value']}` "
                f"(quote copies={t['copies']}, witnesses[{t['source']}]="
                + ", ".join(f"`{w}`" for w in t["witnesses"]) + ")"
            )
    lines.append("")

    return "\n".join(lines) + "\n\n" + "\n\n".join(bodies) + "\n", summaries, tail


def main() -> None:
    report, summaries, tail = build_report()
    _OUT_PATH.write_text(report, encoding="utf-8")

    print(f"wrote {_OUT_PATH}")
    print("\nGLOBAL SUMMARY (anchor: #STATED / likely_valid / needs_adjudication / rubric_prose)")
    for s in summaries:
        print(
            f"  {s['anchor']:>5}: {s['stated']:>2} / {s['likely_valid']:>2} / "
            f"{s['needs_adjudication']:>2} / {s['rubric']:>2}"
        )
    print("\nDELETION-VALID TAIL:")
    if not tail:
        print("  (none)")
    for t in tail:
        print(f"  {t['anchor']:>5}  {t['field']} = {t['value']}  [copies={t['copies']}, src={t['source']}]")


if __name__ == "__main__":
    main()
