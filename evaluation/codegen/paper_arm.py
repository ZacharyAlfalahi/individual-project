"""P1 paper arm — paper → executable strategy, with no specification in between.

The P1 field-key arm (``runner.build_prompt``) gives each model the hand-authored reference
specification. This arm changes one variable: the prompt carries the paper itself, the frozen
canonical text the Librarian reads, plus the strategy's name as the paper labels it. The models,
panel, oracles, grader, sandbox, cache and budget are the field-key arm's, reused unchanged through
``ablation.run_scored_ablation(prompt_fn=build_paper_prompt)``. This is the compound task a
practitioner would hand a general-purpose model.

Contract: the P1 paper-arm mini-contract (not shipped with the repository). Its freeze block must
carry the sha256 of every asset below before ``--execute`` (``paper_arm_freeze_ok``). Unlike the
field-key arm, the paper's own reported results are NOT withheld: they are part of the paper, and
generation is single-shot with no execution feedback, so a model cannot tune toward them.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

from agents.librarian.config.canonical_text import load_canonical_text
from evaluation.codegen.runner import STRATEGIES
from shared.licensed_inputs import require_licensed_input

_REPO_ROOT = Path(__file__).resolve().parents[2]
_HERE = Path(__file__).resolve().parent
_TEMPLATE = _HERE / "prompts_paper" / "template_paper.md"
# The field-key arm's frozen schema and output-contract assets, reused byte-for-byte.
_SHARED_ASSETS = (_HERE / "prompts" / "panel_schema.md", _HERE / "prompts" / "output_contract.md")
CONTRACT = _REPO_ROOT / "docs" / "extensions" / "contracts" / "p1_paper_arm.md"

ARM = "paper"

# The four oracle strategies that also carry a byte-equality result: the 8-cell
# byte-equality denominator (lrf has an oracle but no byte-equality result).
BYTE_EQUALITY_STRATEGIES: tuple[str, ...] = ("drf", "str", "mom6", "crf")


@dataclass(frozen=True)
class PaperTarget:
    """Which paper a strategy comes from, and the name the prompt gives it."""

    paper_id: str
    canonical_text: Path
    label: str


_TEXTS = _REPO_ROOT / "evaluation" / "canonical_texts"

# Labels are the hand-authored construction names of evaluation/gold_specs/enum_*.yaml, written as
# the papers write them (project codes dropped). drf, crf and lrf share one paper, so the label is
# the only thing that tells those three prompts apart.
PAPER_TARGETS: dict[str, PaperTarget] = {
    "drf": PaperTarget("BBW_2019", _TEXTS / "bbw_2019.frozen.yaml", "Downside Risk Factor (DRF)"),
    "str": PaperTarget("DRR_2026", _TEXTS / "drr_2026.frozen.yaml", "short-term reversal factor"),
    "mom6": PaperTarget("JNPS_2013", _TEXTS / "jnps_2013.frozen.yaml",
                        "six-month momentum strategy"),
    "crf": PaperTarget("BBW_2019", _TEXTS / "bbw_2019.frozen.yaml", "Credit Risk Factor (CRF)"),
    "lrf": PaperTarget("BBW_2019", _TEXTS / "bbw_2019.frozen.yaml",
                       "Liquidity Risk Factor (LRF)"),
}
if set(PAPER_TARGETS) != set(STRATEGIES):
    raise ImportError("every P1 oracle strategy needs a paper target (and no other)")


def paper_text(strategy: str) -> str:
    """The frozen canonical text, pages joined exactly as the Librarian joins them."""
    ct = load_canonical_text(PAPER_TARGETS[strategy].canonical_text)
    ct.require_frozen()
    return "\n\n".join(ct.pages)


def build_paper_prompt(strategy: str) -> str:
    target = PAPER_TARGETS[strategy]
    schema, contract = (p.read_text(encoding="utf-8") for p in _SHARED_ASSETS)
    return (
        _TEMPLATE.read_text(encoding="utf-8")
        .replace("{{TARGET_LABEL}}", target.label)
        .replace("{{PAPER_TEXT}}", paper_text(strategy))
        .replace("{{PANEL_SCHEMA}}", schema)
        .replace("{{OUTPUT_CONTRACT}}", contract)
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def paper_arm_asset_hashes() -> dict[str, str]:
    """sha256 of every input the paper-arm prompt is built from, keyed by repo-relative path."""
    paths = [_TEMPLATE, *_SHARED_ASSETS,
             *sorted({require_licensed_input(t.canonical_text, "frozen canonical text")
                      for t in PAPER_TARGETS.values()})]
    return {str(p.relative_to(_REPO_ROOT)): _sha256(p) for p in paths}


def paper_arm_freeze_ok(contract: Path = CONTRACT) -> tuple[bool, str]:
    """``--execute`` requires the contract to exist and to record every asset hash, so the prompt
    cannot change after the freeze without the change showing."""
    if not contract.exists():
        return False, f"paper-arm contract missing: {contract}"
    text = contract.read_text(encoding="utf-8")
    for name, digest in paper_arm_asset_hashes().items():
        if digest not in text:
            return False, (f"asset {name} (sha256 {digest[:16]}…) is not recorded in the "
                           "paper-arm contract freeze block — re-freeze before any generation")
    return True, "paper-arm contract freeze verified"


def sku_verification(runs: list[dict], phase: str) -> tuple[bool | None, str]:
    """Three-state SKU check, mirroring ``scripts/run_p2_codegen.py``. A replayed response carries no
    returned model version, so "no mismatch seen" is not a match: it is ``None`` (unverified)."""
    if phase != "reported":
        return None, "not applicable — the dev pair is never reportable"
    verified = [r for r in runs if r.get("returned_model_version") is not None]
    mismatches = [(r["model_id"], r["returned_model_version"]) for r in verified
                  if r["model_id"] not in str(r["returned_model_version"])]
    if mismatches:
        return False, f"returned SKU did not match the pinned model: {mismatches}"
    if not verified:
        return None, "UNVERIFIED — every response replayed from the cache, which stores no SKU"
    if len(verified) < len(runs):
        return None, (f"PARTIALLY VERIFIED — {len(verified)}/{len(runs)} runs carried a returned "
                      "model version")
    return True, f"every returned SKU matched the pinned model ({len(verified)}/{len(runs)} runs)"


def reportability(record: dict, *, live_record: dict | None = None) -> dict:
    """Named blockers; ``reportable`` is True only when there are none.

    A replay (e.g. a basis served from the shared cache) cannot evidence SKUs itself. It inherits verification only from
    a live record whose prompts are byte-identical (same prompt sha256 per strategy) and whose own
    SKU check passed, and it names that record."""
    sku_match, sku_reason = sku_verification(record.get("runs", []), record.get("phase", ""))
    if (sku_match is None and live_record is not None and record.get("phase") == "reported"
            and live_record.get("prompt_sha256") == record.get("prompt_sha256")
            and live_record.get("sku_match") is True):
        sku_match, sku_reason = True, (
            "verified by the live record with byte-identical prompts, which served these responses "
            f"({live_record.get('sku_reason', 'SKUs matched')})")
    blockers: list[str] = []
    if record.get("phase") != "reported":
        blockers.append("dev phase — the free pair is never reportable")
    if record.get("generation_errors"):
        blockers.append("at least one generation failed on infrastructure")
    if sku_match is False:
        blockers.append(sku_reason)
    elif sku_match is None:
        blockers.append(f"returned model SKUs unverified ({sku_reason})")
    return {"sku_match": sku_match, "sku_reason": sku_reason,
            "reportable_blockers": blockers, "reportable": not blockers}


_ROW_KEYS = ("verdict", "reason", "failed_criteria", "n_overlap", "correlation", "sign_agreement",
             "mean_diff", "tracking_error", "max_abs_diff")


def _rows(record: dict) -> dict[tuple[str, str], dict]:
    return {(r["strategy"], r["model_id"]): {k: r.get(k) for k in _ROW_KEYS}
            for r in record.get("raw_metrics_table", [])}


def _verdict_counts(rows: list[dict | None]) -> dict[str, int]:
    counts = {"RUNS_RIGHT": 0, "RUNS_WRONG": 0, "WONT_RUN": 0, "NO_DATUM": 0}
    for row in rows:
        counts[row["verdict"] if row else "NO_DATUM"] += 1
    return counts


def compare_arms(field_key: dict, paper: dict) -> dict:
    """Pair the two arms' scored cells, one row per (strategy, model), with a verdict transition.

    Counts are given over all ten cells and over the eight byte-equality cells.
    A cell with no scored row (a generation error) is ``NO_DATUM``, never a verdict."""
    fk, pa = _rows(field_key), _rows(paper)
    models = list(dict.fromkeys([*field_key.get("models", []), *paper.get("models", [])]))
    cells = []
    for strategy in STRATEGIES:
        for model in models:
            a, b = fk.get((strategy, model)), pa.get((strategy, model))
            cells.append({
                "strategy": strategy, "model_id": model,
                "byte_equality_cell": strategy in BYTE_EQUALITY_STRATEGIES,
                "field_key": a, "paper": b,
                "transition": f"{a['verdict'] if a else 'NO_DATUM'}→{b['verdict'] if b else 'NO_DATUM'}",
            })

    def _counts(subset):
        return {"field_key": _verdict_counts([c["field_key"] for c in subset]),
                "paper": _verdict_counts([c["paper"] for c in subset])}

    eight = [c for c in cells if c["byte_equality_cell"]]
    transitions: dict[str, int] = {}
    for c in cells:
        transitions[c["transition"]] = transitions.get(c["transition"], 0) + 1
    return {
        "arms": {"field_key": "hand-authored reference specification (P1)",
                 "paper": "frozen canonical paper text + the strategy's label (paper arm)"},
        "models": models,
        "cells": cells,
        "counts_all_10": _counts(cells),
        "counts_byte_equality_8": _counts(eight),
        "transitions": dict(sorted(transitions.items())),
        "prompt_sha256": {"field_key": field_key.get("prompt_sha256"),
                          "paper": paper.get("prompt_sha256")},
        "panel_sha256": {"field_key": field_key.get("panel_sha256"),
                         "paper": paper.get("panel_sha256")},
    }


def _fmt(value, spec: str, scale: float = 1.0) -> str:
    return "NaN" if value is None else format(value * scale, spec)


def _cell_text(row: dict | None) -> str:
    if row is None:
        return "no datum"
    if row["verdict"] == "WONT_RUN":
        return f"WONT_RUN ({row.get('reason') or 'n/a'})"
    return f"{row['verdict']} (ρ {_fmt(row.get('correlation'), '.3f')})"


def render_comparison_md(comparison: dict, *, basis: str) -> str:
    """Human-readable view of ``compare_arms``; every number comes from the comparison dict."""
    lines = [
        f"# P1 paper arm vs field-key arm — {basis}",
        "",
        "Same models, panel, oracles and grader; only the prompt differs "
        "(field-key specification vs the paper text). Generated by "
        "`scripts/build_paper_arm_report.py`.",
        "",
        f"Panels byte-identical: {comparison['panel_sha256']['field_key'] == comparison['panel_sha256']['paper']}",
        "",
        "| Strategy | Model | Field-key arm | Paper arm | Paper overlap (months) | "
        "Paper mean diff (bp/mo) | Paper tracking error (bp/mo) | Paper failed criteria |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for c in comparison["cells"]:
        p = c["paper"] or {}
        lines.append(
            f"| {c['strategy']} | {c['model_id']} | {_cell_text(c['field_key'])} | "
            f"{_cell_text(c['paper'])} | {p.get('n_overlap', '—') if p.get('n_overlap') is not None else '—'} | "
            f"{_fmt(p.get('mean_diff'), '+.1f', 1e4) if p else '—'} | "
            f"{_fmt(p.get('tracking_error'), '.1f', 1e4) if p else '—'} | "
            f"{', '.join(p.get('failed_criteria') or []) or '—'} |")
    for title, key in (("All ten cells", "counts_all_10"),
                       ("Eight byte-equality cells", "counts_byte_equality_8")):
        counts = comparison[key]
        lines += ["", f"## {title}", "", "| Verdict | Field-key arm | Paper arm |", "|---|---|---|"]
        for verdict in ("RUNS_RIGHT", "RUNS_WRONG", "WONT_RUN", "NO_DATUM"):
            lines.append(f"| {verdict} | {counts['field_key'][verdict]} | {counts['paper'][verdict]} |")
    lines += ["", "## Verdict transitions (field-key → paper)", "", "| Transition | Cells |",
              "|---|---|"]
    lines += [f"| {t} | {n} |" for t, n in comparison["transitions"].items()]
    return "\n".join(lines) + "\n"
