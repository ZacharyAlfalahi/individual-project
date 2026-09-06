#!/usr/bin/env python
"""
Closed-book memorisation probe (referents §1, RATIFIED 2026-09-03 with
δ_cb = 15pp; non-gating diagnostic, labelled exploratory).

Design per the registration: the SAME per-field suffixes (instruction + schema)
the production pipeline sends, with the PAPER TEXT block replaced by the frozen
NO-PAPER block below (which must identify the paper — the probe asks what the
model already knows ABOUT a named paper it has not been shown). Same models
(phase_f stack), same decoding (production _parse_json/_answer_from_parsed),
same normalisation (compare_field), one shot per (anchor, model, field), NO
retries. Scored per model against the ratified golds over gold-STATED
comparable fields; the registered comparison is
acc(well-known BBW) − acc(obscure DRR) vs δ_cb = 15pp.

Cost: ~2 anchors x 2 models x ~40 tiny prompts (no paper text) — cents.
Responses cached (crash protection; unique prompt bytes, shared cache dir).
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
from agents.librarian.pipeline import load_gold_list, load_prompt_manifest  # noqa: E402
from agents.librarian.pipeline.form_filler import _field_kind  # noqa: E402
from agents.librarian.pipeline.model_client import FieldQuery  # noqa: E402
from agents.librarian.pipeline.real_client import (  # noqa: E402
    PromptBuilder,
    _answer_from_parsed,
    _parse_json,
    assemble_field_prompt,
    make_backend,
)
from agents.librarian.registries import load_signal_concept_registry  # noqa: E402
from agents.librarian.schema import fields as F  # noqa: E402
from agents.scientist.researcher.cache import ResponseCache  # noqa: E402
from evaluation.gold_specs.gold_loader import load_gold_spec  # noqa: E402
from evaluation.harness.compare_policy import Comparability, compare_field  # noqa: E402
from scripts.run_librarian import _load_dotenv  # noqa: E402

# FROZEN no-paper block (the probe's one deviation from production assembly;
# registered function: identify the paper, state that its text is withheld).
_NO_PAPER_BLOCK = {
    "drf": ("YOU HAVE NOT BEEN SHOWN THE PAPER. The paper in question is: "
            "Bai, Bali & Wen (2019), \"Common Risk Factors in the Cross-Section of "
            "Corporate Bond Returns\", Journal of Financial Economics 131(3). "
            "Answer ONLY from whatever prior knowledge you have of this specific "
            "paper; if you do not know, answer with answered: false.\n\n"),
    "str": ("YOU HAVE NOT BEEN SHOWN THE PAPER. The paper in question is: "
            "Dickerson, Robotti & Rossetti (2026), \"The Corporate Bond Factor "
            "Replication Crisis\" (working paper). "
            "Answer ONLY from whatever prior knowledge you have of this specific "
            "paper; if you do not know, answer with answered: false.\n\n"),
}
ANCHOR_META = {
    "drf": ("bbw", "evaluation/canonical_texts/bbw_2019.frozen.yaml",
            "evaluation/gold_specs/enum_bbw_2019.yaml"),
    "str": ("drr", "evaluation/canonical_texts/drr_2026.frozen.yaml",
            "evaluation/gold_specs/enum_drr_2026.yaml"),
}
_LEG_FIELDS = (F.SORT_KIND, F.BUCKETING_METHOD, F.N_GROUPS, F.STRIPE_AGGREGATION,
               F.CONTROL_MISSING_POLICY, F.LONG_LEG, F.SIGNAL_TRANSFORM,
               F.CONTROL_N_GROUPS)


def _queries(manifest) -> list[FieldQuery]:
    """The production per-field query set (mirrors enumerate_field_prompts'
    membership; method_summary excluded — prose, no compare policy)."""
    def q(name):
        bound = manifest.query_for(name) if name in manifest.field_types else None
        return bound or FieldQuery(field=name, kind=_field_kind(name))

    out = [q(F.FORMATION_STRUCTURE), q(F.ASSET_CLASS)]
    out += [q(n) for n in F.COMMON_FIELDS]
    out.append(FieldQuery(field=F.SORT_SIGNAL, kind="signal_ref"))
    out += [q(n) for n in _LEG_FIELDS]
    out += [q(F.COMBINER), q(F.SAMPLE_START), q(F.SAMPLE_END),
            q(F.CLAIMED_HEADLINE_METRIC)]
    return out


def _gold_map(anchor: str) -> dict[str, tuple[object, str]]:
    spec = load_gold_spec(anchor)
    leg = spec.part2.legs[0]
    out = {F.FORMATION_STRUCTURE: spec.part1.formation_structure,
           F.ASSET_CLASS: spec.part1.asset_class,
           F.SORT_SIGNAL: leg.sort_signal,
           F.COMBINER: spec.part2.combiner.kind}
    for n in F.COMMON_FIELDS:
        out[n] = getattr(spec.part2, n)
    for n in _LEG_FIELDS:
        out[n] = getattr(leg, n)
    if spec.paper_facts is not None:
        out[F.SAMPLE_START] = spec.paper_facts.sample_start
        out[F.SAMPLE_END] = spec.paper_facts.sample_end
        out[F.CLAIMED_HEADLINE_METRIC] = spec.paper_facts.claimed_headline_metric
    return out


def _backends():
    _load_dotenv(_REPO_ROOT / ".env")
    stack = yaml.safe_load((_REPO_ROOT / "docs" / "thresholds.yaml")
                           .read_text(encoding="utf-8"))["librarian"]["model_stack"]
    block = stack["phase_f"]
    out = []
    for side in ("model_a", "model_b"):
        e = block[side]
        out.append((e["model_id"], make_backend(
            e["vendor"], e["model_id"], os.environ.get(e["api_key_env"], ""),
            temperature=float(stack.get("temperature", 0)))))
    return out


def main() -> int:
    registry = load_signal_concept_registry()
    manifest = load_prompt_manifest()
    builder = PromptBuilder.load(registry, manifest)
    cache = ResponseCache(_REPO_ROOT / "runs" / "librarian_cache")
    backends = _backends()

    rows, acc = [], {}
    for anchor, (paper_key, text_path, enum_path) in ANCHOR_META.items():
        ct = load_canonical_text(_REPO_ROOT / text_path)
        label = load_gold_list(_REPO_ROOT / enum_path).constructions[0].name
        gold = _gold_map(anchor)
        for model_id, backend in backends:
            n_correct = n_scored = 0
            for query in _queries(manifest):
                _prefix, suffix = assemble_field_prompt(builder, query, label, ct)
                prompt = _NO_PAPER_BLOCK[anchor] + suffix
                hit = cache.get(prompt, f"probe:{model_id}", 0)
                if hit is None:
                    raw, _v = backend.generate(prompt, builder.max_tokens_for(query.kind))
                    cache.put(prompt, f"probe:{model_id}", 0, raw)
                else:
                    raw = hit
                parsed = _parse_json(raw)
                ans = (_answer_from_parsed(query.field, query.kind, parsed, model_id)
                       if parsed is not None else None)
                g = gold.get(query.field)
                g_val = g.value if hasattr(g, "value") else g
                g_tag = g.tag if hasattr(g, "tag") else "SIGNAL"
                if query.field == F.SORT_SIGNAL:
                    g_val, g_tag = gold[F.SORT_SIGNAL], "STATED"
                if g_tag != "STATED" or g_val is None:
                    continue                              # scored over gold-STATED only
                run_val = ans.raw if (ans is not None and ans.answered) else None
                cmp = compare_field(query.field, g_val, run_val) if run_val is not None else None
                correct = bool(cmp and cmp.comparability == Comparability.COMPARABLE
                               and cmp.equal)
                n_scored += 1
                n_correct += int(correct)
                rows.append({"anchor": anchor, "model": model_id, "field": query.field,
                             "answered": run_val is not None, "correct": correct})
            acc[(anchor, model_id)] = (n_correct, n_scored)
            print(f"[probe] {anchor}/{model_id}: {n_correct}/{n_scored} correct")

    pooled = {a: (sum(c for (aa, _), (c, _) in acc.items() if aa == a),
                  sum(s for (aa, _), (_, s) in acc.items() if aa == a))
              for a in ANCHOR_META}
    acc_known = pooled["drf"][0] / pooled["drf"][1]
    acc_obscure = pooled["str"][0] / pooled["str"][1]
    gap_pp = (acc_known - acc_obscure) * 100
    verdict = "PRIOR FAMILIARITY INFERRED" if gap_pp > 15.0 else "within margin"
    summary = {
        "acc_well_known_bbw": f"{pooled['drf'][0]}/{pooled['drf'][1]} = {acc_known:.1%}",
        "acc_obscure_drr": f"{pooled['str'][0]}/{pooled['str'][1]} = {acc_obscure:.1%}",
        "gap_pp": round(gap_pp, 1), "delta_cb_pp": 15.0, "verdict": verdict,
        "per_cell": {f"{a}/{m}": f"{c}/{s}" for (a, m), (c, s) in acc.items()},
        "no_paper_blocks": _NO_PAPER_BLOCK,
    }
    out = _REPO_ROOT / "results" / "closed_book_probe.json"
    out.write_text(json.dumps({"summary": summary, "rows": rows}, indent=2),
                   encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "no_paper_blocks"},
                     indent=2))
    print(f"[probe] -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
