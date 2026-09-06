#!/usr/bin/env python
"""
Live (or offline-fake) dual-model Librarian extraction on one frozen paper (L1).

The thin entrypoint that turns a frozen ``CanonicalText`` + two ``ModelClient``s
into a ``StrategySpec`` + ``ExtractionTrace``, exercising the whole real pipeline:
enumeration -> per-field dual-model extraction (D9 merge + quote gate) ->
stamp + fail-closed validate -> emit.

Phases (docs/thresholds.yaml -> librarian.model_stack, D33 two-phase policy):
  --phase dev   Phase-D free pair (Gemini 3.1-flash-lite + Mistral free tier). Needs
                GEMINI_API_KEY + MISTRAL_API_KEY (see .env.example). Non-reportable.
  --phase report  Phase-F reported pair (Claude Sonnet + Gemini). Gated on SKU
                authorization (D33); needs ANTHROPIC_API_KEY too.
  --phase fake  Offline wiring proof: FakeModelClients scripted with a handful of
                verbatim-locating BBW answers. No network, no keys. Proves the
                assemble -> run_paper -> emit -> validate path end-to-end.

Prompt assembly is PAPER-TEXT-FIRST (2026-09-02 amendment): the paper text
leads every live prompt as a provider-cacheable prefix; templates/schemas/contract
unchanged.

Enumeration is now LIVE (WS-3): each model's ``extract_enumeration(ct)`` returns a
construction list via the frozen ``enumeration`` run-template (manifest
``run_templates``), and ``enumerate_constructions`` runs the D20 dual-model
agreement gate (a name-set / class disagreement routes the paper to review).
PAPERS[*]["constructions"] is now the FAKE seed only -- it scripts the offline
FakeModelClients so ``--phase fake`` still proves the assemble -> validate path;
the dev/report pairs enumerate against the real paper text.

Pre-Phase-F follow-ups (the dev smoke runs fine without them; reportable baselines
need them):
  * Format/schema failures currently fold into UNKNOWN(not_stated) (only the client's
    ``format_failures`` counter distinguishes them) -- before reportable runs they must
    be a DISTINCT trace signal so the contract §3.6 gate can separate the quote/format
    bucket from genuine silence.
  * The quote gate is strict L1 (the frozen ladder): some real model spans locate only
    at L2 (OCR folds) -> UNKNOWN(quote_match_failure) -> review (kept strict-L1 by
    decision, 2026-07-16). Nudge the extraction prompts toward short, verbatim,
    locate-friendly spans.
  * An authoritative per-field ``definition`` resource (frozen, hash-stamped);
    the ``control_n_groups`` manifest binding.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import yaml

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from agents.librarian.config import load_canonical_text  # noqa: E402
from agents.librarian.pipeline import (  # noqa: E402
    AssemblyIncomplete,
    Construction,
    ExtractionTrace,
    FakeModelClient,
    ModelAnswer,
    RunProvenance,
    TraceRunHeader,
    enumerate_constructions,
    fill_field,
    fill_method_summary,
    fill_signal_ref,
    load_gold_list,
    load_prompt_manifest,
    run_paper,
)
from agents.librarian.pipeline.form_filler import _field_kind  # noqa: E402
from agents.librarian.pipeline.model_client import FieldQuery  # noqa: E402
from agents.librarian.pipeline.real_client import (  # noqa: E402
    SCOPED_FIELDS_CONTRACT,
    PromptBuilder,
    RealClientError,
    assemble_field_prompt,
    build_client_pair,
)
from agents.librarian.registries import load_signal_concept_registry  # noqa: E402
from shared.reporting.run_manifest import (  # noqa: E402
    build_operational_profile,
    build_run_manifest,
    write_run_manifest,
)
from agents.librarian.registries.silence_policy import load_silence_policy_table  # noqa: E402
from agents.librarian.schema import (  # noqa: E402
    Combiner,
    Leg,
    MethodSummary,
    PaperFacts,
    Part1,
    Part2,
)
from agents.librarian.schema import estimation_fields as EF  # noqa: E402
from agents.librarian.schema.signal_ref import DescribedSignal, LocatedQuote, SignalRef  # noqa: E402
from agents.librarian.schema.strategy_spec import (  # noqa: E402
    EstimationBlock,
    InstrumentRef,
    InstrumentSet,
)
from agents.librarian.schema import fields as F  # noqa: E402
from agents.librarian.validators import load_tag_reason_registry  # noqa: E402
from agents.quant.config import Evidence, Inherited  # noqa: E402


# ---------------------------------------------------------------------------
# Paper registry (frozen canonical text + a provided enumeration).
# ---------------------------------------------------------------------------

PAPERS: dict[str, dict] = {
    "bbw": {
        "paper_id": "BBW_2019",
        "canonical_text": "evaluation/canonical_texts/bbw_2019.frozen.yaml",
        "gold_enum": "evaluation/gold_specs/enum_bbw_2019.yaml",
        # Fake seed (WS-3): BBW's headline sorted-portfolio construction, scripted
        # into the offline FakeModelClients. The quote is verbatim from the frozen
        # text (locates on page 2) so the strategy label ships STATED. The dev/report
        # pairs enumerate live and ignore this.
        "constructions": [
            {"name": "Downside Risk Factor (DRF)", "quote": "downside risk factor (DRF)", "cls": "strategy"},
        ],
    },
    # The other two anchors. Each construction quote is the anchor gold's own
    # formation_structure quote -- table-backed, verified binding against the
    # frozen text in locator_backfill_report.md -- so the strategy label ships
    # STATED and the run is scored against a gold built from the same sentence.
    "jnps": {
        "paper_id": "JNPS_2013",
        "canonical_text": "evaluation/canonical_texts/jnps_2013.frozen.yaml",
        "gold_enum": "evaluation/gold_specs/enum_jnps_2013.yaml",
        "constructions": [
            {
                "name": "Six-Month Momentum (mom6)",
                "quote": (
                    "each month t, bonds are sorted into decile portfolios, P1 to P10, "
                    "based on their cumulative returns over months t −6 to t −1 (formation period)"
                ),
                "cls": "strategy",
            },
        ],
    },
    "drr": {
        "paper_id": "DRR_2026",
        "canonical_text": "evaluation/canonical_texts/drr_2026.frozen.yaml",
        "gold_enum": "evaluation/gold_specs/enum_drr_2026.yaml",
        "constructions": [
            {
                "name": "Short-Term Reversal (str)",
                "quote": (
                    "we sort bonds into deciles each month and form value-weighted portfolios "
                    "(using bond market capitalization) that are long the top decile and short "
                    "the bottom decile"
                ),
                "cls": "strategy",
            },
        ],
    },
    # Scale-layer corpus papers (RQ2 coverage). Enumeration comes from the authored
    # per-paper enum gold via load_gold_list (--enumeration gold, the default), which
    # bypasses the D20 dual-model gate. No fake seed -- --phase fake is the anchor
    # wiring proof only (its scripted field quotes locate in BBW 2019 alone). Frozen
    # text + enum gold both verified: every enum quote locates at L1.
    "bbw2021": {
        "paper_id": "BBW_2021",
        "canonical_text": "evaluation/canonical_texts/bbw_2021.frozen.yaml",
        "gold_enum": "evaluation/gold_specs/enum_bbw_2021.yaml",
    },
    "dfps": {
        "paper_id": "DFPS_2026",
        "canonical_text": "evaluation/canonical_texts/dfps_2026.frozen.yaml",
        "gold_enum": "evaluation/gold_specs/enum_dfps_2026.yaml",
    },
    # Scope B (2026-09-04, registered hybrid): the fitted-model paper. family=
    # "estimation" routes to make_estimation_assembler (8 asked fields; 3 prose
    # registered not-asked; instruments run-template).
    "kpp": {
        "paper_id": "KPP_2023",
        "canonical_text": "evaluation/canonical_texts/kpp_2023.frozen.yaml",
        "gold_enum": "evaluation/gold_specs/enum_kpp_2023.yaml",
        "family": "estimation",
    },
    # T2 prospective set (registered 2026-09-04; hand-authored enum golds, quotes
    # 33/33 located against the frozen texts at registration). One-shot,
    # publish-as-found (T2-SEL discipline).
    "hvz": {
        "paper_id": "HVZ_2017",
        "canonical_text": "evaluation/canonical_texts/hvz_2017.frozen.yaml",
        "gold_enum": "evaluation/gold_specs/enum_hvz_2017.yaml",
    },
    "cgnst": {
        "paper_id": "CGNST_2017",
        "canonical_text": "evaluation/canonical_texts/cgnst_2017.frozen.yaml",
        "gold_enum": "evaluation/gold_specs/enum_cgnst_2017.yaml",
    },
    "klz": {
        "paper_id": "KLZ_2017",
        "canonical_text": "evaluation/canonical_texts/klz_2017.frozen.yaml",
        "gold_enum": "evaluation/gold_specs/enum_klz_2017.yaml",
    },
    "bektic": {
        "paper_id": "BEKTIC_2018",
        "canonical_text": "evaluation/canonical_texts/bektic_2018.frozen.yaml",
        "gold_enum": "evaluation/gold_specs/enum_bektic_2018.yaml",
    },
    "bwwss": {
        "paper_id": "BWWSS_2019",
        "canonical_text": "evaluation/canonical_texts/bwwss_2019.frozen.yaml",
        "gold_enum": "evaluation/gold_specs/enum_bwwss_2019.yaml",
    },
    # T4(b) synthetic evaluation instrument (end-to-end known-answer test). Registered mode
    # is gold-enum (a single SBM construction); the planted answer key lives at
    # evaluation/synthetic/planted_key_synth_2026.yaml. Not a scale-layer corpus paper and
    # never enters the RQ1/RQ3 denominators.
    "synth": {
        "paper_id": "SYNTH_2026",
        "canonical_text": "evaluation/canonical_texts/synth_2026.frozen.yaml",
        "gold_enum": "evaluation/gold_specs/enum_synth_2026.yaml",
    },
    # T5 Arm B must-refuse papers (evaluation/adversarial/reject_set.yaml, count_prereg 8).
    # Deliberately NO gold_enum: a gold-less paper falls through to LIVE enumeration even
    # under --enumeration gold, and enumeration/assembly refusing (exit 2/3) is exactly
    # the graded behaviour. Frozen texts live under the GITIGNORED
    # evaluation/adversarial/frozen/ (never canonical_texts/ -- the design-touched
    # consistency test is scoped there by design), so these entries are runnable only on
    # a machine that has frozen the local reject PDFs. Never a corpus/RQ1/RQ3 member.
    "reject_hxz": {"paper_id": "HXZ_REJECT",
                   "canonical_text": "evaluation/adversarial/frozen/reject_hxz.frozen.yaml"},
    "reject_kpj": {"paper_id": "KPJ_REJECT",
                   "canonical_text": "evaluation/adversarial/frozen/reject_kpj.frozen.yaml"},
    "reject_hlz": {"paper_id": "HLZ_REJECT",
                   "canonical_text": "evaluation/adversarial/frozen/reject_hlz.frozen.yaml"},
    "reject_gkx": {"paper_id": "GKX_REJECT",
                   "canonical_text": "evaluation/adversarial/frozen/reject_gkx.frozen.yaml"},
    "reject_gl": {"paper_id": "GL_REJECT",
                  "canonical_text": "evaluation/adversarial/frozen/reject_gl.frozen.yaml"},
    "reject_bpw": {"paper_id": "BPW_REJECT",
                   "canonical_text": "evaluation/adversarial/frozen/reject_bpw.frozen.yaml"},
    "reject_bkmx": {"paper_id": "BKMX_REJECT",
                    "canonical_text": "evaluation/adversarial/frozen/reject_bkmx.frozen.yaml"},
    "reject_dmr": {"paper_id": "DMR_REJECT",
                   "canonical_text": "evaluation/adversarial/frozen/reject_dmr.frozen.yaml"},
}


# A field the run never ASKED about, as opposed to one the paper was silent on.
# B2 (2026-09-02): `not_extracted` is now a REGISTERED tag-reason row (a statement
# about THIS RUN, never about the paper), so the G3 scorer buckets these NOT_ASKED
# by reason -- retiring the earlier note-prefix sniffing that rode `not_stated`.
# The note keeps the human-readable detail (and the historical prefix, so old
# Phase-D run dirs still score identically via the reader's fallback).
NOT_EXTRACTED_NOTE_PREFIX = "not extracted"


def _not_extracted(detail: str) -> Inherited:
    """UNKNOWN for a field this run did not ask about (never a claim of silence)."""
    return Inherited(
        None, "UNKNOWN",
        Evidence(note=f"{NOT_EXTRACTED_NOTE_PREFIX} ({detail})", unknown_reason="not_extracted"),
    )


# AssemblyIncomplete now lives in agents/librarian/pipeline/failures.py (CI-9,
# 2026-09-06): run_paper catches it PER CONSTRUCTION and records a typed review
# event, so one unresolved sort signal no longer aborts the paper's remaining
# constructions. Imported above; re-exported here so existing importers
# (tests, tooling) keep working.


# ---------------------------------------------------------------------------
# The per-construction assembler (the callable run_paper injects).
# ---------------------------------------------------------------------------

# Every field name a live run can extract (validates --fields fail-loud: a typo'd
# allowlist entry must never silently skip the target). Capped fields + the
# always-run structural/paper_facts names (listing an always-run name is legal).
_EXTRACTABLE_FIELDS: frozenset = frozenset((
    *F.COMMON_FIELDS,
    F.SORT_KIND, F.BUCKETING_METHOD, F.N_GROUPS, F.STRIPE_AGGREGATION,
    F.CONTROL_MISSING_POLICY, F.LONG_LEG, F.SIGNAL_TRANSFORM, F.CONTROL_N_GROUPS,
    "control_axis",
    F.FORMATION_STRUCTURE, F.ASSET_CLASS, F.METHOD_SUMMARY, F.SORT_SIGNAL, F.COMBINER,
    F.SAMPLE_START, F.SAMPLE_END, F.CLAIMED_HEADLINE_METRIC,
))


def make_assembler(model_a, model_b, registry, manifest, field_limit=None,
                   field_allowlist=None):
    """Build the ``assemble_strategy(construction, canonical_text, prov)`` closure
    run_paper drives. Fills Part 1 + all 38 Part-2 fields via the live per-field
    dual-model merge; silent fields degrade to UNKNOWN (blank is the safe state).

    ``field_limit`` (int or None) caps how many of the non-structural per-value
    fields (the 28 common + the leg-inherited fields) are extracted LIVE -- the
    rest degrade to UNKNOWN(not_extracted) with no model call. A cheap smoke that
    still exercises the full live path; None = extract everything. The structural
    fields (Part 1, sort_signal, combiner) always run.

    ``field_allowlist`` (set of field names or None -- the C-lever, 2026-09-02):
    when given, ONLY the named non-structural fields are extracted live; every
    other capped field degrades to UNKNOWN(not_extracted), exactly the --limit
    degrade. Built for the T5 targeted-field runs (each perturbation targets one
    field; the sheets grade that field's row, and untargeted fields bucket
    NOT_ASKED in scoring). The always-run set (Part 1, method_summary,
    sort_signal, combiner, paper_facts) is unaffected -- listing one of those
    names is legal and simply restricts nothing extra. None = the default behaviour,
    byte-identical."""

    def _q(field_name):
        # Stamp template hashes into the trace when the field is bound in the
        # manifest; else fall back to a default query (fill_field infers the kind).
        # control_n_groups (v1.1) is not yet bound in the manifest -- follow-up.
        return manifest.query_for(field_name) if field_name in manifest.field_types else None

    def _skipped(cause: str):
        return _not_extracted(cause)

    def assemble(construction: Construction, canonical_text, prov: RunProvenance):
        # method_summary renders with this construction's name (the Protocol
        # carries no construction context) -- set it on any client that supports it.
        # The enumeration quote rides beside it for scoped-field runs (CI-10
        # candidate): inert unless the client's scoped_fields flag is on.
        for m in (model_a, model_b):
            if hasattr(m, "current_strategy_label"):
                m.current_strategy_label = construction.name
            if hasattr(m, "current_strategy_quote"):
                m.current_strategy_quote = construction.quote

        records = []
        live = [0]  # count of live per-value extractions (for --limit)

        def _capped(name):
            """Extract one per-value field live, or skip to UNKNOWN when the field
            is outside the allowlist (targeted run) or the live budget is spent."""
            if field_allowlist is not None and name not in field_allowlist:
                return _skipped("targeted --fields")
            if field_limit is not None and live[0] >= field_limit:
                return _skipped("smoke --limit")
            live[0] += 1
            out = fill_field(name, model_a, model_b, canonical_text, query=_q(name))
            records.append(out.trace)
            return out.value

        # --- strategy label (re-located; STATED iff its quote locates) ---------
        loc = canonical_text.locate(construction.quote)
        if loc is not None:
            strategy_label = Inherited(
                construction.name, "STATED", Evidence(quote=construction.quote, locator=loc)
            )
        else:
            strategy_label = Inherited(
                construction.name,
                "UNKNOWN",
                Evidence(note="strategy-name quote did not locate", unknown_reason="quote_match_failure"),
            )

        # --- Part 1 ------------------------------------------------------------
        fs = fill_field(F.FORMATION_STRUCTURE, model_a, model_b, canonical_text, query=_q(F.FORMATION_STRUCTURE))
        ac = fill_field(F.ASSET_CLASS, model_a, model_b, canonical_text, query=_q(F.ASSET_CLASS))
        method_summary, ms_trace = fill_method_summary(
            model_a, model_b, canonical_text, query=_q(F.METHOD_SUMMARY)
        )
        records += [fs.trace, ac.trace, ms_trace]
        if method_summary is None:
            method_summary = MethodSummary(
                summary=Inherited(
                    None, "UNKNOWN",
                    Evidence(note="method_summary: no locating summary", unknown_reason="not_stated"),
                ),
                quotes=(),
            )
        part1 = Part1(formation_structure=fs.value, asset_class=ac.value, method_summary=method_summary)

        # --- Part 2 common (28) ------------------------------------------------
        common = {name: _capped(name) for name in F.COMMON_FIELDS}

        # --- Leg (sort block, one leg per construction, D19) -------------------
        # sort_signal is the crux -- always extracted live.
        sig = fill_signal_ref(F.SORT_SIGNAL, model_a, model_b, canonical_text, registry)
        records.append(sig.trace)
        records.extend(sig.param_traces)
        if sig.signal_ref is None:
            raise AssemblyIncomplete(
                f"{construction.name!r}: sort_signal did not resolve to a registry concept "
                f"(concept_id tag = {sig.concept_id.tag}); routing to review.",
                paper_id=prov.paper_id,
                construction_name=construction.name,
            )
        # control_axis (2nd sort of a double sort; literal field name -- fields.py
        # has no F.CONTROL_AXIS). Skipped under --limit (single sorts leave it None);
        # under an allowlist it runs only when explicitly targeted.
        run_control_axis = field_limit is None and (
            field_allowlist is None or "control_axis" in field_allowlist
        )
        if run_control_axis:
            ctrl = fill_signal_ref("control_axis", model_a, model_b, canonical_text, registry)
            records.append(ctrl.trace)
            records.extend(ctrl.param_traces)
            control_axis = ctrl.signal_ref
        else:
            control_axis = None

        leg_inh = {
            name: _capped(name)
            for name in (
                F.SORT_KIND, F.BUCKETING_METHOD, F.N_GROUPS, F.STRIPE_AGGREGATION,
                F.CONTROL_MISSING_POLICY, F.LONG_LEG, F.SIGNAL_TRANSFORM, F.CONTROL_N_GROUPS,
            )
        }

        leg = Leg(
            sort_signal=sig.signal_ref,
            control_axis=control_axis,
            sort_kind=leg_inh[F.SORT_KIND],
            bucketing_method=leg_inh[F.BUCKETING_METHOD],
            n_groups=leg_inh[F.N_GROUPS],
            stripe_aggregation=leg_inh[F.STRIPE_AGGREGATION],
            control_missing_policy=leg_inh[F.CONTROL_MISSING_POLICY],
            long_leg=leg_inh[F.LONG_LEG],
            signal_transform=leg_inh[F.SIGNAL_TRANSFORM],
            control_n_groups=leg_inh[F.CONTROL_N_GROUPS],
        )

        comb = fill_field(F.COMBINER, model_a, model_b, canonical_text, query=_q(F.COMBINER))
        records.append(comb.trace)
        part2 = Part2(legs=(leg,), combiner=Combiner(kind=comb.value), **common)

        # --- paper_facts (v1.1, spec-level) ------------------------------------
        # Analysis-only extraction output: RQ1-scorable, and the adapter NEVER
        # reads it (Guard 2). Always extracted live -- these are 4 gold-STATED
        # fields on every anchor, so skipping them under --limit would put a
        # pipeline gap into the coverage denominator.
        pf_start = fill_field(F.SAMPLE_START, model_a, model_b, canonical_text, query=_q(F.SAMPLE_START))
        pf_end = fill_field(F.SAMPLE_END, model_a, model_b, canonical_text, query=_q(F.SAMPLE_END))
        pf_metric = fill_field(
            F.CLAIMED_HEADLINE_METRIC, model_a, model_b, canonical_text,
            query=_q(F.CLAIMED_HEADLINE_METRIC), value_kind="paper_metric",
        )
        records += [pf_start.trace, pf_end.trace, pf_metric.trace]
        paper_facts = PaperFacts(
            sample_start=pf_start.value,
            sample_end=pf_end.value,
            # universe_filter needs a prose field-type (unbuilt) and a scoring
            # rubric (unauthored, D34) -- see the manifest note. Marked NOT ASKED,
            # not silent: the paper is not silent on its universe, we did not ask.
            universe_filter=_not_extracted("universe_filter: no prose field-type yet"),
            claimed_headline_metric=pf_metric.value,
        )

        header = TraceRunHeader(
            paper_id=prov.paper_id,
            strategy_label=construction.name,
            registry_version=prov.registry_version,
            registry_hash=prov.registry_hash,
            silence_table_version=prov.silence_table_version,
            canonical_text_hash=prov.canonical_text_hash,
            model_a_id=prov.model_a_id,
            model_b_id=prov.model_b_id,
            run_id=prov.run_id,
            timestamp=prov.timestamp,
            prompt_template_hashes=prov.prompt_template_hashes,
        )
        trace = ExtractionTrace(header=header, records=tuple(records))
        return part1, part2, strategy_label, trace, paper_facts

    return assemble


def enumerate_field_prompts(builder, manifest, constructions, canonical_text,
                            field_allowlist=None):
    """B-lever (2026-09-02): yield every ``(strategy_label, field, prefix, suffix,
    max_tokens)`` per-field prompt an UNCAPPED live run would issue for these
    constructions -- the batch-prefetch work list (model-agnostic: the same prompt
    bytes go to both models of the pair; only the cache key's model_id differs).

    Mirrors ``make_assembler``'s asking pattern exactly (parity-pinned by
    tests/unit/test_prefetch.py): Part 1 + method_summary, the 28 common fields,
    sort_signal + control_axis (ad-hoc ``signal_ref`` queries, matching
    fill_signal_ref -- NOT manifest-bound), the 8 leg fields, combiner, and the
    3 asked paper_facts -- honouring ``field_allowlist`` the way ``_capped`` does.

    EXCLUDED by design: signal-PARAMETER sub-prompts (which parameters exist
    depends on the concept the models answer, so they are unenumerable statically
    -- the live run fills them as normally-counted residual misses) and
    ``--limit`` runs (a dev smoke; never prefetched). The whole-paper enumeration
    prompt is separate (``assemble_enumeration_prompt``): only gold-enum-less
    papers make it."""

    def _query(name):
        bound = manifest.query_for(name) if name in manifest.field_types else None
        if bound is not None:
            return bound
        # Unbound fallbacks mirror the live callsites exactly: fill_method_summary
        # defaults to kind="method_summary"; fill_field defaults to _field_kind.
        kind = "method_summary" if name == F.METHOD_SUMMARY else _field_kind(name)
        return FieldQuery(field=name, kind=kind)

    def _allowed(name):
        return field_allowlist is None or name in field_allowlist

    for construction in constructions:
        label = construction.name
        names: list[tuple[str, FieldQuery]] = [
            (F.FORMATION_STRUCTURE, _query(F.FORMATION_STRUCTURE)),
            (F.ASSET_CLASS, _query(F.ASSET_CLASS)),
            (F.METHOD_SUMMARY, _query(F.METHOD_SUMMARY)),
        ]
        names += [(n, _query(n)) for n in F.COMMON_FIELDS if _allowed(n)]
        names.append((F.SORT_SIGNAL, FieldQuery(field=F.SORT_SIGNAL, kind="signal_ref")))
        if _allowed("control_axis"):
            names.append(("control_axis", FieldQuery(field="control_axis", kind="signal_ref")))
        names += [(n, _query(n)) for n in (
            F.SORT_KIND, F.BUCKETING_METHOD, F.N_GROUPS, F.STRIPE_AGGREGATION,
            F.CONTROL_MISSING_POLICY, F.LONG_LEG, F.SIGNAL_TRANSFORM, F.CONTROL_N_GROUPS,
        ) if _allowed(n)]
        names.append((F.COMBINER, _query(F.COMBINER)))
        names += [(n, _query(n)) for n in
                  (F.SAMPLE_START, F.SAMPLE_END, F.CLAIMED_HEADLINE_METRIC)]

        for name, query in names:
            prefix, suffix = assemble_field_prompt(builder, query, label, canonical_text)
            yield label, name, prefix, suffix, builder.max_tokens_for(query.kind)


# ---------------------------------------------------------------------------
# Scope B (2026-09-04): the fitted-model (estimated_factor_model) assembler.
# ---------------------------------------------------------------------------

def _merge_instruments(rows_a, rows_b, canonical_text):
    """The D9 EXTENSION for the whole-paper instruments call (documented
    convention, Scope B): ship an InstrumentRef only where BOTH models list the
    same registry concept_id AND a quote locates (model_a's span first, else
    model_b's) -- agreement + the quote gate, exactly the per-field discipline
    lifted to rows. The metadata dials ship STATED only on cross-model
    agreement (casefolded); otherwise UNKNOWN(not_stated) with the disagreement
    noted. Rows in only ONE list, and 'unrecognised' rows (identity cannot be
    equated across models), are counted in the returned note -- dropped
    conservatively, never shipped.

    Returns (InstrumentSet | None, note). None = zero shipped rows (a legal
    spec: ``instruments`` is an optional sibling)."""
    rows_a, rows_b = rows_a or [], rows_b or []

    def _index(rows):
        # Review m4: on duplicate concept_ids within ONE model's list, prefer the
        # FIRST row whose quote locates (an arbitrary last-wins could discard the
        # only locatable span); fall back to the first occurrence.
        out: dict[str, dict] = {}
        for r in rows:
            cid = r.get("concept_id")
            if not cid or cid == "unrecognised":
                continue
            if cid not in out:
                out[cid] = r
            else:
                held, cand = out[cid].get("quote"), r.get("quote")
                held_loc = (isinstance(held, str) and held.strip()
                            and canonical_text.locate(held) is not None)
                cand_loc = (isinstance(cand, str) and cand.strip()
                            and canonical_text.locate(cand) is not None)
                if cand_loc and not held_loc:
                    out[cid] = r
        return out

    by_id_a = _index(rows_a)
    by_id_b = _index(rows_b)
    n_unrec = sum(1 for r in rows_a + rows_b if r.get("concept_id") == "unrecognised")
    shipped, dropped = [], []
    for cid in sorted(set(by_id_a) & set(by_id_b)):
        ra, rb = by_id_a[cid], by_id_b[cid]
        quote = loc = None
        for cand in (ra.get("quote"), rb.get("quote")):
            if isinstance(cand, str) and cand.strip():
                loc = canonical_text.locate(cand)
                if loc is not None:
                    quote = cand
                    break
        if quote is None:
            dropped.append(f"{cid}(quote gate)")
            continue
        ev = Evidence(quote=quote, locator=loc)

        def dial(field_name):
            # Provenance note (review m5): the STATED evidence reuses the
            # instrument-PRESENCE span -- both models agreed on the dial value,
            # and D7 requires a locator, not proof-of-value; the imprecision is
            # accepted and documented here.
            va, vb = ra.get(field_name), rb.get(field_name)
            if (isinstance(va, str) and isinstance(vb, str)
                    and va.strip().casefold() == vb.strip().casefold() and va.strip()):
                return Inherited(va.strip(), "STATED", ev)
            return Inherited(None, "UNKNOWN", Evidence(
                note=f"{field_name}: models disagreed or silent for {cid}",
                unknown_reason="not_stated"))

        # Review m3: a hostile/hallucinated non-str label must degrade, never crash.
        raw_label = ra.get("label") if isinstance(ra.get("label"), str) else (
            rb.get("label") if isinstance(rb.get("label"), str) else cid)
        label = (raw_label or cid).strip() or cid
        shipped.append(InstrumentRef(
            concept_id=Inherited(cid, "STATED", ev),
            source_class=dial("source_class"),
            transform=dial("transform"),
            lag=dial("lag"),
            as_described=DescribedSignal(label=label, quotes=(
                LocatedQuote(text=quote, page=loc.page,
                             char_start=loc.char_start, char_end=loc.char_end),)),
        ))
    note = (f"instruments merge: a={len(rows_a)} b={len(rows_b)} "
            f"shipped={len(shipped)} agreed-but-dropped={dropped} "
            f"unrecognised-rows={n_unrec} (conservative: agreement + quote gate)")
    return (InstrumentSet(instruments=tuple(shipped)) if shipped else None), note


def make_estimation_assembler(model_a, model_b, manifest, instrument_registry):
    """The Scope-B analogue of ``make_assembler`` for the
    ``estimated_factor_model`` family (registered per-paper route, D-hybrid
    2026-09-04): Part 1 as usual; a stub all-UNKNOWN Part 2 (the schema's
    documented pattern -- sort-block fields are structurally inapplicable);
    the 8 ASKED estimation fields through the ordinary dual-model fill; the 3
    prose fields registered UNKNOWN(not_extracted); the instrument set via the
    whole-paper run-template on both models + the D9-extension merge."""

    def _q(name):
        return manifest.query_for(name) if name in manifest.field_types else None

    _stub_note = "fitted-model family: sort-block field structurally inapplicable (Scope B)"

    def assemble(construction, canonical_text, prov):
        for m in (model_a, model_b):
            if hasattr(m, "current_strategy_label"):
                m.current_strategy_label = construction.name
            if hasattr(m, "current_strategy_quote"):
                m.current_strategy_quote = construction.quote
        records = []

        loc = canonical_text.locate(construction.quote)
        if loc is not None:
            strategy_label = Inherited(construction.name, "STATED",
                                       Evidence(quote=construction.quote, locator=loc))
        else:
            strategy_label = Inherited(construction.name, "UNKNOWN", Evidence(
                note="strategy-name quote did not locate",
                unknown_reason="quote_match_failure"))

        fs = fill_field(F.FORMATION_STRUCTURE, model_a, model_b, canonical_text,
                        query=_q(F.FORMATION_STRUCTURE))
        ac = fill_field(F.ASSET_CLASS, model_a, model_b, canonical_text,
                        query=_q(F.ASSET_CLASS))
        method_summary, ms_trace = fill_method_summary(
            model_a, model_b, canonical_text, query=_q(F.METHOD_SUMMARY))
        records += [fs.trace, ac.trace, ms_trace]
        if method_summary is None:
            method_summary = MethodSummary(summary=Inherited(
                None, "UNKNOWN",
                Evidence(note="method_summary: no locating summary",
                         unknown_reason="not_stated")), quotes=())
        part1 = Part1(formation_structure=fs.value, asset_class=ac.value,
                      method_summary=method_summary)

        stub = _not_extracted(_stub_note)
        stub_leg = Leg(
            sort_signal=SignalRef(
                concept_id=Inherited(None, "UNKNOWN",
                                     Evidence(note=_stub_note,
                                              unknown_reason="not_extracted")),
                as_described=DescribedSignal(label="not applicable (fitted-model family)"),
                parameters={}),
            control_axis=None,
            **{n: _not_extracted(_stub_note) for n in (
                F.SORT_KIND, F.BUCKETING_METHOD, F.N_GROUPS, F.STRIPE_AGGREGATION,
                F.CONTROL_MISSING_POLICY, F.LONG_LEG, F.SIGNAL_TRANSFORM,
                F.CONTROL_N_GROUPS)},
        )
        part2 = Part2(**{n: _not_extracted(_stub_note) for n in F.COMMON_FIELDS},
                      legs=(stub_leg,), combiner=Combiner(kind=stub))

        est_vals = {}
        for name in EF.ESTIMATION_FIELDS:
            if name in EF.ESTIMATION_PROSE_FIELDS:
                # Rubric FROZEN 2026-09-04 (kpp_prose_rubric §4 PR-2): the prose
                # fields are ASKED via the method_summary mechanism (no equality
                # gate -- earliest-locating-quote ship; quote gate per rubric
                # §2.6). Graded only by the rubric, never headline (D34).
                ms, ms_trace = fill_method_summary(
                    model_a, model_b, canonical_text, query=_q(name), field=name)
                records.append(ms_trace)
                if ms is not None:
                    est_vals[name] = ms.summary
                else:
                    est_vals[name] = Inherited(
                        None, "UNKNOWN",
                        Evidence(note=f"{name}: no model produced a located prose answer",
                                 unknown_reason="not_stated"))
                continue
            vk = ("int_set" if name == EF.N_FACTORS_TESTED
                  else "int" if name in EF.ESTIMATION_INT_FIELDS else None)
            out = fill_field(name, model_a, model_b, canonical_text,
                             query=_q(name), value_kind=vk)
            records.append(out.trace)
            val = out.value
            if name == EF.N_FACTORS_TESTED and isinstance(val.value, tuple):
                # JSON-native boundary (review n7): the D9 merge compares sorted
                # tuples; the SPEC ships the list so to_dict output is
                # representation-stable against a JSON round trip.
                val = Inherited(list(val.value), val.tag, val.evidence)
            est_vals[name] = val
        estimation = EstimationBlock(**est_vals)

        rows_a = (model_a.extract_instruments(canonical_text)
                  if hasattr(model_a, "extract_instruments") else None)
        rows_b = (model_b.extract_instruments(canonical_text)
                  if hasattr(model_b, "extract_instruments") else None)
        instruments, inst_note = _merge_instruments(rows_a, rows_b, canonical_text)
        print(f"[run_librarian] {inst_note}")

        pf_start = fill_field(F.SAMPLE_START, model_a, model_b, canonical_text,
                              query=_q(F.SAMPLE_START))
        pf_end = fill_field(F.SAMPLE_END, model_a, model_b, canonical_text,
                            query=_q(F.SAMPLE_END))
        pf_metric = fill_field(F.CLAIMED_HEADLINE_METRIC, model_a, model_b,
                               canonical_text, query=_q(F.CLAIMED_HEADLINE_METRIC),
                               value_kind="paper_metric")
        records += [pf_start.trace, pf_end.trace, pf_metric.trace]
        paper_facts = PaperFacts(
            sample_start=pf_start.value, sample_end=pf_end.value,
            universe_filter=_not_extracted("universe_filter: no prose field-type yet"),
            claimed_headline_metric=pf_metric.value)

        header = TraceRunHeader(
            paper_id=prov.paper_id, strategy_label=construction.name,
            registry_version=prov.registry_version, registry_hash=prov.registry_hash,
            silence_table_version=prov.silence_table_version,
            canonical_text_hash=prov.canonical_text_hash,
            model_a_id=prov.model_a_id, model_b_id=prov.model_b_id,
            run_id=prov.run_id, timestamp=prov.timestamp,
            prompt_template_hashes=prov.prompt_template_hashes)
        trace = ExtractionTrace(header=header, records=tuple(records))
        return (part1, part2, strategy_label, trace, paper_facts,
                estimation, instruments, instrument_registry)

    return assemble


# ---------------------------------------------------------------------------
# Client construction.
# ---------------------------------------------------------------------------

def _load_dotenv(path: Path) -> None:
    """Minimal .env loader: KEY=VALUE lines into os.environ (never overrides)."""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip())


def build_clients(phase: str, builder: PromptBuilder, out_dir: Path, paper: dict,
                  min_interval_s: float | None = None, cache_dir: Path | None = None):
    """Return (model_a, model_b). ``fake`` = offline scripted; ``dev``/``report`` =
    live vendor clients read from thresholds.yaml + env keys. ``paper`` supplies the
    fake pair's scripted enumeration seed (WS-3); the live pairs enumerate live.
    ``min_interval_s`` (when given) overrides the stack's per-model call pacing -- an
    operational free-tier rate-limit knob, not a frozen numerical threshold.
    ``cache_dir`` (when given) enables the shared disk replay cache (B1) so an
    interrupted or repeated run never re-pays a vendor call -- an operational knob,
    like pacing, deliberately NOT a thresholds.yaml key."""
    if phase == "fake":
        return _fake_pair(paper)

    _load_dotenv(_REPO_ROOT / ".env")
    thresholds = yaml.safe_load((_REPO_ROOT / "docs" / "thresholds.yaml").read_text(encoding="utf-8"))
    stack = thresholds["librarian"]["model_stack"]
    block = dict(stack["phase_d" if phase == "dev" else "phase_f"])
    block["temperature"] = stack.get("temperature", 0)
    if min_interval_s is not None:
        block["min_interval_s"] = min_interval_s
    env_names = {block["model_a"]["api_key_env"], block["model_b"]["api_key_env"]}
    api_keys = {name: os.environ.get(name, "") for name in env_names}
    return build_client_pair(block, builder, api_keys, archive_dir=out_dir / "raw",
                             cache_dir=cache_dir)


def _fake_pair(paper: dict):
    """Two FakeModelClients scripted with a few verbatim-locating BBW answers --
    an offline proof of the assemble -> validate path (no network/keys). Both are
    seeded with the paper's hand-listed constructions as the enumeration output
    (WS-3): identical seeds -> the D20 agreement gate passes and the run proceeds."""
    scripted = {
        F.SORT_SIGNAL: ("var_5pct", "the 5% VaR"),            # locates p5
        F.FORMATION_STRUCTURE: ("sorted_portfolios", "downside risk factor (DRF)"),  # p2
        F.ASSET_CLASS: ("corporate_bonds", "corporate bond"),
        F.N_GROUPS: (5, "lowest-VaR quintile"),               # "quintile" -> 5, p8
        F.WEIGHTING_SCHEME: ("value", "value-weighted average"),  # p5
    }
    seed = tuple(
        Construction(name=c["name"], quote=c["quote"], cls=c["cls"])
        for c in paper.get("constructions", ())  # corpus papers carry no fake seed
    )

    def _client(cid):
        answers = {}
        for fld, (raw, quote) in scripted.items():
            # For a signal_ref field the concept_id rides in raw, exactly as the
            # ordinary D9 merge token; FakeModelClient ignores the query kind.
            answers[fld] = ModelAnswer(field=fld, answered=True, raw=raw, quote=quote)
        return FakeModelClient(cid, answers, enumeration=seed)

    return _client("fake-a"), _client("fake-b")


# ---------------------------------------------------------------------------
# Main.
# ---------------------------------------------------------------------------

def _canonical_text_hash(ct) -> str:
    payload = json.dumps(ct.to_dict(), sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _clear_stale_run_artefacts(out_dir: Path) -> None:
    """Make a re-run into an existing out_dir IDEMPOTENT. The raw archive is opened in APPEND mode while spec/trace are
    overwritten, so re-using a populated out_dir would accumulate a second
    archive line per field and trip the run_artefacts.check_raw integrity guard
    -- corrupting a paid run's artefacts at scoring time (this exact mismatch
    already happened once: runs/bbw_full.). A re-run is a NEW counted run (D31/
    I3), never a silent continuation, so stale artefacts from the previous run
    are cleared up front; with the disk replay cache the fresh archive is
    reconstructed at zero vendor cost -- true zero-cost resume."""
    for pattern in ("spec_*.json", "trace_*.json", "run_manifest.json"):
        for p in out_dir.glob(pattern):
            p.unlink()
    raw = out_dir / "raw"
    if raw.is_dir():
        for p in raw.glob("*.jsonl"):
            p.unlink()


def _paper_failed(exc: RealClientError) -> int:
    """Typed exit for a model/transport failure DURING extraction (enumeration or
    per-field) -- e.g. retries exhausted on a 429. It is an infra failure, NOT paper
    silence: degrading the un-asked constructions/fields to UNKNOWN would fabricate
    "the paper was silent" from "we could not ask" (the not_extracted != UNKNOWN
    distinction), inflating the RQ1 UNKNOWN count; and v1 emits no partial spec set
    (D31). Stop and route to a typed paper_failed -- re-run later (paced / paid model)
    as a NEW counted run, never a silent resume (I3). The build-time missing-key
    RealClientError is raised in build_clients, BEFORE any extraction call, and stays
    fatal by design."""
    print(f"[run_librarian] paper_failed: extraction client failure: {exc}")
    return 4


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Live dual-model Librarian extraction on one paper.")
    ap.add_argument("--paper", default="bbw", choices=sorted(PAPERS), help="which frozen paper")
    ap.add_argument("--phase", default="fake", choices=("fake", "dev", "report"), help="model pair")
    ap.add_argument("--out", default="runs/bbw_smoke", help="output directory")
    ap.add_argument("--limit", type=int, default=None,
                    help="cap live extraction to the first N per-value fields (cheap smoke; "
                         "rest -> UNKNOWN). Omit to extract everything.")
    ap.add_argument("--enumeration", default="gold", choices=("gold", "live"),
                    help="'gold' (default): use the authored per-paper gold enumeration list "
                         "(evaluation/gold_specs/enum_*.yaml) as the agreed construction set -- "
                         "correct for the gold anchor papers, whose constructions are settled. "
                         "'live': run the D20 dual-model enumeration-agreement gate (scale layer).")
    ap.add_argument("--min-interval-s", type=float, default=None, dest="min_interval_s",
                    help="minimum seconds between calls PER MODEL (rate-limit pacing; overrides the "
                         "stack default). Use on the free tier to avoid 429s.")
    ap.add_argument("--cache-dir", default="runs/librarian_cache", dest="cache_dir",
                    help="shared disk replay cache for live vendor calls (B1). A repeated or "
                         "resumed run replays cached raw responses at zero cost. Default ON; the "
                         "cache is shared ACROSS runs and keyed by (model_id, prompt).")
    ap.add_argument("--no-cache", action="store_true",
                    help="disable the disk replay cache. REQUIRED for any run whose meaning "
                         "depends on call independence (e.g. contract §3.4 repeats / flip-rate): "
                         "a cached replay is byte-identical to the original, not a fresh sample.")
    ap.add_argument("--fields", default=None,
                    help="comma-separated field allowlist (C-lever, targeted runs): only the "
                         "named non-structural fields are extracted live; the rest degrade to "
                         "UNKNOWN(not_extracted). Structural fields + paper_facts always run. "
                         "Unknown names fail loud (a typo must never silently skip everything).")
    # T5 Arm-A variant overrides: run this registered paper's extraction against a
    # DIFFERENT frozen canonical text (a perturbed variant) without registering the
    # 257 variants as PAPERS entries. The override paths are recorded in the run
    # manifest's inputs (hashed), so the exact variant is always attributable.
    ap.add_argument("--canonical-text", default=None, dest="canonical_text_override",
                    help="override the paper's frozen canonical text path (T5 perturbed variant).")
    ap.add_argument("--scoped-fields", action="store_true", dest="scoped_fields",
                    help="construction-scoped field queries (CI-10 candidate): every "
                         "per-field prompt carries this construction's name + enum "
                         "quote, so a multi-construction paper's fields resolve per "
                         "construction instead of paper-level. Default OFF = the "
                         "historical byte-identical assembly. Scoped runs stamp "
                         "SCOPED_FIELDS_CONTRACT into prompt_template_hashes.")
    ap.add_argument("--gold-enum", default=None, dest="gold_enum_override",
                    help="override the paper's gold enumeration list path.")
    args = ap.parse_args(argv)

    paper = dict(PAPERS[args.paper])
    if args.canonical_text_override:
        paper["canonical_text"] = args.canonical_text_override
    if args.gold_enum_override:
        paper["gold_enum"] = args.gold_enum_override

    field_allowlist = None
    if args.fields is not None:
        field_allowlist = {f.strip() for f in args.fields.split(",") if f.strip()}
        unknown = field_allowlist - _EXTRACTABLE_FIELDS
        if unknown:
            ap.error(f"--fields contains unknown field name(s): {sorted(unknown)}. "
                     f"Extractable: {sorted(_EXTRACTABLE_FIELDS)}")
    out_dir = (_REPO_ROOT / args.out) if not Path(args.out).is_absolute() else Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    ct = load_canonical_text(_REPO_ROOT / paper["canonical_text"])
    ct.require_frozen()

    registry = load_signal_concept_registry()
    tag_reason = load_tag_reason_registry()
    manifest = load_prompt_manifest()
    silence_table = load_silence_policy_table()
    builder = PromptBuilder.load(registry, manifest)

    cache_dir = None
    if not args.no_cache and args.cache_dir:
        cache_dir = (_REPO_ROOT / args.cache_dir) if not Path(args.cache_dir).is_absolute() else Path(args.cache_dir)
    model_a, model_b = build_clients(args.phase, builder, out_dir, paper,
                                     min_interval_s=args.min_interval_s, cache_dir=cache_dir)
    if args.scoped_fields:
        # CI-10 candidate: construction-scoped field prompts. Attribute-set like
        # current_strategy_label; a client without the seam (FakeModelClient)
        # simply ignores the flag -- the fake path has no prompt assembly.
        for m in (model_a, model_b):
            if hasattr(m, "scoped_fields"):
                m.scoped_fields = True
    # Clear AFTER the clients build: a missing-key failure (raised in build_clients,
    # fatal by design) must never wipe a previous run's artefacts first (review
    # 2026-09-02 footgun note). Nothing writes to out_dir before this point.
    _clear_stale_run_artefacts(out_dir)

    now = datetime.now(timezone.utc).isoformat()
    t_start = time.monotonic()   # WS-8 wall-clock: covers enumeration + per-field extraction
    prov = RunProvenance(
        paper_id=paper["paper_id"],
        registry_version=registry.version,
        registry_hash=registry.content_hash,
        silence_table_version=silence_table.version,
        canonical_text_hash=_canonical_text_hash(ct),
        model_a_id=model_a.model_id,
        model_b_id=model_b.model_id,
        prompt_template_hashes=(
            # A scoped run's extraction contract differs (the construction-context
            # block heads every field suffix), so its header stamp must differ:
            # append the block's own versioned hash. Unscoped runs keep the
            # historical stamp byte-identical.
            manifest.combined_prompt_hash + ";" + SCOPED_FIELDS_CONTRACT
            if args.scoped_fields else manifest.combined_prompt_hash
        ),
        run_id=f"{args.paper}-{args.phase}-{now}",
        timestamp=now,
    )

    def _early_manifest() -> None:
        """WS-8 on the review/failure exits (B4): the calls already made (enumeration,
        partial extraction) are real spend and are recorded honestly -- partial usage,
        zero spec outputs. Closes the 'documented follow-up' on _emit_run_manifest."""
        _emit_run_manifest(out_dir, prov, args.phase, model_a, model_b,
                           n_specs=0, wall_clock_seconds=time.monotonic() - t_start,
                           inputs=[paper["canonical_text"]])

    # Enumeration source. Two modes:
    #  * gold (default): the authored per-paper gold enumeration list IS the agreed
    #    construction set. Correct for the gold anchor papers -- their constructions are
    #    settled, so field extraction need not be gated behind two models wording the
    #    same list identically (the D20 verbatim-name-set gate, which real independent
    #    models rarely clear on a multi-construction paper).
    #  * live: run the D20 dual-model enumeration-agreement gate (scale-layer papers,
    #    where no authored gold list exists).
    if args.enumeration == "gold" and paper.get("gold_enum"):
        enum = load_gold_list(_REPO_ROOT / paper["gold_enum"])
        print(f"[run_librarian] enumeration=gold ({paper['gold_enum']}): "
              f"{len(enum.constructions)} construction(s), {len(enum.strategies)} strategy")
    else:
        # Live enumeration (WS-3): each model returns its construction list; the D20
        # dual-model agreement gate ships the agreed set (relocated against ct) or
        # routes the paper to review on any name-set / class disagreement.
        try:
            list_a = model_a.extract_enumeration(ct)
            list_b = model_b.extract_enumeration(ct)
        except RealClientError as exc:
            # 429s were observed at THIS enumeration gate on the first live run (WS-3),
            # not only per-field -- same infra-failure semantics as the run_paper catch.
            _early_manifest()
            return _paper_failed(exc)
        enum = enumerate_constructions(ct, list_a, list_b, paper["paper_id"])
        if not enum.agreed:
            print(f"[run_librarian] REVIEW: enumeration disagreement: {enum.disagreement.detail}")
            _early_manifest()
            return 2
        if not enum.constructions:
            print("[run_librarian] REVIEW: enumeration produced zero constructions "
                  "(both models empty or unparseable) -- routing to review rather than "
                  "proceeding with an empty spec set")
            _early_manifest()
            return 2

    if paper.get("family") == "estimation":
        # Scope B (2026-09-04): the fitted-model route is a REGISTERED per-paper
        # mode (KPP is the registered estimated_factor_model paper), never an
        # extraction-dependent branch. --limit/--fields do not apply here.
        instrument_registry = load_signal_concept_registry(
            path=Path("agents/librarian/data/instrument_concept_registry.yaml"))
        assembler = make_estimation_assembler(model_a, model_b, manifest,
                                              instrument_registry)
    else:
        assembler = make_assembler(model_a, model_b, registry, manifest,
                                   field_limit=args.limit,
                                   field_allowlist=field_allowlist)

    limit_note = f" limit={args.limit}" if args.limit is not None else ""
    print(f"[run_librarian] paper={args.paper} phase={args.phase}{limit_note} "
          f"models=({model_a.model_id}, {model_b.model_id})")
    try:
        result = run_paper(
            canonical_text=ct,
            enumeration=enum,
            assemble_strategy=assembler,
            prov=prov,
            registry=registry,
            tag_reason_registry=tag_reason,
        )
    except AssemblyIncomplete as exc:
        # Safety net only: since CI-9 (2026-09-06) run_paper catches assembly
        # failures per construction, so this fires only if one escapes outside
        # the construction loop. Fail-closed as before: review exit.
        print(f"[run_librarian] REVIEW: {exc}")
        _early_manifest()
        return 2
    except RealClientError as exc:
        _early_manifest()
        return _paper_failed(exc)

    _write_outputs(result, out_dir)
    _emit_run_manifest(out_dir, prov, args.phase, model_a, model_b,
                       n_specs=len(result.specs), wall_clock_seconds=time.monotonic() - t_start,
                       inputs=[paper["canonical_text"]])
    _report(result, out_dir, model_a, model_b)
    if result.specs:
        return 0
    # Zero specs: if every construction fell to a typed assembly review, the
    # paper-level outcome is review (exit 2, matching the pre-CI-9 semantics for
    # this condition); otherwise the established zero-specs exit (3).
    if any(getattr(ev, "kind", None) == "assembly_incomplete" for ev in result.events):
        return 2
    return 3


def _emit_run_manifest(out_dir, prov, phase, model_a, model_b, *, n_specs,
                       wall_clock_seconds, inputs=None):
    """WS-8 (§4.7): the per-run operational log sidecar. Tokens/calls/retries are summed
    across both models from their mechanical counters (a FakeModelClient has none, so the
    profile degrades to zeros — recorded, never guessed). Emitted on the completion path
    AND (B4) on the review/failure exits via _early_manifest -- partial usage there is
    real spend and is recorded honestly."""
    def _usage(m):
        return m.operational_usage() if hasattr(m, "operational_usage") else {}
    ua, ub = _usage(model_a), _usage(model_b)
    def _sum(k):
        return (ua.get(k, 0) or 0) + (ub.get(k, 0) or 0)
    write_run_manifest(out_dir, build_run_manifest(
        run_id=prov.run_id,
        driver="run_librarian",
        timestamp=datetime.now(timezone.utc).isoformat(),
        inputs=list(inputs or []),   # C-lever: the (possibly variant-overridden) canonical text, hashed
        configs=["docs/thresholds.yaml"],
        outputs=[str(out_dir / f"spec_{i}.json") for i in range(n_specs)],
        operational_profile=build_operational_profile(
            phase=str(phase),
            model_calls=_sum("model_calls"),
            prompt_tokens=_sum("prompt_tokens") or None,
            completion_tokens=_sum("completion_tokens") or None,
            cache_creation_tokens=_sum("cache_creation_tokens") or None,
            cache_read_tokens=_sum("cache_read_tokens") or None,
            wall_clock_seconds=wall_clock_seconds,
            retries=_sum("retries"),
            capability="llm",
        ),
    ))


def _write_outputs(result, out_dir: Path) -> None:
    for i, (spec, trace) in enumerate(result.specs):
        (out_dir / f"spec_{i}.json").write_text(
            json.dumps(spec.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8"
        )
        (out_dir / f"trace_{i}.json").write_text(
            json.dumps(trace.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8"
        )
    if result.events:
        # CI-9: persist the typed run-record events (assembly reviews, emission
        # refusals) so per-construction outcomes are consumable downstream (the
        # coverage layer), not just printed. Additive -- no existing reader.
        def _ev(ev):
            if hasattr(ev, "to_dict"):
                return ev.to_dict()
            return {"kind": type(ev).__name__, "detail": str(ev)}
        (out_dir / "events.json").write_text(
            json.dumps([_ev(ev) for ev in result.events], indent=2, ensure_ascii=False),
            encoding="utf-8",
        )


def _report(result, out_dir: Path, model_a, model_b) -> None:
    for i, (spec, _trace) in enumerate(result.specs):
        d = spec.to_dict()
        p2 = d["part2"]
        leg = p2["legs"][0]
        p2_tags = [v["tag"] for k, v in p2.items() if isinstance(v, dict) and "tag" in v]
        leg_tags = [v["tag"] for k, v in leg.items() if isinstance(v, dict) and "tag" in v]
        stated = sum(t == "STATED" for t in p2_tags + leg_tags)
        unknown = sum(t == "UNKNOWN" for t in p2_tags + leg_tags)
        print(f"[run_librarian] spec_{i}: label={d['header']['strategy_label']['value']!r} "
              f"({d['header']['strategy_label']['tag']}) | "
              f"sort_signal={leg['sort_signal']['concept_id']['value']!r} | "
              f"Part2 STATED={stated} UNKNOWN={unknown}")
    for ev in result.events:
        print(f"[run_librarian] event: {type(ev).__name__}: {ev}")
    for m in (model_a, model_b):
        ff = getattr(m, "format_failures", 0)
        if ff:
            print(f"[run_librarian] {m.model_id}: {ff} format failure(s)")
    print(f"[run_librarian] wrote {len(result.specs)} spec(s) to {out_dir}")


if __name__ == "__main__":
    raise SystemExit(main())
