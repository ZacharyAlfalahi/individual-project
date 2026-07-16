#!/usr/bin/env python
"""
Live (or offline-fake) dual-model Librarian extraction on one frozen paper (L1).

The thin entrypoint that turns a frozen ``CanonicalText`` + two ``ModelClient``s
into a ``StrategySpec`` + ``ExtractionTrace``, exercising the whole real pipeline:
enumeration -> per-field dual-model extraction (D9 merge + quote gate) ->
stamp + fail-closed validate -> emit.

Phases (docs/thresholds.yaml -> librarian.model_stack, D33 two-phase policy):
  --phase dev   Phase-D free pair (Gemini 2.5 Flash + Mistral free tier). Needs
                GEMINI_API_KEY + MISTRAL_API_KEY (see .env.example). Non-reportable.
  --phase report  Phase-F reported pair (Claude Sonnet + Gemini). Gated on SKU
                authorization (D33); needs ANTHROPIC_API_KEY too.
  --phase fake  Offline wiring proof: FakeModelClients scripted with a handful of
                verbatim-locating BBW answers. No network, no keys. Proves the
                assemble -> run_paper -> emit -> validate path end-to-end.

Enumeration is PROVIDED here (a single hand-listed construction), not extracted
live: the ModelClient Protocol is field-oriented and there is no frozen
enumeration prompt/schema in the manifest yet -- live enumeration extraction is a
separate follow-up needing its own frozen contract. The per-field extraction (the
crux of L1) is fully live.

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
    live enumeration extraction; the ``control_n_groups`` manifest binding.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from agents.librarian.config import load_canonical_text  # noqa: E402
from agents.librarian.pipeline import (  # noqa: E402
    Construction,
    EnumerationResult,
    ExtractionTrace,
    FakeModelClient,
    ModelAnswer,
    RunProvenance,
    TraceRunHeader,
    fill_field,
    fill_method_summary,
    fill_signal_ref,
    load_prompt_manifest,
    run_paper,
)
from agents.librarian.pipeline.real_client import PromptBuilder, build_client_pair  # noqa: E402
from agents.librarian.registries import load_signal_concept_registry  # noqa: E402
from agents.librarian.registries.silence_policy import load_silence_policy_table  # noqa: E402
from agents.librarian.schema import (  # noqa: E402
    Combiner,
    Leg,
    MethodSummary,
    Part1,
    Part2,
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
        # Provided enumeration: BBW's headline sorted-portfolio construction. The
        # quote is verbatim from the frozen text (locates on page 2) so the
        # strategy label ships STATED.
        "constructions": [
            {"name": "Downside Risk Factor (DRF)", "quote": "downside risk factor (DRF)", "cls": "strategy"},
        ],
    },
}


class AssemblyIncomplete(RuntimeError):
    """A construction could not be assembled into a valid spec (e.g. its sort
    signal did not resolve to a registry concept). Surfaced, never silently
    dropped -- an unresolved sort signal means "cannot build this strategy",
    a review outcome, not a blank spec."""


# ---------------------------------------------------------------------------
# The per-construction assembler (the callable run_paper injects).
# ---------------------------------------------------------------------------

def make_assembler(model_a, model_b, registry, manifest, field_limit=None):
    """Build the ``assemble_strategy(construction, canonical_text, prov)`` closure
    run_paper drives. Fills Part 1 + all 38 Part-2 fields via the live per-field
    dual-model merge; silent fields degrade to UNKNOWN (blank is the safe state).

    ``field_limit`` (int or None) caps how many of the non-structural per-value
    fields (the 28 common + the leg-inherited fields) are extracted LIVE -- the
    rest degrade to UNKNOWN(not_extracted) with no model call. A cheap smoke that
    still exercises the full live path; None = extract everything. The structural
    fields (Part 1, sort_signal, combiner) always run."""

    def _q(field_name):
        # Stamp template hashes into the trace when the field is bound in the
        # manifest; else fall back to a default query (fill_field infers the kind).
        # control_n_groups (v1.1) is not yet bound in the manifest -- follow-up.
        return manifest.query_for(field_name) if field_name in manifest.field_types else None

    def _skipped():
        return Inherited(
            None, "UNKNOWN",
            Evidence(note="not extracted (smoke --limit)", unknown_reason="not_stated"),
        )

    def assemble(construction: Construction, canonical_text, prov: RunProvenance):
        # method_summary renders with this construction's name (the Protocol
        # carries no construction context) -- set it on any client that supports it.
        for m in (model_a, model_b):
            if hasattr(m, "current_strategy_label"):
                m.current_strategy_label = construction.name

        records = []
        live = [0]  # count of live per-value extractions (for --limit)

        def _capped(name):
            """Extract one per-value field live, or skip to UNKNOWN once the
            live budget (field_limit) is spent."""
            if field_limit is not None and live[0] >= field_limit:
                return _skipped()
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
                f"(concept_id tag = {sig.concept_id.tag}); routing to review."
            )
        # control_axis (2nd sort of a double sort; literal field name -- fields.py
        # has no F.CONTROL_AXIS). Skipped under --limit (single sorts leave it None).
        if field_limit is None:
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
        return part1, part2, strategy_label, trace

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


def build_clients(phase: str, builder: PromptBuilder, out_dir: Path):
    """Return (model_a, model_b). ``fake`` = offline scripted; ``dev``/``report`` =
    live vendor clients read from thresholds.yaml + env keys."""
    if phase == "fake":
        return _fake_pair()

    _load_dotenv(_REPO_ROOT / ".env")
    thresholds = yaml.safe_load((_REPO_ROOT / "docs" / "thresholds.yaml").read_text(encoding="utf-8"))
    stack = thresholds["librarian"]["model_stack"]
    block = dict(stack["phase_d" if phase == "dev" else "phase_f"])
    block["temperature"] = stack.get("temperature", 0)
    env_names = {block["model_a"]["api_key_env"], block["model_b"]["api_key_env"]}
    api_keys = {name: os.environ.get(name, "") for name in env_names}
    return build_client_pair(block, builder, api_keys, archive_dir=out_dir / "raw")


def _fake_pair():
    """Two FakeModelClients scripted with a few verbatim-locating BBW answers --
    an offline proof of the assemble -> validate path (no network/keys)."""
    scripted = {
        F.SORT_SIGNAL: ("var_5pct", "the 5% VaR"),            # locates p5
        F.FORMATION_STRUCTURE: ("sorted_portfolios", "downside risk factor (DRF)"),  # p2
        F.ASSET_CLASS: ("corporate_bonds", "corporate bond"),
        F.N_GROUPS: (5, "lowest-VaR quintile"),               # "quintile" -> 5, p8
        F.WEIGHTING_SCHEME: ("value", "value-weighted average"),  # p5
    }

    def _client(cid):
        answers = {}
        for fld, (raw, quote) in scripted.items():
            # For a signal_ref field the concept_id rides in raw, exactly as the
            # ordinary D9 merge token; FakeModelClient ignores the query kind.
            answers[fld] = ModelAnswer(field=fld, answered=True, raw=raw, quote=quote)
        return FakeModelClient(cid, answers)

    return _client("fake-a"), _client("fake-b")


# ---------------------------------------------------------------------------
# Main.
# ---------------------------------------------------------------------------

def _canonical_text_hash(ct) -> str:
    payload = json.dumps(ct.to_dict(), sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Live dual-model Librarian extraction on one paper.")
    ap.add_argument("--paper", default="bbw", choices=sorted(PAPERS), help="which frozen paper")
    ap.add_argument("--phase", default="fake", choices=("fake", "dev", "report"), help="model pair")
    ap.add_argument("--out", default="runs/bbw_smoke", help="output directory")
    ap.add_argument("--limit", type=int, default=None,
                    help="cap live extraction to the first N per-value fields (cheap smoke; "
                         "rest -> UNKNOWN). Omit to extract everything.")
    args = ap.parse_args(argv)

    paper = PAPERS[args.paper]
    out_dir = (_REPO_ROOT / args.out) if not Path(args.out).is_absolute() else Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    ct = load_canonical_text(_REPO_ROOT / paper["canonical_text"])
    ct.require_frozen()

    registry = load_signal_concept_registry()
    tag_reason = load_tag_reason_registry()
    manifest = load_prompt_manifest()
    silence_table = load_silence_policy_table()
    builder = PromptBuilder.load(registry, manifest)

    model_a, model_b = build_clients(args.phase, builder, out_dir)

    now = datetime.now(timezone.utc).isoformat()
    prov = RunProvenance(
        paper_id=paper["paper_id"],
        registry_version=registry.version,
        registry_hash=registry.content_hash,
        silence_table_version=silence_table.version,
        canonical_text_hash=_canonical_text_hash(ct),
        model_a_id=model_a.model_id,
        model_b_id=model_b.model_id,
        prompt_template_hashes=manifest.combined_prompt_hash,
        run_id=f"{args.paper}-{args.phase}-{now}",
        timestamp=now,
    )

    enum = EnumerationResult(
        paper_id=paper["paper_id"],
        constructions=tuple(
            Construction(name=c["name"], quote=c["quote"], cls=c["cls"]) for c in paper["constructions"]
        ),
    )

    assembler = make_assembler(model_a, model_b, registry, manifest, field_limit=args.limit)

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
        print(f"[run_librarian] REVIEW: {exc}")
        return 2

    _write_outputs(result, out_dir)
    _report(result, out_dir, model_a, model_b)
    return 0 if result.specs else 3


def _write_outputs(result, out_dir: Path) -> None:
    for i, (spec, trace) in enumerate(result.specs):
        (out_dir / f"spec_{i}.json").write_text(
            json.dumps(spec.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8"
        )
        (out_dir / f"trace_{i}.json").write_text(
            json.dumps(trace.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8"
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
