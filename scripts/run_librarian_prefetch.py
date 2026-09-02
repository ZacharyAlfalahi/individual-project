#!/usr/bin/env python
"""
B-lever (2026-09-02): batch-prefetch Librarian vendor responses into the shared
disk replay cache at the Anthropic Batches API's 50% discount, so the subsequent
run_librarian / corpus / T5 run REPLAYS them at $0.

Flow (runbook order): ``--dry-run`` (enumerate + hit/miss report, zero network)
-> prefetch (submit misses, poll, write cache) -> the normal run (replays;
residual misses -- e.g. answer-dependent signal-parameter sub-prompts, errored
batch entries -- go live and are counted in the run manifest as usual).

Parity guarantees (the correctness spine, each pinned by tests/unit/test_prefetch.py):
  * PROMPT parity: prompts come from the same ``enumerate_field_prompts`` /
    ``assemble_enumeration_prompt`` assembly the live client uses -- byte-equal.
  * KEY parity: cache writes go through the same ``ResponseCache`` class with the
    same (prompt, model_id, seed=0) key derivation ``_disk_get`` reads; the batch
    ``custom_id`` IS that key.
  * REQUEST parity: batch params mirror ``_AnthropicBackend.generate_split``
    exactly (model, max_tokens, temperature, two content blocks with
    cache_control on the paper-text prefix) -- prefetched and live responses are
    drawn from the same request distribution.
  * PAYLOAD parity: cache values are the same ``{"raw_text", "version"}`` JSON
    blob ``_disk_put`` writes.

Scope: only anthropic-vendor models are batchable here (~90% of paid spend).
The Gemini side has a batches SDK surface (google-genai ``client.batches``) but
is deliberately NOT built -- it is ~10% of spend, its live calls already ride
implicit caching, and its misses simply go live (documented non-blocking
follow-up). Gold-enum-less papers (the Arm-B reject set) prefetch ONLY the
whole-paper enumeration prompt -- their runs route to review at the D20 gate
before any per-field call.

Spend sidecar (B3 decision): this script writes its own schema-v2-shaped
operational sidecar (prefetch_manifest.json). A later fully-prefetched
run_librarian manifest legitimately reads ``model_calls: 0`` -- THIS sidecar
carries the spend (tokens from batch result usage; cost derived downstream from
tokens x a cited rate, never invented here).
"""

from __future__ import annotations

import argparse
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
from agents.librarian.pipeline import load_gold_list, load_prompt_manifest  # noqa: E402
from agents.librarian.pipeline.real_client import (  # noqa: E402
    PromptBuilder,
    assemble_enumeration_prompt,
)
from agents.librarian.registries import load_signal_concept_registry  # noqa: E402
from agents.scientist.researcher.cache import ResponseCache  # noqa: E402
from shared.reporting.run_manifest import build_operational_profile  # noqa: E402
from scripts.run_librarian import (  # noqa: E402
    PAPERS,
    _load_dotenv,
    enumerate_field_prompts,
)
from scripts.run_librarian_corpus import SETS  # noqa: E402
from scripts import run_t5_extraction as t5x  # noqa: E402

_CACHE_SEED = 0                       # == RealModelClient._CACHE_SEED
# Anthropic batch limits: 100k requests / 256MB. Each request carries the full
# paper text (~120KB), so the byte cap binds first; stay well under it.
_MAX_BATCH_BYTES = 80_000_000
_MAX_BATCH_REQUESTS = 10_000


def _model_entries(phase: str) -> list[dict]:
    """The pair's model entries from the same thresholds stack build_clients reads."""
    thresholds = yaml.safe_load(
        (_REPO_ROOT / "docs" / "thresholds.yaml").read_text(encoding="utf-8"))
    stack = thresholds["librarian"]["model_stack"]
    block = stack["phase_d" if phase == "dev" else "phase_f"]
    temperature = stack.get("temperature", 0)
    out = []
    for side in ("model_a", "model_b"):
        e = dict(block[side])
        e["temperature"] = float(temperature)   # match build_client_pair's coercion
        out.append(e)
    return out


# ---------------------------------------------------------------------------
# Job enumeration: a job = one (canonical text, constructions-or-enumeration,
# allowlist) unit -- a corpus paper or one T5 variant.
# ---------------------------------------------------------------------------

def corpus_jobs(set_names: list[str]) -> list[dict]:
    jobs = []
    for set_name in set_names:
        for key in SETS[set_name]:
            paper = PAPERS[key]
            jobs.append({
                "job_id": f"{set_name}:{key}",
                "canonical_text": _REPO_ROOT / paper["canonical_text"],
                "gold_enum": (_REPO_ROOT / paper["gold_enum"]) if paper.get("gold_enum") else None,
                "field_allowlist": None,
            })
    return jobs


def t5_jobs(anchor: str | None, limit: int | None) -> list[dict]:
    jobs = []
    n = 0
    for s in t5x.arm_a_sheets():
        if not s["scoreable"] or not s["frozen"].exists():
            continue
        if anchor is not None and s["anchor"] != anchor:
            continue
        if limit is not None and n >= limit:
            break
        n += 1
        # Degrade the same way run_t5_extraction.run_variants does (integrative-
        # review finding): a sheet whose target cannot be asked is skipped here
        # (its run records a CRASH there) -- it must not abort the whole prefetch.
        try:
            allowlist = set(t5x.fields_arg_for(s["field"]).split(","))
        except ValueError as exc:
            print(f"[prefetch] skipping {s['sheet_id']}: {exc}")
            continue
        paper = PAPERS[t5x.PAPER_KEY_OF_ANCHOR[s["anchor"]]]
        jobs.append({
            "job_id": f"t5:{s['sheet_id']}",
            "canonical_text": s["frozen"],
            "gold_enum": _REPO_ROOT / paper["gold_enum"],
            "field_allowlist": allowlist,
        })
    return jobs


def job_prompts(job: dict, builder: PromptBuilder, manifest) -> list[tuple[str, str, int]]:
    """(prefix, suffix, max_tokens) for every prompt this job's live run would
    issue statically. Gold-less job -> the enumeration prompt only (see module
    docstring)."""
    ct = load_canonical_text(job["canonical_text"])
    if job["gold_enum"] is None:
        prefix, suffix = assemble_enumeration_prompt(builder, ct)
        return [(prefix, suffix, builder.run_max_tokens("enumeration"))]
    enum = load_gold_list(job["gold_enum"])
    return [
        (prefix, suffix, max_tokens)
        for _label, _field, prefix, suffix, max_tokens in enumerate_field_prompts(
            builder, manifest, enum.constructions, ct,
            field_allowlist=job["field_allowlist"])
    ]


# ---------------------------------------------------------------------------
# Planning: dedup, cache-hit check, per-model batchability.
# ---------------------------------------------------------------------------

def build_plan(jobs: list[dict], models: list[dict], cache: ResponseCache) -> dict:
    """{"requests": [{custom_id, model_id, prompt, prefix, suffix, max_tokens,
    temperature}], "counts": {...}} -- misses on batchable (anthropic) models,
    deduped by custom_id (identical prompt bytes across jobs collapse)."""
    registry = load_signal_concept_registry()
    manifest = load_prompt_manifest()
    builder = PromptBuilder.load(registry, manifest)

    requests: dict[str, dict] = {}
    counts = {"jobs": len(jobs), "prompts": 0, "hits": 0,
              "misses_batchable": 0, "misses_live_only": 0}
    for job in jobs:
        for prefix, suffix, max_tokens in job_prompts(job, builder, manifest):
            prompt = prefix + suffix
            counts["prompts"] += len(models)
            for m in models:
                if cache.get(prompt, m["model_id"], _CACHE_SEED) is not None:
                    counts["hits"] += 1
                    continue
                if m["vendor"] != "anthropic":
                    counts["misses_live_only"] += 1
                    continue
                key = ResponseCache.key(prompt, m["model_id"], _CACHE_SEED)
                if key not in requests:
                    requests[key] = {
                        "custom_id": key, "model_id": m["model_id"],
                        "prompt": prompt, "prefix": prefix, "suffix": suffix,
                        "max_tokens": max_tokens, "temperature": m["temperature"],
                    }
                counts["misses_batchable"] += 1
    counts["unique_requests"] = len(requests)
    return {"requests": list(requests.values()), "counts": counts}


def _request_params(r: dict) -> dict:
    """EXACTLY _AnthropicBackend.generate_split's request shape."""
    return {
        "model": r["model_id"],
        "max_tokens": r["max_tokens"],
        "temperature": r["temperature"],
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": r["prefix"],
                 "cache_control": {"type": "ephemeral"}},
                {"type": "text", "text": r["suffix"]},
            ],
        }],
    }


def _chunk(requests: list[dict]) -> list[list[dict]]:
    """Split into batches under the request-count and (approximate) byte caps."""
    chunks, cur, cur_bytes = [], [], 0
    for r in requests:
        size = len(r["prefix"]) + len(r["suffix"]) + 1024
        if cur and (cur_bytes + size > _MAX_BATCH_BYTES or len(cur) >= _MAX_BATCH_REQUESTS):
            chunks.append(cur)
            cur, cur_bytes = [], 0
        cur.append(r)
        cur_bytes += size
    if cur:
        chunks.append(cur)
    return chunks


# ---------------------------------------------------------------------------
# Batch submit / poll / collect. The anthropic client is injected via
# _make_anthropic_client so tests drive a fake.
# ---------------------------------------------------------------------------

def _make_anthropic_client():
    import anthropic  # lazy

    _load_dotenv(_REPO_ROOT / ".env")
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        raise SystemExit("[prefetch] ANTHROPIC_API_KEY is not set (.env or environment)")
    return anthropic.Anthropic(api_key=api_key)


def _record_batch(state_path: Path | None, batch_id: str, status: str, n: int) -> None:
    """Persist a submitted batch id IMMEDIATELY (review finding, 2026-09-02): a
    crash between create and collection would otherwise lose the id while the
    batch is still billed server-side -- a blind re-run would resubmit and pay
    twice. --resume reads this ledger back."""
    if state_path is None:
        return
    state = []
    if state_path.exists():
        state = json.loads(state_path.read_text(encoding="utf-8"))
    state = [b for b in state if b["batch_id"] != batch_id]
    state.append({"batch_id": batch_id, "status": status, "requests": n})
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(json.dumps(state, indent=2), encoding="utf-8")


def open_batch_ids(state_path: Path | None) -> list[str]:
    if state_path is None or not state_path.exists():
        return []
    state = json.loads(state_path.read_text(encoding="utf-8"))
    return [b["batch_id"] for b in state if b["status"] == "open"]


def collect_batch(batch_id: str, by_id: dict, cache: ResponseCache, client,
                  poll_interval_s: float, outcome: dict,
                  state_path: Path | None = None) -> None:
    """Poll one batch to ended and write its succeeded results into the replay
    cache. A custom_id absent from ``by_id`` (already collected on an earlier
    run, so no longer a planned miss) is skipped -- its cache entry exists."""
    while True:
        status = client.messages.batches.retrieve(batch_id)
        if getattr(status, "processing_status", None) == "ended":
            break
        time.sleep(poll_interval_s)
    for entry in client.messages.batches.results(batch_id):
        r = by_id.get(entry.custom_id)
        if r is None:
            continue
        rtype = entry.result.type
        if rtype != "succeeded":
            outcome[rtype if rtype in outcome else "errored"] += 1
            continue
        msg = entry.result.message
        raw_text = "".join(b.text for b in msg.content
                           if getattr(b, "type", None) == "text")
        version = getattr(msg, "model", None)
        # PAYLOAD parity with RealModelClient._disk_put.
        cache.put(r["prompt"], r["model_id"], _CACHE_SEED,
                  json.dumps({"raw_text": raw_text, "version": version},
                             ensure_ascii=False))
        outcome["succeeded"] += 1
        u = getattr(msg, "usage", None)
        if u is not None:
            t = outcome["tokens"].setdefault(r["model_id"], {
                "calls": 0, "prompt": 0, "completion": 0,
                "cache_creation": 0, "cache_read": 0})
            t["calls"] += 1
            t["prompt"] += getattr(u, "input_tokens", 0) or 0
            t["completion"] += getattr(u, "output_tokens", 0) or 0
            t["cache_creation"] += getattr(u, "cache_creation_input_tokens", 0) or 0
            t["cache_read"] += getattr(u, "cache_read_input_tokens", 0) or 0
    _record_batch(state_path, batch_id, "collected", 0)


def resume_open_batches(open_ids: list[str], requests: list[dict],
                        cache: ResponseCache, client, poll_interval_s: float,
                        state_path: Path | None) -> dict:
    """Collect previously-submitted batches (--resume). A batch that cannot be
    retrieved (corrupt/foreign ledger id, transient failure) is reported and left
    OPEN for a later resume -- one bad id never aborts the others (integrative-
    review finding)."""
    outcome = {"batch_ids": list(open_ids), "succeeded": 0, "errored": 0,
               "expired": 0, "canceled": 0, "tokens": {}}
    by_id = {r["custom_id"]: r for r in requests}
    for bid in open_ids:
        print(f"[prefetch] resuming open batch {bid}")
        try:
            collect_batch(bid, by_id, cache, client, poll_interval_s, outcome,
                          state_path=state_path)
        except Exception as exc:  # noqa: BLE001 -- skip-and-continue, ledger stays open
            print(f"[prefetch] could not collect {bid} "
                  f"({type(exc).__name__}: {exc}); left open in the ledger")
    return outcome


def run_batches(requests: list[dict], cache: ResponseCache, client,
                poll_interval_s: float = 30.0, state_path: Path | None = None) -> dict:
    """Submit, poll to ended, write succeeded results into the replay cache.
    Errored/expired/canceled entries are counted and LEFT MISSING (the live run
    fills them, counted normally). Every batch id is persisted to ``state_path``
    the moment it is created. Returns the spend/outcome summary."""
    by_id = {r["custom_id"]: r for r in requests}
    outcome = {"batch_ids": [], "succeeded": 0, "errored": 0, "expired": 0,
               "canceled": 0, "tokens": {}}   # tokens keyed by model_id

    for chunk in _chunk(requests):
        batch = client.messages.batches.create(requests=[
            {"custom_id": r["custom_id"], "params": _request_params(r)} for r in chunk
        ])
        outcome["batch_ids"].append(batch.id)
        _record_batch(state_path, batch.id, "open", len(chunk))
        print(f"[prefetch] submitted batch {batch.id} ({len(chunk)} request(s))")
        collect_batch(batch.id, by_id, cache, client, poll_interval_s, outcome,
                      state_path=state_path)
    return outcome


def write_sidecar(out_dir: Path, phase: str, plan_counts: dict, outcome: dict,
                  wall_clock_seconds: float) -> Path:
    """The B3 spend sidecar: schema-v2-shaped per-model operational profiles."""
    profiles = {
        model_id: build_operational_profile(
            phase=phase,
            model_calls=t["calls"],
            prompt_tokens=t["prompt"],
            completion_tokens=t["completion"],
            cache_creation_tokens=t["cache_creation"],
            cache_read_tokens=t["cache_read"],
            wall_clock_seconds=wall_clock_seconds,
        )
        for model_id, t in outcome["tokens"].items()
    }
    sidecar = {
        "kind": "librarian_batch_prefetch",
        # Machine-readable rate flag (review finding): these tokens are billed at
        # the Batches API rate, NOT the live rate -- cost derivation must key on
        # this, never on the profile alone.
        "billing_rate": "anthropic_batch_50pct_of_live",
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "phase": phase,
        "plan": plan_counts,
        "batches": outcome["batch_ids"],
        "results": {k: outcome[k] for k in ("succeeded", "errored", "expired", "canceled")},
        "operational_profiles": profiles,
        "note": ("Batch-prefetched spend lives HERE; a fully-prefetched "
                 "run_librarian manifest legitimately reads model_calls: 0. "
                 "Cost is derived downstream from tokens x the cited batch rate "
                 "(50% of the live rate)."),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "prefetch_manifest.json"
    path.write_text(json.dumps(sidecar, indent=2), encoding="utf-8")
    return path


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="Batch-prefetch Librarian responses into the disk replay cache.")
    ap.add_argument("--set", action="append", default=[], choices=sorted(SETS),
                    dest="sets", help="corpus set(s) to prefetch (repeatable)")
    ap.add_argument("--t5", action="store_true", help="prefetch the T5 Arm-A variants")
    ap.add_argument("--t5-anchor", default=None, choices=sorted(t5x.PAPER_KEY_OF_ANCHOR))
    ap.add_argument("--t5-limit", type=int, default=None)
    ap.add_argument("--phase", required=True, choices=("dev", "report"),
                    help="model pair (never fake; report is gated on D33 authorization)")
    ap.add_argument("--cache-dir", default="runs/librarian_cache")
    ap.add_argument("--out", default=None,
                    help="sidecar dir (default runs/prefetch)")
    ap.add_argument("--poll-interval-s", type=float, default=30.0)
    ap.add_argument("--dry-run", action="store_true",
                    help="enumerate + hit/miss report only; zero network")
    ap.add_argument("--resume", action="store_true",
                    help="collect any still-open batches from the state ledger "
                         "BEFORE submitting new ones (crash recovery; avoids "
                         "resubmitting work an in-flight batch already billed)")
    args = ap.parse_args(argv)

    if not args.sets and not args.t5:
        ap.error("nothing to prefetch: pass --set and/or --t5")

    jobs = corpus_jobs(args.sets)
    if args.t5:
        jobs += t5_jobs(args.t5_anchor, args.t5_limit)
    models = _model_entries(args.phase)
    cache = ResponseCache(args.cache_dir)

    t0 = time.monotonic()
    plan = build_plan(jobs, models, cache)
    c = plan["counts"]
    print(f"[prefetch] {c['jobs']} job(s), {c['prompts']} prompt-slot(s): "
          f"{c['hits']} cached, {c['misses_batchable']} batchable miss(es) "
          f"({c['unique_requests']} unique), {c['misses_live_only']} live-only "
          f"(non-anthropic vendor)")
    if args.dry_run:
        print("[prefetch] dry run: nothing submitted.")
        return 0

    out_dir = Path(args.out) if args.out else (_REPO_ROOT / "runs" / "prefetch")
    state_path = out_dir / "open_batches.json"

    requests = plan["requests"]
    outcome = None
    if args.resume:
        open_ids = open_batch_ids(state_path)
        if open_ids:
            client = _make_anthropic_client()
            outcome = resume_open_batches(open_ids, requests, cache, client,
                                          args.poll_interval_s, state_path)
            # Whatever the open batches filled is no longer a miss.
            requests = [r for r in requests
                        if cache.get(r["prompt"], r["model_id"], _CACHE_SEED) is None]

    if not requests:
        if c["misses_batchable"] == 0 and c["misses_live_only"] > 0:
            print("[prefetch] nothing batchable: no anthropic-vendor model in this "
                  "phase's pair (misses go to the live run).")
        else:
            print("[prefetch] nothing to submit (all cached).")
        if outcome is None:
            return 0
    else:
        client = _make_anthropic_client()
        resumed = outcome
        outcome = run_batches(requests, cache, client,
                              poll_interval_s=args.poll_interval_s,
                              state_path=state_path)
        if resumed is not None:   # fold the resumed collection into the summary
            outcome["batch_ids"] = resumed["batch_ids"] + outcome["batch_ids"]
            for k in ("succeeded", "errored", "expired", "canceled"):
                outcome[k] += resumed[k]
            for mid, t in resumed["tokens"].items():
                agg = outcome["tokens"].setdefault(mid, {
                    "calls": 0, "prompt": 0, "completion": 0,
                    "cache_creation": 0, "cache_read": 0})
                for k, v in t.items():
                    agg[k] += v

    path = write_sidecar(out_dir, args.phase, c, outcome,
                         round(time.monotonic() - t0, 1))
    print(f"[prefetch] {outcome['succeeded']} cached, "
          f"{outcome['errored'] + outcome['expired'] + outcome['canceled']} left to live run; "
          f"sidecar -> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
