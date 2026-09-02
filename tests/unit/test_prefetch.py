"""B-lever (2026-09-02): batch prefetch into the disk replay cache.

The correctness spine, per the build plan:
  * ENUMERATOR-vs-LIVE parity (byte-level): every vendor call a live run makes
    is enumerated, and nothing more -- proven by driving the real assembler with
    prompt-recording backends against the real BBW text + gold enumeration.
  * KEY/PAYLOAD parity: a prefetch-written cache entry replays in the live
    client with ZERO backend calls.
  * Batch mechanics: submit/poll/collect against a mocked anthropic client;
    errored entries stay missing; chunking; the B3 spend sidecar shape.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT))

from agents.librarian.config import load_canonical_text                   # noqa: E402
from agents.librarian.pipeline import load_gold_list, load_prompt_manifest  # noqa: E402
from agents.librarian.pipeline import real_client as rc                   # noqa: E402
from agents.librarian.pipeline.emission import RunProvenance              # noqa: E402
from agents.librarian.pipeline.model_client import FieldQuery             # noqa: E402
from agents.librarian.registries import load_signal_concept_registry      # noqa: E402
from agents.scientist.researcher.cache import ResponseCache               # noqa: E402
from scripts import run_librarian_prefetch as pf                          # noqa: E402
from scripts.run_librarian import enumerate_field_prompts, make_assembler  # noqa: E402

_BBW_CT = _REPO_ROOT / "evaluation" / "canonical_texts" / "bbw_2019.frozen.yaml"
_BBW_ENUM = _REPO_ROOT / "evaluation" / "gold_specs" / "enum_bbw_2019.yaml"


class _RecordingBackend:
    """Answers everything with one universal JSON blob and records every
    (prompt, max_tokens) vendor call -- the live-side half of the parity proof."""

    def __init__(self):
        # value + concept_id both set so every kind parses: enum reads "value",
        # signal_ref reads "concept_id"; the quote locates in the real BBW text.
        self._text = json.dumps({"answered": True, "value": "var_5pct",
                                 "concept_id": "var_5pct", "quote": "the 5% VaR"})
        self.seen: list[tuple[str, int]] = []

    def generate(self, prompt, max_output_tokens):
        self.seen.append((prompt, max_output_tokens))
        return self._text, "rec-v1"


@pytest.fixture(scope="module")
def registry():
    return load_signal_concept_registry()


@pytest.fixture(scope="module")
def manifest():
    return load_prompt_manifest()


@pytest.fixture(scope="module")
def builder(registry, manifest):
    return rc.PromptBuilder.load(registry, manifest)


# ---------------------------------------------------------------------------
# Parity spine.
# ---------------------------------------------------------------------------

def test_enumerator_covers_every_live_vendor_call_byte_exactly(
        monkeypatch, registry, manifest, builder):
    """Drive the REAL assembler (real BBW text, gold construction) with
    recording backends: the enumerator's (prompt bytes, max_tokens) set must
    equal the set of vendor calls EXACTLY -- nothing missed (prefetch covers the
    run), nothing extra (prefetch never pays for a prompt the run won't make),
    each prompt issued exactly once per model (in-memory dedup holds)."""
    back_a, back_b = _RecordingBackend(), _RecordingBackend()
    monkeypatch.setitem(rc._BACKENDS, "stub-a",
                        lambda model_id, api_key, temperature: back_a)
    monkeypatch.setitem(rc._BACKENDS, "stub-b",
                        lambda model_id, api_key, temperature: back_b)
    model_a = rc.RealModelClient(model_id="stub-a", vendor="stub-a", api_key="k",
                                 builder=builder)
    model_b = rc.RealModelClient(model_id="stub-b", vendor="stub-b", api_key="k",
                                 builder=builder)

    ct = load_canonical_text(_BBW_CT)
    enum = load_gold_list(_BBW_ENUM)
    assemble = make_assembler(model_a, model_b, registry, manifest)
    prov = RunProvenance(paper_id="BBW_2019", registry_version="t", registry_hash="t",
                         silence_table_version="t", canonical_text_hash="t",
                         model_a_id="stub-a", model_b_id="stub-b")
    for construction in enum.constructions:
        assemble(construction, ct, prov)

    enumerated = {
        (prefix + suffix, max_tokens)
        for _label, _field, prefix, suffix, max_tokens in enumerate_field_prompts(
            builder, manifest, enum.constructions, ct)
    }
    for back in (back_a, back_b):
        assert set(back.seen) == enumerated
        assert len(back.seen) == len(enumerated)      # each exactly once per model


def test_enumerator_parity_multi_construction_with_disk_cache(
        monkeypatch, registry, manifest, builder, tmp_path):
    """The multi-construction corpus shape (review finding): label-independent
    prompts repeat byte-identically across constructions; with the disk replay
    cache the live run pays each UNIQUE prompt once -- and the enumerator's
    unique set must equal exactly that vendor-call set."""
    back_a, back_b = _RecordingBackend(), _RecordingBackend()
    monkeypatch.setitem(rc._BACKENDS, "stub-a",
                        lambda model_id, api_key, temperature: back_a)
    monkeypatch.setitem(rc._BACKENDS, "stub-b",
                        lambda model_id, api_key, temperature: back_b)
    cache_dir = tmp_path / "cache"
    model_a = rc.RealModelClient(model_id="stub-a", vendor="stub-a", api_key="k",
                                 builder=builder, cache_dir=cache_dir)
    model_b = rc.RealModelClient(model_id="stub-b", vendor="stub-b", api_key="k",
                                 builder=builder, cache_dir=cache_dir)

    from agents.librarian.pipeline import Construction
    ct = load_canonical_text(_BBW_CT)
    c0 = load_gold_list(_BBW_ENUM).constructions[0]
    constructions = [c0, Construction(name="Second Construction (SC)",
                                      quote=c0.quote, cls=c0.cls)]
    assemble = make_assembler(model_a, model_b, registry, manifest)
    prov = RunProvenance(paper_id="BBW_2019", registry_version="t", registry_hash="t",
                         silence_table_version="t", canonical_text_hash="t",
                         model_a_id="stub-a", model_b_id="stub-b")
    for construction in constructions:
        assemble(construction, ct, prov)

    enumerated_unique = {
        (prefix + suffix, max_tokens)
        for _l, _f, prefix, suffix, max_tokens in enumerate_field_prompts(
            builder, manifest, constructions, ct)
    }
    for back in (back_a, back_b):
        assert set(back.seen) == enumerated_unique
        assert len(back.seen) == len(enumerated_unique)   # dedup: each paid once


def test_allowlist_byte_parity_with_targeted_live_run(
        monkeypatch, registry, manifest, builder):
    """T5 paid path (review finding): a targeted (field_allowlist) live run's
    vendor calls equal the enumerator's targeted set, byte-exactly."""
    back_a, back_b = _RecordingBackend(), _RecordingBackend()
    monkeypatch.setitem(rc._BACKENDS, "stub-a",
                        lambda model_id, api_key, temperature: back_a)
    monkeypatch.setitem(rc._BACKENDS, "stub-b",
                        lambda model_id, api_key, temperature: back_b)
    model_a = rc.RealModelClient(model_id="stub-a", vendor="stub-a", api_key="k",
                                 builder=builder)
    model_b = rc.RealModelClient(model_id="stub-b", vendor="stub-b", api_key="k",
                                 builder=builder)

    allow = {"long_leg", "weighting_scheme"}
    ct = load_canonical_text(_BBW_CT)
    enum = load_gold_list(_BBW_ENUM)
    assemble = make_assembler(model_a, model_b, registry, manifest,
                              field_allowlist=allow)
    prov = RunProvenance(paper_id="BBW_2019", registry_version="t", registry_hash="t",
                         silence_table_version="t", canonical_text_hash="t",
                         model_a_id="stub-a", model_b_id="stub-b")
    for construction in enum.constructions:
        assemble(construction, ct, prov)

    enumerated = {
        (prefix + suffix, max_tokens)
        for _l, _f, prefix, suffix, max_tokens in enumerate_field_prompts(
            builder, manifest, enum.constructions, ct, field_allowlist=allow)
    }
    for back in (back_a, back_b):
        assert set(back.seen) == enumerated
        assert len(back.seen) == len(enumerated)


def test_registry_census_no_parameterised_concepts(registry):
    """Census pin (2026-09-02): NO registry concept carries parameters, so the
    answer-dependent signal-parameter sub-prompts the enumerator excludes by
    design are EMPTY in practice -- static prefetch coverage of gold-enum papers
    is 100%. If a parameterised concept ever lands, this fires and the coverage
    note in run_librarian_prefetch's docstring must be revisited (the design
    stays sound: params become counted residual live calls)."""
    ids = list(registry.ids())
    assert ids
    assert [cid for cid in ids if registry.parameter_schema(cid)] == []


def test_allowlist_parity_with_targeted_run(registry, manifest, builder):
    """The enumerator honours field_allowlist the way _capped does: a targeted
    T5-style enumeration is a strict subset and keeps the always-run set."""
    ct = load_canonical_text(_BBW_CT)
    enum = load_gold_list(_BBW_ENUM)
    full = {f for _l, f, _p, _s, _m in enumerate_field_prompts(
        builder, manifest, enum.constructions, ct)}
    targeted = {f for _l, f, _p, _s, _m in enumerate_field_prompts(
        builder, manifest, enum.constructions, ct,
        field_allowlist={"long_leg", "weighting_scheme"})}
    assert targeted < full
    assert {"long_leg", "weighting_scheme", "sort_signal", "method_summary",
            "asset_class", "combiner", "sample_start"} <= targeted
    assert "holding_period" not in targeted
    assert "control_axis" not in targeted             # runs only when listed


def test_prefetched_entry_replays_in_live_client(monkeypatch, builder, tmp_path):
    """KEY + PAYLOAD parity end-to-end: a cache entry written the prefetcher's
    way is a replay HIT for the live client -- zero backend calls."""
    class _Boom:
        calls = 0

        def generate(self, prompt, max_output_tokens):  # pragma: no cover
            raise AssertionError("live vendor call despite prefetched cache entry")

    monkeypatch.setitem(rc._BACKENDS, "boom",
                        lambda model_id, api_key, temperature: _Boom())
    cache_dir = tmp_path / "cache"
    client = rc.RealModelClient(model_id="claude-x", vendor="boom", api_key="k",
                                builder=builder, cache_dir=cache_dir)

    ct = load_canonical_text(_BBW_CT)
    q = FieldQuery("weighting_scheme", "enum")
    label = "the strategy described in this paper"    # the client's default label
    prefix, suffix = rc.assemble_field_prompt(builder, q, label, ct)

    raw = json.dumps({"field": "weighting_scheme", "answered": True,
                      "value": "value", "quote": "value-weighted average"})
    ResponseCache(cache_dir).put(
        prefix + suffix, "claude-x", pf._CACHE_SEED,
        json.dumps({"raw_text": raw, "version": "claude-x-batch"}, ensure_ascii=False))

    ans = client.answer(q, ct)
    assert ans.answered is True
    assert ans.model_id == "claude-x-batch"           # replayed version stamp
    assert client.model_calls == 0                    # WS-8: replay is not a paid call


# ---------------------------------------------------------------------------
# Planning.
# ---------------------------------------------------------------------------

def _bbw_job():
    return {"job_id": "t:bbw", "canonical_text": _BBW_CT, "gold_enum": _BBW_ENUM,
            "field_allowlist": None}


def _models():
    return [{"model_id": "claude-x", "vendor": "anthropic", "temperature": 0},
            {"model_id": "gem-y", "vendor": "gemini", "temperature": 0}]


def test_build_plan_dedup_vendor_split_and_hits(tmp_path):
    cache = ResponseCache(tmp_path / "cache")
    plan = pf.build_plan([_bbw_job(), _bbw_job()], _models(), cache)
    c = plan["counts"]
    n = c["unique_requests"]
    assert n > 30                                     # the full per-construction set
    assert c["misses_batchable"] == 2 * n             # duplicate job collapses
    assert c["misses_live_only"] == 2 * n             # gemini: never batched
    assert c["hits"] == 0
    assert all(r["model_id"] == "claude-x" for r in plan["requests"])

    # Pre-seed ONE anthropic entry -> both duplicate jobs hit it.
    r0 = plan["requests"][0]
    cache.put(r0["prompt"], "claude-x", pf._CACHE_SEED, "seeded")
    plan2 = pf.build_plan([_bbw_job(), _bbw_job()], _models(), cache)
    assert plan2["counts"]["hits"] == 2
    assert plan2["counts"]["unique_requests"] == n - 1


def test_custom_id_is_the_replay_cache_key(tmp_path):
    plan = pf.build_plan([_bbw_job()], _models()[:1], ResponseCache(tmp_path / "cache"))
    r = plan["requests"][0]
    assert r["custom_id"] == ResponseCache.key(r["prompt"], r["model_id"], pf._CACHE_SEED)


# ---------------------------------------------------------------------------
# Batch mechanics (mocked anthropic client).
# ---------------------------------------------------------------------------

def _mk_request(prompt_prefix, prompt_suffix, model_id="claude-x"):
    prompt = prompt_prefix + prompt_suffix
    return {"custom_id": ResponseCache.key(prompt, model_id, pf._CACHE_SEED),
            "model_id": model_id, "prompt": prompt, "prefix": prompt_prefix,
            "suffix": prompt_suffix, "max_tokens": 256, "temperature": 0}


class _FakeBatchClient:
    """messages.batches.{create,retrieve,results} triple; scripts one result per
    custom_id: ("succeeded", text) | ("errored", None). ``preloaded`` maps a
    pre-existing batch_id -> [custom_id] (a batch submitted by a crashed prior
    session); ``retrieve_error`` makes retrieve() raise (crash-window tests)."""

    def __init__(self, script, preloaded=None, retrieve_error=None):
        self._script = script
        self._preloaded = preloaded or {}
        self._retrieve_error = retrieve_error
        self.created: list[list[dict]] = []
        outer = self

        class _Batches:
            def create(self, requests):
                outer.created.append(requests)
                return SimpleNamespace(id=f"batch_{len(outer.created)}")

            def retrieve(self, batch_id):
                if outer._retrieve_error is not None:
                    raise outer._retrieve_error
                return SimpleNamespace(processing_status="ended")

            def results(self, batch_id):
                if batch_id in outer._preloaded:
                    ids = outer._preloaded[batch_id]
                else:
                    ids = [r["custom_id"]
                           for r in outer.created[int(batch_id.split("_")[1]) - 1]]
                for cid in ids:
                    kind, text = outer._script[cid]
                    if kind != "succeeded":
                        yield SimpleNamespace(custom_id=cid,
                                              result=SimpleNamespace(type=kind))
                        continue
                    msg = SimpleNamespace(
                        content=[SimpleNamespace(type="text", text=text)],
                        model="claude-x-2026-01",
                        usage=SimpleNamespace(input_tokens=100, output_tokens=20,
                                              cache_creation_input_tokens=5,
                                              cache_read_input_tokens=50))
                    yield SimpleNamespace(custom_id=cid,
                                          result=SimpleNamespace(type="succeeded",
                                                                 message=msg))

        self.messages = SimpleNamespace(batches=_Batches())


def test_run_batches_writes_succeeded_and_leaves_errored_missing(tmp_path):
    cache = ResponseCache(tmp_path / "cache")
    r1 = _mk_request("P1", "S1")
    r2 = _mk_request("P2", "S2")
    client = _FakeBatchClient({r1["custom_id"]: ("succeeded", '{"answered": true}'),
                               r2["custom_id"]: ("errored", None)})
    outcome = pf.run_batches([r1, r2], cache, client, poll_interval_s=0)

    hit = cache.get(r1["prompt"], "claude-x", pf._CACHE_SEED)
    assert json.loads(hit) == {"raw_text": '{"answered": true}',
                               "version": "claude-x-2026-01"}   # _disk_put payload shape
    assert cache.get(r2["prompt"], "claude-x", pf._CACHE_SEED) is None
    assert (outcome["succeeded"], outcome["errored"]) == (1, 1)
    assert outcome["tokens"]["claude-x"] == {"calls": 1, "prompt": 100, "completion": 20,
                                             "cache_creation": 5, "cache_read": 50}
    # Request params mirror generate_split: 2 blocks, cache_control on the prefix.
    params = client.created[0][0]["params"]
    blocks = params["messages"][0]["content"]
    assert blocks[0]["cache_control"] == {"type": "ephemeral"}
    assert (blocks[0]["text"], blocks[1]["text"]) == ("P1", "S1")
    assert params["max_tokens"] == 256 and params["temperature"] == 0


def test_run_batches_chunks_by_byte_cap(tmp_path, monkeypatch):
    monkeypatch.setattr(pf, "_MAX_BATCH_BYTES", 2_000)
    reqs = [_mk_request(f"P{i}" * 200, f"S{i}") for i in range(3)]
    client = _FakeBatchClient({r["custom_id"]: ("succeeded", "{}") for r in reqs})
    outcome = pf.run_batches(reqs, ResponseCache(tmp_path / "c"), client,
                             poll_interval_s=0)
    assert len(client.created) > 1                    # split into multiple batches
    assert outcome["succeeded"] == 3                  # all collected across chunks


def test_sidecar_shape(tmp_path):
    outcome = {"batch_ids": ["batch_1"], "succeeded": 2, "errored": 1, "expired": 0,
               "canceled": 0,
               "tokens": {"claude-x": {"calls": 2, "prompt": 200, "completion": 40,
                                       "cache_creation": 10, "cache_read": 100}}}
    path = pf.write_sidecar(tmp_path, "report", {"jobs": 1}, outcome, 12.5)
    sidecar = json.loads(path.read_text(encoding="utf-8"))
    assert sidecar["kind"] == "librarian_batch_prefetch"
    prof = sidecar["operational_profiles"]["claude-x"]
    assert prof["model_calls"] == 2
    assert prof["tokens"] == {"prompt": 200, "completion": 40,
                              "cache_creation": 10, "cache_read": 100}
    assert prof["cost_usd"] is None                   # derived downstream, never invented
    assert "model_calls: 0" in sidecar["note"]


def test_batch_id_persisted_before_collection_survives_crash(tmp_path):
    """The crash-window ledger (review finding): create succeeds, retrieve
    crashes -> the batch id is already ON DISK marked open, so --resume can
    collect it instead of resubmitting (double-billing)."""
    r1 = _mk_request("P1", "S1")
    client = _FakeBatchClient({r1["custom_id"]: ("succeeded", "{}")},
                              retrieve_error=ConnectionError("mid-poll crash"))
    state = tmp_path / "open_batches.json"
    with pytest.raises(ConnectionError):
        pf.run_batches([r1], ResponseCache(tmp_path / "c"), client,
                       poll_interval_s=0, state_path=state)
    ledger = json.loads(state.read_text(encoding="utf-8"))
    assert ledger == [{"batch_id": "batch_1", "status": "open", "requests": 1}]


def test_collect_batch_resumes_and_marks_collected(tmp_path):
    """--resume mechanics: a preloaded (crashed-session) batch is collected into
    the cache without any new create; custom_ids no longer in the plan are
    skipped harmlessly; the ledger flips to collected."""
    cache = ResponseCache(tmp_path / "c")
    r1 = _mk_request("P1", "S1")
    state = tmp_path / "open_batches.json"
    state.write_text(json.dumps(
        [{"batch_id": "b_prev", "status": "open", "requests": 2}]), encoding="utf-8")
    client = _FakeBatchClient(
        {r1["custom_id"]: ("succeeded", '{"answered": true}'),
         "gone_from_plan": ("succeeded", "{}")},
        preloaded={"b_prev": [r1["custom_id"], "gone_from_plan"]})

    assert pf.open_batch_ids(state) == ["b_prev"]
    outcome = {"batch_ids": ["b_prev"], "succeeded": 0, "errored": 0,
               "expired": 0, "canceled": 0, "tokens": {}}
    pf.collect_batch("b_prev", {r1["custom_id"]: r1}, cache, client,
                     poll_interval_s=0, outcome=outcome, state_path=state)

    assert client.created == []                       # no new submission
    assert cache.get(r1["prompt"], "claude-x", pf._CACHE_SEED) is not None
    assert outcome["succeeded"] == 1                  # the skipped id is not counted
    assert pf.open_batch_ids(state) == []             # marked collected


def test_run_batches_marks_ledger_collected_on_success(tmp_path):
    r1 = _mk_request("P1", "S1")
    client = _FakeBatchClient({r1["custom_id"]: ("succeeded", "{}")})
    state = tmp_path / "open_batches.json"
    pf.run_batches([r1], ResponseCache(tmp_path / "c"), client,
                   poll_interval_s=0, state_path=state)
    assert pf.open_batch_ids(state) == []


# ---------------------------------------------------------------------------
# Job enumeration.
# ---------------------------------------------------------------------------

def test_corpus_jobs_gold_and_goldless(tmp_path):
    jobs = pf.corpus_jobs(["anchors"])
    assert all(j["gold_enum"] is not None for j in jobs)
    rejects = pf.corpus_jobs(["rejects"])
    assert rejects and all(j["gold_enum"] is None for j in rejects)


@pytest.mark.skipif(not (_REPO_ROOT / "evaluation" / "adversarial" / "frozen").exists(),
                    reason="machine-local frozen variants not present")
def test_t5_jobs_carry_target_allowlists():
    jobs = pf.t5_jobs(None, None)
    assert len(jobs) > 200                            # the scoreable Arm-A set
    by_id = {j["job_id"]: j for j in jobs}
    compound = by_id["t5:drf__long_leg+weighting_scheme__c3_paraphrase"]
    assert compound["field_allowlist"] == {"long_leg", "weighting_scheme"}
    assert all(j["gold_enum"] is not None for j in jobs)
