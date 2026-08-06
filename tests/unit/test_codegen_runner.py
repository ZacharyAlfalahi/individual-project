"""WS-C (P1) — runner seams: cache-first single-shot generation, fence
extraction, phase-F resolution from thresholds, redaction, archive layout,
and the typed blocked-client refusal."""

from __future__ import annotations

import json

import pytest

from agents.scientist.researcher.cache import ResponseCache
from evaluation.codegen.runner import (
    BlockedModelClient,
    GenerationBlockedError,
    archive_run,
    build_prompt,
    extract_code,
    generate_once,
    load_phase_f_models,
    prompt_sha256,
    serialise_gold_spec,
)


class CountingStub:
    """A stub client that counts calls — the cache-hit proof."""

    def __init__(self, response: str):
        self.name = "stub"
        self.response = response
        self.calls = 0

    def generate(self, prompt: str, *, seed: int) -> str:
        self.calls += 1
        return self.response


def test_generate_once_is_cache_first(tmp_path):
    cache = ResponseCache(tmp_path)
    stub = CountingStub("```python\nprint('x')\n```")
    first = generate_once("drf", "stub-model", stub, cache)
    second = generate_once("drf", "stub-model", stub, cache)
    assert first == second and stub.calls == 1        # the second call cost nothing


def test_extract_code_requires_exactly_one_fence():
    assert extract_code("```python\nx = 1\n```") == "x = 1"
    assert extract_code("prose only") is None
    assert extract_code("```python\na\n```\n```python\nb\n```") is None
    assert extract_code("```python\n\n```") is None            # empty block
    assert extract_code("```python\nx = 1\n") is None          # unclosed fence


def test_phase_f_models_come_from_thresholds():
    models = load_phase_f_models()
    assert len(models) == 2
    for m in models:
        assert {"vendor", "model_id", "api_key_env"} <= set(m)


def test_blocked_client_raises_typed_error():
    with pytest.raises(GenerationBlockedError, match="dry-run"):
        BlockedModelClient().generate("prompt", seed=0)


def test_prompt_is_deterministic_and_redacted():
    p1, p2 = build_prompt("drf"), build_prompt("drf")
    assert prompt_sha256(p1) == prompt_sha256(p2)
    assert "claimed_headline_metric" not in serialise_gold_spec("drf")
    assert "claimed_headline_metric" not in p1
    # The prompt embeds the schema and the output contract.
    assert "portfolio_return" in p1 and "PANEL_PATH" in p1


def test_archive_layout(tmp_path):
    entry = archive_run(
        "drf", "stub-model", "x = 1\n", "stdout tail", {"verdict": "WONT_RUN"},
        root=tmp_path,
    )
    assert entry.directory.exists()
    assert (entry.directory / "code.py").read_text() == "x = 1\n"
    meta = json.loads((entry.directory / "meta.json").read_text())
    assert meta["verdict"] == "WONT_RUN" and meta["code_sha256"].startswith(entry.code_sha12)
    assert entry.directory.parts[-3:] == ("drf", "stub-model", entry.code_sha12)
