"""
model_client.py — the LLM client boundary for the Auditor explainer (step 19, §11).

The explainer runs strictly AFTER the deterministic verdict and has ZERO gating
power; its output is checked number-by-number by `numeric_verifier`. This module is
the thin model boundary: a minimal `ExplainerClient` protocol (prompt in → text
out), a `FakeExplainerClient` for offline deterministic tests, and a
`LiveExplainerClient` that REUSES the Librarian's vendor backends
(`agents.librarian.pipeline.real_client.make_backend`) so no new SDK code is written.

All the Librarian backends emit JSON, so the explainer asks the model for a
single-field object `{"explanation": "<prose>"}` — vendor-agnostic, so the free
Gemini-lite dev model and Claude alike work through the same path.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Protocol, runtime_checkable


@dataclass(frozen=True)
class ExplainerResponse:
    """One model response: the raw text (expected to be a JSON object carrying the
    prose) and the vendor's returned model-version string (for provenance)."""

    text: str
    model_version: str | None = None


@runtime_checkable
class ExplainerClient(Protocol):
    def generate(self, prompt: str) -> ExplainerResponse:
        ...


class FakeExplainerClient:
    """Offline, deterministic client for tests. `scripted` is either a fixed string
    returned verbatim, or a callable `(prompt, attempt_index) -> str` for
    attempt-dependent behaviour (e.g. to exercise the retry → fallback path)."""

    def __init__(
        self,
        scripted: str | Callable[[str, int], str],
        *,
        model_version: str = "fake-explainer",
    ) -> None:
        self._scripted = scripted
        self.model_version = model_version
        self.calls = 0

    def generate(self, prompt: str) -> ExplainerResponse:
        if callable(self._scripted):
            text = self._scripted(prompt, self.calls)
        else:
            text = self._scripted
        self.calls += 1
        return ExplainerResponse(text=text, model_version=self.model_version)


class LiveExplainerClient:
    """Reuses a Librarian vendor backend (`make_backend`) for a single free-text
    call. Constructed from a pre-registered `ExplainerModelConfig` + the API key."""

    def __init__(
        self,
        vendor: str,
        model_id: str,
        api_key: str,
        *,
        temperature: float = 0.0,
        max_output_tokens: int = 1024,
    ) -> None:
        from agents.librarian.pipeline.real_client import make_backend

        self.model_id = model_id
        self._max_output_tokens = max_output_tokens
        self._backend = make_backend(vendor, model_id, api_key, temperature)
        # WS-8 (§4.7) mechanical operational counters (mirrors RealModelClient).
        self.model_calls = 0
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.total_retries = 0

    def generate(self, prompt: str) -> ExplainerResponse:
        text, version = self._backend.generate(prompt, self._max_output_tokens)
        self.model_calls += 1
        u = getattr(self._backend, "last_usage", None)
        if u:
            if u.get("prompt") is not None:
                self.total_prompt_tokens += u["prompt"]
            if u.get("completion") is not None:
                self.total_completion_tokens += u["completion"]
        return ExplainerResponse(text=text, model_version=version)

    def operational_usage(self) -> dict:
        """Mechanical token/call/retry totals (WS-8 / §4.7), mirroring RealModelClient."""
        return {
            "model_calls": self.model_calls,
            "prompt_tokens": self.total_prompt_tokens,
            "completion_tokens": self.total_completion_tokens,
            "retries": self.total_retries,
        }


def build_live_explainer_client(
    phase: str,
    api_keys: dict[str, str],
    *,
    thresholds_path=None,
) -> LiveExplainerClient:
    """Build a LiveExplainerClient for `phase` ∈ {'dev','reported'} from the
    pre-registered `auditor.explainer` block + the API keys map (env-name -> key)."""
    from ..thresholds import load_explainer_config

    cfg = load_explainer_config(phase, thresholds_path)
    key = api_keys.get(cfg.api_key_env, "")
    if not key:
        raise ValueError(
            f"missing API key: env var {cfg.api_key_env!r} is unset for the "
            f"{phase!r} explainer model ({cfg.vendor}/{cfg.model_id})"
        )
    return LiveExplainerClient(
        cfg.vendor, cfg.model_id, key,
        temperature=cfg.temperature, max_output_tokens=cfg.max_output_tokens,
    )
