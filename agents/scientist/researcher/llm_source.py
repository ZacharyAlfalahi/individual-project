"""Rung 3 — the generative LLM researcher (spec §8.1/§8.2). Build rung 3 LAST. The generative call
receives ONLY the wall allow-list (INVARIANT 1): the context is assembled by `build_context`
(already magnitude-free), so no realised performance statistic can reach the model. Structured
output is decoded at the ONE boundary (decode_proposal); invalid/duplicate proposals are counted,
never regenerated (§8.2 / prohibition 6) — the generation loop in sources.py owns that.

The model client is INJECTED (like rung 2's embedder). The real D4 phase_d clients (Gemini
3.1-flash-lite + Mistral-small) are DEFERRED — `DeferredModelClient` raises until wired, so the
rung's structure + cache + parsing are built and tested with a stub, gated on model access.
"""

from __future__ import annotations

import json
from typing import Protocol

from .cache import ResponseCache
from .context_builder import build_context


class ModelClient(Protocol):
    name: str

    def generate(self, prompt: str, *, seed: int) -> str: ...


class DeferredModelClient:
    """The real D4 phase_d client is not wired yet (needs model access + SKU/cost authorization).
    Raises rather than silently returning nothing, so a run cannot falsely claim rung 3 executed."""
    name = "deferred"

    def generate(self, prompt: str, *, seed: int) -> str:
        raise RuntimeError(
            "llm_researcher (rung 3) model access not configured — inject a real D4 client "
            "(Gemini 3.1-flash-lite / Mistral-small) or a stub. Deferred by design.")


def build_prompt(context: dict, m: int) -> str:
    """The generative prompt — the magnitude-free context (from the wall allow-list) plus the
    instruction to emit exactly m proposals as JSON. `context` is already magnitude-free."""
    return (
        "You are proposing audit-clean EXTENSIONS of a corrected corporate-bond strategy. Using "
        "ONLY the context below, emit exactly "
        f"{m} proposals as a JSON array; each item = "
        '{"mechanism_ref","template_ref","conditioning_variable","conditioning_lag_months",'
        '"interaction_form","rationale","prediction"}. Never propose a correction or a bias toggle.'
        "\n\nCONTEXT:\n" + json.dumps(context, ensure_ascii=False, sort_keys=True))


def _to_raw(item, *, case, seed, model, prompt_version, library_version, generated_at, i) -> dict:
    if not isinstance(item, dict):
        return {"__malformed__": True}                     # -> decode_proposal rejects (counted)
    cv = item.get("conditioning_variable")
    return {
        "proposal_id": f"prop_llm_{seed}_{i}", "case_id": case.case_id,
        "parent_strategy_id": case.strategy_id, "mechanism_ref": item.get("mechanism_ref"),
        "template_ref": item.get("template_ref"), "rationale": item.get("rationale", ""),
        "prediction": item.get("prediction", ""),
        "config_delta": {"conditioning_variable": cv,
                         "conditioning_lag_months": item.get("conditioning_lag_months"),
                         "interaction_form": item.get("interaction_form")},
        "required_inputs": [cv] if cv else [],
        "generation": {"source": "llm_researcher", "seed": seed, "model": model,
                       "prompt_version": prompt_version, "library_version": library_version,
                       "generated_at": generated_at},
    }


class LLMResearcherSource:
    source_name = "llm_researcher"

    def __init__(self, client: ModelClient, *, cache: ResponseCache | None = None):
        self.client = client
        self.cache = cache

    def candidates(self, case, eligible_results, library, *, seed, m, model, prompt_version,
                   generated_at) -> list[dict]:
        context = build_context(case, eligible_results, library)      # the wall — magnitude-free
        prompt = build_prompt(context, m)
        response = self.cache.get(prompt, model, seed) if self.cache else None
        if response is None:
            response = self.client.generate(prompt, seed=seed)
            if self.cache:
                self.cache.put(prompt, model, seed, response)          # persist every seed (R5)
        try:
            items = json.loads(response)
        except (json.JSONDecodeError, TypeError):
            return []                                                  # unparseable -> 0 candidates
        if not isinstance(items, list):
            return []
        return [_to_raw(it, case=case, seed=seed, model=model, prompt_version=prompt_version,
                        library_version=library.version_hash, generated_at=generated_at, i=i)
                for i, it in enumerate(items[:m])]
