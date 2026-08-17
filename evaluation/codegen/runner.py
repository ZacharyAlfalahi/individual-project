"""P1 codegen ablation (WS-C) — generation runner (build-only).

One-shot-per-(model, strategy) generation from the gold StrategySpec, cached
content-addressed, executed in the sandbox, scored by the rung-3/4 comparator.
Generation is BLOCKED until Phase-F credentials + the frozen mini-contract
are in place (`docs/extensions/contracts/p1_codegen_ablation.md`); the
default client refuses loudly with a typed message, so the only runnable
mode is `--dry-run` (prompt hashes, sandbox self-check, oracle-vs-oracle
sanity — zero spend).

Attempts policy (I3): ONE generation per (model, strategy), temperature 0,
seed 0, count-not-retry — a failed run is a recorded datum, never re-rolled.
Cache (I8): `ResponseCache` (content-addressed by prompt/model/seed,
append-only, atomic) reused verbatim from the Scientist.

Prompt hygiene (C3): the serialised gold spec is REDACTED of the paper's
claimed headline metric before prompting — the model receives the strategy's
construction, never the number it is supposed to reproduce.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Protocol

import yaml

from agents.scientist.researcher.cache import ResponseCache

_REPO_ROOT = Path(__file__).resolve().parents[2]
_PROMPT_DIR = Path(__file__).resolve().parent / "prompts"
_THRESHOLDS = _REPO_ROOT / "docs" / "thresholds.yaml"
_CONTRACT = _REPO_ROOT / "docs" / "extensions" / "contracts" / "p1_codegen_ablation.md"

STRATEGIES: tuple[str, ...] = ("drf", "str", "mom6", "crf", "lrf")

# Keys removed from the serialised gold spec before it enters a prompt: the
# paper's own performance numbers. Recorded verbatim in the mini-contract.
REDACTED_KEYS: tuple[str, ...] = ("claimed_headline_metric",)


class GenerationBlockedError(RuntimeError):
    """Raised by the default client: generation requires Phase-F credentials
    plus the frozen, signed mini-contract (E3)."""


class ModelClient(Protocol):
    name: str

    def generate(self, prompt: str, *, seed: int) -> str: ...


class BlockedModelClient:
    """The default client. Refuses every call with the typed blocker."""

    def __init__(self, name: str = "blocked"):
        self.name = name

    def generate(self, prompt: str, *, seed: int) -> str:
        raise GenerationBlockedError(
            "P1 generation is blocked: Phase-F credentials and the signed frozen "
            "mini-contract are required before any generation call (E3/I5). "
            "Run with --dry-run."
        )


def load_phase_f_models(thresholds_path: Path | None = None) -> list[dict]:
    """The Phase-F pair, read from thresholds at runtime — never hardcoded."""
    doc = yaml.safe_load((thresholds_path or _THRESHOLDS).read_text(encoding="utf-8"))
    try:
        stack = doc["librarian"]["model_stack"]["phase_f"]
    except (KeyError, TypeError) as exc:
        raise KeyError("thresholds.yaml has no librarian.model_stack.phase_f block") from exc
    return [dict(stack["model_a"]), dict(stack["model_b"])]


def _redact(node: object) -> object:
    if isinstance(node, dict):
        return {k: _redact(v) for k, v in node.items() if k not in REDACTED_KEYS}
    if isinstance(node, list):
        return [_redact(v) for v in node]
    return node


def serialise_gold_spec(strategy: str) -> str:
    """The gold StrategySpec as redacted, deterministic JSON."""
    from evaluation.gold_specs.gold_loader import load_gold_spec

    spec = load_gold_spec(strategy)
    return json.dumps(_redact(spec.to_dict()), indent=2, sort_keys=True, default=str)


def build_prompt(strategy: str) -> str:
    template = (_PROMPT_DIR / "template.md").read_text(encoding="utf-8")
    schema = (_PROMPT_DIR / "panel_schema.md").read_text(encoding="utf-8")
    contract = (_PROMPT_DIR / "output_contract.md").read_text(encoding="utf-8")
    return (
        template
        .replace("{{GOLD_SPEC_JSON}}", serialise_gold_spec(strategy))
        .replace("{{PANEL_SCHEMA}}", schema)
        .replace("{{OUTPUT_CONTRACT}}", contract)
    )


def prompt_sha256(prompt: str) -> str:
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()


def prompt_asset_hashes() -> dict[str, str]:
    """sha256 of every prompt asset — the freeze block's inputs (I5)."""
    return {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(_PROMPT_DIR.glob("*.md"))
    }


def generate_once(
    strategy: str,
    model_id: str,
    client: ModelClient,
    cache: ResponseCache,
    *,
    seed: int = 0,
) -> str:
    """Cache-first single-shot generation. A cache hit performs ZERO client
    calls; a miss performs exactly one and persists it immutably — re-running
    the ablation can never spend twice on the same (prompt, model, seed)."""
    prompt = build_prompt(strategy)
    cached = cache.get(prompt, model_id, seed)
    if cached is not None:
        return cached
    response = client.generate(prompt, seed=seed)
    cache.put(prompt, model_id, seed, response)
    return response


def extract_code(response: str) -> str | None:
    """Exactly one fenced Python block; anything else is WONT_RUN
    (malformed_response) — never repaired, per count-not-retry (I3)."""
    marker = "```python"
    if response.count(marker) != 1:
        return None
    _, _, rest = response.partition(marker)
    code, closed, _ = rest.partition("```")
    if not closed:
        return None
    code = code.strip("\n")
    return code if code.strip() else None


@dataclass(frozen=True)
class ArchiveEntry:
    strategy: str
    model_id: str
    code_sha12: str
    directory: Path


def archive_run(
    strategy: str,
    model_id: str,
    code: str,
    stdout_tail: str,
    meta: dict,
    *,
    root: Path | None = None,
) -> ArchiveEntry:
    """Content-addressed archive of one generation run. Code, stdout and meta
    are small text artefacts (committed); return-series CSVs go under
    data/development/codegen/ (gitignored) with their sha recorded in meta."""
    code_sha = hashlib.sha256(code.encode("utf-8")).hexdigest()
    base = (root or (_REPO_ROOT / "runs" / "p1_codegen")) / strategy / model_id / code_sha[:12]
    base.mkdir(parents=True, exist_ok=True)
    (base / "code.py").write_text(code, encoding="utf-8")
    (base / "stdout.txt").write_text(stdout_tail, encoding="utf-8")
    (base / "meta.json").write_text(
        json.dumps({**meta, "code_sha256": code_sha}, indent=2, sort_keys=True, default=str),
        encoding="utf-8",
    )
    return ArchiveEntry(strategy=strategy, model_id=model_id,
                        code_sha12=code_sha[:12], directory=base)


def contract_freeze_ok() -> tuple[bool, str]:
    """I5: `--execute` requires the mini-contract to exist and its recorded
    prompt-asset hashes to match the live assets byte-for-byte."""
    if not _CONTRACT.exists():
        return False, f"mini-contract missing: {_CONTRACT}"
    text = _CONTRACT.read_text(encoding="utf-8")
    for name, digest in prompt_asset_hashes().items():
        if digest not in text:
            return False, (
                f"prompt asset {name} (sha256 {digest[:16]}…) is not recorded in the "
                "mini-contract freeze block — re-freeze before any generation call"
            )
    return True, "contract freeze verified"


def run_ablation(
    strategies: tuple[str, ...] = STRATEGIES,
    *,
    dry_run: bool,
    client_factory: Callable[[dict], ModelClient] | None = None,
    cache_root: Path | None = None,
) -> dict:
    """The ablation loop. `--dry-run` renders + hashes every prompt and calls
    nothing; `--execute` (blocked) generates once per (model, strategy),
    sandboxes, scores, archives."""
    models = load_phase_f_models()
    out: dict = {"dry_run": dry_run, "models": [m["model_id"] for m in models],
                 "prompt_sha256": {}, "runs": []}
    for strategy in strategies:
        out["prompt_sha256"][strategy] = prompt_sha256(build_prompt(strategy))
    if dry_run:
        return out

    ok, msg = contract_freeze_ok()
    if not ok:
        raise GenerationBlockedError(msg)
    cache = ResponseCache(cache_root or (_REPO_ROOT / "runs" / "p1_codegen" / "cache"))
    factory = client_factory or (lambda m: BlockedModelClient(m["model_id"]))
    for strategy in strategies:
        for model in models:
            client = factory(model)
            response = generate_once(strategy, model["model_id"], client, cache)
            out["runs"].append({
                "strategy": strategy,
                "model_id": model["model_id"],
                "configured_vendor": model.get("vendor"),
                "response_chars": len(response),
                "code_extracted": extract_code(response) is not None,
            })
    return out
