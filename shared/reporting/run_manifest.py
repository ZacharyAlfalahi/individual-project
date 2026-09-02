"""
Shared per-run execution manifest (WS-8 / data-layer O11).

A unified sidecar every ``scripts/run_*.py`` driver can emit: execution timestamp +
code hash + input-data hashes + config hashes + output hashes + a per-run
*operational profile* (model calls, tokens in/out, wall-clock, retries, human
interventions, and a cost slot). This is the per-run EXECUTION record the corpus /
Phase-F runs need so the cost / capability-table data actually exists — the gap the
data-layer register flags as O11 ("no unified per-run manifest, no shared run_id").

Cost discipline: ``cost_usd`` is left ``None`` and DERIVED post-hoc from the stored
token counts times a CITED per-model rate — the mechanical figures (tokens, calls,
wall-clock, retries) are captured live so the cost is reconstructable later, and no
uncited cost is ever invented (the same honesty-of-unavailability rule the cost
register applies to trading costs).

It is deliberately distinct from ``agents/reporter/manifest.py`` (the hand-maintained
cross-agent *pointer* file that joins one strategy's artefacts by human assertion).
This one is machine-emitted per driver run.

**Sidecar discipline.** The manifest carries a wall-clock timestamp, so it is NOT
byte-reproducible and MUST NEVER be hashed into a result artefact or a pinned digest
— it records run identity *beside* the results, never inside them. The builder itself
is pure: the timestamp is passed in (never read from a clock here), so it is fully
deterministic and unit-testable; the driver stamps the real time at the call site.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

MANIFEST_SCHEMA = 2   # v2 (2026-08-24): operational_profile gains wall_clock_seconds, retries,
                      # interventions (the WS-8 / §4.7 mechanical operational log).


def _sha256_file(abs_path: Path) -> str | None:
    """sha256 of a file's bytes, or ``None`` if it does not exist. A missing path is a
    fact the manifest records (an un-emitted output, an absent optional input), never a
    raise — the manifest describes what a run touched."""
    p = Path(abs_path)
    return hashlib.sha256(p.read_bytes()).hexdigest() if p.exists() else None


def git_code_hash(repo_root: Path = REPO_ROOT) -> dict:
    """The code identity: commit sha, short sha, and whether the tree is dirty. Falls
    back to ``"unknown"`` rather than raising, so a manifest can be written outside a
    git checkout."""
    def _git(*args: str) -> str:
        try:
            return subprocess.check_output(["git", *args], cwd=repo_root).decode().strip()
        except Exception:
            return "unknown"

    return {
        "commit": _git("rev-parse", "HEAD"),
        "short": _git("rev-parse", "--short", "HEAD"),
        "dirty": _git("status", "--porcelain") not in ("", "unknown"),
    }


def _hash_map(paths: list[str] | None, repo_root: Path) -> dict[str, str | None]:
    """Map each repo-relative (or absolute) path to its sha256 (or None if absent),
    keyed by the path string as given, sorted for a stable, diff-friendly record."""
    out: dict[str, str | None] = {}
    for p in sorted(paths or []):
        ap = Path(p) if Path(p).is_absolute() else (repo_root / p)
        out[str(p)] = _sha256_file(ap)
    return out


def default_operational_profile() -> dict:
    """The zero-cost profile for a deterministic (no-LLM) run. The SCHEMA exists even
    when cost is 0, so a Phase-F run populates the same shape and the cost / capability
    table has data to read (WS-8 rationale)."""
    return {
        "phase": None,              # "D" | "F" | None(deterministic)
        "model_calls": 0,
        "tokens": None,             # {prompt, completion} once a model is called
        "cost_usd": None,           # DERIVED later from tokens x a cited rate; None until cited
        "wall_clock_seconds": None,
        "retries": 0,
        "interventions": [],        # human operator interventions w/ reason codes (operator-fed)
        "capability": "deterministic",
    }


def build_operational_profile(
    *,
    phase: str | None = None,
    model_calls: int = 0,
    prompt_tokens: int | None = None,
    completion_tokens: int | None = None,
    cache_creation_tokens: int | None = None,
    cache_read_tokens: int | None = None,
    wall_clock_seconds: float | None = None,
    retries: int = 0,
    cost_usd: float | None = None,
    interventions: list | None = None,
    capability: str = "llm",
) -> dict:
    """Assemble an operational profile from mechanically-captured values (WS-8 / §4.7).

    Token counts are the load-bearing figures: ``cost_usd`` is DERIVED downstream from the
    stored tokens times a cited per-model rate, so it is left ``None`` here rather than
    invented. A token count that the vendor did not return stays ``None`` (unavailable,
    never guessed). ``interventions`` is a list of ``{reason_code, ...}`` records fed by the
    operator; an empty list means an unattended run.

    Cache-token fields (A-lever, 2026-09-02, additive): on cache-aware vendors
    ``prompt`` is the UNCACHED remainder only -- total prompt volume per run is
    ``prompt + cache_creation + cache_read``, and each bucket bills at its own rate
    (write ~1.25x, read ~0.1x), so the split is required for honest cost derivation."""
    tokens = None
    if prompt_tokens is not None or completion_tokens is not None:
        tokens = {"prompt": prompt_tokens, "completion": completion_tokens}
        if cache_creation_tokens is not None or cache_read_tokens is not None:
            tokens["cache_creation"] = cache_creation_tokens
            tokens["cache_read"] = cache_read_tokens
    return {
        "phase": phase,
        "model_calls": model_calls,
        "tokens": tokens,
        "cost_usd": cost_usd,
        "wall_clock_seconds": wall_clock_seconds,
        "retries": retries,
        "interventions": list(interventions or []),
        "capability": capability,
    }


def build_run_manifest(
    *,
    run_id: str,
    driver: str,
    timestamp: str,
    inputs: list[str] | None = None,
    configs: list[str] | None = None,
    outputs: list[str] | None = None,
    operational_profile: dict | None = None,
    repo_root: Path = REPO_ROOT,
) -> dict:
    """Assemble a unified per-run manifest dict.

    ``timestamp`` is supplied by the caller (an ISO-8601 string) — never read from a
    clock here — so this function is pure and deterministic. ``inputs`` / ``configs`` /
    ``outputs`` are lists of repo-relative paths hashed into the record; a missing path
    hashes to ``None`` (recorded, not raised)."""
    return {
        "manifest_schema": MANIFEST_SCHEMA,
        "run_id": run_id,
        "driver": driver,
        "timestamp": timestamp,
        "code": git_code_hash(repo_root),
        "inputs": _hash_map(inputs, repo_root),
        "configs": _hash_map(configs, repo_root),
        "outputs": _hash_map(outputs, repo_root),
        "operational_profile": operational_profile or default_operational_profile(),
    }


def write_run_manifest(out_dir: str | Path, manifest: dict, *, filename: str = "run_manifest.json") -> Path:
    """Write the manifest as a JSON sidecar into ``out_dir`` and return its path. A
    sidecar only — never fed back into a hashed artefact."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / filename
    path.write_text(json.dumps(manifest, indent=2, default=str))
    return path
