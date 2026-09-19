"""P2 coverage-boundary driver (WS-C) — generation loop (build-only).

Loops Arm A ∪ Arm B and, for each member, builds ONE prompt from the
Librarian-EXTRACTED spec (a variant of ``runner.build_prompt`` that does NOT
call ``load_gold_spec`` — P2 has no gold; it still redacts
``claimed_headline_metric`` via the shared ``runner`` redaction) and generates
TWICE (the Phase-F pair), cache-first, then runs each through the WS-C sandbox.

GATED emission: agreement/divergence NUMBERS are produced downstream by
``p2_metrics`` only when the scale census EXISTS *and*
``corpus.selection.status == 'frozen'``. Until then every mode is fixture /
dry-run: prompts are rendered and hashed, but ZERO generation calls are made
(the CountingStub proof in the tests). The default client is the P1
``BlockedModelClient``, so even a mis-gated ``--execute`` fails closed with a
typed refusal.

Single responsibility: this module produces prompts + (in the gated execute
path) sandbox run records. It does NOT compute agreement numbers — those come
from ``p2_metrics.compute_p2_metrics`` once real return series exist.
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import yaml

from agents.scientist.researcher.cache import ResponseCache
from evaluation.codegen import runner
from evaluation.codegen.census import CensusResult
from evaluation.codegen.live_client import BudgetExceededError
from evaluation.codegen.p2_selector import ArmSelection
from evaluation.codegen.runner import (
    BlockedModelClient,
    ModelClient,
    archive_run,
    extract_code,
    load_models,
    prompt_sha256,
)
from evaluation.codegen.sandbox import SandboxResult, SandboxSpec, run_sandboxed

_REPO_ROOT = Path(__file__).resolve().parents[2]
_THRESHOLDS = _REPO_ROOT / "docs" / "thresholds.yaml"

#: The census-selection status that unlocks number emission (mirrors the
#: auditor.ipca_differential.status convention: draft => proposals only).
FROZEN_STATUS = "frozen"

#: How much of a failed run's stderr the run record carries (the full tail is archived).
_DIAGNOSTIC_CHARS = 600


# ---------------------------------------------------------------------------
# Prompt from the EXTRACTED spec (a variant of runner.build_prompt — NOT gold).
# ---------------------------------------------------------------------------

def serialise_extracted_spec(spec: object) -> str:
    """The Librarian-extracted spec as redacted, deterministic JSON. Accepts a
    StrategySpec-like object (``.to_dict()``) or a plain dict. Reuses the shared
    ``runner`` redaction so ``claimed_headline_metric`` is stripped exactly as on
    the P1 path — the model receives the construction, never the paper's number."""
    payload = spec.to_dict() if hasattr(spec, "to_dict") else spec
    if not isinstance(payload, dict):
        raise TypeError(
            "extracted spec must be a dict or expose .to_dict(); got "
            f"{type(spec).__name__}"
        )
    return json.dumps(runner._redact(payload), indent=2, sort_keys=True, default=str)


def build_prompt_from_spec(spec: object) -> str:
    """Render the codegen prompt for one member from its EXTRACTED spec, reusing
    the frozen P1 prompt assets (``template.md`` / ``panel_schema.md`` /
    ``output_contract.md``). Deliberately does NOT call ``load_gold_spec``."""
    template = (runner._PROMPT_DIR / "template.md").read_text(encoding="utf-8")
    schema = (runner._PROMPT_DIR / "panel_schema.md").read_text(encoding="utf-8")
    contract = (runner._PROMPT_DIR / "output_contract.md").read_text(encoding="utf-8")
    return (
        template
        .replace("{{GOLD_SPEC_JSON}}", serialise_extracted_spec(spec))
        .replace("{{PANEL_SCHEMA}}", schema)
        .replace("{{OUTPUT_CONTRACT}}", contract)
    )


def _generate_from_prompt(
    prompt: str, model_id: str, client: ModelClient, cache: ResponseCache, *, seed: int = 0
) -> str:
    """Cache-first single-shot generation from a prebuilt prompt — mirrors
    ``runner.generate_once`` (which is gold-bound) for the extracted-spec prompt.
    A cache hit performs ZERO client calls."""
    cached = cache.get(prompt, model_id, seed)
    if cached is not None:
        return cached
    response = client.generate(prompt, seed=seed)
    cache.put(prompt, model_id, seed, response)
    return response


# ---------------------------------------------------------------------------
# Gating.
# ---------------------------------------------------------------------------

def sandbox_bounds(thresholds_path: Path | None = None) -> tuple[int, int]:
    """``(wall_clock_seconds, memory_mb)`` from ``p1_codegen.sandbox`` — read at runtime,
    never hard-coded, so one thresholds edit moves P1 and P2 together (fail-loud)."""
    doc = yaml.safe_load((thresholds_path or _THRESHOLDS).read_text(encoding="utf-8"))
    try:
        block = doc["p1_codegen"]["sandbox"]
        return int(block["wall_clock_seconds"]), int(block["memory_mb"])
    except (KeyError, TypeError) as exc:
        raise KeyError("thresholds.yaml has no p1_codegen.sandbox block "
                       "(wall_clock_seconds / memory_mb)") from exc


def corpus_selection_status(thresholds_path: Path | None = None) -> str:
    """``corpus.selection.status`` read at runtime (fail-loud). Only ``'frozen'``
    unlocks the number-emitting execute path."""
    doc = yaml.safe_load((thresholds_path or _THRESHOLDS).read_text(encoding="utf-8"))
    try:
        return str(doc["corpus"]["selection"]["status"])
    except (KeyError, TypeError) as exc:
        raise KeyError("thresholds.yaml has no corpus.selection.status") from exc


# ---------------------------------------------------------------------------
# The driver.
# ---------------------------------------------------------------------------

def run_p2_driver(
    census: CensusResult,
    selection: ArmSelection,
    *,
    dry_run: bool,
    census_available: bool = False,
    phase: str = "reported",
    client_factory=None,
    cache_root: Path | None = None,
    thresholds_path: Path | None = None,
    panel_path: Path | None = None,
    python_bin: Path | None = None,
    sandbox_root: Path | None = None,
) -> dict:
    """Build every member's prompt; in the gated execute path, generate twice per
    member and sandbox each. Returns a typed run record.

    Emission is gated: unless ``dry_run is False`` AND ``census_available`` AND
    ``corpus.selection.status == 'frozen'``, the loop stops after rendering and
    hashing prompts — ZERO generation calls. Number assembly (agreement /
    divergence) is a downstream ``p2_metrics`` step over the produced series."""
    models = load_models(phase, thresholds_path)
    status = corpus_selection_status(thresholds_path)
    gated = bool(census_available and status == FROZEN_STATUS)

    out: dict = {
        "dry_run": dry_run,
        "census_available": census_available,
        "corpus_selection_status": status,
        "gated_emit": bool(gated and not dry_run),
        "models": [m["model_id"] for m in models],
        "arm_a": list(selection.arm_a),
        "arm_b": list(selection.arm_b),
        "arm_a_size": selection.arm_a_size,
        "below_floor": selection.below_floor,
        "prompt_sha256": {},
        "runs": [],
        "generation_errors": [],
    }

    prompts: dict[str, str] = {}
    for pid in selection.members():
        member = census.member(pid)
        if member.extracted_spec is None:
            raise ValueError(
                f"selected member {pid!r} has no extracted spec — cannot build a prompt "
                "(a selected member must have cleared the text-quality bar with a spec)"
            )
        prompt = build_prompt_from_spec(member.extracted_spec)
        prompts[pid] = prompt
        out["prompt_sha256"][pid] = prompt_sha256(prompt)

    if dry_run or not gated:
        out["emitted_numbers"] = False
        return out

    # --- gated execute path ---------------------------------------------------
    cache = ResponseCache(cache_root or (_REPO_ROOT / "runs" / "p2_codegen" / "cache"))
    factory = client_factory or (lambda m: BlockedModelClient(m["model_id"]))
    bin_path = Path(python_bin) if python_bin is not None else Path(sys.executable)
    jail_root = Path(sandbox_root) if sandbox_root is not None else (
        _REPO_ROOT / "runs" / "p2_codegen" / "sandbox"
    )
    if panel_path is not None and "holdout" in Path(panel_path).parts:
        raise RuntimeError(f"codegen panel path touches the holdout partition: {panel_path}")
    wall_clock_s, memory_mb = sandbox_bounds(thresholds_path)
    # ONE client per model, not one per (member, model): a fresh client resets the vendor
    # pacing clock (``LiveCodegenClient._last_call_ts``), which would defeat ``min_interval_s``
    # and turn a vendor rate-limit into a recorded codegen failure at the boundary.
    clients = {m["model_id"]: factory(m) for m in models}
    arm_of = {pid: "A" for pid in selection.arm_a}
    arm_of.update({pid: "B" for pid in selection.arm_b})
    for pid in selection.members():
        prompt = prompts[pid]
        for model in models:
            client = clients[model["model_id"]]
            # Infrastructure failure (vendor rate-limit exhaustion) is a typed generation_error,
            # never a silent drop and never an abort of the whole coverage loop; a budget breach
            # still HALTS. Mirrors evaluation/codegen/ablation.run_scored_ablation.
            try:
                response = _generate_from_prompt(prompt, model["model_id"], client, cache)
            except BudgetExceededError:
                raise
            except Exception as exc:
                out["generation_errors"].append({
                    "paper_id": pid, "model_id": model["model_id"],
                    "error": f"{type(exc).__name__}: {exc}"[:300]})
                out["runs"].append({
                    "paper_id": pid, "arm": arm_of[pid], "model_id": model["model_id"],
                    "code_extracted": False, "sandbox_status": "generation_error",
                    "sandbox_reason": f"{type(exc).__name__}",
                    "output_path": None, "output_sha256": None,
                    "returned_model_version": getattr(client, "last_model_version", None)})
                continue
            code = extract_code(response)
            record: dict = {
                "paper_id": pid,
                "arm": arm_of[pid],
                "model_id": model["model_id"],
                "response_chars": len(response),
                "code_extracted": code is not None,
                # A response with no single fenced block never reaches the sandbox; it is the
                # same WONT_RUN(malformed_response) datum the P1 path records, typed here
                # so the key is never absent (count-not-retry, I3).
                "sandbox_status": "wont_run",
                "sandbox_reason": "malformed_response",
                "output_path": None,
                "output_sha256": None,
            }
            stdout_tail = ""
            stderr_tail = ""
            if code is not None:
                jail_dir = jail_root / pid / model["model_id"]
                # A jail is content-blind: generated code that exits 0 WITHOUT writing leaves
                # any earlier run's out/portfolio_returns.csv in place, and it would be read
                # back (and hashed) as this run's series. Clear the jail first — a stale
                # series scored as a fresh one is fabricated data.
                shutil.rmtree(jail_dir, ignore_errors=True)
                sb: SandboxResult = run_sandboxed(
                    code,
                    SandboxSpec(
                        python_bin=bin_path,
                        jail_dir=jail_dir,
                        panel_path=panel_path,
                        wall_clock_s=wall_clock_s,
                        memory_mb=memory_mb,
                    ),
                )
                stdout_tail = sb.stdout_tail
                stderr_tail = sb.stderr_tail
                record["sandbox_status"] = sb.status
                record["sandbox_reason"] = sb.reason
                record["output_sha256"] = sb.output_sha256
                record["output_path"] = None if sb.output_path is None else str(sb.output_path)
            # The failure diagnostic the taxonomy reads: WHY a run would not execute. Kept
            # short in the record (the full tail is archived beside the code).
            record["stderr_tail"] = stderr_tail[-_DIAGNOSTIC_CHARS:] if stderr_tail else ""
            record["returned_model_version"] = getattr(client, "last_model_version", None)
            if hasattr(client, "operational_usage"):
                record["operational_usage"] = client.operational_usage()
            archive_run(
                pid, model["model_id"], code or "", stdout_tail,
                {**{k: record[k] for k in (
                    "arm", "code_extracted", "sandbox_status", "sandbox_reason",
                    "output_sha256", "returned_model_version")},
                 "stderr_tail": stderr_tail},
                root=jail_root.parent / "archive",
            )
            out["runs"].append(record)

    out["emitted_numbers"] = True
    return out
