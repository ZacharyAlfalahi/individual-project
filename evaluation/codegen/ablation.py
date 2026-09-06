"""P1 codegen ablation (WS-C) — the scored pipeline.

``runner.run_ablation`` renders + generates; THIS module closes the loop the mini-contract describes
(§1/§5): for each (strategy, model) it generates once (cache-first), extracts the single fenced
block, runs it in the WS-C sandbox against the exported corr-family engine panel, scores the output
series against the oracle with the rung-3/4 comparator, archives the run, and tabulates the raw
per-run metrics (§5 mandates the re-thresholdable table).

Separated from ``runner`` so ``runner`` stays generation-only (its dry-run + tests are untouched)
and this module owns the sandbox+score orchestration. Mirrors ``p2_driver``'s generate->sandbox
loop, adding oracle scoring (P2 has no oracle) and budget accounting.

DEV ONLY: the panel + oracles are under ``data/development/``; the sandbox denies network and jails
writes; the holdout is never opened.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import yaml

from agents.scientist.researcher.cache import ResponseCache
from evaluation.codegen.live_client import BudgetExceededError, BudgetGuard
from evaluation.codegen.oracles import load_oracle_series
from evaluation.codegen.panel_export import CODEGEN_PANEL, export_codegen_panel
from evaluation.codegen.runner import (
    GenerationBlockedError,
    ModelClient,
    STRATEGIES,
    archive_run,
    build_prompt,
    contract_freeze_ok,
    extract_code,
    generate_once,
    load_models,
    prompt_sha256,
)
from evaluation.codegen.sandbox import SandboxSpec, parse_output_csv, run_sandboxed
from evaluation.codegen.scoring import ScoreResult, Verdict, load_scoring_thresholds, score_run

_REPO_ROOT = Path(__file__).resolve().parents[2]
_THRESHOLDS = _REPO_ROOT / "docs" / "thresholds.yaml"


def _sandbox_bounds(thresholds_path: Path | None = None) -> tuple[int, int]:
    """(wall_clock_seconds, memory_mb) from p1_codegen.sandbox (fail-loud)."""
    doc = yaml.safe_load((thresholds_path or _THRESHOLDS).read_text(encoding="utf-8"))
    try:
        block = doc["p1_codegen"]["sandbox"]
        return int(block["wall_clock_seconds"]), int(block["memory_mb"])
    except (KeyError, TypeError) as exc:
        raise KeyError("thresholds.yaml has no p1_codegen.sandbox block "
                       "(wall_clock_seconds / memory_mb)") from exc


def _raw_metrics_row(strategy: str, model_id: str, score: ScoreResult) -> dict:
    """One row of the §5 raw per-run similarity table (re-thresholdable by any reader)."""
    r3 = score.rung3 or {}
    r4 = score.rung4 or {}
    return {
        "strategy": strategy,
        "model_id": model_id,
        "verdict": score.verdict.value,
        "reason": score.reason,
        "exact_tier": score.exact_tier,
        "failed_criteria": list(score.failed_criteria),
        "n_overlap": r3.get("n_overlap"),
        "correlation": r3.get("correlation"),
        "sign_agreement": r3.get("sign_agreement"),
        "mean_diff": r3.get("mean_diff"),
        "tracking_error": r3.get("tracking_error"),
        "max_abs_diff": r3.get("max_abs_diff"),
        "delta_average": r4.get("delta_average"),
        "delta_sharpe": r4.get("delta_sharpe"),
        "delta_t_stat": r4.get("delta_t_stat"),
    }


def run_scored_ablation(
    strategies: tuple[str, ...] = STRATEGIES,
    *,
    phase: str = "reported",
    client_factory: Callable[[dict], ModelClient],
    budget: BudgetGuard | None = None,
    panel_path: Path | None = None,
    cache_root: Path | None = None,
    sandbox_root: Path | None = None,
    python_bin: Path | None = None,
    thresholds_path: Path | None = None,
    ensure_panel: bool = True,
) -> dict:
    """The full scored ablation. Requires a real ``client_factory`` (a live pair, or a fake in
    tests); the contract freeze (§8, prompt-asset hashes) must verify before any generation.

    Returns a typed run record: per-run verdicts + metrics, the §5 raw metrics table, the phase +
    reportability stamp, and the budget summary. ``reportable`` is True only for ``phase='reported'``
    with every returned SKU matching the pinned Phase-F model (contract §3)."""
    ok, msg = contract_freeze_ok()
    if not ok:
        raise GenerationBlockedError(msg)

    models = load_models(phase, thresholds_path)
    thresholds = load_scoring_thresholds(thresholds_path)
    wall_clock_s, memory_mb = _sandbox_bounds(thresholds_path)

    import sys
    bin_path = Path(python_bin) if python_bin is not None else Path(sys.executable)
    panel = Path(panel_path) if panel_path is not None else CODEGEN_PANEL
    if ensure_panel and not panel.exists():
        export_codegen_panel(panel)
    if "holdout" in panel.parts:
        raise RuntimeError(f"codegen panel path touches the holdout partition: {panel}")

    cache = ResponseCache(cache_root or (_REPO_ROOT / "runs" / "p1_codegen" / "cache"))
    jail_root = Path(sandbox_root) if sandbox_root is not None else (
        _REPO_ROOT / "runs" / "p1_codegen" / "sandbox")

    out: dict = {
        "phase": phase,
        "models": [m["model_id"] for m in models],
        "panel_path": str(panel),
        "runs": [],
        "raw_metrics_table": [],
        "verdict_counts": {v.value: 0 for v in Verdict},
        "generation_errors": [],
    }
    sku_ok = True

    for strategy in strategies:
        oracle = load_oracle_series(strategy)
        for model in models:
            model_id = model["model_id"]
            client = client_factory(model)
            # A generation call can fail on infrastructure (vendor rate-limit exhaustion, SDK
            # error) — that is NOT a codegen datum (WONT_RUN/RUNS_WRONG/RUNS_RIGHT), so it is
            # recorded as a typed generation_error and the loop continues. One vendor's free-tier
            # 429 must never abort the whole ablation. A budget breach still HALTS (it is a real cap).
            try:
                response = generate_once(strategy, model_id, client, cache)
            except BudgetExceededError:
                raise
            except Exception as exc:
                out["generation_errors"].append({
                    "strategy": strategy, "model_id": model_id,
                    "error": f"{type(exc).__name__}: {exc}"[:300],
                })
                out["runs"].append({
                    "strategy": strategy, "model_id": model_id,
                    "configured_vendor": model.get("vendor"),
                    "code_extracted": False, "sandbox_status": "generation_error",
                    "sandbox_reason": f"{type(exc).__name__}", "verdict": None,
                    "exact_tier": False,
                    "returned_model_version": getattr(client, "last_model_version", None),
                    "operational_usage": client.operational_usage()
                    if hasattr(client, "operational_usage") else None,
                })
                continue
            code = extract_code(response)
            stdout_tail = ""

            if code is None:
                score = score_run(strategy, "wont_run", "malformed_response", oracle, None, thresholds)
                sandbox_status, sandbox_reason, output_sha = "wont_run", "malformed_response", None
            else:
                sb = run_sandboxed(
                    code,
                    SandboxSpec(
                        python_bin=bin_path,
                        jail_dir=jail_root / phase / strategy / model_id,
                        panel_path=panel,
                        wall_clock_s=wall_clock_s,
                        memory_mb=memory_mb,
                    ),
                )
                sandbox_status, sandbox_reason, output_sha = sb.status, sb.reason, sb.output_sha256
                stdout_tail = sb.stdout_tail
                if sb.status == "ok":
                    cand = parse_output_csv(sb.output_path)
                    score = score_run(strategy, "ok", None, oracle, cand, thresholds)
                else:
                    score = score_run(strategy, sb.status, sb.reason, oracle, None, thresholds)

            returned_version = getattr(client, "last_model_version", None)
            usage = client.operational_usage() if hasattr(client, "operational_usage") else None
            # Reportability (contract §3): the reported phase requires the vendor's returned SKU to
            # match the pinned model_id; a mismatch (or a dev run) is non-reportable.
            if phase == "reported" and returned_version is not None and model_id not in str(returned_version):
                sku_ok = False

            archive_run(
                strategy, model_id, code or "", stdout_tail,
                {
                    "phase": phase,
                    "verdict": score.verdict.value,
                    "reason": score.reason,
                    "exact_tier": score.exact_tier,
                    "failed_criteria": list(score.failed_criteria),
                    "sandbox_status": sandbox_status,
                    "sandbox_reason": sandbox_reason,
                    "output_sha256": output_sha,
                    "returned_model_version": returned_version,
                    "rung3": score.rung3,
                    "rung4": score.rung4,
                    "operational_usage": usage,
                },
                root=jail_root.parent / "archive",
            )

            out["runs"].append({
                "strategy": strategy,
                "model_id": model_id,
                "configured_vendor": model.get("vendor"),
                "response_chars": len(response),
                "code_extracted": code is not None,
                "sandbox_status": sandbox_status,
                "sandbox_reason": sandbox_reason,
                "verdict": score.verdict.value,
                "exact_tier": score.exact_tier,
                "returned_model_version": returned_version,
                "operational_usage": usage,
            })
            out["raw_metrics_table"].append(_raw_metrics_row(strategy, model_id, score))
            out["verdict_counts"][score.verdict.value] += 1

    out["prompt_sha256"] = {s: prompt_sha256(build_prompt(s)) for s in strategies}
    # Reportable only when the paid phase ran, every returned SKU matched (§3), AND no run failed
    # on infrastructure (a generation_error means missing data — re-run to fill via the cache).
    out["reportable"] = bool(phase == "reported" and sku_ok and not out["generation_errors"])
    out["sku_match"] = sku_ok
    if budget is not None:
        out["budget"] = budget.summary()
    return out
