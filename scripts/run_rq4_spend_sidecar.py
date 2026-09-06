"""RQ4 paid-run spend sidecar — a reconstructed metering record.

The RQ4 reportable run used paid Phase-F generation without recording runtime authorization or
per-call token telemetry. This script reconstructs spend as far as the response-only cache allows,
explicitly identifying what is and is not recoverable:

  * OUTPUT tokens are estimated from each cached response's length (~chars/4) — a rough but
    reproducible lower-ish estimate; multiplied by the PRE-REGISTERED output price
    (`p1_codegen.budget.prices_usd_per_1m`) gives a metered-from-cache output cost.
  * INPUT tokens were NOT recorded (the cache stores only model/seed/key/response), so input cost
    is NOT reconstructable exactly; a stated per-call upper bound is applied so the total is bounded,
    never fabricated as exact.
  * The free dev pair (gemini-3.1-flash-lite / mistral) is priced $0 — its spend is logged, never paid.

This is a reconstruction, not a runtime meter. It provides a bounded spend estimate for a run that
did not emit a runtime usage record. The result is explicitly reconstructed rather than directly
metered. Deterministic, offline, $0.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import yaml  # noqa: E402

_THRESHOLDS = REPO_ROOT / "docs" / "thresholds.yaml"
_RQ4_CACHE = REPO_ROOT / "runs" / "rq4_funnel" / "cache"
_OUT_DIR = REPO_ROOT / "results" / "scientist" / "rq4_funnel"

_CHARS_PER_TOKEN = 4.0                 # standard rough heuristic for the output-token estimate
_INPUT_UPPER_BOUND_TOKENS = 8000       # STATED assumption: a generous per-call prompt ceiling


class RQ4SidecarError(RuntimeError):
    """A required input (cache / prices) is missing — surfaced loudly, never a silent $0."""


def _git_commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT,
                              capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        return "unknown"


def load_prices(thresholds_path: Path | None = None) -> dict:
    """The pre-registered per-model USD/1M prices (`p1_codegen.budget.prices_usd_per_1m`)."""
    doc = yaml.safe_load((thresholds_path or _THRESHOLDS).read_text(encoding="utf-8"))
    try:
        return dict(doc["p1_codegen"]["budget"]["prices_usd_per_1m"])
    except (KeyError, TypeError) as exc:
        raise RQ4SidecarError(
            "p1_codegen.budget.prices_usd_per_1m missing — cannot price the RQ4 spend") from exc


def reconstruct_spend(cache_records: list[dict], prices: dict) -> dict:
    """Reconstruct the per-model spend from response-only cache records + pre-registered prices.

    ``cache_records`` are the cache dicts (``model`` + ``response``). Output tokens are estimated
    from the response length; input tokens are NOT in the cache and are upper-bounded, not invented."""
    by_model: dict[str, dict] = defaultdict(lambda: {"n_calls": 0, "output_chars": 0})
    for rec in cache_records:
        m = rec.get("model", "unknown")
        by_model[m]["n_calls"] += 1
        by_model[m]["output_chars"] += len(rec.get("response", "") or "")

    models_out: dict[str, dict] = {}
    total_output_cost = 0.0
    total_input_upper_cost = 0.0
    paid_calls = 0
    for m, agg in sorted(by_model.items()):
        price = prices.get(m)
        if price is None:
            raise RQ4SidecarError(f"no pre-registered price for model {m!r} — cannot meter it")
        out_tokens = int(round(agg["output_chars"] / _CHARS_PER_TOKEN))
        out_cost = out_tokens / 1_000_000 * float(price["output"])
        in_ub_tokens = agg["n_calls"] * _INPUT_UPPER_BOUND_TOKENS
        in_ub_cost = in_ub_tokens / 1_000_000 * float(price["input"])
        is_paid = (float(price["input"]) > 0 or float(price["output"]) > 0)
        if is_paid:
            paid_calls += agg["n_calls"]
        total_output_cost += out_cost
        total_input_upper_cost += in_ub_cost
        models_out[m] = {
            "n_calls": agg["n_calls"],
            "paid": is_paid,
            "est_output_tokens": out_tokens,
            "price_usd_per_1m": {"input": float(price["input"]), "output": float(price["output"])},
            "output_cost_usd": round(out_cost, 6),
            "input_tokens_captured": False,
            "input_upper_bound_tokens": in_ub_tokens,
            "input_cost_upper_bound_usd": round(in_ub_cost, 6),
        }

    return {
        "experiment": "rq4_funnel_spend_reconstruction",
        "basis": "retroactive_reconstruction",
        "models": models_out,
        "paid_calls": paid_calls,
        "totals": {
            "output_cost_usd_metered_from_cache": round(total_output_cost, 6),
            "input_cost_usd_UPPER_BOUND": round(total_input_upper_cost, 6),
            "total_estimate_upper_bound_usd": round(total_output_cost + total_input_upper_cost, 6),
        },
        "method": {
            "output_tokens": f"estimated from cached response length (~chars/{int(_CHARS_PER_TOKEN)})",
            "input_tokens": ("NOT captured at run time (cache stores model/seed/key/response only); "
                             f"upper-bounded at {_INPUT_UPPER_BOUND_TOKENS} tokens/call — a stated "
                             "assumption, not a measurement"),
            "prices": "pre-registered p1_codegen.budget.prices_usd_per_1m",
            "cache_dir": str(_RQ4_CACHE.relative_to(REPO_ROOT)),
        },
        "governance": {
            "enforced_signoff_gate_at_runtime": False,
            "usage_metered_at_runtime": False,
            "reconstruction": True,
            "note": ("The RQ4 Phase-F run did not record runtime authorization or token usage. "
                     "This sidecar reconstructs a bounded estimate from the response cache; "
                     "authorization status remains external to this artefact."),
        },
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache", default=str(_RQ4_CACHE))
    ap.add_argument("--out", default=None,
                    help="output path (default results/scientist/rq4_funnel/rq4_spend_sidecar_<date>.json)")
    args = ap.parse_args(argv)

    cache_dir = Path(args.cache)
    files = sorted(cache_dir.glob("*.json"))
    if not files:
        raise RQ4SidecarError(f"no cache records under {cache_dir} — cannot reconstruct RQ4 spend")
    records = [json.loads(f.read_text(encoding="utf-8")) for f in files]

    result = reconstruct_spend(records, load_prices())
    stamp = datetime.now(timezone.utc)
    result["provenance"] = {
        "reconstructor": "scripts/run_rq4_spend_sidecar.py",
        "run_timestamp": stamp.isoformat(),
        "git_commit": _git_commit(),
        "n_cache_files": len(files),
        "dev_only": True,
        "model_calls": 0,
        "spend_usd": 0.0,
    }

    out = Path(args.out) if args.out else _OUT_DIR / "rq4_spend_sidecar.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True, default=str), encoding="utf-8")

    t = result["totals"]
    print(f"RQ4 spend reconstruction: {result['paid_calls']} paid Phase-F calls; output cost "
          f"${t['output_cost_usd_metered_from_cache']:.4f} (metered from cache); total upper bound "
          f"${t['total_estimate_upper_bound_usd']:.4f} (input tokens NOT captured — upper-bounded).")
    print(f"results written: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
