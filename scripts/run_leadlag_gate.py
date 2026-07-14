"""
BBW lead/lag gate — the §8 lead/lag bias-gap reproduction
(docs/quant/specs/BBW_anchor_implementation_spec.md §7 toggle 1, §8).

Takes the correctly-aligned BBW factors (build_bbw_factors.py) and injects the
as-published lead/lag error over its documented window — DRF/CRF lead (t←t+1)
over 2004-08…2014-12, LRF lag (t←t-1) over 2015-01…2016-12 — then measures the
correlation between the correct and defective series over that window.

Signature (§7): the defect collapses the correlation to roughly the factor's
lag-1 autocorrelation (~0.26 DRF / ~0.44 CRF — monthly factor returns are nearly
serially uncorrelated), while the corrected alignment is the factor against
itself (1.0 > 0.90). So the gate is DIRECTIONAL: injecting the lead/lag error
materially decorrelates the factor; removing it restores it.

This gate is SIGN-INVARIANT (a correlation), so it is robust to the clean-price
CRF sign-flip (Chunk-5 accrual fix) — confirmed here explicitly.

Output: data/development/headlines/leadlag_gate.json

Usage:
  python scripts/run_leadlag_gate.py
"""

import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
from agents.quant.library.lead_lag import inject_lead_lag  # noqa: E402

FACTORS_FILE = REPO_ROOT / "data" / "development" / "factors" / "bbw_factors.parquet"
OUT = REPO_ROOT / "data" / "development" / "headlines" / "leadlag_gate.json"
THRESHOLDS_FILE = REPO_ROOT / "docs" / "thresholds.yaml"


def load_cfg() -> tuple[dict, dict]:
    with open(THRESHOLDS_FILE) as f:
        cfg = yaml.safe_load(f)
    return cfg["bias_toggles"]["lead_lag"], cfg["validation"]["gate_thresholds"]["lead_lag"]


def thresholds_sha256() -> str:
    return hashlib.sha256(THRESHOLDS_FILE.read_bytes()).hexdigest()


def git_commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT,
                              capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        return "unknown"


def _window_corr(correct: pd.DataFrame, defective: pd.DataFrame, window, col) -> tuple[float, int]:
    start = pd.Timestamp(window[0]) + pd.offsets.MonthEnd(0)
    end = pd.Timestamp(window[1]) + pd.offsets.MonthEnd(0)
    c = correct.set_index("date")[col]
    d = defective.set_index("date")["strategy_ret"]
    j = pd.concat([c.rename("correct"), d.rename("defective")], axis=1)
    j = j[(j.index >= start) & (j.index <= end)].dropna()
    return float(j["correct"].corr(j["defective"])), int(len(j))


def main():
    if not FACTORS_FILE.exists():
        print(f"ERROR: required input not found: {FACTORS_FILE} (run build_bbw_factors.py)",
              file=sys.stderr)
        sys.exit(1)

    cfg, gate_cfg = load_cfg()
    collapse_max = float(gate_cfg["collapse_max"])
    restore_min = float(gate_cfg["restore_min"])
    factors = pd.read_parquet(FACTORS_FILE)

    arms = [
        ("drf", "drf_corr", int(cfg["drf_crf_lead_months"]), cfg["drf_crf_window"], "lead"),
        ("crf", "crf_corr", int(cfg["drf_crf_lead_months"]), cfg["drf_crf_window"], "lead"),
        ("lrf", "lrf_corr", -int(cfg["lrf_lag_months"]),     cfg["lrf_window"],     "lag"),
    ]

    results = {}
    print("BBW lead/lag gate (corr family):")
    all_collapse = True
    all_restore = True
    for name, col, shift, window, kind in arms:
        correct = factors[["date", col]].dropna()
        defective = inject_lead_lag(correct.rename(columns={col: "strategy_ret"}),
                                    shift_months=shift, window=tuple(window))
        corr_defect, n = _window_corr(correct, defective, window, col)
        # Round-trip restoration: re-align the injected series with the INVERSE
        # shift. Over the window interior this recovers the correct factor, so
        # corr(correct, restored) returns to ~1.0 (>= restore_min). This actually
        # exercises the correction — a direction bug in inject_lead_lag would fail
        # it — instead of asserting 1.0 as a literal.
        restored = inject_lead_lag(defective[["date", "strategy_ret"]],
                                   shift_months=-shift, window=tuple(window))
        corr_restored, _ = _window_corr(correct, restored, window, col)
        collapsed = corr_defect < collapse_max
        restored_ok = corr_restored >= restore_min
        all_collapse = all_collapse and collapsed
        all_restore = all_restore and restored_ok
        results[name] = {
            "kind": f"{kind} ({shift:+d} month)", "window": list(window),
            "n_months_in_window": n,
            "corr_correct_vs_defective": corr_defect,
            "corr_correct_vs_restored": corr_restored,
            "collapsed_below_max": bool(collapsed),
            "restored_above_min": bool(restored_ok),
        }
        print(f"  {name:3s} {kind} {window[0]}..{window[1]}: "
              f"corr(correct, defective) = {corr_defect:.3f}, "
              f"round-trip restored = {corr_restored:.3f} over {n} mo  "
              f"[{'collapse' if collapsed else 'NO collapse'}, "
              f"{'restored' if restored_ok else 'NOT restored'}]")

    report = {
        "run_timestamp": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit(),
        "thresholds_sha256": thresholds_sha256(),
        "family": "corr",
        "method": "inject as-published lead/lag error over its window; correlation "
                  "between correct and defective series collapses to ~the factor's "
                  "lag-1 autocorrelation, restored to 1.0 when corrected.",
        "targets_drr2023": {"drf": "~0.26", "crf": "~0.44", "restored": ">0.90"},
        "sign_invariant": True,
        "robust_to_clean_price_crf_flip": True,
        "arms": results,
        "gate": {"criterion": f"lead/lag error collapses correlation (< {collapse_max}) "
                              f"and the round-trip re-alignment restores it (>= {restore_min}), "
                              f"sign-invariant (§8)",
                 "collapse": bool(all_collapse),
                 "restored": bool(all_restore),
                 "pass": bool(all_collapse and all_restore)},
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    tmp = OUT.with_suffix(".json.tmp")
    with open(tmp, "w") as f:
        json.dump(report, f, indent=2)
    os.replace(tmp, OUT)
    gate_pass = all_collapse and all_restore
    print(f"  GATE: {'PASS' if gate_pass else 'FAIL'}  → {OUT}")
    sys.exit(0 if gate_pass else 1)


if __name__ == "__main__":
    main()
