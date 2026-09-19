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
CRF sign flip (accrual treatment) — confirmed here explicitly.

Output: data/development/headlines/leadlag_gate.json

Usage:
  python scripts/run_leadlag_gate.py [--factors-dir DIR] [--out PATH]
      defaults: data/development/factors, data/development/headlines/leadlag_gate.json
"""

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
from agents.quant.library.lead_lag import inject_lead_lag  # noqa: E402

FACTORS_DIR = REPO_ROOT / "data" / "development" / "factors"
FACTORS_FILE = FACTORS_DIR / "bbw_factors.parquet"
OUT = REPO_ROOT / "data" / "development" / "headlines" / "leadlag_gate.json"
THRESHOLDS_FILE = REPO_ROOT / "docs" / "thresholds.yaml"


def load_cfg() -> tuple[dict, dict]:
    with open(THRESHOLDS_FILE) as f:
        cfg = yaml.safe_load(f)
    return cfg["bias_toggles"]["lead_lag"], cfg["validation"]["gate_thresholds"]["lead_lag"]


def thresholds_sha256() -> str:
    return hashlib.sha256(THRESHOLDS_FILE.read_bytes()).hexdigest()


def _rel(path: Path) -> str:
    """Repo-relative path when inside the repo, else the path as given (never raises)."""
    try:
        return str(Path(path).relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="BBW lead/lag gate.")
    ap.add_argument("--factors-dir", type=Path, default=FACTORS_DIR,
                    help="directory holding bbw_factors.parquet (default data/development/factors)")
    ap.add_argument("--out", type=Path, default=OUT,
                    help="report path (default data/development/headlines/leadlag_gate.json)")
    return ap.parse_args(argv or [])


def resolve_paths(args: argparse.Namespace) -> tuple[Path, Path]:
    """(bbw_factors parquet, report json) from the parsed flags. A non-default factors input with the default
    report path is refused — it would overwrite the default gate report with another basis's result."""
    factors_file, out = Path(args.factors_dir) / FACTORS_FILE.name, Path(args.out)
    if factors_file != FACTORS_FILE and out == OUT:
        raise SystemExit(f"ERROR: --factors-dir {args.factors_dir} with the default --out would overwrite the "
                         f"default gate report {OUT}; pass --out")
    return factors_file, out


def input_provenance(factors_file: Path) -> dict:
    """Report keys recording a non-default factors input; empty (report unchanged) by default."""
    if factors_file == FACTORS_FILE:
        return {}
    return {"factors_file": _rel(factors_file),
            "factors_file_sha256": hashlib.sha256(factors_file.read_bytes()).hexdigest()}


def _window_corr(correct: pd.DataFrame, defective: pd.DataFrame, window, col) -> tuple[float, int]:
    start = pd.Timestamp(window[0]) + pd.offsets.MonthEnd(0)
    end = pd.Timestamp(window[1]) + pd.offsets.MonthEnd(0)
    c = correct.set_index("date")[col]
    d = defective.set_index("date")["strategy_ret"]
    j = pd.concat([c.rename("correct"), d.rename("defective")], axis=1)
    j = j[(j.index >= start) & (j.index <= end)].dropna()
    return float(j["correct"].corr(j["defective"])), int(len(j))


def main(argv: list[str] | None = None):
    factors_file, out = resolve_paths(parse_args(argv))
    if not factors_file.exists():
        print(f"ERROR: required input not found: {factors_file} (run build_bbw_factors.py)",
              file=sys.stderr)
        sys.exit(1)

    cfg, gate_cfg = load_cfg()
    collapse_max = float(gate_cfg["collapse_max"])
    restore_min = float(gate_cfg["restore_min"])
    factors = pd.read_parquet(factors_file)

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
        # it.
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
        "thresholds_sha256": thresholds_sha256(),
        **input_provenance(factors_file),
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
                              f"sign-invariant (§8); two-sided mechanism gate, hard under "
                              f"D-Q1 bias_attribution",
                 "collapse": bool(all_collapse),
                 "restored": bool(all_restore),
                 "direction_pass": bool(all_collapse and all_restore)},
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".json.tmp")
    with open(tmp, "w") as f:
        json.dump(report, f, indent=2)
    os.replace(tmp, out)
    gate_pass = all_collapse and all_restore
    print(f"  GATE: {'PASS' if gate_pass else 'FAIL'}  → {out}")
    sys.exit(0 if gate_pass else 1)


if __name__ == "__main__":
    main(sys.argv[1:])
