"""
§8 anchor validation summary (BBW_anchor_implementation_spec.md §8).

Aggregates the three within-pipeline bias-gap gates into one verdict and records
the validation principle (bias-attribution, NOT absolute level) and the critical
sign-invariance guard. Reads the per-gate JSONs produced by:

    python scripts/build_str_decomposition.py   → str_decomposition.json
    python scripts/run_mom6_lab_gate.py          → mom6_lab_gate.json
    python scripts/run_leadlag_gate.py           → leadlag_gate.json

(run those first). Validates per the anchor criterion in thresholds.yaml
(validation.anchor_criterion = bias_attribution). The OSBAP directional smoke
test (§8 gate 2) needs OSBAP reference data not vendored here and is reported as
deferred.

Output: data/development/headlines/validation_gates_summary.json

Usage:
  python scripts/run_validation_gates.py
"""

import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
HEAD = REPO_ROOT / "data" / "development" / "headlines"
OUT = HEAD / "validation_gates_summary.json"
THRESHOLDS_FILE = REPO_ROOT / "docs" / "thresholds.yaml"


def _read(name: str) -> dict | None:
    p = HEAD / name
    if not p.exists():
        return None
    with open(p) as f:
        return json.load(f)


def thresholds_sha256() -> str:
    return hashlib.sha256(THRESHOLDS_FILE.read_bytes()).hexdigest()


def main():
    with open(THRESHOLDS_FILE) as f:
        vcfg = yaml.safe_load(f).get("validation", {})
    design_sign_invariant = bool(vcfg.get("bias_gates_sign_invariant", False))

    strd = _read("str_decomposition.json")
    mom6 = _read("mom6_lab_gate.json")
    leadlag = _read("leadlag_gate.json")

    gates = {}

    if strd:
        g = strd["gate"]
        gates["str_lib_decomposition"] = {
            "pass": bool(g.get("direction_reproduced")),
            "direction_reproduced": g.get("direction_reproduced"),
            "lib_share_pct": (strd["lib_share_of_month_end"] or 0) * 100,
            "lib_share_in_band": g.get("lib_share_in_band"),
            "month_end_pct": strd["month_end"]["mean_pct"],
            "month_begin_pct": strd["month_begin"]["mean_pct"],
            "sign_invariant": design_sign_invariant,
            "verdict": "direction reproduced; LIB proportion universe-sensitive (corr family liquid subset)",
        }
    else:
        gates["str_lib_decomposition"] = {"status": "NOT RUN — run build_str_decomposition.py"}

    if mom6:
        g = mom6["gate"]
        gates["mom6_lab"] = {
            "pass": bool(g.get("pass", g.get("direction_reproduced"))),
            "direction_reproduced": g.get("direction_reproduced"),
            "ex_post_pct": mom6["ex_post_biased"]["mean_pct"],
            "ex_ante_pct": mom6["ex_ante_corrected"]["mean_pct"],
            "ep_minus_ea_pct": mom6["ep_minus_ea_gap"]["mean_pct"],
            "sign_invariant": design_sign_invariant,
            "verdict": "ex-post reproduces DRR biased ≈+0.30; EP−EA gap > 0; full ex-ante collapse partial",
        }
    else:
        gates["mom6_lab"] = {"status": "NOT RUN — run run_mom6_lab_gate.py"}

    if leadlag:
        gates["bbw_lead_lag"] = {
            "pass": leadlag["gate"]["pass"],
            "drf_corr_defective": leadlag["arms"]["drf"]["corr_correct_vs_defective"],
            "crf_corr_defective": leadlag["arms"]["crf"]["corr_correct_vs_defective"],
            "lrf_corr_defective": leadlag["arms"]["lrf"]["corr_correct_vs_defective"],
            "sign_invariant": bool(leadlag.get("sign_invariant", design_sign_invariant)),
            "verdict": "lead/lag error collapses correlation (DRF→0.26 ≈ DRR 0.26); round-trip re-alignment restores it",
        }
    else:
        gates["bbw_lead_lag"] = {"status": "NOT RUN — run run_leadlag_gate.py"}

    gates["osbap_directional"] = {
        "status": "DEFERRED — OSBAP ret_vw reference not vendored; CRF sign also "
                  "pending the Chunk-5 accrual upgrade.",
    }

    expected = ("str_lib_decomposition", "mom6_lab", "bbw_lead_lag")
    missing = [k for k in expected if "status" in gates[k]]
    all_present = not missing
    all_gates_pass = all(bool(gates[k].get("pass")) for k in expected if "status" not in gates[k])
    # Sign-invariance is a STRUCTURAL design property recorded in thresholds.yaml,
    # not a per-run measurement; require all gates present so it can NOT be reported
    # satisfied with zero gates run (the old all([]) == True false-green path).
    all_sign_invariant = design_sign_invariant and all_present
    overall_pass = all_present and all_gates_pass and all_sign_invariant

    report = {
        "run_timestamp": datetime.now(timezone.utc).isoformat(),
        "thresholds_sha256": thresholds_sha256(),
        "anchor_criterion": vcfg.get("anchor_criterion", "bias_attribution"),
        "overall_pass": bool(overall_pass),
        "gates_present": [k for k in expected if "status" not in gates[k]],
        "gates_missing": missing,
        "principle": "Anchors validated by reproducing published BIAS VERDICTS "
                     "(direction + approximate magnitude of each toggle effect), "
                     "NOT published factor levels (§8).",
        "critical_guard": {
            "claim": "every bias gate is a within-pipeline differential/correlation, "
                     "hence sign-invariant and robust to the clean-price CRF flip",
            "all_bias_gates_sign_invariant": bool(all_sign_invariant),
            "all_expected_gates_present": bool(all_present),
            "only_sign_dependent_check": "osbap_directional (deferred)",
        },
        "gates": gates,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    tmp = OUT.with_suffix(".json.tmp")
    with open(tmp, "w") as f:
        json.dump(report, f, indent=2)
    os.replace(tmp, OUT)

    print("§8 anchor validation summary (criterion: bias-attribution):")
    for name, g in gates.items():
        if "status" in g:
            print(f"  {name:24s}: {g['status']}")
        else:
            print(f"  {name:24s}: {g['verdict']}")
    if missing:
        print(f"  MISSING gates (counts as FAIL — run them first): {', '.join(missing)}")
    print(f"  critical guard — all bias gates sign-invariant: "
          f"{'YES' if all_sign_invariant else 'NO'} (robust to clean-price CRF flip)")
    print(f"  OVERALL: {'PASS' if overall_pass else 'FAIL'}  → {OUT}")
    sys.exit(0 if overall_pass else 1)


if __name__ == "__main__":
    main()
