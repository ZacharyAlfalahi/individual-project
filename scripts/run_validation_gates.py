"""
§8 anchor validation summary (docs/quant/specs/BBW_anchor_implementation_spec.md §8).

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
        if "direction_pass" not in g:
            raise SystemExit(
                "str_decomposition.json uses the pre-D-Q1-split schema (no "
                "'direction_pass') — re-run build_str_decomposition.py before "
                "aggregating; stale-schema artefacts must not feed the summary."
            )
        lib_share_pct = (strd["lib_share_of_month_end"] or 0) * 100
        direction = bool(g["direction_pass"])
        gates["str_lib_decomposition"] = {
            "direction_pass": direction,
            "direction_reproduced": g.get("direction_reproduced"),
            "lib_share_pct": lib_share_pct,
            "magnitude_diagnostics": {
                "status": "soft_diagnostic_descriptive_per_D-Q1",
                "lib_share_in_band": g["magnitude_diagnostics"]["lib_share_in_band"],
            },
            "month_end_pct": strd["month_end"]["mean_pct"],
            "month_begin_pct": strd["month_begin"]["mean_pct"],
            "sign_invariant": design_sign_invariant,
            "verdict": (
                f"direction {'reproduced' if direction else 'NOT reproduced'} (sign-aware); "
                f"LIB share {lib_share_pct:.1f}% of month-end — soft diagnostic, "
                f"universe-sensitive (corr family liquid subset)"
            ),
        }
    else:
        gates["str_lib_decomposition"] = {"status": "NOT RUN — run build_str_decomposition.py"}

    if mom6:
        g = mom6["gate"]
        if "direction_pass" not in g:
            raise SystemExit(
                "mom6_lab_gate.json uses the pre-D-Q1-split schema (no "
                "'direction_pass') — re-run run_mom6_lab_gate.py before "
                "aggregating; stale-schema artefacts must not feed the summary."
            )
        direction = bool(g["direction_pass"])
        ep = mom6["ex_post_biased"]["mean_pct"]
        ea = mom6["ex_ante_corrected"]["mean_pct"]
        gap = mom6["ep_minus_ea_gap"]["mean_pct"]
        gates["mom6_lab"] = {
            "direction_pass": direction,
            "direction_reproduced": g.get("direction_reproduced"),
            "ex_post_pct": ep,
            "ex_ante_pct": ea,
            "ep_minus_ea_pct": gap,
            "magnitude_diagnostics": {
                "status": "soft_diagnostic_descriptive_per_D-Q1",
                "ex_post_in_drr_band": g["magnitude_diagnostics"]["ex_post_in_drr_band"],
                "full_collapse_to_baseline": g["magnitude_diagnostics"]["full_collapse_to_baseline"],
            },
            "sign_invariant": design_sign_invariant,
            "verdict": (
                f"ex-post {ep:+.2f}%/mo vs ex-ante {ea:+.2f}%/mo, EP−EA gap {gap:+.2f} — "
                f"look-ahead direction {'confirmed' if direction else 'NOT confirmed'}; "
                f"full ex-ante collapse partial (soft diagnostic)"
            ),
        }
    else:
        gates["mom6_lab"] = {"status": "NOT RUN — run run_mom6_lab_gate.py"}

    if leadlag:
        g = leadlag["gate"]
        if "direction_pass" not in g:
            raise SystemExit(
                "leadlag_gate.json uses the pre-D-Q1-split schema (no "
                "'direction_pass') — re-run run_leadlag_gate.py before "
                "aggregating; stale-schema artefacts must not feed the summary."
            )
        direction = bool(g["direction_pass"])
        drf_c = leadlag["arms"]["drf"]["corr_correct_vs_defective"]
        crf_c = leadlag["arms"]["crf"]["corr_correct_vs_defective"]
        lrf_c = leadlag["arms"]["lrf"]["corr_correct_vs_defective"]
        gates["bbw_lead_lag"] = {
            "direction_pass": direction,
            "drf_corr_defective": drf_c,
            "crf_corr_defective": crf_c,
            "lrf_corr_defective": lrf_c,
            "sign_invariant": bool(leadlag.get("sign_invariant", design_sign_invariant)),
            "verdict": (
                f"lead/lag error collapses correlation (drf {drf_c:.2f}, crf {crf_c:.2f}, "
                f"lrf {lrf_c:.2f}); round-trip re-alignment "
                f"{'restores' if direction else 'does NOT restore'} it"
            ),
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
    all_gates_pass = all(
        bool(gates[k].get("direction_pass")) for k in expected if "status" not in gates[k]
    )
    # Sign-invariance is a STRUCTURAL design property recorded in thresholds.yaml,
    # not a per-run measurement; require all gates present so it can NOT be reported
    # satisfied with zero gates run (the old all([]) == True false-green path).
    all_sign_invariant = design_sign_invariant and all_present
    bias_attribution_direction_pass = all_present and all_gates_pass and all_sign_invariant

    report = {
        "run_timestamp": datetime.now(timezone.utc).isoformat(),
        "thresholds_sha256": thresholds_sha256(),
        "anchor_criterion": vcfg.get("anchor_criterion", "bias_attribution"),
        "bias_attribution_direction_pass": bool(bias_attribution_direction_pass),
        "schema_note": "'overall_pass' was renamed 'bias_attribution_direction_pass' "
                       "(2026-08-05, system-health-audit HIGH-3): the hard gate is the "
                       "D-Q1 bias-attribution DIRECTION verdict only; magnitude/band "
                       "checks live under each gate's 'magnitude_diagnostics' and are "
                       "soft descriptive diagnostics, never aggregated into the pass. "
                       "No alias is emitted — an old reader must fail here, not "
                       "misread a softer aggregate.",
        "gates_present": [k for k in expected if "status" not in gates[k]],
        "gates_missing": missing,
        "principle": "Anchors validated by reproducing published BIAS VERDICTS: the "
                     "DIRECTION of each toggle effect is the hard gate (D-Q1, "
                     "validation.anchor_criterion = bias_attribution); magnitude/"
                     "proportion bands are soft descriptive diagnostics; published "
                     "factor LEVELS are never targets (§8).",
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
    print(f"  BIAS-ATTRIBUTION DIRECTION GATE (D-Q1): "
          f"{'PASS' if bias_attribution_direction_pass else 'FAIL'}  → {OUT}")
    sys.exit(0 if bias_attribution_direction_pass else 1)


if __name__ == "__main__":
    main()
