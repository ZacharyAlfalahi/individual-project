"""
RQ2 anchor fidelity summary (evaluation contract §7, v1.6 re-scope 2026-08-14).

The RQ2 anchor fidelity PASS CONDITION is contract §7 **gates 1-2** — construction
invariants + rulebook byte-equality — computed here from code + the hand-authored
gold specs via the G2 round-trip harness. The §7 **gate-5 bias-mechanism
differential** (str LIB decomposition, mom6 LAB, BBW lead/lag) is retained as a
reported **RQ3-facing diagnostic** and NO LONGER gates the RQ2 verdict (v1.6
supersedes O5 / D-Q1, which had made gate 5 the pass condition).

The gate-1-2 verdict is independent of the differential: it is computed from the
adapter + golds + the hash-verified standing register, so a missing or stale
differential JSON is a diagnostic status, never a gate failure.

The differential diagnostic reads the per-gate JSONs produced by:
    python scripts/build_str_decomposition.py   -> str_decomposition.json
    python scripts/run_mom6_lab_gate.py          -> mom6_lab_gate.json
    python scripts/run_leadlag_gate.py           -> leadlag_gate.json

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
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evaluation.harness.round_trip import (  # noqa: E402
    gate12_verdict,
    load_verified_standing_subs,
)

HEAD = REPO_ROOT / "data" / "development" / "headlines"
OUT = HEAD / "validation_gates_summary.json"
THRESHOLDS_FILE = REPO_ROOT / "docs" / "thresholds.yaml"

# The anchors whose gold specs compile + byte-equal (str/drf single-leg, crf the
# 3-leg composite, mom6 binding since the JNPS freeze). This set defines RQ2
# anchor fidelity under §7 gates 1-2.
ANCHORS = ("str", "drf", "mom6", "crf")

SCHEMA_NOTE = (
    "'overall_pass' -> 'bias_attribution_direction_pass' (2026-08-05, "
    "system-health-audit HIGH-3) -> 'construction_rulebook_pass' (2026-08-14, "
    "RQ2 v1.6 re-scope): the RQ2 anchor fidelity pass condition is now contract "
    "§7 gates 1-2 (construction invariants + rulebook byte-equality). The §7 "
    "gate-5 bias-mechanism differential is retained under 'differential_diagnostic' "
    "as a reported RQ3-facing diagnostic and NO LONGER gates. No alias is emitted "
    "— an old reader must fail here, not misread a superseded aggregate."
)


def _read(name: str) -> dict | None:
    p = HEAD / name
    if not p.exists():
        return None
    with open(p) as f:
        return json.load(f)


def thresholds_sha256() -> str:
    return hashlib.sha256(THRESHOLDS_FILE.read_bytes()).hexdigest()


# ---------------------------------------------------------------------------
# The PASS: RQ2 anchor fidelity = contract §7 gates 1-2 (construction + rulebook)
# ---------------------------------------------------------------------------

def compute_construction_rulebook_verdict(standing_subs=None) -> dict:
    """Gate-1+2 RQ2 anchor fidelity across every anchor -- THE pass condition.

    Deliberately independent of the (diagnostic) bias differential: it compiles
    each gold through the adapter with the hash-verified standing register and
    checks rulebook byte-equality + the holding_period invariant. The returned
    dict is stable regardless of any differential JSON on disk.
    """
    subs = standing_subs if standing_subs is not None else load_verified_standing_subs()
    per_anchor = {a: gate12_verdict(a, subs) for a in ANCHORS}
    all_pass = all(v["pass"] for v in per_anchor.values())
    return {"construction_rulebook_pass": bool(all_pass), "anchors": per_anchor}


# ---------------------------------------------------------------------------
# The DIAGNOSTIC: §7 gate-5 bias-mechanism differential (NON-GATING under v1.6)
# ---------------------------------------------------------------------------

def _stale_or_gate(g: dict, source: str):
    """Return the gate dict if it carries the post-D-Q1-split schema, else a
    STALE status. NON-GATING: a stale differential JSON is a diagnostic-quality
    issue, flagged (never silent) but no longer a pass/fail input."""
    if "direction_pass" not in g:
        return {"status": f"STALE_SCHEMA — re-run {source} (pre-D-Q1-split schema)"}
    return None


def _parse_str_gate(strd: dict) -> dict:
    stale = _stale_or_gate(strd["gate"], "build_str_decomposition.py")
    if stale:
        return stale
    g = strd["gate"]
    lib_share_pct = (strd["lib_share_of_month_end"] or 0) * 100
    direction = bool(g["direction_pass"])
    return {
        "direction_pass": direction,
        "direction_reproduced": g.get("direction_reproduced"),
        "lib_share_pct": lib_share_pct,
        "month_end_pct": strd["month_end"]["mean_pct"],
        "month_begin_pct": strd["month_begin"]["mean_pct"],
        "verdict": (
            f"direction {'reproduced' if direction else 'NOT reproduced'} "
            f"(sign-aware); LIB share {lib_share_pct:.1f}% of month-end"
        ),
    }


def _parse_mom6_gate(mom6: dict) -> dict:
    stale = _stale_or_gate(mom6["gate"], "run_mom6_lab_gate.py")
    if stale:
        return stale
    g = mom6["gate"]
    ep = mom6["ex_post_biased"]["mean_pct"]
    ea = mom6["ex_ante_corrected"]["mean_pct"]
    gap = mom6["ep_minus_ea_gap"]["mean_pct"]
    direction = bool(g["direction_pass"])
    return {
        "direction_pass": direction,
        "direction_reproduced": g.get("direction_reproduced"),
        "ex_post_pct": ep,
        "ex_ante_pct": ea,
        "ep_minus_ea_pct": gap,
        "verdict": (
            f"ex-post {ep:+.2f}%/mo vs ex-ante {ea:+.2f}%/mo, EP−EA gap "
            f"{gap:+.2f} — look-ahead direction "
            f"{'confirmed' if direction else 'NOT confirmed'}"
        ),
    }


def _parse_leadlag_gate(leadlag: dict) -> dict:
    stale = _stale_or_gate(leadlag["gate"], "run_leadlag_gate.py")
    if stale:
        return stale
    g = leadlag["gate"]
    direction = bool(g["direction_pass"])
    drf_c = leadlag["arms"]["drf"]["corr_correct_vs_defective"]
    crf_c = leadlag["arms"]["crf"]["corr_correct_vs_defective"]
    lrf_c = leadlag["arms"]["lrf"]["corr_correct_vs_defective"]
    return {
        "direction_pass": direction,
        "drf_corr_defective": drf_c,
        "crf_corr_defective": crf_c,
        "lrf_corr_defective": lrf_c,
        "verdict": (
            f"lead/lag error collapses correlation (drf {drf_c:.2f}, crf "
            f"{crf_c:.2f}, lrf {lrf_c:.2f}); round-trip re-alignment "
            f"{'restores' if direction else 'does NOT restore'} it"
        ),
    }


def _safe_gate(json_obj, parser, source: str, missing_msg: str) -> dict:
    """Parse one differential gate — NON-GATING and CRASH-PROOF: a missing JSON
    -> the 'NOT RUN' marker; a stale/partial/malformed JSON -> a status marker,
    never an exception. The report must be written regardless of the
    differential's health (it feeds no pass), so a KeyError/TypeError from a
    partially-written producer JSON must degrade to a status, not crash main()."""
    if not json_obj:
        return {"status": missing_msg}
    try:
        return parser(json_obj)
    except (KeyError, TypeError, ValueError, IndexError) as exc:
        return {"status": f"MALFORMED — re-run {source} (differential JSON missing/"
                          f"malformed expected keys: {exc!r}); NON-GATING, diagnostic only"}


def read_differential_diagnostic() -> dict:
    """The §7 gate-5 bias differential, reported as an RQ3-facing diagnostic.

    Reads the three per-gate JSONs; a missing JSON -> 'NOT RUN', a stale-schema
    JSON -> 'STALE_SCHEMA', a partial/malformed JSON -> 'MALFORMED'. Purely
    descriptive and crash-proof: nothing it returns feeds the RQ2 pass, and a
    degraded differential can never prevent the summary from being written.
    """
    gates = {
        "str_lib_decomposition": _safe_gate(
            _read("str_decomposition.json"), _parse_str_gate,
            "build_str_decomposition.py", "NOT RUN — run build_str_decomposition.py"),
        "mom6_lab": _safe_gate(
            _read("mom6_lab_gate.json"), _parse_mom6_gate,
            "run_mom6_lab_gate.py", "NOT RUN — run run_mom6_lab_gate.py"),
        "bbw_lead_lag": _safe_gate(
            _read("leadlag_gate.json"), _parse_leadlag_gate,
            "run_leadlag_gate.py", "NOT RUN — run run_leadlag_gate.py"),
    }
    return {
        "status": "NON-GATING RQ3-facing diagnostic (v1.6): the bias-mechanism "
                  "differential no longer gates RQ2 anchor fidelity.",
        "gates": gates,
    }


# ---------------------------------------------------------------------------
# Assemble + run
# ---------------------------------------------------------------------------

def assemble_report(verdict: dict, differential: dict, anchor_criterion: str) -> dict:
    """Assemble the summary. The gate-bearing portion (``construction_rulebook_pass``
    + ``anchors``) does not read ``differential`` -- byte-identical with the
    diagnostic block present or absent (pinned by test)."""
    return {
        "run_timestamp": datetime.now(timezone.utc).isoformat(),
        "thresholds_sha256": thresholds_sha256(),
        "anchor_criterion": anchor_criterion,
        "construction_rulebook_pass": verdict["construction_rulebook_pass"],
        "schema_note": SCHEMA_NOTE,
        "principle": (
            "RQ2 anchor fidelity = contract §7 gates 1-2: construction invariants "
            "(direction, group count, weighting, signal lag, holding period, leg "
            "identities) + rulebook byte-equality modulo the authorised register "
            "(G2). Published factor LEVELS are never targets; the §7 gate-5 bias "
            "differential is a reported RQ3-facing diagnostic (v1.6 supersedes "
            "O5 / D-Q1)."
        ),
        "anchors": verdict["anchors"],
        "differential_diagnostic": differential,
    }


def main():
    with open(THRESHOLDS_FILE) as f:
        vcfg = yaml.safe_load(f).get("validation", {})
    anchor_criterion = vcfg.get("anchor_criterion", "construction_rulebook")

    verdict = compute_construction_rulebook_verdict()
    differential = read_differential_diagnostic()
    report = assemble_report(verdict, differential, anchor_criterion)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    tmp = OUT.with_suffix(".json.tmp")
    with open(tmp, "w") as f:
        json.dump(report, f, indent=2)
    os.replace(tmp, OUT)

    print(f"RQ2 anchor fidelity — contract §7 gates 1-2 (criterion: {anchor_criterion}):")
    for a, v in verdict["anchors"].items():
        if v["pass"]:
            print(f"  {a:6s}: PASS  (rulebook byte-equal + holding_period="
                  f"{v['expected_holding_period']})")
        else:
            print(f"  {a:6s}: FAIL  (byte_equal={v['rulebook_byte_equal']}, "
                  f"holding_match={v['holding_period_match']}"
                  f"{'; ' + v['error'] if v['error'] else ''})")
    print("  differential (§7 gate 5): NON-GATING RQ3-facing diagnostic")
    ok = verdict["construction_rulebook_pass"]
    print(f"  CONSTRUCTION+RULEBOOK GATE (§7 gates 1-2): "
          f"{'PASS' if ok else 'FAIL'}  → {OUT}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
