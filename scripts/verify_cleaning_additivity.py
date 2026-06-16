"""
Cross-stage row-count additivity verifier for the corrected TRACE cleaning chain.

Each cleaning script already verifies its OWN row arithmetic against the parquet
it writes (`parquet_row_count_verified`). What no single script can see is the
HANDOFF between stages: that the rows one stage emits are exactly the rows the
next stage ingests. This verifier closes that gap by reading the three
trade-level stage reports and asserting both the per-stage internal sums and the
inter-stage handoffs.

Scope (DEVELOPMENT partition, CORRECTED family):

    preprocess_trace.py            (cleaning_report.json,      rows.*)
      → development_rows ────────────────────────────────┐
    apply_decimal_shift.py         (decimal_shift_report.json, rows_dev.*)
      → input_rows == development_rows  ◄─────────────────┘
      → output_rows ────────────────────────────────────┐
    bounce_back_filter.py          (bounce_back_report.json,   rows_dev.*)
      → input_rows == output_rows  ◄──────────────────────┘

The chain STOPS at bounce-back: build_daily_panel.py group-bys trades into
cusip-days, so trade-level row additivity is no longer meaningful past that
point (its report carries no `input_rows`). The RAW family bypasses
decimal-shift + bounce-back entirely (preprocess → build_daily_panel directly),
so it has no decimal-shift/bounce-back report and is not chained here.

Holdout is not reconciled during development (no holdout decimal-shift/bounce
report exists until the single post-freeze run in weeks 13-14); this verifier
reads only the development-side reports and never touches `/data/holdout/`.

The checking logic (`check_additivity`) is a pure function over the three report
dicts so it is unit-testable without running the pipeline.

Usage:
  python scripts/verify_cleaning_additivity.py
Exit code 0 = all checks pass; 1 = at least one additivity check failed.
"""

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
CLEANING_REPORT = REPO_ROOT / "data" / "development" / "cleaning_report.json"
DECIMAL_SHIFT_REPORT = REPO_ROOT / "data" / "development" / "decimal_shift_report.json"
BOUNCE_BACK_REPORT = REPO_ROOT / "data" / "development" / "bounce_back_report.json"


def load_report(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(
            f"Stage report not found: {path}. Run the cleaning chain "
            "(preprocess_trace → apply_decimal_shift → bounce_back_filter) first."
        )
    with open(path) as f:
        return json.load(f)


def _check(name: str, lhs: int, rhs: int) -> dict:
    """One equality check; returns a structured result row."""
    ok = lhs == rhs
    return {"name": name, "ok": ok, "lhs": lhs, "rhs": rhs,
            "detail": f"{lhs:,} {'==' if ok else '!='} {rhs:,}"}


# Fields each stage report must carry for the additivity checks below.
_REQUIRED = {
    "cleaning": ("rows.raw_total", "rows.dropped_trc_st",
                 "rows.dropped_cancelled_original", "rows.dropped_asof_cd",
                 "rows.dropped_wis_fl", "rows.dropped_interdealer_duplicate",
                 "rows.dropped_invalid_date", "rows.development_rows",
                 "rows.holdout_rows", "rows.final_clean_total"),
    "decimal_shift": ("rows_dev.input_rows", "rows_dev.output_rows",
                      "rows_dev.dropped_pre_ceiling", "rows_dev.dropped_floor",
                      "rows_dev.kept_no_shift", "rows_dev.kept_shift_div10",
                      "rows_dev.kept_shift_div100"),
    "bounce_back": ("rows_dev.input_rows", "rows_dev.kept_rows",
                    "rows_dev.dropped_bounce_back"),
}


def _dig(report: dict, dotted: str):
    """Walk a dotted path; return (value, present)."""
    cur = report
    for part in dotted.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return None, False
        cur = cur[part]
    return cur, True


def _missing_field_failures(cleaning, decimal_shift, bounce_back) -> list:
    """Required fields that are absent or null — returned as check-shaped
    failures so a stale/old-format report yields a clear message instead of a
    bare KeyError mid-computation."""
    failures = []
    for stage, report in (("cleaning", cleaning),
                          ("decimal_shift", decimal_shift),
                          ("bounce_back", bounce_back)):
        for dotted in _REQUIRED[stage]:
            val, present = _dig(report, dotted)
            if not present or val is None:
                name = f"field present: {stage}.{dotted}"
                failures.append({"name": name, "ok": False,
                                 "detail": "missing or null — re-run the stage"})
    return failures


def check_additivity(cleaning: dict, decimal_shift: dict, bounce_back: dict) -> dict:
    """Pure check over the three stage-report dicts (dev, corrected chain).

    Returns {"checks": [...], "failures": [...], "warnings": [...], "ok": bool}.
    Each check is an equality the cleaning chain must satisfy exactly; any
    discrepancy means rows are silently lost or double-counted between stages.
    """
    # Validate required fields up front so a stale/old-format report produces a
    # clear "missing field — re-run the stage" failure rather than a KeyError.
    field_failures = _missing_field_failures(cleaning, decimal_shift, bounce_back)
    if field_failures:
        return {"checks": field_failures, "failures": field_failures,
                "warnings": [], "ok": False}

    c = cleaning["rows"]
    d = decimal_shift["rows_dev"]
    b = bounce_back["rows_dev"]

    checks = [
        # --- preprocess_trace internal (Dick-Nielsen only) ---
        _check(
            "preprocess: raw_total == Σdrops + dev + holdout",
            c["raw_total"],
            (c["dropped_trc_st"] + c["dropped_cancelled_original"]
             + c["dropped_asof_cd"] + c["dropped_wis_fl"]
             + c["dropped_interdealer_duplicate"] + c["dropped_invalid_date"]
             + c["development_rows"] + c["holdout_rows"]),
        ),
        _check(
            "preprocess: final_clean_total == dev + holdout",
            c["final_clean_total"],
            c["development_rows"] + c["holdout_rows"],
        ),
        # --- apply_decimal_shift internal ---
        _check(
            "decimal_shift: input == output + dropped_pre_ceiling + dropped_floor",
            d["input_rows"],
            d["output_rows"] + d["dropped_pre_ceiling"] + d["dropped_floor"],
        ),
        _check(
            "decimal_shift: output == kept_no_shift + div10 + div100",
            d["output_rows"],
            d["kept_no_shift"] + d["kept_shift_div10"] + d["kept_shift_div100"],
        ),
        # --- bounce_back internal ---
        _check(
            "bounce_back: input == kept + dropped",
            b["input_rows"],
            b["kept_rows"] + b["dropped_bounce_back"],
        ),
        # --- inter-stage handoffs (dev, corrected family) ---
        _check(
            "handoff: preprocess.development_rows == decimal_shift.input_rows",
            c["development_rows"],
            d["input_rows"],
        ),
        _check(
            "handoff: decimal_shift.output_rows == bounce_back.input_rows",
            d["output_rows"],
            b["input_rows"],
        ),
    ]

    failures = [ch for ch in checks if not ch["ok"]]

    # Provenance warning: the three reports can come from different pipeline
    # runs. A thresholds_sha256 mismatch is a legitimate "stale inputs" signal,
    # distinct from a logic bug — surface it rather than letting it mask one.
    warnings = []
    shas = {
        "cleaning": cleaning.get("thresholds_sha256"),
        "decimal_shift": decimal_shift.get("thresholds_sha256"),
        "bounce_back": bounce_back.get("thresholds_sha256"),
    }
    distinct = {s for s in shas.values() if s is not None}
    if len(distinct) > 1:
        warnings.append(
            "thresholds_sha256 differs across stages — reports may be from "
            f"different pipeline runs: {shas}"
        )

    # Surface any stage's own self-declared counts_caveat so a stale-report
    # failure (a report predating a counting fix) is self-explanatory rather
    # than looking like a live logic bug.
    for stage_name, rep in (("cleaning", cleaning),
                            ("decimal_shift", decimal_shift),
                            ("bounce_back", bounce_back)):
        caveat = rep.get("counts_caveat")
        if caveat:
            warnings.append(f"{stage_name} report carries a counts_caveat "
                            f"(likely stale — re-run the stage): {caveat}")

    return {"checks": checks, "failures": failures, "warnings": warnings,
            "ok": not failures}


def main() -> int:
    cleaning = load_report(CLEANING_REPORT)
    decimal_shift = load_report(DECIMAL_SHIFT_REPORT)
    bounce_back = load_report(BOUNCE_BACK_REPORT)

    result = check_additivity(cleaning, decimal_shift, bounce_back)

    print("Cross-stage cleaning additivity (development, corrected chain):")
    for ch in result["checks"]:
        print(f"  [{'PASS' if ch['ok'] else 'FAIL'}] {ch['name']}: {ch['detail']}")
    for w in result["warnings"]:
        print(f"  [WARN] {w}")

    if result["ok"]:
        print("All cross-stage additivity checks passed.")
        return 0
    print(f"FAILED: {len(result['failures'])} additivity check(s) broken — "
          "rows are being silently lost or double-counted in the chain.",
          file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
