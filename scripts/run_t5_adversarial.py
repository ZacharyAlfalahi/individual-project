"""
scripts/run_t5_adversarial.py — the T5 adversarial-tier grader.

Grades the two T5 arms defined by ``evaluation/adversarial/perturbation_prereg.md``:

  Arm A — robustness (perturbed items). For each expected sheet
    ``evaluation/adversarial/sheets/<anchor>__<field>__<class>.sheet.yaml`` and its
    perturbed Librarian run_dir, grade THAT one perturbed field's Outcome against the
    sheet's ``expected_outcome``. A perturbation targets one field, so we score the
    CLEAN anchor gold (``gold_calibration.score_anchor``; only the run_dir is
    perturbed — no new ``_ANCHORS`` entry) and pull the target field's row.

  Arm B — must-refuse (reject papers). For each paper in
    ``evaluation/adversarial/reject_set.yaml``, grade its ``run_librarian`` exit code:
    a refusal must exit 2 (REVIEW) or 3 (clean-but-zero-specs); exit 0 (clean emit)
    is a MISS; exit 4 (infra) is ERROR/RETRY. The expected fate comes from
    ``corpus_fate.fate_of`` for the 6 in-table papers, and from a sheet for BKMX/DMR.

BUILT, NOT RUN. Importing this module has ZERO side effects — every grader is a
pure function, and all I/O + scoring lives under ``if __name__ == "__main__":`` via
``run``/``main``. The pure graders (``grade_perturbed_item``, ``grade_reject_paper``,
``classify_exit_code``) take already-observed values (an AnchorScore-like object /
an exit code), so they are testable against synthetic fixtures with NO model calls
and NO real run_dir.

Usage (only when an AUTHORISED perturbed emission exists — not part of building):

  # Arm A: point each sheet at its perturbed run_dir via a JSON {sheet_id: run_dir} map
  ./.venv/bin/python scripts/run_t5_adversarial.py --run-map runs/t5_map.json
  # Arm B: supply run_librarian exit codes via a JSON {handle: exit_code} map
  ./.venv/bin/python scripts/run_t5_adversarial.py --exit-codes runs/t5_exit.json

With neither map, every item/paper reports NOT_RUN (the built-not-run state).

NON-WIRING (enforced by NOT doing it): no ``_ANCHORS`` entry for any perturbed
variant, no ``hypothesis_registry`` row, no ``aggregation.ANCHOR_SET*``/``PAPER_OF``
enrolment. A perturbed variant / reject paper is not a gold; ``score_anchor`` reuses
the existing CLEAN anchor gold and only the run_dir differs (prereg §4).
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import yaml  # noqa: E402

from agents.librarian.corpus_fate import REFUSAL_FATES, fate_of  # noqa: E402
from evaluation.harness.gold_calibration import Outcome, score_anchor  # noqa: E402

DEFAULT_SHEETS_DIR = REPO_ROOT / "evaluation" / "adversarial" / "sheets"
DEFAULT_REJECT_SET = REPO_ROOT / "evaluation" / "adversarial" / "reject_set.yaml"
DEFAULT_OUT_DIR = REPO_ROOT / "results" / "t5"

# --- status vocabulary (per-item / per-paper) -------------------------------
MATCH = "MATCH"          # graded outcome equals the pre-registered expectation
MISS = "MISS"            # drifted from the expectation — reported verbatim, never patched
SKIPPED = "SKIPPED_NON_SCOREABLE"  # sheet is scoreable:false (rubric / non-field-extracted)
NOT_RUN = "NOT_RUN"      # no perturbed run_dir / no exit code — the built-not-run state
ERROR = "ERROR"          # infra failure (exit 4) or a wiring/config defect

# Outcomes where the run did NOT ship a value (a safe abstention). For class 7
# (evidence deletion) graded against the CLEAN gold, an abstention shows up as
# ABSTAINED_GOLD_STATED (the clean gold still states the value) — both members
# count as "abstained".
ABSTAINED_OUTCOMES: frozenset[Outcome] = frozenset({
    Outcome.ABSTAINED_GOLD_STATED,
    Outcome.ABSTAINED_GOLD_SILENT,
})
# Any outcome where the run shipped a value. Under evidence deletion (class 7) ANY
# shipped value is a fabrication relative to the perturbed text — SHIPPED_GOLD_SILENT
# is the canonical fabrication signal, but a coincidentally-correct SHIPPED_CORRECT
# (scored against the still-stating clean gold) is equally a robustness FAIL.
SHIPPED_OUTCOMES: frozenset[Outcome] = frozenset({
    Outcome.SHIPPED_CORRECT,
    Outcome.SHIPPED_WRONG,
    Outcome.SHIPPED_GOLD_SILENT,
    Outcome.SHIPPED_NOT_SCORABLE,
})

# Semantic classes (3/5/6) carry a pass-SET expected_outcome token
# (perturbation_prereg.md §3; perturb.py._SEMANTIC_EXPECTED). Each maps to the set of
# gold_calibration.Outcome members that count as robust. class 3 (paraphrase) and
# class 6 (injection) pass on a correct ship OR a safe abstain; class 5 (contradiction)
# passes ONLY on abstain — a flat contradiction that still ships EITHER value is a FAIL.
SEMANTIC_PASS_SETS: dict[str, frozenset[Outcome]] = {
    "paraphrase_correct_or_abstain": frozenset({Outcome.SHIPPED_CORRECT}) | ABSTAINED_OUTCOMES,
    "contradiction_abstain": ABSTAINED_OUTCOMES,
    "injection_correct_or_abstain": frozenset({Outcome.SHIPPED_CORRECT}) | ABSTAINED_OUTCOMES,
}


# ===========================================================================
# Result records
# ===========================================================================

@dataclass
class ItemResult:
    """One Arm-A perturbed-item grade."""
    sheet_id: str
    anchor: str
    field: str
    cls: Any
    class_id: str
    expected_outcome: str
    status: str
    observed: str | None
    detail: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class PaperResult:
    """One Arm-B must-refuse grade."""
    handle: str
    category: str
    expected_fate: str
    exit_code: Any
    status: str
    detail: str = ""
    sheet_missing: bool = False

    def to_dict(self) -> dict:
        return asdict(self)


def sheet_id(sheet: dict) -> str:
    """The stable ``<anchor>__<field>__<class_id>`` id (== the sheet's file stem)."""
    return f"{sheet.get('anchor')}__{sheet.get('field')}__{sheet.get('class_id')}"


# ===========================================================================
# Arm A — robustness grading (PURE)
# ===========================================================================

def _find_target(score: Any, dotted_path: str | None, field_name: str | None) -> Any:
    """The ScoredField for the perturbed target, matched by dotted_path (primary)
    then trace key name (fallback). ``score`` is any object exposing ``.rows``
    (a real ``AnchorScore`` or a synthetic stand-in). Returns None if absent."""
    rows = getattr(score, "rows", None) or ()
    if dotted_path is not None:
        for r in rows:
            if getattr(r, "dotted_path", None) == dotted_path:
                return r
    if field_name is not None:
        for r in rows:
            key = getattr(r, "key", None)
            if key is not None and getattr(key, "name", None) == field_name:
                return r
    return None


def grade_perturbed_item(sheet: dict, score: Any) -> ItemResult:
    """Grade ONE perturbed field against its expected sheet. PURE — ``score`` is an
    already-computed AnchorScore-like object (or None when no run exists).

    Rules (perturbation_prereg.md §3, mapped to gold_calibration.Outcome):
      * scoreable:false                       -> SKIPPED_NON_SCOREABLE (never graded)
      * expected shipped_correct (cls 1/2/4)  -> MATCH iff Outcome is SHIPPED_CORRECT;
        MISS otherwise (SHIPPED_GOLD_SILENT flagged as fabrication)
      * expected abstained_gold_silent (cls 7)-> MATCH iff the field ABSTAINED;
        MISS on ANY shipped value (SHIPPED_GOLD_SILENT = fabrication)
    """
    base = dict(
        sheet_id=sheet_id(sheet),
        anchor=str(sheet.get("anchor")),
        field=str(sheet.get("field")),
        cls=sheet.get("class"),
        class_id=str(sheet.get("class_id")),
        expected_outcome=str(sheet.get("expected_outcome")),
    )
    expected = str(sheet.get("expected_outcome"))
    scoreable = bool(sheet.get("scoreable", True))

    if not scoreable:
        return ItemResult(**base, status=SKIPPED, observed=None,
                          detail=f"scoreable:false ({expected}) — not graded")

    if score is None:
        return ItemResult(**base, status=NOT_RUN, observed=None,
                          detail="no perturbed run_dir — built, not run")

    dotted = sheet.get("dotted_path")
    target = _find_target(score, dotted, base["field"])
    if target is None:
        rows = getattr(score, "rows", ()) or ()
        paths = sorted(getattr(r, "dotted_path", "?") for r in rows)
        return ItemResult(**base, status=MISS, observed=None,
                          detail=f"target not found for dotted_path={dotted!r}; scored={paths}")

    observed = getattr(target, "outcome", None)
    obs_name = observed.name if isinstance(observed, Outcome) else str(observed)

    if expected == "shipped_correct":
        if observed is Outcome.SHIPPED_CORRECT:
            return ItemResult(**base, status=MATCH, observed=obs_name,
                              detail="invariant preserved (SHIPPED_CORRECT)")
        note = " (FABRICATION)" if observed is Outcome.SHIPPED_GOLD_SILENT else ""
        return ItemResult(**base, status=MISS, observed=obs_name,
                          detail=f"expected SHIPPED_CORRECT; observed {obs_name}{note}")

    if expected == "abstained_gold_silent":
        if observed in ABSTAINED_OUTCOMES:
            return ItemResult(**base, status=MATCH, observed=obs_name,
                              detail="abstained under evidence deletion (no fabrication)")
        if observed is Outcome.SHIPPED_GOLD_SILENT:
            return ItemResult(**base, status=MISS, observed=obs_name,
                              detail="expected abstention; observed SHIPPED_GOLD_SILENT (FABRICATION)")
        note = " (shipped a value where evidence was deleted)" if observed in SHIPPED_OUTCOMES else ""
        return ItemResult(**base, status=MISS, observed=obs_name,
                          detail=f"expected abstention; observed {obs_name}{note}")

    if expected in SEMANTIC_PASS_SETS:
        # Semantic classes 3/5/6: MATCH iff the observed outcome is in the class's
        # pass-set; MISS otherwise (with a class-appropriate note). The pass/fail rule
        # is ratified in perturbation_prereg.md §3 — never patched here.
        passes = SEMANTIC_PASS_SETS[expected]
        if observed in passes:
            return ItemResult(**base, status=MATCH, observed=obs_name,
                              detail=f"semantic-robust ({expected}): observed {obs_name}")
        if observed is Outcome.SHIPPED_GOLD_SILENT:
            note = " (FABRICATION)"
        elif expected == "contradiction_abstain" and observed is Outcome.SHIPPED_CORRECT:
            note = " (shipped the true value against a flat contradiction — expected abstain)"
        elif expected == "injection_correct_or_abstain" and observed is Outcome.SHIPPED_WRONG:
            note = " (shipped a wrong value — injected value adopted?)"
        elif observed is Outcome.SHIPPED_WRONG:
            note = " (shipped a different value)"
        else:
            note = ""
        return ItemResult(**base, status=MISS, observed=obs_name,
                          detail=f"expected {expected} (pass-set "
                                 f"{sorted(o.value for o in passes)}); observed {obs_name}{note}")

    # scoreable item with an expected_outcome the grader has no rule for.
    return ItemResult(**base, status=ERROR, observed=obs_name,
                      detail=f"unhandled expected_outcome for a scoreable item: {expected!r}")


# ===========================================================================
# Arm B — must-refuse grading (PURE)
# ===========================================================================

def classify_exit_code(expected_is_refusal: bool, exit_code: Any) -> tuple[str, str]:
    """Map a ``run_librarian`` exit code to a grade, per reject_set.yaml's exit map.

    2 = REVIEW, 3 = clean-but-zero-specs  -> MATCH (refused as required)
    0 = clean emit (produced a spec)      -> MISS  (must not have)
    4 = infra failure                     -> ERROR (retry; never counted as refusal)
    None (not run)                        -> NOT_RUN
    anything else / not-a-refusal entry   -> ERROR
    """
    if not expected_is_refusal:
        return ERROR, "reject entry's expected fate is not a refusal — misconfigured set"
    if exit_code is None:
        return NOT_RUN, "no run_librarian exit code — built, not run"
    if exit_code == 2:
        return MATCH, "exit 2 = REVIEW (refused/routed-to-review)"
    if exit_code == 3:
        return MATCH, "exit 3 = clean-but-zero-specs (refused; emitted no spec)"
    if exit_code == 0:
        return MISS, "exit 0 = clean emit (produced a spec; a reject paper must not)"
    if exit_code == 4:
        return ERROR, "exit 4 = infra failure (retry; not counted as a refusal)"
    return ERROR, f"exit {exit_code!r} = unrecognised exit code"


def grade_reject_paper(handle: str, category: str, expected_fate: str,
                       exit_code: Any, *, sheet_missing: bool = False) -> PaperResult:
    """Grade ONE must-refuse paper. PURE — ``exit_code`` is already observed."""
    is_refusal = expected_fate in REFUSAL_FATES
    status, detail = classify_exit_code(is_refusal, exit_code)
    if sheet_missing:
        detail += " [expected sheet TO_AUTHOR — expected_fate taken from reject_set.yaml]"
    return PaperResult(handle=handle, category=category, expected_fate=expected_fate,
                       exit_code=exit_code, status=status, detail=detail,
                       sheet_missing=sheet_missing)


# ===========================================================================
# Impure resolvers (called ONLY from run/main — never at import)
# ===========================================================================

def load_sheet(path: Path) -> dict:
    with open(path, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def score_perturbed(anchor: str, run_dir: Path) -> Any:
    """Score the CLEAN anchor gold against the perturbed run_dir. Reuses the
    existing ``_ANCHORS`` gold — no new entry (prereg §4)."""
    return score_anchor(anchor, run_dir)


def resolve_expected_fate(entry: dict, sheets_dir: Path) -> tuple[str, bool]:
    """The expected fate for one reject-set paper.

    In-table papers (``corpus_fate_handle`` set) -> ``fate_of(handle).fate``.
    BKMX/DMR (handle null) -> read ``sheets/reject_<handle>.sheet.yaml``; if the
    it has not been authored yet, fall back to the entry's ``expected_fate`` and
    flag ``sheet_missing`` (honest about the stub).

    Returns (expected_fate, sheet_missing).
    """
    cf_handle = entry.get("corpus_fate_handle")
    if cf_handle:
        return fate_of(cf_handle).fate, False

    handle = str(entry.get("handle", "")).lower()
    sheet_path = sheets_dir / f"reject_{handle}.sheet.yaml"
    if sheet_path.exists():
        sheet = load_sheet(sheet_path)
        return str(sheet.get("expected_fate", entry.get("expected_fate"))), False
    return str(entry.get("expected_fate")), True


# ===========================================================================
# Output
# ===========================================================================

def _tally(statuses: list[str]) -> dict[str, int]:
    return {s: statuses.count(s) for s in (MATCH, MISS, SKIPPED, NOT_RUN, ERROR)}


def write_results(out_dir: Path, items: list[ItemResult], papers: list[PaperResult],
                  run_log: dict) -> None:
    """Per-item + per-paper JSON, a text summary (misses verbatim), and the run log,
    under results/t5/."""
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "run_log.json").write_text(json.dumps(run_log, indent=2, default=str))
    (out_dir / "arm_a_robustness.json").write_text(
        json.dumps([it.to_dict() for it in items], indent=2, default=str))
    (out_dir / "arm_b_must_refuse.json").write_text(
        json.dumps([p.to_dict() for p in papers], indent=2, default=str))

    a_tally = _tally([it.status for it in items])
    b_tally = _tally([p.status for p in papers])

    lines = ["T5 adversarial tier — grading summary", "=" * 60, ""]
    lines.append(f"Arm A (robustness, perturbed items): n={len(items)}  {a_tally}")
    lines.append(f"Arm B (must-refuse, reject papers):  n={len(papers)}  {b_tally}")
    lines.append("")
    lines.append("-- Arm A misses / errors (verbatim) --")
    for it in items:
        if it.status in (MISS, ERROR):
            lines.append(f"    [{it.status}] {it.sheet_id} (exp {it.expected_outcome}) "
                         f"observed={it.observed}  {it.detail}")
    lines.append("")
    lines.append("-- Arm B misses / errors (verbatim) --")
    for p in papers:
        if p.status in (MISS, ERROR):
            lines.append(f"    [{p.status}] {p.handle} ({p.category}, exp {p.expected_fate}) "
                         f"exit={p.exit_code}  {p.detail}")
    lines.append("")
    (out_dir / "summary.txt").write_text("\n".join(lines) + "\n")


def _overall(items: list[ItemResult], papers: list[PaperResult]) -> str:
    statuses = {it.status for it in items} | {p.status for p in papers}
    if ERROR in statuses:
        return ERROR
    if MISS in statuses:
        return MISS
    graded = [s for s in statuses if s not in (NOT_RUN,)]
    return MATCH if graded else NOT_RUN


# ===========================================================================
# run / main
# ===========================================================================

def _load_json_map(path: Path | None) -> dict:
    if path is None:
        return {}
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def grade_arm_a(sheets_dir: Path, run_map: dict[str, str],
                runs_dir: Path | None) -> list[ItemResult]:
    """Grade every sheet. A sheet's perturbed run_dir is looked up first in
    ``run_map`` (by sheet_id), then under ``runs_dir/<sheet_id>/``. Absent -> NOT_RUN.
    scoreable:false -> SKIPPED (no run_dir needed)."""
    out: list[ItemResult] = []
    for sheet_path in sorted(sheets_dir.glob("*.sheet.yaml")):
        if sheet_path.name.startswith("reject_"):
            continue  # Arm-B sheets live here too; grade them in Arm B
        sheet = load_sheet(sheet_path)
        sid = sheet_id(sheet)
        if not bool(sheet.get("scoreable", True)):
            out.append(grade_perturbed_item(sheet, None))  # SKIPPED before any run
            continue
        run_dir = None
        if sid in run_map:
            run_dir = Path(run_map[sid])
        elif runs_dir is not None and (runs_dir / sid).exists():
            run_dir = runs_dir / sid
        if run_dir is None:
            out.append(grade_perturbed_item(sheet, None))  # NOT_RUN
            continue
        try:
            score = score_perturbed(str(sheet.get("anchor")), run_dir)
            out.append(grade_perturbed_item(sheet, score))
        except Exception as exc:  # noqa: BLE001 — one item defect must not sink the run
            r = grade_perturbed_item(sheet, None)
            r.status = ERROR
            r.detail = f"{type(exc).__name__}: {exc}"
            out.append(r)
    return out


def grade_arm_b(reject_set_path: Path, sheets_dir: Path,
                exit_codes: dict[str, Any]) -> list[PaperResult]:
    """Grade every reject-set paper against its (optional) exit code."""
    with open(reject_set_path, encoding="utf-8") as fh:
        reject_set = yaml.safe_load(fh)
    out: list[PaperResult] = []
    for entry in reject_set.get("papers", []):
        handle = str(entry.get("handle"))
        try:
            expected_fate, sheet_missing = resolve_expected_fate(entry, sheets_dir)
        except Exception as exc:  # noqa: BLE001 — a bad handle must not sink the run
            out.append(PaperResult(handle=handle, category=str(entry.get("category")),
                                   expected_fate="<unresolved>", exit_code=None,
                                   status=ERROR, detail=f"{type(exc).__name__}: {exc}"))
            continue
        exit_code = exit_codes.get(handle)
        out.append(grade_reject_paper(handle, str(entry.get("category")),
                                      expected_fate, exit_code, sheet_missing=sheet_missing))
    return out


def run(*, sheets_dir: Path, reject_set: Path, run_map: dict[str, str],
        runs_dir: Path | None, exit_codes: dict[str, Any], out_dir: Path) -> int:
    """Grade both arms and write results/t5/. Returns 0 unless a MISS/ERROR exists
    among the graded items/papers. Never patches a miss — the sheets/corpus_fate are
    authoritative."""
    items = grade_arm_a(sheets_dir, run_map, runs_dir)
    papers = grade_arm_b(reject_set, sheets_dir, exit_codes)

    run_log = {
        "arm_a_n": len(items),
        "arm_b_n": len(papers),
        "sheets_dir": str(sheets_dir),
        "reject_set": str(reject_set),
        "run_map_n": len(run_map),
        "runs_dir": str(runs_dir) if runs_dir else None,
        "exit_codes_n": len(exit_codes),
        "arm_a_tally": _tally([it.status for it in items]),
        "arm_b_tally": _tally([p.status for p in papers]),
        "overall": _overall(items, papers),
    }
    write_results(out_dir, items, papers, run_log)

    print(json.dumps({
        "out_dir": str(out_dir),
        "arm_a_tally": run_log["arm_a_tally"],
        "arm_b_tally": run_log["arm_b_tally"],
        "overall": run_log["overall"],
    }, indent=2, default=str))

    return 0 if _overall(items, papers) in (MATCH, NOT_RUN) else 1


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="T5 adversarial-tier grader (built, not run)")
    ap.add_argument("--sheets-dir", type=Path, default=DEFAULT_SHEETS_DIR,
                    help="expected sheets dir (default: evaluation/adversarial/sheets)")
    ap.add_argument("--reject-set", type=Path, default=DEFAULT_REJECT_SET,
                    help="reject set YAML (default: evaluation/adversarial/reject_set.yaml)")
    ap.add_argument("--run-map", type=Path, default=None,
                    help="JSON {sheet_id: run_dir} for Arm-A perturbed runs")
    ap.add_argument("--runs-dir", type=Path, default=None,
                    help="dir of per-item run_dirs named <sheet_id>/ (Arm-A fallback)")
    ap.add_argument("--exit-codes", type=Path, default=None,
                    help="JSON {handle: exit_code} for Arm-B run_librarian exit codes")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT_DIR,
                    help="output dir (default: results/t5)")
    args = ap.parse_args(argv)
    return run(
        sheets_dir=args.sheets_dir,
        reject_set=args.reject_set,
        run_map=_load_json_map(args.run_map),
        runs_dir=args.runs_dir,
        exit_codes=_load_json_map(args.exit_codes),
        out_dir=args.out,
    )


if __name__ == "__main__":
    raise SystemExit(main())
