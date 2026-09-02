#!/usr/bin/env python
"""
T5 Arm-A extraction runner (C-lever, 2026-09-02): one TARGETED Librarian run per
perturbed variant, feeding scripts/run_t5_adversarial.py.

For every Arm-A expected sheet (evaluation/adversarial/sheets/*.sheet.yaml,
excluding the reject_* Arm-B sheets and any ``scoreable: false`` sheet -- the
grader SKIPs those before needing a run, so paying for them would be pure waste),
this driver invokes run_librarian.main in-process with:

  * ``--canonical-text`` -> the variant's frozen text
    (evaluation/adversarial/frozen/<sheet_id>.frozen.yaml, machine-local/gitignored);
  * the ANCHOR's registered paper entry (its gold enumeration list -- verified:
    load_gold_list never relocates quotes, so the clean gold enum drives perturbed
    texts by design) via the anchor's PAPERS key;
  * ``--fields <target>`` -- the targeted-field allowlist: only the sheet's target
    field(s) (compound '+'-joined targets are split, every component asked) plus
    the always-run structural/paper_facts set is extracted (~8-10 calls per model
    instead of ~42). Every scoreable sheet's target is fully extractable (census
    2026-09-02); a violation fails loud rather than silently running
    structural-only;
  * ``--out runs/t5/<sheet_id>`` -- the grader's ``runs_dir`` convention, plus an
    explicit ``{sheet_id: run_dir}`` map written for ``--run-map``.

Skip-and-continue per variant (one defect must not sink the batch); resume-safe:
the disk replay cache makes a re-run of completed variants free, and each variant's
out_dir is idempotent. Exit codes are recorded per variant; the batch exits 0 when
every attempted variant produced a typed outcome.

Cost note: this runner is where the T5 registration's targeted-field amendment
(docs/evaluation/t5_targeted_mode_amendment_draft_2026-09-02.md) is exercised --
the amendment must be ratified before any paid (--phase report) Arm-A run.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import yaml

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from scripts import run_librarian  # noqa: E402  (same module object the tests patch)
from scripts.run_librarian import _EXTRACTABLE_FIELDS  # noqa: E402

SHEETS_DIR = _REPO_ROOT / "evaluation" / "adversarial" / "sheets"
FROZEN_DIR = _REPO_ROOT / "evaluation" / "adversarial" / "frozen"

# T5 anchors -> the registered PAPERS key whose gold enum drives the run.
PAPER_KEY_OF_ANCHOR = {"drf": "bbw", "mom6": "jnps", "str": "drr"}

_EXIT_NAMES = {0: "ok", 2: "review", 3: "zero_specs", 4: "paper_failed"}


def arm_a_sheets(sheets_dir: Path = SHEETS_DIR) -> list[dict]:
    """Load every scoreable Arm-A sheet, with its id and paths resolved."""
    out = []
    for p in sorted(sheets_dir.glob("*.sheet.yaml")):
        if p.name.startswith("reject_"):
            continue                                   # Arm B
        sheet = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
        sid = f"{sheet.get('anchor')}__{sheet.get('field')}__{sheet.get('class_id')}"
        out.append({
            "sheet_id": sid,
            "path": p,
            "anchor": sheet.get("anchor"),
            "field": sheet.get("field"),
            "scoreable": bool(sheet.get("scoreable", True)),
            "frozen": FROZEN_DIR / f"{sid}.frozen.yaml",
        })
    return out


def fields_arg_for(target_field: str) -> str:
    """The --fields value for one sheet's target. Compound targets are '+'-joined
    (e.g. ``long_leg+weighting_scheme``) -- every component must be asked, else the
    graded field is never extracted and the sheet reports a spurious NOT_ASKED MISS
    (review finding, 2026-09-02). Census fact: every SCOREABLE sheet's components
    are all extractable; the only ""-mapping targets (strategy_label,
    universe_filter) are exclusively non-scoreable and never reach extraction.
    A sheet that violates this fails LOUD here (recorded as CRASH, batch continues)
    rather than silently running structural-only."""
    components = target_field.split("+")
    unextractable = [c for c in components if c not in _EXTRACTABLE_FIELDS]
    if unextractable:
        raise ValueError(
            f"sheet target {target_field!r} has unextractable component(s) "
            f"{unextractable} -- a targeted run cannot ask the graded field; "
            "refusing to run structural-only (would grade NOT_ASKED -> MISS)")
    return ",".join(components)


def run_variants(sheets: list[dict], out_root: Path, passthrough: list[str]) -> dict:
    results: dict[str, dict] = {}
    run_map: dict[str, str] = {}
    crashed = False
    for s in sheets:
        sid = s["sheet_id"]
        if not s["scoreable"]:
            results[sid] = {"outcome": "skipped_non_scoreable"}
            continue
        if not s["frozen"].exists():
            results[sid] = {"outcome": "missing_frozen_text",
                            "frozen": str(s["frozen"])}
            continue
        paper_key = PAPER_KEY_OF_ANCHOR[s["anchor"]]
        out_dir = out_root / sid
        t0 = time.monotonic()
        try:
            argv = ["--paper", paper_key,
                    "--canonical-text", str(s["frozen"]),
                    "--fields", fields_arg_for(s["field"]),
                    "--out", str(out_dir),
                    *passthrough]
            code = run_librarian.main(argv)
        # SystemExit included: argparse errors are BaseException and would
        # otherwise sink the whole batch (review finding, 2026-09-02).
        except (Exception, SystemExit) as exc:  # noqa: BLE001 -- one variant must not sink the batch
            crashed = True
            results[sid] = {"outcome": f"CRASH: {type(exc).__name__}: {exc}"}
            print(f"[t5x] {sid}: CRASH {type(exc).__name__}: {exc}")
            continue
        results[sid] = {"outcome": _EXIT_NAMES.get(code, f"exit_{code}"), "exit": code,
                        "seconds": round(time.monotonic() - t0, 1)}
        run_map[sid] = str(out_dir)
        print(f"[t5x] {sid}: exit {code} ({results[sid]['outcome']})")
    return {"variants": results, "run_map": run_map, "crashed": crashed}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Targeted Librarian extraction over the T5 Arm-A variants.")
    ap.add_argument("--out-root", default=None,
                    help="root for per-variant run dirs (default runs/t5 -- the grader's runs_dir).")
    ap.add_argument("--run-map", default=None, dest="run_map_out",
                    help="write the {sheet_id: run_dir} JSON the grader's --run-map consumes.")
    ap.add_argument("--limit-variants", type=int, default=None,
                    help="run only the first N scoreable variants (smoke).")
    ap.add_argument("--anchor", default=None, choices=sorted(PAPER_KEY_OF_ANCHOR),
                    help="restrict to one anchor's variants.")
    # Forwarded verbatim to run_librarian per variant.
    ap.add_argument("--phase", default="fake", choices=("fake", "dev", "report"))
    ap.add_argument("--min-interval-s", type=float, default=None, dest="min_interval_s")
    ap.add_argument("--cache-dir", default="runs/librarian_cache", dest="cache_dir")
    ap.add_argument("--no-cache", action="store_true")
    args = ap.parse_args(argv)

    sheets = arm_a_sheets()
    if args.anchor:
        sheets = [s for s in sheets if s["anchor"] == args.anchor]
    if args.limit_variants is not None:
        # Keep every non-scoreable sheet (they cost nothing and are reported as
        # skipped) + the first N scoreable ones, preserving sorted-sheet order.
        scoreable_kept = {s["sheet_id"] for s in
                          [x for x in sheets if x["scoreable"]][: args.limit_variants]}
        sheets = [s for s in sheets if not s["scoreable"] or s["sheet_id"] in scoreable_kept]

    out_root = Path(args.out_root) if args.out_root else (_REPO_ROOT / "runs" / "t5")
    out_root.mkdir(parents=True, exist_ok=True)

    passthrough = ["--phase", args.phase, "--enumeration", "gold",
                   "--cache-dir", args.cache_dir]
    if args.min_interval_s is not None:
        passthrough += ["--min-interval-s", str(args.min_interval_s)]
    if args.no_cache:
        passthrough += ["--no-cache"]

    log = run_variants(sheets, out_root, passthrough)

    counts: dict[str, int] = {}
    for rec in log["variants"].values():
        key = rec["outcome"].split(":")[0]
        counts[key] = counts.get(key, 0) + 1
    log_path = out_root / "t5_extraction_log.json"
    # The log describes THIS invocation (filters recorded so a subset log is
    # self-describing and cannot masquerade as a full-batch record).
    log_path.write_text(json.dumps({
        "invocation": {"anchor": args.anchor, "limit_variants": args.limit_variants,
                       "phase": args.phase},
        "counts": counts, **log}, indent=2), encoding="utf-8")
    print(f"[t5x] {len(log['variants'])} sheet(s): {counts} -> {log_path}")

    if args.run_map_out:
        mp = Path(args.run_map_out)
        mp.parent.mkdir(parents=True, exist_ok=True)
        # MERGE into an existing map: a partial re-run (--anchor/--limit-variants)
        # must extend, never shrink, the grading input (review finding, 2026-09-02).
        merged: dict[str, str] = {}
        if mp.exists():
            merged = json.loads(mp.read_text(encoding="utf-8"))
        merged.update(log["run_map"])
        mp.write_text(json.dumps(merged, indent=2), encoding="utf-8")
        print(f"[t5x] run map ({len(merged)} entries) -> {mp}")

    return 1 if log["crashed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
