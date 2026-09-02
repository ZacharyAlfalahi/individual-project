#!/usr/bin/env python
"""
Corpus batch driver over run_librarian (B4): iterate a NAMED paper set, one
in-process ``run_librarian.main([...])`` call per paper, skip-and-continue on the
typed non-success exits, and emit the per-paper outcome maps downstream consumers
read.

Semantics (deliberate):
  * Paper sets are EXPLICIT named lists -- never ``sorted(PAPERS)``. PAPERS also
    holds the T4(b) synthetic instrument and the anchors; a blind "all" would
    sweep instruments into corpus outputs. ``--set`` picks the population.
  * Exit codes 2 (review), 3 (zero specs), 4 (paper_failed) are RECORDED and the
    loop continues -- one paper routing to review or exhausting retries must not
    abort a paid batch (the T5 one-defect-must-not-sink-the-run discipline).
  * A missing-API-key ``RealClientError`` escapes ``run_librarian.main`` from
    build_clients and is FATAL-FOR-ALL: the same key is missing for every paper,
    so continuing would burn the whole batch into paper_failed noise.
  * Any other unexpected exception is recorded as a CRASH for that paper and the
    loop continues; the batch exits non-zero so a crash is never silent.
  * Each paper gets a DISTINCT out dir (``<out-root>/<paper>``): the per-run raw
    archive carries a line-count integrity guard (run_artefacts.check_raw), and
    per-run manifests must not overwrite each other.
  * ``--exit-map`` writes the ``{handle: exit_code}`` JSON that
    scripts/run_t5_adversarial.py ``--exit-codes`` consumes (Arm B). Handles
    default to the PAPERS key; reject papers map through REJECT_HANDLE_OF.

The disk replay cache (run_librarian ``--cache-dir``, default ON) makes an
interrupted batch cheap to resume: re-running replays completed papers' calls
from disk at zero cost.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from scripts import run_librarian  # noqa: E402  (same module object the tests patch)
from agents.librarian.pipeline.real_client import RealClientError  # noqa: E402

# ---------------------------------------------------------------------------
# Named paper sets (PAPERS keys). Membership is EXPLICIT by design -- the corpus
# population is defined by the registered corpus records (docs/thresholds.yaml
# corpus.selection + docs/evaluation/corpus_inventory.md), not by what happens
# to be registered in PAPERS.
# ---------------------------------------------------------------------------

SETS: dict[str, tuple[str, ...]] = {
    "anchors": ("bbw", "jnps", "drr"),
    "corpus": ("bbw2021", "dfps"),        # T3 scale layer (RQ2 coverage denominator)
    "synth": ("synth",),                  # T4(b) instrument -- never a corpus member
}

# PAPERS key -> T5 reject-set handle (evaluation/adversarial/reject_set.yaml
# ``papers[].handle``). The "rejects" set derives from this map so the two cannot
# drift. Arm B runs live enumeration by construction (no gold_enum registered),
# and the exit map keyed by these handles feeds run_t5_adversarial --exit-codes.
REJECT_HANDLE_OF: dict[str, str] = {
    "reject_hxz": "HXZ",
    "reject_kpj": "KPJ",
    "reject_hlz": "HLZ",
    "reject_gkx": "GKX",
    "reject_gl": "GL",
    "reject_bpw": "BPW",
    "reject_bkmx": "BKMX",
    "reject_dmr": "DMR",
}
SETS["rejects"] = tuple(REJECT_HANDLE_OF)

# Exit-code names for the human-readable log (run_librarian's typed exits).
_EXIT_NAMES = {0: "ok", 2: "review", 3: "zero_specs", 4: "paper_failed"}


def run_set(papers: tuple[str, ...], out_root: Path, passthrough: list[str]) -> dict:
    """Run each paper in-process; return the batch log dict. ``passthrough`` is the
    argv tail forwarded to run_librarian.main (phase/enumeration/limit/cache...)."""
    results: dict[str, dict] = {}
    crashed = False
    for key in papers:
        out_dir = out_root / key
        argv = ["--paper", key, "--out", str(out_dir), *passthrough]
        print(f"[corpus] {key}: run_librarian {' '.join(argv)}")
        t0 = time.monotonic()
        try:
            code: int | None = run_librarian.main(argv)
        except RealClientError as exc:
            # Missing key at build_clients -- fatal for every paper. Abort loudly.
            raise SystemExit(
                f"[corpus] FATAL: client build failed ({exc}) -- the same key is missing "
                f"for every paper; fix the environment and re-run (completed papers replay "
                f"from the disk cache at zero cost)."
            ) from exc
        except Exception as exc:  # noqa: BLE001 -- one paper's crash must not sink the batch
            crashed = True
            code = None
            results[key] = {"exit": None, "outcome": f"CRASH: {type(exc).__name__}: {exc}",
                            "out": str(out_dir), "seconds": round(time.monotonic() - t0, 1)}
            print(f"[corpus] {key}: CRASH {type(exc).__name__}: {exc}")
            continue
        results[key] = {"exit": code, "outcome": _EXIT_NAMES.get(code, f"exit_{code}"),
                        "out": str(out_dir), "seconds": round(time.monotonic() - t0, 1)}
        print(f"[corpus] {key}: exit {code} ({results[key]['outcome']})")

    counts: dict[str, int] = {}
    for rec in results.values():
        counts[rec["outcome"] if rec["exit"] is not None else "crash"] = \
            counts.get(rec["outcome"] if rec["exit"] is not None else "crash", 0) + 1
    return {"papers": results, "counts": counts, "crashed": crashed}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Batch Librarian extraction over a named paper set.")
    ap.add_argument("--set", default="corpus", choices=sorted(SETS), dest="paper_set",
                    help="which named paper set to run (explicit lists; never 'everything').")
    ap.add_argument("--out-root", default=None,
                    help="root output directory; each paper gets <out-root>/<paper>. "
                         "Default: runs/corpus_<set>_<phase>.")
    ap.add_argument("--exit-map", default=None,
                    help="also write the {handle: exit_code} JSON map consumed by "
                         "scripts/run_t5_adversarial.py --exit-codes (Arm B).")
    # Everything below is forwarded verbatim to run_librarian.main per paper.
    ap.add_argument("--phase", default="fake", choices=("fake", "dev", "report"))
    ap.add_argument("--enumeration", default="gold", choices=("gold", "live"))
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--min-interval-s", type=float, default=None, dest="min_interval_s")
    ap.add_argument("--cache-dir", default="runs/librarian_cache", dest="cache_dir")
    ap.add_argument("--no-cache", action="store_true")
    args = ap.parse_args(argv)

    papers = SETS[args.paper_set]
    out_root = Path(args.out_root) if args.out_root else (
        _REPO_ROOT / "runs" / f"corpus_{args.paper_set}_{args.phase}")
    out_root.mkdir(parents=True, exist_ok=True)

    passthrough = ["--phase", args.phase, "--enumeration", args.enumeration,
                   "--cache-dir", args.cache_dir]
    if args.limit is not None:
        passthrough += ["--limit", str(args.limit)]
    if args.min_interval_s is not None:
        passthrough += ["--min-interval-s", str(args.min_interval_s)]
    if args.no_cache:
        passthrough += ["--no-cache"]

    log = run_set(papers, out_root, passthrough)

    log_path = out_root / "corpus_log.json"
    log_path.write_text(json.dumps(log, indent=2), encoding="utf-8")
    print(f"[corpus] {len(log['papers'])} paper(s): {log['counts']} -> {log_path}")

    if args.exit_map:
        exit_map = {REJECT_HANDLE_OF.get(k, k): rec["exit"]
                    for k, rec in log["papers"].items()}
        map_path = Path(args.exit_map)
        map_path.parent.mkdir(parents=True, exist_ok=True)
        map_path.write_text(json.dumps(exit_map, indent=2), encoding="utf-8")
        print(f"[corpus] exit map -> {map_path}")

    return 1 if log["crashed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
