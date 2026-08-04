"""Backfill the `bias_class_partition` block into committed AuditCore JSON fixtures.

ADR bias_class_taxonomy §5.2 added a `bias_class_partition` block to
`AuditCore.to_dict()`, immediately after `saturated_bases`. The committed
`results/auditor/run_<git>/*_core.json` fixtures pre-date that block. Re-running the
auditor to regenerate them is unsafe here: it needs the real dev panel and writes to
a NEW git-hash-named directory, breaking the hard-coded fixture paths in the reporter
tests.

Instead this splices the block in DETERMINISTICALLY from data already in each file —
the block is a pure recombination of `saturated_bases.harsanyi_dividends` over
`runnable_toggles`, computed by the SAME production functions the runtime uses, so
the result is byte-identical to what `AuditCore.to_dict()` would emit. Every
pre-existing key/value is left untouched; only the new block is inserted (after
`saturated_bases`, preserving key order). Idempotent: re-running overwrites the block
in place.

Usage:
    ./.venv/bin/python scripts/backfill_bias_class_partition.py            # default dir
    ./.venv/bin/python scripts/backfill_bias_class_partition.py <dir> ...  # explicit dirs
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Self-bootstrap the repo root onto sys.path so the documented invocation
# (`./.venv/bin/python scripts/backfill_bias_class_partition.py`) runs without an
# exported PYTHONPATH — matching the sibling auditor scripts (e.g. run_auditor.py).
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from agents.auditor.checks.algebra import bias_class_partition  # noqa: E402
from agents.auditor.schemas.decomposition import label_to_subset  # noqa: E402

_DEFAULT_DIR = Path("results/auditor/run_45313b2")


def _partition_dict(core: dict) -> dict:
    hd = core["saturated_bases"]["harsanyi_dividends"]
    harsanyi = {label_to_subset(k): float(v) for k, v in hd.items()}
    runnable = tuple(core["runnable_toggles"])
    return bias_class_partition(harsanyi, runnable).to_dict()


def _splice(core: dict, block: dict) -> dict:
    """Return a new dict matching the amended AuditCore.to_dict() ordering:
    `not_applicable_toggles` immediately after `runnable_toggles` (ADR §5.4; empty
    for these all-runnable fixtures) and `bias_class_partition` immediately after
    `saturated_bases`. Every other key/value is left untouched; stale copies of the
    two inserted keys are dropped and re-inserted in position (idempotent)."""
    out: dict = {}
    for key, value in core.items():
        if key in ("not_applicable_toggles", "bias_class_partition"):
            continue  # drop stale copies; re-inserted in the right position below
        out[key] = value
        if key == "runnable_toggles":
            out["not_applicable_toggles"] = list(core.get("not_applicable_toggles", []))
        if key == "saturated_bases":
            out["bias_class_partition"] = block
    if "bias_class_partition" not in out:  # not an AuditCore -> leave unchanged
        return core
    return out


def backfill_file(path: Path) -> bool:
    core = json.loads(path.read_text())
    if "saturated_bases" not in core or "runnable_toggles" not in core:
        return False
    block = _partition_dict(core)
    updated = _splice(core, block)
    # Match run_auditor.write_results exactly (indent=2, default=str, no trailing
    # newline) so the ONLY changes to the file are the inserted keys.
    path.write_text(json.dumps(updated, indent=2, default=str))
    return True


def main(argv: list[str]) -> int:
    dirs = [Path(a) for a in argv[1:]] or [_DEFAULT_DIR]
    total = 0
    for d in dirs:
        for path in sorted(d.glob("*_core.json")):
            if backfill_file(path):
                total += 1
                print(f"backfilled bias_class_partition -> {path}")
            else:
                print(f"skipped (not an AuditCore) -> {path}")
    print(f"done: {total} file(s) updated")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
