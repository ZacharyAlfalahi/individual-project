"""P2 coverage-boundary codegen — CLI (mirrors scripts/run_p1_codegen.py).

`--dry-run` (the ONLY runnable mode until the scale census exists AND
`corpus.selection.status == 'frozen'` AND the frozen zoo-list hash matches):
reports the Phase-F pair, the shared prompt assets, and the P2 execute-gate
state. Zero generation calls, zero spend, no holdout contact.

`--execute`: REFUSES unless (1) the scale census artefact exists, (2)
`corpus.selection.status == 'frozen'`, and (3) the frozen zoo-list sha256 in
`p2_codegen.zoo_list.frozen_sha256` matches the live zoo-list — the three P2
gates. Two of the three fail today (no census; sha is TO_SET),
so `--execute` fails closed.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
for _p in (str(_REPO_ROOT), str(_REPO_ROOT / "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from evaluation.codegen.p2_driver import corpus_selection_status  # noqa: E402
from evaluation.codegen.p2_metrics import load_p2_scoring_thresholds  # noqa: E402
from evaluation.codegen.runner import load_phase_f_models, prompt_asset_hashes  # noqa: E402
from evaluation.codegen.sandbox import detect_mechanism  # noqa: E402

# Build-only artefact paths (the scale census + frozen zoo-list do not exist yet).
_CENSUS_PATH = _REPO_ROOT / "data" / "development" / "codegen" / "p2_census.json"
_ZOO_LIST_PATH = _REPO_ROOT / "data" / "development" / "codegen" / "p2_zoo_list.txt"

_FROZEN_STATUS = "frozen"


def load_zoo_list(path: Path) -> tuple[str, ...]:
    """One paper name per line; blanks and ``#`` comments skipped."""
    names: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if s and not s.startswith("#"):
            names.append(s)
    return tuple(names)


def zoo_list_sha256(names: tuple[str, ...]) -> str:
    return hashlib.sha256(("\n".join(names)).encode("utf-8")).hexdigest()


def census_exists() -> bool:
    return _CENSUS_PATH.is_file()


def zoo_list_freeze_ok(thresholds: dict) -> tuple[bool, str]:
    """The frozen zoo-list sha256 must be set (not TO_SET) and match the
    live zoo-list byte-for-byte."""
    frozen = str(thresholds.get("zoo_list", {}).get("frozen_sha256", "TO_SET"))
    if frozen == "TO_SET":
        return False, "p2_codegen.zoo_list.frozen_sha256 is TO_SET (zoo-list not frozen)"
    if not _ZOO_LIST_PATH.is_file():
        return False, f"zoo-list file missing: {_ZOO_LIST_PATH}"
    live = zoo_list_sha256(load_zoo_list(_ZOO_LIST_PATH))
    if live != frozen:
        return False, (f"zoo-list sha256 {live[:16]}… != frozen {frozen[:16]}… "
                       "— re-freeze before any generation call")
    return True, "zoo-list freeze verified"


def execute_gate() -> tuple[bool, list[str]]:
    """The three P2 gates. Returns (ok, per-gate status lines)."""
    thresholds = load_p2_scoring_thresholds()
    status = corpus_selection_status()
    lines: list[str] = []

    g1 = census_exists()
    lines.append(f"  [{'PASS' if g1 else 'REFUSE'}] scale census exists ({_CENSUS_PATH})")

    g2 = status == _FROZEN_STATUS
    lines.append(f"  [{'PASS' if g2 else 'REFUSE'}] corpus.selection.status == 'frozen' "
                 f"(is '{status}')")

    g3, g3_msg = zoo_list_freeze_ok(thresholds)
    lines.append(f"  [{'PASS' if g3 else 'REFUSE'}] zoo-list freeze — {g3_msg}")

    return (g1 and g2 and g3), lines


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true", default=True)
    mode.add_argument("--execute", dest="dry_run", action="store_false")
    args = ap.parse_args(argv)

    print("P2 coverage-boundary codegen — " + ("DRY RUN (zero generation calls)"
                                               if args.dry_run else "EXECUTE"))
    print(f"sandbox mechanism: {detect_mechanism()}")
    print("shared prompt assets (P1/P2):")
    for name, digest in prompt_asset_hashes().items():
        print(f"  {name}: {digest}")
    print(f"phase_f pair (from thresholds): {[m['model_id'] for m in load_phase_f_models()]}")

    ok, gate_lines = execute_gate()
    print("P2 execute gate:")
    for line in gate_lines:
        print(line)

    if args.dry_run:
        print("dry run complete — no generation performed.")
        return 0

    if not ok:
        print("REFUSED: the P2 execute gate is not satisfied (see the gate lines above); "
              "P2 emits numbers only once the census exists, the selection is frozen, and "
              "the zoo-list hash matches.", file=sys.stderr)
        return 2
    missing = [m["api_key_env"] for m in load_phase_f_models()
               if not os.environ.get(m.get("api_key_env", ""))]
    if missing:
        print(f"REFUSED: missing Phase-F credentials {missing}", file=sys.stderr)
        return 2
    print("REFUSED: live execution additionally requires a wired client_factory under the "
          "approved budget sub-cap; not built in this cycle.", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
