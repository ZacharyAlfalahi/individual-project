"""P1 paper arm — CLI: the field-key code-generation experiment with the paper in place of the spec.

Contract: the P1 paper-arm mini-contract (not shipped). Everything except the prompt is the field-key
arm's (``scripts/run_p1_codegen.py``):
  * the same phase_f pair and budget;
  * the same scoring thresholds and oracles (``basis_inputs.factors_dir(basis)``);
  * the byte-identical engine panel that arm exported for the basis, read-only and sha-checked
    against that arm's recorded ``panel_sha256``.

`--dry-run` (default, zero calls): prints the asset hashes, per-strategy prompt sha256 and size, the
freeze status, the panel check, the oracle-vs-oracle sanity, and the replay-first spend estimate.

`--execute --phase reported --basis B`: refuses unless the paper-arm freeze verifies, the sandbox root
is new (no stale outputs), seatbelt is available (the read-deny list needs it) and the pre-flight is
within caps. It then generates once per (strategy, model), cache-first at
``runs/p1_paper_codegen/cache`` (shared across bases, so a second basis replays for $0). Each script
runs sandboxed with reads of the holdout, the oracle factors, the field-key specs and earlier
generations denied, and is scored and archived. Results land in
``results/codegen_paper_arm/<basis>/p1_paper_results_<phase>.json``.

`--live-record PATH`: on a replay, SKU verification is inherited from the live record that served the
responses, provided its prompt hashes match byte for byte (``paper_arm.reportability``).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
for _p in (str(_REPO_ROOT), str(_REPO_ROOT / "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import pandas as pd  # noqa: E402

from evaluation.codegen.ablation import run_scored_ablation  # noqa: E402
from evaluation.codegen.jsonio import dump_json  # noqa: E402
from evaluation.codegen.live_client import BudgetExceededError, load_budget  # noqa: E402
from evaluation.codegen.paper_arm import (  # noqa: E402
    ARM,
    CONTRACT,
    PAPER_TARGETS,
    build_paper_prompt,
    paper_arm_asset_hashes,
    paper_arm_freeze_ok,
    reportability,
)
from evaluation.codegen.preflight import (  # noqa: E402
    metered_by_model,
    replay_first_factory,
    run_preflight,
)
from evaluation.codegen.runner import STRATEGIES, load_models, prompt_sha256  # noqa: E402
from evaluation.codegen.sandbox import detect_mechanism  # noqa: E402
from scripts import basis_inputs  # noqa: E402
from scripts.run_p1_codegen import DEFAULT_MAX_USD, _load_dotenv, _oracle_sanity  # noqa: E402

OUT_ROOT = _REPO_ROOT / "results" / "codegen_paper_arm"
CACHE_ROOT = _REPO_ROOT / "runs" / "p1_paper_codegen" / "cache"
FIELD_KEY_RECORD = "p1_results_reported.json"


class PaperArmRefusal(RuntimeError):
    """A precondition failed; nothing was generated."""


@dataclass(frozen=True)
class PaperArmPaths:
    out: Path
    panel: Path                 # the field-key arm's exported engine panel (read-only)
    field_key_record: Path      # that arm's results JSON, which pins the panel sha
    factors_dir: Path
    sandbox_root: Path          # the archive lands beside it (``sandbox_root.parent / "archive"``)


def resolve_paths(basis: str, out_root: Path = OUT_ROOT) -> PaperArmPaths:
    codegen = basis_inputs.basis_dir(basis, "codegen")
    out = Path(out_root) / basis
    return PaperArmPaths(out=out, panel=codegen / "engine_panel_corr.parquet",
                         field_key_record=codegen / FIELD_KEY_RECORD,
                         factors_dir=basis_inputs.factors_dir(basis),
                         sandbox_root=out / "sandbox")


def deny_read_paths(basis: str) -> tuple[Path, ...]:
    """Trees the generated code must not read: the holdout, both copies of the oracle factor series,
    the hand-authored field-key specs, and every earlier generation (caches and archives)."""
    codegen = basis_inputs.basis_dir(basis, "codegen")
    return (
        _REPO_ROOT / "data" / "holdout",
        _REPO_ROOT / "data" / "development" / "factors",
        basis_inputs.factors_dir(basis),
        _REPO_ROOT / "evaluation" / "gold_specs",
        _REPO_ROOT / "runs",
        codegen / "archive",
        codegen / "sandbox",
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_panel(paths: PaperArmPaths) -> str:
    """The panel must be development-only and byte-identical to the one the field-key arm ran on."""
    if "holdout" in paths.panel.parts:
        raise PaperArmRefusal(f"panel path touches the holdout partition: {paths.panel}")
    if not paths.panel.exists() or not paths.field_key_record.exists():
        raise PaperArmRefusal(f"field-key arm artefacts missing: {paths.panel} / "
                              f"{paths.field_key_record}")
    recorded = json.loads(paths.field_key_record.read_text(encoding="utf-8")).get("panel_sha256")
    actual = _sha256(paths.panel)
    if actual != recorded:
        raise PaperArmRefusal(f"panel sha256 {actual[:16]}… differs from the field-key arm's "
                              f"recorded {str(recorded)[:16]}… — the arms would not share a panel")
    try:
        basis_inputs.assert_dev_only(pd.read_parquet(paths.panel, columns=["date"])["date"],
                                     f"codegen panel {paths.panel.name}")
    except basis_inputs.BasisError as exc:
        raise PaperArmRefusal(str(exc)) from exc
    return actual


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true", default=True)
    mode.add_argument("--execute", dest="dry_run", action="store_false")
    ap.add_argument("--phase", choices=("dev", "reported"), default="dev",
                    help="dev = free non-reportable pair (default); reported = paid Phase-F pair")
    ap.add_argument("--basis", choices=basis_inputs.BASES, required=True)
    ap.add_argument("--out-root", default=str(OUT_ROOT),
                    help="results root (default results/codegen_paper_arm); use a "
                         "scratch root for a determinism replay")
    ap.add_argument("--live-record", default=None,
                    help="live-run record whose SKU check a replay may inherit")
    ap.add_argument("--max-usd-anthropic", type=float, default=DEFAULT_MAX_USD)
    ap.add_argument("--max-usd-gemini", type=float, default=DEFAULT_MAX_USD)
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    paths = resolve_paths(args.basis, Path(args.out_root))
    print("P1 paper arm — " + ("DRY RUN (zero generation calls)" if args.dry_run
                               else f"EXECUTE (phase={args.phase}, basis={args.basis})"))
    print("paper-arm assets:")
    for name, digest in paper_arm_asset_hashes().items():
        print(f"  {name}: {digest}")
    print("prompt sha256 per strategy (label, characters):")
    for s in STRATEGIES:
        prompt = build_paper_prompt(s)
        print(f"  {s}: {prompt_sha256(prompt)}  ({PAPER_TARGETS[s].label!r}, {len(prompt):,} chars)")
    freeze_ok, freeze_msg = paper_arm_freeze_ok()
    print(f"freeze: {freeze_msg}")

    try:
        panel_sha = verify_panel(paths)
    except PaperArmRefusal as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 2
    print(f"panel: {paths.panel} sha256={panel_sha[:16]}… (matches the field-key arm)")

    paths.out.mkdir(parents=True, exist_ok=True)
    print("oracle-vs-oracle sanity:")
    for line in _oracle_sanity(paths.out / "oracle_sanity", paths.factors_dir):
        print(line)

    models = load_models(args.phase)
    budget = load_budget()
    caps = {"anthropic": args.max_usd_anthropic, "gemini": args.max_usd_gemini}
    try:
        preflight = run_preflight(STRATEGIES, models, cache_root=CACHE_ROOT, budget=budget,
                                  caps_usd=caps, prompt_fn=build_paper_prompt)
    except BudgetExceededError as exc:
        print(f"REFUSED (pre-flight): {exc}", file=sys.stderr)
        return 2
    print(f"pre-flight: {preflight['cache']['hits']} hits, {preflight['cache']['misses']} misses; "
          f"estimated upper-bound spend {preflight['estimate']['per_vendor_usd']} (caps {caps})")

    if args.dry_run:
        print(f"{args.phase} pair: {[m['model_id'] for m in models]}")
        print("dry run complete — no generation performed.")
        return 0

    # --- execute path -------------------------------------------------------
    if not freeze_ok:
        print(f"REFUSED: {freeze_msg}", file=sys.stderr)
        return 2
    if preflight["refused"]:
        print("REFUSED (pre-flight, no call made): " + "; ".join(preflight["violations"]),
              file=sys.stderr)
        return 2
    if paths.sandbox_root.exists() or (paths.sandbox_root.parent / "archive").exists():
        print(f"REFUSED: {paths.sandbox_root} (or its archive) already exists — a stale output "
              "could be scored; use a new --out-root", file=sys.stderr)
        return 2
    if detect_mechanism() != "seatbelt":
        print("REFUSED: the read-deny list needs the seatbelt sandbox", file=sys.stderr)
        return 2
    live_record = None
    if args.live_record:
        live_record = json.loads(Path(args.live_record).read_text(encoding="utf-8"))

    live_models = preflight["live_models"]
    if live_models:
        _load_dotenv(_REPO_ROOT / ".env")
        missing = [m["api_key_env"] for m in models
                   if m["model_id"] in live_models and not os.environ.get(m.get("api_key_env", ""))]
        if missing:
            print(f"REFUSED: missing {args.phase} credentials {missing}", file=sys.stderr)
            return 2
    else:
        print("pure cache replay — no credentials read, no live client constructed.")

    deny = deny_read_paths(args.basis)
    result = run_scored_ablation(
        STRATEGIES, phase=args.phase, client_factory=replay_first_factory(live_models, budget),
        budget=budget, panel_path=paths.panel, cache_root=CACHE_ROOT,
        sandbox_root=paths.sandbox_root, factors_dir=paths.factors_dir, ensure_panel=False,
        prompt_fn=build_paper_prompt, deny_read_paths=deny)

    # The ablation's own reportable/sku_match stamp treats "no SKU seen" as a match; the paper arm
    # replaces it with the three-state, named-blocker verdict.
    result.pop("reportable", None)
    result.pop("sku_match", None)
    result.update(reportability(result, live_record=live_record))
    result.update({
        "arm": ARM,
        "computed_after_the_fact": True,
        "contract": str(CONTRACT.relative_to(_REPO_ROOT)),
        "contract_sha256": _sha256(CONTRACT),
        "asset_sha256": paper_arm_asset_hashes(),
        "targets": {s: {"paper_id": t.paper_id, "label": t.label} for s, t in PAPER_TARGETS.items()},
        "panel_sha256": panel_sha,
        "inputs": {
            "basis": args.basis,
            "panel": str(paths.panel.relative_to(_REPO_ROOT)),
            "field_key_record": str(paths.field_key_record.relative_to(_REPO_ROOT)),
            "field_key_record_sha256": _sha256(paths.field_key_record),
            "factors_dir": str(paths.factors_dir.relative_to(_REPO_ROOT)),
            "sandbox_root": str(paths.sandbox_root),
            "deny_read_paths": [str(p) for p in deny],
            "sandbox_mechanism": detect_mechanism(),
            "live_record": args.live_record,
        },
        "llm_policy": {"preflight": preflight, "metered_by_model": metered_by_model(budget)},
    })

    print(f"verdicts: {result['verdict_counts']}")
    print(f"reportable={result['reportable']} (sku_match={result['sku_match']}); "
          f"blockers={result['reportable_blockers']}")
    print(f"budget: spent ${result['budget']['spent_usd_estimate']:.4f} of "
          f"${result['budget']['usd_cap']:.0f} over {result['budget']['calls']} calls")
    out_path = paths.out / f"p1_paper_results_{args.phase}.json"
    out_path.write_text(dump_json(result), encoding="utf-8")
    print(f"results written: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
