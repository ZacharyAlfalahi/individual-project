"""
Corpus confirmatory FDR (§7.2, D-A9) — BH over the UNION family across the locked anchors.

The full-audit run writes a WITHIN-STRATEGY BH block per report (`fdr.scope ==
"within_strategy"`, a diagnostic — see `agents/auditor/checks/report.py`). The pre-registered
§7.2 confirmatory family is the union across the locked anchors, keyed by (strategy,
coordinate). This driver assembles it from a full-audit run's
reports; it computes NO new inference — p-values are read from each report's `inference` block.

Membership (never hard-coded):
  * strategies = registry rows with `status == "locked"` (config/hypothesis_registry.yaml).
    The pilot (str, D-A33) is excluded, and so is the negative control: its row is locked, so
    `FactorHypothesis.is_confirmatory` is True, but it is a specificity control with no lattice
    report and is not a confirmatory anchor (D-A59: confirmatory = mom6 + drf).
  * coordinates = `confirmatory_coordinates(runnable_toggles)` per report — the runnable
    first-order effects plus the three registered pairs (D-A28).

Fail-loud cross-checks: each report's inference labels equal its confirmatory coordinates; each
p-value equals the report's own within-strategy FDR block; the report q equals the current
`auditor.fdr.q`; all reports share one pre-registration tag.

Dev-only (reads local audit report JSONs; the holdout is never touched).
Usage: python scripts/run_corpus_confirmatory_fdr.py [--run-dir results/auditor/recorded]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from agents.auditor.checks.fdr import corpus_confirmatory_fdr  # noqa: E402
from agents.auditor.confirmatory import confirmatory_coordinates  # noqa: E402
from agents.auditor.schemas.decomposition import subset_label  # noqa: E402
from agents.auditor.thresholds import load_fdr_q  # noqa: E402
from agents.auditor.validation.hypothesis_registry import (  # noqa: E402
    _REGISTRY_PATH,
    FactorHypothesis,
    load_hypothesis_registry,
)

DEFAULT_RUN_DIR = REPO_ROOT / "results" / "auditor" / "recorded"
THRESHOLDS_FILE = REPO_ROOT / "docs" / "thresholds.yaml"


class CorpusFdrError(RuntimeError):
    """A source report is missing or inconsistent — refuse rather than emit a family built on
    mismatched inputs."""


def confirmatory_members(registry: dict[str, FactorHypothesis]) -> tuple[list[str], dict[str, str]]:
    """(members, excluded→reason). Members are the locked anchors, in registry order."""
    members, excluded = [], {}
    for fid, hyp in registry.items():
        if hyp.status == "locked" and hyp.is_locked:
            members.append(fid)
        elif hyp.status == "pilot":
            excluded[fid] = "pilot (D-A33)"
        elif hyp.status == "negative_control":
            excluded[fid] = "negative control — specificity check, not a confirmatory anchor (D-A59)"
        else:
            excluded[fid] = f"not locked (is_locked={hyp.is_locked}, status={hyp.status!r})"
    if not members:
        raise CorpusFdrError("registry has no locked anchors — no confirmatory family to form")
    return members, excluded


def strategy_pvalues(report: dict, strategy: str) -> dict[str, float]:
    """{coordinate label: p} from the report's inference block, cross-checked against the
    registered coordinate set and the report's own within-strategy FDR block."""
    expected = [subset_label(c) for c in confirmatory_coordinates(report["runnable_toggles"])]
    inference = report.get("inference") or {}
    if set(inference) != set(expected):
        raise CorpusFdrError(
            f"{strategy}: inference coordinates {sorted(inference)} != confirmatory "
            f"coordinates {sorted(expected)}")
    fdr = report.get("fdr") or {}
    if fdr.get("scope") != "within_strategy":
        raise CorpusFdrError(f"{strategy}: expected a within_strategy fdr block, got {fdr.get('scope')!r}")
    out: dict[str, float] = {}
    for label in expected:
        p = float(inference[label]["p_value"])
        p_fdr = float(fdr["decisions"][label]["p_value"])
        if p != p_fdr:
            raise CorpusFdrError(f"{strategy}::{label}: inference p {p!r} != fdr-block p {p_fdr!r}")
        out[label] = p
    return out


def build_corpus_confirmatory(reports: dict[str, dict], registry: dict[str, FactorHypothesis],
                              q: float) -> dict:
    """Pure assembly: membership, the pooled BH report, and a per-member comparison with the
    within-strategy adjustment. `reports` maps registry factor id → report dict."""
    members, excluded = confirmatory_members(registry)
    missing = [m for m in members if m not in reports]
    if missing:
        raise CorpusFdrError(f"no report for locked anchor(s) {missing}")

    tags = {reports[m].get("pre_registration_tag") for m in members}
    if len(tags) != 1:
        raise CorpusFdrError(f"reports span pre-registration tags {sorted(map(str, tags))}")
    for m in members:
        report_q = float(reports[m]["fdr"]["q"])
        if report_q != q:
            raise CorpusFdrError(f"{m}: report q {report_q} != auditor.fdr.q {q}")

    per_strategy = {m: strategy_pvalues(reports[m], m) for m in members}
    pooled = corpus_confirmatory_fdr(per_strategy, q).to_dict()

    comparison = []
    for m in members:
        for label, p in per_strategy[m].items():
            d = pooled["decisions"][f"{m}::{label}"]
            w = reports[m]["fdr"]["decisions"][label]
            comparison.append({
                "strategy": m, "coordinate": label, "p_value": p,
                "adjusted_p_within_strategy": w["adjusted_p"],
                "adjusted_p_pooled": d["adjusted_p"],
                "rejected_within_strategy": w["rejected"],
                "rejected_pooled": d["rejected"],
                "pooled_rank": d["rank"],
            })
    comparison.sort(key=lambda row: row["pooled_rank"])

    return {
        "membership": {
            "strategies": members,
            "coordinates_per_strategy": {m: list(per_strategy[m]) for m in members},
            "excluded": excluded,
        },
        "pre_registration_tag": tags.pop(),
        "fdr": pooled,
        "min_adjusted_p_pooled": {
            m: min(r["adjusted_p_pooled"] for r in comparison if r["strategy"] == m) for m in members},
        "comparison": comparison,
    }


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def main(argv: list[str] | None = None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR,
                    help="one run directory holding every locked anchor's <id>_report.json")
    ap.add_argument("--report", action="append", default=[], metavar="ID=PATH",
                    help="per-anchor report path (overrides --run-dir for that anchor); repeatable, "
                         "for runs written to one directory per strategy")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args(argv)

    run_dir = args.run_dir
    registry = load_hypothesis_registry()
    members, _ = confirmatory_members(registry)
    overrides: dict[str, Path] = {}
    for item in args.report:
        fid, sep, path = item.partition("=")
        if not sep or fid not in members:
            raise CorpusFdrError(f"--report expects ID=PATH with ID in {members}; got {item!r}")
        overrides[fid] = Path(path)
    report_paths = {m: overrides.get(m, run_dir / f"{m}_report.json") for m in members}
    for m, p in report_paths.items():
        if not p.is_file():
            raise CorpusFdrError(f"report for {m} missing: {p}")
    reports = {m: json.loads(p.read_text(encoding="utf-8")) for m, p in report_paths.items()}
    run_logs = {}
    for m, p in report_paths.items():
        log_path = p.parent / "run_log.json"
        run_logs[m] = json.loads(log_path.read_text(encoding="utf-8")) if log_path.is_file() else {}

    result = build_corpus_confirmatory(reports, registry, load_fdr_q())
    artifact = {
        "purpose": ("§7.2 corpus confirmatory BH-FDR over the union of the locked anchors' "
                    "confirmatory coordinates, assembled from full-audit reports. "
                    "Re-adjustment only; no new inference."),
        "run_timestamp": datetime.now(timezone.utc).isoformat(),
        "source_run": {
            m: {"path": _rel(p), "sha256": _sha256(p),
                "basis": run_logs[m].get("basis", "clean")}
            for m, p in report_paths.items()
        },
        "thresholds_sha256": _sha256(THRESHOLDS_FILE),
        "hypothesis_registry": {"path": _rel(_REGISTRY_PATH), "sha256": _sha256(_REGISTRY_PATH)},
        **result,
    }
    bases = {v["basis"] for v in artifact["source_run"].values()}
    if len(bases) != 1:
        raise CorpusFdrError(f"reports mix return bases {sorted(bases)} — a family must share one basis")

    if args.out is None and overrides:
        raise CorpusFdrError("--out is required with --report (a per-anchor family must not overwrite "
                             "the default artefact)")
    out = args.out or (REPO_ROOT / "results" / "auditor" / "corpus_confirmatory_fdr.json")
    tmp = out.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(artifact, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    tmp.replace(out)

    fdr = artifact["fdr"]
    print(f"corpus confirmatory FDR: family={fdr['n_family']} q={fdr['q']} rejected={fdr['n_rejected']}")
    for m, v in artifact["min_adjusted_p_pooled"].items():
        print(f"  {m}: min pooled adjusted p = {v:.4f}")
    print(f"  written: {_rel(out)}")
    return artifact


if __name__ == "__main__":
    main()
