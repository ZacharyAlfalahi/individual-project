"""
T3 corpus coverage scoring (B5b) -- contract §5.2/§5.3 over the pre-registered
32-construction denominator.

Joins the CI-5 expected-disposition labels (evaluation/gold_specs/
t3_coverage_labels.yaml: 27 implement + 5 refuse + 1 excluded) against realised
per-construction outcomes, and reports:

  * layered coverage (C_semantic/C_binding/C_execution) over the OBSERVED
    denominator rows (`layered_observed`), reusing the identical §5.2 classifier
    the anchor run uses (agents/quant/config/coverage.layered_coverage) -- never
    a second implementation. NOTE: `layered_observed["C_end_to_end"]` runs over
    the observed subset only; the HEADLINE end-to-end rate is
    `C_end_to_end_full`, charged over the frozen 32 (§8 elastic-coverage
    discipline) -- never headline the observed-subset number;
  * FRR  (§5.3): refused ∩ should-implement, over the 27 (an incorrect refusal);
  * refusal recall: refused ∩ should-refuse, over the 5 (correct caution);
  * FIR events (§5.3 headline safety): executed ∩ should-refuse -- each one is
    listed, never only counted;
  * unobserved rows (no outcome produced -- e.g. the paper degraded to
    paper_failed): kept IN the end-to-end denominator (never silently dropped;
    a shrunk denominator is the elastic-coverage failure §8 forbids) but
    reported in their own bucket, not mis-attributed to a refusal layer.

The excluded row (the 153-signal family) sits outside both denominators by
registration and is only ever echoed back, never scored.

Observed rows are dicts: {"paper_id", "name", "outcome": "executed"|"refused"|
"not_run", "refusal_codes": [...]} (codes required for refused rows; the layer
attribution is theirs). Fail-loud on a phantom (paper_id, name) -- an outcome
for a construction the registered denominator does not contain is a build
error, not data.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

from agents.quant.config.coverage import layered_coverage

_REPO_ROOT = Path(__file__).resolve().parents[2]
_LABELS_PATH = _REPO_ROOT / "evaluation" / "gold_specs" / "t3_coverage_labels.yaml"

_LEGAL_LABELS = frozenset({"implement", "refuse", "excluded"})
_LEGAL_OUTCOMES = frozenset({"executed", "refused", "not_run"})


class CoverageLabelError(ValueError):
    """A labels-file or observed-row integrity failure (fail-loud, never skipped)."""


@dataclass(frozen=True)
class CoverageLabels:
    """The registered expected-disposition key, {(paper_id, name): label}."""

    label_of: dict[tuple[str, str], str]
    version: str

    @property
    def denominator(self) -> int:
        return sum(1 for v in self.label_of.values() if v != "excluded")

    def of(self, kind: str) -> frozenset[tuple[str, str]]:
        return frozenset(k for k, v in self.label_of.items() if v == kind)


def load_coverage_labels(path: str | Path | None = None) -> CoverageLabels:
    """Load + validate the labels file against its own registration (27/5/1)."""
    p = Path(path) if path is not None else _LABELS_PATH
    data = yaml.safe_load(p.read_text(encoding="utf-8"))
    label_of: dict[tuple[str, str], str] = {}
    for paper_id, block in (data.get("papers") or {}).items():
        for name, label in (block.get("constructions") or {}).items():
            if label not in _LEGAL_LABELS:
                raise CoverageLabelError(f"{p}: illegal label {label!r} for {paper_id}/{name}")
            key = (paper_id, name)
            if key in label_of:
                raise CoverageLabelError(f"{p}: duplicate construction {key}")
            label_of[key] = label

    counts = {k: sum(1 for v in label_of.values() if v == k) for k in _LEGAL_LABELS}
    if (counts["implement"], counts["refuse"], counts["excluded"]) != (27, 5, 1):
        raise CoverageLabelError(
            f"{p}: label counts {counts} do not match the CI-5 registration "
            "(27 implement / 5 refuse / 1 excluded) -- a changed partition is a "
            "re-registration, never a load-time surprise"
        )
    return CoverageLabels(label_of=label_of, version=str(data.get("version", "")))


def score_corpus_coverage(observed: list[dict],
                          labels: CoverageLabels | None = None) -> dict:
    """Score realised outcomes against the registered denominator; see module doc."""
    lab = labels if labels is not None else load_coverage_labels()

    by_key: dict[tuple[str, str], dict] = {}
    for row in observed:
        key = (row["paper_id"], row["name"])
        if key not in lab.label_of:
            raise CoverageLabelError(
                f"observed outcome for unregistered construction {key} -- the "
                f"denominator is frozen; a phantom row is a build error"
            )
        if key in by_key:
            raise CoverageLabelError(f"duplicate observed row for {key}")
        outcome = row.get("outcome")
        if outcome not in _LEGAL_OUTCOMES:
            raise CoverageLabelError(f"{key}: illegal outcome {outcome!r}")
        if outcome == "refused" and not row.get("refusal_codes"):
            raise CoverageLabelError(
                f"{key}: a refusal must carry its typed refusal_codes (the layer "
                f"attribution is theirs; an untyped refusal is unattributable)"
            )
        by_key[key] = row

    scored = {k: v for k, v in by_key.items() if lab.label_of[k] != "excluded"}
    excluded_seen = sorted(k for k in by_key if lab.label_of[k] == "excluded")

    implement, refuse = lab.of("implement"), lab.of("refuse")
    unobserved = sorted((implement | refuse) - set(scored))

    # §5.2 layered coverage over the OBSERVED denominator rows (the classifier is
    # the anchor run's own); unobserved rows are charged to the end-to-end rate
    # explicitly below, never laundered into a refusal layer.
    layered = layered_coverage([
        {"strategy_id": f"{k[0]}/{k[1]}",
         "refusal_codes": list(v.get("refusal_codes", [])) if v["outcome"] == "refused" else []}
        for k, v in sorted(scored.items()) if v["outcome"] != "not_run"
    ])

    executed = {k for k, v in scored.items() if v["outcome"] == "executed"}
    refused = {k for k, v in scored.items() if v["outcome"] == "refused"}
    not_run = {k for k, v in scored.items() if v["outcome"] == "not_run"}

    false_refusals = sorted(refused & implement)
    refusal_recall_hits = sorted(refused & refuse)
    fir_events = sorted(executed & refuse)

    def frac(num: int, den: int) -> float | None:
        return (num / den) if den else None

    n_denominator = lab.denominator                    # 32, frozen
    n_unobserved = len(unobserved) + len(not_run)
    return {
        "denominator": n_denominator,
        "n_implement": len(implement),
        "n_refuse": len(refuse),
        "n_executed": len(executed),
        "n_refused": len(refused),
        "n_unobserved": n_unobserved,
        "unobserved": [list(k) for k in unobserved] + [list(k) for k in sorted(not_run)],
        "layered_observed": layered,
        # End-to-end charged over the FULL frozen denominator: an unobserved
        # construction is a coverage loss, not a denominator shrink.
        "C_end_to_end_full": frac(len(executed), n_denominator),
        # §5.3 safety rates, each over its registered frame.
        "FRR": frac(len(false_refusals), len(implement)),
        "false_refusals": [list(k) for k in false_refusals],
        "refusal_recall": frac(len(refusal_recall_hits), len(refuse)),
        "FIR_events": [list(k) for k in fir_events],
        "n_FIR_events": len(fir_events),
        "excluded_observed": [list(k) for k in excluded_seen],
    }
