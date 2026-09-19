"""Reference-arm comparison over the ORACLE set (WS-C).

When the registered Arm B is empty (the deterministic router compiles no scale-layer
member, so there is no size-matched supported control to draw), the coverage-boundary
comparators have nothing to compare. This module computes the SAME two comparators over the
one population where all three implementations DO exist — the oracle set P1
generates against:

  * **inter-model agreement** — the two models' generated series for the same strategy;
  * **codegen-vs-compiler divergence** — each model's series against the hand-built oracle,
    i.e. the deterministic implementation, labelled "neither side is truth" exactly as on the
    P2 path.

It is NOT the registered Arm B and never stands in for it: the population is the anchor /
oracle set built from GOLD specifications, not the scale-layer corpus routed from extracted
ones. It is a descriptive reference rate — what these comparators look like where the
specification is known-good — reported beside the boundary arm.

Reads only artefacts a P1 run produces (no model calls, no generation): the run record, the
sandboxed output CSVs, and the oracle series. Every series read back is integrity-checked
against the hash P1 archived for that run, so a sandbox re-used since cannot be scored as
this run's output.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from pathlib import Path

import pandas as pd

from evaluation.codegen.p2_metrics import (
    agreement_distribution,
    codegen_vs_compiler_divergence,
    member_agreement,
)
from evaluation.codegen.sandbox import parse_output_csv

#: The arm label carried on every row (never "A"/"B" — this is not a registered P2 arm).
ARM = "reference"

#: Where the oracle set's comparison sits relative to truth: the oracle is the deterministic
#: implementation, so unlike P2's boundary arms a correctness verdict DOES exist here (P1
#: reports it). This module deliberately reports only the two P2 comparators, so the two
#: populations are compared like with like.
ORACLE_NOTE = (
    "The oracle is the hand-built deterministic implementation, so on this population a "
    "correctness verdict exists and P1 reports it. Only the two coverage-boundary "
    "comparators are computed here, so the reference rate is like-for-like with the "
    "boundary arm."
)

#: On the boundary arms neither side is truth, because no oracle exists there. On THIS
#: population one does, so the boundary label would be false here and is replaced.
ORACLE_DIVERGENCE_LABEL = (
    "the oracle is the deterministic reference on this population; the coverage-boundary "
    "arms have no such reference"
)


class ReferenceArmError(ValueError):
    """A P1 artefact is missing or no longer matches the run record it belongs to."""


@dataclass(frozen=True)
class SeriesSource:
    """Where one P1 run's generated series was read from, and the hash that ties it to the
    archived run."""

    strategy: str
    model_id: str
    path: str | None = None
    sha256: str | None = None
    status: str = "missing"          # "read" | "no_output" | "missing"

    def to_dict(self) -> dict:
        return {"strategy": self.strategy, "model_id": self.model_id, "path": self.path,
                "sha256": self.sha256, "status": self.status}


@dataclass(frozen=True)
class ReferenceArmResult:
    agreements: tuple = ()
    distribution: dict = field(default_factory=dict)
    divergences: tuple = ()
    sources: tuple[SeriesSource, ...] = ()
    unpaired: tuple[str, ...] = ()      # strategies where only one model produced a series

    def to_dict(self) -> dict:
        return {
            "arm": ARM,
            "oracle_note": ORACLE_NOTE,
            "agreements": [a.to_dict() for a in self.agreements],
            "distribution": self.distribution,
            "divergences": [d.to_dict() for d in self.divergences],
            "series_sources": [s.to_dict() for s in self.sources],
            "unpaired_strategies": list(self.unpaired),
        }


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def archived_output_sha(
    archive_root: Path, strategy: str, model_id: str, *, phase: str | None = None
) -> str | None:
    """The ``output_sha256`` P1 archived for this (strategy, model), or ``None`` when the run
    archived no output (a WONT_RUN).

    The sandbox is phase-scoped but the archive is not, so entries are filtered by the
    archived ``phase`` when one is given — otherwise a dev run and a reported run of the same
    cell look like one ambiguous cell. Fail-loud if two entries still disagree: the caller
    cannot know which of them produced the CSV on disk."""
    base = archive_root / strategy / model_id
    if not base.is_dir():
        return None
    shas = set()
    for meta_path in sorted(base.glob("*/meta.json")):
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if phase is not None and meta.get("phase") not in (None, phase):
            continue
        if meta.get("output_sha256"):
            shas.add(meta["output_sha256"])
    if len(shas) > 1:
        raise ReferenceArmError(
            f"{strategy}/{model_id}: the archive holds {len(shas)} distinct output hashes for "
            f"phase {phase!r} — the cell was generated more than once, so the CSV on disk "
            "cannot be tied to a run"
        )
    return shas.pop() if shas else None


def load_generated_series(
    runs: list[dict],
    sandbox_root: Path,
    archive_root: Path,
    *,
    phase: str = "reported",
    parse: Callable[[Path], pd.Series] = parse_output_csv,
) -> tuple[dict[str, dict[str, pd.Series | None]], tuple[SeriesSource, ...]]:
    """Read each P1 run's sandboxed output back as a series.

    A run P1 recorded as anything but ``ok`` contributes ``None``. A run recorded ``ok`` MUST
    have its CSV on disk and that CSV MUST hash to the value P1 archived — otherwise the
    sandbox has been re-used since and the file is not this run's output (fail loud rather
    than score a stale series)."""
    series: dict[str, dict[str, pd.Series | None]] = {}
    sources: list[SeriesSource] = []
    seen: set[tuple[str, str]] = set()
    for record in runs:
        strategy, model_id = record["strategy"], record["model_id"]
        if (strategy, model_id) in seen:
            raise ReferenceArmError(
                f"{strategy}/{model_id}: two run records for one cell — the later would "
                "silently replace the earlier; fix the run record rather than pick one"
            )
        seen.add((strategy, model_id))
        path = sandbox_root / phase / strategy / model_id / "out" / "portfolio_returns.csv"
        if record.get("sandbox_status") != "ok":
            series.setdefault(strategy, {})[model_id] = None
            sources.append(SeriesSource(strategy, model_id, status="no_output"))
            continue
        if not path.is_file():
            raise ReferenceArmError(
                f"{strategy}/{model_id}: P1 recorded sandbox_status 'ok' but {path} is missing "
                "— the run record and the sandbox disagree"
            )
        digest = _sha256(path)
        archived = archived_output_sha(archive_root, strategy, model_id, phase=phase)
        if archived is None:
            # Without the archived hash nothing ties this CSV to the run that produced it, and
            # a sandbox path is re-used by construction. Refuse rather than score an
            # unverifiable series.
            raise ReferenceArmError(
                f"{strategy}/{model_id}: P1 archived no output hash for this cell (phase "
                f"{phase!r}), so {path} cannot be tied to the run that wrote it; refusing to "
                "score an unverifiable series"
            )
        if archived != digest:
            raise ReferenceArmError(
                f"{strategy}/{model_id}: {path} hashes to {digest[:12]}… but P1 archived "
                f"{archived[:12]}… — the sandbox was re-used after the run; refusing to score "
                "a stale series"
            )
        series.setdefault(strategy, {})[model_id] = parse(path)
        sources.append(SeriesSource(strategy, model_id, str(path), digest, "read"))
    return series, tuple(sources)


def render_reference_report(result: ReferenceArmResult, meta: dict | None = None) -> str:
    """Markdown for the reference arm. Carries the §6 agreement caveat verbatim, states in its
    own header that it is NOT the registered Arm B, and emits no performance number."""
    from evaluation.codegen.p2_report import AGREEMENT_CAVEAT, _blockquote, _fmt

    out = ["# Codegen reference arm — the oracle set", ""]
    out.append(
        "_Where the registered coverage-boundary Arm B is empty (the deterministic router compiles "
        "no member of the scale-layer corpus, so no size-matched supported control exists "
        "there), this is the same pair of comparators computed over the one population where "
        "all three implementations do exist — the oracle set, built from GOLD specifications. "
        "It is a descriptive reference rate, **not** the registered Arm B, and it never "
        "stands in for it._"
    )
    out.append("")
    out.append(f"_{ORACLE_NOTE}_")
    out.append("")
    out.append("## Agreement caveat (contract §6, verbatim)")
    out.append("")
    out.append(_blockquote(AGREEMENT_CAVEAT))
    out.append("")

    dist = result.distribution
    out.append("## Inter-model agreement (the two models against each other)")
    out.append("")
    out.append(f"- strategies with BOTH models' series: {dist.get('n_members', 0)} "
               f"(measurable correlations: {dist.get('n_finite_correlations', 0)}; "
               f"insufficient overlap: {dist.get('n_insufficient_overlap', 0)})")
    unpaired = dist.get("unpaired_strategies") or []
    out.append(f"- strategies with only ONE model's series, so no pair to score: "
               f"{len(unpaired)}{(' — ' + ', '.join(unpaired)) if unpaired else ''}")
    out.append(f"- correlation min / median / max: {_fmt(dist.get('correlation_min'))} / "
               f"{_fmt(dist.get('correlation_median'))} / {_fmt(dist.get('correlation_max'))}")
    out.append(f"- agreement rate: {_fmt(dist.get('agreement_rate'))} "
               f"({dist.get('n_agree', 0)}/{dist.get('n_scored', 0)} scored pairs)")
    sc = dist.get("strata_counts", {})
    out.append(f"- divergence strata — high: {sc.get('high', 0)}, "
               f"medium: {sc.get('medium', 0)}, low: {sc.get('low', 0)}")
    out.append("")
    out.append("| strategy | n_overlap | correlation | sign agreement | stratum | agrees |")
    out.append("|---|---|---|---|---|---|")
    for a in result.agreements:
        out.append(f"| {a.paper_id} | {a.n_overlap} | {_fmt(a.correlation)} | "
                   f"{_fmt(a.sign_agreement)} | {a.stratum} | {a.agrees} |")
    out.append("")

    out.append("## Codegen-vs-compiler divergence")
    out.append("")
    out.append(f"_{ORACLE_DIVERGENCE_LABEL}._")
    out.append("")
    out.append("| strategy | model | correlation | tracking error | max abs diff | n_overlap |")
    out.append("|---|---|---|---|---|---|")
    for d in result.divergences:
        m = d.metrics
        out.append(f"| {d.paper_id} | {d.model_id} | {_fmt(m.get('correlation'))} | "
                   f"{_fmt(m.get('tracking_error'))} | {_fmt(m.get('max_abs_diff'))} | "
                   f"{m.get('n_overlap', 0)} |")
    out.append("")

    out.append("## Series provenance (each read-back tied to the hash P1 archived)")
    out.append("")
    out.append("| strategy | model | status | sha256 |")
    out.append("|---|---|---|---|")
    for s in result.sources:
        out.append(f"| {s.strategy} | {s.model_id} | {s.status} | "
                   f"{(s.sha256 or '—')[:16]} |")
    out.append("")
    if meta:
        out.append("## Provenance")
        out.append("")
        for key in sorted(meta):
            out.append(f"- {key}: {meta[key]}")
    return "\n".join(out)


def build_reference_arm(
    series_by_strategy: dict[str, dict[str, pd.Series | None]],
    oracle_for: Callable[[str], pd.Series],
    thresholds: dict,
    *,
    model_ids: tuple[str, str],
    sources: tuple[SeriesSource, ...] = (),
) -> ReferenceArmResult:
    """The two comparators over the oracle set.

    Inter-model agreement is computed for every strategy where BOTH models produced a series
    (a strategy where one model failed has no pair to score — counted in the distribution's
    own bookkeeping, never silently dropped). The codegen-vs-compiler divergence is computed
    for every (strategy, model) that produced one."""
    a_id, b_id = model_ids
    agreements = []
    divergences = []
    unpaired: list[str] = []
    for strategy in sorted(series_by_strategy):
        pair = series_by_strategy[strategy]
        a, b = pair.get(a_id), pair.get(b_id)
        if a is not None and b is not None:
            agreements.append(member_agreement(strategy, ARM, a, b, thresholds))
        else:
            unpaired.append(strategy)
        if all(pair.get(m) is None for m in (a_id, b_id)):
            continue                       # nothing generated: no oracle load, no rows
        oracle = oracle_for(strategy)
        for model_id in (a_id, b_id):
            generated = pair.get(model_id)
            if generated is not None:
                divergence = codegen_vs_compiler_divergence(
                    strategy, model_id, generated, oracle)
                divergences.append(replace(divergence, label=ORACLE_DIVERGENCE_LABEL))
    distribution = dict(agreement_distribution(tuple(agreements)))
    # A strategy where only one model produced a series has no pair to score. It is NOT in
    # the agreement denominator, so it is counted here rather than vanishing from the headline.
    distribution["n_unpaired_strategies"] = len(unpaired)
    distribution["unpaired_strategies"] = list(unpaired)
    return ReferenceArmResult(
        agreements=tuple(agreements),
        distribution=distribution,
        divergences=tuple(divergences),
        sources=sources,
        unpaired=tuple(unpaired),
    )
