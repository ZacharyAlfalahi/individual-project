"""G4 supplementary PBO — the pre-registered supplementary measurement
(DEV-G4-PBO-1 resolution, E6, 2026-08-06).

The protocol pre-registered `pbo: {scope: realised_family_only, status:
supplementary_caveated}` with the in-file comment "coarse at m = 6; reported
with a caveat, never gating". G4 itself does not compute it.
This module computes it INERTLY, under three conditions:

  (i)  additive only — computed from the FINISHED `ExperimentReport` and the
       driver-held candidate returns, strictly after every gate decision;
       nothing here is importable from a gate module's execution path;
  (ii) provably inert — `tests/unit/test_g4_pbo_supplementary.py` pins that
       no module under `agents/scientist/experimentalist/` imports pbo, and
       that the report's records are byte-identical with and without this
       computation (it takes the report read-only and returns a separate
       block);
  (iii) no decision path is touched: it lives in the `reporting` subpackage,
       sibling to (never inside) the gate stack.

The realised family = the audit-clean proposals whose candidate series
executed (the trials over which G3's joint BH actually operated). The
confirmatory report assembler places the returned block beside the G4
measurements, carrying the caveat verbatim.
"""

from __future__ import annotations

from typing import Iterable, Mapping

import pandas as pd

from shared.stats import pbo_cscv

# The pre-registered caveat, quoted from docs/scientist_protocol.yaml `pbo:`.
PBO_CAVEAT = "coarse at m = 6; reported with a caveat, never gating"
PBO_SCOPE = "realised_family_only"


def _is_realised(record) -> bool:
    """Audit-clean membership read off the record's booleans — the family the
    joint BH ran over (compiled, execution-verified, audit-clean)."""
    b = record.booleans
    return bool(b.compiled and b.execution_verified and b.audit_clean)


def _serialise(obj):
    """A result object as plain data, whatever shape it exposes."""
    if hasattr(obj, "to_dict"):
        return obj.to_dict()
    import dataclasses
    return dataclasses.asdict(obj) if dataclasses.is_dataclass(obj) else obj


def funnel_supplementary(
    reports_by_source: Mapping[str, object],
    returns_by_source: Mapping[str, Mapping[str, pd.Series]],
    *,
    parent_returns: pd.Series | None = None,
    spread: pd.Series | None = None,
) -> dict:
    """The three registered post-hoc diagnostics, computed AFTER every gate
    decision and gating nothing:

      * **deflated Sharpe** — measured inside G4 for each survivor and
        lifted out of the finished records here, never recomputed;
      * **PBO** — the family-level CSCV probability of backtest overfitting, per source
        family, with its pre-registered caveat;
      * **regime decomposition** — the mechanism-consistency comparison over the frozen
        evaluation median, per advanced candidate.

    Read-only over the reports. A missing input yields a typed note, never an exception:
    a supplementary measurement must not be able to take down the run it accompanies."""
    from shared.evaluation.regimes import evaluate_regimes

    out: dict = {
        "computed_after_every_gate_decision": True,
        "gates_nothing": True,
        "sources": {},
    }
    for source, report in sorted(reports_by_source.items()):
        returns = dict(returns_by_source.get(source) or {})
        records = list(getattr(report, "records", ()))
        advanced = list(getattr(report, "advanced", ()))

        dsr = {}
        for record in records:
            measurements = getattr(record, "measurements", None)
            value = getattr(measurements, "deflated_sharpe", None) if measurements else None
            if value is not None:
                dsr[record.proposal_id] = float(value)

        regimes: dict = {}
        for proposal_id in advanced:
            series = returns.get(proposal_id)
            if series is None:
                regimes[proposal_id] = {"status": "no_series_retained"}
                continue
            if spread is None:
                regimes[proposal_id] = {"status": "no_macro_spread_supplied"}
                continue
            regimes[proposal_id] = _serialise(evaluate_regimes(
                series, spread, parent_returns=parent_returns))

        out["sources"][source] = {
            "deflated_sharpe": dsr,
            "n_deflated_sharpe": len(dsr),
            "pbo": family_pbo_supplementary(records, returns),
            "regimes": regimes,
            "n_advanced": len(advanced),
        }
    return out


def family_pbo_supplementary(
    records: Iterable,
    candidate_returns: Mapping[str, pd.Series],
    *,
    n_groups: int = 8,
    test_groups: int = 2,
    purge: int = 0,
    embargo: int = 1,
    months_per_year: int = 12,
) -> dict:
    """The family-level CSCV PBO block for one strategy's proposal family.

    ``records`` is `ExperimentReport.records` (read-only; never mutated);
    ``candidate_returns`` maps proposal_id -> monthly return series (held by
    the driver — the report deliberately retains no series). Returns a
    self-contained dict with a typed ``status``; NEVER raises for family-size
    or overlap shortfalls (a supplementary measurement must not be able to
    take down a run), and never gates anything.
    """
    geometry = {
        "n_groups": n_groups, "test_groups": test_groups,
        "purge": purge, "embargo": embargo,
    }
    base = {"scope": PBO_SCOPE, "caveat": PBO_CAVEAT, "geometry": geometry}

    realised = [r.proposal_id for r in records
                if _is_realised(r) and r.proposal_id in candidate_returns]
    if len(realised) < 2:
        return {**base, "status": "insufficient_family",
                "n_candidates": len(realised),
                "detail": "CSCV needs >= 2 realised candidates with returns"}

    frame = pd.concat(
        {pid: pd.Series(candidate_returns[pid]) for pid in realised},
        axis=1, join="inner",
    ).dropna()
    n_total = max(len(pd.Series(candidate_returns[pid]).dropna()) for pid in realised)
    months_dropped = int(n_total - len(frame))
    if len(frame) < n_groups:
        return {**base, "status": "insufficient_overlap",
                "n_candidates": len(realised), "n_obs": int(len(frame)),
                "months_dropped": months_dropped,
                "detail": f"overlapping months {len(frame)} < n_groups {n_groups}"}

    result = pbo_cscv(frame, n_groups=n_groups, test_groups=test_groups,
                      purge=purge, embargo=embargo, months_per_year=months_per_year)
    return {
        **base,
        "status": "computed",
        "n_candidates": result.n_candidates,
        "n_obs": result.n_obs,
        "n_folds": result.n_folds,
        "pbo": result.pbo,
        "median_oos_relative_rank": result.median_oos_relative_rank,
        "is_best_counts": dict(result.is_best_counts),
        "months_dropped": months_dropped,
    }
