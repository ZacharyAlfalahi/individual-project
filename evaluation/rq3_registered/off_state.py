"""Per-correction OFF-state check (registered).

The registration lists ``off_state_known`` among the four runnability conditions: a
correction is only runnable if the as-published state it toggles away from is DOCUMENTED.
This module checks that condition against what the run recorded: each correction's OFF arm
must be the documented as-published state.

**What this check can establish, and what it cannot.** The OFF arm's *configuration* is
verifiable against the run's recorded baseline signature, and a declared no-op's ON ≡ OFF is
verifiable from the invariance gate. A *series-level* known answer — "the OFF arm reproduces
an independently known published number" — exists for exactly one known error in this project,
the ``lead_lag`` positive control, and it is off-lattice. So this module reports a
configuration-level verification per correction and states the series-level limit rather
than implying a reproduction it cannot perform.

Two correction families carry an explicit recorded signature; the rest toggle a plain
as-published default:

  * ``meas_err`` OFF -> the per-paper baseline price family (raw / bbw_2019 / jostova_2013);
  * ``lab_trim`` OFF -> the paper's ex-post trim specification (or none).
"""

from __future__ import annotations

from dataclasses import dataclass

#: Corrections whose OFF state is a recorded per-strategy signature rather than a default.
SIGNATURE_KEYS = {
    "meas_err": "meas_err_off_family",
    "lab_trim": "expost_trim_off",
}

#: What OFF means for the corrections that toggle a plain as-published default.
DEFAULT_OFF_STATE = {
    "stale_price": "no stale-price mask applied (as published)",
    "survivorship": "no survivorship filter applied (as published)",
    "lib_gap": "no signal-return gap inserted; signal_lag = 0 (as published)",
}

SERIES_LEVEL_LIMIT = (
    "configuration-level only: the OFF arm is checked against the run's recorded baseline "
    "signature, not against an independently known published series. The single series-level "
    "known answer in this project is the off-lattice lead_lag positive control."
)


class OffStateInputError(ValueError):
    """A report or signature block cannot supply what the check needs."""


@dataclass(frozen=True)
class OffStateCheck:
    strategy: str
    correction: str
    off_state: object
    source: str                 # "recorded_signature" | "as_published_default"
    documented: bool
    is_declared_no_op: bool
    invariance_verified: bool   # the gate PROVED ON == OFF for this coordinate

    def to_dict(self) -> dict:
        return {
            "strategy": self.strategy,
            "correction": self.correction,
            "off_state": self.off_state,
            "source": self.source,
            "documented": self.documented,
            "is_declared_no_op": self.is_declared_no_op,
            "invariance_verified": self.invariance_verified,
        }


def check_strategy(strategy: str, report: dict, signatures: dict) -> tuple[OffStateCheck, ...]:
    """Every runnable correction's OFF state for one strategy."""
    invariance = {str(row["toggle_id"]): bool(row.get("is_no_op"))
                  for row in (report.get("invariance") or [])}
    verified = {str(row["toggle_id"]): bool(row.get("returns_identical")
                                            and row.get("config_hashes_differ"))
                for row in (report.get("invariance") or [])}
    out: list[OffStateCheck] = []
    for correction in (report.get("runnable_toggles") or ()):
        key = SIGNATURE_KEYS.get(correction)
        if key is not None:
            if key not in signatures:
                raise OffStateInputError(
                    f"{strategy}: correction {correction!r} needs the recorded signature "
                    f"{key!r}, which the run log does not carry"
                )
            off_state, source, documented = signatures[key], "recorded_signature", True
        else:
            off_state = DEFAULT_OFF_STATE.get(correction)
            source, documented = "as_published_default", off_state is not None
        out.append(OffStateCheck(
            strategy=strategy,
            correction=correction,
            off_state=off_state,
            source=source,
            documented=documented,
            is_declared_no_op=invariance.get(correction, False),
            invariance_verified=verified.get(correction, False),
        ))
    return tuple(out)


def off_state_block(reports: dict[str, dict], baseline_signatures: dict) -> dict:
    """The per-correction OFF-state block over ``{strategy: report}``."""
    checks: list[OffStateCheck] = []
    for strategy, report in sorted(reports.items()):
        signatures = baseline_signatures.get(strategy)
        if signatures is None:
            raise OffStateInputError(
                f"{strategy}: the run log carries no baseline signature block"
            )
        checks.extend(check_strategy(strategy, report, signatures))

    undocumented = [c.to_dict() for c in checks if not c.documented]
    return {
        "question": ("is each correction's OFF arm the DOCUMENTED as-published state?"),
        "scope_limit": SERIES_LEVEL_LIMIT,
        "n_checked": len(checks),
        "n_documented": sum(1 for c in checks if c.documented),
        "n_undocumented": len(undocumented),
        "undocumented": undocumented,
        "all_documented": not undocumented,
        "no_ops_proved_inert": sorted(
            f"{c.strategy}:{c.correction}" for c in checks
            if c.is_declared_no_op and c.invariance_verified),
        "checks": [c.to_dict() for c in checks],
    }
