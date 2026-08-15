"""
scripts/run_p4_grid_sweep.py — P4 optimisation-overfitting sweep (extensions WS-D).

EXPLORATORY measurement of backtest overfitting on the anchor modules (mom6 / str /
drf) at the FULLY-CORRECTED corner, development window only. Optimised parameters
are NEVER promoted — the sweep measures the published-vs-grid-best premium (DSR /
PSR / CSCV-PBO), it selects nothing. Execution is gated by
docs/p4_execution_gate.yaml (manually controlled: rebuild flag + report
hashes + a manual review gate); protocol
docs/extensions/contracts/p4_optimisation_protocol.md.

D1 disclosure (verbatim, carried into every report): "all sweep statistics are
exploratory; once seen they constrain any future pre-registration on these
parameters."

D0 RULING (encoded here, docs/thresholds.yaml p4_grids.mom6): the mom6 "skip" IS
the engine signal_lag = the lib_gap bias toggle. The grid therefore sweeps the
TOTAL signal-to-trade gap (`total_signal_gap_months`) with the corrected floor 1 —
lag 0 is excluded because at the corrected corner it would REINTRODUCE the
enumerated look-ahead bias. Centre = published skip = corrected value = 1.

str's dial (`reversal_window_months`) is CONSTRUCTED (F2): the published design has
no window parameter (the reversal signal is structurally the prior-month excess
return), so its sweep is a family-robustness measurement, reported separately from
the mom6/drf published-vs-optimised premium.

Development panel ONLY (2002-2021). NEVER reads /data/holdout/ (guarded). --dry-run
prints the resolved grids without loading any panel or touching data/.

Usage:
  ./.venv/bin/python scripts/run_p4_grid_sweep.py --dry-run
  ./.venv/bin/python scripts/run_p4_grid_sweep.py            # requires the gate
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import math
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = REPO_ROOT / "scripts"
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(SCRIPTS_DIR))

from agents.quant.library.run_config import (  # noqa: E402
    ConstructionConfig,
    RunConfig,
    corrected,
)
from shared.stats import deflated_sharpe_ratio, pbo_cscv  # noqa: E402

THRESHOLDS_FILE = REPO_ROOT / "docs" / "thresholds.yaml"
CACHE_DIR = REPO_ROOT / "results" / "p4_sweep" / "cells"
DEFAULT_OUT_DIR = REPO_ROOT / "docs" / "extensions" / "reports"

# The D1 disclosure — verbatim; appears in the module docstring above, in the JSON
# report, and in the Markdown report.
D1_DISCLOSURE = (
    "all sweep statistics are exploratory; once seen they constrain any future "
    "pre-registration on these parameters."
)

# Per-module constants.
#   anchor            : gold-spec id (evaluation/gold_specs/gold_loader.py).
#   claimed_direction : the SOURCE PAPER's claimed premium sign. str is DRR-2026's
#               published short-term REVERSAL, claimed -1. The gating direction used
#               for orientation is NOT taken from this — it is DERIVED from the
#               realised parent (published-cell) premium sign at run time and
#               ASSERTED against this claim (SC-SCI-8 / D-Q17). On the corrected dev
#               panel str realises POSITIVE momentum (+0.95%/mo, Sharpe ~+1.06 —
#               state-iii), so its derived direction is +1 and DIVERGES from the -1
#               claim (a logged finding, not an error). -0.99 is DRR's published
#               figure, never our dev build.
#   holding   : holding period in months (mom6 gold holding_period=6 STATED —
#               cross-checked against the gold in resolve_grids; str/drf hold
#               monthly). Drives the CPCV embargo = max(1, holding).
MODULE_META: dict[str, dict] = {
    "mom6": {"anchor": "mom6", "claimed_direction": +1, "holding": 6},
    "str": {"anchor": "str", "claimed_direction": -1, "holding": 1},
    "drf": {"anchor": "drf", "claimed_direction": +1, "holding": 1},
}

# str's published (centre) reversal window: the signal is STRUCTURALLY the
# prior-1-month excess return (concept prior_1m_excess_return -> xret), so the
# constructed dial's centre is 1 by definition, not a thresholds signal constant.
STR_PUBLISHED_K = 1

# Panel column name for the str k>1 CONSTRUCTED signal (k-month trailing
# cumulative xret, built via the audited compute_mom6_signal). Not an engine
# reserved column; attached family-indexed as f"{STR_SWEEP_COLUMN}_corr".
STR_SWEEP_COLUMN = "str_rev"

# CSCV geometry — the shared/stats defaults (textbook Bailey et al. C(8,2)).
PBO_N_GROUPS = 8
PBO_TEST_GROUPS = 2


class P4GridError(ValueError):
    """Grid resolution failed loudly: a malformed p4_grids block, a cap breach, or
    a mismatch between a thresholds centre and a gold-typed value that speaks to
    the same quantity."""


# ---------------------------------------------------------------------------
# Guards + small shared helpers.
# ---------------------------------------------------------------------------

def _guard_no_holdout(paths: Iterable[Path]) -> None:
    """Refuse any input whose path has "holdout" as a path COMPONENT.

    Mirrors scripts/run_anchor_descriptive.py `_guard_no_holdout` exactly:
    the check is `"holdout" in p.parts` — COMPONENT EQUALITY, not substring.
    `data/holdout/x.parquet` raises; a file merely NAMED `holdout_notes.txt`
    does not (no component equals "holdout"). Pinned by test_holdout_guard_raises.
    """
    for p in paths:
        if "holdout" in Path(p).parts:
            raise RuntimeError(f"REFUSED: path touches holdout: {p}")


def load_thresholds(path: Path = THRESHOLDS_FILE) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _git_commit() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT, capture_output=True, text=True, check=True,
        ).stdout.strip()
    except Exception:
        return "unknown"


def _finite_or_none(x) -> float | None:
    """Non-finite floats -> None so every report stays strict-JSON valid
    (allow_nan=False), mirroring run_anchor_descriptive._finite_or_none."""
    if x is None:
        return None
    x = float(x)
    return x if math.isfinite(x) else None


def _atomic_write_json(path: Path, obj: dict) -> None:
    """tmp-then-rename so a crashed run never leaves a half-written cache cell."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, default=str))
    os.replace(tmp, path)


# ---------------------------------------------------------------------------
# Execution gate (WS-D D6) — manually controlled; refusals are TYPED strings.
# ---------------------------------------------------------------------------

def load_gate(path: Path) -> dict:
    """Parse the execution-gate YAML. Missing file is fail-loud (the gate is a
    committed governance artefact, not an optional config)."""
    gate = yaml.safe_load(Path(path).read_text())
    if not isinstance(gate, dict):
        raise ValueError(f"execution gate at {path} is not a YAML mapping")
    return gate


def check_gate(gate: dict, repo_root: Path) -> str | None:
    """None iff execution is allowed; else a TYPED refusal string.

    Allowed iff panel_rebuild_complete is True AND scope_approved is True AND
    every recorded sha256 in panel_report_sha256 is a non-null string equal to
    sha256(file bytes) of the named report under <repo_root>/data/development/.
    The hash pin means the flag cannot be satisfied by an old (pre-rebuild) panel.
    This function is only reached on the execution path — never under --dry-run.
    """
    if gate.get("panel_rebuild_complete") is not True:
        return (
            "P4_GATE_REFUSED(panel_rebuild_incomplete): panel_rebuild_complete is "
            "not true — the post-Phase-3 panel rebuild has not been finalised"
        )
    if gate.get("scope_approved") is not True:
        return (
            "P4_GATE_REFUSED(scope_approved_missing): scope_approved is "
            "not true (E4 — P4 scope approval)"
        )
    hashes = gate.get("panel_report_sha256")
    if not isinstance(hashes, dict) or not hashes:
        return (
            "P4_GATE_REFUSED(no_report_hashes): panel_report_sha256 must record "
            "at least one rebuilt-panel report hash"
        )
    for name in sorted(hashes):
        recorded = hashes[name]
        if not isinstance(recorded, str) or not recorded.strip():
            return (
                f"P4_GATE_REFUSED(null_report_hash): panel_report_sha256[{name!r}] "
                "is null/empty — the rebuilt panel report hash has not been recorded"
            )
        report = Path(repo_root) / "data" / "development" / name
        if not report.exists():
            return (
                f"P4_GATE_REFUSED(report_file_missing): {report} does not exist — "
                "cannot verify the recorded hash against the rebuilt panel report"
            )
        actual = _sha256_file(report)
        if actual != recorded:
            return (
                f"P4_GATE_REFUSED(report_hash_mismatch): {name}: recorded "
                f"{recorded[:12]}... != actual {actual[:12]}... — the gate is "
                "pinned to a different panel build"
            )
    return None


# ---------------------------------------------------------------------------
# Grid resolution — thresholds p4_grids + signals centres -> cells per module.
# ---------------------------------------------------------------------------

def _drf_rank(window: int) -> int:
    """rank_rule ceil_5pct_of_window: keep the "5% VaR" concept faithful as the
    window moves — rank = max(1, ceil(0.05 * window)). 2 at 36, 3 at 48."""
    return max(1, math.ceil(0.05 * window))


def _gold_typed_int(inherited) -> int | str:
    """A gold field's typed int value, or the literal string "gold_untyped" when
    the gold carries no STATED int for it (UNKNOWN / a formula string)."""
    if (
        getattr(inherited, "tag", None) == "STATED"
        and isinstance(inherited.value, int)
        and not isinstance(inherited.value, bool)
    ):
        return int(inherited.value)
    return "gold_untyped"


def _gold_cross_check(published: dict[str, dict]) -> dict:
    """Cross-check thresholds centres against the gold-typed int fields
    (n_groups / signal_lag / holding_period) via load_gold_spec.

    FAIL-LOUD (P4GridError) only where a STATED gold int speaks to the SAME
    quantity as a thresholds centre:
      * mom6 signal_lag (the published Jostova skip, STATED 1) must equal the
        published total_signal_gap_months centre (D0: the skip IS the engine
        signal_lag).
      * mom6 holding_period (STATED 6) must equal MODULE_META's holding constant
        (it drives the CPCV embargo / mom6 information span).
    Everything else is recorded, not compared: n_groups has no swept centre;
    str's STATED signal_lag=0 is the published lib_gap value, NOT the constructed
    reversal-window dial (the sweep runs the corrected corner, floor 1 per D0).
    A gold that cannot LOAD degrades to a recorded warning (test environments may
    lack the artefacts); a genuine value mismatch never degrades.
    """
    from evaluation.gold_specs.gold_loader import load_gold_spec

    out: dict[str, dict] = {}
    for module, meta in MODULE_META.items():
        try:
            spec = load_gold_spec(meta["anchor"])
        except Exception as exc:  # artefacts absent (e.g. test env) -> warn, not fail
            out[module] = {"status": "warning", "detail": f"gold unavailable: {exc}"}
            continue
        typed = {
            "signal_lag": _gold_typed_int(spec.part2.signal_lag),
            "holding_period": _gold_typed_int(spec.part2.holding_period),
            "n_groups": _gold_typed_int(spec.part2.legs[0].n_groups),
        }
        if module == "mom6":
            gap_centre = published["mom6"]["total_signal_gap_months"]
            if typed["signal_lag"] != "gold_untyped" and typed["signal_lag"] != gap_centre:
                raise P4GridError(
                    f"mom6 gold signal_lag={typed['signal_lag']} != published "
                    f"total_signal_gap_months centre {gap_centre} (D0: the skip IS "
                    "the engine signal_lag) — thresholds/gold drift"
                )
            if (
                typed["holding_period"] != "gold_untyped"
                and typed["holding_period"] != meta["holding"]
            ):
                raise P4GridError(
                    f"mom6 gold holding_period={typed['holding_period']} != module "
                    f"holding constant {meta['holding']} — the CPCV embargo / "
                    "information span would be wrong"
                )
        out[module] = {"status": "ok", "typed": typed}
    return out


def resolve_grids(thresholds: dict, *, gold_check: bool = True) -> tuple[dict, dict]:
    """Resolve docs/thresholds.yaml `p4_grids` (+ `signals` centres) into the
    per-module cell lists.

    Returns (grids, meta):
      grids : module -> list of cell param dicts —
        mom6: {"formation_months", "total_signal_gap_months", "min_obs"}
        str : {"reversal_window_months"}
        drf : {"window", "min_obs", "rank"}
      meta  : {"centres", "published", "filtered", "trial_counts", "gold_check"}.

    Fail-loud on: a gap level < 1 (D0 — lag 0 reintroduces lib_gap), an unknown
    min_obs_rule / rank_rule, a published rank that contradicts the rank rule at
    the centre window, a module exceeding max_cells_per_module, or a gold-typed
    mismatch. `gold_check=False` makes the function testable from a plain
    thresholds dict without touching the gold artefacts.
    """
    p4 = thresholds["p4_grids"]
    sig = thresholds["signals"]
    cap = int(p4["max_cells_per_module"])

    # --- mom6: formation offsets around the signals centre x gap levels ------
    f_centre = int(sig["mom6"]["formation_months"])
    formations = sorted(f_centre + int(o) for o in p4["mom6"]["formation_months"]["offsets"])
    if any(f < 1 for f in formations):
        raise P4GridError(f"mom6 formation grid contains a non-positive window: {formations}")
    gaps = sorted(int(g) for g in p4["mom6"]["total_signal_gap_months"]["levels"])
    if min(gaps) < 1:
        raise P4GridError(
            "D0: total_signal_gap_months has floor 1 — a gap of 0 at the corrected "
            f"corner would reintroduce the lib_gap look-ahead; got levels {gaps}"
        )
    if p4["mom6"]["min_obs_rule"] != "equals_formation":
        raise P4GridError(
            f"unknown mom6 min_obs_rule {p4['mom6']['min_obs_rule']!r} "
            "(only 'equals_formation' is registered)"
        )
    mom6_cells = [
        {"formation_months": f, "total_signal_gap_months": g, "min_obs": f}
        for f in formations
        for g in gaps
    ]
    # Published gap centre = the corrected corner's signal_lag (D0: published
    # skip = corrected value = 1) — derived from the RunConfig, not re-typed here.
    published_gap = corrected().construction.signal_lag
    if published_gap not in gaps:
        raise P4GridError(
            f"published gap centre {published_gap} (corrected signal_lag) is not "
            f"in the gap levels {gaps} — the published cell would be missing"
        )

    # --- str: constructed reversal-window dial (levels; centre = 1) ----------
    ks = sorted(int(k) for k in p4["str"]["reversal_window_months"]["levels"])
    if min(ks) < 1:
        raise P4GridError(f"str reversal_window_months must be >= 1; got {ks}")
    str_cells = [{"reversal_window_months": k} for k in ks]
    if STR_PUBLISHED_K not in ks:
        raise P4GridError(
            f"str published window {STR_PUBLISHED_K} is not in the levels {ks}"
        )

    # --- drf: window x min_obs offsets around the var_5pct centres -----------
    w_centre = int(sig["var_5pct"]["window"])
    m_centre = int(sig["var_5pct"]["min_obs"])
    windows = sorted(w_centre + int(o) for o in p4["drf"]["window"]["offsets"])
    min_obss = sorted(m_centre + int(o) for o in p4["drf"]["min_obs"]["offsets"])
    if any(w < 1 for w in windows) or any(m < 1 for m in min_obss):
        raise P4GridError(f"drf grid contains a non-positive value: {windows} / {min_obss}")
    if p4["drf"]["rank_rule"] != "ceil_5pct_of_window":
        raise P4GridError(
            f"unknown drf rank_rule {p4['drf']['rank_rule']!r} "
            "(only 'ceil_5pct_of_window' is registered)"
        )
    published_rank = int(sig["var_5pct"]["rank"])
    if _drf_rank(w_centre) != published_rank:
        raise P4GridError(
            f"rank_rule at the centre window {w_centre} gives {_drf_rank(w_centre)} "
            f"but signals.var_5pct.rank = {published_rank} — centre inconsistency"
        )
    drf_cells: list[dict] = []
    drf_filtered: list[dict] = []
    for w in windows:
        for m in min_obss:
            if m > w:
                # compute_var_5pct requires min_obs <= window; the combo is not a
                # runnable estimand. Recorded, never silently dropped.
                drf_filtered.append({"window": w, "min_obs": m})
                continue
            drf_cells.append({"window": w, "min_obs": m, "rank": _drf_rank(w)})

    grids = {"mom6": mom6_cells, "str": str_cells, "drf": drf_cells}
    for module, cells in grids.items():
        if len(cells) > cap:
            raise P4GridError(
                f"{module} grid has {len(cells)} cells > max_cells_per_module={cap}"
            )

    published = {
        "mom6": {
            "formation_months": f_centre,
            "total_signal_gap_months": published_gap,
            "min_obs": f_centre,
        },
        "str": {"reversal_window_months": STR_PUBLISHED_K},
        "drf": {"window": w_centre, "min_obs": m_centre, "rank": published_rank},
    }
    for module, pub in published.items():
        if pub not in grids[module]:
            raise P4GridError(f"{module} published cell {pub} is not in its own grid")

    meta = {
        "centres": {
            "mom6": {"formation_months": f_centre, "total_signal_gap_months": published_gap},
            "str": {"reversal_window_months": STR_PUBLISHED_K},
            "drf": {"window": w_centre, "min_obs": m_centre, "rank": published_rank},
        },
        "published": published,
        "filtered": {"drf": drf_filtered},
        "trial_counts": {m: len(c) for m, c in grids.items()},
        "gold_check": (
            _gold_cross_check(published) if gold_check else {"status": "skipped"}
        ),
    }
    return grids, meta


# ---------------------------------------------------------------------------
# Content-addressed cell cache key.
# ---------------------------------------------------------------------------

def cell_cache_key(
    module: str,
    params: dict,
    panel_sha: str,
    corner_hash: str,
    pipeline_version: str = "p4-v1",
) -> str:
    """sha256 over a canonical (sorted-keys JSON) encoding of the full cell
    identity. Stable across runs; any change to module / params / panel bytes /
    corner config / pipeline version changes the key."""
    payload = json.dumps(
        {
            "module": module,
            "params": params,
            "panel_sha": panel_sha,
            "corner_hash": corner_hash,
            "pipeline_version": pipeline_version,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# One cell — the ONLY function that touches real data (reachable only through
# the execution gate). Heavy imports are local so --dry-run / unit tests never
# pull the full agent + builder stack.
# ---------------------------------------------------------------------------

def _rebind_score(adapted, column: str):
    """Return a copy of an AdaptResult whose every leg's score Binding points at
    `column` (the str k>1 CONSTRUCTED signal). Mirrors the dataclasses.replace
    pattern of agents/auditor/checks/cell_runner._override_construction."""
    from agents.quant.config import Binding, Evidence, QuantConfig

    new_legs = []
    for lc in adapted.leg_calls:
        cfg = lc.result
        if not isinstance(cfg, QuantConfig):
            raise RuntimeError(
                f"leg {lc.strategy_id!r} carries a non-runnable result "
                f"({type(cfg).__name__}); cannot rebind its score"
            )
        binding = Binding(
            column,
            "BOUND",
            Evidence(
                column=column,
                note=(
                    "P4 str CONSTRUCTED dial: k-month trailing cumulative xret "
                    "(family-robustness measurement, F2) — sweep-only rebind, "
                    "never promoted"
                ),
            ),
        )
        new_legs.append(dataclasses.replace(lc, result=dataclasses.replace(cfg, score=binding)))
    return dataclasses.replace(adapted, leg_calls=tuple(new_legs))


def run_cell(
    module: str,
    params: dict,
    panel: pd.DataFrame,
    subs,
    corner: RunConfig,
    *,
    thresholds: dict | None = None,
) -> dict:
    """Run one grid cell at the fully-corrected corner and return its serialisable
    record {module, params, summary, monthly}.

    Pipeline (the run_quant.py / cell_runner idioms):
      1. Recompute the swept signal on the CORR family via the parameterised
         audited builders — mom6: compute_mom6_signal(ret_corr, formation, min_obs)
         (the gap enters as the signal_lag override, D0); drf:
         compute_var_5pct(ret_corr, window, min_obs, rank, multiplier from
         thresholds); str: k=1 uses the panel's own xret (published structural
         signal, no rebuild), k>1 reuses compute_mom6_signal on xret_corr and the
         leg's score is rebound to the constructed column.
      2. view() the base panel at the corrected corner (corr family + stale mask
         + terminal rows kept), attaching the swept signal family-resolved.
      3. adapt_spec(load_gold_spec(anchor), standing_subs=subs) — each anchor's
         own compiled construction.
      4. _override_construction with signal_lag = (mom6: the cell's total gap;
         str/drf: 1) and expost_trim="none" — the corrected-corner construction,
         exactly as the auditor lattice applies it.
      5. run_strategy (safe_rate/benchmark None — run_quant.py's uniform choice).
    """
    import run_quant  # scripts sibling: summarize_run + the B1 conventions
    from agents.auditor.checks.cell_runner import _override_construction
    from agents.librarian.adapter.adapt import adapt_spec
    from agents.quant.config.runner import StrategyResult, run_strategy
    from agents.quant.library.views import view
    from build_mom6_signal import compute_mom6_signal
    from build_var_5pct import compute_var_5pct
    from evaluation.gold_specs.gold_loader import load_gold_spec

    if module not in MODULE_META:
        raise ValueError(f"unknown module {module!r}; expected one of {sorted(MODULE_META)}")
    thresholds = thresholds if thresholds is not None else load_thresholds()
    meta = MODULE_META[module]

    # (1) swept signal on the corr family (the corner's price family).
    signals: pd.DataFrame | None
    rebind_col: str | None = None
    if module == "mom6":
        sig_in = panel[["cusip", "date", "ret_corr"]].rename(columns={"ret_corr": "ret"})
        signals = compute_mom6_signal(
            sig_in, int(params["formation_months"]), int(params["min_obs"])
        ).rename(columns={"mom6": "mom6_corr"})
        lag = int(params["total_signal_gap_months"])
    elif module == "drf":
        multiplier = float(thresholds["signals"]["var_5pct"]["multiplier"])
        sig_in = panel[["cusip", "date", "ret_corr"]].rename(columns={"ret_corr": "ret"})
        signals = compute_var_5pct(
            sig_in, int(params["window"]), int(params["min_obs"]),
            int(params["rank"]), multiplier,
        ).rename(columns={"var_5pct": "var_5pct_corr"})
        lag = 1  # corrected lib_gap
    else:  # str
        k = int(params["reversal_window_months"])
        if k == 1:
            signals = None  # panel xret as-is — the published structural signal
        else:
            sig_in = panel[["cusip", "date", "xret_corr"]].rename(columns={"xret_corr": "ret"})
            signals = compute_mom6_signal(sig_in, k, k).rename(
                columns={"mom6": f"{STR_SWEEP_COLUMN}_corr"}
            )
            rebind_col = STR_SWEEP_COLUMN
        lag = 1  # corrected lib_gap

    # (2) the engine-shape panel at the corrected corner + the cell's construction.
    cell_rc = dataclasses.replace(
        corner, construction=ConstructionConfig(signal_lag=lag, expost_trim="none")
    )
    viewed = view(panel, cell_rc, signals=signals)

    # (3) compile the anchor's own gold spec (hash-verified standing subs).
    adapted = adapt_spec(load_gold_spec(meta["anchor"]), standing_subs=subs)
    if adapted.refused:
        raise RuntimeError(
            f"{module}: gold spec refused to compile — the anchors are supported, "
            f"so a refusal here is a build error: {[r.to_dict() for r in adapted.refusals]}"
        )

    # (4) str k>1: point the score at the constructed k-month column.
    if rebind_col is not None:
        adapted = _rebind_score(adapted, rebind_col)

    # (5) corrected-corner construction override (auditor lattice semantics).
    overridden = _override_construction(adapted, cell_rc)

    # (6) run.
    result = run_strategy(overridden, viewed, safe_rate=None, benchmark=None)
    if not isinstance(result, StrategyResult):
        raise RuntimeError(
            f"{module}: run_strategy returned {type(result).__name__} for a "
            "non-refused strategy — a compiled cell must run to a StrategyResult"
        )

    monthly_df = result.monthly_returns
    monthly: list[tuple[str, float]] = []
    n_dropped = 0
    for d, r in zip(monthly_df["date"], monthly_df["strategy_ret"]):
        r = float(r)
        if math.isfinite(r):
            monthly.append((pd.Timestamp(d).date().isoformat(), r))
        else:
            n_dropped += 1

    return {
        "module": module,
        "params": dict(params),
        "signal_lag_used": lag,
        "expost_trim_used": "none",
        "summary": run_quant.summarize_run(result),
        "monthly": monthly,
        "n_nonfinite_dropped": n_dropped,
    }


# ---------------------------------------------------------------------------
# Deflation — published vs grid-best, DSR/PSR + CSCV-PBO. Pure given cell dicts.
# ---------------------------------------------------------------------------

def _cell_id(params: dict) -> str:
    return json.dumps(params, sort_keys=True, separators=(",", ":"))


def _returns_matrix(cells: list[dict]) -> tuple[pd.DataFrame, int]:
    """T x N per-month cell-return matrix over the INNER JOIN of months where ALL
    cells have a return (pbo_cscv refuses NaN cells, so the join is explicit and
    the dropped-month count is recorded). Rows in time order (ISO dates sort)."""
    series = {
        _cell_id(c["params"]): pd.Series(
            {d: float(r) for d, r in c["monthly"]}, dtype=float
        )
        for c in cells
    }
    wide = pd.DataFrame(series)          # union of months
    inner = wide.dropna().sort_index()   # months where ALL cells have returns
    return inner, int(len(wide) - len(inner))


def _module_purge(module: str, cells: list[dict]) -> int:
    """The module's information span, from GRID MAXIMA (not the published cell) so
    every candidate's formation information is purged from every CPCV train set:
      mom6: max formation + max total gap + holding (6) — a formation-window month
            can influence realised returns up to that many months later under the
            staggered H=6 overlap;
      str : max k + 1 — the k-month constructed window plus the 1-month
            signal-to-trade gap;
      drf : 1 — the protocol-ruled span for the slow rolling-VaR signal at h=1.
    """
    if module == "mom6":
        return (
            max(int(c["params"]["formation_months"]) for c in cells)
            + max(int(c["params"]["total_signal_gap_months"]) for c in cells)
            + MODULE_META["mom6"]["holding"]
        )
    if module == "str":
        return max(int(c["params"]["reversal_window_months"]) for c in cells) + 1
    if module == "drf":
        return 1
    raise ValueError(f"unknown module {module!r}")


def deflate_module(
    module: str,
    cells: list[dict],
    published_params: dict,
    *,
    months_per_year: int = 12,
) -> dict:
    """Deflate one module's grid: published cell vs grid-best-in-the-derived-direction,
    raw uplift, DSR of the grid-best (n_trials = grid size), PSR-equivalent of the
    published cell (n_trials = 1), and CSCV PBO over the per-month cell returns.

    Pure given cell dicts ({params, summary, monthly}) — unit-testable with synthetic
    cells. The gating direction is DERIVED from the realised parent (published-cell)
    premium sign and ASSERTED against MODULE_META's claimed_direction (SC-SCI-8 /
    D-Q17): a divergence (str realises +1 momentum vs DRR's -1 reversal claim) is a
    logged finding, and grid-best is oriented on the DERIVED sign, never on the paper's
    claim — so a negative claim can no longer invert the orientation of a
    positive-premium parent. The DSR/PSR call convention mirrors
    agents/scientist/experimentalist/robustness.py exactly —
    deflated_sharpe_ratio(sr / sqrt(months_per_year), n_obs, n_trials, sr_std=std of
    per-cell periodic Sharpes); the *_oriented variants apply the derived direction
    first (direction * sr) so the deflation is read in the parent's realised direction.
    """
    if module not in MODULE_META:
        raise ValueError(f"unknown module {module!r}; expected one of {sorted(MODULE_META)}")
    if not cells:
        raise ValueError(f"{module}: no cells to deflate")

    for c in cells:
        s = c["summary"]
        if s.get("sharpe") is None or not math.isfinite(float(s["sharpe"])):
            raise ValueError(
                f"{module}: cell {_cell_id(c['params'])} has a non-finite Sharpe — "
                "the grid cannot be deflated over a degenerate cell"
            )

    published = next((c for c in cells if c["params"] == published_params), None)
    if published is None:
        raise ValueError(
            f"{module}: published cell {published_params} not found in the grid"
        )
    pub_sr = float(published["summary"]["sharpe"])
    pub_mean = float(published["summary"]["average"])

    # DERIVE the gating direction from the realised parent (published-cell) premium
    # sign, then ASSERT it against the source paper's claim (SC-SCI-8 / D-Q17). The
    # str inversion (DRR claims -1 reversal; the corrected dev parent realises +1
    # momentum) surfaces here as a logged divergence, NOT an error — and grid-best is
    # oriented on the DERIVED sign so a wrong claim can never invert a positive parent.
    claimed_direction = MODULE_META[module]["claimed_direction"]
    direction = 1 if pub_mean >= 0.0 else -1
    direction_divergence = direction != claimed_direction

    # Grid-best IN THE DERIVED DIRECTION: argmax of direction * Sharpe.
    best = max(cells, key=lambda c: direction * float(c["summary"]["sharpe"]))
    best_sr = float(best["summary"]["sharpe"])
    best_mean = float(best["summary"]["average"])
    sqrt_mpy = math.sqrt(months_per_year)

    # sr_std = std (ddof=1) of the PER-CELL PERIODIC Sharpes — the cross-trial
    # dispersion expected_max_sharpe deflates against (robustness.py convention).
    periodic = np.array([float(c["summary"]["sharpe"]) / sqrt_mpy for c in cells])
    sr_std = float(np.std(periodic, ddof=1))

    n_trials = len(cells)
    best_n = int(best["summary"]["n_months"])
    pub_n = int(published["summary"]["n_months"])
    dsr_best = deflated_sharpe_ratio(best_sr / sqrt_mpy, best_n, n_trials, sr_std=sr_std)
    dsr_best_oriented = deflated_sharpe_ratio(
        direction * best_sr / sqrt_mpy, best_n, n_trials, sr_std=sr_std
    )
    psr_pub = deflated_sharpe_ratio(pub_sr / sqrt_mpy, pub_n, 1, sr_std=sr_std)
    psr_pub_oriented = deflated_sharpe_ratio(
        direction * pub_sr / sqrt_mpy, pub_n, 1, sr_std=sr_std
    )

    # CSCV PBO over the inner-join T x N matrix; purge from grid maxima (formula
    # in _module_purge), embargo = max(1, holding) — mom6 6, str/drf 1.
    matrix, months_dropped = _returns_matrix(cells)
    purge = _module_purge(module, cells)
    embargo = max(1, MODULE_META[module]["holding"])
    pbo = pbo_cscv(
        matrix,
        n_groups=PBO_N_GROUPS,
        test_groups=PBO_TEST_GROUPS,
        purge=purge,
        embargo=embargo,
        months_per_year=months_per_year,
    )

    return {
        "module": module,
        "direction": direction,
        "claimed_direction": claimed_direction,
        "direction_divergence": direction_divergence,
        "direction_note": (
            None if not direction_divergence else
            f"derived direction {direction:+d} (realised parent premium "
            f"{'positive' if direction > 0 else 'negative'}) diverges from the "
            f"source-paper claim {claimed_direction:+d}; grid-best oriented on the "
            f"derived sign (SC-SCI-8 / D-Q17 inversion fix). For str this is the "
            f"state-iii finding: DRR's -0.99 reversal is not realised on the corrected "
            f"dev panel, which shows positive momentum."
        ),
        "n_trials": n_trials,
        "months_per_year": months_per_year,
        "published": {
            "params": dict(published_params),
            "sharpe": _finite_or_none(pub_sr),
            "mean": _finite_or_none(pub_mean),
            "n_months": pub_n,
            "psr_n_trials_1": _finite_or_none(psr_pub),
            "psr_n_trials_1_oriented": _finite_or_none(psr_pub_oriented),
        },
        "grid_best": {
            "params": dict(best["params"]),
            "sharpe": _finite_or_none(best_sr),
            "mean": _finite_or_none(best_mean),
            "n_months": best_n,
        },
        "uplift": {
            "sharpe": _finite_or_none(best_sr - pub_sr),
            "mean": _finite_or_none(best_mean - pub_mean),
            "sharpe_in_derived_direction": _finite_or_none(direction * (best_sr - pub_sr)),
        },
        "sr_std_periodic": _finite_or_none(sr_std),
        "dsr_grid_best": _finite_or_none(dsr_best),
        "dsr_grid_best_oriented": _finite_or_none(dsr_best_oriented),
        "pbo": {
            **pbo.to_dict(),
            "months_in_matrix": int(len(matrix)),
            "months_dropped_inner_join": months_dropped,
        },
    }


# ---------------------------------------------------------------------------
# Report — stamped JSON + EXPLORATORY-labelled Markdown.
# ---------------------------------------------------------------------------

def _md_module_section(module: str, res: dict, cells: list[dict]) -> list[str]:
    lines: list[str] = []
    title = f"## {module} (EXPLORATORY)"
    if module == "str":
        title = f"## {module} (EXPLORATORY) — CONSTRUCTED dial"
    lines.append(title)
    lines.append("")
    if module == "str":
        lines.append(
            "The `reversal_window_months` dial is **CONSTRUCTED** (F2): the published "
            "design has no window parameter — the reversal signal is structurally the "
            "prior-1-month excess return. This sweep is a **family-robustness "
            "measurement**, distinct from the mom6/drf published-vs-optimised premium, "
            "and is reported separately for that reason. DRR's *claimed* direction is "
            "-1 (reversal), but the gating direction is DERIVED from the realised "
            "parent and here is +1: the corrected dev panel shows positive momentum "
            "(state-iii), so grid-best = most positive Sharpe (SC-SCI-8 / D-Q17)."
        )
        lines.append("")
    pub, best = res["published"], res["grid_best"]
    lines.append(
        f"Table (EXPLORATORY): {module} published vs grid-best-in-derived-direction."
    )
    lines.append("")
    lines.append("| cell | params | sharpe | mean/mo | n_months |")
    lines.append("|---|---|---|---|---|")
    lines.append(
        f"| published | `{_cell_id(pub['params'])}` | {pub['sharpe']} | "
        f"{pub['mean']} | {pub['n_months']} |"
    )
    lines.append(
        f"| grid-best | `{_cell_id(best['params'])}` | {best['sharpe']} | "
        f"{best['mean']} | {best['n_months']} |"
    )
    lines.append("")
    lines.append(f"Table (EXPLORATORY): {module} deflation statistics.")
    lines.append("")
    lines.append("| statistic | value |")
    lines.append("|---|---|")
    lines.append(f"| direction (DERIVED from realised parent, SC-SCI-8/D-Q17) | {res['direction']} |")
    lines.append(f"| claimed_direction (source paper) | {res['claimed_direction']} |")
    if res.get("direction_divergence"):
        lines.append(f"| **direction divergence** | {res['direction_note']} |")
    lines.append(f"| n_trials (grid size) | {res['n_trials']} |")
    lines.append(f"| uplift sharpe (grid-best - published) | {res['uplift']['sharpe']} |")
    lines.append(f"| uplift mean (grid-best - published) | {res['uplift']['mean']} |")
    lines.append(
        f"| uplift sharpe in derived direction | "
        f"{res['uplift']['sharpe_in_derived_direction']} |"
    )
    lines.append(f"| DSR grid-best (n_trials={res['n_trials']}) | {res['dsr_grid_best']} |")
    lines.append(f"| DSR grid-best, oriented | {res['dsr_grid_best_oriented']} |")
    lines.append(f"| PSR published (n_trials=1) | {res['published']['psr_n_trials_1']} |")
    lines.append(
        f"| PSR published, oriented | {res['published']['psr_n_trials_1_oriented']} |"
    )
    lines.append(f"| PBO (CSCV) | {res['pbo']['pbo']} |")
    lines.append(f"| PBO folds / purge / embargo | {res['pbo']['n_folds']} / "
                 f"{res['pbo']['purge']} / {res['pbo']['embargo']} |")
    lines.append(f"| months in matrix (dropped by inner join) | "
                 f"{res['pbo']['months_in_matrix']} ({res['pbo']['months_dropped_inner_join']}) |")
    lines.append("")
    lines.append(f"Table (EXPLORATORY): {module} per-cell grid results.")
    lines.append("")
    lines.append("| params | sharpe | mean/mo | n_months |")
    lines.append("|---|---|---|---|")
    for c in cells:
        s = c["summary"]
        lines.append(
            f"| `{_cell_id(c['params'])}` | {s.get('sharpe')} | "
            f"{s.get('average')} | {s.get('n_months')} |"
        )
    lines.append("")
    return lines


def write_report(results: dict, out_md: Path, out_json: Path) -> None:
    """Write the stamped EXPLORATORY report (strict JSON + Markdown, atomic).

    `results` = {"modules": {module: deflate_module output}, "cells": {module:
    [cell records]}, "grids_meta": resolve_grids meta}. The stamp carries
    run_timestamp / git_commit / thresholds_sha256 / gate-file sha256 / per-module
    trial counts / status EXPLORATORY; the Markdown title and EVERY table caption
    carry the word EXPLORATORY, and the D1 disclosure appears verbatim.
    """
    gate_rel = None
    gate_sha = None
    try:
        thresholds = load_thresholds()
        gate_path = REPO_ROOT / thresholds["p4_grids"]["execution_gate_file"]
        gate_rel = str(gate_path.relative_to(REPO_ROOT))
        gate_sha = _sha256_file(gate_path)
    except Exception:
        pass  # the stamp records null rather than blocking an exploratory write

    trial_counts = {m: len(c) for m, c in results.get("cells", {}).items()}
    stamped = {
        "status": "EXPLORATORY",
        "d1_disclosure": D1_DISCLOSURE,
        "run_timestamp": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "thresholds_sha256": _sha256_file(THRESHOLDS_FILE),
        "gate_file": gate_rel,
        "gate_file_sha256": gate_sha,
        "trial_counts": trial_counts,
        "window": "development 2002-2021 (holdout untouched)",
        "corner": "fully corrected (corr family + stale mask + terminal rows kept; "
                  "signal_lag floor 1 per D0; expost_trim none)",
        **results,
    }

    out_json.parent.mkdir(parents=True, exist_ok=True)
    tmp_json = out_json.with_suffix(out_json.suffix + ".tmp")
    with open(tmp_json, "w") as f:
        json.dump(stamped, f, indent=2, allow_nan=False, default=str)
    os.replace(tmp_json, out_json)

    lines = [
        "# EXPLORATORY — P4 optimisation-overfitting sweep (WS-D)",
        "",
        f"> D1 disclosure: {D1_DISCLOSURE}",
        "",
        "Optimised parameters are NEVER promoted; this report measures the "
        "published-vs-grid-best premium at the fully-corrected corner on the "
        "development window only.",
        "",
        "- status: **EXPLORATORY**",
        f"- run_timestamp: {stamped['run_timestamp']}",
        f"- git_commit: {stamped['git_commit']}",
        f"- thresholds_sha256: {stamped['thresholds_sha256']}",
        f"- gate_file: {stamped['gate_file']} (sha256 {stamped['gate_file_sha256']})",
        f"- trial counts: {json.dumps(trial_counts, sort_keys=True)}",
        f"- window: {stamped['window']}",
        f"- corner: {stamped['corner']}",
        "",
    ]
    for module in ("mom6", "drf", "str"):  # str last — its CONSTRUCTED dial is set apart
        if module in results.get("modules", {}):
            lines.extend(
                _md_module_section(
                    module, results["modules"][module], results.get("cells", {}).get(module, [])
                )
            )
    tmp_md = out_md.with_suffix(out_md.suffix + ".tmp")
    tmp_md.write_text("\n".join(lines))
    os.replace(tmp_md, out_md)
    print(f"Written: {out_md}")
    print(f"Written: {out_json}")


# ---------------------------------------------------------------------------
# Orchestration.
# ---------------------------------------------------------------------------

def _dry_run_preview(grids: dict, meta: dict, gate_path: Path) -> dict:
    """The --dry-run payload: resolved grids, centres, trial counts and a
    cache-key preview — computed WITHOUT loading any panel or touching data/
    (the panel sha is a literal placeholder; the gate is displayed, not enforced,
    because enforcing it would hash files under data/development/)."""
    gate_display: dict = {"file": str(gate_path), "enforced_in_dry_run": False}
    try:
        gate = load_gate(gate_path)
        gate_display.update(
            {
                "panel_rebuild_complete": gate.get("panel_rebuild_complete"),
                "scope_approved": gate.get("scope_approved"),
            }
        )
    except Exception as exc:
        gate_display["error"] = str(exc)

    corner_hash = corrected().hash()
    preview_keys = {
        module: cell_cache_key(
            module, cells[0], "DRY-RUN-PANEL-SHA-UNRESOLVED", corner_hash
        )
        for module, cells in grids.items()
        if cells
    }
    counts = dict(meta["trial_counts"])
    return {
        "mode": "dry-run",
        "status": "EXPLORATORY",
        "d1_disclosure": D1_DISCLOSURE,
        "gate": gate_display,
        "centres": meta["centres"],
        "published": meta["published"],
        "filtered_combos": meta["filtered"],
        "trial_counts": {**counts, "total": sum(counts.values())},
        "gold_check": meta["gold_check"],
        "cache_key_preview_first_cell": preview_keys,
        "grids": grids,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=(
            "P4 optimisation-overfitting sweep (WS-D) — EXPLORATORY; execution "
            "gated by docs/p4_execution_gate.yaml"
        )
    )
    ap.add_argument(
        "--dry-run", action="store_true",
        help="print resolved grids / centres / trial counts / cache-key preview "
             "without loading any panel or touching data/",
    )
    ap.add_argument(
        "--out-dir", type=Path, default=DEFAULT_OUT_DIR,
        help=f"report output directory (default {DEFAULT_OUT_DIR})",
    )
    args = ap.parse_args(argv)

    thresholds = load_thresholds()
    grids, meta = resolve_grids(thresholds)
    gate_path = REPO_ROOT / thresholds["p4_grids"]["execution_gate_file"]

    if args.dry_run:
        print(json.dumps(_dry_run_preview(grids, meta, gate_path), indent=2, default=str))
        return 0

    # ---- execution path (gated) -------------------------------------------
    refusal = check_gate(load_gate(gate_path), REPO_ROOT)
    if refusal is not None:
        print(refusal)
        return 2

    import run_quant  # heavy scripts sibling — only after the gate opens

    base_panel_path = run_quant.BASE_PANEL
    _guard_no_holdout([base_panel_path, THRESHOLDS_FILE, gate_path])

    subs = run_quant.load_standing_subs_verified()
    panel = pd.read_parquet(base_panel_path)
    panel_sha = _sha256_file(base_panel_path)
    corner = corrected()

    cell_records: dict[str, list[dict]] = {}
    module_results: dict[str, dict] = {}
    for module, cells in grids.items():
        records = []
        for params in cells:
            key = cell_cache_key(module, params, panel_sha, corner.hash())
            cache_path = CACHE_DIR / f"{key}.json"
            if cache_path.exists():
                record = json.loads(cache_path.read_text())
            else:
                record = run_cell(module, params, panel, subs, corner, thresholds=thresholds)
                record["cache_key"] = key
                _atomic_write_json(cache_path, record)
            records.append(record)
        cell_records[module] = records
        module_results[module] = deflate_module(module, records, meta["published"][module])

    results = {"modules": module_results, "cells": cell_records, "grids_meta": meta}
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_report(
        results,
        args.out_dir / "p4_sweep_report.md",
        args.out_dir / "p4_sweep_report.json",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
