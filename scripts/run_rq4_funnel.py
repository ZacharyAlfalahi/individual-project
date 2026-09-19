"""RQ4 Scientist development-funnel orchestrator.

This driver composes the existing funnel stages: build the `str` ScientistCase from the corrected
str AuditReport + the entry rule → eligibility census → generate proposals (3 sources × k seeds × m)
→ G0-G5 experimentalist per source → results (economic + agent-quality funnels, the 13-code refusal
profile, the derived Outcome taxonomy + wrong_signed, the advanced survivors) → G5→G6 rehearsal
bridge (dev pseudo-window; the holdout gate NEVER opens).

  --phase dev  (default): the FREE phase_d generative pair (Gemini 3.1-flash-lite + Mistral-small),
                          NON-reportable — a smoke run of the funnel chain.
  --phase reported:       Phase-F pair.

DEV / HOLDOUT DISCIPLINE: every read is development-window only (data/development/, or the basis
outputs built from it); the real holdout is never opened (only run_oneshot_holdout's rehearsal
branch runs, via dev_pseudo_builder — data/development only).

Design decisions (2026-09-05):
  1. CORRECTED PARENT = the auditor-EXACT all-ON cell (reproduced to 1e-10; asserted in
     corrected_str_parent()). all-ON, not lib_gap-only: RQ3/RQ4/Scientist define "corrected" as
     all-ON and forbid a parent below the corrected lattice point; a lib_gap-only cell is an interior
     point the RQ3 spec bars from standalone interpretation.
  2. DIRECTION is DERIVED at runtime from the realised corrected-parent premium sign (dev-window
     only, guarded). A realised sign that differs from the paper's is NOT a leg convention — both
     use the winners−losers leg; such a divergence is a §8 DATA-CLEANING question (corr/MMN
     cleaning can remove the reversal microstructure premium) and is RECORDED, not resolved
     (ARCHITECTURE.md). A −1 would be a finding, not a config edit.
  3. phase_f generative pair = scientist.model_stack.phase_f (built by build_live_pair via
     phase_d_client._client_from); reportable figures require cost authorization before
     `--phase reported`.
  If G5 advances no survivor, the holdout is not opened — a terminal outcome.

EMBEDDER (--embedder {offline,minilm}, default offline):
  offline — the deterministic SHA-256 bag-of-words stub (label `minilm_offline`); the retrieval
            arm is a NON-reportable stub artefact. Its output bytes are the regression bar.
  minilm  — the REGISTERED production embedder (sources.minilm_embedder, all-MiniLM-L6-v2; label
            `minilm`); the retrieval arm becomes reportable. NEVER falls back to the stub: if
            sentence-transformers is absent, minilm_embedder raises. Device selection is
            sources.py's own default (unparameterised — MPS on Apple Silicon, else CPU); instead of
            forcing a device, the run RECORDS provenance: the output JSON header (`embedder` block)
            carries the embedder name, the model name, and the installed sentence-transformers +
            torch versions. MiniLM inference is deterministic for fixed inputs on a fixed
            device/version pair. A minilm run writes a distinct artifact
            (rq4_funnel_<phase>_minilm.json) so it can never clobber the offline artifact
            (rq4_funnel_<phase>.json).

INPUTS (--basis / --audit-report / --factors-dir / --out):
  --basis {total_return,clean}  the maximal panel comes from scripts/basis_inputs.load_basis_inputs
                                (clean-price signals on both bases; only the return leg changes); the
                                audit report, factors dir and output dir then default to
                                results/consistent_basis/<basis>/{audit/full_run/
                                str_report.json, factors, rq4/funnel}.
  --audit-report PATH           the corrected str AuditReport (default
                                results/auditor/str_corrected/str_report.json).
  --factors-dir DIR             BBW-4 (bbw_factors, mktb) AND the crowding bundles (bbw_factors, mktb,
                                str, mom6) are read from DIR by basename — re-pointed in code;
                                docs/thresholds.yaml is never edited. A missing file fails loud.
  --out DIR                     output directory (default results/scientist/rq4_funnel).
  Precedence: explicit flag > basis default > recorded default. With none of --basis /
  --audit-report / --factors-dir the run reads the recorded inputs and writes the recorded bytes
  (run_e2e_traversal sha256-compares rq4_funnel_reported.json).

LLM POLICY (design decision 2026-09-11) — cache-first, budgeted (prepare_budgeted_clients):
  every generative prompt is rebuilt through the wall (build_context → build_prompt, magnitude-free)
  BEFORE any call and each (model, seed, prompt) key is looked up in runs/rq4_funnel/cache. Each miss
  is priced as a per-call UPPER BOUND, the rule evaluation/codegen/preflight.py applies: input tokens =
  max(8000, ceil(planned prompt UTF-8 bytes / 2)); output tokens = the live client's enforced output
  ceiling (scientist.model_stack.max_output_tokens — the max_output_tokens phase_d_client builds every
  live client with); priced at p1_codegen.budget.prices_usd_per_1m. Outside the bound: a failed
  attempt a vendor bills and the client then retries, and any reasoning tokens a vendor bills beyond
  the output ceiling. If the Anthropic or Gemini estimate exceeds its cap (--max-usd-anthropic /
  --max-usd-gemini, default 10.0 = the per-vendor policy ceiling; a non-finite, negative or > 10.0 cap
  is rejected) the run REFUSES before any call. A zero-miss run constructs no live client (no keys
  needed); otherwise a live client is built only for a model with a miss (a hit-only model stays
  replay-only), only the pre-flighted misses may reach a vendor, each at most once, and an unplanned
  call HALTS the run (UnplannedCallRefused — never a generation_error). The record (hits/misses per
  model, the estimate, metered usage) is embedded in the output JSON of a flagged run; for a flagless
  (recorded-inputs) run it is written to the gitignored runs/rq4_funnel/llm_budget/
  rq4_funnel_<phase>[_minilm].llm_budget.json, so the recorded-inputs artifact stays byte-identical
  and its output folder gains no file.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import math
import sys
import types
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[1]
for _p in (str(_REPO_ROOT), str(_REPO_ROOT / "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from scripts import basis_inputs as BI  # noqa: E402
import run_rq4_spend_sidecar as SPEND  # noqa: E402  (the pre-registered price loader + token bounds)
from agents.auditor.schemas.decomposition import subset_label  # noqa: E402
from agents.scientist.experimentalist.audit_checks import load_reporting_delays  # noqa: E402
from agents.scientist.experimentalist.orchestrator import run_experimentalist  # noqa: E402
from agents.scientist.reporting.funnels import agent_quality_funnel, economic_funnel  # noqa: E402
from agents.scientist.researcher import sources as S  # noqa: E402
from agents.scientist.researcher.cache import ResponseCache  # noqa: E402
from agents.scientist.researcher.context_builder import build_context  # noqa: E402
from agents.scientist.researcher.eligibility import evaluate  # noqa: E402
from agents.scientist.researcher.library import (  # noqa: E402
    available_conditioning_variables,
    load_library,
)
from agents.scientist.researcher.llm_source import LLMResearcherSource, build_prompt  # noqa: E402
from agents.scientist.researcher.phase_d_client import load_model_stack  # noqa: E402
from agents.scientist.schemas.case import DevelopmentWindow, HoldoutStatus  # noqa: E402
from agents.scientist.schemas.outcomes import REFUSAL_CODES  # noqa: E402
from evaluation.codegen.preflight import (  # noqa: E402  (the one per-call input-token bound rule)
    INPUT_BYTES_PER_TOKEN,
    INPUT_TOKENS_UPPER_BOUND,
    PER_VENDOR_USD_CEILING,
    input_token_bound,
)
from shared.evaluation.crowding import load_crowding_factor_bundle  # noqa: E402
from shared.evaluation.thresholds import load_crowding_config  # noqa: E402
from shared.handoff.scientist_case import build_scientist_case, load_entry_rule_params  # noqa: E402
from shared.licensed_inputs import require_licensed_input  # noqa: E402

_STR_AUDIT = _REPO_ROOT / "results" / "auditor" / "str_corrected" / "str_report.json"
_DEFAULT_OUT = _REPO_ROOT / "results" / "scientist" / "rq4_funnel"
_DEV_FACTORS = _REPO_ROOT / "data" / "development" / "factors"
_CACHE_ROOT = _REPO_ROOT / "runs" / "rq4_funnel" / "cache"
_BUDGET_SIDECAR_DIR = _REPO_ROOT / "runs" / "rq4_funnel" / "llm_budget"   # gitignored (runs/)
_GEN_AT = "2026-09-05T00:00:00Z"
_STRATEGY_FAMILY = "CHARACTERISTIC_SORT"


def _load_dotenv(path: Path) -> None:
    """Minimal .env loader (mirrors run_p1/p2/librarian): KEY=VALUE -> os.environ (never overrides).
    The Scientist phase_d/f clients read os.environ directly, so the driver must populate it."""
    import os
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, val = line.split("=", 1)
        os.environ.setdefault(key.strip(), val.strip())


# --------------------------------------------------------------------------
# 0. Input selection (--basis / --audit-report / --factors-dir / --out). Precedence: explicit flag
#    > basis default > the recorded default. No input flag = the recorded run, byte-for-byte.
# --------------------------------------------------------------------------

def repo_relative(path) -> str:
    """Repo-relative posix path for provenance (the absolute path when outside the repo)."""
    p = Path(path).resolve()
    try:
        return p.relative_to(_REPO_ROOT).as_posix()
    except ValueError:
        return str(p)


def is_recorded_inputs(basis, audit, factors_dir) -> bool:
    """True when no input flag is set — the run must reproduce the recorded artifact exactly."""
    return basis is None and audit is None and factors_dir is None


def resolve_basis_paths(basis: str | None, *, audit=None, factors_dir=None, out=None,
                        default_audit, default_out, out_leaf: str,
                        audit_leaf: str | None = None) -> dict:
    """{'basis', 'audit', 'factors_dir', 'out'} for one driver.

    Basis defaults live under results/consistent_basis/<basis>/: audit/full_run
    [/<audit_leaf>] (where the basis full audit writes its reports), factors,
    rq4/<out_leaf>. `factors_dir` None means the recorded data/development/factors + the
    thresholds.yaml crowding bundles, untouched. An unknown basis raises BasisError."""
    if basis is not None:
        BI.check_basis(basis)
    if audit is not None:
        audit_path = Path(audit)
    elif basis is None:
        audit_path = Path(default_audit) if default_audit is not None else None
    else:
        audit_path = BI.basis_dir(basis, "audit", "full_run",
                                  *((audit_leaf,) if audit_leaf else ()))
    if factors_dir is not None:
        factors = Path(factors_dir)
    else:
        factors = BI.factors_dir(basis) if basis is not None else None
    if out is not None:
        out_path = Path(out)
    elif basis is None:
        out_path = Path(default_out) if default_out is not None else None
    else:
        out_path = BI.basis_dir(basis, "rq4", out_leaf)
    return {"basis": basis, "audit": audit_path, "factors_dir": factors, "out": out_path}


def resolve_funnel_paths(basis: str | None = None, *, audit_report=None, factors_dir=None,
                         out=None) -> dict:
    return resolve_basis_paths(basis, audit=audit_report, factors_dir=factors_dir, out=out,
                               default_audit=_STR_AUDIT, default_out=_DEFAULT_OUT,
                               audit_leaf="str_report.json", out_leaf="funnel")


def inputs_record(basis: str | None, audit, factors_dir) -> dict:
    """Provenance block for a FLAGGED run (never added to a recorded-inputs run's output)."""
    return {
        "basis": basis,
        "maximal_loader": ("scripts/basis_inputs.load_basis_inputs" if basis is not None
                           else "agents/auditor/ipca_differential/runner.load_dev_inputs (clean)"),
        "basis_provenance": BI.basis_provenance(basis).to_dict() if basis is not None else None,
        "audit_report": repo_relative(audit),
        "factors_dir": (repo_relative(factors_dir) if factors_dir is not None
                        else "data/development/factors (thresholds.yaml crowding bundles)"),
    }


# --------------------------------------------------------------------------
# 1. Case from the corrected str AuditReport (duck-typed shim — the
#    AuditReport has to_dict() but NO from_dict; build_scientist_case reads only
#    3 fields, so exactly those are reconstructed, as the entry-rule unit test does).
# --------------------------------------------------------------------------

def build_case(audit_json: Path = _STR_AUDIT, *, strategy_id: str = "str",
               case_id: str = "rq4_str", corrected_quant_config_ref: str = "qc_str_corrected",
               corrected_run_ref: str = "str_corrected"):
    """Defaults reproduce the recorded str funnel byte-identically; the SC-SCI-16 capability
    benchmark passes the non-entering anchors' ids + report paths through the same shim
    (corrected_run_ref is per-anchor here — each anchor has its own corrected run dir).
    `corrected_run_ref` is case metadata only (not in the generative context)."""
    params = load_entry_rule_params()
    rd = json.loads(require_licensed_input(
        Path(audit_json),
        "corrected AuditReport (local pipeline output, not shipped with the repository)",
    ).read_text(encoding="utf-8"))
    runnable = tuple(rd["runnable_toggles"])
    doe = {frozenset({t}): float(rd["saturated_bases"]["doe_effects"][subset_label(frozenset({t}))])
           for t in runnable}
    dec = {frozenset({t}): types.SimpleNamespace(
               adjusted_p=float(rd["fdr"]["decisions"][subset_label(frozenset({t}))]["adjusted_p"]))
           for t in runnable}
    report = types.SimpleNamespace(
        core=types.SimpleNamespace(saturated=types.SimpleNamespace(doe=doe),
                                   runnable_toggles=runnable),
        fdr=types.SimpleNamespace(decisions=dec))
    case = build_scientist_case(
        report, strategy_id=strategy_id, case_id=case_id,
        corrected_quant_config_ref=corrected_quant_config_ref, corrected_run_ref=corrected_run_ref,
        audit_report_ref=str(audit_json),
        development_window=DevelopmentWindow("2002-07", "2021-12"),
        holdout_status=HoldoutStatus(accessible=False), theta=params.theta, q=params.q)
    return case, params


# --------------------------------------------------------------------------
# 2. Corrected str parent (DEV approximation — see design decision 1) + derived direction.
# --------------------------------------------------------------------------

def corrected_str_parent(basis: str | None = None):
    """Return (panel, base_rulebook, parent_returns, direction) for the auditor-EXACT all-ON
    (fully-corrected) str lattice cell — the RQ4 corrected parent (all-ON per RQ3/RQ4/Scientist
    invariants; a lib_gap-only cell is an interior point RQ3 bars from standalone reading).

    Built through the auditor's OWN machinery (load_dev_inputs → the all-ON lattice RunConfig →
    view → run_cell), so it reproduces str_report.json's all-ON cell mean
    (asserted below against run_cell). For str (single-leg, holding_period=1) run_strategy reduces
    to run_characteristic_sort(panel, to_rulebook(cfg)), so the returned (panel, base_rulebook) is
    exactly what the auditor cell runs — a drop-in the experimentalist reproduces via
    run_characteristic_sort. DEV ONLY: load_dev_inputs reads data/development; asserted below.

    `basis` None = load_dev_inputs() (the recorded run); a basis name = basis_inputs.
    load_basis_inputs(basis) — the same auditor chain, only the panel's return leg differs."""
    from agents.auditor.checks.cell_runner import _override_construction, run_cell
    from agents.auditor.checks.lattice import build_lattice_configs
    from agents.auditor.checks.preflight import derive_scope
    from agents.auditor.ipca_differential.runner import load_dev_inputs
    from agents.quant.config.quant_config import to_rulebook
    from agents.quant.library.characteristic_sort import run_characteristic_sort
    from agents.quant.library.views import view
    from scripts.run_auditor import (
        default_anchor_facts,
        load_anchor_expost_trim_off,
        load_anchor_meas_err_off_family,
        load_anchor_strategy,
    )

    if basis is None:
        maximal, signals, _registry = load_dev_inputs()
    else:
        maximal, signals, _registry = BI.load_basis_inputs(basis)
    strategy = load_anchor_strategy("str")
    facts = default_anchor_facts("str")
    meas_err_off_family = load_anchor_meas_err_off_family("str")
    expost_trim_off = load_anchor_expost_trim_off("str", None)

    pf = derive_scope(strategy.strategy_label, facts)
    configs = build_lattice_configs(
        pf.runnable_toggles, pf.fixed_states, lib_gap_lags=(0, 1),
        not_applicable_toggles=pf.not_applicable_toggles, meas_err_off_family=meas_err_off_family)
    all_on = frozenset(pf.runnable_toggles)
    rc_all_on = next(rc for on_set, rc in configs if on_set == all_on)

    panel = view(maximal, rc_all_on, signals=signals)
    overridden = _override_construction(strategy, rc_all_on, expost_trim_off)
    base_rulebook = to_rulebook(overridden.leg_calls[0].result)

    res = run_characteristic_sort(panel, base_rulebook)
    parent = pd.Series(res["monthly_returns"]["strategy_ret"].to_numpy(),
                       index=pd.DatetimeIndex(res["monthly_returns"]["date"].to_numpy())).sort_index()
    # Self-verify: the (panel, base_rulebook) the experimentalist will use == the auditor cell.
    cell = run_cell(strategy, rc_all_on, all_on, panel, expost_trim_off=expost_trim_off)
    assert parent.index.equals(cell.returns.sort_index().index) and \
        np.allclose(parent.to_numpy(), cell.returns.sort_index().to_numpy(), rtol=0, atol=0), \
        "corrected parent diverges from the auditor all-ON cell"
    # DEV-WINDOW GUARD: the direction sign is DERIVED from this premium, so it must read dev only —
    # otherwise a holdout run's gate would depend on 2022+ data (inviolable data rule).
    if pd.Timestamp(parent.index.max()) >= pd.Timestamp("2022-01-01"):
        raise RuntimeError(f"corrected-parent premium reads beyond dev (max {parent.index.max()}) "
                           "— direction derivation must be dev-only")
    # DERIVED from the dev premium sign; a paper-sign divergence is recorded, not resolved (§8).
    direction = 1 if float(parent.mean()) > 0 else -1
    return panel, base_rulebook, parent, direction


def bbw4_frame(factors_dir: Path | str | None = None) -> pd.DataFrame:
    """The BBW-4 benchmark frame (date, mktb, drf, crf, lrf) at the corrected family, read from
    `factors_dir` (default data/development/factors — the recorded inputs)."""
    fdir = Path(factors_dir) if factors_dir is not None else _DEV_FACTORS
    bbw = pd.read_parquet(require_licensed_input(
        fdir / "bbw_factors.parquet", "BBW factor panel (licensed input or local pipeline output)"))
    mktb = pd.read_parquet(require_licensed_input(
        fdir / "mktb.parquet", "market-beta factor (licensed input or local pipeline output)"))
    frame = bbw.merge(mktb[["date", "mktb_corr"]], on="date")
    return frame[["date", "mktb_corr", "drf_corr", "crf_corr", "lrf_corr"]].rename(
        columns={"mktb_corr": "mktb", "drf_corr": "drf", "crf_corr": "crf", "lrf_corr": "lrf"})


def crowding_config_for(factors_dir: Path | str | None = None):
    """The registered crowding config. With `factors_dir`, every bundle is re-pointed IN CODE to
    <factors_dir>/<same basename> with the same corrected column — a basis run never mixes its
    factors with the recorded ones, and docs/thresholds.yaml is never edited. Fails loud (no
    fallback) when a bundle file is absent from the directory."""
    cfg = load_crowding_config()
    if factors_dir is None:
        return cfg
    fdir = Path(factors_dir)
    bundles = {name: (str(fdir / Path(path).name), column)
               for name, (path, column) in cfg.bundles.items()}
    missing = sorted({path for path, _column in bundles.values() if not Path(path).exists()})
    if missing:
        raise FileNotFoundError(
            f"crowding bundles absent from factors dir {fdir}: {missing} — build every factor on "
            "this basis first (no fallback to the recorded development factors)")
    return dataclasses.replace(cfg, bundles=bundles)


_MACRO_FILES = {
    "baa_aaa_spread": "baa_aaa_spread.parquet",
    "vix": "vix.parquet",
    "term_spread": "term_spread.parquet",
}


def load_macro_series(lo, hi) -> dict:
    """Regime-conditioning series for month_filter proposals, one per variable, indexed by
    month-end date to match panel['date']. RESTRICTED to [lo, hi] (the dev window) so the
    expanding regime median (execution_verifier.expanding_regime_mask, look-ahead-free) is
    computed on development data ONLY — consistent with the /data/development/ data rule and
    extension_1_config.yaml's committed dev median. Holdout months are never read here."""
    macros = {}
    for var, fname in _MACRO_FILES.items():
        df = pd.read_parquet(require_licensed_input(_REPO_ROOT / "data" / "development" / fname, "macro conditioning series"))
        val_col = next(c for c in df.columns if c != "year_month")
        idx = pd.PeriodIndex(df["year_month"], freq="M").to_timestamp(how="end").normalize()
        s = pd.Series(df[val_col].to_numpy(), index=idx, name=var).sort_index()
        macros[var] = s[(s.index >= lo) & (s.index <= hi)]
    return macros


# --------------------------------------------------------------------------
# 3. Retrieval embedders: the offline deterministic stub (default) vs the registered
#    production MiniLM (sources.minilm_embedder) — selected by --embedder, never silently mixed.
# --------------------------------------------------------------------------

_MINILM_MODEL_NAME = "all-MiniLM-L6-v2"


def deterministic_embedder(dim: int = 64):
    def embed(texts):
        out = np.zeros((len(texts), dim), dtype=float)
        for i, t in enumerate(texts):
            for tok in str(t).lower().split():
                out[i, int(hashlib.sha256(tok.encode()).hexdigest(), 16) % dim] += 1.0
            nrm = np.linalg.norm(out[i])
            if nrm > 0:
                out[i] /= nrm
        return out
    return embed


def minilm_header() -> dict:
    """Provenance header for a minilm run: embedder + model name + installed versions (recorded
    because sources.minilm_embedder's device selection is unparameterised — see module docstring).
    importlib.metadata (not a heavy import) — fails loud if the packages are absent."""
    from importlib.metadata import version
    return {
        "embedder": "minilm",
        "model_name": _MINILM_MODEL_NAME,
        "sentence_transformers_version": version("sentence-transformers"),
        "torch_version": version("torch"),
        "note": ("registered production embedder (agents/scientist/researcher/sources.py::"
                 "minilm_embedder); retrieval arm reportable — NOT the offline stub"),
    }


def resolve_embedder(mode: str, injected=None):
    """--embedder mode -> (embed_fn, retrieval model label, retrieval-arm reportable, header|None).

    offline: the deterministic SHA-256 stub (or `injected`, tests only), label `minilm_offline`,
             NON-reportable retrieval arm, no header — the default, byte-identical output.
    minilm:  sources.minilm_embedder() — RAISES (RuntimeError) if sentence-transformers is absent;
             there is NO fallback to the stub. Label `minilm`, retrieval arm reportable."""
    if mode == "minilm":
        return S.minilm_embedder(), "minilm", True, minilm_header()
    if mode == "offline":
        return injected or deterministic_embedder(), "minilm_offline", False, None
    raise ValueError(f"--embedder must be 'offline' or 'minilm', got {mode!r}")


def output_filename(phase: str, embedder_mode: str) -> str:
    """offline writes rq4_funnel_<phase>.json (the byte-identical regression bar); minilm writes a
    distinct _minilm name that can never collide with the offline artifact
    (results/scientist/rq4_funnel/rq4_funnel_<phase>.json)."""
    if embedder_mode == "offline":
        return f"rq4_funnel_{phase}.json"
    return f"rq4_funnel_{phase}_minilm.json"


def llm_budget_filename(phase: str, embedder_mode: str) -> str:
    """Sidecar name for a recorded-inputs run's LLM budget record, written under the gitignored
    _BUDGET_SIDECAR_DIR (keeps the artifact byte-identical and the recorded folder unchanged)."""
    return output_filename(phase, embedder_mode)[: -len(".json")] + ".llm_budget.json"


# --------------------------------------------------------------------------
# 4. Cache-first, budgeted generative clients (design decision 2026-09-11). Planning rebuilds each
#    prompt through the SAME wall the source uses (build_context → build_prompt), so neither the
#    pre-flight nor any live call ever sees a realised performance statistic.
# --------------------------------------------------------------------------

DEFAULT_MAX_USD = {"anthropic": PER_VENDOR_USD_CEILING, "gemini": PER_VENDOR_USD_CEILING}
MAX_USD_POLICY_CEILING = PER_VENDOR_USD_CEILING   # registered: p1_codegen.budget.per_vendor_usd_ceiling
_LLM_POLICY = (
    "cache-first (design decision 2026-09-11): every prompt rebuilt through the wall and looked up "
    "before any call; each miss priced as a per-call upper bound (the evaluation/codegen/preflight.py "
    f"rule): input max({INPUT_TOKENS_UPPER_BOUND}, ceil(planned prompt UTF-8 bytes/"
    f"{INPUT_BYTES_PER_TOKEN})) tokens + output at the live client's enforced ceiling "
    "scientist.model_stack.max_output_tokens, at p1_codegen.budget.prices_usd_per_1m; outside the "
    "bound: a failed attempt a vendor bills and the client then retries, and any reasoning tokens a "
    "vendor bills beyond the output ceiling; refuse before any call if a vendor estimate exceeds its "
    "cap; a zero-miss run builds no live client; a live client is built only for a model with a "
    "miss; only pre-flighted misses may reach a vendor, each at most once; an unplanned call halts "
    "the run")


def check_usd_cap(value) -> float:
    """A per-vendor spend cap as a float: finite, non-negative and at most the per-vendor policy
    ceiling. A NaN cap would never compare above an estimate, so it is refused, never accepted."""
    cap = float(value)
    if not math.isfinite(cap) or cap < 0:
        raise ValueError(f"spend cap must be a finite, non-negative USD amount; got {value!r}")
    if cap > MAX_USD_POLICY_CEILING:
        raise ValueError(f"spend cap ${cap:.2f} exceeds the "
                         f"${MAX_USD_POLICY_CEILING:.2f} per-vendor policy ceiling")
    return cap


def usd_cap_arg(text: str) -> float:
    """argparse `type` for --max-usd-*: check_usd_cap with a clear CLI error."""
    try:
        return check_usd_cap(text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


class LLMBudgetExceeded(RuntimeError):
    """The pre-flight refused — raised BEFORE any live client is constructed or called."""

    def __init__(self, message: str, record: dict):
        super().__init__(message)
        self.record = record


class UnplannedCallRefused(RuntimeError):
    """A call the pre-flight did not plan (an unplanned key, a repeat, or any call to a replay-only
    model). Never recorded as a generation_error: the generation loops re-raise it so the run
    halts instead of silently dropping an arm."""


def planned_prompt(case, eligible_results, library, m: int) -> str:
    """The exact prompt LLMResearcherSource.candidates sends: the magnitude-free wall context
    (build_context re-asserts the allow-list) serialised by build_prompt."""
    return build_prompt(build_context(case, eligible_results, library), m)


def plan_cache_misses(prompts, model_names, *, k: int, cache_root) -> dict:
    """{model: {hits, misses, miss_keys, miss_input_tokens}} over the unique (model, seed, prompt)
    keys that seeds 0..k-1 will request; miss_input_tokens sums input_token_bound over each missed
    key's own planned prompt. File-existence check only: the cache is never created or written here."""
    root = Path(cache_root)
    plan = {}
    for name in model_names:
        keyed = {ResponseCache.key(p, name, seed): p for p in set(prompts) for seed in range(k)}
        miss = {key for key in keyed if not (root / f"{key}.json").exists()}
        plan[name] = {"hits": len(keyed) - len(miss), "misses": len(miss), "miss_keys": miss,
                      "miss_input_tokens": sum(input_token_bound(keyed[key]) for key in miss)}
    return plan


def estimate_spend(plan: dict, specs: list[dict], prices: dict, *,
                   output_tokens_per_call: int) -> dict:
    """Per-call upper-bound USD for the planned misses, per model and per vendor: each miss's input
    bound (plan_cache_misses) + `output_tokens_per_call`, the live client's enforced output ceiling.
    A model with misses and no pre-registered price is listed (the budget check refuses it), never
    priced at $0."""
    models, by_vendor, unpriced = {}, {}, []
    for spec in specs:
        name, vendor = spec["model_id"], spec["vendor"]
        n, in_tokens = plan[name]["misses"], plan[name]["miss_input_tokens"]
        price = prices.get(name)
        row = {
            "vendor": vendor, "misses": n,
            "input_tokens_upper_bound": in_tokens,
            "output_tokens_per_call": output_tokens_per_call,
            "output_tokens_basis": ("scientist.model_stack.max_output_tokens (the live client's "
                                    "enforced output ceiling)"),
            "price_usd_per_1m": price,
        }
        if price is None:
            row["usd"] = None
            if n:
                unpriced.append(name)
        else:
            usd = (in_tokens * float(price["input"])
                   + n * output_tokens_per_call * float(price["output"])) / 1_000_000
            row["usd"] = usd
            by_vendor[vendor] = by_vendor.get(vendor, 0.0) + usd
        models[name] = row
    return {"models": models, "by_vendor_usd": by_vendor, "unpriced_models_with_misses": unpriced}


def budget_violations(estimate: dict, caps: dict) -> list[str]:
    out = [f"{m}: cache misses but no pre-registered price"
           for m in estimate["unpriced_models_with_misses"]]
    for vendor, usd in sorted(estimate["by_vendor_usd"].items()):
        cap = caps.get(vendor)
        if cap is None and usd > 0:
            out.append(f"{vendor}: estimated ${usd:.4f} with no cap")
        elif cap is not None and usd > cap:
            out.append(f"{vendor}: estimated ${usd:.4f} > cap ${cap:.2f}")
    return out


class BudgetedClient:
    """A ModelClient forwarding ONLY the pre-flighted cache misses (each key at most once) to a
    live client. Anything else — an unplanned key, a second call for a key, or any call when no
    live client was built (zero-miss replay) — raises before reaching a vendor."""

    def __init__(self, name: str, *, allowed_keys=(), live=None, price: dict | None = None):
        self.name = name
        self._allowed = set(allowed_keys)
        self._live = live
        self._price = price
        self.live_calls = 0

    def generate(self, prompt: str, *, seed: int) -> str:
        key = ResponseCache.key(prompt, self.name, seed)
        if self._live is None or key not in self._allowed:
            raise UnplannedCallRefused(f"unplanned live call refused for {self.name!r} seed {seed}: "
                                       "only pre-flighted cache misses may reach a vendor")
        self._allowed.discard(key)
        self.live_calls += 1
        return self._live.generate(prompt, seed=seed)

    def usage(self) -> dict:
        out = {"live_client_constructed": self._live is not None, "live_calls": self.live_calls}
        op = getattr(self._live, "operational_usage", None)
        if callable(op):
            u = dict(op())
            out["operational_usage"] = u
            if (self._price is not None and u.get("prompt_tokens") is not None
                    and u.get("completion_tokens") is not None):
                out["metered_usd"] = (u["prompt_tokens"] * float(self._price["input"])
                                      + u["completion_tokens"] * float(self._price["output"])
                                      ) / 1_000_000
        return out


def llm_usage(clients) -> dict:
    return {c.name: c.usage() for c in clients if isinstance(c, BudgetedClient)}


def prepare_budgeted_clients(prompts, *, stack_key: str, k: int, cache_root, client_factory,
                             max_usd: dict | None = None,
                             thresholds_path: Path | None = None) -> tuple[list, dict]:
    """Pre-flight + wrap -> (BudgetedClients in model_stack order, record). Raises LLMBudgetExceeded
    BEFORE `client_factory` is called; calls it (once) only when some model has a miss, with the
    model ids that have one, and it must build exactly those — a hit-only model gets no live client
    (its BudgetedClient refuses any call), so its key is never read."""
    caps = dict(DEFAULT_MAX_USD)
    caps.update({v: check_usd_cap(c) for v, c in (max_usd or {}).items() if c is not None})
    ms = load_model_stack(thresholds_path) if thresholds_path is not None else load_model_stack()
    try:
        specs = [dict(ms[stack_key]["model_a"]), dict(ms[stack_key]["model_b"])]
    except (KeyError, TypeError) as exc:
        raise RuntimeError(
            f"scientist.model_stack.{stack_key}.{{model_a,model_b}} missing — fail loud") from exc
    names = [s["model_id"] for s in specs]
    prices = SPEND.load_prices(thresholds_path)
    plan = plan_cache_misses(prompts, names, k=k, cache_root=cache_root)
    # the output ceiling every live client is built with (phase_d_client._client_from, same default)
    estimate = estimate_spend(plan, specs, prices,
                              output_tokens_per_call=int(ms.get("max_output_tokens", 2048)))
    record = {
        "policy": _LLM_POLICY, "stack": stack_key, "cache_root": repo_relative(cache_root),
        "k": k, "n_prompts": len(set(prompts)),
        "cache": {n: {"hits": plan[n]["hits"], "misses": plan[n]["misses"]} for n in names},
        "preflight_estimate": estimate, "caps_usd": caps,
    }
    violations = budget_violations(estimate, caps)
    if violations:
        record.update(decision="refused", violations=violations, live_clients_constructed=False)
        raise LLMBudgetExceeded("LLM budget pre-flight refused before any live call: "
                                + "; ".join(violations), record)
    need_live = [n for n in names if plan[n]["misses"]]
    live = {}
    if need_live:
        live = {c.name: c for c in client_factory(need_live)}
        absent = [n for n in need_live if n not in live]
        unrequested = sorted(set(live) - set(need_live))
        if absent or unrequested:
            raise RuntimeError(f"client factory must build live clients for exactly {need_live}: "
                               f"absent {absent}, unrequested {unrequested}")
    clients = [BudgetedClient(n, allowed_keys=plan[n]["miss_keys"], live=live.get(n),
                              price=prices.get(n))
               for n in names]
    record.update(decision="live_calls_for_misses" if need_live else "replay_only",
                  violations=[], live_clients_constructed=bool(need_live))
    return clients, record


def build_live_pair(phase: str, models=None) -> list:
    """Live generative clients for a phase's pair (.env loaded — the clients read os.environ),
    restricted to the model ids in `models` (None = both). Each client is built exactly as
    build_phase_d_clients / build_phase_f_clients build it (phase_d_client._client_from), so a model
    outside `models` never has its client constructed or its key looked up. Reached only through
    prepare_budgeted_clients, for the models whose misses the pre-flight found within budget."""
    _load_dotenv(_REPO_ROOT / ".env")
    from agents.scientist.researcher.phase_d_client import _client_from
    stack_key = "phase_d" if phase == "dev" else "phase_f"
    ms = load_model_stack()
    try:
        specs = [ms[stack_key]["model_a"], ms[stack_key]["model_b"]]
    except (KeyError, TypeError) as exc:
        raise RuntimeError(
            f"scientist.model_stack.{stack_key}.{{model_a,model_b}} missing — fail loud") from exc
    return [_client_from(spec, ms) for spec in specs
            if models is None or spec["model_id"] in models]


def _write_json(out_dir, name: str, obj: dict) -> Path | None:
    if out_dir is None:
        return None
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / name
    path.write_text(json.dumps(obj, indent=2, sort_keys=True, default=str), encoding="utf-8")
    return path


# --------------------------------------------------------------------------
# The funnel.
# --------------------------------------------------------------------------

def run_rq4_funnel(*, phase: str = "dev", llm_clients=None, embedder=None,
                   embedder_mode: str = "offline", k: int = 5, m: int = 6,
                   run_rehearsal: bool = True, out_dir: Path | None = None,
                   cache_root: Path | None = None, basis: str | None = None,
                   audit_report: Path | str | None = None,
                   factors_dir: Path | str | None = None,
                   max_usd: dict | None = None, client_factory=None,
                   supplementary: bool = False) -> dict:
    """Chain the funnel end-to-end on the DEV window. `llm_clients` (list of ModelClient) overrides
    the generative pair (tests inject a stub); default = the phase's pair behind the cache-first
    budget pre-flight (prepare_budgeted_clients; `max_usd` per-vendor caps, `client_factory(models)`
    builds live clients for the model ids with a miss — tests inject a mock). `embedder_mode`
    ('offline' | 'minilm') selects the retrieval embedder (see resolve_embedder); `embedder`
    injects a stand-in in offline mode only (tests). `basis` / `audit_report` / `factors_dir`
    select the inputs (None = recorded)."""
    recorded = is_recorded_inputs(basis, audit_report, factors_dir)
    paths = resolve_funnel_paths(basis, audit_report=audit_report, factors_dir=factors_dir)
    if recorded:
        case, params = build_case()
    else:
        case, params = build_case(paths["audit"],
                                  corrected_run_ref=repo_relative(paths["audit"].parent))
    inputs = None if recorded else inputs_record(basis, paths["audit"], paths["factors_dir"])
    if not case.failed_check_ids:
        result = {"entered": False, "note": "str did not enter — no correction cleared entry"}
        if inputs is not None:                             # a flagged run records its non-entry
            result["inputs"] = inputs
            _write_json(out_dir, output_filename(phase, embedder_mode), result)
        return result

    library = load_library()
    available = available_conditioning_variables()
    elig = [evaluate(mm, strategy_family=_STRATEGY_FAMILY, holding_period=1,
                     templates=library.templates, variable_families=library.variable_families,
                     available_variables=available) for mm in library.mechanisms]

    # --- LLM pre-flight BEFORE any data load or live call (refusal is cheap) ---------------
    cache_root = cache_root or _CACHE_ROOT
    llm_budget = None
    if llm_clients is None:
        llm_clients, llm_budget = prepare_budgeted_clients(
            [planned_prompt(case, elig, library, m)],
            stack_key="phase_d" if phase == "dev" else "phase_f", k=k, cache_root=cache_root,
            max_usd=max_usd,
            client_factory=client_factory or (lambda models: build_live_pair(phase, models)))

    panel, base_rulebook, parent_returns, direction = corrected_str_parent(basis)
    macros = load_macro_series(panel["date"].min(), panel["date"].max())
    bbw4 = bbw4_frame(paths["factors_dir"])
    crowding_cfg = crowding_config_for(paths["factors_dir"])
    crowding_factors = load_crowding_factor_bundle(crowding_cfg)
    reporting_delays = load_reporting_delays()

    embed, retrieval_label, retrieval_reportable, embedder_header = resolve_embedder(
        embedder_mode, embedder)

    # --- 3. generate proposals: random / retrieval / generative-LLM(×clients) --------------
    source_specs = [
        ("random_eligible", S.RandomEligibleSource(), "random_eligible"),
        ("retrieval_only", S.RetrievalOnlySource(embed), retrieval_label),
    ]
    for c in llm_clients:
        source_specs.append((f"llm_{c.name}", LLMResearcherSource(c, cache=ResponseCache(cache_root)),
                             c.name))

    generation: dict[str, list] = {}
    gen_errors: list[dict] = []
    for src_name, source, model_name in source_specs:
        try:
            generation[src_name] = S.run_all_seeds(
                source, case, elig, library, k=k, m=m, model=model_name,
                prompt_version="v1", generated_at=_GEN_AT)
        except UnplannedCallRefused:                   # a stale pre-flight halts the run — never
            raise                                      # a silently dropped arm
        except Exception as exc:                       # a generative pair member can rate-limit out
            gen_errors.append({"source": src_name, "error": f"{type(exc).__name__}: {exc}"[:300]})

    # --- 4. G0-G5 experimentalist per source (seed 0 advances) ------------------------------
    reports: dict[str, dict] = {}
    all_advanced: list[str] = []
    # --supplementary only: the candidate series the report deliberately does not retain, kept
    # aside for the post-hoc diagnostics. Empty (and the collector unused) on every other run.
    experiment_reports: dict[str, object] = {}
    returns_by_source: dict[str, dict] = {}
    for src_name, seed_results in generation.items():
        proposals0 = seed_results[0].proposal_set.proposals if seed_results else ()
        collected: dict | None = {} if supplementary else None
        rep = run_experimentalist(
            case, proposals0, library, panel=panel, base_rulebook=base_rulebook,
            bbw4_factors=bbw4, holding_period=1, signal_lookback=1, available_variables=available,
            macros=macros, m=m, q=params.q, cap=None, direction=direction,
            crowding_config=crowding_cfg, crowding_factors=crowding_factors,
            reporting_delays=reporting_delays, collect_returns=collected)
        if supplementary:
            experiment_reports[src_name] = rep
            returns_by_source[src_name] = collected or {}
        refusal_profile = {code: 0 for code in REFUSAL_CODES}
        refusal_profile.update(Counter(r.refusal_code.value for r in rep.records
                                       if getattr(r, "refusal_code", None) is not None))
        reports[src_name] = {
            "economic_funnel": economic_funnel(rep.records),
            "agent_quality_funnel": agent_quality_funnel(seed_results),
            "outcome_funnel": rep.funnel,                    # derived Outcome taxonomy
            "wrong_signed": rep.wrong_signed,
            "refusal_profile": refusal_profile,
            "advanced": list(rep.advanced),
        }
        if src_name == "retrieval_only" and retrieval_reportable:
            # minilm mode ONLY (offline output stays byte-identical): the stub-artefact caveat does
            # not apply — record that the registered production embedder ran.
            reports[src_name]["reportable"] = True
            reports[src_name]["embedder_note"] = embedder_header["note"]
        all_advanced.extend(rep.advanced)

    # --- 5/6. G5 -> G6 rehearsal bridge (dev pseudo-window; gate NEVER opens) ---------------
    rehearsal = {"ran": False, "advanced_total": len(all_advanced)}
    if run_rehearsal:
        if all_advanced:
            from run_oneshot_holdout import _run_rehearsal    # reuse the canonical rehearsal path
            rc = _run_rehearsal()
            rehearsal.update({"ran": True, "exit_code": rc,
                              "note": "advanced survivors flow to the one-shot holdout runner; the "
                                      "rehearsal exercises the bridge on the dev pseudo-window "
                                      "(per-survivor series extraction is a reportable-run step)"})
        else:
            rehearsal["note"] = "0 advanced survivors → holdout legitimately not opened (terminal)"

    if llm_budget is not None:
        llm_budget["metered_usage"] = llm_usage(llm_clients)
    result = {
        "entered": True, "phase": phase, "reportable": False,
        "direction": direction, "corrected_parent_mean_per_month": float(parent_returns.mean()),
        "failed_check_ids": list(case.failed_check_ids),
        "k": k, "m": m,
        "sources": list(generation.keys()),
        "generation_errors": gen_errors,
        "reports": reports,
        "rehearsal": rehearsal,
    }
    if supplementary:
        # Registered post-hoc diagnostics: DSR (measured in G4), PBO and the regime decomposition.
        # Computed strictly after every gate decision; nothing below gates anything.
        from agents.scientist.reporting.supplementary import funnel_supplementary
        result["supplementary"] = funnel_supplementary(
            experiment_reports, returns_by_source,
            parent_returns=parent_returns,
            spread=(macros or {}).get("baa_aaa_spread"))
        result["supplementary"]["computed_after_the_fact"] = True
    if embedder_header is not None:                          # minilm ONLY — offline byte-identical
        result["embedder"] = embedder_header
    if inputs is not None:                                   # flagged runs ONLY — recorded bytes kept
        result["inputs"] = inputs
        if llm_budget is not None:
            result["llm_budget"] = llm_budget
    if out_dir is not None:
        _write_json(out_dir, output_filename(phase, embedder_mode), result)
        if inputs is None and llm_budget is not None:          # gitignored — never the out dir
            _write_json(_BUDGET_SIDECAR_DIR, llm_budget_filename(phase, embedder_mode), llm_budget)
    return result


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--phase", choices=("dev", "reported"), default="dev")
    ap.add_argument("--embedder", choices=("offline", "minilm"), default="offline",
                    help="offline = deterministic stub (label minilm_offline, retrieval arm "
                         "NON-reportable, the default); minilm = the registered "
                         "production embedder (label minilm, retrieval arm reportable; raises "
                         "if sentence-transformers is absent — no stub fallback)")
    ap.add_argument("--no-rehearsal", action="store_true")
    ap.add_argument("--supplementary", action="store_true",
                    help="additionally compute the registered post-hoc diagnostics "
                         "(deflated Sharpe, PBO, regime decomposition). They are "
                         "computed after every gate decision and gate nothing.")
    ap.add_argument("--basis", choices=BI.BASES, default=None,
                    help="return basis of the maximal panel (default: the recorded clean run via "
                         "load_dev_inputs); sets basis defaults for the flags below")
    ap.add_argument("--audit-report", default=None,
                    help="corrected str AuditReport (default results/auditor/str_corrected/"
                         "str_report.json; with --basis: results/consistent_basis/"
                         "<basis>/audit/full_run/str_report.json)")
    ap.add_argument("--factors-dir", default=None,
                    help="dir with bbw_factors/mktb/str/mom6 parquets (default data/development/"
                         "factors via thresholds.yaml; with --basis: .../<basis>/factors)")
    ap.add_argument("--out", default=None,
                    help="output dir (default results/scientist/rq4_funnel; with --basis: "
                         "results/consistent_basis/<basis>/rq4/funnel)")
    ap.add_argument("--max-usd-anthropic", type=usd_cap_arg, default=DEFAULT_MAX_USD["anthropic"],
                    help="refuse before any call if the estimated Anthropic spend exceeds this "
                         "(finite, 0..10.0 — the per-vendor policy ceiling)")
    ap.add_argument("--max-usd-gemini", type=usd_cap_arg, default=DEFAULT_MAX_USD["gemini"],
                    help="refuse before any call if the estimated Gemini spend exceeds this "
                         "(finite, 0..10.0 — the per-vendor policy ceiling)")
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    paths = resolve_funnel_paths(args.basis, audit_report=args.audit_report,
                                 factors_dir=args.factors_dir, out=args.out)

    if args.phase == "dev":
        print("RQ4 funnel — DEV (free phase_d pair), NON-reportable smoke run. Corrected parent is "
              "the auditor-EXACT all-ON cell; direction derived dev-only.")
    if not is_recorded_inputs(args.basis, args.audit_report, args.factors_dir):
        print(f"inputs: basis={args.basis or 'clean (load_dev_inputs)'} "
              f"audit_report={paths['audit']} "
              f"factors_dir={paths['factors_dir'] or 'data/development/factors'}")
    if args.embedder == "minilm":
        hdr = minilm_header()
        print(f"retrieval embedder: PRODUCTION minilm ({hdr['model_name']}; sentence-transformers "
              f"{hdr['sentence_transformers_version']}, torch {hdr['torch_version']}) — retrieval "
              f"arm reportable; output file ({output_filename(args.phase, args.embedder)})")
    try:
        result = run_rq4_funnel(phase=args.phase, embedder_mode=args.embedder,
                                run_rehearsal=not args.no_rehearsal,
                                out_dir=paths["out"], basis=args.basis,
                                audit_report=args.audit_report, factors_dir=args.factors_dir,
                                max_usd={"anthropic": args.max_usd_anthropic,
                                         "gemini": args.max_usd_gemini},
                                supplementary=args.supplementary)
    except LLMBudgetExceeded as exc:
        print(f"REFUSED (no live call made): {exc}")
        print(json.dumps(exc.record, indent=2, sort_keys=True, default=str))
        return 2
    if not result.get("entered"):
        print(result.get("note"))
        return 0
    print(f"entered=True direction={result['direction']} "
          f"corrected_parent_mean={result['corrected_parent_mean_per_month']:.6f}/mo")
    print(f"sources: {result['sources']}  generation_errors: {len(result['generation_errors'])}")
    for src, rep in result["reports"].items():
        print(f"  [{src}] outcomes={rep['outcome_funnel']} advanced={len(rep['advanced'])} "
              f"wrong_signed={rep['wrong_signed']}")
    print(f"rehearsal: {result['rehearsal']}")
    print(f"results written under: {paths['out']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
