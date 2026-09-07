"""RQ4 Scientist development-funnel orchestrator.

This driver composes the existing funnel stages: build the `str` ScientistCase from the committed corrected AuditReport + the
entry rule → eligibility census → generate proposals (3 sources × k seeds × m) → G0-G5
experimentalist per source → results (economic + agent-quality funnels, the 13-code refusal
profile, the derived Outcome taxonomy + wrong_signed, the advanced survivors) → G5→G6 rehearsal
bridge (dev pseudo-window; the holdout gate NEVER opens).

  --phase dev  (default): the FREE phase_d generative pair (Gemini 3.1-flash-lite + Mistral-small),
                          NON-reportable — the wiring smoke.
  --phase reported:       Phase-F pair.

DEV / HOLDOUT DISCIPLINE: every read is under data/development/; the real holdout is never opened
(only run_oneshot_holdout's rehearsal branch runs, via dev_pseudo_builder — data/development only).

Design decisions (2026-09-05):
  1. CORRECTED PARENT = the auditor-EXACT all-ON cell (mean +0.0019679/mo, reproduced to 1e-10) —
     see corrected_str_parent(). all-ON, not lib_gap-only: RQ3/RQ4/Scientist define "corrected" as
     all-ON and forbid a parent below the corrected lattice point; a lib_gap-only cell is an interior
     point the RQ3 spec bars from standalone interpretation.
  2. DIRECTION = +1, DERIVED at runtime from the realised corrected-parent premium (dev-window only,
     guarded). The +0.197%/mo (ours, momentum) vs DRR's −0.17 (reversal) is NOT a leg convention —
     both use the winners−losers leg; the sign divergence is a documented §8 DATA-CLEANING result
     (corr/MMN cleaning removes the reversal microstructure premium) and is RECORDED, not resolved
     (ARCHITECTURE.md). A −1 would be a finding, not a config edit.
  3. phase_f generative pair WIRED (build_phase_f_clients + scientist.model_stack.phase_f);
     reportable figures still need cost authorization before `--phase reported`.
  If G5 advances ZERO survivors, the holdout is legitimately not opened — a terminal outcome, not a gap.

EMBEDDER (--embedder {offline,minilm}, default offline):
  offline — the deterministic SHA-256 bag-of-words stand-in (label `minilm_offline`); the retrieval
            arm is a NON-reportable stub artefact. Byte-identical to the pre-flag behaviour.
  minilm  — the REGISTERED production embedder (sources.minilm_embedder, all-MiniLM-L6-v2; label
            `minilm`); the retrieval arm becomes reportable. NEVER falls back to the stub: if
            sentence-transformers is absent, minilm_embedder raises. Device selection is
            sources.py's own default (unparameterised — MPS on Apple Silicon, else CPU); rather
            than touching sources.py to force device='cpu', the run RECORDS provenance: the output
            JSON header (`embedder` block) carries the embedder name, the model name, and the
            installed sentence-transformers + torch versions. MiniLM inference is deterministic
            for fixed inputs on a fixed device/version pair. A minilm run writes a distinct
            artifact (rq4_funnel_<phase>_minilm.json) so it can never clobber the recorded
            offline artifacts (rq4_funnel_reported.json / results/rq4_funnel.*).
"""

from __future__ import annotations

import argparse
import hashlib
import json
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

from agents.auditor.schemas.decomposition import subset_label  # noqa: E402
from agents.scientist.experimentalist.audit_checks import load_reporting_delays  # noqa: E402
from agents.scientist.experimentalist.orchestrator import run_experimentalist  # noqa: E402
from agents.scientist.reporting.funnels import agent_quality_funnel, economic_funnel  # noqa: E402
from agents.scientist.researcher import sources as S  # noqa: E402
from agents.scientist.researcher.cache import ResponseCache  # noqa: E402
from agents.scientist.researcher.eligibility import evaluate  # noqa: E402
from agents.scientist.researcher.library import (  # noqa: E402
    available_conditioning_variables,
    load_library,
)
from agents.scientist.researcher.llm_source import LLMResearcherSource  # noqa: E402
from agents.scientist.schemas.case import DevelopmentWindow, HoldoutStatus  # noqa: E402
from agents.scientist.schemas.outcomes import REFUSAL_CODES  # noqa: E402
from shared.evaluation.crowding import load_crowding_factor_bundle  # noqa: E402
from shared.evaluation.thresholds import load_crowding_config  # noqa: E402
from shared.handoff.scientist_case import build_scientist_case, load_entry_rule_params  # noqa: E402
from shared.licensed_inputs import require_licensed_input  # noqa: E402

_STR_AUDIT = _REPO_ROOT / "results" / "auditor" / "str_corrected" / "str_report.json"
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
# 1. Case from the committed corrected str AuditReport (duck-typed shim — the
#    AuditReport has to_dict() but NO from_dict; build_scientist_case reads only
#    3 fields, so we reconstruct exactly those, as the entry-rule unit test does).
# --------------------------------------------------------------------------

def build_case(audit_json: Path = _STR_AUDIT):
    params = load_entry_rule_params()
    rd = json.loads(Path(audit_json).read_text(encoding="utf-8"))
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
        report, strategy_id="str", case_id="rq4_str",
        corrected_quant_config_ref="qc_str_corrected", corrected_run_ref="str_corrected",
        audit_report_ref=str(audit_json),
        development_window=DevelopmentWindow("2002-07", "2021-12"),
        holdout_status=HoldoutStatus(accessible=False), theta=params.theta, q=params.q)
    return case, params


# --------------------------------------------------------------------------
# 2. Corrected str parent (DEV approximation — see design decision 1) + derived direction.
# --------------------------------------------------------------------------

def corrected_str_parent():
    """Return (panel, base_rulebook, parent_returns, direction) for the auditor-EXACT all-ON
    (fully-corrected) str lattice cell — the RQ4 corrected parent (all-ON per RQ3/RQ4/Scientist
    invariants; a lib_gap-only cell is an interior point RQ3 bars from standalone reading).

    Built through the auditor's OWN machinery (load_dev_inputs → the all-ON lattice RunConfig →
    view → run_cell), so it reproduces str_report.json's all-ON cell mean = ∅ + endpoint_gap =
    +0.0019679/mo (verified to 1e-10). For str (single-leg, holding_period=1) run_strategy reduces
    to run_characteristic_sort(panel, to_rulebook(cfg)), so the returned (panel, base_rulebook) is
    exactly what the auditor cell runs — a drop-in the experimentalist reproduces via
    run_characteristic_sort. DEV ONLY: load_dev_inputs reads data/development; asserted below."""
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

    maximal, signals, _registry = load_dev_inputs()
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
    direction = 1 if float(parent.mean()) > 0 else -1           # DERIVED; +1 is the documented §8
    return panel, base_rulebook, parent, direction              # data-cleaning result (record, not resolve)


def bbw4_frame() -> pd.DataFrame:
    """The BBW-4 benchmark frame (date, mktb, drf, crf, lrf) at the corrected family."""
    bbw = pd.read_parquet(require_licensed_input(_REPO_ROOT / "data" / "development" / "factors" / "bbw_factors.parquet", "BBW factor panel"))
    mktb = pd.read_parquet(require_licensed_input(_REPO_ROOT / "data" / "development" / "factors" / "mktb.parquet", "market-beta factor"))
    frame = bbw.merge(mktb[["date", "mktb_corr"]], on="date")
    return frame[["date", "mktb_corr", "drf_corr", "crf_corr", "lrf_corr"]].rename(
        columns={"mktb_corr": "mktb", "drf_corr": "drf", "crf_corr": "crf", "lrf_corr": "lrf"})


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
# 3. Retrieval embedders: the offline deterministic stand-in (default) vs the registered
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

    offline: the deterministic SHA-256 stand-in (or `injected`, tests only), label `minilm_offline`,
             NON-reportable retrieval arm, no header — the pre-flag behaviour, byte-identical.
    minilm:  sources.minilm_embedder() — RAISES (RuntimeError) if sentence-transformers is absent;
             there is NO fallback to the stub. Label `minilm`, retrieval arm reportable."""
    if mode == "minilm":
        return S.minilm_embedder(), "minilm", True, minilm_header()
    if mode == "offline":
        return injected or deterministic_embedder(), "minilm_offline", False, None
    raise ValueError(f"--embedder must be 'offline' or 'minilm', got {mode!r}")


def output_filename(phase: str, embedder_mode: str) -> str:
    """offline keeps the historical name (rq4_funnel_<phase>.json — the byte-identical regression
    bar); minilm writes a distinct _minilm name that can never collide with the recorded artifacts
    (results/scientist/rq4_funnel/rq4_funnel_reported.json, results/rq4_funnel.*)."""
    if embedder_mode == "offline":
        return f"rq4_funnel_{phase}.json"
    return f"rq4_funnel_{phase}_minilm.json"


# --------------------------------------------------------------------------
# The funnel.
# --------------------------------------------------------------------------

def run_rq4_funnel(*, phase: str = "dev", llm_clients=None, embedder=None,
                   embedder_mode: str = "offline", k: int = 5, m: int = 6,
                   run_rehearsal: bool = True, out_dir: Path | None = None,
                   cache_root: Path | None = None) -> dict:
    """Chain the funnel end-to-end on the DEV window. `llm_clients` (list of ModelClient) overrides
    the generative pair (tests inject a stub); default = phase_d free pair. `embedder_mode`
    ('offline' | 'minilm') selects the retrieval embedder (see resolve_embedder); `embedder`
    injects a stand-in in offline mode only (tests)."""
    case, params = build_case()
    if not case.failed_check_ids:
        return {"entered": False, "note": "str did not enter — no correction cleared entry"}

    library = load_library()
    available = available_conditioning_variables()
    elig = [evaluate(mm, strategy_family=_STRATEGY_FAMILY, holding_period=1,
                     templates=library.templates, variable_families=library.variable_families,
                     available_variables=available) for mm in library.mechanisms]

    panel, base_rulebook, parent_returns, direction = corrected_str_parent()
    macros = load_macro_series(panel["date"].min(), panel["date"].max())
    bbw4 = bbw4_frame()
    crowding_cfg = load_crowding_config()
    crowding_factors = load_crowding_factor_bundle(crowding_cfg)
    reporting_delays = load_reporting_delays()

    embed, retrieval_label, retrieval_reportable, embedder_header = resolve_embedder(
        embedder_mode, embedder)
    if llm_clients is None:
        _load_dotenv(_REPO_ROOT / ".env")               # phase_d_client reads os.environ directly
        from agents.scientist.researcher.phase_d_client import (
            build_phase_d_clients,
            build_phase_f_clients,
        )
        llm_clients = list(build_phase_d_clients() if phase == "dev" else build_phase_f_clients())

    cache_root = cache_root or (_REPO_ROOT / "runs" / "rq4_funnel" / "cache")

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
        except Exception as exc:                       # a generative pair member can rate-limit out
            gen_errors.append({"source": src_name, "error": f"{type(exc).__name__}: {exc}"[:300]})

    # --- 4. G0-G5 experimentalist per source (seed 0 advances) ------------------------------
    reports: dict[str, dict] = {}
    all_advanced: list[str] = []
    for src_name, seed_results in generation.items():
        proposals0 = seed_results[0].proposal_set.proposals if seed_results else ()
        rep = run_experimentalist(
            case, proposals0, library, panel=panel, base_rulebook=base_rulebook,
            bbw4_factors=bbw4, holding_period=1, signal_lookback=1, available_variables=available,
            macros=macros, m=m, q=params.q, cap=None, direction=direction,
            crowding_config=crowding_cfg, crowding_factors=crowding_factors,
            reporting_delays=reporting_delays)
        refusal_profile = {code: 0 for code in REFUSAL_CODES}
        refusal_profile.update(Counter(r.refusal_code.value for r in rep.records
                                       if getattr(r, "refusal_code", None) is not None))
        reports[src_name] = {
            "economic_funnel": economic_funnel(rep.records),
            "agent_quality_funnel": agent_quality_funnel(seed_results),
            "outcome_funnel": rep.funnel,                    # derived Outcome taxonomy (supersedes T1/2/3)
            "wrong_signed": rep.wrong_signed,
            "refusal_profile": refusal_profile,
            "advanced": list(rep.advanced),
        }
        if src_name == "retrieval_only" and retrieval_reportable:
            # minilm mode ONLY (offline output stays byte-identical): the stub-artefact caveat no
            # longer applies — record that the registered production embedder ran.
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
    if embedder_header is not None:                          # minilm ONLY — offline byte-identical
        result["embedder"] = embedder_header
    if out_dir is not None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / output_filename(phase, embedder_mode)).write_text(
            json.dumps(result, indent=2, sort_keys=True, default=str), encoding="utf-8")
    return result


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--phase", choices=("dev", "reported"), default="dev")
    ap.add_argument("--embedder", choices=("offline", "minilm"), default="offline",
                    help="offline = deterministic stub (label minilm_offline, retrieval arm "
                         "NON-reportable, historical behaviour); minilm = the registered "
                         "production embedder (label minilm, retrieval arm reportable; raises "
                         "if sentence-transformers is absent — no stub fallback)")
    ap.add_argument("--no-rehearsal", action="store_true")
    ap.add_argument("--out", default=str(_REPO_ROOT / "results" / "scientist" / "rq4_funnel"))
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)

    if args.phase == "dev":
        print("RQ4 funnel — DEV (free phase_d pair), NON-reportable wiring run. Corrected parent is "
              "the auditor-EXACT all-ON cell (+0.0019679/mo); direction derived dev-only (+1).")
    if args.embedder == "minilm":
        hdr = minilm_header()
        print(f"retrieval embedder: PRODUCTION minilm ({hdr['model_name']}; sentence-transformers "
              f"{hdr['sentence_transformers_version']}, torch {hdr['torch_version']}) — retrieval "
              f"arm reportable; output file ({output_filename(args.phase, args.embedder)})")
    result = run_rq4_funnel(phase=args.phase, embedder_mode=args.embedder,
                            run_rehearsal=not args.no_rehearsal,
                            out_dir=Path(args.out))
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
    print(f"results written under: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
