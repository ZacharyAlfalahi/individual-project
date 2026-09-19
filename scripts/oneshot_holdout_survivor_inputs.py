"""One-shot holdout stage-2 inputs from a SEEDED panel: the corrected `str` parent, the RQ4 survivors, the BBW-4 benchmark.

The registered one-shot holdout run evaluates the RQ4 survivors on the holdout window; this module derives the
survivors and the benchmark from the holdout inventory OUTSIDE the ``oneshot_holdout`` package
(the one-shot holdout import firewall bars ``agents.auditor`` from ``oneshot_holdout/``; constructing the parent
cell needs the Auditor's own lattice machinery, exactly as the descriptive holdout runner does).
Nothing here reads ``/data/holdout/``: it takes
in-memory seeded frames from the gated builder and is validated on DEVELOPMENT data first
(``scripts/oneshot_holdout_prepare_dev_pins.py``), where every series must reproduce the funnel run exactly.

  parent    — the Auditor's all-ON (fully corrected) str cell on the seeded panel, self-verified against run_cell
              at zero tolerance (the construction ``run_rq4_funnel.corrected_str_parent`` uses on development).
  survivors — the advanced proposals named in the funnel artefact, REGENERATED deterministically (random /
              retrieval / cached LLM outputs; a cache miss refuses), compiled and executed through the
              experimentalist's own compile_g1a / execute_g1b on the seeded panel.
  benchmark — BBW-4 (mktb, drf, crf, lrf; corrected family) from the seeded factor frames.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from shared.licensed_inputs import require_licensed_input

REPO_ROOT = Path(__file__).resolve().parents[1]
HOLDOUT_FLOOR = pd.Timestamp("2022-01-01")


@dataclass(frozen=True)
class SurvivorSpec:
    source: str
    proposal_id: str
    template_ref: str
    mechanism: str | None
    template_mode: str
    conditioning_variable: str | None
    conditioning_lag_months: int | None
    interaction_form: str | None

    def to_dict(self) -> dict:
        return dict(self.__dict__)


def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def advanced_by_source(funnel_artefact: Path) -> dict[str, list[str]]:
    doc = json.loads(require_licensed_input(
        funnel_artefact, "RQ4 funnel artefact (local pipeline output, not shipped with the repository)").read_text())
    return {src: list(rep["advanced"]) for src, rep in doc["reports"].items() if rep["advanced"]}


# --------------------------------------------------------------------------------------------
# parent cell (Auditor all-ON) on an arbitrary seeded panel
# --------------------------------------------------------------------------------------------

def parent_panel_and_rulebook(maximal: pd.DataFrame, signals: pd.DataFrame):
    """``(panel, base_rulebook, parent_series)`` for the all-ON str cell on THIS panel — the chain
    ``run_rq4_funnel.corrected_str_parent`` runs on development, without its dev-only guard, and
    self-verified against the Auditor's ``run_cell`` at zero tolerance."""
    from agents.auditor.checks.cell_runner import _override_construction, run_cell
    from agents.quant.config.quant_config import to_rulebook
    from agents.quant.library.characteristic_sort import run_characteristic_sort
    from agents.quant.library.views import view
    from scripts.run_holdout_oos_descriptive import _anchor_setup

    strategy, expost_trim_off, configs, all_on = _anchor_setup("str")
    rc_all_on = next(rc for on_set, rc in configs if on_set == all_on)
    panel = view(maximal, rc_all_on, signals=signals)
    overridden = _override_construction(strategy, rc_all_on, expost_trim_off)
    base_rulebook = to_rulebook(overridden.leg_calls[0].result)
    res = run_characteristic_sort(panel, base_rulebook)
    parent = pd.Series(res["monthly_returns"]["strategy_ret"].to_numpy(),
                       index=pd.DatetimeIndex(res["monthly_returns"]["date"].to_numpy())).sort_index()
    cell = run_cell(strategy, rc_all_on, all_on, panel, expost_trim_off=expost_trim_off)
    if not (parent.index.equals(cell.returns.sort_index().index)
            and np.array_equal(parent.to_numpy(), cell.returns.sort_index().to_numpy())):
        raise RuntimeError("corrected parent diverges from the auditor all-ON cell")
    return panel, base_rulebook, parent


# --------------------------------------------------------------------------------------------
# survivors: regenerate by id, compile, execute
# --------------------------------------------------------------------------------------------

def regenerate_survivors(audit_report: Path, funnel_artefact: Path, *, cache_root: Path | None = None,
                         embedder_mode: str = "minilm", k: int = 5, m: int = 6):
    """Regenerate the funnel's seed-0 proposal sets and return ``[(SurvivorSpec, proposal, compiled)]`` for
    the advanced ids, in funnel-artefact order. LLM sources replay the cache only (zero misses, no live
    client); random / retrieval are deterministic. Refuses if an advanced id is not regenerated."""
    import run_rq4_funnel as F
    from agents.scientist.experimentalist.compiler import compile_g1a
    from agents.scientist.researcher import sources as S
    from agents.scientist.researcher.cache import ResponseCache

    cache_root = Path(cache_root) if cache_root else F._CACHE_ROOT
    case, _params = F.build_case(Path(audit_report), corrected_run_ref=str(Path(audit_report).parent))
    if not case.failed_check_ids:
        raise RuntimeError("the audit report does not enter RQ4 — no survivors to derive")
    library = F.load_library()
    available = F.available_conditioning_variables()
    elig = [F.evaluate(mm, strategy_family=F._STRATEGY_FAMILY, holding_period=1, templates=library.templates,
                       variable_families=library.variable_families, available_variables=available)
            for mm in library.mechanisms]
    clients, budget = F.prepare_budgeted_clients(
        [F.planned_prompt(case, elig, library, m)], stack_key="phase_f", k=k, cache_root=cache_root,
        max_usd={"anthropic": 0.0, "gemini": 0.0},           # replay only: any miss refuses before a call
        client_factory=lambda models: F.build_live_pair("reported", models))
    if budget["decision"] != "replay_only":
        raise RuntimeError(f"survivor regeneration needs a pure cache replay; got {budget['decision']}")
    embed, label, _rep, _hdr = F.resolve_embedder(embedder_mode, None)
    specs = [("random_eligible", S.RandomEligibleSource(), "random_eligible"),
             ("retrieval_only", S.RetrievalOnlySource(embed), label)]
    for c in clients:
        specs.append((f"llm_{c.name}", F.LLMResearcherSource(c, cache=ResponseCache(cache_root)), c.name))
    wanted = advanced_by_source(funnel_artefact)
    out, found = [], set()
    for src_name, source, model in specs:
        ids = wanted.get(src_name, [])
        if not ids:
            continue
        gen = S.run_all_seeds(source, case, elig, library, k=k, m=m, model=model,
                              prompt_version="v1", generated_at=F._GEN_AT)
        for p in gen[0].proposal_set.proposals:
            if p.proposal_id not in ids:
                continue
            template = library.templates[p.template_ref]
            g1a, compiled = compile_g1a(p, template, case, holding_period=1, available_variables=available)
            if not g1a.passed or compiled is None:
                raise RuntimeError(f"{src_name}/{p.proposal_id}: compile failed on regeneration")
            d = p.model_dump()
            cd = d["config_delta"]
            spec = SurvivorSpec(src_name, p.proposal_id, p.template_ref, d.get("mechanism_ref") or d.get("mechanism_id"),
                                template.get("execution", {}).get("mode"), cd.get("conditioning_variable"),
                                cd.get("conditioning_lag_months"), cd.get("interaction_form"))
            out.append((spec, p, compiled))
            found.add((src_name, p.proposal_id))
    missing = {(s, i) for s, ids in wanted.items() for i in ids} - found
    if missing:
        raise RuntimeError(f"advanced proposals not regenerated: {sorted(missing)}")
    return out, case, library


def execute_survivors(survivors, panel: pd.DataFrame, base_rulebook: dict, macros: dict) -> dict[str, pd.Series]:
    """``{survivor_id: monthly returns}`` via the experimentalist's own execute_g1b; any G1b failure refuses."""
    from agents.scientist.experimentalist.execution_verifier import execute_g1b

    series = {}
    for spec, _p, compiled in survivors:
        macro = None
        if compiled.template_mode == "month_filter" and compiled.panel_transform is not None:
            macro = macros.get(compiled.panel_transform.variable)
            if macro is None:
                raise RuntimeError(f"{spec.source}/{spec.proposal_id}: no series for "
                                   f"{compiled.panel_transform.variable}")
        g1b, res = execute_g1b(compiled, panel, base_rulebook, macro=macro, holding_period=1)
        if not g1b.passed or res is None:
            raise RuntimeError(f"{spec.source}/{spec.proposal_id}: execution failed ({g1b.refusal_code})")
        s = res.candidate_returns.copy()
        s.index = pd.DatetimeIndex(s.index)
        series[f"{spec.source}:{spec.proposal_id}"] = s.sort_index()
    return series


def bbw4_from_frames(bbw: pd.DataFrame, mktb: pd.DataFrame) -> pd.DataFrame:
    frame = bbw.merge(mktb[["date", "mktb_corr"]], on="date")
    return frame[["date", "mktb_corr", "drf_corr", "crf_corr", "lrf_corr"]].rename(
        columns={"mktb_corr": "mktb", "drf_corr": "drf", "crf_corr": "crf", "lrf_corr": "lrf"})


def dev_portion_mean(series: pd.Series) -> float:
    idx = pd.to_datetime(series.index)
    dev = series[idx < HOLDOUT_FLOOR]
    return float(dev.mean()) if len(dev) else float("nan")


def holdout_portion(series: pd.Series) -> tuple[float, int]:
    """(mean, n_months) over the holdout months (>= 2022-01) — the plain level the descriptive holdout runner reports."""
    idx = pd.to_datetime(series.index)
    hold = series[idx >= HOLDOUT_FLOOR]
    return (float(hold.mean()) if len(hold) else float("nan")), int(len(hold))


# --------------------------------------------------------------------------------------------
# the stage-2 inputs
# --------------------------------------------------------------------------------------------

def derive_stage2_inputs(seeded: dict, *, audit_report: Path, funnel_artefact: Path, cache_root: Path | None = None,
                         dev_pins: dict | None = None, atol: float = 1e-9):
    """``(survivors: list[SurvivorInput], benchmarks, record)`` from the seeded frames
    ``{"maximal", "signals", "macros", "factors_bbw", "factors_mktb"}``. With ``dev_pins`` (from
    ``oneshot_holdout_prepare_dev_pins.py``) every series' DEVELOPMENT-portion mean must reproduce its development pin —
    the seeded build's development half must reproduce the development run before stage 2 evaluates any
    holdout month."""
    from agents.scientist.experimentalist.oneshot_holdout.stage2_evaluate import SurvivorInput

    panel, base_rulebook, parent = parent_panel_and_rulebook(seeded["maximal"], seeded["signals"])
    survivors, _case, _library = regenerate_survivors(audit_report, funnel_artefact, cache_root=cache_root)
    # Mirror run_rq4_funnel.load_macro_series exactly: each regime series is restricted to the PANEL's own date
    # span before the expanding median is formed (the median's history starts with the panel, not with FRED's).
    lo, hi = pd.Timestamp(panel["date"].min()), pd.Timestamp(panel["date"].max())
    macros = {k: v[(v.index >= lo) & (v.index <= hi)] for k, v in (seeded.get("macros") or {}).items()}
    series = execute_survivors(survivors, panel, base_rulebook, macros)
    bbw4 = bbw4_from_frames(seeded["factors_bbw"], seeded["factors_mktb"])

    ph_mean, ph_n = holdout_portion(parent)
    record = {"parent_dev_portion_mean": dev_portion_mean(parent),
              "parent_holdout_mean": ph_mean, "parent_holdout_n_months": ph_n,
              "survivors": [s.to_dict() | {"survivor_id": f"{s.source}:{s.proposal_id}",
                                           "dev_portion_mean": dev_portion_mean(series[f"{s.source}:{s.proposal_id}"]),
                                           "holdout_mean": holdout_portion(series[f"{s.source}:{s.proposal_id}"])[0],
                                           "holdout_n_months": holdout_portion(series[f"{s.source}:{s.proposal_id}"])[1],
                                           "n_months_total": int(len(series[f"{s.source}:{s.proposal_id}"]))}
                            for s, _p, _c in survivors],
              "bbw4_dev_portion_means": {c: dev_portion_mean(bbw4.set_index("date")[c]) for c in ("mktb", "drf", "crf", "lrf")},
              "self_verify": None}
    if dev_pins is not None:
        checks = [("parent", record["parent_dev_portion_mean"], dev_pins["parent_dev_mean"])]
        pinned = {s["survivor_id"]: s["dev_mean"] for s in dev_pins["survivors"]}
        checks += [(s["survivor_id"], s["dev_portion_mean"], pinned[s["survivor_id"]]) for s in record["survivors"]]
        checks += [(f"bbw4.{c}", record["bbw4_dev_portion_means"][c], dev_pins["bbw4_dev_means"][c])
                   for c in ("mktb", "drf", "crf", "lrf")]
        bad = [(n, got, want) for n, got, want in checks if not abs(got - want) <= atol]
        record["self_verify"] = {"atol": atol, "checks": [{"name": n, "recomputed": g, "pin": w} for n, g, w in checks],
                                 "passed": not bad}
        if bad:
            raise RuntimeError(f"seeded development portion does not reproduce the development pins: {bad}")
    inputs = [SurvivorInput(survivor_id=f"{s.source}:{s.proposal_id}", returns=series[f"{s.source}:{s.proposal_id}"],
                            parent_returns=parent, is_extension_1=False) for s, _p, _c in survivors]
    return inputs, {"bbw4": bbw4}, record
