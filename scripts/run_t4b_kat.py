"""
scripts/run_t4b_kat.py — the T4(b) synthetic-paper KAT driver (SYNTH_2026).

Grades one T4(b) synthetic-paper run against the planted key, SEAM BY SEAM. This
is "built, not executed": a scaffold a later AUTHORISED run will use. Importing
this module has ZERO side effects — every seam runs inside a function, guarded by
``if __name__ == "__main__":``.

  # Part B only (deterministic, self-contained — no live emission needed):
  ./.venv/bin/python scripts/run_t4b_kat.py

  # Part A + Part B (needs a Librarian emission for SYNTH_2026):
  ./.venv/bin/python scripts/run_t4b_kat.py --run-dir runs/synth_2026

ALL expectations are loaded from ``evaluation/synthetic/planted_key_synth_2026.yaml``
(no graded number is hard-coded here). The DGP recipe constants (seed 2026,
survivorship magnitude 0.30, the mom6 leg parameters) are the drift-gate's own,
reused verbatim; they are cross-checked against the key's ``dgp``/``seam_3`` blocks
as INFO so a key edit surfaces rather than silently diverging.

SEAMS
-----
  Part A (seams 1-3) — need a Librarian emission (an ``spec_0.json`` + ``trace_0.json``
  under ``--run-dir``). Without ``--run-dir`` they are marked NOT_RUN and only Part B
  is graded.
    seam 1  enumeration = the single "Synthetic Bond Momentum (SBM)" construction.
    seam 2  ``pair_fields`` + ``compare_field`` of a KEY-DERIVED expected spec vs the
            run's shipped fields (20 STATED + 18 UNKNOWN in part2; universe_filter =
            NOT_ASKED; method_summary = rubric). Uses pair_fields/compare_field
            DIRECTLY — never ``score_anchor`` (SYNTH has no ``_ANCHORS`` entry by
            design) and never the RQ1/RQ3 aggregation (``ANCHOR_SET*``/``PAPER_OF``).
    seam 3  ``adapt_spec`` on the KEY-DERIVED spec: zero refusals, variant False,
            combiner single_leg, binding mom6 BOUND, leg kwargs per ``seam_3_compile``.
            Adapting the EMITTED spec is left NOT_GRADED (no typed spec deserialiser
            exists — it needs run_librarian to hand back the typed StrategySpec).

  Part B (seams 4-6) — deterministic, self-contained, no run_dir. Rebuilds the panel
  (SyntheticSpec(seed=2026) -> make_clean_maximal_panel -> inject_survivorship(0.30) ->
  rename score_* -> mom6_*), builds the mom6-bound single-leg strategy, runs
  ``run_scenario(metric="average")`` and ``run_bootstrap(...)``, and grades every value
  in ``seam_4``/``seam_5``/``seam_6`` against the key. Misses are reported VERBATIM,
  never patched (errata discipline: the key is never edited after a live run).

NON-WIRING (enforced by NOT doing it): no ``score_anchor``/``_ANCHORS``; no
``evaluation/harness/aggregation.py`` enrolment; SYNTH_2026 never enters an RQ1/RQ3
denominator (key ``non_wiring``: hypothesis_registry_row FORBIDDEN, rq1_denominators
EXCLUDED).
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import yaml  # noqa: E402

# --- Part B (deterministic) imports — light, proven bit-exact by the drift gate ---
from agents.auditor.checks.bootstrap import run_bootstrap  # noqa: E402
from agents.auditor.checks.preflight import derive_scope  # noqa: E402
from agents.auditor.data.synthetic_panel import (  # noqa: E402
    Scenario,
    SyntheticSpec,
    inject_survivorship,
    make_clean_maximal_panel,
)
from agents.auditor.hashing import hash_series  # noqa: E402
from agents.auditor.schemas.toggle import TOGGLE_IDS, ToggleFacts  # noqa: E402
from agents.auditor.validation.layer_b_fixtures import run_scenario  # noqa: E402
from agents.librarian.adapter.result import (  # noqa: E402
    AdaptResult,
    CombinerInstruction,
    LegCall,
)
from agents.quant.config import Binding, Evidence, Inherited, build_quant_config  # noqa: E402

DEFAULT_KEY_PATH = REPO_ROOT / "evaluation" / "synthetic" / "planted_key_synth_2026.yaml"
DEFAULT_OUT_DIR = REPO_ROOT / "results" / "t4b"

# Status vocabulary for a single graded check.
MATCH = "MATCH"          # reproduced within the key's stated rule
MISS = "MISS"            # drifted — reported verbatim, never patched
NOT_RUN = "NOT_RUN"      # seam not exercised (Part A without --run-dir)
NOT_GRADED = "NOT_GRADED"  # wired but no policy / needs a live run to grade
INFO = "INFO"            # descriptive, never fails a seam
ERROR = "ERROR"          # the seam raised while grading (a build/wiring defect)


# ===========================================================================
# Check + Seam accumulation
# ===========================================================================

@dataclass
class Check:
    name: str
    status: str
    detail: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class SeamResult:
    seam: str
    title: str
    checks: list[Check] = field(default_factory=list)

    # --- append helpers -----------------------------------------------------
    def add(self, name: str, status: str, detail: str = "") -> None:
        self.checks.append(Check(name, status, detail))

    def info(self, name: str, detail: str) -> None:
        self.add(name, INFO, detail)

    def not_graded(self, name: str, detail: str) -> None:
        self.add(name, NOT_GRADED, detail)

    def exact(self, name: str, got: float, exp: float) -> None:
        """Bit-exact float equality (the seam-4/5 match rule: executed == exact)."""
        got_f, exp_f = float(got), float(exp)
        d = abs(got_f - exp_f)
        self.add(name, MATCH if got_f == exp_f else MISS,
                 f"got={got_f!r} exp={exp_f!r} |delta|={d:.3e}")

    def within(self, name: str, got: float, exp: float, tol: float) -> None:
        got_f, exp_f, tol_f = float(got), float(exp), float(tol)
        d = abs(got_f - exp_f)
        self.add(name, MATCH if d <= tol_f else MISS,
                 f"got={got_f!r} exp={exp_f!r} |delta|={d:.3e} tol={tol_f:.1e}")

    def eq(self, name: str, got: Any, exp: Any) -> None:
        self.add(name, MATCH if got == exp else MISS, f"got={got!r} exp={exp!r}")

    def truth(self, name: str, ok: bool, detail: str) -> None:
        self.add(name, MATCH if ok else MISS, detail)

    def ts(self, name: str, got: Any, exp_iso: str) -> None:
        import pandas as pd
        g, e = pd.Timestamp(got), pd.Timestamp(exp_iso)
        self.add(name, MATCH if g == e else MISS, f"got={g.date()} exp={e.date()}")

    # --- verdict ------------------------------------------------------------
    @property
    def verdict(self) -> str:
        statuses = {c.status for c in self.checks}
        if ERROR in statuses:
            return ERROR
        if not self.checks or statuses == {NOT_RUN}:
            return NOT_RUN
        if MISS in statuses:
            return MISS
        return MATCH  # only MATCH / INFO / NOT_GRADED remain

    def to_dict(self) -> dict:
        return {
            "seam": self.seam,
            "title": self.title,
            "verdict": self.verdict,
            "n_match": sum(c.status == MATCH for c in self.checks),
            "n_miss": sum(c.status == MISS for c in self.checks),
            "n_not_graded": sum(c.status == NOT_GRADED for c in self.checks),
            "checks": [c.to_dict() for c in self.checks],
        }


# ===========================================================================
# Key loading
# ===========================================================================

def load_key(path: str | Path = DEFAULT_KEY_PATH) -> dict:
    """Load the planted-answer key. The single source of every graded expectation."""
    with open(path, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


# ===========================================================================
# Part B — the engine (drift-gate code, reused verbatim)
# ===========================================================================

def build_panel_and_signals(spec_seed: int = 2026, surv_mag: float = 0.30):
    """The key's dgp.recipe (order matters): clean maximal panel ->
    inject_survivorship -> rename score_* to mom6_* AFTER injection (inject reads
    score_raw)."""
    spec = SyntheticSpec(seed=spec_seed)
    panel, signals = make_clean_maximal_panel(spec)
    panel, signals = inject_survivorship(panel, signals, surv_mag)
    signals = signals.rename(columns={"score_raw": "mom6_raw", "score_corr": "mom6_corr"})
    return panel, signals


def mom6_strategy(label: str = "synthetic") -> AdaptResult:
    """Mirror synthetic_panel.score_strategy but bind COLUMN 'mom6' (not 'score'),
    single-leg, groups 5, equal, lag 0, long Q5 (group 4) short Q1 (group 0), hold 1,
    no trim — the engine-truth strategy the key's seam_3_compile describes."""
    mom6 = Binding("mom6", "BOUND", Evidence(column="mom6"))
    cfg = build_quant_config(
        label,
        mom6,
        groups=Inherited(5, "DESIGN", Evidence(note="synthetic groups")),
        weighting=Inherited("equal", "DESIGN", Evidence(note="synthetic weighting")),
        signal_lag=Inherited(0, "DESIGN", Evidence(note="synthetic base lag")),
        long_group=Inherited(4, "DESIGN", Evidence(note="top group")),
        short_group=Inherited(0, "DESIGN", Evidence(note="bottom group")),
        holding_period=Inherited(1, "DESIGN", Evidence(note="monthly rebalance")),
        trim=None,
    )
    leg = LegCall(strategy_id=f"{label}::0", kwargs={}, result=cfg)
    return AdaptResult(
        strategy_label=label,
        leg_calls=(leg,),
        combiner=CombinerInstruction(kind="single_leg"),
    )


def run_engine(key: dict):
    """Rebuild the panel + strategy and run the scenario + bootstrap. Returns
    (run, boot). Bootstrap params/seed are read from the key (not hard-coded)."""
    panel, signals = build_panel_and_signals()
    strat = mom6_strategy()
    scenario = Scenario(panel, signals, strat, ("survivorship",), 0.30)
    run = run_scenario(scenario, metric="average")

    bp = ((key.get("seam_6_audit") or {}).get("bootstrap") or {})
    params = bp.get("params") or {}
    boot = run_bootstrap(
        run.lattice.cells,
        run.common,
        TOGGLE_IDS,
        n_replicates=int(params.get("n_replicates", 200)),
        data_driven_block_months=int(params.get("data_driven_block_months", 6)),
        min_effective_blocks=int(params.get("min_effective_blocks", 3)),
        holding_period=int(params.get("holding_period", 1)),
        seed=int(bp.get("seed", 3026)),
    )
    return run, boot, panel


# --- seam graders -----------------------------------------------------------

def grade_seam_4(run, panel, key: dict) -> SeamResult:
    """seam 4 — as-published (all-OFF cell), native support = what the paper prints."""
    s = SeamResult("seam_4", "execution as-published (all-OFF, native)")
    k = key["seam_4_execution_as_published"]
    off = run.lattice.cell_for(frozenset())
    m = off.metrics_native

    s.exact("average", m.average, k["average"])
    s.exact("t_stat", m.t_stat, k["t_stat"])
    s.exact("annualised_average", m.annualised_average, k["annualised_average"])
    s.exact("bumpiness", m.bumpiness, k["bumpiness"])
    s.exact("sharpe", m.sharpe, k["sharpe"])
    s.eq("n_months", int(m.n_months), int(k["n_months"]))
    s.eq("nw_lags_used", int(m.nw_lags_used), int(k["nw_lags_used"]))
    s.add("return_series_sha256",
          MATCH if hash_series(off.returns) == k["return_series_sha256"] else MISS,
          f"got={hash_series(off.returns)} exp={k['return_series_sha256']}")

    # window (descriptive fields the key also pins)
    s.ts("first_date", m.first_date, k["first_date"])
    s.ts("last_date", m.last_date, k["last_date"])

    # membership diagnostics
    nb = off.n_bonds
    s.exact("avg_bonds_per_month", float(nb.mean()), k["avg_bonds_per_month"])
    s.eq("n_bonds_min", int(nb.min()), int(k["n_bonds_min"]))
    s.eq("n_bonds_max", int(nb.max()), int(k["n_bonds_max"]))

    # first three months (key rounds to 10 dp)
    import pandas as pd
    ser = off.returns.sort_index()
    for iso, exp in (k.get("first_three_months") or {}).items():
        got = ser.get(pd.Timestamp(iso))
        if got is None:
            s.add(f"first_three_months[{iso}]", MISS, "month absent from the OFF series")
        else:
            s.truth(f"first_three_months[{iso}]", round(float(got), 10) == float(exp),
                    f"got={round(float(got), 10)!r} exp={exp!r}")

    # printed claims: printed == round(exact, 2dp)
    pc = k.get("printed_claims") or {}
    printed = {
        "mean_pct_per_month": (round(m.average * 100, 2), pc.get("mean_pct_per_month")),
        "t_stat": (round(m.t_stat, 2), pc.get("t_stat")),
        "ann_mean_pct": (round(m.annualised_average * 100, 2), pc.get("ann_mean_pct")),
        "monthly_vol_pct": (round(m.bumpiness * 100, 2), pc.get("monthly_vol_pct")),
        "sharpe": (round(m.sharpe, 2), pc.get("sharpe")),
    }
    for name, (got2, exp2) in printed.items():
        if exp2 is None:
            continue
        s.truth(f"printed[{name}]", got2 == float(exp2), f"round2dp={got2} printed={exp2}")
    return s


def grade_seam_5(run, key: dict) -> SeamResult:
    """seam 5 — corrected endpoint (all-ON cell), native support."""
    s = SeamResult("seam_5", "corrected endpoint (all-ON, native)")
    k = key["seam_5_corrected_endpoint"]
    on = run.lattice.cell_for(frozenset(TOGGLE_IDS))
    m = on.metrics_native

    s.exact("average", m.average, k["average"])
    s.exact("t_stat", m.t_stat, k["t_stat"])
    s.exact("annualised_average", m.annualised_average, k["annualised_average"])
    s.exact("bumpiness", m.bumpiness, k["bumpiness"])
    s.exact("sharpe", m.sharpe, k["sharpe"])
    s.eq("n_months", int(m.n_months), int(k["n_months"]))
    s.add("return_series_sha256",
          MATCH if hash_series(on.returns) == k["return_series_sha256"] else MISS,
          f"got={hash_series(on.returns)} exp={k['return_series_sha256']}")
    s.ts("first_date", m.first_date, k["first_date"])
    s.ts("last_date", m.last_date, k["last_date"])
    return s


def grade_seam_6(run, boot, key: dict) -> SeamResult:
    """seam 6 — the 2^5 lattice audit on common support (metric 'average')."""
    s = SeamResult("seam_6", "audit (2^5 lattice, common support)")
    k = key["seam_6_audit"]

    # preflight
    facts = [ToggleFacts(t, runnable=True) for t in TOGGLE_IDS]
    pf = derive_scope(run.lattice.strategy_label, facts)
    exp_pf = k.get("preflight") or {}
    s.eq("preflight.audit_scope", pf.audit_scope, exp_pf.get("audit_scope"))
    s.eq("preflight.runnable", sorted(pf.runnable_toggles), sorted(exp_pf.get("runnable", [])))
    s.eq("lattice_cells", len(run.lattice.cells), int(k["lattice_cells"]))

    # common support
    common = run.common
    cs = k.get("common_support") or {}
    s.eq("common_support.n_months", len(common), int(cs["n_months"]))
    s.ts("common_support.first", common.min(), cs["first"])
    s.ts("common_support.last", common.max(), cs["last"])

    # two-valued Y (structural: OFF cells share one value, ON cells share another)
    yk = k.get("Y_two_valued") or {}
    tol = float(yk.get("within_half_tolerance", 1e-12))
    y_off = [v for kk, v in run.Y.items() if "survivorship" not in kk]
    y_on = [v for kk, v in run.Y.items() if "survivorship" in kk]
    off_spread = (max(y_off) - min(y_off)) if y_off else float("inf")
    on_spread = (max(y_on) - min(y_on)) if y_on else float("inf")
    s.truth("Y.off_single_valued", off_spread <= tol,
            f"distinct={len(set(y_off))} spread={off_spread:.3e} tol={tol:.1e}")
    s.truth("Y.on_single_valued", on_spread <= tol,
            f"distinct={len(set(y_on))} spread={on_spread:.3e} tol={tol:.1e}")
    if y_off:
        s.within("Y.survivorship_off_cells", max(y_off), yk["survivorship_off_cells"], tol)
    if y_on:
        s.within("Y.survivorship_on_cells", max(y_on), yk["survivorship_on_cells"], tol)

    # first-order DOE effects
    doe = run.doe_first_order
    kd = k.get("doe_first_order") or {}
    surv = kd.get("survivorship") or {}
    s.exact("doe.survivorship", doe["survivorship"], surv["expected"])
    obs_sign = (1 if doe["survivorship"] > 0 else -1 if doe["survivorship"] < 0 else 0)
    s.eq("doe.survivorship.sign", obs_sign, int(surv["sign"]))
    for t in ("meas_err", "stale_price", "lab_trim", "lib_gap"):
        entry = kd.get(t) or {}
        s.within(f"doe.{t}", doe[t], entry.get("expected", 0.0),
                 float(entry.get("tolerance", 1e-12)))

    # bootstrap CIs
    ci = boot.doe_ci()[frozenset({"survivorship"})]
    exp_ci = (k.get("bootstrap") or {}).get("survivorship_ci95") or [None, None]
    s.exact("boot.survivorship_ci_lo", ci[0], exp_ci[0])
    s.exact("boot.survivorship_ci_hi", ci[1], exp_ci[1])
    s.truth("boot.survivorship_ci_excludes_0", not (ci[0] <= 0.0 <= ci[1]),
            f"ci=[{ci[0]:.6e}, {ci[1]:.6e}]")
    all_ci = boot.doe_ci()
    for t in ("meas_err", "stale_price", "lab_trim", "lib_gap"):
        lo, hi = all_ci[frozenset({t})]
        s.truth(f"boot.{t}_ci_includes_0", lo <= 0.0 <= hi, f"ci=[{lo:.3e}, {hi:.3e}]")

    # x5 dominance
    inj = abs(doe["survivorship"])
    max_other = max(abs(float(doe[t])) for t in ("meas_err", "stale_price", "lab_trim", "lib_gap"))
    ratio = inj / max_other if max_other > 0 else float("inf")
    dom = k.get("dominance_x5") or {}
    floor = _parse_ge(dom.get("injected_over_max_other"), default=5.0)
    s.truth("dominance_x5", ratio >= floor, f"ratio={ratio:.6e} floor={floor}")
    return s


def _parse_ge(spec: Any, *, default: float) -> float:
    """Pull the numeric floor out of a '>= 5' style threshold; fall back to default."""
    if isinstance(spec, (int, float)):
        return float(spec)
    if isinstance(spec, str):
        for tok in spec.replace(">=", " ").replace(">", " ").split():
            try:
                return float(tok)
            except ValueError:
                continue
    return float(default)


def grade_part_b(key: dict) -> list[SeamResult]:
    """Seams 4-6 — always runnable, deterministic, proven bit-exact by the drift gate."""
    run, boot, panel = run_engine(key)
    return [
        grade_seam_4(run, panel, key),
        grade_seam_5(run, key),
        grade_seam_6(run, boot, key),
    ]


# ===========================================================================
# Part A — the KEY-DERIVED expected StrategySpec (seams 1-3)
# ===========================================================================

# NOTE (provenance scaffold, load-bearing): every STATED field needs an Inherited
# whose Evidence carries a verbatim quote AND a Locator (page + char span). The key
# records the quote + 1-based PDF page but NOT the char span (that requires locating
# the quote in the frozen canonical text — a live-pipeline op). Char spans are
# therefore scaffolded (0..len(quote)); the *value* and *tag* are faithful to the
# key, and grading reads ONLY the value/tag (compare_field never inspects the
# locator). Replace the scaffold with a real locate against
# evaluation/canonical_texts/synth_2026.frozen.yaml for the live run.

def _key_derived_spec(key: dict):
    """Build the expected StrategySpec from the key's seam_2_extraction fields.

    Constructs the full spec (header + Part1 + Part2 + paper_facts) so ``adapt_spec``
    and ``pair_fields`` can consume it. UNKNOWN fields carry value=None + tag UNKNOWN
    with the key's reason as the note; STATED fields carry the key's value + a
    scaffold locator (see module note)."""
    from agents.librarian.schema import (
        Combiner,
        DescribedSignal,
        Leg,
        MethodSummary,
        PaperFacts,
        Part1,
        Part2,
        SignalRef,
        SpecHeader,
        StrategySpec,
    )
    from agents.quant.config import Locator
    from agents.quant.config.concept_column import load_concept_column_table

    def ev_stated(quote: str, page_1based: int) -> Evidence:
        q = str(quote)
        # 1-based PDF page in the key -> 0-based Locator.page (key page_convention).
        page0 = max(0, int(page_1based) - 1)
        return Evidence(quote=q, locator=Locator(page=page0, char_start=0, char_end=len(q)))

    def stated(value, quote: str, page: int) -> Inherited:
        return Inherited(value, "STATED", ev_stated(quote, page))

    def unknown(entry: dict) -> Inherited:
        note = (entry or {}).get("reason") or (entry or {}).get("note") or "not_stated"
        return Inherited(None, "UNKNOWN", Evidence(note=str(note)))

    def inh(entry: dict) -> Inherited:
        """A common/leg field entry -> Inherited, dispatching on its tag."""
        if (entry or {}).get("tag") == "STATED":
            return stated(entry["value"], entry["quote"], entry["page"])
        return unknown(entry)

    s2 = key["seam_2_extraction"]
    p1 = s2["part1"]
    sb = s2["sort_block"]
    common = s2["common"]
    pf = s2["paper_facts"]

    concept_table = load_concept_column_table()

    # --- header ---
    label_name = key["seam_1_enumeration"]["expected_constructions"][0]["name"]
    label_quote = key["seam_1_enumeration"]["expected_constructions"][0]["quote"]
    label_page = key["seam_1_enumeration"]["expected_constructions"][0].get("page", 1)
    header = SpecHeader(
        paper_id=key["paper_id"],
        strategy_label=stated(label_name, label_quote, label_page),
        registry_version=concept_table.registry_version,   # guarantees the D27(1) handshake
        registry_hash="key-derived-scaffold",
        silence_table_version="key-derived-scaffold",
        canonical_text_hash=str((key.get("provenance") or {}).get("canonical_text_hash", "key-derived")),
    )

    # --- Part 1 ---
    ms = p1.get("method_summary") or {}
    method_rubric = "; ".join(ms.get("rubric", [])) or "method summary (rubric)"
    part1 = Part1(
        formation_structure=inh(p1["formation_structure"]),
        asset_class=inh(p1["asset_class"]),
        # method_summary is graded by rubric (NO_POLICY); the summary text is scaffold.
        method_summary=MethodSummary(
            summary=stated(method_rubric,
                           p1["formation_structure"]["quote"],
                           p1["formation_structure"]["page"])),
    )

    # --- Part 2: sort block (leg + combiner) ---
    sig = sb["sort_signal"]
    sort_signal = SignalRef(
        concept_id=stated(sig["concept_id"], sig["quote"], sig["page"]),
        as_described=DescribedSignal(label=str(sig.get("note") or sig["concept_id"])),
    )
    leg = Leg(
        sort_signal=sort_signal,
        control_axis=None,   # key: control_axis is null (single univariate sort)
        sort_kind=inh(sb["sort_kind"]),
        bucketing_method=unknown(sb["bucketing_method"]),
        n_groups=inh(sb["n_groups"]),
        stripe_aggregation=unknown(sb["stripe_aggregation"]),
        control_missing_policy=unknown(sb["control_missing_policy"]),
        long_leg=inh(sb["long_leg"]),
        signal_transform=unknown(sb["signal_transform"]),
        control_n_groups=unknown(sb["control_n_groups"]),
    )
    combiner = Combiner(kind=unknown(sb["combiner"]))

    # --- Part 2: the 28 common fields (schema order) ---
    common_fields = {name: inh(common[name]) for name in _COMMON_FIELD_NAMES}
    part2 = Part2(combiner=combiner, legs=(leg,), **common_fields)

    # --- paper_facts ---
    def pf_metric() -> Inherited:
        m = pf["claimed_headline_metric"]
        return stated(m["value"], m["quote"], m["page"])

    paper_facts = PaperFacts(
        sample_start=inh(pf["sample_start"]),
        sample_end=inh(pf["sample_end"]),
        # universe_filter: the current driver emits UNKNOWN 'not extracted' (NOT_ASKED);
        # it is a rubric/NOT_ASKED field, so its value is never graded.
        universe_filter=Inherited(
            None, "UNKNOWN",
            Evidence(note="not extracted (NOT_ASKED until the prose field-type lands)")),
        claimed_headline_metric=pf_metric(),
    )

    return StrategySpec(header=header, part1=part1, part2=part2, paper_facts=paper_facts)


# The 28 Part-2 common field names, in schema order (mirrors
# strategy_spec._COMMON_INHERITED_FIELDS; a divergence would raise loudly in Part2).
_COMMON_FIELD_NAMES: tuple[str, ...] = (
    "eligibility_missing_policy", "return_availability_policy", "signal_lag",
    "lag_convention", "min_bonds", "min_bonds_granularity", "tie_break_policy",
    "weighting_scheme", "weighting_base", "weight_timing", "strategy_side",
    "empty_leg_policy", "transaction_cost_convention", "return_label",
    "rebalance_frequency", "holding_period", "overlap_convention", "cohort_weighting",
    "burn_in_policy", "missing_return_policy", "realisation_min_survivors",
    "return_compounding", "significance_convention", "hac_lags", "annualisation",
    "rf_convention", "benchmark_model", "expost_trim",
)

# Fields graded by rubric / not-asked (never a value comparison — mirrors
# field_pairing.RUBRIC_FIELDS + the universe_filter NOT_ASKED driver gap).
_RUBRIC_OR_NOT_ASKED: frozenset[str] = frozenset({"method_summary", "universe_filter"})


def grade_seam_1(run_dir: Path, key: dict) -> SeamResult:
    """seam 1 — enumeration is the single 'Synthetic Bond Momentum (SBM)' construction."""
    s = SeamResult("seam_1", "enumeration (single construction)")
    exp = key["seam_1_enumeration"]["expected_constructions"]
    exp_name = exp[0]["name"]

    spec_0 = run_dir / "spec_0.json"
    if not spec_0.exists():
        s.add("emission_present", MISS, f"missing {spec_0}")
        return s
    try:
        emitted = json.loads(spec_0.read_text(encoding="utf-8"))
        got_name = (((emitted.get("header") or {}).get("strategy_label") or {}).get("value"))
    except Exception as exc:  # noqa: BLE001
        s.add("read_spec_0", ERROR, f"{type(exc).__name__}: {exc}")
        return s

    s.eq("construction_name", got_name, exp_name)
    s.truth("single_construction", not (run_dir / "spec_1.json").exists(),
            "exactly one strategy emitted (no spec_1.json)")
    s.eq("expected_construction_count", len(exp), 1)
    return s


def grade_seam_2(run_dir: Path, key: dict) -> SeamResult:
    """seam 2 — field extraction fidelity via pair_fields + compare_field DIRECTLY
    (never score_anchor / the RQ1 aggregation). Grades the run's shipped fields
    against the KEY-DERIVED expected spec's tags + values."""
    s = SeamResult("seam_2", "extraction (pair_fields + compare_field)")
    from evaluation.harness.compare_policy import Comparability, compare_field
    from evaluation.harness.field_pairing import pair_fields
    from evaluation.harness.run_artefacts import load_run

    spec = _key_derived_spec(key)
    art = load_run(run_dir)
    paired, excluded = pair_fields(spec, art)

    # Self-consistency: the key-derived gold's part2 tag counts must equal the key's
    # declared expected_tag_counts (validates the builder before we grade the run).
    stated_ct = unknown_ct = 0
    for p in paired:
        if not p.dotted_path.startswith("part2."):
            continue
        if p.key.name in _RUBRIC_OR_NOT_ASKED:
            continue
        # control_axis is the injected None comparand (fabrication-detection cell),
        # not a schema silence field — the key's 18 UNKNOWN excludes it. It is still
        # graded per-field below (a run asserting a control axis is a MISS).
        if p.key.name == "control_axis":
            continue
        if p.gold_tag == "STATED":
            stated_ct += 1
        elif p.gold_tag == "UNKNOWN":
            unknown_ct += 1
    exp_counts = key["seam_2_extraction"].get("expected_tag_counts") or {}
    s.eq("gold.part2_stated_count", stated_ct, int(exp_counts.get("part2_stated", stated_ct)))
    s.eq("gold.part2_unknown_count", unknown_ct, int(exp_counts.get("part2_unknown", unknown_ct)))
    s.info("excluded_paths", f"{len(excluded)} path(s): {sorted(excluded)}")

    # Per-field grading (mirrors gold_calibration._classify, without its RQ1 taxonomy).
    for p in paired:
        name = p.key.name
        run = p.run
        if p.excluded_rubric or name in _RUBRIC_OR_NOT_ASKED:
            s.not_graded(f"{p.dotted_path}", "rubric / NOT_ASKED (declared weaker)")
            continue
        if run is None or getattr(run, "not_extracted", False):
            s.add(f"{p.dotted_path}", NOT_GRADED, "run did not ask this field")
            continue
        gold_stated = p.gold_tag == "STATED"
        if run.final_tag != "STATED":
            # run silent: correct iff the key is also silent on this field
            s.truth(f"{p.dotted_path}", not gold_stated,
                    f"run silent ({run.final_tag}); gold_tag={p.gold_tag}")
            continue
        # run shipped a value
        if not gold_stated:
            s.add(f"{p.dotted_path}", MISS,
                  f"run shipped {run.normalised_a!r} but key says UNKNOWN (fabrication)")
            continue
        outcome = compare_field(p.key, p.gold_value, run.normalised_a)
        if outcome.comparability is not Comparability.COMPARABLE:
            s.not_graded(f"{p.dotted_path}", f"not scorable: {outcome.comparability.value} ({outcome.note})")
        else:
            s.truth(f"{p.dotted_path}", bool(outcome.equal),
                    f"gold={outcome.gold_normalised!r} run={outcome.run_normalised!r}")
    return s


def grade_seam_3(run_dir: Path, key: dict) -> SeamResult:
    """seam 3 — compile. adapt_spec on the KEY-DERIVED spec: zero refusals, variant
    False, combiner single_leg, binding mom6 BOUND, leg kwargs per seam_3_compile.
    When the run dir carries an emitted spec, it is ALSO loaded typed
    (spec_loader, 2026-09-04), adapted, and its G2 rulebook byte-compared to the
    key-derived rulebook; a spec-less run dir grades those checks NOT_GRADED."""
    s = SeamResult("seam_3", "compile (adapt_spec)")
    from agents.librarian.adapter import adapt_spec

    spec = _key_derived_spec(key)
    result = adapt_spec(spec)
    k = key["seam_3_compile"]

    s.truth("key_derived.not_refused", not result.refused,
            f"refused={result.refused} refusals={[getattr(r, 'code', r) for r in result.refusals]}")
    s.eq("key_derived.variant", result.variant, bool(k.get("variant", False)))
    combiner_kind = result.combiner.kind if result.combiner is not None else None
    s.eq("key_derived.combiner", combiner_kind, k.get("combiner"))
    s.eq("key_derived.n_legs", len(result.leg_calls), len(spec.part2.legs))

    if result.leg_calls and result.leg_calls[0].result is not None:
        cfg = result.leg_calls[0].result
        lk = k.get("leg") or {}
        binding = lk.get("binding") or {}
        # binding mom6 BOUND
        score = getattr(cfg, "score", None)
        s.eq("leg.binding.column", getattr(score, "value", None), binding.get("column"))
        s.eq("leg.binding.status", getattr(score, "tag", None), binding.get("status"))
        # leg kwargs per seam_3_compile
        for name in ("groups", "weighting", "signal_lag", "long_group",
                     "short_group", "holding_period", "min_bonds"):
            if name not in lk:
                continue
            got = getattr(getattr(cfg, name, None), "value", None)
            s.eq(f"leg.{name}", got, lk[name])
        s.info("leg.trim_rule", repr(getattr(getattr(cfg, "trim_rule", None), "value", None)))
    else:
        s.add("leg.result_present", MISS, "adapt_spec produced no leg config")

    # Emitted-spec adaptation through the typed deserialiser. When the run emitted
    # a spec: load it typed, adapt it, and byte-compare its G2 rulebook against the
    # key-derived rulebook. A run with NO spec (e.g. a Guard-1 emission refusal,
    # the live outcome) stays NOT_GRADED with the reason recorded.
    spec_path = run_dir / "spec_0.json"
    if not spec_path.exists():
        s.not_graded("emitted_spec.adapt",
                     "no spec_0.json in the run dir (emission refused/absent); "
                     "nothing to adapt")
        s.not_graded("rulebook_byte_equal",
                     "no emitted spec -> no emitted rulebook to byte-compare")
        return s

    from agents.librarian.pipeline.spec_loader import spec_from_dict
    from agents.quant.config.quant_config import to_rulebook

    emitted = spec_from_dict(json.loads(spec_path.read_text(encoding="utf-8")))
    em_result = adapt_spec(emitted)
    s.eq("emitted_spec.adapt.refused", em_result.refused, False)
    if (not em_result.refused and em_result.leg_calls
            and em_result.leg_calls[0].result is not None
            and result.leg_calls and result.leg_calls[0].result is not None):
        rb_key = json.dumps(to_rulebook(result.leg_calls[0].result), sort_keys=True)
        rb_em = json.dumps(to_rulebook(em_result.leg_calls[0].result), sort_keys=True)
        s.truth("rulebook_byte_equal", rb_em == rb_key,
                "emitted rulebook == key-derived rulebook (canonical bytes)"
                if rb_em == rb_key else
                f"BYTE MISMATCH: emitted {rb_em[:120]}... vs key {rb_key[:120]}...")
    else:
        s.not_graded("rulebook_byte_equal",
                     "emitted spec refused adapt (or produced no leg config); "
                     "nothing to byte-compare")
    return s


def grade_part_a(run_dir: Path | None, key: dict) -> list[SeamResult]:
    """Seams 1-3. NOT_RUN unless --run-dir carries a Librarian emission. Each seam is
    guarded so a construction/wiring defect surfaces as an ERROR check, never a crash
    that would swallow Part B."""
    titles = {
        "seam_1": "enumeration (single construction)",
        "seam_2": "extraction (pair_fields + compare_field)",
        "seam_3": "compile (adapt_spec)",
    }
    if run_dir is None:
        out = []
        for sid, title in titles.items():
            sr = SeamResult(sid, title)
            sr.add("run_dir", NOT_RUN, "no --run-dir: Part A needs a Librarian emission")
            out.append(sr)
        return out

    graders = (
        ("seam_1", grade_seam_1),
        ("seam_2", grade_seam_2),
        ("seam_3", grade_seam_3),
    )
    out = []
    for sid, fn in graders:
        try:
            out.append(fn(run_dir, key))
        except Exception as exc:  # noqa: BLE001 — a seam defect must not sink the run
            sr = SeamResult(sid, titles[sid])
            sr.add("grader", ERROR, f"{type(exc).__name__}: {exc}")
            out.append(sr)
    return out


# ===========================================================================
# Output
# ===========================================================================

def write_results(out_dir: Path, seams: list[SeamResult], run_log: dict) -> None:
    """Per-seam JSON + a text summary + the run log, under results/t4b/."""
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "run_log.json").write_text(json.dumps(run_log, indent=2, default=str))
    for sr in seams:
        (out_dir / f"{sr.seam}.json").write_text(json.dumps(sr.to_dict(), indent=2, default=str))

    lines = ["T4(b) synthetic-paper KAT (SYNTH_2026) — seam grading", "=" * 60, ""]
    for sr in seams:
        d = sr.to_dict()
        lines.append(f"[{d['verdict']:9s}] {sr.seam}: {sr.title}"
                     f"  (match={d['n_match']} miss={d['n_miss']} not_graded={d['n_not_graded']})")
        for c in sr.checks:
            if c.status in (MISS, ERROR):  # surface every failure verbatim
                lines.append(f"    - {c.status}: {c.name}  {c.detail}")
    lines.append("")
    (out_dir / "summary.txt").write_text("\n".join(lines) + "\n")


def _overall(seams: list[SeamResult]) -> str:
    verdicts = {sr.verdict for sr in seams}
    if ERROR in verdicts:
        return ERROR
    if MISS in verdicts:
        return MISS
    graded = [sr for sr in seams if sr.verdict != NOT_RUN]
    return MATCH if graded else NOT_RUN


# ===========================================================================
# main
# ===========================================================================

def run(run_dir: Path | None, *, key_path: Path, out_dir: Path) -> int:
    """Grade Part B (always) + Part A (iff run_dir). Returns 0 iff no MISS/ERROR
    among the graded seams. Never patches a miss — the key is authoritative."""
    key = load_key(key_path)
    seams = grade_part_a(run_dir, key) + grade_part_b(key)

    run_log = {
        "paper_id": key.get("paper_id"),
        "key_path": str(key_path),
        "key_status": key.get("status"),
        "run_dir": str(run_dir) if run_dir else None,
        "part_a": "graded" if run_dir else "NOT_RUN (no --run-dir)",
        "part_b": "graded (deterministic, self-contained)",
        "seams": {sr.seam: sr.verdict for sr in seams},
    }
    write_results(out_dir, seams, run_log)

    print(json.dumps({
        "paper_id": key.get("paper_id"),
        "out_dir": str(out_dir),
        "seams": {sr.seam: sr.to_dict()["verdict"] for sr in seams},
        "part_b_bit_exact": all(sr.verdict == MATCH for sr in seams if sr.seam in ("seam_4", "seam_5", "seam_6")),
        "overall": _overall(seams),
    }, indent=2, default=str))

    return 0 if _overall(seams) in (MATCH, NOT_RUN) else 1


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="T4(b) synthetic-paper KAT driver (SYNTH_2026)")
    ap.add_argument("--run-dir", type=Path, default=None,
                    help="Librarian emission dir (spec_0.json + trace_0.json) for Part A "
                         "(seams 1-3). Omit to grade only Part B (seams 4-6).")
    ap.add_argument("--key", type=Path, default=DEFAULT_KEY_PATH,
                    help="planted key YAML (default: evaluation/synthetic/planted_key_synth_2026.yaml)")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT_DIR,
                    help="output dir (default: results/t4b)")
    args = ap.parse_args(argv)
    return run(args.run_dir, key_path=args.key, out_dir=args.out)


if __name__ == "__main__":
    raise SystemExit(main())
