# Multi-Agent LLMs for Replicating, Auditing, and Extending Corporate Bond Trading Strategies

Given a published corporate bond trading-strategy paper, the system extracts a machine-readable specification of the strategy, re-implements and backtests it on an audited TRACE/FISD data layer, measures how much of the published performance is an artifact of known data biases, and proposes pre-registered extensions on the bias-corrected strategy. LLMs extract and propose; every number is produced deterministically and verified.

Five-agent pipeline — **Librarian** (paper → spec extraction) → **Quant** (deterministic compilation on an audited backtest library) → **Auditor** (differential bias attribution) → **Scientist** (pre-registered extensions) → **Reporter** (verified research notes).

> Detailed specifications, decision registers, validation records, and licensed data are maintained internally by intention and are not part of this public repository; in-code references to `docs/…` paths and to numbered `§` sections (e.g. `§3.1`, `§13.2`) are provenance labels citing sections of that internal record.

---

## Setup

Requires Python 3.12 (the version the project venv is built and tested against).

```bash
# 1. Create and activate the virtual environment
python3.12 -m venv .venv
source .venv/bin/activate          # ← run this in EVERY new shell, before any command below

# 2. Install the runtime, test/lint, PDF-parsing, and LLM-SDK dependencies
pip install -r scripts/requirements_preprocess.txt \
            -r scripts/requirements_parser.txt \
            -r scripts/requirements_librarian.txt
```

### API keys (live Librarian runs only)

Copy `.env.example` → `.env`. The Phase-D dev pair needs `GEMINI_API_KEY` + `MISTRAL_API_KEY`; the Phase-F reported pair (cost-gated) adds `ANTHROPIC_API_KEY`. The test suite and the data/factor pipeline need no keys.

### Data governance (run once after cloning)

FISD and TRACE Enhanced are **licensed data that must never be committed or pushed**:

```bash
bash scripts/install_hooks.sh
```

This installs pre-commit and pre-push hooks that refuse any FISD/TRACE data (`data/fisd/**`, raw `data/trace_enhanced_*`, and any `.parquet`/`.csv`/`.csv.gz` under `data/`). The `data-governance` CI workflow is the un-bypassable backstop.

### Input data (licensed — supply your own)

The build pipeline reads licensed raw inputs that are **not** in the repo: **TRACE Enhanced** → `data/trace_enhanced_repull.csv.gz` (consumed by §1) and **FISD reference tables** → `data/fisd/*.parquet` (consumed by §4). Without them the build steps cannot run. The **test suite needs none of this** — it runs on synthetic fixtures, so a fresh clone can run `python -m pytest tests/unit/` immediately.

---

## The five agents

| Agent | RQ | Role | LLM in the path? |
|---|---|---|---|
| **Librarian** | RQ1 | Dual-LLM extraction of a frozen paper into a `StrategySpec` + `ExtractionTrace`. A field is STATED only if both models agree on value **and** verbatim quote; UNKNOWN is a valid answer, not an error. | Two models, cross-checked |
| **Quant** | RQ2 | Deterministic compilation: `StrategySpec → adapter → QuantConfig → audited runner` over the hand-built library; closed-enum family routing with typed refusal. | None |
| **Auditor** | RQ3 | Differential bias attribution: each strategy across its `2^k` bias-toggle lattice; the as-published vs corrected gap decomposed with full inference. | None in the analytical core; verifier-gated explainer only |
| **Scientist** | RQ4 | Extensions on a fully corrected parent (extends, never repairs), from three proposal sources; BH-FDR + CPCV qualification, one protected holdout. | Proposal generation only — no generative call ever sees a realised performance statistic |
| **Reporter** | — | Deterministic research-note + claim-ledger emitter; every numeric token asserted against typed pipeline output by `agents/reporter/verify.py`. | None |

### Librarian (RQ1)

```bash
python scripts/run_librarian.py --phase fake       # offline wiring proof — no network, no keys
python scripts/run_librarian.py --phase dev        # Phase-D free pair (Gemini flash-lite + Mistral)
python scripts/run_librarian.py --phase report     # Phase-F reported pair (Claude Sonnet + Gemini) — cost-gated
python scripts/run_librarian_corpus.py             # corpus-wide driver
python scripts/run_librarian_prefetch.py --dry-run # then prefetch: Anthropic Batches API at 50% off; the live run replays at $0
```

Pipeline: live dual-model enumeration (agreement gate; disagreement routes the paper to review) → per-field dual-model extraction (merge + verbatim-quote gate) → stamp + fail-closed validate → emit. Live prompts lead with the paper text so provider prefix caching applies; `--fields` targets single fields (~8–10 calls/model instead of ~42). Responses land in a shared disk replay cache keyed by `(prompt, model_id, seed)`; the Batches prefetch writes to the same cache with byte-exact prompt/key/request/payload parity.

**Evaluation:** `scripts/run_g3_score.py` (the RQ1 scoring chain: per-field accuracy, agreement tables, decomposition; Phase-D runs refuse to render headlines), `scripts/run_rq1_ablation.py`, and the T5 adversarial tier — `run_t5_extraction.py` (Arm A: targeted extraction on perturbed paper variants) + `run_t5_adversarial.py` (grading); Arm B is the must-refuse reject set at the enumeration gate.

### Quant (RQ2)

```bash
python scripts/run_quant.py --anchor mom6   # or --anchor all
```

Compiles each supported anchor's `StrategySpec` through the adapter into a `QuantConfig` and runs it on the audited engine over the development panel (2002–2021); an unsupported family emits a typed `ConfigRefusal`, counted in the RQ2 coverage denominator rather than crashing. Codegen/coverage tiers: `scripts/run_p1_codegen.py`, `scripts/run_p2_codegen.py`, `evaluation/harness/t3_coverage.py`. The hand-built library the Quant configures — and never modifies — is the factor layer below (§8–§13).

### Auditor (RQ3)

```bash
python scripts/run_auditor.py --anchor mom6 --check core-sync-1   # core lattice, dev panel
python scripts/run_full_audit.py --anchor all                     # full stack — refuses until the DSR pre-registration is committed (fail-loud, by design)
python -m pytest tests/unit/test_auditor_*.py -q                  # Layer A/B gates, calibration, inference
```

Runs each strategy through a `2^k` lattice of bias-toggle combinations (`meas_err`, `stale_price`, `survivorship`, `lib_gap`, `lab_trim`) — return series differing ONLY in toggle settings — and decomposes the endpoint gap three ways: **corner marginals**, **DOE effects**, and **Shapley shares** (exactly additive), backed by a synchronised fixed-block bootstrap, HAC-vs-bootstrap routing, BH-FDR, Bayesian and hierarchical layers, and economic significance / deflated Sharpe. `agents/auditor/checks/` is deterministic — zero LLM; the explainer is downstream, every numeric token checked against the typed report, with a deterministic renderer fallback. Validation gates: Layer A algebraic recovery against an independent Shapley oracle, Layer B synthetic bias injection + FPR/MDE calibration. The IPCA-differential extension lives in `agents/auditor/ipca_differential/` (`scripts/run_ipca_differential.py`).

### Scientist (RQ4)

```bash
python scripts/run_oneshot_holdout.py --rehearsal   # full pipeline on the dev pseudo-window; the REAL one-shot is refused here
```

Entry is conditional on RQ3: a strategy enters only where a validated correction survived FDR, exceeded the materiality floor, and reduced the premium. Fixed six proposals per strategy from three sources (random / retrieval / generative) sharing one schema, gate stack, and evaluator; BH-FDR per source, then CPCV-qualified survivors advance to the single protected holdout (2022-01..2025-09 — never read during development). Supporting drivers: `scripts/run_p4_grid_sweep.py`, `scripts/run_r1_promotion_diagnostics.py`, and the executability/reachability censuses.

### Reporter

```bash
python scripts/run_reporter.py   # render / publish / check / scaffold
```

Deterministic research-note and claim-ledger emitter — no LLM anywhere; it never regenerates numbers from prose.

---

## Data & factor layer

The data layer produces a **dual-family** monthly panel (`raw` = as-published, `corr` = bias-corrected); the gap between the two endpoint views is the measured bias. On top sit the **anchor factors** — the BBW (2019) four-factor model (MKTB, DRF, LRF, CRF), short-term reversal, 6-month momentum — and the audited **IPCA estimator**. Each stage hash-logs a report JSON.

### 1. TRACE cleaning → dual families
```bash
python scripts/preprocess_trace.py         # Dick-Nielsen cleaning + dev/holdout split → trace_clean_raw
python scripts/apply_decimal_shift.py      # WRDS-MMN decimal-shift (corrected family)
python scripts/bounce_back_filter.py       # DRR (2026) bounce-back → trace_clean_corr
```
Holdout (2022-01..2025-09) is written once and **never read during development**.

### 2. Daily layer + distressed filters
```bash
python scripts/build_daily_panel.py --family raw     # VWAP → (cusip, day); repeat with --family corr
python scripts/apply_distressed_filters.py           # DRR App. A.3 filters, corr daily only
```

### 3. Risk-free rate
```bash
python scripts/download_rf_rate.py         # FRED TB3MS (needs internet; no API key)
```

### 4. FISD reference preprocessing
```bash
python scripts/build_fisd_reference.py
```
CUSIP-keyed reference + monthly as-of ratings (no look-ahead), the universe filter, rating ladder, survivorship exit dates, and the value-weight size proxy.

### 5. Maximal monthly panel
```bash
python scripts/build_monthly_panel.py
```
Outer-joins the two daily families and merges the FISD characteristics → `data/development/monthly_panel_maximal.parquet`.

### 6. Signals
```bash
python scripts/build_var_5pct.py           # BBW 5% VaR
python scripts/build_gamma_illiq.py        # BPW γ illiquidity
python scripts/build_mom6_signal.py        # trailing-6m cumulative return
python scripts/build_bond_vol.py           # 24m return-vol (IPCA instrument)
```
All family-indexed (`_raw`/`_corr`), under `data/development/signals/`.

### 7. Endpoint views (bias-toggle registry)
```bash
python scripts/export_endpoint_views.py
```
A `RunConfig` drives `agents/quant/library/views.py` to apply the view-level toggles, materialising the all-OFF (as-published) and all-ON (corrected) panels.

### 8. Characteristic-sort engine
Signal-agnostic quintile long-short engine (equal/size-weighted legs, single and independent double sorts, Newey–West inference) at `agents/quant/library/characteristic_sort.py`. Configured by the Quant agent; modifications require review.

### 9. Total-return upgrade
```bash
python scripts/build_total_return_panel.py   # (P + AI + C) growth; 30E/360 accrual
```
The **headline basis** for the anchor factors; zero-coupon bonds stay byte-identical (the control-group invariance regression).

### 10. Anchor factors
```bash
python scripts/build_mktb.py          # market factor (VW par excess return)
python scripts/build_str.py           # short-term reversal
python scripts/build_mom6.py          # 6-month momentum (staggered H=6)
python scripts/build_bbw_factors.py   # DRF / LRF / REV + CRF composite
```
5×5 rating×signal bivariate sorts on the audited engine; on the development sample the corr-family MKTB is +0.50%/mo (≈ DRR-2023's 0.47) with DRF/LRF/CRF correctly signed.

### 11. Bias toggles + validation gates
```bash
python scripts/run_validation_gates.py     # aggregate verdict (plus per-gate drivers: str decomposition, mom6 lab, lead/lag, accrual)
```
Anchors are validated by reproducing published **bias verdicts** (direction + magnitude of each toggle effect), not absolute factor levels.

### 12. IPCA estimator (KPP)
```bash
python -m pytest tests/synthetic/test_ipca_battery.py -q   # 33-test certification battery
```
`agents/quant/library/ipca.py`: hand-built, numpy-only Kelly–Palhares–Pruitt estimator — ALS with per-iteration identification, wild-bootstrap tests, recursive OOS, tangency/spread strategies with costs — configured (never authored) at run time via `agents/quant/library/configs/kpp_ipca.yaml`; `mypy --strict` + `ruff` clean.

### 13. IPCA characteristic panel + shakedown
```bash
python scripts/build_ipca_panel.py    # 7-instrument bond-centric feed
python scripts/run_ipca_shakedown.py  # in-sample K-sweep + recursive OOS
```
**Interface-validation only — NON-COMPARABLE to KPP** (7 bond-only instruments vs KPP's 29-instrument DtS model); every artifact carries that stamp.

---

## Tests

```bash
python -m pytest tests/unit/ -q                          # full unit suite (~225 test files)
python -m pytest tests/synthetic/ -q                     # incl. the IPCA certification battery
```

With the venv active these run on the project interpreter; a different pandas means you forgot `source .venv/bin/activate`. Every signal, factor, bias toggle, and agent has synthetic-fixture-with-known-answer unit tests; the holdout firewall (`tests/conftest.py`) blocks any read of `data/holdout/`.
