# Multi-agent LLM for Corporate Bond Strategy Replication & Bias Detection

Six-agent LLM pipeline: PDF ingestion → strategy extraction → backtesting → bias auditing → extension proposals → research notes.

> Detailed specifications, decision registers, validation records, and licensed data are maintained internally by intention and are not part of this public repository; in-code references to `docs/…` are provenance labels for that internal record.

---

## Setup

Requires Python 3.12 (the version the project venv is built and tested against).

```bash
# 1. Create and activate the virtual environment
python3.12 -m venv .venv
source .venv/bin/activate          # ← run this in EVERY new shell, before any command below

# 2. Install the runtime, test/lint, and PDF-parsing dependencies
pip install -r scripts/requirements_preprocess.txt -r scripts/requirements_parser.txt
```

### Data governance (run once after cloning)

FISD and TRACE Enhanced are **licensed data that must never be committed or pushed**.
Activate the local guard hooks:

```bash
bash scripts/install_hooks.sh
```

This installs pre-commit and pre-push hooks that refuse any FISD/TRACE data
(`data/fisd/**`, raw `data/trace_enhanced_*`, and any `.parquet`/`.csv`/`.csv.gz`
under `data/`). The `data-governance` CI workflow is the un-bypassable backstop.

### Input data (licensed — supply your own)

The build pipeline (§1–§13) reads licensed raw inputs that are **not** in the repo:

- **TRACE Enhanced** → `data/trace_enhanced_repull.csv.gz` (consumed by §1)
- **FISD reference tables** → `data/fisd/*.parquet` (consumed by §4)

Place these under `data/` before running the build steps; without them §1/§4 cannot run and every later stage has nothing to build on. The **test suite needs none of this** — it runs on synthetic fixtures, so a fresh clone can run `python -m pytest tests/unit/` immediately.

---

## What's Built

The data layer produces a **dual-family** monthly panel (`raw` = as-published, `corr` = bias-corrected) that the bias-toggle registry's view layer materialises into two endpoint panels; the gap between them is the measured bias. On top of it sits the **anchor factor layer** — the BBW (2019) four-factor model (MKTB, DRF, LRF, CRF), standalone short-term reversal (`str`) and 6-month momentum (`mom6`), plus the RQ2/RQ3 bias toggles that make them anchors. Each stage hash-logs a report JSON. The first **audited scale-layer module** — the KPP IPCA estimator — sits alongside the anchors (§12), fed by a buildable bond-centric characteristic panel and exercised by an interface-validation shakedown (§13).

### 1. TRACE cleaning → dual families
```bash
python scripts/preprocess_trace.py        # Dick-Nielsen (2009/2014) + dev/holdout split → trace_clean_raw
python scripts/apply_decimal_shift.py      # WRDS-MMN decimal-shift (corrected family)
python scripts/bounce_back_filter.py       # DRR (2026) Table A.2 bounce-back  → trace_clean_corr
```
`raw` carries the as-published junk (no corrections); `corr` adds decimal-shift + bounce-back. Holdout (2022-01..2025-09, 45 months) is written once and **never read during development**.

### 2. Daily layer + distressed filters
```bash
python scripts/build_daily_panel.py --family raw     # VWAP → (cusip, day), raw
python scripts/build_daily_panel.py --family corr
python scripts/apply_distressed_filters.py           # DRR App. A.3 filters 1–4, corr daily only
```

### 3. Risk-free rate
```bash
python scripts/download_rf_rate.py         # FRED TB3MS → rf_rate.parquet (needs internet; no API key)
```

### 4. FISD reference preprocessing
```bash
python scripts/build_fisd_reference.py
```
Turns the licensed FISD reference tables (`data/fisd/*`) into a clean, CUSIP-keyed reference + a monthly **as-of credit-rating** panel (no look-ahead), plus a coverage/quality report. Derives the corporate **universe** filter, the **rating** ladder (1–22), survivorship **exit dates** (maturity / default / defeased), and the value-weight **size proxy** (`offering_amt`, since FISD `amount_outstanding` is ~84% zero). Outputs under `data/development/fisd/`. All rules live in `thresholds.yaml:fisd`.

### 5. Maximal monthly panel (+ FISD merge)
```bash
python scripts/build_monthly_panel.py
```
Outer-joins the two daily families into the dual-family maximal panel and **merges FISD characteristics** as shared, family-agnostic columns: `size` (value-weighting input), `universe_eligible`, `rating`, `investment_grade`, `maturity`, `time_to_maturity`, and a populated `exit_reason`. → `data/development/monthly_panel_maximal.parquet`.

### 6. Signals
```bash
python scripts/build_var_5pct.py           # BBW (2019) 5% VaR        (var_5pct_raw/_corr)
python scripts/build_gamma_illiq.py        # BPW (2011) γ illiquidity (gamma_raw/_corr)
python scripts/build_mom6_signal.py        # Jostova trailing-6m cumulative return (mom6_raw/_corr)
python scripts/build_bond_vol.py           # 24m return-vol / KPP TOTAL_VOL (bond_vol_raw/_corr) — IPCA §13
```
All signals are family-indexed (`_raw`/`_corr`) and written under `data/development/signals/`.

### 7. Endpoint views (bias-toggle registry)
```bash
python scripts/export_endpoint_views.py
```
A `RunConfig` drives `agents/quant/library/views.py` to select a price family and apply the view-level toggles — meas_err (family), stale-price mask, **universe restriction**, **survivorship** (terminal-row drop), signal lag, ex-post trim — materialising `monthly_panel_uncorrected.parquet` (all-OFF / as-published) and `monthly_panel_corrected.parquet` (all-ON / corrected).

### 8. Characteristic-sort engine
Signal-agnostic quintile long-short engine (equal/size-weighted legs, single and independent double sorts, monthly rebalancing, Newey–West HAC inference). Lives at `agents/quant/library/characteristic_sort.py`; configured by the Quant agent; modifications require review.

### 9. Total-return upgrade (accrued interest + coupon)
```bash
python scripts/build_total_return_panel.py   # §2.1 total return = (P + AI + C) growth; 30E/360 accrual
```
The clean-price basis flips CRF's sign and flattens LRF (the credit/liquidity premia live in high-coupon bonds). The accrual engine (`agents/quant/library/accrual.py`) rebuilds the panel on total return — zero-coupon bonds stay byte-identical (the control-group invariance regression). This is the **headline basis** for the anchor factors.

### 10. Anchor factors (BBW four-factor + str + mom6)
```bash
python scripts/build_mktb.py          # market factor (VW par excess return)
python scripts/build_str.py           # short-term reversal (losers − winners)
python scripts/build_mom6.py          # 6-month momentum (deciles, EW, P10 − P1, staggered H=6)
python scripts/build_bbw_factors.py   # DRF / LRF / REV + CRF composite
```
Three independent 5×5 rating×{VaR, γ, REV} bivariate sorts on the audited engine (step 8); par value-weighted, correction-agnostic. On the development sample (2002–2021; licensed data not distributed), the corr-family MKTB is +0.50%/mo (≈ DRR-2023 0.47), with DRF/LRF/CRF correctly signed. Library: `agents/quant/library/{market_factor,bbw_factors}.py`.

### 11. Bias toggles + §8 validation gates
```bash
python scripts/build_str_decomposition.py   # str LIB: month-end vs month-begin (CEIV) decomposition
python scripts/run_mom6_lab_gate.py         # mom6 ex-post vs ex-ante winsorization (look-ahead)
python scripts/run_leadlag_gate.py          # BBW lead/lag error → correlation collapse → restore
python scripts/run_accrual_validation.py    # levels-corrected + differentials-static + Z-invariance
python scripts/run_validation_gates.py      # aggregate §8 verdict
```
Anchors are validated by reproducing published **bias verdicts** (direction + magnitude of each toggle effect), not absolute factor levels (`thresholds.yaml:validation`). Toggles: ex-post/ex-ante winsorization (`winsorize.py`), lead/lag injection (`lead_lag.py`), the str LIB gap (`intramonth_prices.py`).

### 12. IPCA estimator (KPP) — audited Quant library module
```bash
python -m pytest tests/synthetic/test_ipca_battery.py -q   # 33-test certification battery (§9)
python -m mypy --strict agents/quant/library/ipca.py       # + ruff check — clean
```
`agents/quant/library/ipca.py` is the hand-built, **numpy-only** IPCA estimator (Kelly–Palhares–Pruitt): rank-transform + per-month sufficient statistics, the ALS estimator with per-iteration identification, the two wild-bootstrap tests (Γ_α and per-characteristic), recursive out-of-sample estimation, tangency (recursive + in-sample) and spread strategies with costs/turnover and the smoothing-γ cost curve, the fit metrics, and the §10 `context_table` acceptance harness — all **configured, never authored**, at run time via the gold spec `agents/quant/library/configs/kpp_ipca.yaml`. It consumes per-month matrices, enforces a hard `train_end` wall, and raises `ContractViolation` on contract violations (never imputes). Acceptance is the synthetic battery (`tests/synthetic/test_ipca_battery.py`, §9 tests 1–14: subspace/factor recovery, alpha-test size/power, rotation-invariance, idempotency, determinism, …) plus `mypy --strict` + `ruff` clean and a zero-side-effect import.

### 13. IPCA characteristic panel + bond-centric shakedown (Workstream B)
```bash
python scripts/build_bond_vol.py      # 24m return-vol (instrument #7 + the VOL scaler)
python scripts/build_ipca_panel.py    # assemble the 7-instrument feed → ipca_panel_corr.parquet
python scripts/run_ipca_shakedown.py  # in-sample K-sweep + recursive OOS → run-log + context-table
```
The buildable FISD+TRACE instrument subset — short-term reversal, mom6, VaR, γ-illiquidity, rating, time-to-maturity, 24m return-vol (+ constant) — assembled with the **next-return lag** (instruments at m−1, return at m; adjacent months only), complete-case selection, and **VOLScaled010** returns, emitted as the module's per-month `(Z, R)` feed (which passes the module's own `validate_panel` + wall by construction; 209 months, ~1,700 bonds/month). The shakedown runs the estimator end-to-end on this real feed. **Interface-validation only — NON-COMPARABLE to KPP**: a 7-instrument bond-only, VOL-scaled model is structurally different from KPP's 29-instrument DtS model, so every artifact carries that stamp and it **cannot be cited for RQ1/RQ2/RQ3**. The 15 equity/accounting characteristics and the DtS lane remain blocked (CRSP/Compustat access; spread/yield/OAS).

---

## Tests

```bash
python -m pytest tests/unit/ -q                          # 372 passing
python -m pytest tests/synthetic/test_ipca_battery.py -q # IPCA certification battery (33; ~90s)
```
With the venv active these run on the project interpreter (pandas 3.0.3); a different pandas means you forgot `source .venv/bin/activate`. Each signal, factor, and bias toggle has synthetic-fixture-with-known-answer unit tests (e.g. hand-computed γ covariance, accrued interest, leg directions on a 5×5 grid); the IPCA module adds the synthetic certification battery and is `mypy --strict` + `ruff` clean (toolchain installed by the Setup `pip install`). The holdout firewall (`tests/conftest.py`) blocks any read of `data/holdout/`.
