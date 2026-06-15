# Multi-agent LLM for Corporate Bond Strategy Replication & Bias Detection

Six-agent LLM pipeline: PDF ingestion → strategy extraction → backtesting → bias auditing → extension proposals → research notes.

> Detailed specifications, decision registers, validation records, and licensed data are maintained internally by intention and are not part of this public repository; in-code references to `docs/…` are provenance labels for that internal record.

---

## Setup

Requires Python 3.12 (PyBondLab/Numba constraint).

```bash
python -m venv .venv
source .venv/bin/activate
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
Full policy and branch-protection setup: `docs/data_governance.md`.

---

## What's Built

The data layer produces a **dual-family** monthly panel (`raw` = as-published, `corr` = bias-corrected) that the bias-toggle registry's view layer materialises into two endpoint panels; the gap between them is the measured bias. Each stage hash-logs a report JSON.

### 1. TRACE cleaning → dual families
```bash
python scripts/preprocess_trace.py        # Dick-Nielsen (2009/2014) + dev/holdout split → trace_clean_raw
python scripts/apply_decimal_shift.py      # WRDS-MMN decimal-shift (corrected family)
python scripts/bounce_back_filter.py       # DRR (2026) Table A.2 bounce-back  → trace_clean_corr
```
`raw` carries the as-published junk (no corrections); `corr` adds decimal-shift + bounce-back. Holdout (2022-01..2025-09, 45 months) is written once and **never read during development**. See `docs/trace_preprocessing.md`, `docs/bounce_back_filter_spec.md`.

### 2. Daily layer + distressed filters
```bash
python scripts/build_daily_panel.py --family raw     # VWAP → (cusip, day), raw
python scripts/build_daily_panel.py --family corr
python scripts/apply_distressed_filters.py           # DRR App. A.3 filters 1–4, corr daily only
```

### 3. Risk-free rate
```bash
python scripts/download_rf_rate.py         # FRED TB3MS → data/development/rf_rate.parquet
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
python scripts/build_var_5pct.py           # BBW (2019) 5% VaR, family-indexed (var_5pct_raw/_corr)
```

### 7. Endpoint views (bias-toggle registry)
```bash
python scripts/export_endpoint_views.py
```
A `RunConfig` drives `agents/quant/library/views.py` to select a price family and apply the view-level toggles — meas_err (family), stale-price mask, **universe restriction**, **survivorship** (terminal-row drop), signal lag, ex-post trim — materialising `monthly_panel_uncorrected.parquet` (all-OFF / as-published) and `monthly_panel_corrected.parquet` (all-ON / corrected).

### 8. Characteristic-sort engine
Signal-agnostic quintile long-short engine (equal/size-weighted legs, single and independent double sorts, monthly rebalancing, Newey–West HAC inference). Lives at `agents/quant/library/characteristic_sort.py`; configured by the Quant agent, never modified (per ARCHITECTURE.md). See `docs/characteristic_sort_engine_spec.md` and `…_implementation.md`.

---

## Tests

```bash
pytest tests/ -v
```
