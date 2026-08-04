# Gold Spec — KPP (Kelly, Palhares & Pruitt 2023, *Modeling Corporate Bond Returns*)

_Status: **INSTANTIATED 2026-08-03 (schema v1.2).** The FIRST fitted-latent-factor-model gold
(IPCA on corporate bonds) — a separate, non-pooled RQ1 sub-metric (evaluation contract v1.3 / D42),
never averaged into the sort G3 number. Construction is the estimation block + a 29-instrument set
(Table A.I), NOT `legs[]+combiner`; the sort block is a minimal all-UNKNOWN stub the loader injects
in code (never scored)._
_Frozen canonical text: `evaluation/canonical_texts/kpp_2023.frozen.yaml` (42 pages, PyMuPDF 1.28.0,
L1/v3 ladder, `source_sha256 5e5399df…31da7f`). Pages are 1-based PDF pages of `papers/pdf/KPP.pdf`.
Every STATED quote is an exact L1 substring of the frozen text (verified 2026-08-03 via
`locate_quote`; see `locator_backfill_report.md`). Rule library OFF; zero INFERRED; zero DESIGN._
_Dissection: `papers/processed/kelly_2023_ipca.md` (Table A.I transcription + engine reconciliation)._

## Header

```
paper:                   KPP_2023
strategy_label:          IPCA
registry_version:        v1
silence_policy_version:  v1.2
canonical_text_hash:     normalise_sha256 abe11ff63b76df2e46a775a70f1feb397d10fa59984bafd38d312b178b01a79e (config/canonical_text.yaml recipe, PyMuPDF 1.28.0 / L1 / v3 ladder); source_sha256 5e5399dfae90d0e637c38e577b15ec2ae0a242e15e8d3fc3301585803031da7f (frozen kpp_2023)
```

## Part 1

```
formation_structure:  value: estimated_factor_model   tag: STATED
                      quote: "a new conditional factor model for individual corporate bond returns"   page: 2

asset_class:          value: corporate_bonds   tag: STATED
                      quote: "individual corporate bond returns"   page: 2

method_summary:       A conditional latent-factor model (Instrumented PCA, IPCA) for individual corporate
                      bond returns. Bond and firm characteristics instrument time-varying factor loadings; the
                      model is estimated by alternating least squares over K = 1..5 factors (headline K = 5),
                      with the main specification restricting the conditional intercept (Gamma_alpha) to zero.
                      Returns are excess returns scaled by Duration times Spread; characteristics are
                      cross-sectionally rank-standardized to [-0.5, 0.5]. Reported both in-sample and in a
                      recursive out-of-sample scheme (first OOS month 36 months after the sample start), with a
                      wild bootstrap of the zero-alpha null. Locating quotes:
                      (1) "a new conditional factor model for individual corporate bond returns" page 2;
                      (2) "the IPCA model restricted to have no intercepts" page 16;
                      (3) "The first out-of-sample test observation is 36 months after the start of our sample" page 17
```

## Estimation block

```
model_family:                    value: instrumented_pca   tag: STATED
                                 quote: "based on instrumented principal components analysis"   page: 2

estimation_algorithm:            value: alternating_least_squares   tag: STATED
                                 quote: "a fast-converging alternating least squares algorithm"   page: 6

n_factors_tested:                value: {1, 2, 3, 4, 5}   tag: STATED
                                 quote: "models with K = 1, . . . , 5"   page: 23

n_factors_preferred:             value: 5   tag: STATED
                                 quote: "Our main five-factor IPCA model explains 51% of the panel variation"   page: 3

intercept_spec:                  value: restricted   tag: STATED
                                 quote: "the IPCA model restricted to have no intercepts"   page: 16
                                 # Main model restricts Gamma_alpha = 0 (reconciles to kpp_ipca.yaml model.alpha: false).
                                 # The appendix estimates the unrestricted (alpha != 0) alternative and bootstraps
                                 # H0: alpha = 0 -- that is the inference_method, not a second intercept spec.

return_variable:                 value: bond_excess_return_dts_scaled   tag: STATED
                                 quote: "returns divided by Duration times Spread (DtS)"   page: 38

characteristic_preprocessing:    value: rank_standardize_to_pm_half   tag: STATED
                                 quote: "we cross-sectionally rank, demean, and scale"   page: 13

managed_portfolio_construction:  value: characteristic_managed_portfolios   tag: STATED
                                 quote: "characteristic-managed portfolios"   page: 6

estimation_mode:                 value: both   tag: STATED
                                 quote: "The first out-of-sample test observation is 36 months after the start of our sample"   page: 17
                                 # Reported both in-sample (Tables I-II) and recursive out-of-sample (Table III).

oos_split:                       value: 36   tag: STATED
                                 quote: "The first out-of-sample test observation is 36 months after the start of our sample"   page: 17

inference_method:                value: wild_bootstrap   tag: STATED
                                 quote: "We use a wild-bootstrap following KPS using a t-distribution with seven degrees of freedom and 1,000 simulations"   page: 38
```

## Instruments

_The 29 Table A.I characteristics. Each: `concept_id` (instrument-registry id; evidence = the
`as_described` quote), `source_class` (coarse data-domain of the paper-described construction; evidence
= the appendix definition), and `as_described` (the paper's own words). Per-instrument `transform` and
`lag` are uniformly UNKNOWN and injected by the loader (the uniform rank-standardization is captured in
`estimation.characteristic_preprocessing`; no per-instrument lag is stated). Two ids reuse the sort
registry verbatim (`past_6m_cumulative_return`, `credit_rating`); VaR is the new `bond_var_36m`._

```
concept_id: bond_age
source_class: bond   quote: "Bond age is measured in years"   page: 36
as_described: {label: "Bond age", quote: "Bond age is measured in years", page: 36}
```
```
concept_id: coupon
source_class: bond   quote: "Coupon, face value, and are attributes of the bond issue"   page: 36
as_described: {label: "Coupon", quote: "Coupon, face value, and are attributes of the bond issue", page: 36}
```
```
concept_id: face_value
source_class: bond   quote: "Coupon, face value, and are attributes of the bond issue"   page: 36
as_described: {label: "Face value", quote: "Coupon, face value, and are attributes of the bond issue", page: 36}
```
```
concept_id: book_to_price
source_class: equity   quote: "Book-to-price is the sum of shareholders"   page: 36
as_described: {label: "Book-to-price", quote: "Book-to-price is the sum of shareholders", page: 36}
```
```
concept_id: debt_to_ebitda
source_class: accounting   quote: "Debt-to-EBITDA uses total debt"   page: 36
as_described: {label: "Debt-to-EBITDA", quote: "Debt-to-EBITDA uses total debt", page: 36}
```
```
concept_id: duration
source_class: bond   quote: "is the derivative of the bond value to the credit spread"   page: 36
as_described: {label: "Duration", quote: "is the derivative of the bond value to the credit spread", page: 36}
```
```
concept_id: equity_momentum_6m
source_class: equity   quote: "Mom. 6m equity is 6-2 momentum of the firm"   page: 36
as_described: {label: "Mom. 6m equity", quote: "Mom. 6m equity is 6-2 momentum of the firm", page: 36}
```
```
concept_id: earnings_to_price
source_class: equity   quote: "Earnings-to-price"   page: 13
as_described: {label: "Earnings-to-price", quote: "Earnings-to-price", page: 13}
```
```
concept_id: equity_market_cap
source_class: equity   quote: "Equity market cap."   page: 13
as_described: {label: "Equity market cap.", quote: "Equity market cap.", page: 13}
```
```
concept_id: equity_volatility
source_class: equity   quote: "Equity volatility"   page: 13
as_described: {label: "Equity volatility", quote: "Equity volatility", page: 13}
```
```
concept_id: firm_total_debt
source_class: accounting   quote: "Firm total debt"   page: 13
as_described: {label: "Firm total debt", quote: "Firm total debt", page: 13}
```
```
concept_id: past_6m_cumulative_return
source_class: bond   quote: "mom. 6m is 6-2 momentum of bond returns"   page: 36
as_described: {label: "Mom. 6m", quote: "mom. 6m is 6-2 momentum of bond returns", page: 36}
```
```
concept_id: industry_momentum_6m
source_class: bond   quote: "mom. 6m industry is industry-adjusted 6-2 momentum of bond returns"   page: 36
as_described: {label: "Mom. 6m industry", quote: "mom. 6m industry is industry-adjusted 6-2 momentum of bond returns", page: 36}
```
```
concept_id: momentum_6m_x_rating
source_class: bond   quote: "mom. 6m × ratings is bond return momentum"   page: 36
as_described: {label: "Mom. 6m x ratings", quote: "mom. 6m × ratings is bond return momentum", page: 36}
```
```
concept_id: book_leverage
source_class: accounting   quote: "Book leverage shareholder"   page: 36
as_described: {label: "Book leverage", quote: "Book leverage shareholder", page: 36}
```
```
concept_id: market_leverage
source_class: accounting   quote: "Market leverage is"   page: 36
as_described: {label: "Market leverage", quote: "Market leverage is", page: 36}
```
```
concept_id: turnover_volatility
source_class: accounting   quote: "Turnover volatility is the quarterly standard deviation of sales divided by assets"   page: 37
as_described: {label: "Turnover volatility", quote: "Turnover volatility is the quarterly standard deviation of sales divided by assets", page: 37}
```
```
concept_id: spread
source_class: bond   quote: "Spread is the option-adjusted spread of the bond"   page: 36
as_described: {label: "Spread", quote: "Spread is the option-adjusted spread of the bond", page: 36}
```
```
concept_id: operating_leverage
source_class: accounting   quote: "Operating leverage is sales minus EBITDA, divided by EBITDA"   page: 37
as_described: {label: "Operating leverage", quote: "Operating leverage is sales minus EBITDA, divided by EBITDA", page: 37}
```
```
concept_id: profitability
source_class: accounting   quote: "Profitability is sales minus cost-of"   page: 37
as_described: {label: "Profitability", quote: "Profitability is sales minus cost-of", page: 37}
```
```
concept_id: profitability_change
source_class: accounting   quote: "Profitability change is the five-year change in profitability"   page: 37
as_described: {label: "Profitability change", quote: "Profitability change is the five-year change in profitability", page: 37}
```
```
concept_id: credit_rating
source_class: bond   quote: "Rating is a number from 1 to 22, where 1 is S&P AAA and 22 is already in default"   page: 36
as_described: {label: "Rating", quote: "Rating is a number from 1 to 22, where 1 is S&P AAA and 22 is already in default", page: 36}
```
```
concept_id: distance_to_default
source_class: accounting   quote: "is defined by Shumway (2001)"   page: 37
as_described: {label: "Distance-to-default", quote: "is defined by Shumway (2001)", page: 37}
```
```
concept_id: bond_skewness
source_class: bond   quote: "Bond skewness is bond return"   page: 37
as_described: {label: "Bond skewness", quote: "Bond skewness is bond return", page: 37}
```
```
concept_id: spread_momentum_6m
source_class: bond   quote: "mom. 6m log(Spread) is the log of the spread six"   page: 36
as_described: {label: "Mom. 6m log(Spread)", quote: "mom. 6m log(Spread) is the log of the spread six", page: 36}
```
```
concept_id: spread_to_d2d
source_class: accounting   quote: "Spread-to-D2D is the option-adjusted spread, divided by one minus the cumulative distribution function"   page: 37
as_described: {label: "Spread-to-D2D", quote: "Spread-to-D2D is the option-adjusted spread, divided by one minus the cumulative distribution function", page: 37}
```
```
concept_id: bond_volatility
source_class: bond   quote: "Bond volatility is bond return volatility over the past 24"   page: 37
as_described: {label: "Bond volatility", quote: "Bond volatility is bond return volatility over the past 24", page: 37}
```
```
concept_id: bond_var_36m
source_class: bond   quote: "Value-at-risk is the second lowest credit excess return over the past"   page: 37
as_described: {label: "Value-at-risk", quote: "Value-at-risk is the second lowest credit excess return over the past", page: 37}
```
```
concept_id: vix_beta
source_class: macro   quote: "VIX beta is the sum of coefficients on current and lagged VIX"   page: 37
as_described: {label: "VIX beta", quote: "VIX beta is the sum of coefficients on current and lagged VIX", page: 37}
```

## paper_facts

```
sample_start:            value: 1999-01   quote: "The sample is January 1999 through December 2020"   page: 16
sample_end:              value: 2020-12   quote: "The sample is January 1999 through December 2020"   page: 16
universe_filter:         value: "discard bond-month observations with spreads < 50 bps or > 2,000 bps at the beginning of the period; bonds with duration < 0.25 years discarded"
                         quote: "We discard bond-month observations with extreme bond spreads"   page: 12
                         # Paper prints 50 bps (p12); the engine kpp_ipca.yaml uses 20 (build spec flags 50 as a
                         # text error). The gold records the paper (see dissection §5).
claimed_headline_metric: value: {mean: 51.0, t_stat: 0.0, unit: oos_panel_r2_pct}
                         quote: "Our main five-factor IPCA model explains 51% of the panel variation"   page: 3
                         # KPP's headline is an OOS panel R-squared (51%), not a mean/t-stat premium; encoded in the
                         # {mean, t_stat, unit} composite as mean=51.0, t_stat=0.0 (n/a), unit=oos_panel_r2_pct.
```

## Self-checks

1. **Tag audit.** Part 1 STATED×2 + method_summary; estimation STATED×11; instruments 29 × (concept_id
   STATED + source_class STATED), transform/lag UNKNOWN (loader-injected); paper_facts STATED×4. **Zero
   INFERRED; zero DESIGN.** `source_class` values are the coarse data-domain of the paper-described
   construction (evidence = the appendix definition), not a token the paper prints — see the dissection's
   "source_class semantics" note; borderline rows (Spread-to-D2D, Book/Earnings-to-price, VIX beta) are
   flagged there for review.
2. **Engine reconciliation** (vs `agents/quant/library/configs/kpp_ipca.yaml` / `ipca_spec.md` §8):
   model_family→IPCA ✓ · ALS ✓ · K∈{1..5}, headline 5→`model.K` ✓ · restricted (Gamma_alpha=0)→`model.alpha:
   false` ✓ · DtS scaling→`scaling.lane: DTSScaled25` ✓ · rank→[-0.5,0.5]→`data_contract.rank_map` ✓ ·
   OOS burn-in 36→`window.oos_burn_in_months: 36` ✓ · wild bootstrap {1000, dof 7}→`bootstrap` ✓. **One
   divergence (paper wins in the gold):** universe spread bound — paper 50 bps, engine 20 (dissection §5).
3. **Locator backfill.** Every STATED quote RE-VERIFIED 2026-08-03 as an exact L1 substring of the FROZEN
   canonical text (`kpp_2023.frozen.yaml`, `locate_quote(..., "L1")`) — zero NOT-FOUND, zero cross-page,
   zero offset drift (`scripts/regenerate_locator_backfill.py kpp`).

---

**Non-pooling.** This gold's field-accuracy is reported as a SEPARATE fitted-model sub-metric
(contract v1.3 / D42), physically isolated from the sort G3 number (`ANCHOR_SET = str/drf/mom6`). See
`docs/evaluation/evaluation_contract_v1.md` §(v1.3) and `docs/librarian/registers/decision-log_librarian.md` D42.
