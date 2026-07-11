# Gold Spec — DRF (BBW 2019)

_Status: **DRAFT — authored 2026-07-11 under schema v1 with ◇ (v1.1) fields pre-filled from the paper.**
Schema v1.1 (control axis + paper_facts) has NOT landed (`code-brief_schema-v1.1_2026-07-09.md`: committed NO).
Re-stamp against v1.1 on landing; expected diff = ◇ field stamps only, zero value changes._
_Pages are canonical-text page indices (1-based, = PDF pages) of `evaluation/canonical_texts/bbw_2019.frozen.yaml`.
Every STATED quote is an exact substring of `normalise(page, L1)` — locator backfill results in
`gold_specs/locator_backfill_report.md`._

## Header

```
paper:                   BBW_2019
strategy_label:          DRF
strategy_quote:          "To con- struct the downside risk factor for corporate bonds, for each month from July 2004 to December 2016, we form bivariate portfolios by independently sorting bonds into five quintiles based on their credit rating and five quin- tiles based on their downside risk (measured by 5% VaR)."   page: 14
registry_version:        v1
silence_policy_version:  v1
canonical_text_hash:     normalise_sha256 3d846fef7d710af17c4a183b69353e705ed11e957d7df9390fcba35ac89eee95 (config/canonical_text.yaml recipe, PyMuPDF 1.28.0 / L1 / v2 ladder); source_sha256 fcb58bf7bd433f82e722d284776da639e038f1bc5e0c919d8943f9a04cbde91a (frozen bbw_2019)
```

## Part 1 — three fields (all CORE)

```
formation_structure:  value: sorted_portfolios   tag: STATED
                      quote: "we form bivariate portfolios by independently sorting bonds into five quintiles based on their credit rating and five quin- tiles based on their downside risk (measured by 5% VaR)"   page: 14

asset_class:          value: corporate_bonds   tag: STATED
                      quote: "To con- struct the downside risk factor for corporate bonds, for each month from July 2004 to December 2016, we form bivariate portfolios"   page: 14

method_summary:       Each month, all sample bonds are independently sorted into five credit-rating
                      quintiles and five quintiles of downside risk, measured as the 5% VaR (the second
                      lowest monthly return over the past 36 months, ×−1). Within the resulting 5×5 grid,
                      the factor is long the highest-VaR quintile and short the lowest-VaR quintile,
                      value-weighted, averaged across the five rating quintiles. The output is DRF, a
                      monthly long-short downside-risk factor return series (July 2004–December 2016).
                      Locating quotes: (1) construction sentence, page 14 (header strategy_quote);
                      (2) "The downside risk factor, DRF, is the value-weighted av- erage return difference between the highest-VaR portfolio and the lowest-VaR portfolio across the rating portfolios." page 14;
                      (3) "Downside risk is the 5% VaR of corporate bond return, defined as the second lowest monthly return observation over the past 36 months." page 6
```

## Part 2 · Sort block (1 leg)

### CORE

```
[MARKER] sort_signal:   concept_id: var_5pct   params: {}
                        as_described: {label: "downside risk (measured by 5% VaR): second lowest monthly return over past 36 months, multiplied by −1",
                                       quote: "Downside risk is the 5% VaR of corporate bond return, defined as the second lowest monthly return observation over the past 36 months. The original VaR measure is multiplied by –1 so that a higher VaR indicates higher downside risk.",
                                       page: 6}
                        # 36-month window + ≥24-obs screen are part of the concept identity as the paper
                        # states it: "A bond is included in VaR calculation if it has at least 24 monthly return ob- servations in the 36-month rolling window before the test month." (page 6)

         sort_kind:     value: independent   tag: STATED
                        quote: "We construct the bond factors in a similar vein to Fama and French (2015) and rely on independent sorts."   page: 14
                        # ⚠ TRAP documented: BBW's Table 3 bivariate-CONTROL analysis is sequential
                        # ("by first sorting corporate bonds based on credit rating ... Then, within each
                        # control quintile, corporate bonds are further sorted into subquintiles based on
                        # their 5% VaR", page 11). That is analysis, not the factor. The FACTOR construction
                        # is stated independent (this quote + Table 6 caption, page 15). An extraction that
                        # answers "conditional" has read Table 3 instead of §5.1.

◇        control_signal: concept_id: credit_rating   params: {}
                        as_described: {label: "credit rating as the first sorting variable",
                                       quote: "it is natural to use credit risk (proxied by credit rating) as the first sorting variable in the construction of these new bond market factors",
                                       page: 14}

◇        control_n_groups: value: 5   tag: STATED
                        quote: "independently sorting bonds into five quintiles based on their credit rating"   page: 14

[MARKER] n_groups:      value: 5   tag: STATED
                        quote: "five quin- tiles based on their downside risk (measured by 5% VaR)"   page: 14

         long_leg:      value: highest_signal   tag: STATED
                        quote: "The downside risk factor, DRF, is the value-weighted av- erage return difference between the highest-VaR portfolio and the lowest-VaR portfolio across the rating portfolios."   page: 14

         combiner:      single_leg
```

### TAIL

```
bucketing_method:        value: UNKNOWN   reason: not_stated
                         searched_note: "§5.1 + Table 2/3/6 captions say only 'quintiles'; no breakpoint
                         universe or equal-count statement. 'Quintile' names the buckets, not the method
                         (FF-style papers say quintiles while using external breakpoints), so equal_count
                         is not quotable. Engine default equal_count applies downstream."
stripe_aggregation:      value: UNKNOWN   reason: not_stated
                         searched_note: "Main text says the DRF spread is averaged 'across the rating
                         portfolios' (p14); Table 6 caption says 'within each rating portfolio' (p15).
                         Neither states whether the five rating-stripe spreads are averaged equally or by
                         value. 'Value-weighted' in the DRF sentence most naturally modifies the portfolio
                         returns, not the stripe average. Genuine silence -> default equal."
control_missing_policy:  value: UNKNOWN   reason: not_stated
                         searched_note: "Nothing in §3-§5 on bonds lacking a rating at formation (drop vs
                         pool). Searched data §3.1, §4.2, §5.1. Not specified."
signal_transform:        value: UNKNOWN   reason: not_stated
                         searched_note: "No winsorisation/standardisation of VaR anywhere ('winsoriz'
                         absent from the full text). Raw signal ranked."
```

## Part 2 · Common block (spec-level)

### CORE

```
weighting_scheme:   value: value   tag: STATED
                    quote: "The downside risk factor, DRF, is the value-weighted av- erage return difference between the highest-VaR portfolio and the lowest-VaR portfolio across the rating portfolios."   page: 14

⚠ weighting_base:   value: par   tag: STATED
                    quote: "The portfolios are value weighted using amount outstanding as weights."   page: 9
                    # NOT the engine's design default written in: the PAPER states the base. "Amount
                    # outstanding" is a face-value (par-family) quantity — it contains no price, so it is
                    # not market_value. Two honesty caveats, recorded deliberately:
                    # (1) the base sentence attaches to the §4 characteristic-sorted portfolios (body p9 +
                    #     Table 2/3 captions); the §5.1 factor text repeats only "value-weighted". Fn.28
                    #     ties the factor to those portfolios: "We rely on the independently sorted 5 × 5 portfolios to construct the factors to be consistent with our univariate and bivariate portfolio re- sults from quintile portfolios." (page 15).
                    #     If maximal strictness is preferred, demote to UNKNOWN(not_stated) — flagged for review.
                    # (2) paper base = amount OUTSTANDING; engine par weight = offering_amt
                    #     (BBW_anchor_implementation_spec §2.4) — a within-par-family divergence, real data.

holding_period:     value: UNKNOWN   reason: not_stated
                    searched_note: "'holding period' absent from the full text. Portfolios are re-formed
                    monthly ('for each month from July 2004 to December 2016, we form bivariate
                    portfolios', p14) and Table 2 reports 'the next-month average excess return' (p9), which
                    jointly entail one-month holds for the factor, but no sentence states a holding period;
                    monthly re-formation alone does not exclude overlapping-cohort designs (cf. JNPS).
                    Silence -> engine default 1, which matches the entailed value."

rebalance_frequency: value: monthly   tag: STATED
                    quote: "for each month from July 2004 to December 2016, we form bivariate portfolios by independently sorting bonds"   page: 14

strategy_side:      value: long_short   tag: STATED
                    quote: "the value-weighted av- erage return difference between the highest-VaR portfolio and the lowest-VaR portfolio"   page: 14

signal_lag:         value: UNKNOWN   reason: not_stated
                    searched_note: "Two stated pieces: the VaR window sits 'in the 36-month rolling window
                    before the test month' (p6) and returns are 'one-month-ahead' / 'next-month' (pp. 3, 9),
                    which together imply the window ends at the formation month-end, i.e. lag 0 — but no
                    single sentence states the signal-to-formation offset, so the value is not quotable
                    without stitching. Silence -> engine default 0, which matches the implication."

return_label:       value: UNKNOWN   reason: not_stated
                    searched_note: "For the factor series the paper never labels months. Table 2's
                    'next-month average excess return' (p9) describes the quintile analysis; the DRF series
                    is the corresponding monthly spread. Default realisation matches."

missing_return_policy: value: UNKNOWN   reason: not_stated
                    searched_note: "Searched §3.1 filters, §3.2 return construction, §5.1. Return exists
                    under the two month-end/month-begin price scenarios (p5); nothing on how a position
                    with a missing next-month return, default, or delisting is treated inside a portfolio
                    month. Bias-relevant silence, worth probing in RQ2."
```

### TAIL

```
eligibility_missing_policy:  UNKNOWN(not_stated)   note: "Only the VaR-specific screen is stated (≥24 obs
                             in 36-month window, p6); general missing-characteristic handling unstated."
return_availability_policy:  UNKNOWN(not_stated)   note: "Return realisation requires the two price
                             scenarios of §3.2 (p5), but no explicit require-next-month-return rule for
                             portfolio membership. Searched §3.2 + §4.1."
lag_convention:              UNKNOWN(not_stated)   note: "Month-end language throughout §3.2 ('from the
                             end of month t −1 to the end of month t', p5); lag counting never formalised."
min_bonds:                   UNKNOWN(not_stated)   note: "No minimum-bonds-per-portfolio rule found
                             (§4, §5, table captions)."
min_bonds_granularity:       UNKNOWN(not_stated)   note: "See min_bonds."
tie_break_policy:            UNKNOWN(not_stated)   note: "Not addressed."
weight_timing:               UNKNOWN(not_stated)   note: "Whether weights refresh within the hold is moot
                             at monthly re-formation; not stated."
empty_leg_policy:            UNKNOWN(not_stated)   note: "Not addressed."
transaction_cost_convention: UNKNOWN(not_stated)   note: "No cost overlay stated for factor returns ->
                             gross. Searched §5.1 + Table 6."
overlap_convention:          UNKNOWN(not_stated)   note: "No overlapping-cohort language anywhere (contrast
                             JNPS). Monthly single-period construction."
cohort_weighting:            UNKNOWN(not_stated)   note: "No cohorts (see overlap_convention)."
burn_in_policy:              UNKNOWN(not_stated)   note: "Factor simply starts July 2004 once the 36-month
                             VaR window is available from the July 2002 panel ('DRF and CRF cov- ers the
                             period from July 2004 to December 2016.', p15); no ramp policy stated."
realisation_min_survivors:   UNKNOWN(not_stated)   note: "Not addressed."
return_compounding:          UNKNOWN(not_stated)   note: "Monthly arithmetic returns per Eq. (1) (p5);
                             cross-month compounding not applicable at 1-month analysis horizon."
significance_convention:     UNKNOWN(not_stated) for Table 6 Panel A premia   note: "Newey-West is stated
                             for the sort tables ('Newey-West adjusted t -statistics are given in
                             parentheses.', p9 Table 2 caption) and Panel B alphas; Table 6 Panel A's
                             mean/t-stat rows carry no explicit convention sentence."
hac_lags:                    UNKNOWN(not_stated)   note: "Lag count never printed."
annualisation:               UNKNOWN(not_stated)   note: "All results quoted in % per month; never
                             annualised."
rf_convention:               value: subtract_rf   tag: STATED
                             quote: "We denote R i, t as bond i 's excess return, R i,t = r i,t −r f,t, where r f, t is the risk-free rate proxied by the one-month Treasury bill rate."   page: 5
benchmark_model:             value: other   tag: STATED   note: "Alphas measured against the ten-factor
                             model (five stock + five bond factors); headline metric below is the raw premium."
                             quote: "Model 3 is the ten-factor model that combines the five stock and five bond market factors."   page: 15
expost_trim:                 UNKNOWN(not_stated)   note: "No return winsorisation/truncation stated
                             anywhere ('winsoriz'/'outlier'/'truncat' absent except characteristics
                             elsewhere). Cross-ref lab_trim: BBW is the anchor WITHOUT a stated ex-post
                             return filter."
```

## ◇ paper_facts (spec-level, v1.1)

```
sample_start:            value: 2004-07   quote: "DRF and CRF cov- ers the period from July 2004 to December 2016."   page: 15
                         # Panel/data start is earlier: "we rely on the transaction records reported in the enhanced version of the TRACE for the sample period July 2002 to December 2016" (page 5). The
                         # DRF factor window is what the fidelity harness must match.
sample_end:              value: 2016-12   quote: "DRF and CRF cov- ers the period from July 2004 to December 2016."   page: 15
universe_filter:         value: "Enhanced TRACE, July 2002–December 2016; remove: non-US-market/144A/non-USD/foreign-issuer bonds; structured notes, MBS/ABS, agency-backed, equity-linked; convertibles; price <$5 or >$1000; floating coupon; <1 year to maturity; when-issued/locked-in/special-condition trades and >2-day settlement; cancelled/corrected/reversed records; trades <$10,000 (as stated, filters 1–9)"
                         quote: "Remove bonds that are not listed or traded in the US public market, which include bonds issued through private placement, bonds issued under the 144A rule, bonds that do not trade in US dollars, and bond issuers not in the jurisdiction of the United States."   page: 5
claimed_headline_metric: value: {mean: 0.70, t_stat: 3.60, unit: pct_per_month}
                         quote: "The value-weighted DRF factor has an economically and statistically significant risk premium of 0.70% per month with a t -statistic of 3.60."   page: 15
```

## Self-checks

1. **Tag audit.** STATED×14 all quote+page; UNKNOWN×22 all with searched notes; zero INFERRED; zero
   DESIGN (weighting_base is STATED here because BBW prints the base — see the ⚠ caveat for the
   demote-to-UNKNOWN option).
2. **Rulebook cross-check** (vs `docs/quant/specs/BBW_anchor_implementation_spec.md` §3–§4): groups 5×5 ✓ ·
   independent ✓ · direction high-VaR−low-VaR ✓ · monthly/1-month hold ✓ (gold: UNKNOWN→default 1) ·
   VaR5 definition incl. 36m/24-obs ✓. **Divergence (real data): paper weights by amount
   outstanding; engine by offering_amt (par proxy).** Paper wins in the gold.
3. **Locator backfill.** Run and recorded in `locator_backfill_report.md` (all quotes L1-located
   against the frozen canonical text with char offsets).
