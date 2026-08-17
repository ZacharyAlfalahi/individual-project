# Gold Spec — LRF (BBW 2019)

_Status: **INSTANTIATED 2026-08-15 (schema v1.1).** BBW canonical text frozen
(`evaluation/canonical_texts/bbw_2019.frozen.yaml`, same source_sha256 as drf/crf); every STATED quote is an
exact L1 substring with a binding locator (`locator_backfill_report.md`, generated via
`scripts/regenerate_locator_backfill.py lrf`). LRF is BBW's liquidity risk factor: a single-leg bivariate
rating×illiquidity sort — structurally identical to DRF except the sort signal is the Bao–Pan–Wang gamma
illiquidity measure (ILLIQ) instead of the 5% VaR, and the sample window/headline differ. Mirrors the DRF
gold field-for-field. Built as a P1 codegen oracle (`data/development/factors/bbw_factors.parquet` →
`lrf_corr`)._
_Pages are canonical-text page indices (1-based, = PDF pages) of `evaluation/canonical_texts/bbw_2019.frozen.yaml`.
Every STATED quote is an exact substring of `normalise(page, L1)` — locator backfill results in
`gold_specs/locator_backfill_report.md`._

## Header

```
paper:                   BBW_2019
strategy_label:          LRF
strategy_quote:          "Liquidity risk factor ( LRF ) is con- structed by independently sorting corporate bonds into 5 × 5 quintiles based on illiquidity and credit rating. LRF is the value-weighted aver- age return difference between the highest-illiquidity portfolio minus the lowest-illiquidity portfolio within each rating portfolio."   page: 15
registry_version:        v1
silence_policy_version:  v1.1
canonical_text_hash:     normalise_sha256 abe11ff63b76df2e46a775a70f1feb397d10fa59984bafd38d312b178b01a79e (config/canonical_text.yaml recipe, PyMuPDF 1.28.0 / L1 / v3 ladder); source_sha256 fcb58bf7bd433f82e722d284776da639e038f1bc5e0c919d8943f9a04cbde91a (frozen bbw_2019)
```

## Part 1 — three fields (all CORE)

```
formation_structure:  value: sorted_portfolios   tag: STATED
                      quote: "Liquidity risk factor ( LRF ) is con- structed by independently sorting corporate bonds into 5 × 5 quintiles based on illiquidity and credit rating. LRF is the value-weighted aver- age return difference between the highest-illiquidity portfolio minus the lowest-illiquidity portfolio within each rating portfolio."   page: 15

asset_class:          value: corporate_bonds   tag: STATED
                      quote: "the newly proposed factors of corporate bonds"   page: 15

method_summary:       Each month, all sample bonds are independently sorted into five credit-rating
                      quintiles and five quintiles of bond illiquidity, measured as the Bao–Pan–Wang
                      (2011) gamma illiquidity measure (ILLIQ). Within the resulting 5×5 grid, the factor
                      is long the highest-illiquidity quintile and short the lowest-illiquidity quintile,
                      value-weighted, averaged across the five rating quintiles. The output is LRF, a
                      monthly long-short liquidity-risk factor return series (August 2002–December 2016).
                      Locating quotes: (1) construction sentence, page 15 (header strategy_quote);
                      (2) "The liquid- ity risk factor, LRF , is the value-weighted average return difference between the highest-illiquidity and the lowest- illiquidity portfolios across the rating portfolios." page 14;
                      (3) "we follow Bao, Pan, and Wang (2011) to construct bond- level illiquidity measure, ILLIQ , which aims" page 8
```

## Part 2 · Sort block (1 leg)

### CORE

```
[MARKER] sort_signal:   concept_id: bpw_gamma   params: {}
                        as_described: {label: "bond illiquidity (measured by the Bao–Pan–Wang (2011) gamma measure ILLIQ): the negative autocovariance of daily price changes",
                                       quote: "we follow Bao, Pan, and Wang (2011) to construct bond- level illiquidity measure, ILLIQ , which aims",
                                       page: 8}
                        # LRF's sort signal is the BPW illiquidity measure ILLIQ (gamma), the sole
                        # structural difference from DRF (whose signal is var_5pct). Same concept as
                        # CRF's leg-2 control (bpw_gamma); binds to column `gamma` downstream.

         sort_kind:     value: independent   tag: STATED
                        quote: "We construct the bond factors in a similar vein to Fama and French (2015) and rely on independent sorts."   page: 14
                        # Same §5.1 sentence that grounds DRF/CRF: the FACTOR construction is independent.
                        # (BBW's Table 3 bivariate-CONTROL analysis is sequential — that is analysis, not
                        # the factor. An extraction that answers "conditional" has read Table 3, not §5.1.)

◇        control_axis:  concept_id: credit_rating   params: {}
                        # Field name per the 2026-07-11 brief amendment: keeps the shipped code name
                        # `control_axis` (control_signal was a brief-coined alias, rejected).
                        as_described: {label: "credit rating as the first sorting variable",
                                       quote: "it is natural to use credit risk (proxied by credit rating) as the ﬁrst sorting variable in the construction of these new bond market factors",
                                       page: 14}

◇        control_n_groups: value: 5   tag: STATED
                        quote: "5 × 5 quintiles based on illiquidity and credit rating"   page: 15

[MARKER] n_groups:      value: 5   tag: STATED
                        quote: "5 × 5 quintiles based on illiquidity and credit rating"   page: 15

         long_leg:      value: highest_signal   tag: STATED
                        quote: "The liquid- ity risk factor, LRF , is the value-weighted average return difference between the highest-illiquidity and the lowest- illiquidity portfolios across the rating portfolios."   page: 14

         combiner:      single_leg
```

### TAIL

```
bucketing_method:        value: UNKNOWN   reason: not_stated
                         searched_note: "§5.1 + Table 6 caption say only 'quintiles' / '5 × 5'; no breakpoint
                         universe or equal-count statement. 'Quintile' names the buckets, not the method
                         (FF-style papers say quintiles while using external breakpoints), so equal_count
                         is not quotable. Engine default equal_count applies downstream."
stripe_aggregation:      value: UNKNOWN   reason: not_stated
                         searched_note: "The LRF spread is the illiquidity spread 'across the rating
                         portfolios' (p14) / 'within each rating portfolio' (p15). Neither states whether the
                         five rating-stripe spreads are averaged equally or by value. 'Value-weighted' in the
                         LRF sentence most naturally modifies the portfolio returns, not the stripe average.
                         Genuine silence -> default equal."
control_missing_policy:  value: UNKNOWN   reason: not_stated
                         searched_note: "Nothing in §3-§5 on bonds lacking a rating at formation (drop vs
                         pool). Searched data §3.1, §4.2, §5.1. Not specified."
signal_transform:        value: UNKNOWN   reason: not_stated
                         searched_note: "No winsorisation/standardisation of the ILLIQ signal anywhere
                         ('winsoriz' absent from the full text). Raw signal ranked."
```

## Part 2 · Common block (spec-level)

### CORE

```
weighting_scheme:   value: value   tag: STATED
                    quote: "The liquid- ity risk factor, LRF , is the value-weighted average return difference between the highest-illiquidity and the lowest- illiquidity portfolios across the rating portfolios."   page: 14

⚠ weighting_base:   value: par   tag: STATED
                    quote: "The portfolios are value weighted using amount outstanding as weights."   page: 9
                    # NOT the engine's design default written in: the PAPER states the base. "Amount
                    # outstanding" is a face-value (par-family) quantity — it contains no price, so it is
                    # not market_value. Two honesty caveats, recorded deliberately (shared with drf/crf):
                    # (1) the base sentence attaches to the §4 characteristic-sorted portfolios (body p9 +
                    #     Table 2/3 captions); the §5.1 factor text repeats only "value-weighted". Fn.28
                    #     ties the factor to those portfolios: "We rely on the independently sorted 5 × 5 portfolios to construct the factors to be consistent with our univariate and bivariate portfolio re- sults from quintile portfolios." (page 15).
                    #     If maximal strictness is preferred, demote to UNKNOWN(not_stated) — flagged for review.
                    # (2) paper base = amount OUTSTANDING; engine par weight = offering_amt
                    #     (BBW_anchor_implementation_spec §2.4) — a within-par-family divergence, real data.

holding_period:     value: UNKNOWN   reason: not_stated
                    searched_note: "'holding period' absent from the full text. Portfolios are re-formed
                    monthly ('The liquidity risk and the return reversal factors are constructed similarly
                    using independent sorts.', p14, back onto 'for each month ... we form bivariate
                    portfolios', p14) and Table 2 reports 'the next-month average excess return' (p9), which
                    jointly entail one-month holds for the factor, but no sentence states a holding period.
                    Silence -> engine default 1, which matches the entailed value."

rebalance_frequency: value: monthly   tag: STATED
                    quote: "for each month from July 2004 to December 2016, we form bivariate portfolios by independently sorting bonds"   page: 14
                    # The only quotable "for each month ... we form bivariate portfolios" cadence sentence
                    # is the DRF one; LRF is "constructed similarly using independent sorts" (p14), i.e. the
                    # SAME monthly re-formation. The quote grounds the monthly cadence; LRF's distinct window
                    # (August 2002–December 2016) lives in paper_facts.sample_start below.

strategy_side:      value: long_short   tag: STATED
                    quote: "the value-weighted average return difference between the highest-illiquidity and the lowest- illiquidity portfolios"   page: 14

signal_lag:         value: UNKNOWN   reason: not_stated
                    searched_note: "The ILLIQ measure is estimated on daily transaction prices within the
                    formation month, and returns are 'one-month-ahead' / 'next-month' (pp. 3, 9), which
                    together imply the signal window ends at the formation month-end (lag 0) — but no single
                    sentence states the signal-to-formation offset, so the value is not quotable without
                    stitching. Silence -> engine default 0, which matches the implication."

return_label:       value: UNKNOWN   reason: not_stated
                    searched_note: "For the factor series the paper never labels months. Table 2's
                    'next-month average excess return' (p9) describes the quintile analysis; the LRF series
                    is the corresponding monthly spread. Default realisation matches."

missing_return_policy: value: UNKNOWN   reason: not_stated
                    searched_note: "Searched §3.1 filters, §3.2 return construction, §5.1. Return exists
                    under the two month-end/month-begin price scenarios (p5); nothing on how a position
                    with a missing next-month return, default, or delisting is treated inside a portfolio
                    month. Bias-relevant silence, worth probing in RQ2."
```

### TAIL

```
eligibility_missing_policy:  UNKNOWN(not_stated)   note: "No general missing-characteristic handling stated
                             for the illiquidity/rating sort (LRF has no VaR-style ≥24-obs screen). Searched
                             §3.3 (ILLIQ construction), §5.1."
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
burn_in_policy:              UNKNOWN(not_stated)   note: "LRF simply starts August 2002 (the TRACE panel
                             start), needing no 36-month VaR burn-in — the ILLIQ signal is available from the
                             panel start ('LRF and REV cover the period from August 2002 to December 2016.',
                             p15); no ramp policy stated."
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
                             anywhere ('winsoriz'/'outlier'/'truncat' absent for returns). Cross-ref
                             lab_trim: BBW is the anchor WITHOUT a stated ex-post return filter."
```

## ◇ paper_facts (spec-level, v1.1)

```
sample_start:            value: 2002-08   quote: "LRF and REV cover the period from August 2002 to December 2016."   page: 15
                         # LRF/REV start August 2002 (the TRACE panel start), NOT July 2004: LRF needs no
                         # 36-month VaR burn-in (contrast DRF/CRF). Panel/data start: "we rely on the transaction records reported in the enhanced version of the TRACE for the sample period July 2002 to December 2016" (page 5). The LRF factor window is what the fidelity harness must match.
sample_end:              value: 2016-12   quote: "LRF and REV cover the period from August 2002 to December 2016."   page: 15
universe_filter:         value: "Enhanced TRACE, July 2002–December 2016; remove: non-US-market/144A/non-USD/foreign-issuer bonds; structured notes, MBS/ABS, agency-backed, equity-linked; convertibles; price <$5 or >$1000; floating coupon; <1 year to maturity; when-issued/locked-in/special-condition trades and >2-day settlement; cancelled/corrected/reversed records; trades <$10,000 (as stated, filters 1–9)"
                         quote: "Remove bonds that are not listed or traded in the US public market, which include bonds issued through private placement, bonds issued under the 144A rule, bonds that do not trade in US dollars, and bond issuers not in the jurisdiction of the United States."   page: 5
claimed_headline_metric: value: {mean: 0.52, t_stat: 5.02, unit: pct_per_month}
                         quote: "Liquidity risk factor (LRF) 0.52 5.02"   page: 15
```

## Self-checks

1. **Tag audit.** STATED×14 all quote+page; UNKNOWN×22 all with searched notes; zero INFERRED; zero
   DESIGN (weighting_base is STATED here because BBW prints the base — see the ⚠ caveat for the
   demote-to-UNKNOWN option). Identical STATED/UNKNOWN partition to the DRF gold.
2. **Rulebook cross-check** (vs `agents/quant/library/bbw_factors.py` LRF config): groups 5×5 ✓ ·
   independent ✓ · sort signal bpw_gamma (illiquidity), long high-ILLIQ − low-ILLIQ ✓ · rating control ✓ ·
   monthly/1-month hold ✓ (gold: UNKNOWN→default 1) · value-weighted by par ✓. **Divergence (real data):
   paper weights by amount outstanding; engine by offering_amt (par proxy).** Paper wins in the gold.
   The single structural difference from DRF is the sort signal (bpw_gamma vs var_5pct); window and
   headline (0.52% / t=5.02, Aug 2002–Dec 2016) also differ.
3. **Locator backfill.** Run and recorded in `locator_backfill_report.md` (all quotes L1-located
   against the frozen canonical text with char offsets; markdown-bootstrap path, like crf/kpp).

---

**Anchor targets.** The fidelity `gate` (pass/fail), `replication_target`, and `raw_expectation` for this anchor live in the evaluation-layer register `docs/quant/registers/anchor_targets.md` (O5 / D-Q7 / A4.4) — kept out of this spec (not new `paper_facts` fields) to protect RQ1's field-level-accuracy denominator.
