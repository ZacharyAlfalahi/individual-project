# Gold Spec — CRF (BBW 2019)

_Status: **INSTANTIATED 2026-08-04 (schema v1.1; multi-leg).** BBW canonical text frozen
(`evaluation/canonical_texts/bbw_2019.frozen.yaml`, same source_sha256 as drf); every STATED quote is an
exact L1 substring with a binding locator (`locator_backfill_report.md`, generated via
`scripts/regenerate_locator_backfill.py crf`). CRF is BBW's credit-risk factor: THREE independent 5×5
rating-signal sorts combined by `equal_average` (D28). The FIRST multi-leg gold — RQ1 + RQ2 only
(excluded from RQ3/RQ4)._
_Pages are canonical-text page indices (1-based, = PDF pages) of `evaluation/canonical_texts/bbw_2019.frozen.yaml`.
Every STATED quote is an exact substring of `normalise(page, L1)`._

## Header

```
paper:                   BBW_2019
strategy_label:          CRF
strategy_quote:          "These independent sorts also produce three credit risk factors so that the final credit risk factor (CRF) is defined as the average of the three factors of credit risk."   page: 3
registry_version:        v1
silence_policy_version:  v1.1
canonical_text_hash:     normalise_sha256 abe11ff63b76df2e46a775a70f1feb397d10fa59984bafd38d312b178b01a79e (config/canonical_text.yaml recipe, PyMuPDF 1.28.0 / L1 / v3 ladder); source_sha256 fcb58bf7bd433f82e722d284776da639e038f1bc5e0c919d8943f9a04cbde91a (frozen bbw_2019)
```

## Part 1 — three fields (all CORE)

```
formation_structure:  value: sorted_portfolios   tag: STATED
                      quote: "These independent sorts also produce three credit risk factors so that the final credit risk factor (CRF) is defined as the average of the three factors of credit risk."   page: 3

asset_class:          value: corporate_bonds   tag: STATED
                      quote: "the newly proposed factors of corporate bonds"   page: 15

method_summary:       Each month, all sample bonds are independently double-sorted, credit rating as the
                      first axis, against three signals in turn — the 5% VaR, the Bao–Pan–Wang illiquidity
                      measure (ILLIQ), and the one-month return reversal (REV) — into three independent 5×5
                      grids. Within each grid the credit leg is long the lowest-rating (highest credit risk)
                      bonds and short the highest-rating (lowest credit risk) bonds, value-weighted, averaged
                      across the five signal quintiles, giving CRF_VaR, CRF_ILLIQ and CRF_REV. The credit
                      risk factor CRF is the equal-weighted average of the three: CRF = 1/3(CRF_VaR +
                      CRF_ILLIQ + CRF_REV), a monthly long-short series (July 2004–December 2016).
                      Return-basis note (§1.6): BBW construct monthly returns under two scenarios — end-of-
                      month t−1 → end-of-month t, OR beginning-of-month t → end-of-month t — and when both
                      are available scenario one (the end-to-end, unbiased basis) wins. The preference is a
                      load-bearing construction choice, recorded here as prose (no schema field).
                      Locating quotes: (1) construction sentence, page 3 (header strategy_quote);
                      (2) "the average monthly excess returns for the 5 × 5 portfolios independently sorted on Rating and VaR, Rating and ILLIQ, and Rating and REV." page 14;
                      (3) composite sentence "Credit risk factor ( CRF) is the average of the CRF obtained from forming the DRF, LRF, and REV" page 15.
```

## Part 2 · Leg 1 — CRF_VaR

### CORE

```
[MARKER] sort_signal:   concept_id: credit_rating   params: {}
                        as_described: {label: "credit rating as the first sorting variable (1=AAA … 22=D)",
                                       quote: "it is natural to use credit risk (proxied by credit rating) as the first sorting variable in the construction of these new bond market factors",
                                       page: 14}

         sort_kind:     value: independent   tag: STATED
                        quote: "We construct the bond factors in a similar vein to Fama and French (2015) and rely on independent sorts."   page: 14

◇        control_axis:  concept_id: var_5pct   params: {}
                        as_described: {label: "downside risk (measured by 5% VaR): second lowest monthly return over past 36 months, multiplied by −1",
                                       quote: "Downside risk is the 5% VaR of corporate bond return, defined as the second lowest monthly return observation over the past 36 months. The original VaR measure is multiplied by –1 so that a higher VaR indicates higher downside risk.",
                                       page: 6}

◇        control_n_groups: value: 5   tag: STATED
                        quote: "the average monthly excess returns for the 5 × 5 portfolios independently sorted on Rating and VaR, Rating and ILLIQ, and Rating and REV."   page: 14

[MARKER] n_groups:      value: 5   tag: STATED
                        quote: "the average monthly excess returns for the 5 × 5 portfolios independently sorted on Rating and VaR, Rating and ILLIQ, and Rating and REV."   page: 14

         long_leg:      value: highest_signal   tag: STATED
                        quote: "The credit risk factor, CRF VaR, is the value-weighted aver- age return difference between the lowest-rating (i.e., high- est credit risk) portfolio and the highest-rating (i.e., lowest credit risk) portfolio across the VaR portfolios."   page: 14
                        # sort_signal is credit_rating on the DRR scale 1=AAA…22=D, so long = lowest-rating
                        # (highest credit risk) = HIGHEST numeric rating value = highest_signal. The most
                        # error-prone field (§2): long is the WORST credit, short is the BEST credit.
```

### TAIL

```
bucketing_method:        value: UNKNOWN   reason: not_stated
                         searched_note: "§5.1 + Table 6 caption say only 'quintiles' / '5 × 5'; no breakpoint
                         universe or equal-count statement. Engine default equal_count applies downstream."
stripe_aggregation:      value: UNKNOWN   reason: not_stated
                         searched_note: "Each CRF_x is the credit spread 'across the [signal] portfolios'
                         (p14); whether the five signal-stripe spreads are averaged equally or by value is not
                         stated. Genuine silence -> default equal."
control_missing_policy:  value: UNKNOWN   reason: not_stated
                         searched_note: "Nothing on bonds lacking a VaR value at formation (drop vs pool).
                         Searched §3-§5. Not specified."
signal_transform:        value: UNKNOWN   reason: not_stated
                         searched_note: "No winsorisation/standardisation of the VaR control ('winsoriz'
                         absent from the full text). Raw signal ranked."
```

## Part 2 · Leg 2 — CRF_ILLIQ

### CORE

```
[MARKER] sort_signal:   concept_id: credit_rating   params: {}
                        as_described: {label: "credit rating as the first sorting variable (1=AAA … 22=D)",
                                       quote: "it is natural to use credit risk (proxied by credit rating) as the first sorting variable in the construction of these new bond market factors",
                                       page: 14}

         sort_kind:     value: independent   tag: STATED
                        quote: "We construct the bond factors in a similar vein to Fama and French (2015) and rely on independent sorts."   page: 14

◇        control_axis:  concept_id: bpw_gamma   params: {}
                        as_described: {label: "Bao–Pan–Wang bond-level illiquidity measure ILLIQ (negative autocovariance of price changes)",
                                       quote: "we follow Bao, Pan, and Wang (2011) to construct bond- level illiquidity measure, ILLIQ, which aims to extract the transitory component from bond price.",
                                       page: 8}

◇        control_n_groups: value: 5   tag: STATED
                        quote: "the average monthly excess returns for the 5 × 5 portfolios independently sorted on Rating and VaR, Rating and ILLIQ, and Rating and REV."   page: 14

[MARKER] n_groups:      value: 5   tag: STATED
                        quote: "the average monthly excess returns for the 5 × 5 portfolios independently sorted on Rating and VaR, Rating and ILLIQ, and Rating and REV."   page: 14

         long_leg:      value: highest_signal   tag: STATED
                        quote: "The credit risk factor, CRF VaR, is the value-weighted aver- age return difference between the lowest-rating (i.e., high- est credit risk) portfolio and the highest-rating (i.e., lowest credit risk) portfolio across the VaR portfolios."   page: 14
```

### TAIL

```
bucketing_method:        value: UNKNOWN   reason: not_stated
                         searched_note: "As CRF_VaR: only 'quintiles' / '5 × 5' stated; engine default equal_count."
stripe_aggregation:      value: UNKNOWN   reason: not_stated
                         searched_note: "Averaging of the five ILLIQ-stripe credit spreads not stated -> default equal."
control_missing_policy:  value: UNKNOWN   reason: not_stated
                         searched_note: "Bonds lacking an ILLIQ value at formation not addressed. Searched §3.3.3, §5.1."
signal_transform:        value: UNKNOWN   reason: not_stated
                         searched_note: "No winsorisation/standardisation of the ILLIQ control stated. Raw signal ranked."
```

## Part 2 · Leg 3 — CRF_REV

### CORE

```
[MARKER] sort_signal:   concept_id: credit_rating   params: {}
                        as_described: {label: "credit rating as the first sorting variable (1=AAA … 22=D)",
                                       quote: "it is natural to use credit risk (proxied by credit rating) as the first sorting variable in the construction of these new bond market factors",
                                       page: 14}

         sort_kind:     value: independent   tag: STATED
                        quote: "We construct the bond factors in a similar vein to Fama and French (2015) and rely on independent sorts."   page: 14

◇        control_axis:  concept_id: prior_1m_excess_return   params: {}
                        as_described: {label: "one-month return reversal (REV): previous-month return, sorted 5×5 with credit rating",
                                       quote: "Return re- versal factor ( REV) is constructed by independently sorting corporate bonds into 5 × 5 quintiles based on the previous month return and credit rating.",
                                       page: 15}

◇        control_n_groups: value: 5   tag: STATED
                        quote: "the average monthly excess returns for the 5 × 5 portfolios independently sorted on Rating and VaR, Rating and ILLIQ, and Rating and REV."   page: 14

[MARKER] n_groups:      value: 5   tag: STATED
                        quote: "the average monthly excess returns for the 5 × 5 portfolios independently sorted on Rating and VaR, Rating and ILLIQ, and Rating and REV."   page: 14

         long_leg:      value: highest_signal   tag: STATED
                        quote: "The credit risk factor, CRF VaR, is the value-weighted aver- age return difference between the lowest-rating (i.e., high- est credit risk) portfolio and the highest-rating (i.e., lowest credit risk) portfolio across the VaR portfolios."   page: 14
                        # CONFIRM-ON-READ (spec §3): REV is the PREVIOUS-month return, a one-month-lagged sort
                        # signal that may differ from the VaR/ILLIQ legs. signal_lag is a COMMON-block field
                        # (one per strategy) — the REV-lag caveat is recorded in the common signal_lag note.
```

### TAIL

```
bucketing_method:        value: UNKNOWN   reason: not_stated
                         searched_note: "As CRF_VaR: only 'quintiles' / '5 × 5' stated; engine default equal_count."
stripe_aggregation:      value: UNKNOWN   reason: not_stated
                         searched_note: "Averaging of the five REV-stripe credit spreads not stated -> default equal."
control_missing_policy:  value: UNKNOWN   reason: not_stated
                         searched_note: "Bonds lacking a previous-month return at formation not addressed. Searched §5.1."
signal_transform:        value: UNKNOWN   reason: not_stated
                         searched_note: "No winsorisation/standardisation of the REV control stated. Raw signal ranked."
```

## Part 2 · Combiner

```
combiner:   value: equal_average   tag: STATED
            quote: "Credit risk factor ( CRF) is the average of the CRF obtained from forming the DRF, LRF, and REV"   page: 15
            # D28: multi-leg construction. The equal_average combiner is grounded on its OWN composite
            # sentence (not borrowed from any leg's sort_kind). The full formula also appears verbatim:
            # "CRF = 1 / 3(CRF VaR + C RF I LLI Q + C RF REV)" (p15); the clean front fragment is used here
            # because the formula's subscript glyphs render with internal spaces (§1.1 caveat).
```

## Part 2 · Common block (spec-level)

### CORE

```
weighting_scheme:   value: value   tag: STATED
                    quote: "The credit risk factor, CRF VaR, is the value-weighted aver- age return difference between the lowest-rating (i.e., high- est credit risk) portfolio and the highest-rating (i.e., lowest credit risk) portfolio across the VaR portfolios."   page: 14

⚠ weighting_base:   value: par   tag: STATED
                    quote: "The portfolios are value weighted using amount outstanding as weights."   page: 9
                    # Shared with drf: paper base = amount OUTSTANDING (par-family, no price); engine par
                    # weight = offering_amt (BBW_anchor_implementation_spec §2.4) — a within-par-family
                    # divergence, real data. Same standing par-proxy row as drf.

holding_period:     value: UNKNOWN   reason: not_stated
                    searched_note: "'holding period' absent from the full text. Portfolios re-formed monthly;
                    Table entries report next-month returns, jointly entailing one-month holds, but no sentence
                    states a holding period. Silence -> engine default 1."

rebalance_frequency: value: monthly   tag: STATED
                    quote: "for each month from July 2004 to December 2016, we form bivariate portfolios by independently sorting bonds"   page: 14

strategy_side:      value: long_short   tag: STATED
                    quote: "The credit risk factor, CRF VaR, is the value-weighted aver- age return difference between the lowest-rating (i.e., high- est credit risk) portfolio and the highest-rating (i.e., lowest credit risk) portfolio across the VaR portfolios."   page: 14

signal_lag:         value: UNKNOWN   reason: not_stated
                    searched_note: "The rating first-sort carries no explicit lag; REV is the previous-month
                    return (leg 3 caveat). No single sentence states the signal-to-formation offset for the
                    credit legs. Silence -> engine default 0."

return_label:       value: UNKNOWN   reason: not_stated
                    searched_note: "For the CRF series the paper never labels months. Default realisation matches."

missing_return_policy: value: UNKNOWN   reason: not_stated
                    searched_note: "Return exists under the two month-end/month-begin price scenarios (p5,
                    §1.6); nothing on how a position with a missing next-month return is treated inside a
                    portfolio month. Bias-relevant silence, worth probing in RQ2."
```

### TAIL

```
eligibility_missing_policy:  UNKNOWN(not_stated)   note: "Only the VaR-specific ≥24-obs/36-month screen is
                             stated (p6); general missing-characteristic handling unstated."
return_availability_policy:  UNKNOWN(not_stated)   note: "Return realisation requires the two price scenarios
                             of §3.2 (p5), but no explicit require-next-month-return rule for membership."
lag_convention:              UNKNOWN(not_stated)   note: "Month-end language throughout §3.2; lag counting never formalised."
min_bonds:                   UNKNOWN(not_stated)   note: "No minimum-bonds-per-portfolio rule found (§4, §5, Table 6)."
min_bonds_granularity:       UNKNOWN(not_stated)   note: "See min_bonds."
tie_break_policy:            UNKNOWN(not_stated)   note: "Not addressed."
weight_timing:               UNKNOWN(not_stated)   note: "Whether weights refresh within the hold is moot at monthly re-formation; not stated."
empty_leg_policy:            UNKNOWN(not_stated)   note: "Not addressed."
transaction_cost_convention: UNKNOWN(not_stated)   note: "No cost overlay stated for factor returns -> gross. Searched §5.1 + Table 6."
overlap_convention:          UNKNOWN(not_stated)   note: "No overlapping-cohort language anywhere. Monthly single-period construction."
cohort_weighting:            UNKNOWN(not_stated)   note: "No cohorts (see overlap_convention)."
burn_in_policy:              UNKNOWN(not_stated)   note: "CRF simply starts July 2004 once the 36-month VaR
                             window is available ('DRF and CRF cov- ers the period from July 2004 to December 2016.', p15); no ramp policy stated."
realisation_min_survivors:   UNKNOWN(not_stated)   note: "Not addressed."
return_compounding:          UNKNOWN(not_stated)   note: "Monthly arithmetic returns per Eq. (1) (p5); cross-month compounding not applicable at the 1-month horizon."
significance_convention:     UNKNOWN(not_stated) for Table 6 Panel A premia   note: "Newey-West is stated for
                             the sort tables and Panel B alphas; Table 6 Panel A's mean/t-stat rows carry no explicit convention sentence."
hac_lags:                    UNKNOWN(not_stated)   note: "Lag count never printed."
annualisation:               UNKNOWN(not_stated)   note: "All results quoted in % per month; never annualised."
rf_convention:               value: subtract_rf   tag: STATED
                             quote: "We denote R i, t as bond i 's excess return, R i,t = r i,t −r f,t, where r f, t is the risk-free rate proxied by the one-month Treasury bill rate."   page: 5
benchmark_model:             value: other   tag: STATED   note: "Alphas measured against the ten-factor
                             model (five stock + five bond factors); the CRF headline metric below is the raw premium. Note the CRF 10-factor alpha loses significance (§1.2)."
                             quote: "Model 3 is the ten-factor model that combines the five stock and five bond market factors."   page: 15
expost_trim:                 UNKNOWN(not_stated)   note: "DEFINITIVE ABSENCE (§1.4): full-text search finds NO
                             winsorisation, trimming, or return-level outlier filter anywhere in BBW
                             ('winsoriz'/'outlier'/'truncat' absent for returns). BBW is the anchor WITHOUT a
                             stated ex-post return filter — the absence is established, not merely unread. Cross-ref lab_trim."
```

## ◇ paper_facts (spec-level, v1.1)

```
sample_start:            value: 2004-07   quote: "DRF and CRF cov- ers the period from July 2004 to December 2016."   page: 15
                         # §1.3 window ambiguity (RESOLVED): body text elsewhere says "August 2002 to December
                         # 2016", but Table 6's header and this sentence give July 2004 for DRF and CRF (LRF/REV
                         # from August 2002). July 2004 is correct — 24 months after July 2002, matching fn 13's
                         # ≥24-monthly-return VaR screen. Both quotes worth recording as a resolved defect (an
                         # Auditor demonstration case).
sample_end:              value: 2016-12   quote: "DRF and CRF cov- ers the period from July 2004 to December 2016."   page: 15
universe_filter:         value: "Enhanced TRACE, July 2002–December 2016; remove: non-US-market/144A/non-USD/foreign-issuer bonds; structured notes, MBS/ABS, agency-backed, equity-linked; convertibles; price <$5 or >$1000; floating coupon; <1 year to maturity; when-issued/locked-in/special-condition trades and >2-day settlement; cancelled/corrected/reversed records; trades <$10,000 (as stated, filters 1–9). CALLABLES RETAINED (fn 11, ~67% of the sample); only convertibles removed for optionality."
                         quote: "Remove bonds that are not listed or traded in the US public market, which include bonds issued through private placement, bonds issued under the 144A rule, bonds that do not trade in US dollars, and bond issuers not in the jurisdiction of the United States."   page: 5
claimed_headline_metric: value: {mean: 0.43, t_stat: 2.78, unit: pct_per_month}
                         quote: "significant premiums of 0.43% per month ( t -stat. = 2.78)"   page: 15
```

## Self-checks

1. **Tag audit.** STATED: 3 legs × (sort_signal, sort_kind, control_axis, control_n_groups, n_groups,
   long_leg) + combiner + Part 1 (formation_structure, asset_class) + common (weighting_scheme,
   weighting_base, rebalance_frequency, strategy_side, rf_convention, benchmark_model) + paper_facts
   (sample_start, sample_end, universe_filter, claimed_headline_metric). Zero INFERRED; zero DESIGN.
2. **Rulebook cross-check** (vs `agents/quant/library/bbw_factors.py` CRF_* configs): three legs, rating as
   score axis, controls {var_5pct, bpw_gamma→gamma, prior_1m_excess_return→xret}, long_group 4 = worst
   credit ✓; groups 5×5 ✓; independent ✓; value-weighted by par ✓; `combiner: equal_average` (÷ available) ✓.
   Reconciliation: `prior_1m_excess_return` binds to column `xret` (D27), so `factor_rulebook("crf_rev")`
   control is reconciled `rev`→`xret` for byte-equality (never the frozen concept table).
3. **Locator backfill.** Generated + recorded in `locator_backfill_report.md` (all quotes L1-located
   against the frozen canonical text with char offsets; markdown-bootstrap path, like KPP).

---

**Anchor targets.** CRF is RQ1 + RQ2 only. It carries NO ±15% fidelity target and NO RQ3 bias-differential
(excluded from the lattice / anchor set). The realised empirical CRF sign is a diagnostic, never a gate.
