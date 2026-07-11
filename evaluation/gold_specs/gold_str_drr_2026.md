# Gold Spec — str (Dickerson, Robotti & Rossetti 2026)

_Status: **DRAFT — authored 2026-07-11 under schema v1 with ◇ (v1.1) fields pre-filled.** Extraction
is complete; the spec remains **BLOCKED-ON-RENAME** for instantiation until the reversal column
(score/rev/str_reversal) is reconciled to one canonical name on the Quant side. Re-stamp on schema
v1.1 landing._
_Target construction: the paper's **unadjusted single-sort** factor (Table 1 Panel A, column (1)) —
the standard-practice construction whose bias the project studies, not the signal-/return-adjusted
variants._
_No frozen canonical text exists for this paper yet. Pages are 1-based PDF pages of
`papers/pdf/DRR (2026).pdf` parsed with the frozen recipe (PyMuPDF 1.28.0, L1/v2 ladder). Quotes
verified as exact L1 substrings (`locator_backfill_report.md`)._

## Header

```
paper:                   DRR_2026
strategy_label:          str
strategy_quote:          "In Panel A, we sort bonds into deciles each month and form value-weighted portfolios (using bond market capitalization) that are long the top decile and short the bottom decile."   page: 17
registry_version:        v1
silence_policy_version:  v1
canonical_text_hash:     PENDING-FREEZE — recipe normalise_sha256 3d846fef7d710af17c4a183b69353e705ed11e957d7df9390fcba35ac89eee95; source_sha256 d481c99e38fbb012a87f33ecedbcba6614a35e0c28555e39ad459246edf42b58
```

## Part 1 — three fields (all CORE)

```
formation_structure:  value: sorted_portfolios   tag: STATED
                      quote: "we sort bonds into deciles each month and form value-weighted portfolios (using bond market capitalization) that are long the top decile and short the bottom decile"   page: 17

asset_class:          value: corporate_bonds   tag: STATED
                      quote: "Corporate bond factor research faces a replication crisis. We construct a 'factor zoo' of 108 corporate bond factors"   page: 3

method_summary:       Each month bonds are ranked on the short-term reversal signal (str) and sorted into
                      deciles. The factor is long the top decile P10 and short the bottom decile P1,
                      value-weighted by bond market capitalization, with month-end prices feeding both the
                      signal and the next month's return (the paper's unadjusted Approach 1). The output is
                      a monthly long-short factor return series, 2002-09 to 2024-12. Locating quotes:
                      (1) header strategy_quote, page 17; (2) "Approach 1 (Unadjusted) uses the month-end price P end i,t for both signal computation and return measurement, which is the standard practice in the literature." page 15;
                      (3) "str denotes short-term reversal" page 10
```

## Part 2 · Sort block (1 leg)

### CORE

```
[MARKER] sort_signal:   concept_id: prior_1m_excess_return   params: {}
                        as_described: {label: "short-term reversal (str); month-end price feeds the signal",
                                       quote: "For example, cs denotes credit spread, str denotes short-term reversal, and mom6_1 denotes six-month momentum.",
                                       page: 10}
                        # Concept resolution: registry v1 alias 'short-term reversal' → prior_1m_excess_return.
                        # The main text never prints an explicit formula for str; the one-month structure is
                        # corroborated by the theory section's reversal-signal treatment
                        # ("for rever- sal signals where ηi,t = δi,t −δi,t−1", page 61) and by Approach 1's
                        # signal-at-month-end-t timing (page 15). Registry aliasing carries the mapping;
                        # if the Internet Appendix (not in corpus) defines str otherwise, re-open.

         sort_kind:     value: single   tag: STATED
                        quote: "In Panel A (single-sort), the unad- justed factor earns −0.99% per month (t = −4.46), but the signal-adjusted factor earns only −0.09% (t = −0.51)."   page: 17
                        # The paper's own label for this construction. Panel B's within-firm variant is a
                        # different construction (issuer-demeaned), deliberately NOT this anchor.

◇        control_signal: none
                        # Single sort; consistency pair holds (single ⇔ control none).

◇        control_n_groups: value: UNKNOWN   reason: not_stated
                        searched_note: "No control axis exists; field vacuous for this spec."

[MARKER] n_groups:      value: 10   tag: STATED
                        quote: "we sort bonds into deciles each month"   page: 17

         long_leg:      value: highest_signal   tag: STATED
                        quote: "Panel A sorts bonds into deciles each month, going long P10 and short P1."   page: 37
                        # Long the top decile OF THE SIGNAL. Under the registry concept's positive
                        # prior-return convention, P10 = past winners, and the stated unadjusted premium is
                        # negative (−0.99): losers outperform winners pre-adjustment, which is the reversal
                        # story and the LIB mechanism (loser prices bounce back). See divergence flag in
                        # self-check 2 — the engine spec's leg convention is the sign-mirror of this.

         combiner:      single_leg
```

### TAIL

```
bucketing_method:        value: UNKNOWN   reason: not_stated
                         searched_note: "'deciles' names the buckets; the breakpoint universe is only
                         parameterised in the §5.2 design-space grid ('breakpoint universe (all bonds,
                         investment grade only, large bonds only)', p33), and Table 1 does not state which
                         applies. Default equal_count on the full eligible sample."
stripe_aggregation:      value: UNKNOWN   reason: not_stated
                         searched_note: "Single sort; no stripes. Vacuous."
control_missing_policy:  value: UNKNOWN   reason: not_stated
                         searched_note: "No control axis. Vacuous."
signal_transform:        value: UNKNOWN   reason: not_stated
                         searched_note: "No winsorisation/standardisation of the str signal stated for the
                         unadjusted construction; the paper's whole point is that Approach 1 uses the raw
                         month-end-price signal."
```

## Part 2 · Common block (spec-level)

### CORE

```
weighting_scheme:   value: value   tag: STATED
                    quote: "form value-weighted portfolios (using bond market capitalization)"   page: 17

⚠ weighting_base:   value: market_value   tag: STATED
                    quote: "Portfolios are value-weighted (using bond market capitalization) with excess returns over the one-month T-bill rate."   page: 37
                    # The paper STATES the base — bond market capitalization = price-based market value.
                    # This is the anti-contamination case the template warns about, in reverse: do NOT
                    # write par here. Engine par weighting is a downstream DESIGN substitution and a real
                    # paper≠engine divergence (self-check 2).

holding_period:     value: UNKNOWN   reason: not_stated
                    searched_note: "No holding-period sentence for the Table 1 factors. Monthly
                    re-formation ('we sort bonds into deciles each month', p17) plus month-end returns
                    r_{t+1} (p15) entail one-month holds, but no sentence states it; contrast their
                    mom6_1, where a staggered six-month hold IS stated in a figure caption. Silence ->
                    engine default 1, which matches the entailment."

rebalance_frequency: value: monthly   tag: STATED
                    quote: "we sort bonds into deciles each month"   page: 17

strategy_side:      value: long_short   tag: STATED
                    quote: "Panel A sorts bonds into deciles each month, going long P10 and short P1."   page: 37

signal_lag:         value: 0   tag: STATED
                    quote: "Portfolio weights ωi,t are computed from signals observed at P end i,t, and returns are the month-end returns rEnd i,t+1."   page: 15
                    # Signal observed at the month-end t price = measured at formation; a stated zero lag
                    # under the month-end convention. (This zero lag is exactly what creates LIB.)

return_label:       value: realisation   tag: STATED
                    quote: "Portfolio weights ωi,t are computed from signals observed at P end i,t, and returns are the month-end returns rEnd i,t+1."   page: 15
                    # Factor month = realisation month t+1 of the month-t formation.

missing_return_policy: value: UNKNOWN   reason: not_stated
                    searched_note: "Handled upstream by the stated return-availability rule (see TAIL):
                    bonds without a qualifying trade are 'excluded for that month'. Defaulted bonds are
                    NOT dropped: 'Dropping defaulted bonds from the sample censors extreme outcomes and can bias measured return distributions. We track bonds through and after default.' (p9), with explicit
                    default-return formulas in Appendix A.5 ('When a bond enters or trades under default, we assume coupon payments cease and adjust the return formula accordingly.', p53). No residual
                    within-month missing-return policy is stated beyond these two rules."
```

### TAIL

```
eligibility_missing_policy:  UNKNOWN(not_stated)   note: "Beyond the price-availability rule (below),
                             missing-signal handling is not stated for Table 1; §5.1 filters are the
                             studied variations, not the baseline."
return_availability_policy:  value: require_next_month_return   tag: STATED
                             quote: "A valid return requires such a price in both months t and t + 1 (bonds without a trade in either window are excluded for that month)."   page: 9
return (price convention)    note: "The monthly price is the last available daily price within the last five business days of the month (New York Stock Exchange (NYSE) calendar)." (p9) — recorded here
                             because it pins the t and t+1 windows the rule quantifies over.
lag_convention:              UNKNOWN(not_stated)   note: "Month-end measurement is explicit (p9, p15) but
                             lag counting is never formalised as a convention."
min_bonds:                   UNKNOWN(not_stated)   note: "Only fn.17's assurance that 'even the most
                             restrictive configurations contain a sufficient number of bonds' (p33); no
                             minimum rule."
min_bonds_granularity:       UNKNOWN(not_stated)   note: "See min_bonds."
tie_break_policy:            UNKNOWN(not_stated)   note: "Not addressed."
weight_timing:               UNKNOWN(not_stated)   note: "Weights computed at formation ('Portfolio
                             weights ωi,t are computed from signals observed at P end i,t', p15) — but
                             whether the market-cap weight itself is the formation-month value is not
                             separately stated; moot at 1-month holds."
empty_leg_policy:            UNKNOWN(not_stated)   note: "Empty legs discussed only for the §5.2 grid
                             ('Sixteen specifications (0.088%) pro- duce months with empty long or short
                             legs', p33), not the baseline construction."
transaction_cost_convention: UNKNOWN(not_stated)   note: "Table 1 premia carry no cost overlay; LIB is
                             framed as an implementation gap, not a modelled cost. Default gross."
overlap_convention:          UNKNOWN(not_stated)   note: "No overlapping cohorts for str (contrast their
                             mom6_1 'staggered six-month holding period'). Monthly single-period
                             construction."
cohort_weighting:            UNKNOWN(not_stated)   note: "No cohorts."
burn_in_policy:              UNKNOWN(not_stated)   note: "Factor sample simply starts 2002-09 (one month
                             after the 2002-08 panel start; the signal needs the prior month). No ramp
                             policy stated."
realisation_min_survivors:   UNKNOWN(not_stated)   note: "Not addressed."
return_compounding:          UNKNOWN(not_stated)   note: "Monthly arithmetic returns per Eq. (A.11);
                             not applicable beyond that."
significance_convention:     value: hac_t_of_mean (Newey-West)   tag: STATED
                             quote: "t-statistics (Newey-West, lags = ⌊T 0.25⌋) in parentheses. Sample: 2002-09 to 2024-12, T=268."   page: 37
hac_lags:                    value: floor(T^0.25)   tag: STATED   note: "Formula, not a fixed int; with
                             T=268 this is 4. Same quote as significance_convention (p37)."
annualisation:               UNKNOWN(not_stated)   note: "All results in % per month."
rf_convention:               value: subtract_rf   tag: STATED
                             quote: "Excess returns subtract the Fama-French one-month T-bill rate."   page: 9
benchmark_model:             value: capm   tag: STATED   note: "Single-factor BOND CAPM (MKTB), not the
                             equity CAPM. Headline metric below is the raw premium; α = −0.77 (t = −3.58)
                             vs CAPMB is the stated alpha."
                             quote: "We use a single-factor bond CAPM (CAPMB) comprising MKTB to compute alphas throughout the paper"   page: 9
expost_trim:                 UNKNOWN(not_stated)   note: "No trim is stated for the unadjusted Table 1
                             construction; ex-post winsorisation is the LAB treatment the paper studies in
                             OTHER papers' constructions (§4), and §5 calls the no-filter case 'the
                             baseline (unfiltered) specification' (p32). Silence -> default none, which is
                             also the paper's design intent. Cross-ref lab_trim: for str the bias lever is
                             LIB (price-noise), not LAB (trim)."
```

## ◇ paper_facts (spec-level, v1.1)

```
sample_start:            value: 2002-09   quote: "t-statistics (Newey-West, lags = ⌊T 0.25⌋) in parentheses. Sample: 2002-09 to 2024-12, T=268."   page: 37
                         # Panel/database window is stated separately: "The sample spans August 2002 to December 2024 (269 months), with 52,656 unique bonds and an average of 6,790 bond-month observations per month." (page 10). The factor loses 2002-08 to the signal's prior-month requirement.
sample_end:              value: 2024-12   quote: "t-statistics (Newey-West, lags = ⌊T 0.25⌋) in parentheses. Sample: 2002-09 to 2024-12, T=268."   page: 37
universe_filter:         value: "TRACE enhanced + 144A merged with Mergent FISD; retain USD-denominated, fixed-rate, non-convertible, non-asset-backed bonds with $1,000 par value and original maturity ≥1 year; Dick-Nielsen (2014) cleaning + the paper's decimal-shift and bounce-back correctors; bonds tracked through and after default (as stated)"
                         quote: "We retain USD-denominated, fixed-rate, non-convertible, non-asset-backed bonds with $1,000 par value and original maturity of at least one year. Standard filters from Dick-Nielsen (2014) remove cancellations, corrections, reversals, and agency-side du- plicates."   page: 8
claimed_headline_metric: value: {mean: -0.99, t_stat: -4.46, unit: pct_per_month}
                         quote: "In Panel A (single-sort), the unad- justed factor earns −0.99% per month (t = −4.46), but the signal-adjusted factor earns only −0.09% (t = −0.51)."   page: 17
                         # Sign convention: this is the printed long-P10-short-P1 series. Figures flip it
                         # for display: "The str factor is sign-corrected to have a positive premium." (p19).
                         # The harness must compare sign-aware against the P10−P1 convention, not the
                         # display sign.
```

## Self-checks

1. **Tag audit.** STATED×17 all quote+page; UNKNOWN×19 all with searched notes; zero INFERRED; zero
   DESIGN (weighting_base is STATED market_value — the paper prints the base).
2. **Rulebook cross-check** (vs `docs/quant/specs/BBW_anchor_implementation_spec.md` §5.1): single-sort ✓ ·
   monthly ✓ · deciles ✓ (engine §5.1 wording says VW par "quintiles" via §3.2/§2.4 defaults —
   engine's own §12 should confirm decile wiring for str). **Three divergences (paper wins in the
   gold): (i) weighting base — paper market capitalization, engine par offering_amt; (ii) LEG
   CONVENTION — paper long P10 (highest prior return: winners−losers, printed premium −0.99);
   engine §5.1 'Construction A' builds losers−winners and claims that convention produces −0.99 and
   matches the downloadable series. Both cannot be true: under the paper's stated P10−P1 the
   losers−winners mirror is +0.99. The gold records the paper. Engine team: re-verify §5.1's sign
   note against the stored DRR series before locking the target; (iii) sample end 2024-12 vs any
   engine window assumption.** Divergences are data, not errors in the gold.
3. **Locator backfill.** Quotes verified as exact L1 substrings of the parsed PDF
   (`locator_backfill_report.md`); re-run when a frozen canonical text exists (PENDING-FREEZE).
