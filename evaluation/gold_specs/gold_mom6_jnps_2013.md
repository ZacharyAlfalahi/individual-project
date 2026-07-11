# Gold Spec — mom6 (Jostova, Nikolova, Philipov & Stahel 2013, RFS)

_Status: **AUTHORED 2026-07-11 under schema v1; ◇ (v1.1) fields pre-filled** (control axis trivially
`none`; paper_facts filled). Re-stamp on v1.1 landing; expected diff = ◇ stamps only._
_No frozen canonical text exists for this paper yet. Pages are 1-based PDF pages of
`papers/pdf/Momentum in Corporate Bond Returns.pdf` parsed with the frozen recipe (PyMuPDF 1.28.0,
L1/v2 ladder, watermark strip v3). Quotes verified as exact L1 substrings
(`locator_backfill_report.md`); formal locator backfill re-runs when the canonical text is frozen._

## Header

```
paper:                   JNPS_2013
strategy_label:          mom6
strategy_quote:          "Specifically, each month t, bonds are sorted into decile portfolios, P1 to P10, based on their cumulative returns over months t −6 to t −1 (formation period)."   page: 9
registry_version:        v1
silence_policy_version:  v1
canonical_text_hash:     PENDING-FREEZE — recipe normalise_sha256 3d846fef7d710af17c4a183b69353e705ed11e957d7df9390fcba35ac89eee95; source_sha256 7e80f8cb919de4161822df63da1069310d8a113b0562ccd2d3d6420d4f707242
```

## Part 1 — three fields (all CORE)

```
formation_structure:  value: sorted_portfolios   tag: STATED
                      quote: "each month t, bonds are sorted into decile portfolios, P1 to P10, based on their cumulative returns over months t −6 to t −1 (formation period)"   page: 9

asset_class:          value: corporate_bonds   tag: STATED
                      quote: "This paper documents significant momentum in a comprehensive sample of 81,491 U.S. corporate bonds"   page: 1

method_summary:       Each month t, bonds are ranked on their cumulative return over months t−6 to t−1 and
                      sorted into deciles P1–P10. The strategy is long the winner decile P10 and short the
                      loser decile P1, equally weighted within portfolios, with positions held over months
                      t+1 to t+6 (one skip month), so the strategy month-t return equally averages the six
                      overlapping cohorts alive that month. The output is a monthly long-short momentum
                      return series, January 1973–June 2011. Locating quotes: (1) header strategy_quote,
                      page 9; (2) "The momentum strategy is long the winner portfolio, P10, and short the loser portfolio, P1." page 9;
                      (3) "The overall momentum strategy month-t return is the equally weighted average month-t return of strategies implemented in the prior month and strategies formed up to six months earlier." page 9
```

## Part 2 · Sort block (1 leg)

### CORE

```
[MARKER] sort_signal:   concept_id: past_6m_cumulative_return   params: {}
                        as_described: {label: "cumulative returns over months t−6 to t−1 (formation period)",
                                       quote: "based on their cumulative returns over months t −6 to t −1 (formation period)",
                                       page: 9}

         sort_kind:     value: single   tag: STATED
                        quote: "each month t, bonds are sorted into decile portfolios, P1 to P10, based on their cumulative returns over months t −6 to t −1 (formation period)"   page: 9
                        # One sort axis described; no control axis anywhere in §2. The quote states a
                        # plain single decile sort.

◇        control_signal: none
                        # Single sort; consistency pair holds (single ⇔ control none).

◇        control_n_groups: value: UNKNOWN   reason: not_stated
                        searched_note: "No control axis exists; field vacuous for this spec. Default
                        (= n_groups) never bites."

[MARKER] n_groups:      value: 10   tag: STATED
                        quote: "bonds are sorted into decile portfolios, P1 to P10"   page: 9

         long_leg:      value: highest_signal   tag: STATED
                        quote: "The momentum strategy is long the winner portfolio, P10, and short the loser portfolio, P1."   page: 9

         combiner:      single_leg
```

### TAIL

```
bucketing_method:        value: UNKNOWN   reason: not_stated
                         searched_note: "'decile portfolios' names the buckets; no breakpoint universe or
                         equal-count statement (§2 + Table 2 caption). Default equal_count."
stripe_aggregation:      value: UNKNOWN   reason: not_stated
                         searched_note: "Single sort; no stripes. Vacuous."
control_missing_policy:  value: UNKNOWN   reason: not_stated
                         searched_note: "No control axis. Vacuous."
signal_transform:        value: UNKNOWN   reason: not_stated
                         searched_note: "No transform of the formation-period cumulative return is stated.
                         NB fn.16's sample-wide return elimination (see expost_trim) removes return
                         observations >99.5th pct and therefore touches formation-window inputs too; the
                         paper does not describe it as a signal transform. Fn.26 truncates characteristics
                         (not the sort signal) at the 99th pct."
```

## Part 2 · Common block (spec-level)

### CORE

```
weighting_scheme:   value: equal   tag: STATED
                    quote: "Portfolio returns are equally weighted across their constituent bonds."   page: 9

⚠ weighting_base:   value: UNKNOWN   reason: not_stated
                    searched_note: "Equal-weighted throughout; a par-vs-market-value base has no bite and
                    the paper never states one. Do not inject the engine's par default here."

holding_period:     value: 6   tag: STATED
                    quote: "The portfolios are held over months t +1 to t +6 (holding period)."   page: 9

rebalance_frequency: value: monthly   tag: STATED
                    quote: "each month t, bonds are sorted into decile portfolios, P1 to P10"   page: 9
                    # Monthly formation of a new cohort; positions overlap (see overlap_convention).

strategy_side:      value: long_short   tag: STATED
                    quote: "The momentum strategy is long the winner portfolio, P10, and short the loser portfolio, P1."   page: 9

signal_lag:         value: 1   tag: STATED
                    quote: "Following the equity momentum literature, we skip one month between the formation and holding periods to avoid potential biases from bid-ask bounce and short-term price reversal."   page: 9
                    # Window t−6..t−1 ends one month before the sort month t ('skip one month'); under the
                    # month-end convention that is a stated one-month signal lag at formation.

return_label:       value: realisation   tag: STATED
                    quote: "The overall momentum strategy month-t return is the equally weighted average month-t return of strategies implemented in the prior month and strategies formed up to six months earlier."   page: 9
                    # The series is indexed by the realisation month t of previously formed cohorts.

missing_return_policy: value: UNKNOWN   reason: not_stated
                    searched_note: "Searched §1 (data), §2, and the appendix: no statement on how a
                    cohort position with a missing month-(t+k) return, default, or delisting is treated
                    inside the 6-month hold. Bias-relevant silence for bonds."
```

### TAIL

```
eligibility_missing_policy:  UNKNOWN(not_stated)   note: "Whether a bond needs all six formation-window
                             returns (vs a partial cumulative return) is never stated (§2; searched
                             'consecutive', 'at least'). Genuine silence on formation eligibility."
return_availability_policy:  UNKNOWN(not_stated)   note: "No require-next-month-return rule stated for
                             cohort membership."
lag_convention:              UNKNOWN(not_stated)   note: "Month-indexed timeline (t−6..t−1, t+1..t+6);
                             lag counting never formalised beyond that."
min_bonds:                   UNKNOWN(not_stated)   note: "No minimum-bonds rule found (§2 + captions)."
min_bonds_granularity:       UNKNOWN(not_stated)   note: "See min_bonds."
tie_break_policy:            UNKNOWN(not_stated)   note: "Not addressed."
weight_timing:               UNKNOWN(not_stated)   note: "EW within portfolios; whether cohort weights
                             refresh monthly within the hold is not stated."
empty_leg_policy:            UNKNOWN(not_stated)   note: "Not addressed."
transaction_cost_convention: UNKNOWN(not_stated)   note: "Headline profits carry no cost overlay; costs
                             are analysed separately ('Panel C of Table 10 summarizes gross and net momentum profits for key subsamples.', p40) -> default gross matches."
overlap_convention:          value: overlapping   tag: STATED
                             quote: "The overall momentum strategy month-t return is the equally weighted average month-t return of strategies implemented in the prior month and strategies formed up to six months earlier."   page: 9
                             # The template's mom6 note anticipated exactly this: a genuinely STATED
                             # overlap field. Also: "This allows for standard statistical inference based on nonoverlapping returns." (p9).
cohort_weighting:            value: equal   tag: STATED
                             quote: "the equally weighted average month-t return of strategies implemented in the prior month and strategies formed up to six months earlier"   page: 9
burn_in_policy:              UNKNOWN(not_stated)   note: "First months with <6 live cohorts never
                             discussed."
realisation_min_survivors:   UNKNOWN(not_stated)   note: "Not addressed."
return_compounding:          UNKNOWN(not_stated)   note: "Formation signal is a cumulative 6-month return
                             (necessarily compounded); the strategy series itself is monthly arithmetic.
                             No explicit statement of either."
significance_convention:     UNKNOWN(not_stated) for Table 2 premia   note: "Table 2 caption states only
                             'The t-statistics of the P10-P1 returns are in parentheses in the last
                             column.' (p10). Newey-West IS stated for the alpha regressions: 'The
                             coefficients are estimated using OLS with Newey-West-adjusted standard
                             errors.' (p12) — but not for the headline premium t-stat."
hac_lags:                    UNKNOWN(not_stated)   note: "Never printed."
annualisation:               UNKNOWN(not_stated)   note: "All results in bps/% per month."
rf_convention:               value: none   tag: STATED
                             quote: "the momentum portfolio excess return over the risk-free rate or the momentum strategy return rp,t =RP 10,t −RP 1,t"   page: 12
                             note: "The strategy return is the raw P10−P1 difference (rf cancels);
                             rf-subtraction applies only to portfolio-level alpha regressions."
benchmark_model:             value: other   tag: STATED   note: "Alphas vs combinations of equity (MKT,
                             SMB, HML, momentum) and bond (term, default) factors; headline metric below
                             is the raw premium."
                             quote: "The coefficients are estimated using OLS with Newey-West-adjusted standard errors."   page: 12
expost_trim:                 value: truncate   tag: STATED   bounds: right tail, >99.5th percentile (≈ returns >30%/month), full-sample threshold
                             quote: "To ensure that the results are not driven by outliers, we have eliminated return observations above the 99.5th percentile (returns above 30% per month). The results are robust to alternative cutoffs."   page: 7
                             # Cross-ref lab_trim: this is THE look-ahead lever for the mom6 anchor —
                             # 'eliminated' = truncation on a full-sample percentile (ex-post). DRR-2026
                             # quantifies the resulting LAB. NOTE the engine adjudicated the repair as a
                             # winsorise/clip (wins, right, 99.5 per DRR FilterClass —
                             # BBW_anchor_implementation_spec §5.2); the PAPER's own words say 'eliminated'
                             # (drop). Paper wins in this gold; divergence flagged below.
```

## ◇ paper_facts (spec-level, v1.1)

```
sample_start:            value: 1973-01   quote: "bond-month observations on 81,491 U.S. corporate bonds (8,159 per month on average) by 9,709 issuers from January 1973 to June 2011"   page: 2
sample_end:              value: 2011-06   quote: "bond-month observations on 81,491 U.S. corporate bonds (8,159 per month on average) by 9,709 issuers from January 1973 to June 2011"   page: 2
universe_filter:         value: "IG + NIG U.S. corporate bonds merged from Lehman, DataStream, Bloomberg (quote-based) and TRACE, FISD (trade-based), 1973–2011; per-database eliminations of preferred shares, non-USD, unusual coupons, warrants, MBS/ABS, convertibles, unit deals; obvious data-entry errors removed (as stated)"
                         quote: "We eliminate preferred shares, non-U.S. dollar denominated bonds, bonds with unusual coupons, bonds with warrants, bonds that are mortgage backed or asset backed, convertible, or part of unit deals."   page: 5
claimed_headline_metric: value: {mean: 0.37, t_stat: 3.90, unit: pct_per_month}
                         quote: "past six- month bond winners outperform losers by 37 basis points (bps) per month (t-value of 3.90) over a six-month holding period"   page: 2
```

## Self-checks

1. **Tag audit.** STATED×17 all quote+page; UNKNOWN×20 all with searched notes; zero INFERRED; zero
   DESIGN.
2. **Rulebook cross-check** (vs `docs/quant/specs/BBW_anchor_implementation_spec.md` §5.2): J=6 over t−6..t−1 ✓ ·
   skip 1 ✓ · H=6 staggered/overlapping ✓ · deciles ✓ · EW ✓ · leg winners−losers (P10−P1) ✓ ·
   headline 37bps/3.90 vs engine's quoted "+0.30%/mo attributable to winsorization" (different
   claim, no conflict). **Divergence (real data): trim OPERATION — paper says 'eliminated'
   (truncate); engine adjudicated clip/winsorise from DRR's FilterClass.** Paper wins in the gold;
   the engine's §7.1 adjudication is a DESIGN call downstream.
3. **Locator backfill.** Quotes verified as exact L1 substrings of the parsed PDF
   (`locator_backfill_report.md`); re-run against the frozen canonical text when this paper is
   frozen (PENDING-FREEZE).
