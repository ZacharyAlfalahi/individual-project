"""WS-A (P3 / SC-SCI-11) — reporter wiring tests: the posterior table renders
from fixture data with every numeric token claim-bound, the full note still
passes the five-layer verifier, and pre-amendment `HoldoutView` serialisation
is byte-stable (no `posteriors` key when absent, so recorded hashes hold)."""

from __future__ import annotations

from agents.reporter.bundle import HoldoutView, PriorPosteriorView
from agents.reporter.renderer import render_holdout, render_note
from agents.reporter.verify import verify_document
from shared.reporting.canonical import canonical_hash

from _reporter_fixtures import extension_bundle, make_bundle, make_proposal

_POSTERIORS = (
    PriorPosteriorView(
        prior_label="wide", prior_sigma=0.02,
        p_alpha_positive=0.9632, post_mean=0.008,
        post_ci_low=-0.0008, post_ci_high=0.0168,
    ),
    PriorPosteriorView(
        prior_label="sceptical", prior_sigma=0.0025,
        p_alpha_positive=0.71, post_mean=0.002,
        post_ci_low=-0.0021, post_ci_high=0.0061,
    ),
)


def _bundle_with_posteriors():
    base = extension_bundle()
    proposal = make_proposal(
        diagnostics={"alpha": 0.0012, "alpha_t": 2.4},
        holdout=HoldoutView(
            sharpe_sign=1, sharpe_ci_low=0.10, sharpe_ci_high=0.60,
            paired_difference=0.02, posteriors=_POSTERIORS,
        ),
    )
    return make_bundle(
        docs=dict(base.docs),
        stage_status={
            "extraction": base.stages["extraction"].status,
            "compilation": base.stages["compilation"].status,
            "execution": base.stages["execution"].status,
            "audit": base.stages["audit"].status,
            "scientist": base.stages["scientist"].status,
        },
        proposals=(proposal,),
    )


def test_to_dict_is_stable_without_posteriors():
    view = HoldoutView(sharpe_sign=1, sharpe_ci_low=0.1, sharpe_ci_high=0.6,
                       paired_difference=0.02)
    d = view.to_dict()
    assert "posteriors" not in d
    assert canonical_hash(d) == canonical_hash({
        "sharpe_sign": 1, "sharpe_ci_low": 0.1, "sharpe_ci_high": 0.6,
        "paired_difference": 0.02,
    })


def test_render_holdout_emits_claim_bound_posterior_rows():
    bundle = _bundle_with_posteriors()
    frag = render_holdout(bundle)
    assert "posterior re-expression under the pre-stated priors" in frag.text
    assert "never independent corroboration" in frag.text
    posterior_claims = [c for c in frag.claims if ".holdout.posterior." in c.claim_id]
    # 5 numeric tokens per prior row (sigma, P(alpha>0), mean, ci_low, ci_high).
    assert len(posterior_claims) == 5 * len(_POSTERIORS)
    for claim in posterior_claims:
        assert claim.displayed_value in frag.text
    # The rendered fragment attaches no qualitative adjective to the posterior.
    for banned in ("encouraging", "promising", "strong evidence", "suggestive"):
        assert banned not in frag.text.lower()


def test_full_note_with_posteriors_passes_the_verifier():
    bundle = _bundle_with_posteriors()
    doc = render_note(bundle)
    report = verify_document(doc, bundle)
    assert report.ok, report.errors


def test_note_without_posteriors_is_unchanged_against_fixture_baseline():
    # The pre-amendment fixture path must render and verify exactly as before.
    doc = render_note(extension_bundle())
    report = verify_document(doc, extension_bundle())
    assert report.ok, report.errors
    assert "posterior re-expression" not in doc.note_markdown
