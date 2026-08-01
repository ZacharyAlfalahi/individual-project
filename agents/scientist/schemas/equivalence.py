"""The proposal equivalence key (spec §4): [mechanism_id, template_id, canonicalised config
delta]. TWO consumers must agree on it — the Researcher's generation dedup (sources.py) and the
Experimentalist's G0 duplicate check (validator.py) — so it lives here, stated once, to guarantee
"duplicate" means the same thing on both sides."""

from __future__ import annotations

from .proposal import ExtensionProposal


def equivalence_key(proposal: ExtensionProposal) -> tuple:
    cd = proposal.config_delta
    return (
        proposal.mechanism_ref,
        proposal.template_ref,
        cd.conditioning_variable,
        cd.conditioning_lag_months,
        cd.interaction_form,
    )
