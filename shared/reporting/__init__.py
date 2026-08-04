"""shared/reporting/ — reusable, deterministic reporting primitives (docs/reporter/reporter_spec_v0.2.md §8, D2).

The claim-ledger types (`ClaimSpec`, `ClaimRecord`, `EvidenceBlock`, `SourceLocator`)
and the canonical-JSON contract live here rather than under `agents/reporter/` so a
future corpus-level reporter ("Scope C") can reuse them without a package move. Nothing
here computes a research quantity; these are typed carriers and a serialiser.
"""
