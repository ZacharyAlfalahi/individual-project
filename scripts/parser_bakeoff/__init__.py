"""
Parser & quote-locator bake-off (parser brief §7).

Two modules:
  * ``build_canonical_text`` -- PyMuPDF PDF -> L0 canonical text (§5 shape) with a
    double-parse determinism check.
  * ``run_bakeoff`` -- the fixture grid over the L0/L1/L2 ladder, the §2 decision
    rule, the report, and (on pass) the frozen recipe ``config/canonical_text.yaml``.

Deterministic, no LLM, no network at runtime (parser brief §9). The normalisation
ladder + matcher live in ``agents/librarian/config`` (``normalise`` / ``locate``);
this package is the offline experiment harness that consumes them.
"""
