"""
Librarian agent -- reads a corporate-bond factor paper and emits a
provenance-tagged ``StrategySpec`` (paper language only). Everything downstream
consumes it.

[P1] scope -- the static, testable substrate the extraction
machinery (later components) fills.

  * ``schema/``     -- the frozen ``StrategySpec`` dataclasses (SpecHeader,
                       Part1, Part2, Leg, Combiner) and the SignalRef value
                       object (D22). Every fact-bearing field is an
                       ``Inherited[T]`` (D6 -- no parallel ``Fact[T]``), reusing
                       the frozen provenance layer under ``agents.quant.config``.
  * ``validators/`` -- the tag-reason registry (D24), the closed-domain
                       declarations (D24 part 2), and the Librarian-output spec
                       validator enforcing the D8 negatives.
  * ``data/``       -- the tag-reason registry and domain tables as landed data
                       (never code).

The wall (D3). The StrategySpec speaks *paper vocabulary* only -- e.g.
``long_leg = lowest_signal`` with a located quote, never engine coordinates
(``long_group = 0``). The deterministic Quant-side adapter alone speaks engine
coordinates; nothing in this package imports the engine or its column names.
Consequently the Librarian never emits ``DESIGN`` (D8): the ``Inherited`` type
admits the tag, but ``validate_librarian_spec`` forbids it -- DESIGN is a
project decision made config-side, never a product of extraction. Signal
identity is a menu pick (D22): a ``SignalRef`` carries a ``concept_id`` from the
Signal Concept Registry (or the ``"unrecognised"`` escape), its parameters, and
the paper's own words -- but no column names (the concept->column table lives on
the Quant side of the wall).
"""
