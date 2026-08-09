"""one-shot holdout — the one-shot holdout builder & evaluator (SC-SCI-12/13 machinery).

The single code path that ever reads ``data/holdout/`` (only through the ``holdout.py``
single-access gate). Everything here exists to make that one execution boring:
mechanical window derivation, a fail-loud pre-run checklist, an append-only marker
state machine, a pre-registered failure protocol, and a development rehearsal that
gates the real run. Implementation + rehearsal-green is where this stops — the real
one-shot executes only on an explicit written go-ahead, and is additionally blocked
on G6 survivors, the P3 moderate-prior artefact, and the E9 cost decision (spec §7).
"""
