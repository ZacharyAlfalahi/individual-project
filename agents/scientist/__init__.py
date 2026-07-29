"""
agents.scientist — the RQ4 Scientist agent (deterministic-spine foundation).

INVARIANT 1 (the wall) is enforced at IMPORT LEVEL for this package (R2): NO module under
`agents/scientist/` may import a magnitude-bearing Auditor schema
(`agents.auditor.schemas.{audit_report,audit_core,decomposition}`,
`agents.auditor.checks.{economic,inference,bayes}`). The single seam that turns a magnitude
into a verdict lives in `shared/handoff/scientist_case.py` — outside this package on purpose.
Toggle *identities* (`agents.auditor.schemas.toggle`) carry no magnitudes and are a permitted,
required reuse. The wall is checked statically by `tests/unit/test_scientist_wall_import.py`.
"""
