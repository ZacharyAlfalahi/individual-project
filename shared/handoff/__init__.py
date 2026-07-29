"""
shared.handoff — the Auditor->Scientist seam (R2).

This package — and ONLY this package — is permitted to import the magnitude-bearing Auditor
schemas. It reads an `AuditReport`, applies the pre-registered entry rule, and emits a
magnitude-free `ScientistCase`. `agents/scientist/` may not reach across this line
(`test_wall_import_invariant`).
"""
