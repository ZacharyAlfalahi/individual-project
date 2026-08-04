"""agents/reporter/ — the deterministic Reporter agent (docs/reporter/reporter_spec_v0.2.md).

For one strategy run it emits a Markdown research note, a claim ledger binding every number
to its typed source, and rows in a canonical JSONL registry. No language model anywhere in
this build. It reads frozen/tagged upstream artefacts by IMPORT only and edits nothing under
`agents/librarian/`, `agents/quant/`, `agents/auditor/`. The reusable ledger primitives live
in `shared/reporting/` (D2); everything else is here.
"""
