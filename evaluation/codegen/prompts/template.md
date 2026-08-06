# Task

You are given the complete extracted specification of a published corporate
bond trading strategy (the same typed specification the project's compiler
consumes). Write a complete, self-contained Python script that implements
this strategy on the provided bond-month panel and outputs its monthly
long-short portfolio return series.

Rules:
- Implement EXACTLY what the specification states. Where the specification is
  silent, make the most standard choice and say so in a code comment.
- Use only the Python standard library, pandas and numpy.
- Read the panel from the parquet file at the environment variable
  `PANEL_PATH`. Do not read or write anything else except the output.
- No network access. No randomness.
- Return your answer as ONE fenced Python code block and nothing else.

## Strategy specification (typed extraction from the paper)

```json
{{GOLD_SPEC_JSON}}
```

## Panel schema

{{PANEL_SCHEMA}}

## Output contract

{{OUTPUT_CONTRACT}}
