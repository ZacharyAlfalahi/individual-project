# Task

You are given the full text of a published corporate bond research paper. Write
a complete, self-contained Python script that implements the strategy named
below, as the paper describes it, on the provided bond-month panel and outputs
its monthly long-short portfolio return series.

Strategy to implement: {{TARGET_LABEL}}

Rules:
- Implement EXACTLY what the paper states for this strategy. Where the paper is
  silent, make the most standard choice and say so in a code comment.
- Use only the Python standard library, pandas and numpy.
- Read the panel from the parquet file at the environment variable
  `PANEL_PATH`. Do not read or write anything else except the output.
- No network access. No randomness.
- Return your answer as ONE fenced Python code block and nothing else.

## Paper text

<<<
{{PAPER_TEXT}}
>>>

## Panel schema

{{PANEL_SCHEMA}}

## Output contract

{{OUTPUT_CONTRACT}}
