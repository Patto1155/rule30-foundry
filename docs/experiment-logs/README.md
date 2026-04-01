# Experiment Logs

Use two log styles:

- Canonical frontier logs: `docs/experiment-logs/{LETTER}_{slug}.md`
- Auxiliary runs or postmortems: `docs/experiment-logs/YYYY-MM-DD-short-slug.md`

Canonical lettered logs are for the main Rule 30 research thread.
Do not consume a frontier letter for a quick cleanup experiment unless you are intentionally changing the canonical experiment map.

Recommended fields:

- date
- goal
- setup
- commands or method
- observations
- conclusion
- next step

If the run depends on GPU or packed-bit code, include:

- verification against a naive reference
- any impossible-output checks used as invariants
- whether the result is exploratory, confirmatory, or just a cleanup/control run
