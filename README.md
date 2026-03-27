# rule30-foundry

GPU-heavy local research and benchmarking workbench for the Wolfram Rule 30 prize problems.

## Project Goals

- Use the three official Wolfram Rule 30 prize problems as the core subject of the project.
- Approach the problems with code, not just informal mathematical discussion.
- Maximize useful local GPU usage as part of the work.
- Use the project as a serious first benchmark of the PC after cleanup and setup.
- Prefer a local-first workflow, including the option to use `llama.cpp` with `qwen-2.5-7b-instruct` as a helper model.
- Focus compute on exact search, large-scale scans, analysis, and hypothesis generation rather than brute-force simulation for its own sake.

## Research Discipline

Empirical work is not enough by itself. Each major compute effort should aim to produce at least one of:

- a counterexample,
- a precise conjecture strong enough to try to prove,
- or a reduced formal statement that can be attacked with symbolic methods.

This repo should avoid vague outputs like "looks random" unless they are attached to exact artifacts that can drive a proof-oriented next step.

## What the GPU Is For

The GPU should be used where it actually adds value:

- exact long-run Rule 30 simulation when throughput matters,
- block counting and frequency analysis,
- autocorrelation and correlation scans,
- candidate-period rejection,
- batched searches over recurrences, predictors, circuits, or transducers,
- local LLM-assisted experiment triage if that improves the workflow.

The goal is not just to run bigger jobs. The goal is to use local compute to generate proof-relevant artifacts.

## Success Criteria

This project is successful if it does both of the following:

- meaningfully stresses and benchmarks the local machine, especially the GPU,
- produces outputs that could plausibly contribute to a real attack on the Rule 30 prize problems.

---

## Workspace Layout

- `docs/` — research notes, problem statements, experiments, and references
- `docs/problem-statements/` — structured writeups of each target problem
- `docs/idea-bank/` — short candidate ideas, heuristics, and dead ends
- `docs/experiment-logs/` — dated experiment notes and results
- `docs/references/` — source notes, links, and citations
- `docs/templates/` — reusable markdown templates for consistent logging

## Working Rule

Keep each experiment small enough to summarize in one log entry:

1. State the hypothesis.
2. State the exact input or setup.
3. Record the output or observation.
4. Note whether the result changes the next hypothesis.

## Suggested Workflow

1. Pick one problem statement.
2. Add or refine an idea in `docs/idea-bank/`.
3. Run a bounded experiment.
4. Record the result in `docs/experiment-logs/`.
5. Promote anything useful into a problem statement or reference note.
