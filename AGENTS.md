# Agent Notes

Use this workspace as a disciplined scratchpad, not as an ad hoc notebook dump.

## Before you touch anything

```bash
python tools/verify_all.py
```

One command, ~2 s on a fresh clone. It runs the golden self-test, the hash
manifest, the manifest-is-current check, each canonical bitstream against the
independent reference, both lints, and the test suite, and prints PASS / FAIL /
SKIP per stage. Run it before an experiment and again before you commit.

`SKIP` is not `PASS`. The canonical bitstreams (`data/center_col_*.bin`) are
gitignored, so on a machine that has not regenerated them those stages check
nothing at all - which is exactly how a bad bitstream would go unnoticed.

Two standing conventions the tooling now enforces, both of which cost this repo
real months:

- **`bitorder='little'` for every `data/center_col_*.bin`.** `gpu/rule30_sim.py`
  writes LSB-first; NumPy defaults to MSB-first. A bare `np.unpackbits(data)`
  returns the true stream with every 8-bit block reversed - 49.95% of positions
  differ while the bit mean is *identical*, so no aggregate check catches it.
  `tools/lint_bitorder.py` rejects a bare call; annotate a genuinely
  order-agnostic one with `# bitorder-exempt: <reason>`.
  `tools/gen_golden_reference.py` is MSB-first by deliberate, documented
  exception - its independence is the whole point. Do not "fix" it.
- **LF line endings for everything under `data/`.** `data/** -text` in
  `.gitattributes` stops git normalising them for you, so `csv.writer` needs an
  explicit `lineterminator="\n"` and `Path.write_text` needs `newline=""`.
  `tests/test_manifest_determinism.py` fails the build if either is forgotten.

A ~50% bit difference between two streams is **never** a kernel bug. It means
they are uncorrelated: a packing or seed mismatch. A real kernel bug diverges
*late*.

**New here? Read
[`docs/handover/2026-08-29-data-integrity-and-dfao-curve.md`](docs/handover/2026-08-29-data-integrity-and-dfao-curve.md)
first** - it is the most recent state of play: what is verified, what was
retracted and why, and what is open. Then the ordered work plan in
[`docs/handover/2026-08-30-next-session-plan.md`](docs/handover/2026-08-30-next-session-plan.md).
Then `docs/AGENT_QUICKSTART.md`, then `docs/WORKFLOW.md`. The quickstart gives agents the tool map and prize-facing
triage rules; the workflow file is the operating manual for the GPU-accelerated,
verification-first loop.

## Before Proposing Experiments

Read these first, in order:

1. `docs/idea-bank/theoretical-reframe-2026-03-28.md`
   Why A-L hit a ceiling and why more "looks random" tests are low value.
2. `docs/theory/README.md`
   Read this before proposing theory-motivated work. Check whether it exists.
3. `docs/experiment-logs/README.md`
   Current log and naming conventions.

Key instinct: "Does this experiment answer a question that theory says is answerable, or is it just another random-looking test?"

If it is the latter, step back and consult the theory docs before proposing anything.

## Current Frontier

Canonical frontier:

- `M`: causal sensitivity / dynamical geometry
- `N`: column mutual information / transfer entropy
- `O`: 2D fractal or spacetime-complexity analysis
- `P`: invariant measure / entropy-rate work

Current state as of `2026-04-01`:

- `M` has been run with a corrected packed-bit implementation. See `docs/experiment-logs/M_causal_sensitivity.md`.
- Column MI / TE has also been run via `experiments/column_mi.py`.
- `O` (2D fractal) and `P` (invariant measure) are still open.
- Auxiliary cleanup experiments `fft_autocorr.py` and `compress_probe.py` have also been run.

Important naming caveat:

- A prior session introduced filename drift: `docs/experiment-logs/N_compress_probe.md` and `docs/experiment-logs/O_column_mi.md` do not match the original theory-driven `M/N/O/P` mapping.
- Do not assign new experiment letters casually.
- Do not assume "next experiment is Q" unless the canonical mapping has been reconciled first.
- If a session adds an auxiliary experiment, prefer a dated filename or an `aux_` slug instead of consuming a frontier letter.

## Defaults

- Put concise problem framing in `docs/problem-statements/`
- Put speculative approaches in `docs/idea-bank/`
- Put experiment results in `docs/experiment-logs/`
- Put citations or source links in `docs/references/`
- Put reusable markdown templates in `docs/templates/`

## Experiment Naming

Canonical frontier experiments should use:

- Log files: `docs/experiment-logs/{LETTER}_{slug}.md`
- Scripts: `experiments/{slug}.py`

Auxiliary runs, cleanup experiments, postmortems, and implementation notes may use dated filenames:

- `docs/experiment-logs/YYYY-MM-DD-short-slug.md`

Do not reuse a canonical frontier letter for an auxiliary experiment.

## Logging Standard

For each experiment log, capture:

- date
- goal
- setup
- result
- interpretation
- next step

If the run depended on GPU code or packed-bit kernels, also record:

- what was verified against a naive reference
- what sanity checks were used to reject impossible outputs
- whether the result is a direct consequence of the local rule or a higher-level empirical finding

## GPU Kernels

Simulation runs on two kernels: a trusted per-step reference (`rule30_batch_step`
in `experiments/rule30_open_utils.py`) and a fused multi-step fast path
(`rule30_multistep` in `gpu/rule30_fast.py`, ~K-fold fewer launches, three output
modes). The fast path is wired in transparently and falls back to the reference
on any error. **Before touching `K`, `T`, or the halo logic, read
`docs/GPU_KERNELS.md`** — it has the `K < 64·m` correctness argument and the
verification/benchmark recipe.

## Driving Experiments — `ca_lab.py`

For brute-force CA exploration, prefer the CLI over writing a one-off script. It
prints JSON to stdout (human table to stderr with `--pretty`) over the verified,
GPU-accelerated stack (`eca_sim` fields + `coarse_grain_fast` closure, 61× the
old loop). A 10-rule × 3-shear sweep runs in ~8 s.

```bash
python ca_lab.py sweep --rules 30,45,90,110 --shears 0,0.25,1 --steps 1200 --null
python ca_lab.py closure --rule 30 --shear 0 --steps 1600
python ca_lab.py sim --rule 110 --steps 400 --width 400
```

Coarse-grain reducibility ladder (b=2): linear rules 90/150/60/105 close at 1.0;
Rule 30 sits at the bottom (~0.72, tied with 45, just above the 0.52 noise floor)
— irreducible. See `docs/experiment-logs/2026-06-13-coarse-grain-same-statistics-null.md`.

## Implementation Guardrails

For bit-packed Rule 30 code, treat these as mandatory:

- Verify packed CPU/GPU kernels against a naive cell-by-cell implementation before trusting any result.
- Do not rely only on the first 20 center bits. Also test perturbation propagation across at least one 64-bit word boundary.
- Never infer geometric left/right from variable names alone. Confirm bit ordering and neighbor direction with a tiny naive test.
- In a radius-1 cellular automaton, `first_divergence < distance` is impossible. Treat that as a hard failure.
- "Never reached within N steps" is right-censored. Do not report it as "never" without qualification.
- If a metric is near zero, define a noise floor or baseline before calling it asymmetric or structured.
- Do not cite a number from `data/` without checking `docs/DATA_INTEGRITY.md` first. The March-vs-June kernel gap is CLOSED (the 10M bitstream is byte-identical across the fix), but experiments **I-L are known-bad** and pending a re-run.
- **Always pass `bitorder='little'` when unpacking `center_col_*.bin`.** `gpu/rule30_sim.py` writes LSB-first; numpy's default is MSB-first. Omitting it silently returns the stream with every 8-bit block reversed - ~50% of bit positions differ while the bit mean is unchanged, so aggregate checks will not catch it. This bug invalidated experiments I-L. The golden reference is MSB-first by deliberate, documented exception.
- Verify new packed kernels against `data/golden/center_col_golden_1M.bin` via `python tools/verify_data.py --bitstream <file>`, not only against a 20-bit prefix.
- If you add a file under `data/`, add it to `.gitignore` as an explicit `!` exception and rerun `python tools/make_manifest.py`. Do not `git add -f`.
- Do not hardcode absolute paths. 16 scripts under `experiments/` hardcode a path into `D:/APATPROJECTS/rule30-research/`, which is a **stale checkout of this same repository** (same git remote; its HEAD is a strict ancestor of ours). Make paths repo-relative; do not add a seventeenth.
- One-step TE at distance 1 may just restate the update rule. Do not oversell it as a novel discovery without literature support.

## Good Behavior

- Prefer short, atomic notes over long essays.
- Keep each experiment reproducible.
- If an idea fails, write down why it failed.
- If a result is ambiguous, record what would disambiguate it.
- Clean up transient progress logs or ignore them in git.
