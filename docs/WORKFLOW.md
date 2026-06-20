# Workflow Playbook (read this first)

This repo studies whether Rule 30 (and other elementary CA) admit a *shortcut* —
a coarse-graining / projection that predicts the dynamics faster than running the
CA. The work is GPU-accelerated and discipline-heavy: **correctness is verified
before any speed or science is claimed.** This file is the operating manual so a
new agent is productive immediately. Read `AGENTS.md` next for naming/logging
conventions and the experiment frontier.

## The loop

1. **Plan** the smallest change that answers a real question (see AGENTS.md
   "Before Proposing Experiments" — avoid yet-another looks-random test).
2. **Implement** on a branch (`feat/...`), never directly on `main`.
3. **Verify bit-exact against a trusted reference** before claiming anything. New
   GPU/packed paths must match a naive or previously-trusted implementation
   *exactly* (`np.array_equal`), including across 64-bit word boundaries and at
   the tape edges. The reference of record is `experiments/rule30_open_utils.py`
   (its packed CPU path is checked against a naive cell-by-cell Rule 30).
4. **Benchmark honestly.** If the reference now delegates to your fast path, force
   the slow path for the baseline (`RULE30_NO_FAST=1`). Report warm numbers and
   note run-to-run variance.
5. **Drive experiments via `ca_lab.py`**, not one-off scripts (see below).
6. **Log** the result in `docs/experiment-logs/` (date, goal, setup, result,
   interpretation, next step; plus what was verified for GPU work).
7. **Commit** per logical step with the verification result in the message. Push
   when asked; open a PR with `gh` when asked.

## Driving experiments — `ca_lab.py`

One JSON-emitting CLI over the verified, GPU-resident stack. `--pretty` adds a
human table on stderr; stdout stays pure JSON (pipe to `jq`).

```bash
python ca_lab.py sweep   --rules 30,45,90,110 --shears 0,0.25,1 --steps 1200 --null
python ca_lab.py closure --rule 30 --shear 0 --steps 1600
python ca_lab.py search  --rule 30 --b 3 --shear 0 --budget 300000 --msearch 80000
python ca_lab.py sim     --rule 110 --steps 400 --width 400
```

- `sweep` / `closure` — exhaustive b=2 closure (all 2^16 projections, GPU).
- `search` — b>=3 projection search (2^512 space; GPU population search).
- `sim` — field statistics for a rule.

## The compute stack (what to reuse, never re-derive)

| Layer | Module | Notes |
|---|---|---|
| Simulation (reference) | `experiments/rule30_open_utils.py` | trusted per-step kernels; CPU twin = ground truth |
| Simulation (fast, Rule 30) | `gpu/rule30_fast.py` | fused multi-step, 3 modes; see `docs/GPU_KERNELS.md` |
| Simulation (any rule) | `experiments/eca_sim.py` | arbitrary Wolfram rule, verified vs naive |
| b=2 closure (exhaustive) | `experiments/coarse_grain_fast.py` | 16^4 histogram + GPU contraction, 61x |
| general-b closure + search | `experiments/coarse_grain_bk.py` | uncapped; `closure_batch` (any b) + `search_projection` |
| CLI | `ca_lab.py` | the entry point you should be calling |

### GPU-resident pattern

Keep data on the GPU across the whole pipeline (sim → block codes → closure →
search); avoid host round-trips. Subsample (`m_max`) only inside an inner search
loop for speed; **re-score the final/best result on the full, uncapped transition
set** before reporting it. The `m_max=20000` cap in the old code was a leftover
from an O(M)-per-projection loop and is no longer needed for the histogram path.

## Scientific discipline (the part that makes results trustworthy)

- **Verify before claims.** A speedup or closure number means nothing until the
  path is byte-identical to a trusted reference.
- **Validity gate + null comparison are mandatory for any search/metric:**
  - A *positive control* must score as expected, or the tool is too weak to trust
    its negatives. For coarse-graining, the correct b>=3 control is a **shift rule
    (170/240)** — its coarse field is exactly `coarse[t+1,x]=coarse[t,x+1]` for any
    projection, so closure must reach ~1.0 at any b/r. (Linear rules 90/150 are
    **not** valid b>=3 controls — they are not exactly coarse-grainable at r=1;
    their perfect b=2 closure was special to 2x sublattice alignment.)
  - A *null* sets the bar. For "is Rule 30 special?", the null is rule 45 (chaotic,
    same local statistics), not just an i.i.d. coin.
  - **A search overfits finite samples**, so the null must be **searched at equal
    budget and M** — its searched closure is the floor to clear (~0.52 at b=3,
    M~40k). Enumeration (b=2) does not have this issue.
- **Radius-1 facts:** information moves at exactly 1 cell/step; `first_divergence
  < distance` is impossible — treat as a hard failure. "Never within N steps" is
  right-censored, not "never".

### Correctness lessons already paid for (don't repeat)

- **Boundary leak:** a fused multi-step kernel must re-pin out-of-tape halo words
  (`g<0` or `g>=n_words`) to 0 *every* step, or Rule 30 fills them and bits leak
  across the edge in ~2 steps. This was hidden by spike-only space-time tests
  (edges stay 0) and caught only by a **random IC**. Always test a random IC.
- **Wrong positive control:** see the shift-rule note above — choosing a control
  that *can't* hit the target in principle produces a false "tool is broken".

## Environment gotchas

- Run Python with `PYTHONUTF8=1` (console is cp1252; prints crash otherwise).
- GPU: cuPy `RawKernel` (NVRTC) works out of the box. cuPy **linalg** (`einsum`,
  `matmul`) needs cuBLAS — `nvidia-cublas-cu12` is installed; modules probe once
  and fall back to CPU if absent.
- The harness blocks foreground `sleep`. For long jobs use `run_in_background:
  true` and wait for the completion notification (don't poll); output is
  block-buffered until the process exits.
- Git may warn `LF will be replaced by CRLF` — benign on this Windows checkout.
- Two local clones exist: **`rule30-foundry` is canonical**; `rule30-research` is
  older and only additionally has `tools/ca_observatory/`.
