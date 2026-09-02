# Handover — Unified Fused Rule 30 Kernel (3 modes) + docs + experiments

**Repo:** `D:\APATPROJECTS\rule30-foundry` (git clone; the older local copy
`D:\APATPROJECTS\rule30-research` additionally has `tools/ca_observatory/`, a
FastAPI+React explorer — ignore unless asked. All shared scripts differ only by
CRLF.)

**Env:** GTX 1060 (SM 6.1), CuPy works; **PyTorch is CPU-only** here. Console is
cp1252 — run Python with `PYTHONUTF8=1` to avoid UnicodeEncodeError on prints.
Prefer `run_in_background:true` with a single `until grep -q DONE …; do sleep N; done`
waiter; do NOT spawn nested until-loops (they leak shells).

**Non-negotiable discipline:** correctness before claims. Every new GPU path must
be byte-verified against a trusted reference before any speedup is reported. The
gold reference is `experiments/rule30_open_utils.py` (its packed CPU path is
verified against a naive Rule 30 via `verify_*` helpers). Open boundary = phantom
zeros beyond the tape.

---

## Current state (what already exists, verified)

- `gpu/rule30_fast.py` — fused multi-step kernel, **single-tape center column
  only**. Verified bit-identical to reference; **185× faster** (30k steps:
  4320ms→23ms). Method: shared-memory halo tiling, K=48 steps/launch, m=ceil(K/64)
  halo words; inner tile stays clean because edge corruption spreads 1 cell/step,
  so safe while K < 64·m. See `docs/experiment-logs/2026-06-13-fused-multistep-kernel.md`.
- New experiments added this session (all run + logged): `damage_velocity.py` (Q),
  `left_edge_cone.py` (R), `linear_complexity.py` (S), `coarse_grain_search.py` (T).
  Naming caveat: `AGENTS.md` says canonical frontier is M/N/O/P and "don't assume
  next is Q" — Q/R/S/T jumped that; reconcile or relabel `aux_`.

---

## Task 1 — Generalize the kernel to 3 modes (1 unified kernel)

Replace the single-tape kernel in `gpu/rule30_fast.py` with ONE batched kernel
`rule30_multistep` taking two output flags:

- `write_center` → emit the center bit every substep. Output `[n_variants, n_steps+1]`.
- `write_rows`   → emit every block's inner-T words every substep to a spacetime
  buffer. Output (per variant) `[n_steps, n_words]` packed → unpack to `[n_steps, n_cells]`.

Add a variant dim via `blockIdx.y` (base offset `var*n_words`); center/rows
indexed with a per-variant stride. Pass 1-element dummy arrays for whichever
output is disabled (CuPy needs valid args; the flag guards the write). Keep
K=48, T=256, m=1.

Entry points:
- `simulate_center_columns_batch_fast(initial_rows, n_steps, center_cell)` → `[V, n_steps+1]`
- `simulate_center_column_fast(...)` → wraps V=1 (keep existing signature)
- `simulate_spacetime_fast(initial_row, n_steps)` → `[n_steps, n_cells]`

**Verify (must pass before reporting anything):**
1. center single & batched (random + spike + perturbations) == `simulate_center_columns_batch(gpu=False)`.
2. spacetime == `simulate_spacetime(gpu=True/False)` (which is naive-verified).
Extend the in-file `verify()` to cover all three; fail loudly on mismatch.

Note honest expectation: center-column modes win big (launch-bound); **spacetime
win is modest** (you must write O(area) rows regardless) and is capped by VRAM —
use the `base_step` arg to time-chunk large fields.

## Task 2 — Wire the fast path in (zero API change)

In `rule30_open_utils.simulate_center_columns_batch_from_packed`, at the top of
the `if gpu and GPU_AVAILABLE:` branch, lazy-import `rule30_fast` and delegate;
on ANY exception fall through to the existing kernel. This speeds up every
single-tape AND batched caller (period/bit/autocorr/linear-complexity/crypto +
causal/damage sweeps) with no signature change. Guard against circular import
(import inside the function). Re-run an existing experiment (e.g.
`linear_complexity.py --test`) to confirm identical output.

## Task 3 — Docs for future agents

Write `docs/GPU_KERNELS.md`: the kernel family, the K<64·m safety bound (with the
1-cell/step corruption argument), the 3 modes + APIs, coverage table (which
experiment uses which mode), the verification recipe, and the measured perf
table. Add a one-line pointer in `AGENTS.md` under a new "Performance" note so the
next agent finds it before re-simulating slowly.

## Task 4 — Run experiments

1. **Re-benchmark** all 3 modes (single center, batched center, spacetime) vs the
   old paths; put the table in `GPU_KERNELS.md`.
2. **Coarse-graining stronger-null follow-up** (the real open question). Current
   `coarse_grain_search.py` shows R30 coarse fields are ~0.20 predictable above
   marginal, but i.i.d. is too weak a null (any projection of a local
   deterministic field leaks). Add: (a) a **same-local-statistics null** — repeat
   the search on another chaotic ECA (rule 45, rule 90) by parameterizing the sim
   rule; if R30 ≈ 45 ≈ 90 the signal is generic local leakage, not reducibility;
   (b) an **r/b-scaling check** — does closure climb toward 1.0 as neighborhood r
   or block b grows (genuine coarse rule) or plateau (leakage)? Report verdict.
   Also note: sheared/cone-aligned supercells did NOT beat axis-aligned at b=2
   (my hypothesis failed) — don't re-pursue shear unless b≥3 changes it.
3. Optional speed: the b=2 enumeration is the real bottleneck (~226s, CPU/numpy),
   not the sim. Vectorize across projections (precompute a transition histogram
   over block-code 4-tuples once per shear, then score all LUTs by bincount on the
   histogram) before scaling to b=3 (where 2^512 projections force a learned/torch
   search instead of enumeration).

**Definition of done:** 3 verified kernel modes; fast path wired in with a
passing regression; `GPU_KERNELS.md` written; benchmark table filled; coarse-grain
verdict (trivial-leakage vs real) with the stronger null. Keep claims matched to
evidence — if a speedup isn't verified bit-identical, don't report it.
