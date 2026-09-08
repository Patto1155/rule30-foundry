# Handover — current

**Overwrite this file in place at the end of a session.** Do not add a new
dated handover file. Git history keeps every superseded version, and
`docs/handover/archive/` holds the five that predate this convention.

Keep the narrative sections under ~120 lines. The repo map below is fixed
reference and does not count against that. This file is context and reasoning
— the traps, the "why", the things that are true but not obvious. It is not
the claim record (`docs/CLAIM_LEDGER.md`) and not the work queue
(`docs/STATUS.md`).

Last updated: 2026-09-08 · Branch: `codex/period-word-sieve`

---

## Start here, in this order

1. `CLAUDE.md` — auto-loaded. Three hard rules and the health command.
2. `docs/STATUS.md` — the work queue. **What to do next lives there, not here.**
3. This file — why the queue looks the way it does.
4. `docs/theory/README.md` — the theory gate. What is already proved (do not
   re-measure it) and which routes are closed.

Then: `python tools/verify_all.py`. If it is not green, fix that first.

## The one-paragraph summary

GPU-backed empirical research on Wolfram's three Rule 30 prize problems. The
repo's real output is not "results" but **graded claims**: an eight-level
ladder from Observation to Theorem, enforced by tooling. A claim that cannot
be mechanically re-verified is worth little here, and one recorded above its
evidence gets retracted loudly (it has happened twice, and both retractions are
still in the ledger as warnings). Assume any impressive-sounding number you
find in a README is historical until the ledger confirms its grade.

## Why the work is sequenced this way

The bounded period-word sieve found no nonconstant exclusion: all 21 primitive
nonconstant words of lengths 2..6 have nonperiodic-neighbor loop certificates
at widths 3,5,7,9. Only constant controls synchronize. See
`docs/experiment-logs/2026-09-08-periodic-word-sieve.md` for the independent
audit and coverage. This supplies no composition law for an infinite family;
do not turn the finite method obstruction into a claim about the actual seed.

The September 2 plan prioritized merging the PR stack, adding CI, and then
running the annihilator search. Those steps landed; historical directions to
land #18, create the missing CI directory, or start C2 must not be replayed.
Use `docs/STATUS.md` for remaining work and `docs/CLAIM_LEDGER.md` for evidence.

The enduring lessons are about gates and the prize object. Annihilator rank
searches can be vacuous in both directions; consult the existing counting and
Reed-Muller gates before widening the search. Shortcuts are official Problem 3,
while balance is Problem 2. Random-looking center prefixes, low model accuracy,
and left-edge compression do not by themselves settle either problem.

The B1 recovery and new periodic-trace controls below make the same point:
find the condition that selects the actual single-seed orbit, rather than
assuming a locally feasible pattern or abstract map branch is that orbit.

## Traps that have actually bitten someone

- **`AGENTS.md` is not auto-loaded by Claude Code; `CLAUDE.md` is.** A rule
  that must always apply goes in `CLAUDE.md` or it is advisory only.
- **Bit order.** `gpu/rule30_sim.py` writes LSB-first, NumPy defaults to
  MSB-first. A bare `np.unpackbits` reverses each 8-bit block: 49.95% of
  positions differ while the bit mean is *identical*, so no aggregate check
  catches it. `tools/lint_bitorder.py` rejects bare calls.
- **A ~50% bit difference between two streams is never a kernel bug.** It means
  a packing or seed mismatch. Real kernel bugs diverge *late*.
- **`SKIP` is not `PASS`.** See A3.
- **Vacuous negatives.** Run `python experiments/counting_bound.py --pretty`
  before any "searched class `M`, found no fit" experiment. If
  `log2|M| < n` the negative is guaranteed. A certificate was retracted in
  2026-08 for exactly this, and its own random control had returned the *same*
  negative — which is a red flag, not a reassurance.
- **Vacuous controls.** A negative control that passes while testing nothing is
  worse than none. The DRAT self-test originally truncated a proof of an
  instance that was UNSAT by unit propagation, so the checker accepted *any*
  proof, including an empty one.
- **`SystemExit` is not an `Exception`.** A module-scope `raise SystemExit`
  when pysat was absent silently dropped 23 tests past `except Exception`
  guards. Pinned by `tests/test_import_safety.py`.
- **The status lint is filename-driven.** `docs/STATUS.md` must cite the newest
  dated file in `docs/experiment-logs/`. Add a log dated later and the build
  fails until STATUS is updated. That is the mechanism, not a bug.

## Lessons from the September 8 recovery

The old seven-name branch deletion command was stale: those refs were already
absent. Nine other live refs were merged. They were deleted only after checking
each tip against main, using atomic deletion with exact-tip leases. Automatic
deletion after merge is enabled. Consult live refs, not a historical name list.

B1 was described as unrun, but a full result and log existed untracked in the
older local checkout. The raw result is now preserved with its original hash;
the exact producing revision is unknown. Fresh finite seed gates and local
event algebra pass, but the billions of transitions between nodes were not
replayed. Two abstract leaves double and eight do not before the cutoff; the
actual diagonal remains unresolved. See
`docs/experiment-logs/2026-09-08-b1-recovery.md` for reproduction and limits.

The periodic-trace work found another boundary-condition trap: every trace has
an initial left-half-line realization. Fixing a bounded seed patch does not fix
that whole half-line. The stronger follow-up gives two closed output loops at
each tested width, so alternating centers can coexist with nonperiodic neighbors
in those open strips. Fixing a complete right initial row removes their free
choices, but does not imply a periodic response. The missing condition is
compatibility with the actual left seed dynamics at the same onset. See
`docs/theory/periodic-trace-constraints.md` sections 6 and 7.

PR #32 landed the initial research/recovery branch; #33 added the annihilator
preflight dispatch. Its safety-margin refusal is a policy, not a theorem that
a kernel exists. Distinct-window counts still need verified input provenance;
the manifest declares them and the gate does not measure them.

---

## Repo map

```
rule30-foundry/
├── CLAUDE.md               Auto-loaded entry point. Hard rules + pointers.
├── AGENTS.md               Naming, logging standard, implementation guardrails.
├── README.md               Public face. Historical summary — ledger wins.
│
├── ca_lab.py               CLI for brute-force CA exploration. Prefer over
│                             one-off scripts. JSON to stdout, --pretty to stderr.
├── prize_lab.py            Exact center-column work: GF(2) recurrences, finite
│                             kernel lower bounds, DFAO SAT encodings.
├── run_all.py              Smoke-test runner for the old experiments (--test).
├── run_session.py          Session driver.
├── run_after_sim.py        Post-simulation pipeline.
│
├── docs/
│   ├── STATUS.md           ← THE WORK QUEUE. Current state lives only here.
│   ├── CLAIM_LEDGER.md     ← WHAT THE REPO KNOWS. 34 graded rows.
│   ├── BRANCHING.md        Branch/merge policy. Max stack depth 1.
│   ├── theory/README.md    The theory gate. Read before proposing theory work.
│   ├── AGENT_QUICKSTART.md Tool map + prize-facing triage filter.
│   ├── WORKFLOW.md         The operating loop.
│   ├── GPU_KERNELS.md      Read before touching K, T, or halo logic.
│   ├── COMPUTE_PLAN.md     Renting compute: bandwidth, not VRAM.
│   ├── handover/
│   │   ├── CURRENT.md      This file. Overwrite in place.
│   │   └── archive/        Five superseded handovers, pre-2026-09-02.
│   ├── experiment-logs/    46 logs. Dated ones are canonical; letters A–S are
│   │                         the original frontier series.
│   ├── problem-statements/ Formal framing per prize problem.
│   ├── idea-bank/          Speculative directions.
│   ├── theory/             Proofs, incl. finite-prefix counting bound.
│   └── templates/          Experiment-log template.
│
├── experiments/            53 scripts, one per experiment.
│   ├── rule30_open_utils.py    TRUSTED reference path for packed Rule 30.
│   ├── eca_sim.py              Verified arbitrary-Wolfram-rule simulator.
│   ├── counting_bound.py       RUN BEFORE ANY "no fit found" EXPERIMENT.
│   ├── period_search_exact.py  Exact exhaustive period scan (Certificate).
│   ├── dfao_drat_proofs.py     s*(n) certification, both bounds.
│   ├── diagonal_recursion.py   O(1) pattern map (Certificate verifier).
│   ├── pattern_map_walk.py     Item 14. Validated to d=5e7, no ledger row yet.
│   ├── wedge_profile.py        Settled-wedge decomposition.
│   └── detection_power_probe.py  Measures what the ML suite can/cannot see.
│
├── gpu/
│   ├── rule30_sim.py       CuPy CUDA kernel. Writes LSB-first.
│   ├── rule30_fast.py      Fused multi-step fast path. Read GPU_KERNELS.md.
│   └── tape_geometry.py    Tape layout.
│
├── tools/                  Integrity layer. All of this is Tier 0.
│   ├── verify_all.py       ← ONE COMMAND FOR TRUST. Run before and after work.
│   ├── lint_ledger.py      Ledger citations + single-status-home + staleness.
│   ├── lint_bitorder.py    Rejects bare np.packbits / np.unpackbits.
│   ├── check_clone_integrity.py  Catches CRLF corruption from git checkout.
│   ├── gen_golden_reference.py   Independent CPU reference. MSB-first by
│   │                               deliberate exception — do not "fix" it.
│   ├── verify_data.py      Bitstreams vs the golden reference.
│   ├── make_manifest.py    Regenerates data/MANIFEST.sha256. --check in CI.
│   └── build_sat_toolchain.sh    Builds cadical + drat-trim into third_party/.
│
├── tests/                  13 modules, 115 tests. Detection tests, not just
│                             clean-repo assertions.
├── data/
│   ├── MANIFEST.sha256     Hash anchors. The integrity root.
│   ├── golden/             Golden reference bitstreams.
│   ├── prize/              Prize-facing artifacts.
│   ├── wedge/              Left-edge / diagonal artifacts.
│   └── center_col_*.bin    GITIGNORED. Absent ⇒ verify stages SKIP.
└── data-viz/               README SVG animations.
```
