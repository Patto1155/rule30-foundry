# Handover — current

**Overwrite this file in place at the end of a session.** Do not add a new
dated handover file. Git history keeps every superseded version, and
`docs/handover/archive/` holds the five that predate this convention.

Keep the narrative sections under ~120 lines. The repo map below is fixed
reference and does not count against that. This file is context and reasoning
— the traps, the "why", the things that are true but not obvious. It is not
the claim record (`docs/CLAIM_LEDGER.md`) and not the work queue
(`docs/STATUS.md`).

Last updated: 2026-09-02 · Branch: `claude/agent-context-bootstrap` (PR #21)

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

## Agreed plan — sequence A1 → A2 → C2

Approved 2026-09-02. Full option analysis is in the session that produced this
file; the short version:

**A1 — land the PR stack. This is the bottleneck, and it outranks new work.**
Four PRs are open and `main` has none of it: `main` ← #18 ← #19 ← #20, with
#21 a sibling of #20. #19 alone carries 12,805 additions of certified results.
Work is being produced faster than it is merged, and every new branch has to
reason about which layer its files came from. Land #18 first.

**A2 — add CI running `tools/verify_all.py`.** There is **no `.github/`
directory at all**. The verification tooling is excellent and nothing runs it
automatically, so every check depends on an agent choosing to. ~20 lines of
YAML converts "agents should verify" into "agents cannot merge without
verifying." Highest leverage-per-hour item in the repo.

**A3 — make the bitstreams reachable** (do alongside A2). `data/center_col_*.bin`
are gitignored, so on any fresh machine three of the most important verify
stages report SKIP. Every containerised agent is checking a fraction of what it
believes it is. Publish `center_col_10M.bin` as a GitHub Release asset, or
commit a checkpointed prefix. This also unblocks B2.

**C2 — algebraic annihilator search. The research bet.** Rule 30's ANF is
degree 2. Search for low-degree algebraic relations over `w`-bit windows via
the rank of the monomial matrix (standard Courtois–Meier algebraic
cryptanalysis). Per the ledger this has **not been done**, it aims squarely at
Problem 2, and it is parity/algebra-native — so it sees precisely what the
neural experiments are blind to. It also fails informatively: rank deficiency
is a shortcut, full rank is a counting-bound-backed negative.

**B3 — extend `s*(n)` past n=48.** Good filler between the above. Re-costed at
~28× cheaper than originally planned (207 solves in 133 s standalone, vs 105 in
3747 s under the pysat harness). Extends an existing Certificate rather than
opening a new claim.

**E1 — write up the eight Theorem rows.** More valuable than it first appears.
The DRAT-certified `s*(n)` curve is a novel, citable artifact.

### Explicitly de-prioritised

- **More neural experiments.** The ledger states the ceiling is partly the
  models'. I/K/L are blind to long-lag XOR: the suite fails 6 of 6 budgets on
  `s[i-13] ⊕ s[i-27]` — fully determined by 27 bits inside a 64-bit context —
  while learning period-31 and `s[i-3] ⊕ s[i-5]` instantly.
- **Item 14 pattern-map walk.** ~26 min of CPU from a ledger row and already
  validated to `d = 5e7`, so run it as a palate cleanser. It is **not prize
  progress**: the ledger grades left-edge structure as disjoint from the prize
  object, since `settle(T) ≈ 1.34·T > T` means the center column is never in
  the settled region at any horizon.

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

## Outstanding chore

**Seven merged branches still need deleting.** All seven are verified ancestors
of `origin/main`; deletion failed from the session container with `HTTP 403`
(egress policy on delete-refs), so it must be run from a normal checkout:

```bash
git push origin --delete \
  agent/rule30-20260620T230002Z agent/rule30-20260622T230003Z \
  agent/rule30-20260624T230002Z agent/rule30-20260626T230002Z \
  agent/rule30-20260629T230002Z feat/dfao-min-state-curve \
  claude/branch-merge-strategy-smfppf
```

Then enable **Settings → General → Automatically delete head branches**.
Policy: `docs/BRANCHING.md`.

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
