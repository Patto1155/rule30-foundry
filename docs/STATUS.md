# Status — what is in flight

**This is the only file in the repo that describes current state.** What the
repo *knows* lives in [`CLAIM_LEDGER.md`](CLAIM_LEDGER.md); what is *next*
lives here. Overwrite this file in place; git history keeps the old versions.
No other file may carry a "current state as of" section —
`tools/lint_ledger.py` enforces it.

Updated: 2026-09-09 · Newest log: `docs/experiment-logs/2026-09-09-seven-step-followup.md`

## Where the three prize problems stand

| # | Problem | Best current result | Grade |
|---|---|---|---|
| 1 | Does the center column repeat? | No period `p <= 5,000,000` in the first 10M bits — **decided exactly**, all 9,999,936 candidates, 0 survivors. Cannot resolve the problem: eventual periodicity is asymptotic. | Certificate |
| 2 | Are 0s and 1s equidistributed? | Bias < 0.05% over 10M bits. Uniform Bernoulli(1/2) is invariant (proved, left-permutivity) — which is *not* the same as the single seed's limiting frequency, the actual question. | Theorem + Observation |
| 3 | Is there a shortcut for the nth bit? | None found. Base-2 minimal-DFAO size is now certified through MSD `s*(56)=13` (LSD remains certified through n=48). No GF(2) annihilator of degree `<= 3` over windows to 64 bits, nor degree `<= 4` to 32 bits. ML routes (I/K/L) are scoped down. A published shortcut claim is under audit, separating warm query from cold `n -> c_n` cost. | Certificate (s*(n)) |

## Open work, ranked

Sequence approved 2026-09-02: **A1 → A2 → C2**, with B3 as filler. Reasoning
and the full option analysis: [`handover/CURRENT.md`](handover/CURRENT.md).
A1, A2, C2 and C2b are done (see *Recently closed*); **B3 is next**.

| # | Work | Kind | Cost | Blocks |
|---|---|---|---|---|
| A3 | **Make bitstreams reachable** — no local, release, Actions, or LFS copy found on 2026-09-09; manifest hashes remain anchored | Infra | ~½ day | B2 |
| C2c | **Annihilators over non-consecutive bit selections** — every window searched so far is a run of adjacent bits, so long-lag structure (the I/K/L blind spot) is untested | Research | Days | — |
| C2d | **Multi-word window codes**, lifting `w` past the `uint64` cap of 64 | Research | ~½ day | — |
| B3 | Extend `s*(n)` — MSD n=56 is now exact at 13 states; benchmark MSD n=64 next, then the harder LSD n=56 states individually | Research | Hours | — |
| E1 | Write up the eight Theorem rows; `s*(n)` is citable | Writing | Days | — |
| B2 | Exact period search on 46M — no code change, extends to `p <= 2.3e7` | Research | Minutes | A3 |
| B1 | All seven computationally reachable early zero words were tested. Phase plus one tail-boundary bit selects every successor (4 train, 3 held out), but computing that bit still depends on a transient cutoff growing to 117,323 (~1.335d). Settled-only rules fail; stop before the first period-32 branch at `d=1,420,878,969`. | Research | Proof target needed | — |
| P1-local | Actual seed rows reject most saved alternating-loop alignments before the center mismatch (5,574 exact starts tested), but this excludes only those walks. Period-word sieve still yields no nonconstant all-onset exclusion. | Research | Proof target needed | — |

**De-prioritised:** more neural experiments (the ceiling is partly the models'
— I/K/L are blind to long-lag XOR). Item 14 is worth closing but the ledger
grades left-edge structure as disjoint from the prize object.

## Chores

- **Branch cleanup completed.** The merged backlog was removed on 2026-09-08.
  On 2026-09-09 the twelve test branches were audited, their test-only content
  landed in PR #37, and the bad orbit-cycle source edit was excluded. All twelve
  obsolete refs were then removed atomically with exact-tip leases. Automatic
  deletion after merge is enabled. Unmerged `fix/rule110-showcase` is untouched.

## Delivery and running experiments

PRs #32–#38 are merged; all passed CPU and DRAT CI. They cover trace/B1
recovery, annihilator preflight, output loops, the period-word sieve, official
numbering, twelve imported test suites, and the seven-step follow-up with its
transient selector. No long
experiment or monitor is running. The older checkout's unrelated edits remain
untouched.

The #18 → #19 stack landed on 2026-09-02; #20 and #21 had already been merged
into #19. The three-deep delegation stack is gone: #27, #28 and #29 landed on
2026-09-08 in bottom-up order, which is [`BRANCHING.md`](BRANCHING.md) §2's own
remedy. Nothing is stacked. The merged-branch backlog has been cleared (see
*Chores*); historical branch names in [`BRANCHING.md`](BRANCHING.md) are not a
live deletion list.

### Why #20 and #21 were merged rather than left open

They were siblings on the same base, and each was green on its own — but
**together they failed**, and nothing either branch could run would have shown
it. `lint_ledger`'s `STALE-STATUS` check (added by #21) requires this file to
name the newest dated experiment log; #20 added a newer one. Two green PRs, a
red merge.

That is a standing hazard, not a one-off: *any* PR adding an experiment log
will red-line *any* concurrent PR that touches this file. The check is right
to exist — it is what stops STATUS drifting behind results — so it was kept as
is and the merge was resolved by updating this file, which is what should have
happened anyway. See [`BRANCHING.md`](BRANCHING.md).

## Recently closed

- **G1**: #33 wired the existing annihilator helper into preflight for positive,
  negative, and undeclared search claims. Strict distinct-window parameters and
  both boundaries are checked. A heuristic safety-margin shortfall is explicitly
  distinguished from a forced kernel. All 51 focused gate tests and both CI
  jobs pass. Schema: `docs/WORKHORSE.md`.
- **P1 bounded follow-up**: verified two-loop nonperiodic output constructions
  at seven widths. Finite right-completion checks expose the free-edge condition;
  the actual-seed snapshot probes do not establish an asymptotic invariant.
  `docs/experiment-logs/2026-09-08-periodic-trace-output-loops.md`.

- **The delegation stack**: #27, #28 and #29 landed on 2026-09-08 in bottom-up
  order. `main` carries `tools/gates.py`, `tools/workhorse.py`,
  `tools/codex_worker.py`, `tools/providers.py`, `tools/worker_pool.py` and
  `tools/agent_loop.py`, with [`WORKHORSE.md`](WORKHORSE.md),
  [`CODEX_WORKER.md`](CODEX_WORKER.md), [`WORKER_POOL.md`](WORKER_POOL.md) and
  [`AGENT_LOOP.md`](AGENT_LOOP.md).

  The through-line is that **grunt work goes to an outside model and the
  repo's rules are enforced as code, not prose**. The `gates-trap` stage in
  `verify_all` asserts the counting-bound gate still refuses a known-vacuous
  manifest, so a gate that stops gating breaks the build rather than going
  quiet.

  **What has and has not been exercised.** The council answered a real review
  on 2026-09-04 and reproduced `counting_bound.py`'s own verdict on a planted
  vacuous negative. The pool has run live against both providers and two model
  families. But `workhorse.py --agent codex` has still never implemented an
  experiment, and `queue/b1-pattern-map-walk.json` — the one queued real task —
  **has never been run**. The pilot that would settle whether any of this is
  trustworthy is B1 and B2 on hardware already owned, with traps planted,
  checking the validation layer catches every invalid result. Until that runs
  the machinery is argued for rather than demonstrated. No hardware bought and
  none justified: [`COMPUTE_PLAN.md`](COMPUTE_PLAN.md) §1 is titled *"The
  premise for renting was wrong"*, and the GPU simulator still does not
  checkpoint, which makes spot instances actively wrong.
- **#30**: three defects that each silently discarded paid-for work, all found
  by pointing the pool at a model family it had never run against. A task that
  changed nothing committed nothing, so `investigate` mode — which is forbidden
  from touching what it audits — left its report only in the gitignored `runs/`
  tree; three audits from the 15-task run were lost that way and are not
  recoverable. `max_tokens` was never sent, so OpenRouter defaulted it from the
  advertised context and the backend refused with HTTP 400 on turn zero. And
  the provider's real error was discarded in favour of a routing-level generic.
  Every task branch now carries `queue/results/<task_id>.json`.

  **Known limitation, unfixed:** `agent_loop.py` reads only `message.content`.
  A thinking model returns `reasoning`/`reasoning_details` as well, and
  OpenRouter wants the latter echoed back across tool-calling turns. One audit
  made 12 successful tool calls over 13 turns with every turn's `content`
  empty — the work was done and thrown away. Prefer non-thinking models for
  pool work until that is handled.
- **#24**: reduced from a competing council implementation to `briefs/` alone
  and landed. It and #25 built the council independently; #25's client won the
  add/add conflict, having been exercised end to end. What survived is the
  brief template, which `main` had no equivalent of and which states the
  failure that makes an outside reviewer worthless: *state the claim, not your
  confidence in it* — a brief asserting that a bound was correctly applied buys
  agreement, not review. That is `CLAUDE.md`'s independence argument about
  subagents, applied to the brief rather than the model.
- **C2 + C2b**: algebraic annihilator search — no GF(2) relation of degree
  `<= 3` over windows up to 64 bits, nor degree `<= 4` up to 32 bits, in all 20
  of 24 cells that clear both gates
  ([log](experiment-logs/2026-09-03-algebraic-annihilator.md)). The gate is the
  result worth remembering: this model class is vacuous in *both* directions,
  and the Reed–Muller ceiling `2^w - 2^(w-d)` voids every search at `w <= 22`
  regardless of the sequence. A first pass reported an apparent shortcut at
  `w = 20` that was a sorted-subsample artifact at parameters that were vacuous
  anyway; full-stream verification caught it. A second, worse defect was caught in **review**, not here: the golden input was decoded LSB-first when the golden files are MSB-first by documented exception, so the whole first grid ran on a byte-block-reversed stream at an unchanged bit mean of 0.500222. Reranked verdicts: none — every rank was full before and after. `load_bits` now verifies the decode against a naive center column instead of trusting a convention. Two routes closed alongside it:
  **space-time patches**, where the local rule is itself a degree-2 relation so
  the search succeeds by construction — all 6 rule instances lie in the kernel
  (0 violations), a forced positive. Whether the kernel is *only* the rule
  ideal is undetermined and no longer claimed, and
  **`w <= 22`** for any degree. Width turned out to be the cheap axis — `w=64`
  at `d=2` costs `D=2081` and 12 s, against `D=41449` for `w=32` at `d=4`.
- **Codex council**: #25 landed on 2026-09-04. `main` carries
  `tools/council.py`, the VM-side dispatcher under `tools/codex_dispatcher/`,
  and [`CODEX_COUNCIL.md`](CODEX_COUNCIL.md). The point is independence: every
  grade in the ledger is currently produced and checked by one agent lineage,
  and this puts a differently trained model on the same claim. It is not
  authority — a council answer does not promote a ledger row, and disagreement
  is the useful output.

  **Working end to end, verified 2026-09-04.** The startup probe confirms
  `--sandbox` and `--output-last-message` on codex 0.153.1, so reviews run
  sandboxed read-only and the answer comes from the exact last-message file
  rather than a parsed transcript.

  The first real review was a deliberately vacuous negative: "no DFAO with
  ≤24 states reproduces the first 10,000 center-column bits, therefore no
  finite automaton does, grade it Certificate." It rejected the grade on
  counting grounds, put `log2|M| < 254` against `n = 10,000`, and separately
  caught the quantifier error (≤24 states cannot support "no finite
  automaton"). `experiments/counting_bound.py --verdict 24:10000` independently
  gives `244.078` and `VACUOUS` — so the outside model's arithmetic and verdict
  both agree with the repo's own tool on a case the repo has been burned by.

  Operational detail worth keeping, because it is not what was assumed: the
  **egress allowlist updates live** in an already-running session — the proxy
  enforces policy, not the container — but the **environment variables only
  land on a container restart**. A session that can suddenly reach the host
  while `CODEX_COUNCIL_*` is still unset is in that intermediate state, not
  broken.

  Same PR added `.claude/agents/`: `counting-bound`, `theory-gate`, `verifier`,
  enforcing gates this repo mandates in prose and has never enforced
  mechanically. `CLAUDE.md` authorises their use and records why they are not a
  substitute for the council — they share a model lineage with whoever spawns
  them, so agreement among them is not corroboration.
- **A1**: the #18 → #19 stack landed on 2026-09-02. `main` carries Tier 0
  tooling, the Nersissian audit, and the agent context bootstrap.
- **A2**: CI added — `.github/workflows/verify.yml` runs
  `tools/verify_all.py` on every push to `main` and every PR. Two jobs: a
  CPU-only run, and a second that builds the SAT toolchain so the DRAT stage
  behind `s*(n)` genuinely runs (8 stages PASS there, against 7 without it).
  Both pass `--allow-skip` with an explicit stage list, so a stage that starts
  skipping because its input vanished fails the build instead of going green —
  the repo's "SKIP is not PASS" rule, enforced rather than documented.
  CI's dependencies are pinned in `requirements-ci.txt`: numpy alone, measured,
  not guessed.
- Golden reference closed byte-for-byte at 10M bits, three independent
  reproductions ([log](experiment-logs/2026-08-30-golden-reference-10M.md)).
- `s*(n)` promoted to Certificate, 207/207 DRAT-verified
  ([log](experiment-logs/2026-08-30-dfao-drat-certification.md)).
- Exact exhaustive period search
  ([log](experiment-logs/2026-08-30-period-search-exact.md)).
- I–L un-retracted but scoped down after a detection-power probe
  ([log](experiment-logs/2026-08-30-rerun-il-bitorder.md)).
- New `g(n)` smallest-grammar curve
  ([log](experiment-logs/2026-08-30-grammar-min-size-curve.md)).
- Nersissian shortcut claim audited end-to-end, separating the advertised warm
  `O(log n)` query from the cold-start `n -> c_n` cost that Problem 3 actually
  asks about ([log](experiment-logs/2026-09-01-nersissian-end-to-end-audit.md)).
  No `Omega(n)` established for the compressed representation yet — the next
  step is reconstructing it faithfully.
- `docs/COMPUTE_PLAN.md` D11 corrected: the fixed-`n` circuit definition was
  vacuous (a hard-coded constant), replaced by the whole index function `C(k)`.
