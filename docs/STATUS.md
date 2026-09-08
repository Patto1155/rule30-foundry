# Status — what is in flight

**This is the only file in the repo that describes current state.** What the
repo *knows* lives in [`CLAIM_LEDGER.md`](CLAIM_LEDGER.md); what is *next*
lives here. Overwrite this file in place; git history keeps the old versions.
No other file may carry a "current state as of" section —
`tools/lint_ledger.py` enforces it.

Updated: 2026-09-08 · Newest log: `docs/experiment-logs/2026-09-01-nersissian-end-to-end-audit.md`

## Where the three prize problems stand

| # | Problem | Best current result | Grade |
|---|---|---|---|
| 1 | Does the center column repeat? | No period `p <= 5,000,000` in the first 10M bits — **decided exactly**, all 9,999,936 candidates, 0 survivors. Cannot resolve the problem: eventual periodicity is asymptotic. | Certificate |
| 2 | Is there a shortcut for the nth bit? | None found. `s*(n)` minimal-DFAO curve certified to n=48 with DRAT proofs. ML routes (I/K/L) are **scoped down** — blind to long-lag XOR structure. | Certificate (s*(n)) |
| 3 | Are 0s and 1s equidistributed? | Bias < 0.05% over 10M bits. Uniform Bernoulli(1/2) is invariant (proved, left-permutivity) — which is *not* the same as the single seed's limiting frequency, the actual question. A published shortcut claim is now under audit, with the warm query separated from the cold `n -> c_n` cost. | Theorem + Observation |

## Open work, ranked

Sequence approved 2026-09-02: **A1 → A2 → C2**, with B3 as filler. Reasoning
and the full option analysis: [`handover/CURRENT.md`](handover/CURRENT.md).
A1 and A2 are done (see *Recently closed*). **C2 has an open PR** — [#23](https://github.com/Patto1155/rule30-foundry/pull/23) claims to
close C2 *and* C2b — so the next decision is reviewing and landing that, not
starting the work.

| # | Work | Kind | Cost | Blocks |
|---|---|---|---|---|
| A3 | **Make bitstreams reachable** (Release asset or committed prefix) | Infra | ~½ day | B2 |
| C2 | **Algebraic annihilator search** — low-degree GF(2) relations over `w`-bit windows, via monomial-matrix rank. **Open as [#23](https://github.com/Patto1155/rule30-foundry/pull/23)**, awaiting review, not awaiting work | Research | Days | — |
| B3 | Extend `s*(n)` past n=48 — re-costed ~28× cheaper | Research | Hours | — |
| E1 | Write up the eight Theorem rows; `s*(n)` is citable | Writing | Days | — |
| B2 | Exact period search on 46M — no code change, extends to `p <= 2.3e7` | Research | Minutes | A3 |
| B1 | Item 14 pattern-map walk — palate cleanser, **not prize progress**. Queued as `queue/b1-pattern-map-walk.json` and passes preflight, but **has never been run**: it is the cheapest thing that would show the gates working on a real task | Research | ~26 min CPU | — |

**De-prioritised:** more neural experiments (the ceiling is partly the models'
— I/K/L are blind to long-lag XOR). Item 14 is worth closing but the ledger
grades left-edge structure as disjoint from the prize object.

## Chores

- **Delete 7 merged branches** — verified safe, blocked from the container by
  an egress `HTTP 403`. Command in [`handover/CURRENT.md`](handover/CURRENT.md).
  Then enable auto-delete head branches.

## In-flight branches

| PR | Branch | State |
|---|---|---|
| [#23](https://github.com/Patto1155/rule30-foundry/pull/23) | `claude/rule-30-foundry-env-tlbr9x` | Open. **C2 + C2b**, not A2 — this row said "A2: the `verify` workflow" long after A2 landed. Claims no GF(2) annihilator of degree <= 3 over 64-bit windows, nor degree <= 4 over 32-bit, with a self-verifying certificate. Based on `c64ff83`, which `main` is now five commits past, so it needs the base merged in before it can land. |
| [#24](https://github.com/Patto1155/rule30-foundry/pull/24) | `claude/rule-30-annihilators-phvmz7` | Open since 2026-09-03, based on `c64ff83`. "Add the council: dispatch briefs to external reviewer models" — **apparently superseded by #25**, which landed the council on 2026-09-04. Worth closing rather than rebasing, but that is the author's call. Note the branch name and the title disagree, so check which it actually is first. |
| [#30](https://github.com/Patto1155/rule30-foundry/pull/30) | `claude/recent-prs-review-llfk1b` | Open, based on current `main`. Commits every delegated task's report into its own branch. **If it also edits this file, it and this PR will collide** — the standing hazard in *Why #20 and #21 were merged* below. |

The #18 -> #19 stack landed on 2026-09-02; #20 and #21 had already been merged
into #19. **The three-deep stack is gone**: #27, #28 and #29 all landed within
thirty seconds on 2026-09-08, in bottom-up order, which is what
[`BRANCHING.md`](BRANCHING.md) §2's remedy prescribes. Nothing is stacked now
— the three open PRs are siblings on `main`, though #23 and #24 are based on a
commit five behind it. The 7 merged branches listed in
[`BRANCHING.md`](BRANCHING.md) are still awaiting deletion (see *Chores*), and
#27-#29's branches now join them.

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

- **The delegation stack**: #27, #28 and #29 all landed on 2026-09-08. `main`
  now carries `tools/gates.py`, `tools/workhorse.py`, `tools/codex_worker.py`,
  `tools/providers.py`, `tools/worker_pool.py` and `tools/agent_loop.py`, with
  [`WORKHORSE.md`](WORKHORSE.md), [`CODEX_WORKER.md`](CODEX_WORKER.md),
  [`WORKER_POOL.md`](WORKER_POOL.md) and [`AGENT_LOOP.md`](AGENT_LOOP.md).
  `CLAUDE.md` gained the *Delegation* section that routes work between them.

  The through-line is that **grunt work goes to an outside model and the
  repo's rules are enforced as code, not as prose**. `gates.py` turns
  CLAUDE.md's three expensive rules into preflight and postflight checks a
  runner cannot bypass; the `gates-trap` stage in `verify_all` asserts the
  counting-bound gate still refuses a known-vacuous manifest, so a gate that
  stops gating breaks the build rather than going quiet.

  **What has and has not been exercised.** The council answered a real review
  on 2026-09-04 and reproduced `counting_bound.py`'s own verdict on a planted
  vacuous negative. The worker pool ran live against both providers. But
  `workhorse.py --agent codex` has still never implemented an experiment, and
  `queue/b1-pattern-map-walk.json` — the one queued real task — has not been
  run. The pilot that would settle whether any of this is trustworthy is B1
  and B2 on hardware already owned, with traps planted, checking the
  validation layer catches every invalid result. Until that runs, the
  machinery is built and argued for rather than demonstrated.

  **No hardware bought, and none justified.**
  [`COMPUTE_PLAN.md`](COMPUTE_PLAN.md) §1 is titled *"The premise for renting
  was wrong"* and §5 puts all of §3 under about $20; the GPU simulator still
  does not checkpoint, which makes spot instances actively wrong.
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
