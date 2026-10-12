# Rule 30 recovery and continuation handover

Historical inspection record: 12 October 2026 (Sydney). Prepared for an agent continuing Patrick's `Patto1155/rule30-foundry` project.

This is an archived handover, not a replacement for `docs/STATUS.md` or `docs/CLAIM_LEDGER.md`. It records what this session inspected and what earlier sessions reported. Reconcile those sources before research, merging, cleanup, or expensive computation.

## Objective and first action

Primary objective: mathematical progress toward Prize 1, excluding eventual periodicity of the centre column from the single-black-cell Rule 30 seed. Period-two exclusion is an intermediate target, not the complete prize.

**First recover the later local history. Do not start a new research round from the September GitHub snapshot under the assumption that it includes October work.**

1. Locate any surviving checkout, especially the historically used `/workspace/scratch/719aefb6dc61/rule30-foundry`. This inspection found no `rule30-foundry` directory under the available `/workspace/scratch` tree at depth four. Absence here does not prove the historical checkout is lost everywhere.
2. Inventory working trees, uncommitted changes, branches, reflogs and remotes. Preserve valuable tips before altering refs. Do not reset, replace, or delete an existing checkout to simplify recovery.
3. Search for `consolidate/research-and-tools` and the local commits listed below. Check actual objects and ancestry; short hashes in conversation records are recovery leads, not verified accessible objects.
4. If recovered, compare its tree with the published research branch, rerun verification and read its newer reports. If unavailable, explicitly record the missing history and proceed from the published branch with that limitation.
5. Read applicable `AGENTS.md`, `CLAUDE.md`, `docs/STATUS.md`, `docs/CLAIM_LEDGER.md`, `docs/handover/CURRENT.md`, `docs/THEORY_FIRST.md` and `docs/theory/README.md`. Run `python tools/verify_all.py` before experiments and before committing.

## Evidence boundary

**Inspected through GitHub in this session:** repository metadata; all ten published branches; their comparisons with main; all 220 commits reachable from main; main's status and ledger; research-branch status, September 20 transport report, agent instructions and ledger linter.

**Retrieved from prior conversation records, not independently replayed here:** September 21–October 8 local research, verification reports and consolidation. These reports are valuable but must not automatically promote ledger grades.

No new mathematical experiment or proof was performed during this audit. No branch consolidation, deletion or research merge was performed. The audit's original deliverable was a chat summary; this document makes its findings available to the next agent.

## Published refs inspected

| Ref | Exact tip | Relationship to inspected main |
|---|---|---|
| `main` | `b450406deace9f8adabc7b5b95d45b85e2c32432` | 220 reachable commits; latest commit 18 Sep 2026 |
| `research/slanted-cone-charge` | `44d11eebaab4e8486c195dee1e5c61a36860851f` | 1 commit ahead; published 20 Sep |
| `claude/lucid-dirac-d0h0rr` | `5efc6ef41ac3358f74f24e5fe6922af0634a3818` | 4 ahead, 0 behind |
| `claude/vibrant-darwin-eagjut` | `c5aaa0b74234cda3b6944632c7698e0220d88cc1` | 1 ahead, 19 behind |
| `claude/nice-babbage-1elmxd` | `2ed51a857f467edbd4880c2d7efdb497f8ad943b` | 1 ahead, 29 behind |
| `docs/new-idea-brief` | `973bebc8d50fcc8612c4c00e7eaeb63775be8278` | 1 ahead, 29 behind |
| `fix/rule110-showcase` | `fe86ed177ee72442c87068b8e6155c6bc877d572` | 1 ahead, 177 behind |
| Three `codex/investigate-*` refs below | `222c9ccd115af1ffebd6bf0b54f4d0b3f721fca8` | Each 0 ahead, 70 behind |

The three Codex branches are:
- `codex/investigate-audit-every-file-under-experiments-and-g-a46798ae`
- `codex/investigate-claude-md-rule-3-says-all-three-prize-pr-1b487ab4`
- `codex/investigate-tools-verify-all-py-builds-its-stage-lis-cec39bbe`

There were nine distinct additional commits across non-main branch histories, not nine independent research breakthroughs. No pre-handover published branch tip had a commit newer than September 20. Counts and refs are historical observations; refresh them on arrival.

The research tip is a **snapshot**, not the complete original local commit chain. Its commit message identifies source local commit `e5595b39f28b0a7da98790e32abd09e75e7fa214` and source tree `7659e91fa0c3f35992faeccb99c6bfa80064a0fe`. Compared with main it changes **125 files** and adds **19,651 lines**. One visible commit therefore contains many research rounds.

The lucid branch adds theory ordering/portfolio tooling and alternating-trace probes. All its changed filenames also appear in the research snapshot's changed-file list; **this is not a byte-for-byte equivalence check** and does not itself justify deleting or merging the branch.

The vibrant branch adds four structural probes: branch dependency, branch-summary pruning, wedge hybrid cost and seed/window invariants. Read `docs/experiment-logs/2026-09-15-four-structural-probes.md` on that branch. The nice branch adds Jen/Kopra literature notes; the idea branch adds `docs/NEW_IDEA_BRIEF.md`; Rule 110 contains an older animation fix. The three Codex tips are already ancestors of main; this does not establish that any uncommitted work elsewhere has been preserved.

## Later local work reported by prior sessions

| Date | Recovery pointer | Reported outcome; verify against recovered files |
|---|---|---|
| 21 Sep | Library `rule30-proof-strategy-audit-2026-09-21.md` | Transfer-map/renewal proof architecture audited. Proposed support-growth inequality remained unsupported. |
| 25–27 Sep | `22ab58f7`; handover commit `613b4011` | All-range additive-rank obstruction, sharper periodic-patch bound; 8 checks passed, 3 skipped. Clean and unpushed at reporting. All-onset ancestry criterion also explored; no future centre-trace restriction obtained. |
| 30 Sep | `d7eecd17`; report `2026-09-30-reconciliation-and-coherent-clearing.md` | Reconciled local/published history and claims; replayed tracked 10M certificate; proved adjacent terminal-clearing times differ by exactly one. 10 passed, 2 skipped (46M data and SAT/DRAT tools). Unpushed. |
| 30 Sep | Subsequent research discussion | Joint left–right splice reformulated finite-prefix reachability; fixed-ancestor gap remained. A phase error was caught; route parked, no new exclusion. |
| 4 Oct | `7733d4d`, reported based on main `b450406` | Atomic fsynced checkpoint/resume, corruption checks and locks. Kill/resume reproduced 1,000,003 bits exactly; 21 tests and 8 checks passed. Unpushed; 46M run unstarted at reporting. |
| 4 Oct | Adaptive selector session; no recovered commit identified here | Adaptive scoring, two-step lookahead and per-choice audit trail; 260 checked results replayed. Fixed/random beat adaptive under wall-time; median centre coverage matched. Publication was attempted, not established by the inspected branch inventory. |
| 8 Oct | `consolidate/research-and-tools`, `7dd93382` | Consolidated research/tools while preserving 11 source tips; reconciled claims and corrected finite-measurement overstatements. 9 passed, 3 skipped, 0 failed; four structural artifacts replayed; kill/resume reproduced 1,000,003 bits exactly. This branch was absent from the inspected GitHub inventory. |

An earlier prompt also identifies local `25d4d09a60df34e7ff884ca745b7bcf3547e16ef` on the research branch. The relationship among these local objects must be reconstructed from git, not inferred from session dates.

The October 8 review of `openai/math` reportedly recommended independent theorem specifications, compatibility preservation and bounded false-match testing. It produced workflow lessons, not a new Rule 30 proof.

## Mathematical orientation and scope traps

The promising but unfinished route uses the exact left transfer map `G = T_1 o T_0`, necessary modulo-four conditions, and physically coherent right-boundary renewal events. Left-only survivors or independently satisfiable finite prefixes are not full physical infinite orbits from one fixed finite ancestor.

The September 21 audit described a 740-state renewal core `D`, gap-three/gap-five inverse obligations, and a desired bound of the form `bitlength(y_m) >= h_m + 4*l_m - K(r)`. **That bound was unsupported.** Two exceptional components of sizes 3,106 and 3,716 were reported unresolved in that formulation.

The separately inspected September 20 forced-right analysis used a 16-bit invariant `K` with 13,526 states, a universal attraction basin of 21,457 states and entry rank at most twelve macrosteps. Exact lifts eliminated apparent components of sizes 9 and 29. It reduced persistent nonentry to a 2,445-state component `Q`, with eventual residence by macrotime at most 100 if entry never occurs. Finite-support persistence in Q remained open. **Do not identify Q with either later exceptional component merely from their roles; reconcile definitions and projections first.**

That report also established scoped obstructions to nonnegative width-three cut-density potentials, a restricted signed width-two family, and additive decreasing ranks through range eight. Later sessions report an all-range extension; retrieve its proof before relying on the extension. These obstructions do not eliminate arbitrary order-sensitive, growing-cut or cone-conditioned mechanisms.

All-gap event/threshold identities generalised the encoding of forced periodic centre words. A forced period-three right-half example defeated a naive generalisation of the alternating zero-run rank. It was **not** a full finite Rule 30 period-three orbit.

Adjacent terminal-clearing times differing by one closes a proposed strengthening; it does not supply the required initial-support growth. No reviewed/retrieved session claimed a new nonconstant centre-period exclusion.

Useful published pointers:
- `docs/experiment-logs/2026-09-20-transport-entry-phase.md`
- `docs/theory/right-invariant-entry.md`
- `docs/theory/weighted-renewal-transport.md`
- `docs/theory/zero-tail-rank-followup.md`
- `docs/theory/phase-word-transport.md`
- `docs/theory/transport-entry-hostile-review.md`
- September 19 logs on periodic fibres, predecessor coherence, zero-tail composition, multi-route transport and slanted-cone charge.

## Prize evidence and verification discipline

The inspected ledger includes the older exact 10M finite-period certificate (no prefix period p <= 5,000,000), and base-two MSD minimum DFAO size `s*(64)=15`. Neither finite certificate proves asymptotic aperiodicity or general computational irreducibility. No nth-bit shortcut is known; Bernoulli invariance is not the single seed's limiting-frequency theorem.

The right-special-factor route is an equivalent restatement of Prize 1, not an independent mechanism. Main's ledger separates the Condrey period-one preprint as an external proof candidate; do not silently treat it as a repo-certified theorem or as proof-assistant verification of its main claims.

Canonical `data/center_col_*.bin` files unpack little-endian; independent golden references deliberately use MSB-first. Check geometry, packing and first divergence before interpreting disagreements. A mean near 0.5 cannot validate a stream.

Use counting bounds before interpreting searched-class negatives as complexity evidence. Budget exhaustion means UNKNOWN. Finite-prefix reachability does not establish a single fixed ancestor supporting all horizons. Agreement among same-lineage agents is not independent validation.

Earlier reported checks are not fresh checks. September 20 publication recorded 8 passes, 3 expected skips (canonical bitstreams and SAT/DRAT toolchain). The 46M artifact/run must be located or reproduced; neither its recovery nor a new 46M run was established here.

## Continuation after recovery

Choose one precisely stated obligation with a mechanism and a falsifier. The surviving central question is whether actual finite-right renewal histories impose ordered predecessor obligations that force unbounded initial left support. Alternatively attack persistent exceptional-component residence using finite support, or develop a substantially different mechanism. Do not treat another local density search, a larger enumeration or another splice reformulation as automatically useful.

Before any consolidation, compare actual trees and claims, preserve source tips, run appropriate checks and resolve incompatible evidence. The earlier user approved consolidation work, but this audit did not merge branches or establish that the local consolidated branch was published.

## Suggested skills and tools

- GitHub connector: refresh refs, inspect files/PRs, recover accessible objects and publish reviewable changes.
- `handoff`: compact continuation records with verified references.
- Personal Context: retrieve earlier session reports when local history is missing; conversation reports remain secondary evidence.
- Library: retrieve the named September 21 audit and continuation documents if needed.
- Repository scripts: `tools/verify_all.py`, claim/bit-order lints and the existing focused reproducers.
- Independent review where warranted; preserve the distinction between model agreement and checkable proof.

## Checks performed while preparing this file

A fresh temporary checkout of the published research tip was obtained after the historical-checkout search. The handover was placed under the archived-handover directory, which the status linter explicitly permits. Claim and bit-order lints, integrity/golden/manifest checks and the gate trap passed; canonical 10M/46M bitstream stages and SAT/DRAT verification skipped for missing inputs/tools.

Two full verification invocations were mistakenly overlapped in the same checkout. Both completed 897 tests with ten unittest skips, but failed worktree-count/working-tree checks while the other invocation was mutating temporary worktrees. These are not clean full-suite passes. After both processes stopped, all three distinct affected tests were rerun serially and passed (3/3): the Codex worker untouched-checkout/no-leak test, worker-pool dry-run test and concurrent-worktree preservation test. No code was changed to repair them. Run the full gate serially on arrival; do not run separate suites against the same git metadata concurrently.

The document is an audit and recovery path; mathematical claims remain governed by recovered evidence and the claim ledger.
