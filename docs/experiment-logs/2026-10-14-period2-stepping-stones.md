# 2026-10-14 — period-two stepping stones

## Goal and setup

Build cheaply checked lemmas toward excluding eventual period two, replay
renewal mechanisms, and retain bounded evidence without promoting it to an
asymptotic claim. Base: main `4e1d746b6d02a65cbf45dec8cc5209986d9c4dac`
(merged PR #49). CPU only. Native subagents A–E use isolated worktrees;
subagent agreement is throughput, not independent corroboration. The lead
owns grades and integration. This build awaits a different-model-family audit
before merge.

## Decisions and premise corrections

- Read CLAUDE, STATUS, CLAIM_LEDGER, period2_attempt, and the period-ladder log
  before implementation. PR #49's review addendum controls the period-one
  grade: Proof candidate until a complete machine check or refereed proof.
- Git clone and `git lfs pull` succeeded. The initial clone was slow, not
  blocked; connector reads confirmed the same main commit and PR merge.
- Theory gate precedes B–E: `queue/results/period2-stepping-theory-gate.md`.
- Constant-zero control `r=0` must give zero completion weight. Condrey's
  nonzero-right completion is the linear-growth positive control.
- Sublinear infinite support does not give a finite alternating row. A finite
  prefix cannot refute existential `alpha(r), C(r)`; kill tests reject precise
  numerical bounds, recurrences, or other explicit candidate statements.
- The 740-state component and older 2,445-state component must be reproduced
  separately; their shared role does not establish identity.
- SAT no-fit queries run finite-row counting first and are exact exclusions,
  not evidence of complexity. Recurrence searches must likewise name the
  searched finite class.
- Agent A starts immediately. B–E start only after the theory-gate verdict.
  Agents never edit the ledger, STATUS, or this log.
- Replaced the absolute worker-pool 4.5-second deadline with a measured
  four-task serial baseline and a relative parallel threshold. The focused
  overlap test passes; it is never skipped.
- The first integrated full-suite attempt exposed an unintended Mathlib
  download in nested worker worktrees: elan was globally installed, but
  gitignored formal/.lake sources were absent there. That run was stopped
  before a verdict. Lean now explicitly SKIPs on absent source inputs and
  isolated workers name that permitted SKIP, as CI already does. The parent
  has the source inputs and actually builds/audits Lean. A focused test checks
  both source-absent and source-present cases. No failed build is hidden as a
  PASS; a missing input still checks nothing.
- Constructive L96 upper witnesses were found by exact enumeration of right
  integers below2^16: phase0 radius82/right287/left2894922832184770530049302;
  phase1 radius83/right929/left9508728737446193460969909. Counting first:
  log2|M|=16 vs length96, restricted-class exact minimization only, never
  complexity evidence or global minimum. C independently simulates the rows.
- Git and LFS push/dry-run fail because no GitHub credentials are available
  to Git. Connector writes can publish the reviewable tree, but provide no
  LFS-object upload. New proof objects will be preserved in a portable bundle;
  the draft PR must explicitly require LFS upload as well as cross-lineage
  audit before merge. Ordinary-blob fallback is deliberately not used.

## Initial verification

Baseline `python tools/verify_all.py`: **8 PASS, 3 SKIP, 0 FAIL**.
The initial absent inputs were canonical10M, canonical46M, and SAT toolchain.
Canonical10M was restored with `tools/convert_golden_bitstream.py`: manifest
SHA256 `6f8670b4a89826c8228d6a165047792e91551dedfb2853b8f12572d466b7547e`,
fresh CPU100k and naive4096 prefix agreement. SAT toolchain built with
`tools/build_sat_toolchain.sh` (CaDiCaL3.0.1 and drat-trim). Old retained
SAT proofs are being replayed; no replay verdict is inferred before completion.

## Results and integration

| Work | Reviewed result |
|---|---|
| A |19 sorry-free Lean declarations, complete axiom printouts and rejection tests. Local OR/NOR, finite/time-shift/onset, constant-one construction, ring7 orbit and recovered phase-slip identity checked. Full Condrey uniqueness/exclusion unfinished. |
| B |All65536 right seeds atN128;255 combined witness seeds at4096;256 sampled24-bit seeds at1024;18 selected finalists at100000. Ten long rows exactly share one occupied cell every7 depths and weight14286. Full saved-data/enumeration/sample/descent replay passes. |
| C |Combined phase0 exactL8..66, including sourced interpolation at65; phase1 exactL8..63. L96 checked brackets[54,82] and[54,83].1122 query records:625SAT,420UNSAT,77UNKNOWN. UNKNOWN remains UNKNOWN. Low-order rational/digit-affine guesses fail; arbitrary recurrence/k-regularity is not excluded. All CNFs, witnesses, DRATs, tamper cases and archive hashes checked. |
| D |Recovered K13526/basin21457/rank12, components9/29/2445, lifted small-component elimination, and45 physical credit-obstruction rows reproduced. Reported740 core is source-unavailable, not refuted. Seed32 is inQ outsideK on64 checked returns. |
| E |32 candidates generated:18 killed,14 bounded survivors,0 untested. Round1 uses64 seeds/depth256; round2 uses321 seeds/depth4096. Seven admitted open obligations retain exact proof filters. No survivor is a theorem. |
| F |Explicit gap chain and exact recovered depth7 checker. P7 (`7W_N(r)>=N-7bitlength(r)`) has no violation in16777216 tested inequalities (all65536 seeds, allN<=256), or128 deterministic96-bit seeds throughN4096. Rule90 separation remains a proof obligation. |

The sparse completion is not a new discovery: the recovered
`golden-mean-phase-slip.md` already explains its depth7 return conditionally
on the whole physical right-output word being0101. The new machine check
of its local coboundary does not prove spatial transport. Likewise the
finite ring does not prove that ordinary finite right seed32 has the
required boundary forever. The sharper universal P7 is an open conjecture;
after an exact seed32 theorem the minimum-seed comparison is equivalent to
P7 by integer rounding, not weaker. K-only gap3/5 charging misses the
observed exceptional-Q hard cases.

Lead-owned grades: checked foundation propositions are Theorem; finite
support/candidate/renewal/P7 checks are bounded Certificates; newly retained
horizon proofs are locally checked Certificates with explicit remote-delivery
blocker. Period one stays Proof candidate. Period two stays open. No grade
rests on subagent agreement, the unrecovered740 report, or an infinite
extrapolation from prefixes.

Lead replay commands and raw verdicts are in `queue/results/period2-*`.
The baseline539 DRAT records were also replayed before the new instrument.
The first completed integrated verifier failed seven promoted queue entries
using an unknown mechanism key, plus the nested worker verification of that
same queue. The controls were moved to a separate `proof_filters` field;
the gate itself was unchanged. All29 focused theory-workflow tests pass.
The failed run is retained as `queue/results/period2-failed-schema-verifier.txt`.

Final `python tools/verify_all.py --verbose --allow-skip
bitstream:center_col_46M.bin`:
**11 PASS,1 SKIP,0 FAIL**;759 tests run,6 named skips,0 failures. Raw verdict:
`queue/results/period2-final-verifier.txt`. The skipped stage is the absent
46M stream, and checked nothing. Inside unittest, the three46M subcase skips
are `CanonicalBitstreamTest.test_lsb_decode_matches_oeis`,
`test_msb_decode_is_exactly_the_per_byte_reversal`, and
`test_the_two_decodes_disagree_on_about_half_the_positions`. GPU/CuPy is absent
for `MeasureGpuPathTest.test_gpu_path_agrees_with_cpu` and
`GPUDispatchTest.test_gpu_path_skips_when_cupy_absent`/
`test_gpu_result_agrees_with_cpu`. Their names/reasons are retained in the
renewal verifier report. Worker-pool overlap ran and passed.

All48 new LFS objects (231015592 bytes) are preserved in the saved portable
`rule30-period2-proof-bundle.tar`, including a SHA256/path manifest and an
import script. The script checks bytes against the PR's exact pointers and
populates an authenticated clone's local LFS cache; upload remains a separate
`git lfs push`. GitHub connector publication supplies the reviewable tree,
but not these LFS bytes. The PR therefore remains draft and blocked on
authenticated LFS upload and a different-model-family audit before merge.

## Not run / not verified

- Full Condrey Theorems2–3 completeness/uniqueness and Corollaries4–5;
  generic recurrence-to-alternating formal equivalence beyond the explicit
  alternating-tail hypothesis.
- Infinite seed32 physical boundary, all-r density/P7 proof, unbounded
  support proof, spatial cut functional and uniform endpoint bounds,
  exceptional-Q exclusion, or eventual period-two exclusion.
- Every enumerated right seed and every SAT witness toN100000; this horizon
  was used for18 selected finalists only. No larger-depth universal search.
- Exact L96 minima;77 solver records remain UNKNOWN. No exhaustive recursive
  witness family, complete OEIS search, or full infinite k-regularity test.
- Later740-state/3106/3716 component and terminal-clearing replay: source
  commits absent from fetched refs. No fixed-weight cap6 extension or new LP
  discovery run; the named existing certificate was replayed.
- 46M stream/GPU/CuPy; outside-provider/referee/novelty audit; remote LFS
  upload and merge. Those are not reported as passes.

Next exact falsifier: minimize prefix slack
`min_{1<=N<=D}(7W_N(r)-N+7bitlength(r))` over adversarial finite right seeds,
starting with plateau witnesses and slow rows, withD>7bitlength(r). One
negative exactly reconstructed prefix kills P7, but not the original
existential-density target. A surviving numerical bound still needs an
OR-specific proof retaining physical lineage and spatial terminal credit.

## Addendum: cross-lineage audit (2026-10-14)

A session on a different model family audited this PR. No research claim
failed. It checked:

- **Lean:** independent `#print axioms` for all 19 declarations; Rule 30
  orientation; satisfiability of the `LeftOne` and `AlternatingTail`
  hypotheses.
- **Reimplementation:** a diagonal-recursion simulator that agrees with
  `completion()` on 300/300 random cases, with all 273 stored
  support-growth rows, and with the depth-100000 finalists.
- **Witnesses:** re-simulation of all 1,335 SAT witnesses, plus the 75 edge
  and 2 constructive witnesses. Record consistency and the L65 monotonicity
  argument also hold.

It could not replay any DRAT proof, because the 48 LFS objects were not on
the server.

Changes made in review:

- `tools/verify_lean.py` now catches `sorry` warnings, which Lean prints on
  stdout, and audits `private`/`protected` theorems. A planted
  `private theorem hidden_gap : False := sorry` now fails the gate.
- The Lean project imports Lean core only, so the unused Mathlib dependency
  is removed. The CI `drat` job now installs elan and runs the `lean` stage
  instead of skipping it.
- The ledger's P7 row is marked low-power: the `7·bitlength(r)` credit makes
  it testable only at tiny r. The ledger's Lean row now describes the
  phase-slip lemma as algebra only.
- New top obligation `period2-uniform-deficit`, `7W_N(r) >= N-8` for every
  r. It rests on the audit's search, where the largest deficit `N-7W_N` was
  8 and it never exceeded 4 for r<2^24.
