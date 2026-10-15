# Column reduction and the finite language ladder

Date: 2026-10-15.

Goal: reproduce the supplied column-reduction claims, test relaxed and
physical deficit bounds, and close formal prerequisites without promoting
finite evidence into a period-two proof.

Setup: CPU only, pinned Lean4.19.0, CaDiCaL3.0.1, independent drat-trim,
NumPy and pinned python-sat cardinality encodings. Six implementation
agents used isolated worktrees; a theory gate preceded work, and a
separate static recombination review checked scopes and certificate replay.
The base is `research/period2-stepping-stones`, since its tip was not an
ancestor of main. GitHub API reads were denied, so no API merge state was
inferred from a failed CLI request.

Results and grade changes:

| Item | Result | Grade and limit |
| --- | --- | --- |
| Constant-trace fibres and eventual period-one exclusion | Lean completeness for Condrey Theorems2–3, finite nonzero evolution and arbitrary-onset constant exclusion; 50 total checked declarations including existing and column lemmas | Theorem, exact formal statements; period two remains open |
| Column reduction | Checkpoint weights9/10 and physical N190/W26/deficit8 reproduced; 200 random rows cross-checked and all65536 right seeds checked for no11/no00000 through even time80 | Exact finite checks; odd-left/no11 Lean checked, no00000 not Lean |
| Free-u minima | 20 certified minima N4..80 by4; N81 minimum9 follows from N80 lower9 plus supplied witness; no-11 N79 minimum10 independently certified; censored N300[9,43], N500[9,72] | Certificate; asymptotic density undecided |
| Physical factor language | Exact factor sets through12, all twelve minimal bans certified by local cones; counts2,3,5,8,12,17,25,36,50,68,91,119 | Certificate; no regularity or all-length characterization |
| L1/L2/L3 | Explicit `(01)^45 000101`, N191/W26/deficit9 kills all three relaxed bounds | Certificate; physically impossible via forbidden0101000101 |
| L4 and physical SAT | L4 N191/N300 UNKNOWN; physical K64/128/256, some N<=300 UNKNOWN | No exclusion obtained; retain holes |
| Transients | Physical zeros177..190 and next1 at191 reproduced; L4 finite-tail maximum14 is uncensored-only | Bounded observation; no potential or universal reset |

The full finite scopes, controls, per-query metadata and next proof gap
are in `docs/theory/period2_column_ladder.md` and the component documents.
The raw artifacts are under `runs/p2-free-density-2026-10-15`,
`runs/p2-u-language-2026-10-15`, `runs/p2-transient-2026-10-15`, and
`runs/p2-deficit-sat-2026-10-15`.

The theory gate corrected the proposed odd-right latch to odd-left,
rejected the finite-factor-to-SFT inference and the finite-influence-to-
no-transducer inference, and retained all physical/free-boundary
distinctions. Review strengthened physical certificate replay to bind
CNFs, source digests, seed support, fixed-seed controls, and witness N/W
metadata to their claimed scope; tampered records are rejected.

The original baseline full suite exposed two existing environment-sensitive
test defects: missing-Codex behavior was tested without hiding the installed
executable, and a stale test demanded Mathlib although the base project
imports Lean core only. Both fixtures were corrected with focused checks.
A pool timing check also failed under concurrent solver load; final suite
execution is serialized after the research workers stop. These failures
are not evidence against the research claims.

Next: the precise all-N L4 deficit conjecture in the recombination note.
Resume its retained N191 SAT query with a larger bounded budget; one
reconstructed witness kills that relaxation. A surviving query is not a
proof and does not eliminate the physical support-growth obligation.

Not run/proved: asymptotic free density, period-two exclusion, an endpoint-
bounded potential, infinite physical seed32/96/160 exact7, Lean no00000,
sharp horizons, unavailable inherited LFS proof replay, GPU/46M data, or
cross-lineage audit. Publication and final verification outcomes are
recorded below only after those operations complete.
