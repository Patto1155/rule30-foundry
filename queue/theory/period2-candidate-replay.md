# Finite candidate kill tests

The candidate queue records 20 first-round statements, four in each requested
field, followed by 12 data-conditioned statements. Eighteen have retained exact
counterexamples; fourteen passed their stated finite test domains. **None is a
proved infinite lemma.** Seven second-round survivors distinguish the same
finite Rule 90 control and enter the obligation queue as open, ranked targets.
The other survivors require an OR-specific derivation before admission.

Numerical coefficients were conjecture choices, not confidence intervals or
estimated existential constants. In particular the failure of
`6 W_N(r) >= N-6 bitlength(r)` at `r=32,N=259` rejects exactly that statement,
not positive seed-specific density. The successful finite tests of the analogous
coefficient seven statement do not establish a universal minimum density.

Round one tested all 64 right integers below64 through depth256. Round two used
64 base seeds plus the frozen B SAT/adversarial/finalist corpus (321 distinct
right seeds in the generated queue), through depth4096. The three seeds32,96,160
were singled out because B's longer measurements found density close to1/7 and
C's SAT plateaux included these right halves. The residue-cylinder extrapolation
`r=32 mod64` fails at224. This is why the exact small-seed obligations retain
explicit seed quantifiers rather than extrapolating a residue family.

Replay after the support worker's file has been incorporated:

```sh
python experiments/period2_candidates.py --verify queue/theory/candidates.json
```

In isolated worktrees append
`--support-dir ../rule30-support/experiments`. The checker rebuilds retained
completions, checks every surviving statement over its stored finite domain,
replays discriminating Rule90 witnesses, and rejects a deliberately altered
completion certificate. Constant-zero controls distinguish the Rule30 latch
from Rule90's finite `r=1` completion.

Generate the first round with the default command, and the second with
`--round2 --depth 4096 --support-data runs/support-growth-2026-10-14/results.json`.
Running the second command replaces that round rather than duplicating it.
No recurrence fit, DFAO fit, or no-fit model-class search was performed, so a
counting-bound gate is inapplicable to these explicit statement counterexamples.

Not run: an infinite support-growth proof, a right grammar invariant proof,
Lean, SAT or DRAT experiments, a positive entropy proof, GPU, external-family
review, or the full repository verifier in this isolated worker. The lead runs
the full verifier after combining worktrees. Targeted replay and ledger lint
passed. `support_growth.py` is intentionally supplied by B rather than copied
into a divergent implementation.
