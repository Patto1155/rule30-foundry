# Primitive period-word sieve

- Date: 2026-09-08
- Goal: find periodic center words that force a periodic adjacent column,
  uniformly over transient length and open-strip boundary choices.
- Base: `b765ba6` (main after PR #34).
- Setup: all primitive binary necklaces of lengths 1..6; widths 3, 5, 7, 9.
- Artifact: `runs/period-word-sieve-2026-09-08/sieve.json`.

## Reproduction and checks

```bash
python experiments/period_word_sieve.py --scan
python experiments/period_word_sieve.py --verify runs/period-word-sieve-2026-09-08/sieve.json
python -m unittest discover -s tests -p 'test_period*.py'
```

There are 2, 1, 2, 3, 6, 9 primitive rotation classes by length: 23 words
and 92 word/width cases. Rotating a word only relabels the graph's time phase,
so one representative covers every onset phase. No complement symmetry is
assumed. The verifier reports the configured period and width coverage and
rejects missing, duplicated, or changed records.

The independent audit builds transitions using cell tuples and rule-number
truth-table lookup, then finds components by mutual forward/backward
reachability. The search uses Boolean updates and Kosaraju components.
Both check every internal p-step path in each recurrent component for an
output mismatch. Every infinite finite-graph walk eventually remains in one
such component; thus passing this test proves eventual p-periodicity of the
left output. A violating path can be closed and repeated inside its component.
For stronger failures, the saved equal-length loop pairs are checked directly
by `verify_output_loops`; the proof of nonperiodic concatenations is in
`docs/theory/periodic-trace-constraints.md`, section 6.

This is an exhaustive local implication check, not a finite-prefix model fit:
there is no sample size n or fitted class M to which the counting gate applies.
No canonical bitstream is consumed. Constants are positive controls and `01`
is the previously certified negative control. Twenty-five focused tests pass,
including rotation invariance, independent updates, and artifact tampering.

## Results

| Center words | Widths | Result |
|---|---|---|
| `1` | 3, 5, 7, 9 | Forces eventual period 1 |
| `0` | 5, 7, 9 | Forces eventual period 1 |
| `0` | 3 | Nonperiodic left-output loop pair |
| All 21 nonconstant primitive words, lengths 2..6 | 3, 5, 7, 9 | Nonperiodic left-output loop pair in every case |

Seven implications pass. All 85 failures have two directly verified closed
walks with different equal-length output blocks. There are no cases lacking
a recurrent component, as required by arbitrary-trace realizability.

The constant controls have elementary explanations. If the center is 1,
its left neighbor is 0. If the center is 0, its two neighbors agree, and
the right neighbor obeys `r(t+1) = r(t) OR x_2(t)`; hence it eventually
stabilizes. Width 3 does not impose the right neighbor's update, while
width 5 does. These facts were already available; the sieve adds no excluded
nonconstant word.

## Interpretation and next step

The proposed sieve has produced no new all-onset single-seed exclusion.
Every tested nonconstant word fails even the weaker demand that its neighbor
eventually have *some* period. This is a bounded obstruction to the method,
not proof that any such tail can occur on the seed or that it survives every
larger width.

The intended prize connection uses the adjacent-column obstruction discussed
in the [official prize announcement](https://writings.stephenwolfram.com/2019/10/announcing-the-rule-30-prizes/).
The [NKS note](https://www.wolframscience.com/nks/notes-10-10--cryptographic-properties-of-rule-30/)
also describes reconstruction from a column plus selected cells of its neighbor.
Those are background results, not discoveries of this experiment.

No composition law for a new infinite family emerged: only the two constant
words passed, and repeating them does not create new minimal periods. Stop
this bounded sweep here. Further work needs either a proved constraint from
the complete seed/exterior, or a justified mechanism by which wider strips
eliminate the saved branching witnesses. Merely enumerating more short words
is not supported by this outcome.
