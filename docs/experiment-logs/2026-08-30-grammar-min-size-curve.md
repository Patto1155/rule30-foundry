# Experiment Log

- Date: 2026-08-30
- Title: Smallest-Grammar Upper-Bound Curve `g(n)` for the Rule 30 Center Column
- Claim Level: **Robust observation** — matched 7-seed null band, a positive
  control with 174x separation, and the counting null cleared at every point.
  Not a Certificate: the exact smallest grammar is NP-hard, so `g(n)` here is
  an *upper* bound from a heuristic, not a minimum.
- Run date: 2026-08-30 (58 s)
- Artifact: `data/prize/2026-08-30-grammar-min-size-curve.json`
- Verifier: `python experiments/grammar_min_size.py --gate-only`

## Goal

Measure a minimal-description curve over a **third** model class, and the one
closest to Prize Problem 2. The repo has done this for LFSRs (Experiment S,
`L(n) = n/2`) and for DFAOs (`s*(n)`, certified 2026-08-30). Straight-line
grammars are the natural next step because a small straight-line grammar *is*
a small program that prints the prefix — which is exactly the shape a "faster
algorithm for the nth center cell" would have to take.

    g(n) = number of rules in the smallest straight-line program in Chomsky
           normal form deriving center bits 0..n-1

Every rule is `A_i -> X Y` with `X, Y` from `{0,1}` plus the nonterminals below
`i`, so an SLP with `g` rules is an explicit finite object of size `2g`.

## The caveat that is not a footnote

**This is an upper-bound curve.** Computing the exactly smallest grammar is
NP-hard, so every `g(n)` below is what Re-Pair found. The true `g*(n)` is at
most that. The comparison against the random control is therefore between two
upper bounds produced by the same deterministic algorithm — a fair comparison,
but a weaker object than `s*(n)`, where the minima are exact and now certified.

Stated as a claim: **this experiment cannot rule out that some cleverer grammar
construction compresses the center column and not the control.** What it rules
out is that the standard one does.

## Admission Rule, stated before running

(`docs/theory/finite-prefix-counting-bound.md`.) Rule `i` chooses an ordered
pair from `{0,1} u {A_1..A_{i-1}}`, so

    |M(g)| = prod_{i=1..g} (i+1)^2 = ((g+1)!)^2
    log2|M(g)| = 2 * log2((g+1)!)

and a negative over this class on `n` bits is vacuous unless
`log2|M(g)| >= n`. The threshold `g_null(n) = min{ g : 2*log2((g+1)!) >= n }`
is reported beside every measurement.

Note the two bounds run in opposite directions and both are needed:
`g_null(n)` is a **lower** bound on the exact `g*(n)` for almost all strings
(by counting), while Re-Pair gives an **upper** bound. The true curve lies
between them, and the measurement is only informative where the upper bound
sits above the lower one. It does, everywhere, by a factor of about 2.5.

## Setup

- Tool: `experiments/grammar_min_size.py` (new)
- Algorithm: Re-Pair — repeatedly replace the most frequent adjacent pair —
  then the residual start sequence is folded into a left-deep chain of binary
  rules so the whole object is a CNF SLP and the counting bound applies to a
  single number. Omitting that last step would undercount the grammar.
- Adjacent pairs are counted with overlap and replaced greedily left to right.
  For runs, that can pick a different pair than a strictly non-overlapping
  Re-Pair would. It cannot make a result unsound — whatever pair is chosen, the
  grammar still derives the input, so the size is still an upper bound — and
  the identical rule is applied to the center column and to every control.
- Sequences from `dfao_min_states.sequence_bits`, the same generator as the
  `s*(n)` experiment.
- `n` = 64 … 65536 by doubling; 7 random seeds (30–36), matching the `s*(n)` band.

## Correctness gate (run before any science)

All seven checks pass; the run refuses to measure otherwise.

| Check | Result |
|---|---|
| center / thue-morse / random n=512 grammars derive their own input | PASS |
| all-ones n=4096 compresses to O(log n) rules | PASS (g=12) |
| thue-morse n=4096 far below random | PASS (34 vs 732) |
| center prefix matches the hash-anchored golden reference over 4096 bits | PASS |
| random g(n) clears the counting null at n=4096 | PASS (732 vs 300) |

Every grammar produced anywhere in the run is expanded and compared against its
input, not just the gate's. A grammar that does not derive its own string would
make every number here fiction, so it is checked rather than assumed.

The gate's cross-check against `data/golden/center_col_golden_1M.bin` matters
independently: it ties this experiment's center bits to the hash-anchored,
independently generated reference rather than trusting the generator.

## Result

```
      n  center  rand min  rand max   TM  g_null  c*log2(n)/n  r*log2(n)/n  verdict
     64      25        22        29   15      12        2.344        2.531   inside
    128      48        44        47   17      20        2.625        2.406   above
    256      78        72        80   21      34        2.438        2.406   inside
    512     131       122       135   23      57        2.303        2.303   inside
   1024     231       228       242   28      98        2.256        2.256   inside
   2048     413       401       412   29     170        2.218        2.191   above
   4096     746       725       740   34     300        2.186        2.139   above
   8192    1328      1314      1344   35     536        2.107        2.104   inside
  16384    2406      2400      2419   41     966        2.056        2.061   inside
  32768    4438      4426      4458   41    1754        2.032        2.031   inside
  65536    8175      8155      8228   47    3210        1.996        1.997   inside
```

- **8 of 11 inside the 7-seed random band, 3 above, 0 below.** As with `s*(n)`,
  where the center column deviates it deviates *upward* — the direction that
  means "harder to compress", not "structured".
- **All 11 clear the counting null**, by roughly 2.5x. The measurement is in
  the informative regime at every point.
- **Detection power is not marginal and does not decay.** Thue-Morse stays
  essentially flat (15 rules at n=64, 47 at n=65536 — logarithmic, as a
  morphic sequence must be) while the random band grows linearly in `n/log n`.
  At n=65536 the separation is **174x**. A genuinely compressible sequence is
  found instantly at every scale tested.
- **The two curves converge rather than separate.** Normalising by the
  information-theoretic rate, `g(n)*log2(n)/n` settles to **1.996** for the
  center column and **1.997** for the random median at n=65536. The ratio
  `g_center / g_random` there is **0.9996**.

## Interpretation

**No grammar-based shortcut, over three orders of magnitude of prefix length.**

The center column's smallest-grammar curve is quantitatively indistinguishable
from that of a fair coin — to within 0.04% at n=65536 — while the same
measurement separates a genuinely compressible sequence from noise by a factor
of 174. Both curves follow `g(n) ~ 2n/log2(n)`, the rate for an incompressible
string, and the agreement gets *tighter* as `n` grows, which is the opposite of
what any structure that switched on at scale would produce.

This is the strongest of the three model-class negatives in the repo, for two
reasons. First, reach: `s*(n)` is certified only to n=48 and `L(n)` to a
comparable scale, while this runs to n=65536 — a factor of 1365 further.
Second, relevance: the DFAO and LFSR classes are narrow, but a straight-line
grammar is a general-purpose program that prints the prefix, so a negative here
speaks more directly to "is there a faster algorithm for the nth cell".

Scope, stated plainly, because the reach makes it tempting to overclaim:

- It is an **upper-bound** curve. A better grammar construction is not excluded.
- Re-Pair is a greedy heuristic and is known to be off the optimum by up to a
  logarithmic factor in the worst case. A shortcut hiding entirely inside that
  gap, for the center column and not for seven random controls, is not ruled
  out by this.
- 65536 bits is still a prefix. It says nothing asymptotic.
- The prize asks for the `n`th cell in sub-quadratic time. A small grammar
  would suffice for that; the absence of one that Re-Pair can find does not
  prove no such algorithm exists, because an algorithm need not take the form
  of a compressed representation of the prefix.

## Relation to the anti-goals

`docs/handover/2026-08-30-next-session-plan.md` lists "more aggregate
randomness tests on the center column" as an anti-goal, and this experiment
sits close to that line. What keeps it on the right side is the same thing that
distinguished `s*(n)` from the retracted DFAO certificate: it does not report
"the sequence looks random under statistic X". It measures the *minimum
description length over an explicitly named model class*, states `log2|M|` for
that class before running, checks that the measurement clears the counting
threshold, and compares against both a matched null and a positive control.
The output is a curve with a rate constant, not a p-value.

## Next Step

1. **Close the upper/lower gap.** The counting null (`g_null`) and Re-Pair
   bracket `g*(n)` a factor of ~2.5 apart. Exact smallest-grammar solving via
   SAT/ILP at small `n` — the same design as `s*(n)`, and the same route to a
   Certificate — would turn this from a heuristic comparison into exact minima.
   n ≤ 64 looks tractable given how cheap the DFAO instances turned out to be.
2. **A second heuristic as a robustness check.** Sequitur or a greedy
   longest-match variant. If the center column were compressible in a way
   Re-Pair specifically misses, a second algorithm disagreeing would show it.
3. Do **not** simply push `n` higher. The curves have already converged to
   three decimal places; another octave adds reach, not information, and the
   binding limitation is the upper-bound gap, not the prefix length.
