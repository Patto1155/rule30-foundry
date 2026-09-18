# 2026-09-18 — A theory-first loop, and the first instrument built for it

The repo produces finite exclusions well and has produced no theorem since the
diagonal recursion. This adds the loop whose artifact is a proof obligation, the
gates that refuse a doomed one before any compute, and the first instrument
designed inside it.

Nothing here is a prize result. The new measurement is a finite-prefix
exclusion, like the ones it is meant to succeed.

## 1. Why the exclusion loop cannot finish

All three prizes, in their believed direction, are lower bounds: no period
exists, density is exactly 1/2, cost is at least `O(n)`. Against a statement of
that shape computation can only find the unlikely positive, or accumulate finite
exclusions that never compose. `s*(64)=15` is a theorem about 64 bits and says
nothing about bit 65.

Set against that, every result here that generalised came from algebra on paper:
left-permutivity giving `v_right = 1`, Bernoulli invariance collapsing Prize 2
onto the single seed, sideways determinism, the diagonal recursion and its
period-propagation lemma, and the Reed–Muller bound that voided every
annihilator search at `w <= 22` **before it ran**. Theory has outyielded compute
here by a wide margin while the loop stayed overwhelmingly computational.

## 2. The counting squeeze, stated exactly

The counting bound wants `log2|M| >= n` or a negative says nothing. The
forced-positive gate wants `log2|M| <= n` or a fit exists by dimension alone.
**They meet at one point.** So any design whose evidence is a bare negative is
informative only there — which is exactly why the DFAO programme needs a fresh
instance ~10x harder for each additional state.

Two designs escape it, and `tools/theory_triage.py` now enforces both:

- **`extrapolation`** — fit on `N` coefficients, require prediction of `H` that
  took no part in the fit. No parameter count forces that.
- **`curve-shape`** — the claim is plateau versus growth across at least three
  sizes, against a null and a positive control, not any single point.

## 3. The gates

`tools/theory_triage.py` runs three stages, cheapest first.

**Mechanism gate, free.** `docs/theory/README.md` section 4 lists twelve closed
routes; across them there are six mechanisms, and every obligation must name
which it is near and why it escapes. Naming none is a refusal.

**Arithmetic gate, free.** Where an obligation declares a searchable class, the
gate recomputes `log2|M|` against `n` from the repo's own `counting_bound.py`
rather than believing the prose. It reproduces the repo's published verdicts:

| case | verdict |
|---|---|
| DFAO `s<=5`, `n=128` | **FAIL, vacuous** — the certificate retracted 2026-08 |
| DFAO `s=20`, `n=128` | PASS, 64.9 bits of margin |
| annihilator `w=22, d=3` on 10M | **FAIL, vacuous** — matches the Reed–Muller closure |
| algebraic relation `D=E=99`, `N=8192` | **FAIL, forced positive** |
| same at `D=E=16` as a bare negative | **FAIL, vacuous** |
| same with `evidence: curve-shape` + controls | PASS |

**Jev ranking, ~$0.00006.** Among survivors, which obligation gets the lead's
next hour. A typed choice over a closed set, far cheaper than reasoning over the
ledger in the agent's own tokens. It ranks attention and judges no mathematics.

A note on layering, from the live run. The syntactic gate passed a deliberately
weak obligation ("one more point on the `s*(n)` curve") because its counting
arithmetic is genuinely fine — `log2|M| = 192.9` against `n = 128`. Jev then
ranked it **last at p=0.01**. The free gate catches malformed and arithmetically
doomed work; the cheap call catches plausible-but-worthless. Neither alone is
enough.

## 4. The first instrument: algebraic relations over F_2(x)

Christol's theorem: a sequence over `F_2` is 2-automatic exactly when its
generating function is algebraic over `F_2(x)`. So the question the repo answers
by SAT at `n=64` is the same question as *does a bounded-degree algebraic
relation exist* — and that is a kernel over `F_2`, not a SAT instance.

`experiments/algebraic_relation.py` reports the minimal coefficient budget
`C* = (D+1)(E+1)` admitting a relation, and requires any fit to predict
held-out coefficients.

**Positive control, the strongest available.** The instrument independently
recovers the textbook Thue-Morse relation `(1+x)^3 f^2 + (1+x)^2 f + x = 0`,
returning exactly `D=2, E=3, C*=12` — and that budget stays **flat from N=256 to
N=8192**, six lengths, which is the plateau signature a genuine shortcut in the
center would show. Detection power does not decay with length.

**Measurement.**

| N | thue-morse | random | center |
|---|---|---|---|
| 2048 | `C*=12`, extrapolates | none | none |
| 4096 | `C*=12`, extrapolates | none | none |
| 8192 | `C*=12`, extrapolates | none | none |

The center column admits **no algebraic relation with `D, E <= 16` on prefixes
to 8192 bits**, and is indistinguishable from the matched random null while
Thue-Morse separates cleanly. The N=8192 center sweep takes 10.6 s.

**What this buys against the DFAO route.**

| | DFAO via SAT | algebraic relation |
|---|---|---|
| prefix | `n=64` | `N=8192` |
| class | 14 states, `log2\|M\|` 120.6 | 289 free coefficients |
| deciding run | 353 s solve, 2.26 GB proof, 598 s check | 10.6 s, no proof artifact |

**What it does not buy.** This is still a finite-prefix exclusion and does not
prove non-automaticity; no finite `N` can. The gain is two orders of magnitude
more prefix and a much larger class for a fraction of the compute, in a setting
where the asymptotic content lives in a curve rather than a point — and where
the question now sits in transcendence over `F_2(x)`, which has real proof
machinery, instead of in DFAO state counts, which has none.

Because this is a bare negative it would rank as a plain observation. Section 4b
supersedes it: the measurable quantity underneath is the *smallest budget that
fits*, which is always defined, and that curve against a 7-seed null is what the
ledger records at **Robust observation**. The `s*(n)` row is untouched either
way, and nothing here is a Certificate or a Theorem.

## 4b. Turning the absence into a curve

The section above reports "no relation under this budget", which is an absence,
and the ledger rightly grades an absence low. The fix is to stop asking whether
a fit exists under a fixed budget and ask instead for **the smallest budget that
fits**, which always exists: once the column count passes `N` a kernel is forced.

Define `E_min(D)` as the smallest coefficient degree admitting a relation of
algebraic degree `D`. That is ordering-independent, which a raw column count is
not, and

```
C*(N) = min over D of (D+1)(E_min(D)+1)
```

is then always defined. One elimination pass per `D` serves every `E`, because
columns are added in groups of constant `e` and the first group that closes a
dependency gives `E_min` directly.

**Measured, 7 random seeds for the null, max degree 8:**

| N | null band (7 seeds) | center | `C*/N` | center inside band | thue-morse |
|---|---|---|---|---|---|
| 128 | 120..128 | 126 | 0.984 | **yes** | 12 |
| 256 | 249..256 | 255 | 0.996 | **yes** | 12 |
| 512 | 508..512 | 510 | 0.996 | **yes** | 12 |
| 1024 | 1020..1024 | 1020 | 0.996 | **yes** | 12 |

The center column's algebraic complexity is **maximal**, landing inside the
7-seed random band at 4 of 4 lengths. Thue-Morse stays flat at `C*=12` for an
**85x separation** at N=1024 that does not decay with length, so the instrument
retains full detection power exactly where the center shows none.

This is the direct analogue of the certified `L(n) = n/2` linear-complexity
result, one model class up, and it is the design the theory gate calls the
template: a curve against a null, sitting at the counting threshold which is
also the maximum.

**Independent re-computation.** `C*` is re-decided by numpy convolution and
uint8 Gaussian elimination, sharing no code with the packed carry-less-multiply
and bitmap-elimination path. The two agree exactly on `(C*, D, E)` for
Thue-Morse, center and random. An instrument this cheap is easy to get subtly
wrong, and a rank computation that silently disagrees with itself would produce
exactly the kind of confident wrong answer this repo has retracted before.

**Still finite.** Maximal algebraic complexity to N=1024 is not a proof of
non-automaticity and no finite N can be. Extending the curve needs a packed-word
or GPU rank rather than more wall clock: the elimination is `O(N^3/64)`.

## 5. Reproduce

```bash
python experiments/algebraic_relation.py --self-test
python experiments/algebraic_relation.py --sequences center,random,thue-morse \
    --sizes 2048,4096,8192 --max-degree 16 --max-ext-degree 16 --pretty
python tools/theory_triage.py queue/theory/obligations.json \
    --policy jev --out runs/theory/triage-NNN
```

`--policy fixed` takes the first survivor and makes no model call; it is the
baseline any claim of selector benefit has to beat.
