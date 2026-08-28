# Experiment Log — Period-16 Is False: The Period Doubles at d = 87866

- Date: 2026-08-19
- Title: Direct refutation of the period-16 conjecture, and the growth law that replaces it
- Claim Level: **Certificate** (the refutation) + **Robust observation** (the growth law)
- Prize: Problem 3 context. Structural work on the seed orbit, not a shortcut claim.

## Result in one line

> **The period-16 conjecture is FALSE.** Left diagonal `d = 87867` has minimal
> period **32**. The period is not bounded — it grows like `≈ 2·log2(d)`.

## Goal

Decide the last open Proof candidate in the ledger:

> **Conjecture (period-16).** Every left diagonal is eventually periodic with
> period dividing 16.

`2026-08-19-period-doubling-criterion.md` reduced this (Lemmas A and B) to:
period-16 holds through `D` iff every consecutive-word collision below `D` has an
even-parity predecessor. This experiment checks that directly, from ground truth.

## Method — and why the planned approach was abandoned

The plan had been to iterate the 16-bit pattern map to `d ~ 10^6`, resolving each
zero word by local simulation. Two changes:

1. **The branch points are an artifact of the map, not of the problem.** The map
   discards the diagonal transient, and at a zero word the transient is exactly
   what disambiguates. Simulating the cone directly has **no branch points** —
   every settled word is ground truth.
2. **`left_diagonals` cannot scale, but only because of its output array.** Its
   big-int CA advances *all* diagonals in one bitwise op and is `O(T)` in
   memory; the `(T+1, D)` array is what would need 1.34 TB here.

So `experiments/period16_walk.py` runs the same big-int CA and records only a
**packed periodic tail** (1024 rows, `D/8` bytes each), then extracts the
**minimal period** of every diagonal.

> **Critically, it does not assume 16.** Testing only `rows[i] == rows[i+16]`
> makes a period-32 diagonal look *unsettled*. That is precisely what hid this
> result before: the settled count silently dropped at `d ≈ 87868` and was read
> as "not yet settled" rather than "no longer period 16".

## Result 1 — the refutation

Verified at **two independent simulation sizes** (`T = 151375` and `T = 300000`
— different widths, different tail phase, agreeing exactly):

| d | minimal period | note |
|---|---|---|
| 87864 | 16, word `0x28a8` | |
| 87865 | 16, word `0x28a8` | **collision** `w_{d-2} = w_{d-1}` |
| 87866 | 1 (all zeros) | zero word — exactly as **Lemma A** predicts |
| 87867 | **32** | doubled — exactly as **Lemma B** predicts |
| 87870, 87900 | 32 | |

`popcount(0x28a8) = 5`, **odd**. Both lemmas fired as proved.

Full walk to `d < 10^6` (`T = 1476886`, 393 s, all 10^6 diagonals settled):

```
periods : {1: 16, 2: 10, 4: 56, 8: 668, 16: 87136, 32: 912114, 64: 0, 128: 0, 256: 0}
first d with period > 16 : 87867
VERDICT : REFUTED
```

## Result 2 — Lemma B is confirmed at every event, 7 for 7

| zero word `d` | period before → after | predecessor at that period | popcount | parity | predicted | OK |
|---|---|---|---|---|---|---|
| 2 | 1 → 2 | `0x1` | 1 | odd | 2 | ✔ |
| 7 | 2 → 4 | `0x2` | 1 | odd | 4 | ✔ |
| 28 | 4 → 8 | `0xe` | 3 | odd | 8 | ✔ |
| 399 | 8 → 16 | `0xd0` | 3 | odd | 16 | ✔ |
| 53207 | 16 → 16 | `0x28c3` | 6 | even | 16 | ✔ |
| 58286 | 16 → 16 | `0x6f74` | 10 | even | 16 | ✔ |
| **87866** | **16 → 32** | `0x28a8` | 5 | **odd** | **32** | ✔ |

**0 failures.** The lemmas are confirmed exactly while the conjecture built on
them is refuted — the two even-parity collisions correctly *kept* the period at
16, and the odd one doubled it.

### Correction to the previous log

`2026-08-19-period-doubling-criterion.md` recorded that the four early zero words
had **even**-parity predecessors and that "doubling never fired". **That was
wrong**, and the cause is worth recording:

> A period-`p` diagonal (`p < 16`) still yields a 16-bit word — `16/p` copies of
> its `p`-bit word. Taking popcount over all 16 bits therefore **doubles** the
> true popcount whenever `16/p` is even, forcing even parity. Lemma B's parity
> must be taken **at the minimal period**.

At their true periods all four early predecessors have **odd** parity and the
period doubled every time: that is how the period reached 16 at all. Fixed in
`experiments/period_doubling.py` (`minimal_period_of_word`), which now reports
`doubling fired: True`.

## Result 3 — the growth law that replaces the conjecture

Doubling to period `2^k` occurs at the `k`-th odd-parity collision. Collisions
between period-`p` diagonals need equality of `p`-bit words, so they arrive at
rate `2^-p`, and each doubles with probability `~1/2`. Hence period `2^k` should
begin near `d ~ 2^(2^(k-1))`:

| period | observed start `d` | `2^(2^(k-1))` |
|---|---|---|
| 2 | 2 | 2 |
| 4 | 7 | 4 |
| 8 | 28 | 16 |
| 16 | 399 | 256 |
| 32 | 87866 | 65536 |

Equivalently, and more usefully:

> **`period(d) ≈ 2·log2(d)`, rounded down to a power of two.**

Checks: `d=28 → 9.6 → 8`; `d=399 → 17.3 → 16`; `d=87866 → 32.8 → 32`;
`d=10^6 → 39.9 → 32`. All correct.

**Next doubling (32 → 64) requires a 32-bit collision: expected `d ~ 2·2^32 ≈
8.6×10^9`.** This is why period 32 persists unbroken across the entire
`87867 ≤ d < 10^6` range — 912114 diagonals with no further event — and it
explains why the conjecture looked so solid: period 16 genuinely holds over
`399 < d < 87866`, a stretch of ~87000 diagonals.

## Interpretation

The corrected statement is stronger and more interesting than the conjecture:

- **"Every left diagonal is eventually periodic" remains a Theorem** — unaffected.
  The periods simply grow, exactly as the period-propagation lemma always allowed.
- The period is **unbounded but grows only logarithmically**, so the settled
  wedge is still `O(t·log t)`-describable rather than `O(t)`.
- The 276M-cell wedge Certificate is **unaffected as a certificate** — it was run
  at `T=30000, d=15000`, entirely inside the period-16 region — but its `O(1)`
  16-bit pattern map is **valid only for `d < 87866`**. Generalizing it needs
  `2·log2(d)`-bit words.
- Left-edge structure remains **disjoint from the prize object**: `settle(d) ≈
  1.34·d > d` is untouched, so none of this yields a centre-column shortcut.

### Methodological note

Three separate failures in this thread had the same shape — *a negative quoted
from a range, or a representation, too small to contain the falsifying event*:

1. the retracted DFAO certificate (model class too small — counting bound);
2. the original period-16 evidence (test range below the event rate — power);
3. **this one** (16-bit representation too small to express the event at all).

The third is the sharpest: the analysis code could not represent a period-32
diagonal, so the refuting object was invisible to it and showed up only as a
drop in the "settled" count. Before trusting a negative, ask not just whether
the range is large enough, but whether the *representation can express the
counterexample*.

## Next Step

1. Generalize the wedge pattern map to `2·log2(d)`-bit words and re-issue the
   compression certificate without the period-16 assumption.
2. Confirm the growth law at the next event by probing near `d ~ 8.6×10^9` —
   note this needs `T ~ 1.2×10^10`, far beyond a direct big-int CA, so it needs
   the certificate generalization first, not more simulation.
3. **Do not** re-assert any period bound without stating the representation
   width and the event rate that bound it.

## Commands

```bash
python experiments/period16_walk.py --diagonals 12000  --pretty          # validates vs known ground truth
python experiments/period16_walk.py --diagonals 1000000 --keep 1024 --pretty \
    --out data/wedge/period16_walk_1e6.json                              # 393 s
```

Exits non-zero on a refutation (it found one), a Lemma A violation, or a
Lemma B failure.

## Artifacts

- `data/wedge/period16_walk_1e6.json` — period histogram over `d < 10^6`, all
  zero words, all collisions, and the Lemma B event table.
- `data/wedge/diagonal_propagator.json` — the rolling-window propagator's
  bit-exact verification (312,012,000 cells, `np.array_equal` on every diagonal)
  and its honest benchmark.
