# Periodic free-boundary density test

## Result

The exact sideways Rule 30 map was exhausted for every periodic free
even-time right-boundary word of period `p <= 6`. All 126 orbits repeated
within the declared limit of 100,000 states per word; none was censored. No
eventual spatial cycle has initial-left density below `1/8`, so this family
does **not** falsify a uniform inequality

```text
8 W_N - N >= -C.
```

The minimum observed density is `1/7`. At period 2, the boundary word `01`
in low-bit-first temporal order has transient length 1 and the explicit
spatial output cycle

```text
0000001
```

Thus its cycle charge is `8*1 - 7 = 1`, positive rather than divergent in the
negative direction. Periods 4 and 6 also attain `1/7`. The period minima are:

| period | minimum density | maximum visited states | censored words |
|---:|---:|---:|---:|
| 1 | `1/2` | 5 | 0 |
| 2 | `1/7` | 8 | 0 |
| 3 | `1/2` | 86 | 0 |
| 4 | `1/7` | 36 | 0 |
| 5 | `68/155` | 188 | 0 |
| 6 | `1/7` | 230 | 0 |

The `1/7` mode is a genuine Rule 30 spacetime ether. A direct periodic-ring
truth-table calculation gives the exact 7-by-4 witness

```text
1000000
1100001
0010011
1111110
1000000
```

Each row is the Rule 30 successor of the preceding row, including the spatial
wrap, and column 1 is `0101`. Repeating the first row in space therefore gives
an exact alternating-centre solution whose initial-left ones have zero-based
bit indices `0, 7, 14, ...`, equivalently distances `1, 8, 15, ...` from the
centre. Consequently any
uniform inequality `alpha * W_N - N >= -C` for this family must have
`alpha >= 7`. The proposed coefficient 8 is not contradicted.

## Setup and evidence

For `p`-bit temporal words, with bit zero representing phase zero, the state
is `(Eprev, E, Oprev, O)`. One leftward spatial step is

```text
Enew = O XOR (E OR Eprev)
Onew = rotate_time(E) XOR (O OR Oprev)
```

where `rotate_time(E)[k] = E[k+1 mod p]`. The start state for boundary `s` is
`(s, 0, 0, allones)`, and each step emits the low bit of `Enew`.

Two independent checks had zero mismatches:

- phase-by-phase comparison of both word equations with the Rule 30 truth
  table, including the temporal period wrap;
- comparison of the first `p` emitted spatial bits with the exact finite-cone
  constructor for every right initial word through period 4.

The displayed ether adds a third check using a direct cell-by-cell Rule 30
truth table on a periodic spatial ring; it does not call the sideways map.

The controls are finite-cone constructor baselines only. They do not exercise
the periodic word map or its orbit scanner. For Rule 15, all four tested
two-bit right initial boundaries give an exact
all-zero 12-bit initial-left prefix while the centre alternates. This is the
genuine zero-left alternating control. For Rule 90, the same four boundaries
give four distinct exact left prefixes, so its boundary constraints are
materially different.

The saved JSON is replayable rather than merely archival:

```bash
python experiments/periodic_boundary_density.py \
  --verify runs/theory/periodic-boundary-density.json
```

Verification requires the declared period-6 and 100,000-state scope,
recomputes every exact orbit, and compares the complete parsed report,
including counts, cycle witnesses, and rational densities.

## General periodic nonzero-density lemma

Every finite-period boundary in this family has positive eventual density.
Suppose instead that a spatial cycle emitted zero at every step. Because an
emission becomes the next state's phase-zero `E` bit, phase-zero `E` vanishes
at every state on the cycle. The `Enew` equation then forces phase-zero `O` to
vanish. The `Onew` equation forces phase-one `E` to vanish, after which the
`Enew` equation forces phase-one `O` to vanish. Induction through the `p`
temporal phases makes both `E` and `O` the all-zero word at every state.

The resulting all-zero four-word state has only itself as a predecessor: its
successor's first and third components first force current `E=O=0`, and the
two update equations then force `Eprev=Oprev=0`. It therefore cannot be
reached from `(s, 0, 0, allones)`. Since the state space is finite, every orbit
eventually cycles, and that cycle contains at least one emitted one.

## Gap and next implication

The lemma proves positivity separately for each finite period. It gives no
lower bound uniform in `p`; the cycle length could grow faster than its number
of ones. The scan through period 6 therefore neither proves density at least
`1/8` nor supplies a global cone-charge constant.

The next exact question is whether a cycle-level invariant can bound the gap
between successive emitted ones by 7, or more generally prove
`8 * cycle_ones >= cycle_length`. The physical `0000001` ether shows that a
bound of 6 zeros is already sharp. A larger-period cycle
with density below `1/8` would immediately falsify the uniform charge claim;
a finite negative transient alone would not.
