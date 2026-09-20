# A robust right-boundary invariant with three short entry triggers

## Goal and scope

This construction strengthens right coherence by an invariant under the actual
forced right evolution, rather than by appending a list of forbidden trace
words. It is conditional on entering an explicit set. It does **not** prove
that every finite right half enters, that the left initial weight grows, or
that a finite configuration cannot alternate forever.

Use the same nearest-cell-first integer convention and forced centre `01` as
in `alternating-right-boundary-coherence.md`:

```text
R_b(x) = ((x << 1) | b) XOR (x | (x >> 1)),
H = R_1 o R_0,                 s_k = H^k(x) mod 2.
```

## Finite invariant and universal entry cylinders

Project to the low sixteen right cells. Define the four possible successors
of a projected state `v` by

```text
E(v,a) = H(v + 2^16 a) mod 2^16,      a in {0,1,2,3}.
```

The extra two bits are exactly the dependency-cone width for two updates.
Forget all predecessor coherence and allow the next `a` to be chosen
adversarially at each step. Every actual finite or infinite right trajectory
still projects to a path in this larger graph. Therefore invariance under
**all four** edges is a sound sufficient condition for an actual trajectory.

Let

```text
S = {v : v mod 2 = 0 or v mod 8 = 7}.
```

Starting with `K_0=S`, repeatedly delete every state having any successor
outside the current set. The decreasing sequence reaches a fixed point `K`
after 59 strict deletion rounds. It contains 13,526 states and satisfies

```text
K subset S,       E(K,{0,1,2,3}) subset K.
```

A smaller sufficient certificate starts with the three entire cylinders

| Low eight bits, integer | Nearest-cell-first word |
|---:|---|
| 167 | `11100101` |
| 168 | `00010101` |
| 170 | `01010101` |

These cylinders contain 768 sixteen-bit states. Their forward closure `C`
under all four edges has 1,452 states after 19 strict expansion rounds, and
all its 5,808 edges stay in `C`. Every state of `C` satisfies `S`.

Consequently, whenever any of the three eight-cell words occurs immediately
to the right of the forced centre at an even time, all subsequent even times
satisfy

```text
s_k=1  implies the first three right cells are 111.       (1)
```

The eight-cell word can be followed by **any** finite or infinite tail.
This is a forward-invariant implication, not a statement that a finite
trace prefix uniquely reconstructs the right half.

## The induced renewal restriction

The already checked three-bit return graph has

```text
7 -> {2,6},    6 -> {0,4},    4 -> 2,    2 -> 0,
0 -> {1,5,7}.
```

Inside `K`, only `7` is an allowed odd state. Starting at `7`, the next odd
state therefore occurs after exactly three or five macrosteps. The smaller
closure `C` actually has the projected edges

```text
0 -> 7,    2 -> 0,    4 -> 2,    6 -> 4,    7 -> {2,6}.
```

Thus after entry, the right output is a shift of a concatenation of
`100` and `10000`: successive ones have either two or four zeros between
them. This is a **necessary renewal relaxation**, not a claim that every
concatenation is realized. Shared predecessors may restrict it further.

The output restriction follows indefinitely from finite graph closure. No
long trajectory simulation is used to justify the induction. In particular,
after entry the words `101` and `10001` never occur, and every output window
of length `M` has between `floor(M/5)` and `ceil(M/3)` ones.

## Coupling to the seven-column mechanism

The exact first-slab formula in `golden-mean-phase-slip.md` becomes

```text
W_7 = 4 - 2s_1 - s_2 - s_3 >= 2,                         (2)
```

because the renewal restriction permits at most one one in three consecutive
positions. This doubles the first-slab minimum compared with the no-`11`
relaxation. It still cannot be restarted at an arbitrary spatial cut.

For the old production term `P_i=1[s_i s_(i+1) s_(i+2)=000]`, a complete
inter-one interval of length three contributes zero occurrences, and an
interval of length five contributes two. If a union of complete intervals
has total length `M`, with `r_3` short and `r_5` long intervals, then

```text
M = 3r_3 + 5r_5,
number of ones = r_3 + r_5,
number of 000 starts = 2r_5 = M - 3(number of ones).
```

This makes production an exact count of long renewal intervals. Together
with the existing phase-slip coboundary identity, it gives a concrete
boundary variable to transport. It does not itself pay for arbitrary
initial-row slabs. The separate second-slab calculation is recorded in
`phase-slip-transport-followup.md`; this invariant construction does not rely
on that calculation.

## Finite-right examples and missing implication

Exact forward evaluation gives these first entries into `K`:

| Initial finite right integer | First entry, macrosteps |
|---:|---:|
| 0 | 147 |
| 1 | 146 |
| 2 | 0 |
| 3 | 146 |
| 4 | 145 |
| 17 | 3 |
| 12345 | 12 |

These are examples, not an induction over support width or weight. Nothing
here establishes `finite initial right half => eventual K entry`. Nor does
entry prove that the resulting renewal word is eventually periodic. Neither
assertion should be inserted into a prize argument without a separate proof.

The useful split is now precise: prove eventual entry (or show that persistent
nonentry requires infinite initial right support), and prove left growth for
the sound renewal relaxation or a more coherent successor model. Both remain
additional obligations; the invariant alone closes neither.

## Reproduction and independent checks

```bash
python experiments/right_coherence_followup.py
```

The script reconstructs the graph, derives `K` and `C`, verifies all closure
edges, and verifies the first-return spacings. The packed right map is checked
against a separate cell-by-cell ECA truth-table cone evaluator on all
262,144 eighteen-bit inputs. A separate reviewer independently recomputed
all 5,808 closure transitions cell-by-cell and reproduced both cardinalities.
Agreement is not the evidence: exhaustive transition coverage plus forward
induction is the evidence.
