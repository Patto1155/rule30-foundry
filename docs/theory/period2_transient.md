# Alternating-boundary transients: exact indices and local language obstructions

This note gives a conditional periodic identity and finite tail tables. It
proves neither the uniform deficit conjecture nor physical reachability of a
relaxed boundary word. Initial left depth is one-based: bit `j-1` is `e_j`.

## The physical 14-zero example

Independent forward simulation reproduces `r=0x98f9d8a0`. Its even-time
column-one sequence starts `(01)^45 00010000100100100001…`. The first 26
ones of the completion occur at `1,8,…,176`. The next one is **191**;
**177 through 190 inclusive** are the 14 zeros. Thus `W_190=26` and
`190-7W_190=8`. The last one is 176, not 182. The handover's phrase
“after position `7m`” for `u=(01)^m w` has the wrong scale: those `2m`
even-time inputs specify a temporal cone of depth `4m`.

A zero-run bound concerns one interval; it is not equivalent to a bound on
`N-7W_N` at every prefix. Repeated losses and any earlier weight surplus
also matter. The physical `r=17` zero slab at depths 22–28 still refutes a
stateless seven-column reset.

## Conditional exact-seven identity, with the state retained

Assume the **whole** free input is `u=(01)^infinity`. By the odd-time
boundary erasure in column reduction, choose its right-column representative
`d(t)=(0,0,1,1)` by time modulo four. Put `y_0=(0,1,0,1)` and apply

`y_j(t)=y_(j-1)(t+1) XOR (y_(j-1)(t) OR y_(j-2)(t))`.

The complete temporal columns, in residues `t=0,1,2,3`, are:

| j | column |
|---|---|
| -1 | 0011 |
| 0 | 0101 |
| 1 | 1101 |
| 2 | 0110 |
| 3 | 0011 |
| 4 | 0001 |
| 5 | 0001 |
| 6 | 0011 |
| 7 | 0101 |

This is a finite exact proof: each four-bit row checks all time residues,
and the **adjacent pair** at 6,7 equals the pair at -1,0. Determinism of
the sideways recurrence therefore repeats these seven columns indefinitely.
Consequently `e_j=1 iff j=1 mod7`, and `W_N=ceil(N/7)`. The script checks
the cyclic identity; this proof is not Lean checked. It proves the base
case conditional on an infinite alternating `u`; it does not prove that
physical seeds 32,96,160 supply that input. Their `exact7` obligations
remain distinct.

The pair and its ordered temporal phases are reusable state. A return of
one column alone, or a weight count alone, would not justify this induction.
The OR operation is used in every row; replacing it with Rule90's XOR
recurrence does not return this adjacent pair. The constant-zero target with
right zero retains its zero completion, so the argument is conditioned on
the alternating target and alternating input, not an initial-condition
invariant. Finite influence experiments do not rule out every finite-state
transducer, and no such claim is made here.

## A killed relaxed bound and the next language

Even `L3` (avoid `11,00000,101001,0100101`) does **not** imply the proposed
14-zero transient bound. The exact finite input

`u=(01)^45 000101` (96 bits)

has ones at 176 and 192, hence **15 zeros at 177–191**. It has `W_191=26`
and deficit 9. All bits needed to reconstruct through depth192 are supplied;
no unspecified extension influences this witness. This is also a direct
counterexample to the constant-eight deficit inequality over L3.

The word contains the factor **0101000101**. Agent C's separate DRAT-checked
local physical-cone calculation excludes that factor; hence this is not a
physical counterexample. It does not contain `0101010000`.

For clarity define the finite relaxation **L4** here as L3 plus the eight
additional forbidden factors certified in C's length≤12 factor experiment:

`010010001, 0101000101, 0101010000, 01010001001, 10010001001,`
`010010000101, 010100010001, 100100010000`.

This is a necessary finite-factor restriction, not an exact characterization
of the physical language and not a claim that its minimal forbidden factors
stop at length12. Proof objects and their replay belong to C's artifact.

At `m=45`, a quiet departure `w` starting `00` illustrates what these
constraints carry. The six-bit prefixes determine these exact next ones:

| tail prefix | zero run after176 | next one | survives |
|---|---:|---:|---|
| 000010 | 10 | 187 | L3 |
| 000100 | 14 | 191 | L4 |
| 000101 | 15 | 192 | L3 |
| 001000,001001,001010 | 8 | 185 | L2 |

The L3 constraint removes the `001…` branch; the additional L4 factors
remove `000010` and `000101`. These finite exclusions identify a concrete
mechanism for this example. They supply neither a general potential nor an
all-`m` identity.

## Exhaustive scope and reproduction

`runs/p2-transient-2026-10-15/results.json` contains every per-`m` maximum
and global accepted-word and censor counts for `m=1,…,48`, all **12-bit
tails**, and languages free, L1, L2, L3, L4. Membership is checked on the entire concatenation, including
junction factors. The statistic is the first gap between successive ones
that differs from seven; unobserved endpoints are explicitly censored.
The largest **uncensored** zero runs in this family are respectively
15,15,15,15,14. These maxima are not assertions about censored words or
longer tails. The L4 maximizer `m=42,tail=010101000100` has the same
96-bit input as `(01)^45 000100`, and reproduces the physical example's
first 14-zero gap.

After integration with A's reference module:

```bash
python experiments/period2_transient.py --output runs/p2-transient-2026-10-15/results.json
python -m unittest tests.test_period2_transient -v
```

Before integration, pass `--reference-file` pointing to A's module. The
script imports A for all relaxed reconstructions, compares 64 deterministic
48-bit physical rows at depth96 against `support_growth.completion`, and
checks the physical example with independent dictionary forward simulation.

Not run by this worker: full verifier, Lean, SAT/DRAT, physical realisability
search for other words, infinite seed32 boundary proof, construction of an
endpoint-bounded potential, uniform deficit proof, or external-lineage audit.
No STATUS, ledger, or experiment log was edited.
