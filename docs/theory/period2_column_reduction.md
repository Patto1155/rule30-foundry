# Phase-zero column reduction

For Rule 30, `F(x)_i = x_(i-1) XOR (x_i OR x_(i+1))`. Prescribe
`c(t)=t mod 2`, and put `y_j(t)=x_(-j)(t)`, so `y_0=c` and
`y_-1=x_1`. Left inversion gives

```
y_j(t) = y_(j-1)(t+1) XOR (y_(j-1)(t) OR y_(j-2)(t)).
e_j = y_j(0),  W_N = sum_(j=1)^N e_j.
```

The triangular light cone requires the right temporal column only at times
`0..N-1`. Computing successive columns with decreasing time masks therefore
returns exactly the first N initial negative cells; increasing N preserves
all previously computed bits. This is a restatement of established inversion,
with an explicit implementation and independent checks.

## Three boundary facts and the corrected side

1. **Odd left latch:** `y_1(2s+1)=0 XOR (1 OR x_1(2s+1))=1`.
   At even times, `y_1(2s)=1 XOR u_s`, where `u_s=x_1(2s)`.
   Hence the entire first left column, and inductively every subsequent left
   column, depends only on u. Arbitrary changes to odd right-column bits are
   erased. The assertion concerns the **left** neighbor. The right neighbor
   is not always one at odd times: the driven right seed `r=0` has
   `x_1(1)=0`.
2. **No 11 in u:** at an even time, let the first three positive cells be
   `a,b,d`. With prescribed centres zero then one, two steps give
   `u_(s+1)=NOT a AND NOT b AND NOT d`. Thus `u_s=1` implies
   `u_(s+1)=0`. This is the existing NOR return, not a new analytic result.
3. **No 00000 in u:** this is an existing analytic result recorded in
   [renewal replay](renewal_replay.md). A separate finite local proof is
   reproducible by enumerating the nine initial positive cells: the outputs
   `u_s..u_(s+4)` have the complete set

   ```
   00001 00010 00100 00101 01000 01001
   01010 10000 10001 10010 10100 10101
   ```

   Eight physical steps need only cells 1..9 and the prescribed centre at
   times 0..7. Cells beyond 9 cannot affect these outputs, by radius-one
   locality. All 512 assignments are enumerated by `r<512`; shifting the
   origin of time to any even time gives the same local argument. The
   accompanying script checks the larger finite scope below. This local
   enumeration is not a Lean proof.

`formal/Rule30/ColumnReduction.lean` contains six Lean-checked declarations:
left inversion, local odd-left latch, local NOR return, local no-11, and
the odd-left/no-11 statements along every normalized alternating orbit.
`lake env lean Rule30/ColumnReduction.lean` checked all six declarations
with Lean v4.19.0; their printed axiom dependencies contain only `propext`
and `Quot.sound` (or no axioms), never `sorryAx`. The no-00000 fact has
no new Lean declaration in this module.

## Reference API

`experiments/period2_column_reduction.py` uses chronological temporal inputs
and the same packed spatial convention as `support_growth.py`:

- `column1(r,T)` returns a tuple of `T+1` bits at times 0 through T, inclusive.
  The positive half evolves with its centre externally fixed to `t mod 2`.
- `completion_from_u(u,N)` accepts an iterable of at least `ceil(N/2)` binary
  integer bits and returns an integer with bit `j-1=e_j`, `1<=j<=N`.
- `completion(r,N)` derives u from the finite positive integer r and returns
  the same packed completion.
- `completion_from_column1(d,N)` accepts all right temporal bits at times
  `0..N-1`; it supports the odd-bit erasure check.
- `forward_check(r,left,N)` independently simulates a full physical finite
  row with a cell dictionary and the elementary eight-entry truth table.
  It asserts the centre at every time 0 through N; the centre is not forced
  during this independent replay.

## Reproduction and exact scope

```
python -m unittest tests.test_period2_column_reduction
python experiments/period2_column_reduction.py --verify --exhaustive-bits 16
```

The second command checks:

- Free input `01010100110101101101010010010001100100010`: N=81, W=9.
- No-11 input `(01)^17 0^6`: N=79, W=10.
- Physical `r=0x98f9d8a0`: N=190, W=26, deficit `N-7W=8`,
  checked against the older packed implementation and independent forward CA.
- 200 deterministic random 128-bit seeds, N=80: odd-right-bit erasure,
  agreement with `support_growth`, independent physical forward replay.
- Every `0<=r<65536`: no 11 and no 00000 in the 41-letter even column
  at times 0,2,...,80. This is an exact finite check, not an exhaustive
  enumeration of all infinite trajectories.

The tests additionally compare 32 small cases with the independent
`period2_identity_check.forced_left` implementation, cross a 64-bit packing
boundary, check prefix consistency, and enumerate the complete nine-cell
five-output cone.

The two supplied language checkpoints are **witness weights only**. Their
minimality needs a separate lower-bound proof, such as an independently
checked UNSAT certificate; this reference module makes no minimum claim.
Likewise the physical witness establishes an attained deficit of eight, not
an all-row upper bound.

## Mechanism and limits

The erasure uses Rule 30's OR latch: replacing OR by XOR (Rule 90) makes the
odd first-left column depend on the right input. The forced alternating
centre also excludes the all-zero full orbit, which remains a zero orbit
under the ordinary forward rule. These are universal conditional finite-row
identities, not statistical evidence about the single prize seed.

Neither boundary-word avoidance nor finite witness weights prove an initial
left density bound, infinite support, or period-two exclusion. No SAT/DRAT
minimum proofs, GPU experiment, asymptotic density claim, or all-repository
verifier run is part of this reference module's focused acceptance.
