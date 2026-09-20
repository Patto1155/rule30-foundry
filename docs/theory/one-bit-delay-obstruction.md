# One fixed bit can unlock an unbounded delay

## Result

There is no constant `K` such that adding one absent initial bit always obeys

```text
q(n + 2^p) <= q(n) + K.
```

This fails even with the added-bit position fixed.  Let

```text
n_R = sum(2^(7j), j=0,...,R-1)
m_R = n_R - 2^42                         (R >= 7).
```

Then

```text
q(m_R) = 21
```

for every `R >= 7`, whereas for odd `R` the period-seven truncation theorem
gives

```text
q(n_R) = (7R-1)/2.
```

Consequently

```text
q(m_R + 2^42) - q(m_R) = (7R-43)/2
```

along odd `R`, and this difference is unbounded.

## Exact proof

The six-one truncation is

```text
n_6 = 1 + 2^7 + 2^14 + 2^21 + 2^28 + 2^35
    = 0x810204081.
```

Exact integer iteration of

```text
T_b(n) = (n >> 1) XOR (n OR (n << 1) OR b),
G = T_1 T_0
```

gives the following residues for macrotimes `k=0,...,21`:

```text
G^k(n_6) mod 4: 1212121212121212121213
```

Thus the first residue in `{0,3}` is at `k=21`, so `q(n_6)=21`.

The latch test at macrotime `k` depends only on initial indices
`0,...,2k+1`.  Hence all tests through and including `k=21` depend only on
indices `0,...,43`.  For every `R >= 7`, deleting the one at index 42 makes
`m_R` agree with `n_6` on this entire window: both have ones exactly at
indices `0,7,14,21,28,35` there.  The next possible one of `m_R` is at index
49.  Therefore the two trajectories have identical latch results through
the first failure, proving `q(m_R)=q(n_6)=21`.

Adding the missing bit at index 42 reconstructs `n_R`.  For odd `R`, its
exact period-seven formula is `(7R-1)/2`, which completes the proof.

Equivalently, the locality step used here is the general failure-shielding
lemma

```text
q(x)=h and y == x (mod 2^(2h+2))  implies  q(y)=h.
```

Indeed all passing tests before `h` and the failing test at `h` see only the
shared prefix.

## Scope and implication

This rules out any deletion or insertion induction in which one bit is
allowed only a bounded loss, even if that allowance depends on the bit's
position: the position here is always 42.  The missing bit acts as a gate.
Once restored, it unlocks survivor credit stored in an arbitrarily long
already-present outer ether tail.

The result does not rule out a bound in terms of total initial weight, a
fixed-weight bound `B_R`, or an insertion allowance that grows with the
weight or structure of the retained outer tail.  In particular, it does not
show that the outer ones should be charged to the restored bit.  It shows
that a compositional proof must preserve their dormant credit across the
interior hole; a bounded marginal-bit allowance cannot do so.

## Full finite-row version

The unbounded insertion gain also occurs for actual centre traces.  Let the
centre initially be zero, the left half be `n_R`, and the right half be

```text
v_R = sum_(j=0,...,R-1) 2^(5+7j).
```

Here right bit `j` is physical site `j+1`.  The infinite period-seven ether
has ones at physical sites `x=6 mod 7` and centre trace `0101...`.  The finite
row first differs from it at left site `-(7R+1)` and right site `7R+6`.
No difference can reach the centre before time `7R+1`.  At that time the
leftmost input in the centre's cone is the unique differing input. Iterated
left permutivity makes its coefficient one, so the centre flips.  Thus its
exact first mismatch with alternation is

```text
tau(restored row) = 7R+1.
```

For `R>=7`, delete left bit 42, the physical site `-43`.  This is now the
uniquely nearest defect; all truncation defects are at distance at least 50.
The same causal and permutivity argument gives `tau(deleted row)=43`.
Restoring this fixed bit therefore increases the actual alternating-prefix
length by `7R-42`, unboundedly, with the right half held fixed in each pair.

The independent full-row cone checker in
`tools/verify_interface_obstructions.py` replays the restored/deleted times
`50/43`, `57/43`, and `64/43` for `R=7,8,9`.  The all-`R` proof is the unique
extreme-input argument above. This concerns the first finite centre-trace
prefix, not whether a later eventual alternating trace can begin.
