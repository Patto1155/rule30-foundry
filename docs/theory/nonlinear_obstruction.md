# The additive closed form and Rule 30's quadratic forcing

Every identity below is **proved algebraically** over F₂ for all times and
every finite initial row. The finite checks are separate safeguards, not the
proof. This identifies an obstruction to one homogeneous additive derivation;
it gives **no computational lower bound**, exclusion of eventual periods, or
impossibility theorem for alternative coordinates.

## 1. Correcting the baseline

Write `a_i` for a row and use geometric coordinates increasing to the right.
The exact algebraic normal forms are

\[
\begin{aligned}
N_{30}(a)_i &= a_{i-1}+a_i+a_{i+1}+a_i a_{i+1},\\
H_{90}(a)_i &= a_{i-1}+a_{i+1},\\
L_{150}(a)_i &= a_{i-1}+a_i+a_{i+1}.
\end{aligned}
\]

**Proof.** `b OR c = b+c+bc` in F₂, so substituting into Rule 30's
`left XOR (center OR right)` gives the first line. The other two are their
truth-table definitions. Each of the eight neighborhoods is checked by the
script below.

Rule 30 therefore differs from Rule 90 by **both a centre linear term and a
quadratic term**. Comparing directly to Rule 90 and calling the quadratic term
the sole difference is false. First pass from Rule 90 to Rule 150, which is
still additive; only then isolate Rule 30's nonlinear forcing.

For a finite row define its Laurent polynomial
`F(z)=Σ_i a_i z^i`. Set

\[
H(z)=z^{-1}+z,\qquad L(z)=z^{-1}+1+z,\qquad
Q(a)(z)=\sum_i a_i a_{i+1}z^i.
\]

Multiplication by H or L is exactly the corresponding additive update.
Finite propagation preserves finite Laurent support at every finite time.

## 2. What survives from the Lucas derivation

For Rule 90, with initial polynomial F₀,

\[
F_t^{90}=H^t F_0,
\qquad
H^t=\sum_{k=0}^t\left(\binom tk\bmod2\right)z^{2k-t}.
\]

If `t=Σ_j ε_j 2^j` with binary digits εⱼ, repeated squaring in characteristic
two gives the usual Lucas parity product:

\[
H^t=\prod_{j:\epsilon_j=1}
\bigl(z^{-2^j}+z^{2^j}\bigr).
\]

**Proof.** The binomial theorem gives the first expansion. In F₂,
`(x+y)^(2^j)=x^(2^j)+y^(2^j)`, so factoring t into its binary digits gives the
second. Equating their coefficients is the binomial-parity identity; no
unverified literature lemma is needed here.

Exactly the same characteristic-two step is valid for Rule 150:

\[
F_t^{150}=L^tF_0,\qquad
L^t=\prod_{j:\epsilon_j=1}
\bigl(z^{-2^j}+1+z^{2^j}\bigr).
\]

The trinomial product may contain cancellations between equal exponents;
there is no claim that each factor choice produces a distinct monomial.
In particular, for `t=2^j`,

\[
(L^{2^j}a)_i=a_{i-2^j}+a_i+a_{i+2^j}.
\]

Thus the extra **linear** centre term changes the Green kernel but does not
destroy the sparse binary-time additive formula. For the single-cell seed,
Rule 90 has centre trace `1,0,0,…`: odd times have no central monomial; at
positive even time `2k`, `binom(2k,k)` is even. The product proves the latter
too: a signed sum of distinct positive powers of two cannot be zero because
its largest term exceeds the sum of all smaller ones. Rule 150 instead has
centre value 1 at every positive dyadic time.

## 3. The exact replacement for the homogeneous formula

Let aₜ be the actual Rule 30 row, Fₜ its polynomial, and Qₜ=Q(aₜ). Then

\[
F_{t+1}=LF_t+Q_t,
\qquad
\boxed{F_t=L^tF_0+\sum_{s=0}^{t-1}L^{t-1-s}Q_s.}
\tag{1}
\]

**Proof.** The recurrence is the ANF. For t=0 the empty sum vanishes. Apply L
to the formula at t, add Qₜ, and obtain the formula at t+1. Induction proves
(1) for all times. This is the discrete Duhamel identity with Rule 150's
explicit additive Green kernel.

Let `K_n(r)=[z^r]L(z)^n`. Equation (1) includes the exact centre formula

\[
a_t(0)=\sum_i K_t(-i)a_0(i)
+\sum_{s=0}^{t-1}\sum_i K_{t-1-s}(-i)a_s(i)a_s(i+1).
\tag{2}
\]

**Proof.** Extract the coefficient of z⁰ in (1). Every sum is finite because
each row and each kernel has finite support. The K kernels have the explicit
binary product in section 2, but the forcing factors involve the unknown
intermediate Rule 30 orbit. Equation (2) is exact and is not a shortcut unless
those factors can themselves be obtained more cheaply.

At a dyadic time this gives

\[
a_{2^j}(i)=a_0(i-2^j)+a_0(i)+a_0(i+2^j)
+\left[\sum_{s<2^j}L^{2^j-1-s}Q_s\right]_i.
\tag{3}
\]

The Frobenius step for **L survives**; discarding the second line of (3) is
the exact invalid step when transporting the homogeneous additive derivation
to Rule 30. Its coefficient may vanish through cancellations at some sites
or times; nonzero individual Qₛ does not imply a nonzero final remainder.

## 4. Failure of superposition and the prize seed

For any rows a and b,

\[
N_{30}(a+b)_i+N_{30}(a)_i+N_{30}(b)_i
=a_i b_{i+1}+b_i a_{i+1}.
\tag{4}
\]

**Proof.** All three linear terms cancel. Expand
`(a_i+b_i)(a_(i+1)+b_(i+1))`; the two pure quadratic terms cancel, leaving
the two mixed terms. The script exhausts the 16 assignments of these four
bits. For a=δ₀ and b=δ₁, the defect is δ₀, an explicit finite-row
counterexample to superposition. Hence the homogeneous linear-propagator
method fails globally in the original cell variables. This does **not**
establish that every possible change of variables must fail.

On the actual single-cell prize seed, F₀=1 and Q₀=0. Both Rules 30 and 150
have F₁=z⁻¹+1+z, so Q₁=z⁻¹+1. Equation (1) gives

\[
F_2^{30}=L^2+Q_1=z^{-2}+z^{-1}+z^2,
\qquad F_2^{150}=z^{-2}+1+z^2.
\]

The first quadratic correction to the centre is therefore at t=2:
`a_2^30(0)=0`, while `a_2^150(0)=1`. This establishes relevance of the forcing
to the actual seed, without extrapolating a two-step fact into hardness.

## 5. Required proof filters

**Rule 90 test — passes.** Introduce the switches
`N_(μ,λ)(a)=H a+μ a+λ Q(a)`. Rule 30 is (μ,λ)=(1,1), Rule 150 is (1,0),
and Rule 90 is (0,0). The induction in (1) applies with L replaced by H+μ
and every forcing term multiplied by λ. At λ=0 the quadratic remainder is
identically zero. At μ=0 as well, the original Rule 90 Lucas formula is
recovered. The additivity defect (4) also carries a factor λ. Thus the
obstruction uses the quadratic portion of the OR latch; it does not wrongly
reject Rule 90's additive shortcut or exclude its periodic centre column.

**Zero test — passes.** For F₀=0 all three dynamics stay zero and every Qₛ=0.
All displayed identities specialize to `0=0`. No conclusion of
nonperiodicity, nonzero forcing, or computational hardness is inferred for
this orbit. More generally, the all-time algebra works for every initial
condition, which is legitimate for an identity and insufficient for a
single-seed lower bound.

## 6. Check scope and remaining obligation

Acceptance:

```bash
python experiments/nonlinear_obstruction_check.py
python tools/lint_ledger.py
```

The first command compares independent Wolfram truth-table dynamics with the
binomial Rule 90 kernel, Rule 150 Frobenius kernel, and full Rule 30 Duhamel
sum. It checks **all 32 five-cell initial rows with infinite zero exterior,
at every integer time from 0 to 64 inclusive** (2,080 seed/time pairs).
It also exhausts the local ANF and mixed-product identities, checks all
dyadic kernels in range, includes the all-zero row, and independently
replays the single-seed first correction. It exhausts a specified small seed
class, **not all configurations whose light cone reaches time 64**. There is
no periodic finite-tape boundary and no discarded support.

The narrowly located algebraic obstruction is now an exact formula. What
remains open for Prize 3 is whether the **seed-specific centre forcing sum**
in (2) admits a useful compressed or rapidly accessible representation, or
whether a lower bound can be proved in an explicitly specified computation
model. A proof that a particular change of variables cannot remove this
forcing would need its allowed transformations defined and a separate
argument. None is supplied here. This work establishes neither a new prize
rung nor an unconditional complexity barrier.
