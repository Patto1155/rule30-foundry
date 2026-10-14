# Forced-completion measurements

Label: **lemma discovery over finite rows; not a single-seed measurement**.

Reproduce from the repository root:

```sh
python experiments/support_growth.py --extra-witnesses runs/support-growth-2026-10-14/horizon-witness-input.json
python experiments/support_growth.py --verify runs/support-growth-2026-10-14/results.json
python -m unittest tests.test_support_growth -v
```

The instrument drives the positive half by the prescribed centre, collects its
nearest temporal column `d`, and solves the Rule30 local equation sideways:
`a = (c >> 1) XOR (c OR d)`. Each leftward iteration discards one unavailable
future time bit. This determines the next initial negative cell exactly, with
all horizons sharing the same initial positive integer. Rule90 substitutes
`XOR d` for `XOR (c OR d)`. Independent truth-table forward simulation and the
older repeated-forward triangular inversion agree on 192 cases at depth24;
an additional truth-table check crosses depth64. No use of floating arithmetic
enters completion bits or weights.

Scope: all65536 right integers below2^16 at depth128;256 deterministic samples
below2^24 at depth1024;255 distinct positive halves of existing and newly supplied
SAT witnesses at depth4096;18 selected short-horizon finalists at depth100000.
The supplied phase-one witnesses are advanced one full Rule30 step: their
positive half becomes `R_1(r)` independently of their negative half. This
preserves the finite positive-seed meaning and starts the target at phase zero.
The saved witness-input snapshot makes this conversion reproducible.

The minimum enumerated depth128 weight is19 (r32 and others). The smallest
measured depth100000 weight is14286. The seed32 completion is exactly the
period7 word with ones at depths1 mod7 through100000 (not an infinite proof).
Its finite weights are19,37,74,147,586,1429,14286 at depths128,256,512,1024,4096,
10000,100000. Slow rows look linear at these checkpoints; no finite check can
refute the existential seed-dependent alpha,C conjecture. Short-depth ranking
selects long runs: this does not establish a global minimum at depth100000.

Controls: the Rule30 constant-zero target has completion weight0 for r0,
and roughly half the depth for the tested nonzero right seeds. Rule90 with
constant-zero target and r1 has the finite completion left1 at every tested
depth. This is the separating finite-support control; no finite-completion
claim is made about Rule90's alternating target.

Benchmark for r1: depth1024 about0.00069s, depth10000 about0.020s,
depth100000 about1.28s. Whole measurement run about26.2s in this container;
record-level elapsed times are machine-dependent and not acceptance assertions.
The verifier reconstructs all listed bits and weights, the *entire*65536-row
minimum ranking, deterministic samples, and adversarial coordinate descent.

Not run: all65536 seeds to100000; all255 witness seeds to100000; larger than
100000 prefixes; a proof of asymptotic density or period-two exclusion; GPU;
Lean; outside-model audit. Full repository verification is owned by the lead
at integration, rather than duplicated concurrently by this worker.
