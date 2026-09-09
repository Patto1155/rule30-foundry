# Seven-step follow-up: repository hygiene and bounded probes

- Date: 2026-09-09
- Base: main after PR #37.
- Goal: execute the seven bounded next steps selected after the period-word
  sieve without turning method checks into prize claims.

## 1. Official prize numbering

PR #36 corrected active documentation to the official order: Problem 1 is
nonperiodicity, Problem 2 is equidistribution, and Problem 3 is computational
effort. Archived handovers retain their historical text. Both CI jobs passed.

## 2–3. Test-branch audit and import

All twelve `codex/test-*` branches shared base `222c9ccd`, and none of their
target experiment modules changed between that base and current main. Their
test files passed separately. PR #37 imported 173 test methods and passed both
CI jobs. The orbit-cycle branch's one production edit was excluded: its Floyd
phase-two change is incorrect, while its tests confirm main's implementation.
After verifying every imported test blob on main, all twelve obsolete remote
branches were deleted atomically with exact-tip leases.

## 4. Canonical bitstream availability

Neither `center_col_10M.bin` nor `center_col_46M.bin` is available under the
project drive, AgentPlatform storage, WSL home/temp, or Downloads. The GitHub
repository has no release assets, Actions artifacts, or LFS objects containing
them. The manifest still anchors their expected SHA256 values:

- 10M: `6f8670b4a89826c8228d6a165047792e91551dedfb2853b8f12572d466b7547e`
- 46M: `f281f11bb5d93132248213b3b353857814fd4b8dc754c606e846f3c2c3f67b5c`

Consequently local `verify_all` must still report both bitstream stages as
explicit skips. A3 remains the blocker for the 46M exact-period extension.

## 5. A small actual-transient branch selector

```bash
python experiments/zero_word_regression.py --pretty \
  --out runs/b1-selector-2026-09-09.json
```

The five existing gates pass. At the known period-16 ambiguity, the actual
seed has equal predecessor words `0xd0d0`, zero at diagonal 399, and then
chooses `0x9f60` rather than complementary candidate `0x609f`. The alternatives
lead to distinct later collision words. This is an executable small example of
the missing branch selector. It does not select the first recovered period-32
branch at `d=1,420,878,969`; doing that requires transient information at that
distance, so B1's actual period-32 diagonal remains unresolved.

## 6. Saved loops against actual seed strips

```bash
python experiments/seed_strip_compatibility.py \
  --loops runs/periodic-boundary-2026-09-08/output-loops.json \
  --max-onset 4096 --out runs/seed-strip-compatibility-2026-09-09.json
python experiments/seed_strip_compatibility.py \
  --loops runs/periodic-boundary-2026-09-08/output-loops.json \
  --verify runs/seed-strip-compatibility-2026-09-09.json
```

The search starts comparisons only when an actual centered seed strip equals
a saved loop state exactly. It finds 5,574 such alignments. In 4,503, the left
neighbor disagrees before the actual center stops following the loop's
alternating trace, so this is information beyond an ordinary center mismatch.
Only one width-5 alignment follows an entire saved four-step strip block; no
other full block does. Tests compare the packed seed evolution with an
independent cell dictionary beyond 64 steps and reject altered artifacts.

This eliminates these two particular saved walks at most alignments. It does
not eliminate other open-strip paths, all later onsets, or an eventually
periodic actual center. It therefore supplies no single-seed nonperiodicity
proof.

## 7. One DFAO extension benchmark

The targeted base-2 MSD instance at `n=56`, 12 states, previously timed out in
the Python solver harness. Standalone CaDiCaL 3.0.1 solved it UNSAT in 41.113 s;
the 197,932,048-byte proof was accepted by `drat-trim` in 48.162 s. The stable
CNF SHA256 is
`592ecdbd5a6d15852ad6612a1c6fb0eb88cb5d9345216cce86c3191c80d506d1`.

```bash
python tools/verify_dfao_n56.py \
  --artifact runs/dfao-n56-msd-s12-2026-09-09.json
# Fast static check above. Full certificate reproduction:
bash tools/build_sat_toolchain.sh
python tools/verify_dfao_n56.py \
  --artifact runs/dfao-n56-msd-s12-2026-09-09.json --reprove
```

The verifier also checks the existing directly certified `s*(48)=12` MSD
anchor and independently evaluates the existing 13-state witness on all 56
bits. Prefix monotonicity transfers the certified exclusions of states 1..11
from 48 to 56 bits; the new DRAT proof excludes state 12. Therefore the exact
MSD value is `s*(56)=13`. The conservative counting-null threshold is 10
states, so the result is outside the vacuous regime.

The run justifies extending the MSD curve: the formerly timed-out instance
completed with proof checking in under 90 seconds. The next bounded target is
MSD `n=64`; the LSD `n=56` frontier should first be benchmarked one state at a
time because three states previously timed out.

## 8. Transient invariant branch selector

```bash
python experiments/transient_branch_selector.py --steps 130000 \
  --out runs/transient-branch-selector-2026-09-09.json
python experiments/transient_branch_selector.py --steps 130000 \
  --verify runs/transient-branch-selector-2026-09-09.json
```

The seven zero words in the existing exact `d < 1,000,000` simulation are the
complete reachable training and validation set: `2, 7, 28, 399, 53207, 58286,
87866`. The first four are the chronological training cases and the last three
are held out. A selector using only the phase of the zero diagonal's last 1
succeeds on 3/7. It fails because the predecessor is itself still transient at
four final resets, including `d=399`.

After both inputs reach their periodic tails, the phase plus the successor's
one-bit boundary value selects the correct continuation on 4/4 training and
3/3 validation cases. At the largest reachable event this is six stored bits.
This is a compact *representation* of the missing information, but no compact
way to compute it was found: the exact tail cutoff grows from 1 to 117,323 and
is about `1.335*d` at the largest event. Six simple selectors derived only from
the diagonal index and settled predecessor reach at most 2/3 held-out cases;
the best training rule reaches only 1/3 held-out.

The stop condition therefore fired. The experiment does not simulate toward
the period-32 ambiguity at `d=1,420,878,969`. Promotion requires a recurrence
that propagates the boundary bit from bounded augmented state without scanning
the growing transient; merely naming the bit does not compress its computation.
