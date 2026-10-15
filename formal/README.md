# Checked Rule 30 foundations

Pinned Lean 4.19.0, importing Lean core only (no Mathlib dependency).

```sh
cd formal
lake update
lake build
# Or, from repository root:
python tools/verify_lean.py
```

The verifier builds the project, rechecks every project `.lean` source outside
`.lake`, discovers every `theorem`/`lemma` declaration (including nested
namespaces and modifiers), and generates an independent axiom audit importing
all project modules. Handwritten `#print` lines are not required. Nested comments
and string literals cannot create fake declarations. Unsupported declaration
names or namespaces fail closed. It rejects `sorry` warnings and `sorryAx`, and
allows only `propext`, `Classical.choice`, and `Quot.sound`.

Acceptance checked **47 declarations**: 41 in `Basic.lean`, six in
`ColumnReduction.lean`. `checked-axioms.txt` contains the actual generated audit
and version; rerun the verifier to check the current sources. Ten focused Python
verifier tests passed. The integrated build uses Agent A's column module.

| Lemmas | Exact scope |
|---|---|
| `zero_latch`, `alternating_return` | Rule 30 OR identities by Boolean case analysis |
| `step_finite`, `evolve_finite`, `evolve_add`, `trace_shift` | Finite-support and time-shift infrastructure |
| `onset_reduction_evolve`, `onset_reduction` | An alternating tail gives a finite nonzero row with normalized trace; the stronger form records `T` and `y = evolve x T`, preserving the original interface |
| `localRule_left_injective`, `right_half_evolve_unique`, `triangular_uniqueness` | Same initial nonnegative half and complete centre trace force equality of entire rows, without a finiteness assumption (Condrey Lemma 1) |
| `oneLeft_adjacent`, `oneLeft_invariant`, `leftOne_step`, `leftOne_evolve`, `leftOne_constant_trace`, `oneCompletion_left` | Constant-one checkerboard construction |
| `constant_one_complete`, `leftOne_not_finite`, `finite_not_constant_one` | Entire constant-one fibre and its exclusion for finite rows (Condrey Theorem 3) |
| `zeroLeft_update`, `zeroClass_centre`, `zeroClass_step`, `zeroClass_trace`, `zeroCompletion_class`, `zeroCompletion_trace` | Constant-zero construction when the first positive one is at `m`; the invariant class moves to `max 1 (m-1)` (Condrey Theorem 2 existence) |
| `constant_zero_complete`, `zero_evolve`, `constant_zero_empty_right` | Completeness: the explicit left completion is forced; an empty nonnegative right half forces the zero row (Theorem 2) |
| `first_right_one`, `zeroClass_not_finite`, `finite_nonzero_not_constant_zero` | Constant-zero exclusion for finite **nonzero** rows, retaining the zero-row control |
| `ring7_step0`–`ring7_step3`, `ring7_boundary` | Four-step seven-cell ring orbit and boundary bits |
| `ringIndex7_left`, `ringIndex7_right`, `lift_step` | Ring-to-infinite-periodic lift with index `(-position) mod 7`; increasing ring indices point physically left. The lift commutes with the standard Rule 30 step |
| `temporal_phase_slip` | Exact Boolean/integer temporal coboundary under adjacent-11-free hypotheses; no spatial transport conclusion |
| `column_left_inversion`, `odd_left_latch`, `column_right_return`, `column_no11`, `alternating_odd_left`, `alternating_u_no11` | Agent A's local column identities, odd-time **left** latch, and physical no-adjacent-11 even right output |

`AlternatingTail` asserts an alternating formula with either phase. No theorem
excludes eventual period two. The ring lift is spatially periodic and does not
supply a finite physical right half. Triangular uniqueness concerns the complete
trace; a separate finite-prefix/light-cone theorem is not asserted here.

The constant-fibre constructions and completeness are now checked. Extending
the constant-trace exclusions to arbitrary eventual onsets still requires a
finite-nonzero evolution lemma. Sharp prefix horizons and Condrey Corollary 5
are not formalized. The no-`00000` constraint is not formalized in Lean.

Rule 90 filter: the fibre constructions use the OR explicitly; triangular
uniqueness, finiteness, and time shifts alone are infrastructure. Zero filter:
`finite_nonzero_not_constant_zero` explicitly requires nonzero, while the empty
right-half theorem retains the zero configuration. Normalized alternating tails
exclude zero by their trace hypothesis.

On sandboxes lacking `/proc/<pid>/exe`, the verifier detects the exact Lean
startup error and compiles `tools/lean_proc_path.c` to redirect executable-path
discovery to `/proc/self/exe`. It changes no proof evaluation or kernel logic.
This acceptance run used the normal toolchain path without that shim.

Not run by the formalization worker: full repository suite, SAT/DRAT, GPU,
external lineage audit, period-two exclusion, physical-boundary/support-growth
proof, eventual-constant extension, or sharp-horizon formalization. The lead
owns full integration validation.
