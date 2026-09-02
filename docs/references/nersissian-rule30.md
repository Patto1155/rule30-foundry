# Tigran Nersissian — Rule 30 shortcut material

Primary/public material used by `experiments/nersissian_audit.py`:

- Wolfram Cloud notebook/paper: https://www.wolframcloud.com/obj/b04b6551-fecf-465d-b02d-63d95abd751c
- Wolfram Community discussion: https://community.wolfram.com/groups/-/m/t/1802242

## Claim being audited

The public material describes fast cell evaluation once a mathematically evaluated row/support representation is already available. That query-time statement must be kept separate from the cost of constructing the row/support representation itself.

For Prize Problem 3, the relevant accounting target is cold-start computation:

`n -> every n-dependent support object -> c_n`.

A warm `O(log n)` lookup does not by itself imply an end-to-end `o(n)` algorithm if producing the lookup structure costs `Omega(n)` or stores `Omega(n)` n-dependent information.

## Reconstructed baseline used in this repo

The current audit implements the explicit support recurrence

`S_m = Inc((S_(m-1) * S_(m-2)) Δ S_(m-1) Δ S_(m-2))`

with `S_1={0}`, `S_2={1}`. Here `*` is OR-convolution with multiplicity modulo two, and `Δ` is symmetric difference. The resulting support is evaluated using Lucas' theorem for binomial parity.

This is deliberately an explicit representation. It is **not** claimed to reproduce the source's compressed/masked dyadic-block evaluator. The next research step is a faithful reconstruction of that representation with the same cold/warm accounting interface.
