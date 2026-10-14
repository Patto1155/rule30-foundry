# Period-ladder infrastructure replay — 2026-10-14

Infrastructure only. Rule 90 test and zero test: not applicable; no mathematical claim is made.

## Bitstream diagnosis and restoration

The tracked golden file is 1,250,000 bytes. `tools/gen_golden_reference.py` and `AGENTS.md` explicitly document MSB-first golden packing; canonical `data/center_col_*.bin` files are LSB-first.

Before attributing a cause, freshly simulated the first 100,000 steps with `tools.gen_golden_reference.center_packed` (2.283 seconds), decoded the fresh output MSB-first, and checked its first 4096 bits against `center_naive(4096)`.

| Decode / comparison | First divergent index | Differing bits / 100,000 |
|---|---:|---:|
| Golden decoded little versus fresh simulation | 0 | 49,944 |
| Golden decoded big versus fresh simulation | None | 0 |
| Fresh packed versus naive, first 4096 | None | 0 |

The fresh prefix was `110111001100010110010011`; golden little gave `001110111010001111001001`.

The mismatch is exactly a packing convention change, proved beyond the sampled prefix by the converted full-file SHA256 matching the independently anchored canonical manifest entry.

- Golden SHA256: `9c9e302620bab806c0f9e1b697d68ba93ca334c10ddea095ec7b8138c200c4e4` (manifest entry 37).
- Converted canonical SHA256: `6f8670b4a89826c8228d6a165047792e91551dedfb2853b8f12572d466b7547e` (manifest entry 182).
- `tools/convert_golden_bitstream.py` checks source and destination anchors, a fresh 100k CPU prefix, and a naive4096 check before writing. It leaves the independent golden source unchanged.

Acceptance run:

```bash
python tools/convert_golden_bitstream.py
sha256sum data/center_col_10M.bin
python tools/verify_data.py --bitstream data/center_col_10M.bin
python tools/lint_bitorder.py
```

All passed. `verify_data` compared all 10,000,000 decoded bits to the golden reference.

## SAT toolchain

`bash tools/build_sat_toolchain.sh` completed locally, using its unchanged defaults. CI skip configuration was not modified.

- CaDiCaL version: 3.0.1.
- CaDiCaL source commit: `c60730422e758ef1cebe7aeddf2dda31c996bf04`.
- CaDiCaL executable SHA256: `8669559c89600c75a0cf013dd9b7fcb5f6c7f125ef36a2033b2e89ca682b074c`.
- drat-trim source SHA256: `d834b649f437e091597f5347f259b9f681087f89ca0844d0cee250a1a1a0c2ee`.
- drat-trim executable SHA256: `e4f27f613745bacdbce174fbc9609bc8a52c299015e6d3ab68f996d42d6248d2`. Upstream C source has no version flag; the source hash identifies this build.
- Executables are `third_party/cadical` and `third_party/drat-trim` (gitignored).

`python experiments/dfao_drat_proofs.py --self-test` passed SAT and UNSAT controls, independent DRAT acceptance, truncated-proof rejection, and rejection of a refutation of a satisfiable formula.

Targeted n56 acceptance command:

```bash
python tools/verify_dfao_n56.py --artifact runs/dfao-n56-msd-s12-2026-09-09.json --reprove
```


Fresh replay passed: 984 variables, 14,164 clauses, UNSAT in 36.589 seconds, independently DRAT-VERIFIED in 41.342 seconds. The CNF SHA256 was `592ecdbd5a6d15852ad6612a1c6fb0eb88cb5d9345216cce86c3191c80d506d1`; the 197,932,048-byte fresh proof hash was `e23f79254d50a644ff7a12c8888543e26ee8cf352b79363adc70f2c5bba4e00c`, exactly the tracked certificate hash. The verifier's static n48 lower-bound and n56 13-state witness checks also passed, reproducing existing `s_star_56_msd = 13`.

The replay tool uses a temporary directory and removes the raw fresh proof after checking; retained replay metadata is `queue/results/period-ladder-n56-reproof.json`. No new finite claim or changed grade is implied.

## Full verifier and not run

The pre-change mandatory `python tools/verify_all.py` finished with 8 PASS, 3 SKIP, 0 FAIL; unittest took 87.2 seconds. The three SKIPs were 10M, 46M, and drat-toolchain. Parent owns the post-integration full verifier; this worker used targeted checks after restoration.

Not run:

- No fresh 1M or 10M regeneration: unnecessary after exact full-file reencoding/hash proof.
- No 46M restoration or regeneration: that golden source is absent and the task did not require this costly regeneration.
- No GPU or CuPy tests: environment has no GPU or CuPy.
- No new full verify_all after changes in this worktree: parent requested one integrated owner to avoid concurrent suites.
- No upstream CaDiCaL `make test`: repository independent checker self-test was run instead.
- No commits, CI run, proof claim, or ledger/status/log edits.
