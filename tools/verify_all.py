#!/usr/bin/env python
"""One command for trust: run every integrity check this repo has.

Trust used to be four commands and some tribal knowledge. That is the wrong
price: if checking the repo is cheap, agents run it before every experiment;
if it is expensive, they skip it, and you get another five-month bug computed
on a byte-reversed bitstream.

Stages, in order (each is independently runnable; see its own --help):

  clone-integrity    no LF->CRLF corruption from git's initial checkout
  golden-self-test   naive == packed, OEIS A051023 prefix matches
  manifest+golden    every tracked artifact still hashes the same
  manifest-current   data/MANIFEST.sha256 is what make_manifest regenerates
  bitstream:<name>   each canonical bitstream agrees with the golden reference
  gates-trap         tools/gates.py still refuses the known-vacuous manifest
  lint-bitorder      no bare np.packbits / np.unpackbits
  lint-ledger        no ledger row cites evidence that is not there
  unittest           python -m unittest discover -s tests

A stage whose input is absent is reported SKIP, not FAIL: the canonical
bitstreams are gitignored, so a fresh clone legitimately cannot run them. SKIP
is printed loudly rather than hidden, because "everything passed" on a machine
that checked nothing is the failure mode this tool exists to prevent.

SKIP is honest but it is not evidence. In CI, where nobody reads the log, an
unexpected SKIP is indistinguishable from a pass -- so `--allow-skip` names the
stages whose inputs are known to be absent and turns every *other* SKIP into a
failure. A stage that starts skipping because its input quietly vanished then
breaks the build instead of going green.

Usage:
    python tools/verify_all.py
    python tools/verify_all.py --verbose     # stream each stage's output
    python tools/verify_all.py --allow-skip 'bitstream:*' --allow-skip drat-toolchain
"""

from __future__ import annotations

import argparse
import fnmatch
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
BITSTREAMS = ("data/center_col_10M.bin", "data/center_col_46M.bin")

PASS, FAIL, SKIP = "PASS", "FAIL", "SKIP"


class Stage:
    __slots__ = ("name", "argv", "skip_reason")

    def __init__(self, name: str, argv: list[str],
                 skip_reason: str | None = None):
        self.name = name
        self.argv = argv
        self.skip_reason = skip_reason


def build_stages() -> list[Stage]:
    py = sys.executable
    stages = [
        # First: a bad checkout makes every later stage fail with hash
        # mismatches that read like corrupted data rather than a git filter.
        Stage("clone-integrity", [py, "tools/check_clone_integrity.py"]),
        Stage("golden-self-test",
              [py, "tools/gen_golden_reference.py", "--self-test"]),
        Stage("manifest+golden", [py, "tools/verify_data.py", "--all"]),
        Stage("manifest-current", [py, "tools/make_manifest.py", "--check"]),
    ]
    for rel in BITSTREAMS:
        present = (REPO_ROOT / rel).exists()
        stages.append(Stage(
            f"bitstream:{Path(rel).name}",
            [py, "tools/verify_data.py", "--bitstream", rel],
            skip_reason=None if present else
            "not present (gitignored; regenerate with gpu/rule30_sim.py)"))
    toolchain = [REPO_ROOT / "third_party" / n for n in ("cadical", "drat-trim")]
    stages.append(Stage(
        "drat-toolchain",
        [py, "experiments/dfao_drat_proofs.py", "--self-test"],
        skip_reason=None if all(t.exists() for t in toolchain) else
        "SAT toolchain absent (build it: bash tools/build_sat_toolchain.sh)"))
    # The trap manifest is the shape of the experiment whose certificate was
    # retracted in 2026-08: a DFAO class far below the prefix it is tested
    # against. `--expect-fail` passes only when the gate still refuses it. A
    # gate that silently stops gating is worse than no gate, because the
    # runner would then report PASS on exactly the run that must not happen.
    trap = REPO_ROOT / "queue" / "trap-vacuous-dfao.json"
    stages.append(Stage(
        "gates-trap",
        [py, "tools/gates.py", "preflight", "queue/trap-vacuous-dfao.json",
         "--no-external", "--expect-fail"],
        skip_reason=None if trap.exists() else
        "queue/trap-vacuous-dfao.json absent"))
    stages += [
        Stage("lint-bitorder", [py, "tools/lint_bitorder.py"]),
        Stage("lint-ledger", [py, "tools/lint_ledger.py"]),
        Stage("unittest", [py, "-m", "unittest", "discover", "-s", "tests"]),
    ]
    return stages


def skip_permitted(name: str, allowed: tuple[str, ...] | None) -> bool:
    """Is this stage allowed to SKIP without failing the run?

    `allowed is None` means --allow-skip was never passed: the permissive
    interactive default, where every SKIP is fine. Otherwise only stage names
    matching one of the globs may skip. Kept separate from main() so it can be
    tested without re-entering verify_all, which would recurse through its own
    unittest stage.
    """
    if allowed is None:
        return True
    return any(fnmatch.fnmatch(name, pat) for pat in allowed)


def run_stage(stage: Stage, verbose: bool) -> tuple[str, float, str]:
    if stage.skip_reason:
        return SKIP, 0.0, stage.skip_reason
    started = time.time()
    proc = subprocess.run(stage.argv, cwd=REPO_ROOT,
                          capture_output=not verbose, text=True)
    elapsed = time.time() - started
    output = "" if verbose else (proc.stdout or "") + (proc.stderr or "")
    return (PASS if proc.returncode == 0 else FAIL), elapsed, output


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--verbose", action="store_true",
                    help="stream each stage's output instead of capturing it")
    ap.add_argument("--allow-skip", action="append", metavar="GLOB",
                    help="stage name (fnmatch glob) that is permitted to SKIP. "
                         "Passing this at all switches on strict mode: a SKIP "
                         "by any stage NOT matching one of these globs becomes "
                         "a failure. Repeatable.")
    args = ap.parse_args()

    # None means the flag was never passed -> permissive, the interactive
    # default. Anything else, including an empty list, means strict.
    allowed = None if args.allow_skip is None else tuple(args.allow_skip)

    stages = build_stages()
    width = max(len(s.name) for s in stages)
    results = []
    failures = []
    unexpected_skips = []

    for stage in stages:
        if args.verbose:
            print(f"\n=== {stage.name} ===", flush=True)
        status, elapsed, output = run_stage(stage, args.verbose)
        results.append((stage.name, status, elapsed))
        if status == SKIP:
            permitted = skip_permitted(stage.name, allowed)
            marker = SKIP if permitted else "SKIP!"
            print(f"{marker:<{len(SKIP)}}  {stage.name:<{width}}  {output}",
                  flush=True)
            if not permitted:
                unexpected_skips.append(stage.name)
        else:
            print(f"{status}  {stage.name:<{width}}  {elapsed:6.1f}s",
                  flush=True)
        if status == FAIL:
            failures.append((stage, output))

    n_pass = sum(1 for _, s, _ in results if s == PASS)
    n_skip = sum(1 for _, s, _ in results if s == SKIP)
    n_fail = len(failures)

    print("-" * (width + 22))
    for stage, output in failures:
        print(f"\n--- {stage.name} output ---")
        print(output.rstrip() or "(no output)")
        print(f"--- rerun: {' '.join(stage.argv)}")

    if unexpected_skips:
        print("\n--- unexpected SKIP ---")
        for name in unexpected_skips:
            print(f"  {name} skipped, and --allow-skip does not cover it.")
        print("  Either restore the stage's input or say so explicitly with "
              "--allow-skip.")

    verdict = "FAIL" if (n_fail or unexpected_skips) else "OK"
    print(f"\nverify_all: {n_pass} passed, {n_skip} skipped "
          f"({len(unexpected_skips)} unexpected), {n_fail} failed  {verdict}")
    if n_skip and not n_fail and not unexpected_skips:
        print("  Note: skipped stages checked nothing. The canonical "
              "bitstreams are")
        print("  the artifacts the prize-facing claims rest on; run this on a "
              "machine")
        print("  that has them before treating the repo as verified.")
    return 1 if (n_fail or unexpected_skips) else 0


if __name__ == "__main__":
    raise SystemExit(main())
