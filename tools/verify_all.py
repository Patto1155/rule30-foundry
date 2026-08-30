#!/usr/bin/env python
"""One command for trust: run every integrity check this repo has.

Trust used to be four commands and some tribal knowledge. That is the wrong
price: if checking the repo is cheap, agents run it before every experiment;
if it is expensive, they skip it, and you get another five-month bug computed
on a byte-reversed bitstream.

Stages, in order (each is independently runnable; see its own --help):

  golden-self-test   naive == packed, OEIS A051023 prefix matches
  manifest+golden    every tracked artifact still hashes the same
  manifest-current   data/MANIFEST.sha256 is what make_manifest regenerates
  bitstream:<name>   each canonical bitstream agrees with the golden reference
  lint-bitorder      no bare np.packbits / np.unpackbits
  lint-ledger        no ledger row cites evidence that is not there
  unittest           python -m unittest discover -s tests

A stage whose input is absent is reported SKIP, not FAIL: the canonical
bitstreams are gitignored, so a fresh clone legitimately cannot run them. SKIP
is printed loudly rather than hidden, because "everything passed" on a machine
that checked nothing is the failure mode this tool exists to prevent.

Usage:
    python tools/verify_all.py
    python tools/verify_all.py --verbose     # stream each stage's output
"""

from __future__ import annotations

import argparse
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
    stages += [
        Stage("lint-bitorder", [py, "tools/lint_bitorder.py"]),
        Stage("lint-ledger", [py, "tools/lint_ledger.py"]),
        Stage("unittest", [py, "-m", "unittest", "discover", "-s", "tests"]),
    ]
    return stages


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
    args = ap.parse_args()

    stages = build_stages()
    width = max(len(s.name) for s in stages)
    results = []
    failures = []

    for stage in stages:
        if args.verbose:
            print(f"\n=== {stage.name} ===", flush=True)
        status, elapsed, output = run_stage(stage, args.verbose)
        results.append((stage.name, status, elapsed))
        if status == SKIP:
            print(f"{SKIP}  {stage.name:<{width}}  {output}", flush=True)
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

    verdict = "FAIL" if n_fail else "OK"
    print(f"\nverify_all: {n_pass} passed, {n_skip} skipped, "
          f"{n_fail} failed  {verdict}")
    if n_skip and not n_fail:
        print("  Note: skipped stages checked nothing. The canonical "
              "bitstreams are")
        print("  the artifacts the prize-facing claims rest on; run this on a "
              "machine")
        print("  that has them before treating the repo as verified.")
    return 1 if n_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
