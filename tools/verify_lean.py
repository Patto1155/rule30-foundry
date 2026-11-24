#!/usr/bin/env python
"""Check the Lean 4 proofs under formal/ and refuse anything short of a proof.

    python tools/verify_lean.py            # exit 0 only if every file checks
    LEAN=/path/to/bin/lean python tools/verify_lean.py

Passing means, for each file: `lean` exits 0, the source contains no `sorry`
or `axiom` declaration, every listed theorem is reported by `#print axioms`
with nothing beyond Lean's three standard axioms, and the statement text the
ledger quotes is still present verbatim (so a proof cannot be "kept green" by
weakening what it says).

Toolchain: Lean 4.19.0, core library only - no Mathlib, no lake build.
    https://github.com/leanprover/lean4/releases/tag/v4.19.0
Exit code 2 means the toolchain is absent, which is SKIP, never PASS.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LEAN_VERSION = "4.19.0"
ALLOWED_AXIOMS = {"propext", "Classical.choice", "Quot.sound"}

FILES = {
    "formal/Rule30/PeriodOne.lean": {
        "theorems": [
            "Rule30.no_eventually_constant_column",
            "Rule30.centre_column_not_eventually_constant",
            "Rule30.centre_column_gap",
        ],
        "statements": [
            "def localRule (l c r : Bool) : Bool := l ^^ (c || r)",
            "fun i => localRule (x (i - 1)) (x i) (x (i + 1))",
            "∃ R : Int, ∀ i : Int, (i < -R ∨ R < i) → x i = false",
            "def seed : Configuration := fun i => decide (i = 0)",
            ("theorem no_eventually_constant_column {x : Configuration} "
             "(hf : Finite x)\n    (hn : ∃ i, x i = true) (c : Int) (b : Bool) :\n"
             "    ¬ ∃ T : Nat, ∀ t : Nat, T ≤ t → evolve x t c = b := by"),
            ("theorem centre_column_not_eventually_constant (T : Nat) (b : Bool) :\n"
             "    ∃ t : Nat, T ≤ t ∧ evolve seed t 0 = b := by"),
            ("theorem centre_column_gap (T : Nat) (b : Bool) :\n"
             "    ∃ t : Nat, T ≤ t ∧ t ≤ 4 * T + 2 ∧ evolve seed t 0 = b := by"),
        ],
    },
}


def find_lean() -> str | None:
    cand = os.environ.get("LEAN") or shutil.which("lean")
    return cand if cand and Path(cand).exists() else None


def check_source(rel: str, spec: dict) -> list[str]:
    """Static checks that need no toolchain."""
    text = (ROOT / rel).read_text(encoding="utf-8")
    code = re.sub(r"/-.*?-/", "", text, flags=re.S)
    code = re.sub(r"--.*", "", code)
    problems = []
    if re.search(r"\bsorry\b", code):
        problems.append("contains `sorry`")
    if re.search(r"^\s*(axiom|unsafe|opaque)\b", code, flags=re.M):
        problems.append("declares an axiom / unsafe / opaque")
    if "native_decide" in code:
        problems.append("uses native_decide")
    for stmt in spec["statements"]:
        if stmt not in text:
            problems.append(f"statement changed or missing: {stmt[:60]!r}")
    for thm in spec["theorems"]:
        if f"#print axioms {thm}" not in text:
            problems.append(f"no `#print axioms {thm}`")
    return problems


def check_output(spec: dict, returncode: int, output: str) -> list[str]:
    problems = []
    if returncode != 0:
        problems.append(f"lean exited {returncode}")
    if re.search(r"\b(error|warning: declaration uses 'sorry')", output):
        problems.append("lean reported an error or sorry")
    for thm in spec["theorems"]:
        m = re.search(
            rf"'{re.escape(thm)}' (does not depend on any axioms|"
            rf"depends on axioms: \[([^\]]*)\])", output)
        if not m:
            problems.append(f"{thm}: no axiom report")
            continue
        axioms = {a.strip() for a in (m.group(2) or "").split(",") if a.strip()}
        extra = axioms - ALLOWED_AXIOMS
        if extra:
            problems.append(f"{thm}: non-standard axioms {sorted(extra)}")
    return problems


def main() -> int:
    failed = False
    for rel, spec in FILES.items():
        for p in check_source(rel, spec):
            print(f"FAIL  {rel}: {p}")
            failed = True
    if failed:
        return 1
    lean = find_lean()
    if lean is None:
        print("SKIP  lean not found (set LEAN=... or put lean on PATH); "
              f"need Lean {LEAN_VERSION}")
        return 2
    version = subprocess.run([lean, "--version"], capture_output=True,
                             text=True).stdout.strip()
    if f"version {LEAN_VERSION}" not in version:
        print(f"FAIL  wrong toolchain: {version!r}, need {LEAN_VERSION}")
        return 1
    for rel, spec in FILES.items():
        proc = subprocess.run([lean, rel], cwd=ROOT, capture_output=True,
                              text=True)
        problems = check_output(spec, proc.returncode,
                                proc.stdout + proc.stderr)
        for p in problems:
            print(f"FAIL  {rel}: {p}")
        failed |= bool(problems)
        if not problems:
            print(f"PASS  {rel}: {len(spec['theorems'])} theorems, "
                  f"no sorry, standard axioms only ({version})")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
