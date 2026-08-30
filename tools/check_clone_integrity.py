#!/usr/bin/env python
"""Detect the fresh-Windows-clone corruption before it looks like bad science.

`.gitattributes` carries `data/** -text` precisely so git never rewrites line
endings in the manifest-anchored artifacts. It does not win during `git clone`'s
*initial* checkout: with `core.autocrlf=true`, 162 tracked `data/` files land
with CRLF and fail their SHA-256 anchors. Later checkouts honour the attribute
correctly, which is what makes this so easy to misdiagnose - deleting a file and
checking it out again produces a *correct* file, so the obvious experiment
exonerates the obvious suspect.

`git status` reports the tree clean, because the files round-trip through the
same filter. So the first symptom is `verify_data.py` printing a wall of hash
mismatches, which reads like corrupted data or a bad experiment rather than a
checkout defect. That is the failure this stage exists to short-circuit.

The signature is `i/lf w/crlf`: LF in the index, CRLF in the working tree.

`i/crlf w/crlf` is NOT this defect - that file is CRLF in the index itself, so
it is byte-identical on every platform and its anchor is correct as recorded.
`data/prize/2026-08-15-dfao-min-state-curve.json` is one such file. Do not
"fix" it.

Usage:
    python tools/check_clone_integrity.py
    python tools/check_clone_integrity.py --paths data docs
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

REPAIR = """  Repair (from the repo root):
    git config core.autocrlf false
    git rm --cached -r data
    git reset --hard HEAD

  The `git rm --cached` step is not optional. Without it git considers the
  files clean and will not rewrite them, so deleting and re-checking-out
  leaves most of them corrupted."""


def ls_files_eol(paths: list[str]) -> list[tuple[str, str, str]]:
    """Return (index_eol, worktree_eol, path) for each tracked file."""
    proc = subprocess.run(["git", "ls-files", "--eol", "--", *paths],
                          cwd=REPO_ROOT, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip() or "git ls-files failed")
    rows = []
    for line in proc.stdout.splitlines():
        if "\t" not in line:
            continue
        attrs, path = line.split("\t", 1)
        fields = attrs.split()
        index_eol = next((f[2:] for f in fields if f.startswith("i/")), "")
        work_eol = next((f[2:] for f in fields if f.startswith("w/")), "")
        rows.append((index_eol, work_eol, path.strip()))
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--paths", nargs="*", default=["data"],
                    help="pathspecs to check (default: data)")
    args = ap.parse_args()

    if not (REPO_ROOT / ".git").exists():
        print("clone-integrity: not a git checkout; nothing to check")
        return 0

    try:
        rows = ls_files_eol(args.paths)
    except (OSError, RuntimeError) as exc:
        print(f"clone-integrity: cannot run git ({exc}); skipping")
        return 0

    corrupted = [p for i_eol, w_eol, p in rows
                 if i_eol == "lf" and w_eol == "crlf"]

    if not corrupted:
        print(f"clone-integrity: OK ({len(rows)} tracked files, "
              "no LF->CRLF checkout corruption)")
        return 0

    autocrlf = subprocess.run(["git", "config", "--get", "core.autocrlf"],
                              cwd=REPO_ROOT, capture_output=True, text=True)
    setting = autocrlf.stdout.strip() or "(unset)"

    print(f"clone-integrity: FAIL - {len(corrupted)} tracked files are CRLF in "
          "the working tree but LF in the index.")
    print(f"  core.autocrlf = {setting}")
    print("  These files no longer match their SHA-256 anchors in "
          "data/MANIFEST.sha256.")
    print("  `git status` will report the tree clean: the files round-trip "
          "through the")
    print("  same filter, so nothing looks wrong until hashes are checked.")
    print()
    for path in corrupted[:10]:
        print(f"    {path}")
    if len(corrupted) > 10:
        print(f"    ... and {len(corrupted) - 10} more")
    print()
    print(REPAIR)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
