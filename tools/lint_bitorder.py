#!/usr/bin/env python
"""Static check: no bare np.packbits / np.unpackbits in this repo.

Why this exists
---------------
`gpu/rule30_sim.py` writes center-column dumps with `bitorder='little'`
(LSB-first). NumPy's default is MSB-first. A bare `np.unpackbits(data)` on
such a file therefore returns the true stream *with every consecutive 8-bit
block reversed*: 49.95% of bit positions differ, while the bit mean is
identical. Every aggregate statistic this repo watches - bias, monobit, block
frequency - is invariant under that permutation, so the defect survived five
months and invalidated README experiments I, J, K and L.

Fixing four call sites does not stop a fifth. This lint makes the bug
unwritable: every `packbits`/`unpackbits` call under the scanned roots must
either pass `bitorder=` explicitly or carry an exemption comment saying why
the byte-level convention does not matter there.

Exemption
---------
Put `# bitorder-exempt: <reason>` on the call's own line, or on the line
immediately above it. The reason is mandatory; a bare marker is rejected.
Legitimate reasons look like:

  - a deliberately independent reference implementation whose convention must
    NOT track the kernel it checks (`tools/gen_golden_reference.py`)
  - re-packing an already-correctly-decoded bit array into bytes to feed a
    byte-oriented consumer such as zlib, where any consistent convention gives
    the same answer

"I could not be bothered" is not one of them.

Usage:
    python tools/lint_bitorder.py            # exit 1 on any hit
    python tools/lint_bitorder.py --quiet    # exit code only
"""

from __future__ import annotations

import argparse
import ast
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SCAN_ROOTS = ("experiments", "gpu", "tools")
WATCHED = frozenset({"packbits", "unpackbits"})
EXEMPT_MARKER = "# bitorder-exempt:"


class Finding:
    __slots__ = ("relpath", "lineno", "func", "source")

    def __init__(self, relpath: str, lineno: int, func: str, source: str):
        self.relpath = relpath
        self.lineno = lineno
        self.func = func
        self.source = source

    def __str__(self) -> str:
        return (f"{self.relpath}:{self.lineno}: {self.func}() has no "
                f"bitorder= and no exemption\n      {self.source.strip()}")


def _called_name(node: ast.Call) -> str | None:
    """Return 'packbits'/'unpackbits' for np.packbits(...) or packbits(...)."""
    func = node.func
    if isinstance(func, ast.Attribute) and func.attr in WATCHED:
        return func.attr
    if isinstance(func, ast.Name) and func.id in WATCHED:
        return func.id
    return None


def _has_reason(line: str) -> bool:
    _, sep, reason = line.partition(EXEMPT_MARKER)
    return bool(sep and reason.strip())


def _is_exempt(lines: list[str], lineno: int) -> bool:
    """True if an exemption comment with a reason covers this line.

    The marker may sit on the call's own line, or anywhere in the contiguous
    block of comment-only lines immediately above it - so a multi-line
    justification works and does not have to be crammed onto one line. A blank
    line or any code between the comment and the call ends the block, which
    stops one statement's exemption from covering the next.

    `lineno` is 1-based, as ast reports it.
    """
    idx = lineno - 1
    if not 0 <= idx < len(lines):
        return False
    if _has_reason(lines[idx]):
        return True
    idx -= 1
    while idx >= 0 and lines[idx].lstrip().startswith("#"):
        if _has_reason(lines[idx]):
            return True
        idx -= 1
    return False


def scan_source(source: str, relpath: str) -> list[Finding]:
    tree = ast.parse(source, filename=relpath)
    lines = source.splitlines()
    findings: list[Finding] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        name = _called_name(node)
        if name is None:
            continue
        if any(kw.arg == "bitorder" for kw in node.keywords):
            continue
        # **kwargs could carry it; assume the author knows and let it pass.
        if any(kw.arg is None for kw in node.keywords):
            continue
        if _is_exempt(lines, node.lineno):
            continue
        src = lines[node.lineno - 1] if node.lineno <= len(lines) else ""
        findings.append(Finding(relpath, node.lineno, name, src))
    return findings


def python_files(root: Path = REPO_ROOT) -> list[Path]:
    out: list[Path] = []
    for name in SCAN_ROOTS:
        out.extend(sorted((root / name).rglob("*.py")))
    return out


def scan_repo(root: Path = REPO_ROOT) -> list[Finding]:
    findings: list[Finding] = []
    for path in python_files(root):
        rel = path.relative_to(root).as_posix()
        findings.extend(scan_source(path.read_text(encoding="utf-8"), rel))
    return findings


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--quiet", action="store_true",
                    help="print nothing; signal via exit code")
    args = ap.parse_args()

    findings = scan_repo()
    if findings:
        if not args.quiet:
            print(f"lint_bitorder: {len(findings)} bare call(s)")
            for f in findings:
                print(f"FAIL  {f}")
            print("\n  gpu/rule30_sim.py writes center-column dumps LSB-first;")
            print("  numpy defaults to MSB-first. Pass bitorder= explicitly, or")
            print(f"  annotate with '{EXEMPT_MARKER} <reason>' if the byte-level")
            print("  convention genuinely does not matter at that call site.")
        return 1
    if not args.quiet:
        n = len(python_files())
        print(f"lint_bitorder: {n} files scanned, no bare packbits/unpackbits")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
