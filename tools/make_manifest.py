#!/usr/bin/env python
"""Regenerate data/MANIFEST.sha256.

Covers every tracked artifact under data/ plus, if present locally, the
untracked canonical bitstreams. Bitstreams are recorded under an "untracked"
section so their hashes are anchored in git even though the bytes are not.

    python tools/make_manifest.py
"""

from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
MANIFEST = REPO_ROOT / "data" / "MANIFEST.sha256"

# Canonical bitstreams: not in git, but their hashes belong in the manifest.
BITSTREAMS = ["data/center_col_10M.bin", "data/center_col_46M.bin"]


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def tracked_data_files() -> list[str]:
    out = subprocess.run(
        ["git", "ls-files", "data/"], cwd=REPO_ROOT,
        capture_output=True, text=True, check=True).stdout.split()
    return sorted(p for p in out if not p.endswith("MANIFEST.sha256"))


def main() -> int:
    lines = [
        "# SHA-256 manifest for rule30-foundry data artifacts.",
        "# Regenerate: python tools/make_manifest.py",
        "# Verify:     python tools/verify_data.py --manifest",
        "#",
        "# Tracked artifacts",
    ]
    for rel in tracked_data_files():
        path = REPO_ROOT / rel
        if path.exists():
            lines.append(f"{sha256_file(path)}  {rel}")

    present = [b for b in BITSTREAMS if (REPO_ROOT / b).exists()]
    lines += [
        "#",
        "# Canonical bitstreams (not tracked in git; hashes anchored here).",
        "# Absent entries are expected on a fresh clone.",
    ]
    for rel in present:
        lines.append(f"{sha256_file(REPO_ROOT / rel)}  {rel}")
    for rel in BITSTREAMS:
        if rel not in present:
            lines.append(f"# UNANCHORED  {rel}  (not present on this machine)")

    MANIFEST.write_text("\n".join(lines) + "\n")
    print(f"wrote {MANIFEST} "
          f"({len(tracked_data_files())} tracked, {len(present)} bitstream)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
