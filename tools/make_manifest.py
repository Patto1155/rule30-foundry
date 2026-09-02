#!/usr/bin/env python
"""Regenerate data/MANIFEST.sha256.

Covers every tracked artifact under data/ plus, if present locally, the
canonical bitstreams. Bitstreams are recorded under an "untracked" section so
their hashes are anchored in git even though the bytes are not.

    python tools/make_manifest.py          # rewrite the manifest
    python tools/make_manifest.py --check  # exit 1 if it is out of date

Two properties this file is careful about, both learned the hard way:

  1. **Anchors are never dropped.** An earlier version wrote `# UNANCHORED`
     for any bitstream missing from the local disk. Since the bitstreams are
     gitignored, running it on a fresh clone - or any machine that had not
     regenerated them - would silently *delete* the recorded hashes of the two
     canonical artifacts, destroying the only anchor tying the A-H results to
     specific bytes. Hashes for absent bitstreams are now carried forward from
     the existing manifest.

  2. **LF, always.** `Path.write_text` translates "\\n" to os.linesep, so the
     manifest came out CRLF on Windows and LF elsewhere: byte-different files
     with identical content. `data/** -text` in .gitattributes (correctly)
     stops git from papering over that. Same defect as `csv.writer`'s CRLF
     default, in the integrity file itself.
"""

from __future__ import annotations

import argparse
import hashlib
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
MANIFEST = REPO_ROOT / "data" / "MANIFEST.sha256"

# Canonical bitstreams: not in git, but their hashes belong in the manifest.
BITSTREAMS = ["data/center_col_10M.bin", "data/center_col_46M.bin"]

HEADER = [
    "# SHA-256 manifest for rule30-foundry data artifacts.",
    "# Regenerate: python tools/make_manifest.py",
    "# Verify:     python tools/verify_data.py --manifest",
    "#",
    "# Tracked artifacts",
]


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def tracked_data_files(root: Path = REPO_ROOT) -> list[str]:
    out = subprocess.run(
        ["git", "ls-files", "data/"], cwd=root,
        capture_output=True, text=True, check=True).stdout.split()
    return sorted(p for p in out if not p.endswith("MANIFEST.sha256"))


def recorded_hashes(manifest: Path) -> dict[str, str]:
    """Path -> hash for every real (non-comment) entry in an existing file."""
    if not manifest.exists():
        return {}
    found: dict[str, str] = {}
    for line in manifest.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        digest, _, rel = line.partition("  ")
        if digest and rel:
            found[rel] = digest
    return found


def build_manifest_text(root: Path = REPO_ROOT) -> tuple[str, list[str]]:
    """Return (manifest text, warnings). Pure: reads disk, writes nothing."""
    previous = recorded_hashes(root / "data" / "MANIFEST.sha256")
    warnings: list[str] = []
    lines = list(HEADER)

    for rel in tracked_data_files(root):
        path = root / rel
        if path.exists():
            lines.append(f"{sha256_file(path)}  {rel}")

    lines += [
        "#",
        "# Canonical bitstreams (not tracked in git; hashes anchored here).",
        "# A bitstream absent from this machine keeps its previously anchored",
        "# hash -- regenerating the manifest must never drop an anchor.",
    ]
    for rel in BITSTREAMS:
        path = root / rel
        anchored = previous.get(rel)
        if path.exists():
            digest = sha256_file(path)
            if anchored and anchored != digest:
                warnings.append(
                    f"{rel} hash CHANGED: anchored {anchored} -> {digest}. "
                    "Either the kernel output changed or the file is corrupt; "
                    "do not accept this without an explanation.")
            lines.append(f"{digest}  {rel}")
        elif anchored:
            # Identical to the line a machine that *has* the file would write,
            # so the manifest is byte-identical everywhere. Marking it absent
            # here would make the file machine-dependent, which is exactly the
            # property tests/test_manifest_determinism.py pins down.
            lines.append(f"{anchored}  {rel}")
        else:
            lines.append(f"# UNANCHORED  {rel}  (never hashed on any machine)")

    return "\n".join(lines) + "\n", warnings


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true",
                    help="do not write; exit 1 if the manifest is out of date")
    args = ap.parse_args()

    text, warnings = build_manifest_text()
    for w in warnings:
        print(f"WARN  {w}", file=sys.stderr)

    if args.check:
        current = (MANIFEST.read_bytes() if MANIFEST.exists() else b"")
        if current != text.encode("utf-8"):
            print("FAIL  data/MANIFEST.sha256 is out of date "
                  "(run python tools/make_manifest.py)")
            return 1
        print("manifest: up to date")
        return 0

    # newline="" so "\n" reaches the file verbatim on every platform.
    with MANIFEST.open("w", encoding="utf-8", newline="") as fh:
        fh.write(text)

    n_tracked = sum(1 for rel in tracked_data_files()
                    if (REPO_ROOT / rel).exists())
    present = [b for b in BITSTREAMS if (REPO_ROOT / b).exists()]
    print(f"wrote {MANIFEST} ({n_tracked} tracked, {len(present)} bitstream "
          f"present, {len(BITSTREAMS) - len(present)} carried forward)")
    return 1 if warnings else 0


if __name__ == "__main__":
    raise SystemExit(main())
