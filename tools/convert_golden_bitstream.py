#!/usr/bin/env python
"""Restore canonical LSB-first bytes from the documented MSB-first golden file.

The golden reference remains unchanged. Check both manifest hashes before
writing, and check a freshly simulated CPU prefix before trusting the source.
Run from any directory; default paths are relative to this repository.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from gen_golden_reference import center_naive, center_packed

ROOT = Path(__file__).resolve().parents[1]


def manifest_hash(path: Path, manifest: Path) -> str:
    relative = path.resolve().relative_to(ROOT).as_posix()
    for line in manifest.read_text(encoding="utf-8").splitlines():
        digest, _, name = line.partition("  ")
        if name == relative:
            return digest
    raise ValueError(f"no manifest anchor for {relative}")


def sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def convert(source: Path, destination: Path, manifest: Path,
            prefix_steps: int = 100_000) -> dict:
    raw = source.read_bytes()
    if sha256(raw) != manifest_hash(source, manifest):
        raise ValueError("source does not match its manifest anchor")
    if prefix_steps < 4096 or prefix_steps > len(raw) * 8:
        raise ValueError("prefix_steps must be between 4096 and source bit length")

    golden = np.unpackbits(np.frombuffer(raw, dtype=np.uint8), bitorder="big")
    fresh = np.unpackbits(center_packed(prefix_steps), bitorder="big")[:prefix_steps]
    if not np.array_equal(center_naive(4096), fresh[:4096]):
        raise ValueError("fresh packed simulation failed the naive 4096-bit check")
    differences = np.flatnonzero(golden[:prefix_steps] != fresh)
    if differences.size:
        raise ValueError(f"source MSB prefix first diverges at {int(differences[0])}")

    converted = np.packbits(golden, bitorder="little").tobytes()
    if sha256(converted) != manifest_hash(destination, manifest):
        raise ValueError("converted bytes do not match the canonical manifest anchor")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(converted)
    return {"source": source.relative_to(ROOT).as_posix(),
            "destination": destination.relative_to(ROOT).as_posix(),
            "bytes": len(converted), "sha256": sha256(converted),
            "fresh_cpu_prefix_bits": prefix_steps,
            "naive_prefix_bits": 4096, "prefix_first_divergence": None}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path,
                        default=ROOT / "data/golden/center_col_golden_10M.bin")
    parser.add_argument("--out", type=Path,
                        default=ROOT / "data/center_col_10M.bin")
    parser.add_argument("--manifest", type=Path, default=ROOT / "data/MANIFEST.sha256")
    parser.add_argument("--prefix-steps", type=int, default=100_000)
    args = parser.parse_args()
    print(json.dumps(convert(args.source.resolve(), args.out.resolve(),
                             args.manifest.resolve(), args.prefix_steps), indent=2))


if __name__ == "__main__":
    main()
