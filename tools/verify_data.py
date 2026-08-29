#!/usr/bin/env python
"""Verify repo data artifacts against recorded hashes and the golden reference.

Three checks, each independently runnable:

  --manifest   every file listed in data/MANIFEST.sha256 still hashes the same
  --golden     data/golden/center_col_golden_1M.bin matches its recorded hash
  --bitstream  a candidate center-column .bin agrees with the golden reference
               over its first 1,000,000 bits

Bit order matters and is not uniform in this repo: gpu/rule30_sim.py writes
center-column dumps LSB-first while the golden reference is MSB-first. Reading
one with the other's convention yields a stream ~50% different from bit 0,
which looks like total corruption but is only a packing mismatch. --bitstream
tries both orders by default; --bitorder pins one.

The third is the one that matters for the open integrity question: the
headline A-L results were generated before the packed-kernel fixes of
2026-06-15, and nothing has re-anchored them since. Running

    python tools/verify_data.py --bitstream data/center_col_10M.bin

tells you in seconds whether the bitstream those results were computed from
still agrees with an independent implementation.

Exit code is 0 only if every requested check passes.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
MANIFEST = REPO_ROOT / "data" / "MANIFEST.sha256"
GOLDEN = REPO_ROOT / "data" / "golden" / "center_col_golden_1M.bin"
GOLDEN_BITS = 1_000_000

# Bit-packing convention. gen_golden_reference.py writes the golden file
# MSB-first (numpy's default), but gpu/rule30_sim.py writes center-column
# dumps with bitorder="little" (LSB-first). Decoding one with the other's
# convention scrambles the stream into something ~50% different from bit 0,
# which reads as catastrophic corruption but is only a packing mismatch.
# --bitorder defaults to trying both.
GOLDEN_BITORDER = "big"
ORDERS = ("big", "little")
BITORDER_ALIAS = {"msb": "big", "lsb": "little"}
ORDER_LABEL = {"big": "MSB-first", "little": "LSB-first"}


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def check_manifest() -> bool:
    if not MANIFEST.exists():
        print(f"FAIL  manifest missing: {MANIFEST}")
        return False
    ok = True
    checked = missing = 0
    for line in MANIFEST.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        digest, _, rel = line.partition("  ")
        target = REPO_ROOT / rel
        if not target.exists():
            print(f"MISS  {rel}")
            missing += 1
            continue
        actual = sha256_file(target)
        checked += 1
        if actual != digest:
            print(f"FAIL  {rel}\n      expected {digest}\n      actual   {actual}")
            ok = False
    print(f"manifest: {checked} verified, {missing} absent, "
          f"{'OK' if ok else 'MISMATCH'}")
    return ok


def check_golden() -> bool:
    if not GOLDEN.exists():
        print(f"FAIL  golden reference missing: {GOLDEN}")
        return False
    if not MANIFEST.exists():
        print("WARN  no manifest; cannot verify golden hash")
        return False
    rel = GOLDEN.relative_to(REPO_ROOT).as_posix()
    expected = None
    for line in MANIFEST.read_text().splitlines():
        digest, _, name = line.strip().partition("  ")
        if name == rel:
            expected = digest
    if expected is None:
        print(f"FAIL  {rel} not listed in manifest")
        return False
    actual = sha256_file(GOLDEN)
    if actual != expected:
        print(f"FAIL  golden hash mismatch\n      expected {expected}\n"
              f"      actual   {actual}")
        return False
    print(f"golden: {rel} OK ({GOLDEN_BITS:,} bits)")
    return True


def _decode(raw: np.ndarray, order: str) -> np.ndarray:
    return np.unpackbits(raw, bitorder=order)[:GOLDEN_BITS]


def check_bitstream(candidate: Path, bitorder: str = "auto") -> bool:
    if not candidate.exists():
        print(f"FAIL  bitstream not found: {candidate}")
        print("      (raw .bin files are not tracked in git; regenerate with")
        print("       gpu/rule30_sim.py or point --bitstream at a local copy)")
        return False
    if not GOLDEN.exists():
        print(f"FAIL  golden reference missing: {GOLDEN}")
        return False

    gold_bits = np.unpackbits(np.fromfile(GOLDEN, dtype=np.uint8),
                              bitorder=GOLDEN_BITORDER)[:GOLDEN_BITS]

    need_bytes = GOLDEN_BITS // 8
    raw = np.fromfile(candidate, dtype=np.uint8, count=need_bytes)

    # A center-column dump may be one byte per bit or bit-packed. Detect it.
    if raw.size >= need_bytes and set(np.unique(raw[:4096]).tolist()) <= {0, 1}:
        cands = [("byte-per-bit", None,
                  np.fromfile(candidate, dtype=np.uint8,
                              count=GOLDEN_BITS)[:GOLDEN_BITS])]
    else:
        orders = ORDERS if bitorder == "auto" else [BITORDER_ALIAS[bitorder]]
        cands = [("bit-packed", o, _decode(raw, o)) for o in orders]

    best = None
    for layout, order, bits in cands:
        if bits.size < GOLDEN_BITS:
            print(f"FAIL  {candidate.name} has only {bits.size:,} bits, "
                  f"need {GOLDEN_BITS:,}")
            return False
        ndiff = int(np.count_nonzero(bits != gold_bits))
        if ndiff == 0:
            how = f"{layout}, {ORDER_LABEL[order]}" if order else layout
            print(f"bitstream: {candidate.name} ({how}) agrees with golden "
                  f"over {GOLDEN_BITS:,} bits  OK")
            return True
        if best is None or ndiff < best[2]:
            best = (layout, order, ndiff, bits)

    layout, order, ndiff, bits = best
    how = f"{layout}, {ORDER_LABEL[order]}" if order else layout
    diff = np.flatnonzero(bits != gold_bits)
    print(f"FAIL  {candidate.name} ({how}) diverges from golden reference")
    print(f"      first divergence at bit {diff[0]:,}")
    print(f"      {ndiff:,} of {GOLDEN_BITS:,} bits differ "
          f"({ndiff / GOLDEN_BITS:.4%})")

    if order is not None and bitorder == "auto":
        print("      both bit orders were tried; this is the closer of the two")
    elif order is not None:
        other = "lsb" if order == "big" else "msb"
        print(f"      only {ORDER_LABEL[order]} was tried "
              f"(--bitorder {bitorder}); retry with --bitorder {other}")

    if ndiff / GOLDEN_BITS > 0.4:
        print("      ~50% differing with divergence at bit 0 means the two")
        print("      streams are uncorrelated -- almost always a packing or")
        print("      seed mismatch, not a kernel bug")
    elif diff[0] > 0:
        print("      a late first divergence usually means a word-boundary or")
        print("      padding bug, not a seed or rule-table bug")
    return False


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", action="store_true")
    ap.add_argument("--golden", action="store_true")
    ap.add_argument("--bitstream", type=Path, metavar="PATH")
    ap.add_argument("--bitorder", choices=("auto", "msb", "lsb"),
                    default="auto",
                    help="bit-packing order of the candidate file "
                         "(default: try both)")
    ap.add_argument("--all", action="store_true",
                    help="run manifest and golden checks")
    args = ap.parse_args()

    if not any([args.manifest, args.golden, args.bitstream, args.all]):
        args.all = True

    results = []
    if args.manifest or args.all:
        results.append(check_manifest())
    if args.golden or args.all:
        results.append(check_golden())
    if args.bitstream:
        results.append(check_bitstream(args.bitstream, args.bitorder))

    return 0 if all(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
