#!/usr/bin/env bash
# Build the SAT toolchain that experiments/dfao_drat_proofs.py needs:
#
#   cadical    a DRAT-emitting SAT solver
#   drat-trim  an independent DRAT proof checker
#
# Both land in third_party/ (gitignored). Neither is vendored: the point of a
# DRAT certificate is that the checker shares no code with the solver, and
# that a reader can rebuild both from upstream rather than trusting a binary
# committed by whoever wrote the claim.
#
#   bash tools/build_sat_toolchain.sh
#   python experiments/dfao_drat_proofs.py --self-test
#
# Note: pysat's Cadical153(with_proof=True) is NOT a substitute. Its
# get_proof() output has no terminating empty clause and drat-trim rejects it.
# See the module docstring of experiments/dfao_drat_proofs.py.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST="$ROOT/third_party"
BUILD="$DEST/.build"
mkdir -p "$DEST" "$BUILD"

CADICAL_REPO="${CADICAL_REPO:-https://github.com/arminbiere/cadical.git}"
DRAT_TRIM_SRC="${DRAT_TRIM_SRC:-https://raw.githubusercontent.com/marijnheule/drat-trim/master/drat-trim.c}"

if [ ! -x "$DEST/cadical" ]; then
  echo "==> building cadical"
  rm -rf "$BUILD/cadical"
  git clone --depth 1 "$CADICAL_REPO" "$BUILD/cadical"
  ( cd "$BUILD/cadical" && ./configure && make -j"$(nproc 2>/dev/null || echo 4)" )
  cp "$BUILD/cadical/build/cadical" "$DEST/cadical"
else
  echo "==> cadical already built"
fi

if [ ! -x "$DEST/drat-trim" ]; then
  echo "==> building drat-trim"
  curl -sSL -o "$BUILD/drat-trim.c" "$DRAT_TRIM_SRC"
  cc -O2 -o "$DEST/drat-trim" "$BUILD/drat-trim.c" -lm
else
  echo "==> drat-trim already built"
fi

echo
echo "solver : $DEST/cadical  ($("$DEST/cadical" --version))"
echo "checker: $DEST/drat-trim"
echo
echo "next: python experiments/dfao_drat_proofs.py --self-test"
