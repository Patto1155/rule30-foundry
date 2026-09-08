#!/bin/bash
# Make a web session's checkout able to run tools/verify_all.py.
#
# CI installs requirements-ci.txt before it runs verify_all; a Claude Code on
# the web container did not, so a fresh session started with numpy absent.
# That is not a cosmetic difference. Four test modules fail to IMPORT without
# numpy, and unittest reports an import failure as an error rather than as a
# skip -- so verify_all came up FAIL on a clean checkout of main, and the
# obvious reading ("the branch broke something") is wrong. Worse for the
# worker pool: every delegated worker runs verify_all inside its own worktree,
# so one missing package turns every worker's verification red at once and
# tells the lead nothing about the work.
#
# Deliberately not installed here: the SAT toolchain
# (tools/build_sat_toolchain.sh). It is a long C build, and its absence is an
# honest, allow-listed SKIP rather than a failure -- see the drat-toolchain
# stage in tools/verify_all.py.
set -euo pipefail

if [ "${CLAUDE_CODE_REMOTE:-}" != "true" ]; then
  exit 0
fi

cd "${CLAUDE_PROJECT_DIR:-$(dirname "$0")/../..}"

# --upgrade is omitted on purpose: the pin is a floor, and a container whose
# cache already satisfies it should be a no-op rather than a re-resolve.
python -m pip install --quiet --disable-pip-version-check -r requirements-ci.txt

python - <<'PY'
import numpy
print(f"session-start: numpy {numpy.__version__} ready")
PY
