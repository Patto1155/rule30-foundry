#!/usr/bin/env bash
# Setup script for the `council` Claude Code cloud environment.
#
# Paste this into the environment's "Setup script" box (claude.ai -> Code ->
# Environments), or run it in a session to repair a half-configured container.
# It runs BEFORE Claude starts, so `codex` is already authenticated by the time
# any Bash or MCP call reaches it - tools/council.py never has to seed anything.
#
# Idempotent on purpose: a rerun must not clobber a token Codex has since
# refreshed in place. See docs/COUNCIL.md for what has to be set where.
set -euo pipefail

echo "[council] installing Codex CLI"
npm install -g @openai/codex >/dev/null 2>&1 || {
  echo "[council] npm install failed - is npm on PATH and is the registry allowlisted?" >&2
  exit 1
}

# Optional: only needed if you run `python tools/council.py serve` as an MCP
# server. The CLI path is stdlib-only and needs none of this.
pip install --quiet mcp 2>/dev/null || echo "[council] mcp not installed (CLI path unaffected)"

if [ -n "${CODEX_AUTH_B64:-}" ]; then
  mkdir -p "$HOME/.codex"
  chmod 700 "$HOME/.codex"
  if [ -f "$HOME/.codex/auth.json" ]; then
    echo "[council] ~/.codex/auth.json already present, leaving it alone"
  else
    printf '%s' "$CODEX_AUTH_B64" | base64 -d > "$HOME/.codex/auth.json"
    chmod 600 "$HOME/.codex/auth.json"
    mode=$(python3 -c 'import json,pathlib,os;print(json.loads(pathlib.Path(os.environ["HOME"]+"/.codex/auth.json").read_text()).get("auth_mode","?"))' 2>/dev/null || echo '?')
    echo "[council] seeded ~/.codex/auth.json (auth_mode=$mode; expected: chatgpt)"
  fi
else
  echo "[council] CODEX_AUTH_B64 unset - codex-backed roles will fail. Set it in the environment."
fi

# The repo's own dependencies. Keep this last so a council failure above is
# still visible rather than buried under pip output.
if [ -f requirements-ci.txt ]; then
  pip install --quiet -r requirements-ci.txt
fi

echo "[council] setup complete"
