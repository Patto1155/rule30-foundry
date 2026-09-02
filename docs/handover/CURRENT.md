# Handover — current

**Overwrite this file in place at the end of a session.** Do not add a new
dated handover file. Git history keeps every superseded version, and
`docs/handover/archive/` holds the five that predate this convention.

Handover was append-only until 2026-09-02: five files, 1,353 lines, no index,
with `AGENTS.md` hardcoding three of them by name. A fresh agent could not tell
which was current without opening several, and the pointers went stale every
time a sixth was written.

Keep this file **under ~120 lines**. It is narrative context for the next
session — the traps, the reasoning, the things that are true but not obvious.
It is not the claim record (`docs/CLAIM_LEDGER.md`) and not the work queue
(`docs/STATUS.md`).

---

Last session: 2026-09-02 · Branch: `claude/agent-context-bootstrap`

## What this session did

Restructured how a fresh agent builds context. No research content changed; no
ledger row moved.

- Added `CLAUDE.md` — auto-loaded by Claude Code, which does **not** read
  `AGENTS.md`. Every guardrail in `AGENTS.md` was previously opt-in: it only
  applied if the agent happened to go read it. Now the three that have each
  cost real months (counting bound, bit order, single seed) are injected at
  session start.
- Added `docs/STATUS.md` as the single home for current state, and **deleted
  the competing snapshots**: `AGENTS.md` § Current Frontier was frozen at
  `2026-04-01` and still listed `O` and `P` as the open frontier; `README.md`
  still reported Problem 1 as "no period found up to 1,000,000 steps" when the
  exact scan had decided 9,999,936 candidates.
- Collapsed handover to this file plus `archive/`.
- Extended `tools/lint_ledger.py` so the above cannot silently recur.
- Added `docs/BRANCHING.md` after finding 7 of 12 remote branches were fully
  merged into `main` and never deleted.

## Traps worth knowing

- **`AGENTS.md` is not auto-loaded by Claude Code.** `CLAUDE.md` is. If you add
  a rule that must always apply, it goes in `CLAUDE.md` or it is advisory only.
- **The status lint is filename-driven.** It requires `docs/STATUS.md` to cite
  the newest dated file in `docs/experiment-logs/`. Add a log dated later than
  the one STATUS cites and the build fails until you update STATUS. That is the
  intended behaviour, not a bug — it is the mechanism that keeps STATUS honest.
- **The PR stack is 3 deep** (#18 → #19 → #20) and this branch is a *sibling*
  of #20, not a fourth layer. Land #18 first; see `docs/BRANCHING.md`.

## Next

See `docs/STATUS.md` § Open work, ranked. The cheapest open item remains the
item-14 pattern-map walk — roughly 26 minutes of CPU from a ledger row.
