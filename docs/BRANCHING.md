# Branching and merge policy

## Why this file exists

An audit on 2026-09-02 found 12 remote branches, of which **7 were fully merged
into `main` and had simply never been deleted**:

```
agent/rule30-20260620T230002Z      merged
agent/rule30-20260622T230003Z      merged
agent/rule30-20260624T230002Z      merged
agent/rule30-20260626T230002Z      merged
agent/rule30-20260629T230002Z      merged
feat/dfao-min-state-curve          merged
claude/branch-merge-strategy-smfppf  merged (and never had a PR)
```

At the same time the open PRs formed a **3-deep stack** — `main` ← #18 ← #19 ←
#20 — so no work could land without three sequential merges, and each agent
had to reason about which layer its files came from.

Neither is a research problem. Both are pure friction, and both are cheap to
prevent.

## Rules

### 1. Branch from `main`

Default, no exceptions worth memorising. `git fetch origin main && git checkout
-b <name> origin/main`.

### 2. Stack at most one level, and only under duress

A stacked branch (based on an open PR rather than `main`) is allowed **only**
when it genuinely cannot compile or run without files that exist solely on the
parent branch. When you stack:

- State the dependency in the first line of the PR body: *"Stacked on #NN, not
  `main`."*
- Never stack on a branch that is itself stacked. **Maximum depth from `main`
  is 1.** If you need a third layer, the second layer should have landed first.
- Prefer being a *sibling* of an existing stacked PR over being a fourth layer.

If the stack is already deeper than this — as it was when this file was written
— land the bottom of it before opening anything new.

### 3. Delete the branch when the PR merges

Enable **Settings → General → Automatically delete head branches** on the
repository. A merged branch has no readers: its commits are in `main` and its
PR page is permanent. Leaving it costs every future agent one more line to read
in `git branch -a` and one more "is this live?" question.

For the backlog, deleting a merged branch is safe and reversible — the commits
are reachable from `main`:

```bash
git push origin --delete <branch>
```

### 4. Naming

`<kind>/<slug>` where kind is one of:

| Kind | For |
|---|---|
| `feat/` | New experiment, tool, or capability |
| `fix/` | Correcting something wrong in `main` |
| `research/` | Exploratory work that may not land |
| `claude/` | Agent-authored session branches |

Do not use timestamps as slugs (`agent/rule30-20260620T230002Z` tells a reader
nothing). Say what the branch does.

### 5. One concern per PR

PR #19 carried Tier 0 tooling *and* five research items *and* three merged
laptop commits — 12,805 additions across 54 files. It is a good PR, but it
could not be reviewed incrementally and it blocked #20 behind it for two days.
Split when the parts are independently landable.

### 6. Stale branches

A branch with no commits for 14 days is either landed or abandoned. Open a PR
or delete it. `research/` branches are exempt while their PR is open and has a
stated next step.

## Checking before you branch

```bash
git fetch --prune origin
git branch -r --merged origin/main | grep -v 'origin/main$'   # safe to delete
git branch -r --no-merged origin/main                          # actually live
```
