# Brief template — copy, fill, dispatch

    python tools/council.py review --file briefs/<slug>.md
    python tools/council.py math --file briefs/<slug>.md      # or redteam

A brief is the *entire* context the reviewer gets. It cannot ask a follow-up
question, it cannot see this conversation, and it cannot read the repo —
`tools/council.py` is read-only and has no checkout, by design. A brief that
says "the grid described above" buys you a confident answer to a question
nobody asked.

The role (`math`, `redteam`, `review`) is a prompt framing, not a different
model. It changes how the reply is argued, not what the reviewer can see, so
it is never a substitute for putting the material in the brief.

Two rules that are the whole reason this template exists:

1. **State the claim, not your confidence in it.** If you write "we correctly
   applied the Reed–Muller bound", you have bought agreement, not review.
   Write what was computed and ask whether it supports the conclusion.
2. **Say what "done" looks like.** Ranked actions with costs, or a specific
   yes/no with a reason. Otherwise you get an essay.

Delete this header and everything below the line that does not apply.

---

## Context

What the repo is doing, in five sentences a stranger can act on. Name the
object precisely — for this repo that is almost always *the center column of
Rule 30 from the single-black-cell initial condition* (see
`docs/theory/README.md` §0; ensemble and random-IC quantities are not prize
progress and a reviewer who does not know that will waste your budget).

## The artifact under review

The claim, number, diff, or argument. Inline it — a path is not context the
reviewer can follow. For numeric results give the parameters that produced
them (`d`, `w`, `D`, sample counts), not just the verdict.

## What I want back

Priority-ordered questions. Be concrete and adversarial; forbid summarising the
brief back. State the output shape you want (e.g. "3–5 ranked actions with
expected cost, what a positive and a negative outcome each mean").

## Known-closed routes

What is already proved or already ruled out, so the reviewer does not spend its
budget re-deriving it. `docs/theory/README.md` is the gate.

## Ground rules

- Cite sources for literature claims; say "I could not verify this" rather than
  guessing. An invented citation costs more than a missing one.
- Flag anything in the above you believe is wrong, including the framing.
