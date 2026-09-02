#!/usr/bin/env python
"""Static check: docs/CLAIM_LEDGER.md must not cite evidence that isn't there.

Why this exists
---------------
The ledger is this repo's honesty mechanism, and it had been quietly wrong for
two weeks. The retracted DFAO row pointed at a replacement "`s*(n)` curve row
below" that did not exist, and the experiment log backing it was still a
template full of `RESULT_SECTION` placeholders sitting under a
"Claim Level: Certificate" header. Nothing catches that except reading the
whole file carefully, which is exactly the thing nobody does twice.

Checks
------
1. Every path cited in a backticked span in a ledger row exists on disk.
   Exception: the canonical bitstreams are gitignored, so they cannot be
   required to exist. They are instead required to be *hash-anchored* in
   data/MANIFEST.sha256 - the honest form of the same question.
2. No ledger row contains an unfilled `*_SECTION` placeholder.
3. Every row graded Certificate names a verifier command, and that command's
   script exists.
4. No file under docs/experiment-logs/ contains a placeholder token, and
   every one of them is valid UTF-8.
5. docs/STATUS.md exists, carries no placeholder, and cites the newest dated
   experiment log. Adding a log dated later than the one STATUS cites fails
   the build until STATUS is updated.
6. No file outside the allowlist declares its own current state. AGENTS.md
   carried a "Current state as of `2026-04-01`" section that went five months
   stale while remaining the second file every new agent was told to read.

Usage:
    python tools/lint_ledger.py            # exit 1 on any finding
    python tools/lint_ledger.py --quiet
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
LEDGER = REPO_ROOT / "docs" / "CLAIM_LEDGER.md"
LOG_DIR = REPO_ROOT / "docs" / "experiment-logs"
MANIFEST = REPO_ROOT / "data" / "MANIFEST.sha256"

# Top-level directories a citation can point into. A backticked span that does
# not start with one of these is prose or maths, not a path.
PATH_ROOTS = ("docs/", "data/", "experiments/", "tools/", "gpu/", "tests/")

# Gitignored artifacts: anchored by hash, not by presence.
UNTRACKED_OK = ("data/center_col_10M.bin", "data/center_col_46M.bin")

BACKTICKED = re.compile(r"`([^`]+)`")
PLACEHOLDER = re.compile(r"\b[A-Z][A-Z0-9]*(?:_[A-Z0-9]+)*_SECTION\b|"
                         r"\bTODO_[A-Z_]+\b|<FILL[^>]*>")
# A path token: no spaces, and the characters a repo path is made of.
PATH_TOKEN = re.compile(r"[A-Za-z0-9_./+-]+")
CERT_LEVEL = re.compile(r"\*{0,2}Certificate\*{0,2}")

# Files permitted to describe current state. docs/STATUS.md is the single
# writable home for it; archived handovers and experiment logs are historical
# records, so a frozen snapshot is correct there by definition.
STATUS_ALLOWED = (
    "docs/STATUS.md",
    "docs/handover/archive/",
    "docs/experiment-logs/",
)

# Narrow on purpose: this matches a section that *declares* frozen state, not
# a row that records when something happened. "Refuted 2026-08-19" in the
# ledger is a fact with a date and must not trip this.
FROZEN_STATE = re.compile(
    r"(?i)^(?:.*?\bcurrent\s+(?:state|status)\s+as\s+of\b.*"
    r"|\s*#{1,6}\s*current\s+frontier\s*)$")

DATED_LOG = re.compile(r"^(\d{4}-\d{2}-\d{2})-.+\.md$")

# Prose that *documents* the rule has to be able to quote the banned phrase.
# Same shape as lint_bitorder's `# bitorder-exempt: <reason>`: the marker is
# cheap, and the mandatory reason is what stops it becoming a silencer.
# The reason must start with an actual word: `<!-- status-exempt: -->`
# has a non-space run after the colon (`-->`) and would otherwise pass.
STATUS_EXEMPT = re.compile(r"(?i)status-exempt:\s*(?!-->)[A-Za-z0-9]")


class Finding:
    __slots__ = ("kind", "lineno", "detail")

    def __init__(self, kind: str, lineno: int, detail: str):
        self.kind = kind
        self.lineno = lineno
        self.detail = detail

    def __str__(self) -> str:
        return f"{self.kind} (line {self.lineno}): {self.detail}"


def anchored_paths(manifest: Path) -> set[str]:
    if not manifest.exists():
        return set()
    out = set()
    for line in manifest.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            _, _, rel = line.partition("  ")
            if rel:
                out.add(rel)
    return out


def cited_paths(span: str) -> list[str]:
    """Path-looking tokens inside one backticked span.

    A span may be a bare path, or a whole command line
    (`python tools/verify_data.py --bitstream data/center_col_10M.bin`),
    so every token is considered. A trailing `:123` line reference and any
    trailing punctuation are stripped.
    """
    found = []
    for token in PATH_TOKEN.findall(span):
        token = token.rstrip(".,;)")
        token = re.sub(r":\d+$", "", token)
        if token.startswith(PATH_ROOTS):
            found.append(token)
    return found


def is_table_row(line: str) -> bool:
    stripped = line.strip()
    return (stripped.startswith("|") and stripped.endswith("|")
            and set(stripped) != set("|- "))


def check_ledger(root: Path = REPO_ROOT) -> list[Finding]:
    ledger = root / "docs" / "CLAIM_LEDGER.md"
    if not ledger.exists():
        return [Finding("FAIL", 0, f"ledger missing: {ledger}")]

    anchored = anchored_paths(root / "data" / "MANIFEST.sha256")
    findings: list[Finding] = []

    for lineno, line in enumerate(ledger.read_text(encoding="utf-8").splitlines(), 1):
        if not is_table_row(line):
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) < 2:
            continue
        claim, level = cells[0], cells[1]

        for placeholder in PLACEHOLDER.findall(line):
            findings.append(Finding(
                "PLACEHOLDER", lineno,
                f"unfilled '{placeholder}' in row: {claim[:70]}"))

        for span in BACKTICKED.findall(line):
            for rel in cited_paths(span):
                if (root / rel).exists():
                    continue
                if rel in UNTRACKED_OK:
                    if rel not in anchored:
                        findings.append(Finding(
                            "UNANCHORED", lineno,
                            f"'{rel}' is gitignored and absent, and its hash "
                            "is not anchored in data/MANIFEST.sha256"))
                    continue
                findings.append(Finding(
                    "MISSING", lineno,
                    f"cited path '{rel}' does not exist "
                    f"(row: {claim[:60]})"))

        if CERT_LEVEL.fullmatch(level):
            scripts = [rel for span in BACKTICKED.findall(cells[2] if len(cells) > 2 else "")
                       for rel in cited_paths(span) if rel.endswith(".py")]
            if not scripts:
                findings.append(Finding(
                    "NO-VERIFIER", lineno,
                    "row graded Certificate names no verifier script in its "
                    f"Evidence column (row: {claim[:60]})"))

    return findings


def check_logs(root: Path = REPO_ROOT) -> list[Finding]:
    log_dir = root / "docs" / "experiment-logs"
    findings: list[Finding] = []
    if not log_dir.is_dir():
        return findings
    for path in sorted(log_dir.rglob("*.md")):
        rel = path.relative_to(root).as_posix()
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError as exc:
            # Six logs were cp1252 (0x97 em dash), which renders as mojibake on
            # GitHub and crashes any UTF-8 reader. Report, do not crash.
            findings.append(Finding(
                "ENCODING", 0, f"{rel}: not valid UTF-8 ({exc.reason} at "
                               f"byte {exc.start})"))
            continue
        for lineno, line in enumerate(text.splitlines(), 1):
            for placeholder in PLACEHOLDER.findall(line):
                findings.append(Finding(
                    "PLACEHOLDER", lineno,
                    f"{rel}: unfilled '{placeholder}'"))
    return findings


def newest_dated_log(root: Path) -> str | None:
    """Filename of the most recent dated log, by the date in its name.

    Filename dates, not mtime: a fresh clone gives every file the checkout
    time, so mtime would make this check pass vacuously.
    """
    log_dir = root / "docs" / "experiment-logs"
    if not log_dir.is_dir():
        return None
    dated = [p.name for p in log_dir.glob("*.md") if DATED_LOG.match(p.name)]
    return max(dated) if dated else None


def check_status(root: Path = REPO_ROOT) -> list[Finding]:
    """docs/STATUS.md is the single home for current state, and is current."""
    findings: list[Finding] = []
    status = root / "docs" / "STATUS.md"

    if not status.exists():
        return [Finding("FAIL", 0, "docs/STATUS.md missing: the repo has no "
                                   "single home for current state")]

    text = status.read_text(encoding="utf-8")
    for lineno, line in enumerate(text.splitlines(), 1):
        for placeholder in PLACEHOLDER.findall(line):
            findings.append(Finding(
                "PLACEHOLDER", lineno, f"docs/STATUS.md: unfilled "
                                       f"'{placeholder}'"))

    newest = newest_dated_log(root)
    if newest and newest not in text:
        findings.append(Finding(
            "STALE-STATUS", 0,
            f"docs/STATUS.md does not cite the newest experiment log "
            f"'{newest}'. Update STATUS.md when you add a log, or the repo's "
            "stated state silently falls behind its results."))

    return findings


def check_single_status_home(root: Path = REPO_ROOT) -> list[Finding]:
    """No file outside the allowlist may declare its own current state."""
    findings: list[Finding] = []
    for path in sorted(root.rglob("*.md")):
        if ".git" in path.parts:
            continue
        rel = path.relative_to(root).as_posix()
        if rel.startswith(STATUS_ALLOWED):
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        lines = text.splitlines()
        for lineno, line in enumerate(lines, 1):
            if not FROZEN_STATE.match(line):
                continue
            prev = lines[lineno - 2] if lineno >= 2 else ""
            if STATUS_EXEMPT.search(line) or STATUS_EXEMPT.search(prev):
                continue
            findings.append(Finding(
                "DUPLICATE-STATUS", lineno,
                f"{rel} declares its own current state "
                f"({line.strip()[:60]!r}). Current state belongs in "
                "docs/STATUS.md only - a second copy goes stale silently."))
    return findings


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    findings = (check_ledger() + check_logs() + check_status()
                + check_single_status_home())
    if findings:
        if not args.quiet:
            print(f"lint_ledger: {len(findings)} finding(s)")
            for f in findings:
                print(f"FAIL  {f}")
        return 1
    if not args.quiet:
        rows = sum(1 for line in LEDGER.read_text(encoding="utf-8").splitlines()
                   if is_table_row(line))
        logs = len(list(LOG_DIR.rglob("*.md"))) if LOG_DIR.is_dir() else 0
        print(f"lint_ledger: {rows} ledger rows, {logs} experiment logs, "
              "all cited paths present")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
