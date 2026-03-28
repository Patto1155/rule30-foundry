# Bug Bounty Options & Ideas

> **Stage:** Discussing options only. Nothing built or committed to yet.

---

## Available Assets

- **Python 3.13** — scripting, HTTP clients, parsing
- **Playwright (Node.js)** — browser automation, JS-heavy app crawling
- **Claude Code (agentic)** — code review, pattern matching, report writing
- **GTX 1060 6GB** — not directly useful for web bounties, but available for ML-assisted analysis if needed
- **Git workflow** — already established in this repo

---

## Platform Options

| Platform | Notes |
|----------|-------|
| HackerOne | Largest, most competitive. Good for learning program structure. |
| Bugcrowd | Similar to H1, slightly less saturated |
| Intigriti | European focus, growing |
| Synack | Vetted/paid red team — harder to get in but less competition |
| Open Bug Bounty | XSS-focused, non-intrusive only |

---

## Program Type Options

| Type | Difficulty | Notes |
|------|-----------|-------|
| Web apps (JS-heavy SPAs) | Medium | Good fit for Playwright automation |
| APIs (REST/GraphQL) | Medium | Well-defined attack surface, good tooling |
| Source-available programs | Lower entry | Can static-analyse code directly |
| Mobile (Android/iOS) | Higher | Needs different toolchain |
| Smart contracts | Very different | Requires Solidity knowledge |

---

## Agentic Pipeline Ideas

### Idea 1: JS Recon Pipeline
1. Playwright crawls target, collects all JS bundles + API calls
2. Claude analyses JS for hardcoded secrets, insecure patterns, exposed endpoints
3. Auto-generate a findings report

### Idea 2: Endpoint Differ
- Periodically spider a target's JS/API surface
- Diff against previous snapshot
- Flag new endpoints/parameters for manual review

### Idea 3: Source Code Auditor
- For programs that provide source access
- Feed to Claude with a checklist: SSRF, IDOR, SQLi, path traversal, auth bypass
- Triage by severity

---

## Open Questions

- Which platform to start on?
- Passive recon only first, or dive into a live program?
- How much time per week to allocate?
- Solo or join an existing team/Discord community?
