# The council — external reviewers on demand

Stable reference. What the repo *knows* is in [`CLAIM_LEDGER.md`](CLAIM_LEDGER.md);
what is *in flight* is in [`STATUS.md`](STATUS.md). This file only describes how
the mechanism works.

Every expensive mistake in this repo was **silent**: a byte-reversed bitstream
that kept a 0.5 bit mean, a counting bound that made a negative result vacuous,
a Certificate row citing a template full of placeholders. None of them was
caught by a check; each was caught by a reader who did not share the author's
assumptions. `tools/council.py` buys that reader on demand from a model outside
this session, and writes the reply to a file so it can be reviewed instead of
scrolling past.

A reviewer's reply is **not evidence**. It has not run `tools/verify_all.py`,
cannot see the gitignored bitstreams, and its numbers are unverified. A reply
that matters gets promoted into `docs/experiment-logs/` under the normal logging
standard, where the ledger can cite it; nothing in `briefs/` is tracked.

## Roles

`python tools/council.py roles` prints the live table. As configured:

| Role | Backend | Repo access | Use it for |
|---|---|---|---|
| `review` | Codex (ChatGPT sub) | yes, read-only | diffs, code review, "what breaks this" |
| `math` | Codex (ChatGPT sub) | yes, read-only | counting bounds, vacuity, proof gaps |
| `lit` | OpenRouter | no | prior art, citations, "is this known" |
| `redteam` | OpenRouter | no | adversarial reading of a claim |

Codex roles spend your ChatGPT subscription's rate limits — the same 5-hour and
weekly windows as your own interactive Codex use. Reserve them for jobs that
need repo access or heavy reasoning; send literature fan-out to OpenRouter,
where it costs pennies.

## Running it

```bash
python tools/council.py roles                       # the table above
python tools/council.py doctor                      # is this machine wired up?
python tools/council.py ask math briefs/c2c.md --repo --out briefs/out/c2c-math.md
python tools/council.py ask lit - < brief.md        # stdin, reply on stdout
```

Briefs start from [`briefs/TEMPLATE.md`](../briefs/TEMPLATE.md). The reviewer
sees only the brief (plus read-only repo files with `--repo` on a Codex role),
so a brief that says "the grid described above" gets a confident answer to a
question nobody asked.

`.claude/agents/dispatcher.md` wraps the same loop as a subagent, for when you
want several briefs written and fanned out without spending the main session's
context on it.

## Configuring the cloud environment

Do this once, at claude.ai → Code → Environments. **Create a new environment
named `council`; do not edit Default** — the Codex token is readable by anyone
with access to the environment, so it should not be present in every session you
ever open.

**1. Network access → Custom**, with *"Also include default list of common
package managers"* ticked (npm has to reach the registry):

```
chatgpt.com
*.chatgpt.com
auth.openai.com
*.openai.com
openrouter.ai
*.frame.claudeusercontent.com
```

The Trusted preset does **not** cover the OpenAI endpoints — it is package
registries, GitHub, container and cloud SDKs — so Codex cannot connect under it.
The last line keeps published artifacts readable, which Trusted gives you for
free and Custom does not. Drop `openrouter.ai` if you configure it as an API
credential instead (see step 4); credential-proxied hosts bypass the allowlist.

**2. Environment variables.** On a machine you trust:

```bash
codex login                             # browser, ChatGPT account
jq -r .auth_mode ~/.codex/auth.json     # must print: chatgpt
base64 -w0 ~/.codex/auth.json; echo     # macOS: base64 -i ~/.codex/auth.json
```

Then set `CODEX_AUTH_B64` to that single line. Base64 because `auth.json` is
full of double quotes and the variable editor is not a shell — a raw paste
breaks on the first one.

**3. Setup script.** Paste the contents of
[`tools/council_env_setup.sh`](../tools/council_env_setup.sh). It installs the
Codex CLI, seeds `~/.codex/auth.json`, and installs `requirements-ci.txt`. It
runs before Claude starts, so `codex` is authenticated by the time any Bash call
reaches it.

**4. API credentials** (optional, OpenRouter only). Name `openrouter`, allowed
site `openrouter.ai`, header `Authorization` with prefix `Bearer`. The proxy
injects it in flight, so the key is never visible to Claude or to
`council.py` — which is why `run_openrouter` tolerates an empty
`OPENROUTER_API_KEY`.

## Smoke test, first session in a new container

```bash
python tools/council.py doctor
codex exec --skip-git-repo-check --sandbox read-only "Reply with exactly: council online"
```

## When it breaks

| Symptom | Cause | Fix |
|---|---|---|
| `codex exited N with no answer`, 401 in the tail | seeded token has aged out | run `codex exec "ok"` on the machine you logged in from — that refreshes `auth.json` — then re-base64 it into `CODEX_AUTH_B64` |
| `codex is not installed` | setup script did not run, or npm was blocked | rerun `bash tools/council_env_setup.sh`; check the package-manager default list is ticked |
| `openrouter unreachable` | host not allowlisted and no API credential | add `openrouter.ai` under Custom, or configure the credential |
| `auth_mode` is not `chatgpt` | you logged in with an API key, not the subscription | `codex logout && codex login` |

Expect to refresh the seed roughly weekly: Codex renews the token in place
inside the container, and the container is ephemeral, so the *seed* ages even
though your local login does not.

## Boundaries

- Reviewers run `--sandbox read-only`. They report; Claude applies. Two agents
  writing one tree is a lost afternoon, and a patch from something that cannot
  run `tools/verify_all.py` is a suggestion.
- Never put a credential, `auth.json`, or an environment variable value into a
  brief. Briefs leave the machine.
- Don't route a review role back to Claude. The independence is the product.
