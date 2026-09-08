# The Codex council — setup and threat model

An independent second opinion on a claim, from a model that is not this one.

Every grade in [`CLAIM_LEDGER.md`](CLAIM_LEDGER.md) is currently produced and
checked by the same agent lineage. Subagents do not fix that — they share a
model lineage and therefore its blind spots. What this adds is a differently
trained model with no stake in the repo's prior conclusions.

It is not authority. It is a source of disagreement worth explaining.

```
Claude container  --HTTPS-->  nginx (TLS, rate limit)  -->  dispatcher  -->  codex exec
   council.py                        VM :443                 127.0.0.1:8080
```

## The blocker, first, because everything else is wasted without it

The Claude container's egress is a **default-deny host allowlist**. Measured
2026-09-04, not assumed:

| Destination | Result |
|---|---|
| the VM's IP, `:443` | 403 to CONNECT — `connect_rejected` |
| `example.com:443`, `1.1.1.1:443` | 403 — the same denial |
| `pypi.org`, `github.com` | reachable |

A 403 on a bare IP *and* on `example.com` means this is a host allowlist, not
an SSH-specific or port-specific block. Two consequences:

- **ssh was never going to work**, and neither does swapping it for HTTPS on
  its own. Port 22 was not the problem. An earlier plan diagnosed the wrong
  layer.
- **The dispatcher host must be added to the environment's network policy**
  before anything below matters. That is a change to the environment (see
  <https://code.claude.com/docs/en/claude-code-on-the-web>).

  The two halves of that change do *not* land together, which is worth knowing
  before you debug the gap. The **allowlist updates live**: policy is enforced
  by the egress proxy, not baked into the container, so an already-running
  session starts reaching the host without being restarted. The **environment
  variables do not**: a process reads its environment once at start, so
  `CODEX_COUNCIL_URL` and `CODEX_COUNCIL_TOKEN` only appear after the container
  restarts — which is what starting a new session gets you.

  So a session that can `curl` the host while `council.py` still exits 2 with
  "not configured" is half-updated, not broken. Measured on 2026-09-04: the
  same host went 403 → 200 inside one session, while the variables stayed
  absent until the container was reprovisioned.

Do not attempt to tunnel through an already-allowlisted host. The proxy's own
README is explicit that policy denials are reported, not routed around.

## Before you start

You need, on the VM side: a GCP project with the instance running, and `codex`
installed **and logged in on the VM itself**. A `codex login` done in Cloud
Shell does not carry over — different filesystem, different `~/.codex`. This
is the single most common failure: the service starts cleanly and then returns
502 on every request.

```bash
gcloud compute ssh codex-dispatcher --zone=australia-southeast1-b
# then, ON THE VM:
sudo apt update && sudo apt install -y nodejs npm git python3 bubblewrap
sudo npm install -g @openai/codex
codex login --device-auth
codex exec --skip-git-repo-check "Reply with exactly: VM Codex works"
```

Do not continue until that last command prints `VM Codex works` **on the VM**.

## 1. Pin the IP before you name it

An instance's external IP is ephemeral by default: it changes when the VM
stops and starts. Since the next steps put that address into DNS, a TLS
certificate, and an allowlist, letting it move breaks all three at once, in a
way that reads as a network fault.

```bash
gcloud compute addresses create codex-dispatcher-ip \
  --region=australia-southeast1 --addresses <CURRENT_IP>
```

## 2. Give it a name with a real certificate

The egress proxy re-terminates TLS and **validates the certificate it sees**.
A self-signed cert fails there, not here, so the error surfaces as a connection
problem rather than a trust problem. You need a real cert, which means a real
DNS name.

- Own a domain: point an A record at the reserved IP.
- Don't: `<ip-with-dashes>.sslip.io` resolves to that IP, and Let's Encrypt
  will issue for it. Fine for this; be aware it is a third-party wildcard DNS
  service and subject to LE rate limits.

## 3. Open the port

An instance created without the `https-server` tag has no ingress rule for 443.

```bash
gcloud compute firewall-rules create allow-codex-https \
  --allow=tcp:443,tcp:80 --target-tags=codex-dispatcher
gcloud compute instances add-tags codex-dispatcher \
  --zone=australia-southeast1-b --tags=codex-dispatcher
```

Port 80 is only for the ACME challenge; the nginx config redirects it.

## 4. Install the dispatcher

Copy `tools/codex_dispatcher/` to the VM, then:

```bash
sudo bash install.sh <your-hostname> <the-user-that-ran-codex-login>
```

It generates a token, installs the service and the nginx site, obtains a
certificate, and prints the two environment values you need. It is idempotent
and does **not** rotate the token on a re-run.

Verify from anywhere:

```bash
curl -sS https://<your-hostname>/health
```

## 5. Wire up the Claude side

In the environment's settings, set:

```
CODEX_COUNCIL_URL=https://<your-hostname>/ask
CODEX_COUNCIL_TOKEN=<the token install.sh printed>
```

Add `<your-hostname>` to the environment's **network allowlist**. Then start a
new session and check the two layers separately — this is why `/health` is
unauthenticated:

```bash
python tools/council.py --check          # allowlist works? (no token needed)
echo "2+2=5" | python tools/council.py redteam    # token works?
```

## Using it

```bash
echo "<claim>" | python tools/council.py redteam
python tools/council.py review --file docs/experiment-logs/foo.md
python tools/council.py math --json < argument.txt
python tools/council.py redteam --dry-run < claim.txt   # sends nothing
```

A role is a **prompt framing**, not a different model. All three carry the
repo's known failure modes — the counting bound, bit order, the single seed,
right-censoring, noise floors — because a reviewer without them returns
generic advice.

Record what comes back the way any other evidence is recorded: in the
experiment log, with the disagreement stated. A council answer is not a grade,
and agreement from it does not promote a row in the ledger.

## Threat model

The endpoint runs an agent on a VM using your ChatGPT credentials. A leaked
token buys an attacker model usage on your account and whatever `codex exec`
can reach on that box. Public IPs are scanned continuously; assume it will be
found.

What the design does about it:

| Control | Where |
|---|---|
| Bearer token, `hmac.compare_digest`, ≥32 chars enforced at startup | `server.py` |
| Refuses to start with no token — an unauthenticated dispatcher is an open door | `server.py` |
| Prompt passed as one argv element, never through a shell | `build_codex_argv` |
| Model allowlist, so a leaked token cannot pick an expensive model | `server.py` |
| `--sandbox read-only`, **probed** rather than assumed | `probe_codex_flags` |
| Rate limit 10/min | `nginx.conf.example` |
| Loopback bind; TLS only at nginx | unit + nginx |
| Token never logged; prompts logged as length + SHA-256 prefix | `prompt_fingerprint` |

What it does **not** do: source-IP restriction. The container's egress
addresses are not stable or published, so an IP allowlist is not practical —
the token is the only thing standing between the internet and this endpoint.
Rotate it by editing `/etc/codex-dispatcher.env` and restarting the service.

Cheapest risk reduction available: `gcloud compute instances stop
codex-dispatcher` when you are not using it.

## Troubleshooting

| Symptom | Cause |
|---|---|
| `council.py` exits 2, "not configured" | env vars not set, set on a different environment, or set after this container started — they need a restart, unlike the allowlist |
| 403, no `X-Dispatcher` header | host not on the egress allowlist (this one *does* apply to a running session, so a 403 here is the entry itself, not staleness) |
| 401/403 **with** `X-Dispatcher: codex` | token mismatch — the dispatcher was reached |
| 502 on every request | `codex login` was done as the wrong user; check `~/.codex/auth.json` for the service user |
| 504 after ~60s | nginx `proxy_read_timeout` left at its default; it must exceed `CODEX_TIMEOUT_S` |
| Works locally on the VM, fails externally | firewall tag missing (step 3) |

`journalctl -u codex-dispatcher -f` on the VM shows role, model, duration, and
a prompt fingerprint — never the prompt itself.
