#!/usr/bin/env bash
# Install the Codex dispatcher on the VM. Run ON THE VM, with sudo.
#
#   sudo bash install.sh <public-hostname> [codex-user]
#
# The hostname must be a real DNS name, not the bare IP: the caller's egress
# proxy re-terminates TLS and validates the certificate, so a self-signed cert
# on an IP will fail there rather than here, in a way that looks like a network
# fault. If you do not own a domain, <ip-with-dashes>.sslip.io resolves to that
# IP and Let's Encrypt will issue for it.
#
# Idempotent: safe to re-run. Re-running does NOT rotate the token.

set -euo pipefail

HOST="${1:-}"
CODEX_USER="${2:-${SUDO_USER:-}}"
ENV_FILE=/etc/codex-dispatcher.env
APP_DIR=/opt/codex-dispatcher
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# The token is generated early and printed at the very end. Under `set -e` any
# failure in between -- a certbot problem, say -- aborts before that print and
# leaves the operator with a token they have never seen. Say where it lives.
trap 'st=$?; if [[ $st -ne 0 && -f "$ENV_FILE" ]]; then
  echo >&2
  echo "install did not complete (exit $st)." >&2
  echo "The token was already generated. Read it with:" >&2
  echo "  sudo grep CODEX_COUNCIL_TOKEN $ENV_FILE" >&2
  echo "Re-running this script is safe and keeps that token." >&2
fi' EXIT

if [[ -z "$HOST" ]]; then
  echo "usage: sudo bash install.sh <public-hostname> [codex-user]" >&2
  exit 2
fi
if [[ -z "$CODEX_USER" ]]; then
  echo "could not determine the codex user; pass it as the second argument" >&2
  exit 2
fi
if [[ $EUID -ne 0 ]]; then
  echo "run with sudo" >&2
  exit 2
fi
if ! id "$CODEX_USER" >/dev/null 2>&1; then
  echo "no such user: $CODEX_USER" >&2
  exit 2
fi

# The service authenticates as whoever ran `codex login`. Checking now turns a
# silent 502-on-every-request into a clear failure before anything is exposed.
if [[ ! -f "/home/$CODEX_USER/.codex/auth.json" ]]; then
  echo "ERROR: /home/$CODEX_USER/.codex/auth.json is missing." >&2
  echo "Run 'codex login --device-auth' AS $CODEX_USER first." >&2
  echo "A login done in Cloud Shell does not carry over to this VM." >&2
  exit 1
fi

echo "==> installing service files"
install -d -m 755 "$APP_DIR"
install -m 755 "$HERE/server.py" "$APP_DIR/server.py"

if [[ -f "$ENV_FILE" ]]; then
  echo "==> $ENV_FILE exists; keeping the existing token"
else
  echo "==> generating a token"
  TOKEN="$(openssl rand -hex 32)"
  cat > "$ENV_FILE" <<ENVEOF
CODEX_COUNCIL_TOKEN=$TOKEN
CODEX_BIND=127.0.0.1
CODEX_PORT=8080
CODEX_DEFAULT_MODEL=gpt-5.6-sol
CODEX_ALLOWED_MODELS=gpt-5.6-sol
CODEX_TIMEOUT_S=900
CODEX_MAX_PROMPT=100000
ENVEOF
  chmod 600 "$ENV_FILE"
  chown root:root "$ENV_FILE"
fi

echo "==> installing systemd unit"
sed "s/__CODEX_USER__/$CODEX_USER/g" "$HERE/codex-dispatcher.service" \
  > /etc/systemd/system/codex-dispatcher.service
systemctl daemon-reload
systemctl enable --now codex-dispatcher
sleep 2
systemctl is-active --quiet codex-dispatcher || {
  echo "ERROR: service did not start. Logs:" >&2
  journalctl -u codex-dispatcher -n 30 --no-pager >&2
  exit 1
}

echo "==> local health check"
curl -fsS http://127.0.0.1:8080/health && echo

echo "==> installing nginx site for $HOST"
apt-get install -y -qq nginx certbot python3-certbot-nginx >/dev/null
sed "s/__CODEX_HOST__/$HOST/g" "$HERE/nginx.conf.example" \
  > /etc/nginx/sites-available/codex-dispatcher
rm -f /etc/nginx/sites-enabled/default

# Two constraints that together dictate the order here, and that an earlier
# version of this script got wrong in a way that cost a run:
#
#   - The real site cannot be enabled before the certificate exists. Its
#     server block names ssl_certificate paths, and nginx refuses to start
#     when they are absent.
#   - The ACME challenge needs something actually listening on :80. The
#     earlier version disabled the default site AND moved the dispatcher site
#     out of sites-enabled before calling certbot, which left nginx serving
#     no server blocks at all -- so Let's Encrypt got "Connection refused"
#     and the whole install aborted under `set -e`, before printing the token.
#
# So: stand up a challenge-only :80 site, get the cert, then swap in the real
# one. "Connection refused" here means nothing is listening; a *timeout* would
# mean the firewall rule or the network tag is missing instead.
if [[ ! -d "/etc/letsencrypt/live/$HOST" ]]; then
  echo "==> obtaining a certificate for $HOST"
  install -d -m 755 /var/www/html
  cat > /etc/nginx/sites-available/codex-acme <<ACMEEOF
server {
    listen 80 default_server;
    server_name $HOST;
    root /var/www/html;
}
ACMEEOF
  ln -sf /etc/nginx/sites-available/codex-acme \
         /etc/nginx/sites-enabled/codex-acme
  nginx -t
  systemctl restart nginx
  certbot certonly --webroot -w /var/www/html -d "$HOST" \
    --non-interactive --agree-tos --register-unsafely-without-email
  rm -f /etc/nginx/sites-enabled/codex-acme
fi

ln -sf /etc/nginx/sites-available/codex-dispatcher \
       /etc/nginx/sites-enabled/codex-dispatcher
nginx -t
systemctl restart nginx

echo
echo "==> done. Verify from outside:"
echo "    curl -sS https://$HOST/health"
echo
echo "Set these in the Claude Code environment (Settings -> Environment):"
echo "    CODEX_COUNCIL_URL=https://$HOST/ask"
echo -n "    CODEX_COUNCIL_TOKEN="
grep '^CODEX_COUNCIL_TOKEN=' "$ENV_FILE" | cut -d= -f2-
echo
echo "Then add $HOST to the environment's network allowlist and start a NEW"
echo "session -- an existing session keeps the policy it started with."
