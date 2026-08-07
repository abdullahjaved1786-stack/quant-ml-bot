#!/usr/bin/env bash
# One-command deployment for Oracle Cloud Always-Free VPS (Ubuntu 22.04+).
# Idempotent — safe to re-run.

set -euo pipefail

REPO_DIR="/opt/quant-ml-bot"
SERVICE_NAME="quant-bot"
SERVICE_FILE="deploy/${SERVICE_NAME}.service"

log()  { printf "\033[1;34m[deploy]\033[0m %s\n" "$*"; }
fail() { printf "\033[1;31m[fail]\033[0m %s\n" "$*" >&2; exit 1; }

[[ $EUID -eq 0 ]] || fail "Run as root: sudo bash $0"

if ! command -v python3.11 >/dev/null 2>&1; then
    log "Installing Python 3.11..."
    apt-get update
    apt-get install -y software-properties-common
    add-apt-repository -y ppa:deadsnakes/ppa
    apt-get update
    apt-get install -y python3.11 python3.11-venv python3.11-dev
fi

if ! id -u quant >/dev/null 2>&1; then
    log "Creating 'quant' system user..."
    useradd --system --create-home --shell /bin/bash quant
fi

log "Preparing ${REPO_DIR}..."
mkdir -p "${REPO_DIR}"
chown quant:quant "${REPO_DIR}"

if [[ ! -d "${REPO_DIR}/.git" ]] && [[ ! -f "${REPO_DIR}/main.py" ]]; then
    log "Cloning repository..."
    if [[ -n "${QUANT_BOT_REPO:-}" ]]; then
        sudo -u quant git clone "${QUANT_BOT_REPO}" "${REPO_DIR}"
    else
        fail "Set QUANT_BOT_REPO env var to your git URL before running."
    fi
fi

cd "${REPO_DIR}"
chown -R quant:quant "${REPO_DIR}"

log "Creating venv + installing deps..."
sudo -u quant python3.11 -m venv .venv
sudo -u quant .venv/bin/pip install --upgrade pip
sudo -u quant .venv/bin/pip install -r requirements.txt joblib

log "Creating data + secrets dirs..."
mkdir -p data data/models secrets
chown -R quant:quant data secrets
chmod 700 secrets

if [[ ! -f .env ]]; then
    log "Creating placeholder .env (edit before enabling Sheets)..."
    cat > .env <<'EOF'
GOOGLE_SHEET_NAME=
GOOGLE_SHEET_CREDS=/opt/quant-ml-bot/secrets/gcp.json
EOF
    chown quant:quant .env
    chmod 600 .env
fi

log "Installing systemd service..."
cp "${SERVICE_FILE}" /etc/systemd/system/${SERVICE_NAME}.service
systemctl daemon-reload
systemctl enable ${SERVICE_NAME}
systemctl restart ${SERVICE_NAME}

sleep 2
if systemctl is-active --quiet ${SERVICE_NAME}; then
    log "Service is RUNNING. Tailing logs:"
    journalctl -u ${SERVICE_NAME} -n 30 --no-pager
else
    fail "Service failed to start. Check: journalctl -u ${SERVICE_NAME}"
fi

log "Deployment complete."
log "Useful commands:"
log "  sudo systemctl status  ${SERVICE_NAME}"
log "  sudo journalctl -u ${SERVICE_NAME} -f"
log "  sudo systemctl stop   ${SERVICE_NAME}"
