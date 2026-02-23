#!/usr/bin/env bash
set -euo pipefail

die() { echo "Error: $*" >&2; exit 2; }

usage() {
  cat <<'EOF'
Serve a directory tree with a simple "video wall" UI.

Server:
  bash scripts/utils/serve_gallery.sh --root /mnt --port 18080 --bind 127.0.0.1

Laptop (SSH forward):
  ssh -N -L 18080:127.0.0.1:18080 user@server
  open http://127.0.0.1:18080/__gallery__/

Args:
  --root       Root directory to expose (default: /mnt)
  --port       Port (default: 18080)
  --bind       Bind host (default: 127.0.0.1)
  --ssh-host   If set, prints the ssh -L command to run on your laptop
  --local-port Local port for ssh -L (default: same as --port)
EOF
}

ROOT="/mnt/nvme-fast/zbs/ctrl-world/Ctrl-World/synthetic_data"
PORT="18081"
BIND="127.0.0.1"
SSH_HOST=""
LOCAL_PORT=""
PYTHON_BIN="${PYTHON_BIN:-python3}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --root) ROOT="${2:-}"; shift 2 ;;
    --port) PORT="${2:-}"; shift 2 ;;
    --bind) BIND="${2:-}"; shift 2 ;;
    --ssh-host) SSH_HOST="${2:-}"; shift 2 ;;
    --local-port) LOCAL_PORT="${2:-}"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown arg: $1" >&2; usage; exit 2 ;;
  esac
done

[[ -d "${ROOT}" ]] || die "--root is not a directory: ${ROOT}"
[[ -n "${LOCAL_PORT}" ]] || LOCAL_PORT="${PORT}"

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  command -v python >/dev/null 2>&1 && PYTHON_BIN="python"
  command -v python3 >/dev/null 2>&1 && PYTHON_BIN="python3"
fi
command -v "${PYTHON_BIN}" >/dev/null 2>&1 || die "python not found"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SERVER="${SCRIPT_DIR}/media_gallery_server.py"

echo "Root:    ${ROOT}"
echo "Bind:    ${BIND}"
echo "Port:    ${PORT}"
echo "Gallery: http://${BIND}:${PORT}/__gallery__/"
if [[ -n "${SSH_HOST}" ]]; then
  echo ""
  echo "SSH forward (run on your laptop):"
  echo "  ssh -N -L ${LOCAL_PORT}:127.0.0.1:${PORT} ${SSH_HOST}"
  echo "Then open:"
  echo "  http://127.0.0.1:${LOCAL_PORT}/__gallery__/"
fi
echo ""
echo "Press Ctrl-C to stop."
exec "${PYTHON_BIN}" "${SERVER}" --root "${ROOT}" --bind "${BIND}" --port "${PORT}"
