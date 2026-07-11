#!/usr/bin/env bash
# Запуск Qdrant bare metal (без Docker).
# Скачивает бинарник при первом запуске, затем запускает локально.
#
# Использование:
#   bash scripts/start_qdrant_local.sh
#   bash scripts/start_qdrant_local.sh --port 6333
#   bash scripts/start_qdrant_local.sh --stop
#
# Параметры:
#   --port PORT     REST API порт (default: 6333)
#   --gport PORT    gRPC порт (default: 6334)
#   --data PATH     Путь к данным (default: ./rag_data/qdrant_storage)
#   --version VER   Версия Qdrant (default: v1.14.1)
#   --stop          Остановить запущенный процесс
#   -h, --help      Показать справку
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

# ── Defaults ──────────────────────────────────────────────────────────
REST_PORT="6333"
GRPC_PORT="6334"
DATA_DIR="./rag_data/qdrant_storage"
QDRANT_VERSION="v1.14.1"
STOP="false"
QDRANT_BIN="$PROJECT_DIR/.local/bin/qdrant"
PID_FILE="$PROJECT_DIR/.local/qdrant.pid"

# ── Parse args ────────────────────────────────────────────────────────
usage() {
    sed -n '/^# Использование/,/^# *$/p' "$0" | sed 's/^# *//'
    exit 0
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --port)     REST_PORT="$2"; shift 2 ;;
        --gport)    GRPC_PORT="$2"; shift 2 ;;
        --data)     DATA_DIR="$2"; shift 2 ;;
        --version)  QDRANT_VERSION="$2"; shift 2 ;;
        --stop)     STOP="true"; shift ;;
        -h|--help)  usage ;;
        *)          echo "Unknown option: $1"; usage ;;
    esac
done

# ── Stop mode ─────────────────────────────────────────────────────────
if [[ "$STOP" == "true" ]]; then
    if [[ -f "$PID_FILE" ]]; then
        PID=$(cat "$PID_FILE")
        if kill -0 "$PID" 2>/dev/null; then
            echo "Stopping Qdrant (PID $PID)..."
            kill "$PID"
            rm -f "$PID_FILE"
            echo "✓ Stopped"
        else
            echo "Qdrant not running (stale PID file)"
            rm -f "$PID_FILE"
        fi
    else
        echo "No PID file found. Try: pkill qdrant"
    fi
    exit 0
fi

# ── Check architecture ────────────────────────────────────────────────
ARCH=$(uname -m)
case "$ARCH" in
    x86_64)  ARCH="amd64" ;;
    aarch64) ARCH="arm64" ;;
    *)       echo "Unsupported architecture: $ARCH"; exit 1 ;;
esac

# ── Download if not present ───────────────────────────────────────────
if [[ ! -f "$QDRANT_BIN" ]]; then
    echo "Qdrant not found. Downloading ${QDRANT_VERSION}..."
    mkdir -p "$(dirname "$QDRANT_BIN")"

    DOWNLOAD_URL="https://github.com/qdrant/qdrant/releases/download/${QDRANT_VERSION}/qdrant-${ARCH}-unknown-linux-gnu.tar.gz"
    echo "  URL: $DOWNLOAD_URL"

    TMP_DIR=$(mktemp -d)
    curl -sL "$DOWNLOAD_URL" -o "$TMP_DIR/qdrant.tar.gz"
    tar -xzf "$TMP_DIR/qdrant.tar.gz" -C "$TMP_DIR"
    mv "$TMP_DIR/qdrant" "$QDRANT_BIN"
    chmod +x "$QDRANT_BIN"
    rm -rf "$TMP_DIR"

    echo "✓ Downloaded to $QDRANT_BIN"
fi

# ── Check if already running ──────────────────────────────────────────
if [[ -f "$PID_FILE" ]]; then
    OLD_PID=$(cat "$PID_FILE")
    if kill -0 "$OLD_PID" 2>/dev/null; then
        echo "Qdrant already running (PID $OLD_PID) on port ${REST_PORT}"
        echo "Stop with: bash scripts/start_qdrant_local.sh --stop"
        exit 0
    fi
    rm -f "$PID_FILE"
fi

# ── Start ─────────────────────────────────────────────────────────────
mkdir -p "$DATA_DIR"

echo "─────────────────────────────────────────"
echo "  Qdrant (bare metal)"
echo "─────────────────────────────────────────"
echo "  REST API:   http://localhost:${REST_PORT}"
echo "  gRPC:       localhost:${GRPC_PORT}"
echo "  Data:       ${DATA_DIR}"
echo "  Binary:     ${QDRANT_BIN}"
echo "  Version:    ${QDRANT_VERSION}"
echo "─────────────────────────────────────────"
echo ""

QDRANT__SERVICE__HTTP_PORT="$REST_PORT" \
QDRANT__SERVICE__GRPC_PORT="$GRPC_PORT" \
QDRANT__STORAGE__STORAGE_PATH="$DATA_DIR" \
nohup "$QDRANT_BIN" > "$PROJECT_DIR/.local/qdrant.log" 2>&1 &

echo $! > "$PID_FILE"
echo "Qdrant starting (PID $(cat "$PID_FILE"))..."

# Wait for readiness
for i in $(seq 1 30); do
    if curl -sf "http://localhost:${REST_PORT}/healthz" >/dev/null 2>&1; then
        echo "✓ Qdrant is ready on http://localhost:${REST_PORT}"
        echo ""
        echo "Set in your .env:"
        echo "  QDRANT_URL=http://localhost:${REST_PORT}"
        exit 0
    fi
    sleep 1
done

echo "✗ Qdrant did not start in 30 seconds. Check: cat .local/qdrant.log"
exit 1
