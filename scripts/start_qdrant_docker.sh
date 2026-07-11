#!/usr/bin/env bash
# Запуск Qdrant через Docker (рекомендуется для локальной разработки).
#
# Использование:
#   bash scripts/start_qdrant_docker.sh
#   bash scripts/start_qdrant_docker.sh --port 6333
#   bash scripts/start_qdrant_docker.sh --stop
#
# Параметры:
#   --port PORT     REST API порт (default: 6333)
#   --gport PORT    gRPC порт (default: 6334)
#   --data PATH     Путь к данным (default: ./rag_data/qdrant_storage)
#   --stop          Остановить и удалить контейнер
#   -h, --help      Показать справку
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

# ── Defaults ──────────────────────────────────────────────────────────
REST_PORT="6333"
GRPC_PORT="6334"
DATA_DIR="./rag_data/qdrant_storage"
CONTAINER_NAME="graphrag-qdrant"
STOP="false"

# ── Parse args ────────────────────────────────────────────────────────
usage() {
    sed -n '/^# Использование/,/^# *$/p' "$0" | sed 's/^# *//'
    exit 0
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --port)    REST_PORT="$2"; shift 2 ;;
        --gport)   GRPC_PORT="$2"; shift 2 ;;
        --data)    DATA_DIR="$2"; shift 2 ;;
        --stop)    STOP="true"; shift ;;
        -h|--help) usage ;;
        *)         echo "Unknown option: $1"; usage ;;
    esac
done

# ── Env fallback ─────────────────────────────────────────────────────
if [[ "$DATA_DIR" == "./rag_data/qdrant_storage" && -n "${QDRANT_DATA_DIR:-}" ]]; then
    DATA_DIR="$QDRANT_DATA_DIR"
fi

# ── Stop mode ─────────────────────────────────────────────────────────
if [[ "$STOP" == "true" ]]; then
    echo "Stopping Qdrant container..."
    docker stop "$CONTAINER_NAME" 2>/dev/null && echo "✓ Stopped" || echo "Not running"
    docker rm "$CONTAINER_NAME" 2>/dev/null && echo "✓ Removed" || true
    exit 0
fi

# ── Check Docker ──────────────────────────────────────────────────────
if ! command -v docker &>/dev/null; then
    echo "Error: Docker not found. Install: https://docs.docker.com/get-docker/"
    exit 1
fi

# ── Remove old container if exists ────────────────────────────────────
docker rm -f "$CONTAINER_NAME" 2>/dev/null || true

# ── Start ─────────────────────────────────────────────────────────────
mkdir -p "$DATA_DIR"

echo "─────────────────────────────────────────"
echo "  Qdrant (Docker)"
echo "─────────────────────────────────────────"
echo "  REST API:   http://localhost:${REST_PORT}"
echo "  gRPC:       localhost:${GRPC_PORT}"
echo "  Data:       ${DATA_DIR}"
echo "  Container:  ${CONTAINER_NAME}"
echo "─────────────────────────────────────────"
echo ""

docker run -d \
    --name "$CONTAINER_NAME" \
    -p "${REST_PORT}:6333" \
    -p "${GRPC_PORT}:6334" \
    -v "$(realpath "$DATA_DIR"):/qdrant/storage" \
    qdrant/qdrant:latest

echo "Waiting for Qdrant to be ready..."
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

echo "✗ Qdrant did not start in 30 seconds. Check: docker logs $CONTAINER_NAME"
exit 1
