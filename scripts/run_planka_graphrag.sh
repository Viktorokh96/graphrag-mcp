#!/usr/bin/env bash
# Запуск чистого Docker-инстанса GraphRAG (порт 8766, свои данные)
#
# Использование:
#   bash scripts/run_planka.sh
#   EMBEDDING_API_KEY=sk-... bash scripts/run_planka.sh
#   PORT=8777 bash scripts/run_planka.sh
#
# Переменные окружения (опциональны):
#   EMBEDDING_API_KEY   API-ключ для провайдера эмбеддингов
#   PORT                HTTP порт (default: 8766)
#   QDRANT_PORT         Qdrant REST порт (default: 6335)
#   DATA_DIR            Директория данных хоста (default: ./rag_data_clean)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

PROJECT_NAME="${PROJECT_NAME:-graphrag-planka}"
PORT="${PORT:-8766}"
QDRANT_PORT="${QDRANT_PORT:-6335}"
DATA_DIR="${DATA_DIR:-./rag_data_clean}"

# ── API Key ──────────────────────────────────────────────────────────
if [[ -z "${EMBEDDING_API_KEY:-}" ]]; then
    echo "ERROR: EMBEDDING_API_KEY not set."
    echo "  export EMBEDDING_API_KEY=sk-..."
    echo "  or: EMBEDDING_API_KEY=sk-... bash scripts/run_planka.sh"
    exit 1
fi

# ── Prepare data dir ─────────────────────────────────────────────────
mkdir -p "$DATA_DIR"

# ── Build config from .env.clean + overrides ─────────────────────────
ENV_FILE="$PROJECT_DIR/.env.clean"
if [[ ! -f "$ENV_FILE" ]]; then
    echo "ERROR: $ENV_FILE not found"
    exit 1
fi

# ── Stop existing if running ─────────────────────────────────────────
if docker compose --project-name "$PROJECT_NAME" ps --quiet 2>/dev/null | grep -q .; then
    echo "Stopping existing $PROJECT_NAME containers..."
    docker compose --project-name "$PROJECT_NAME" down
fi

# ── Launch ───────────────────────────────────────────────────────────
echo "─────────────────────────────────────────"
echo "  GraphRAG — Planka (clean instance)"
echo "─────────────────────────────────────────"
echo "  HTTP API:   http://localhost:${PORT}"
echo "  MCP SSE:    http://localhost:${PORT}/mcp"
echo "  OpenAPI:    http://localhost:${PORT}/docs"
echo "  WebUI:      http://localhost:${PORT}/webui/"
echo "  Qdrant:     http://localhost:${QDRANT_PORT}"
echo "  Data dir:   ${DATA_DIR}"
echo "─────────────────────────────────────────"
echo ""

EMBEDDING_API_KEY="$EMBEDDING_API_KEY" \
HTTP_PORT="$PORT" \
QDRANT_PORT="$QDRANT_PORT" \
STORE_DATA_DIR="$DATA_DIR" \
QDRANT_DATA_DIR="$DATA_DIR/qdrant_storage" \
    docker compose \
        --project-name "$PROJECT_NAME" \
        --env-file "$ENV_FILE" \
        up --build -d

# ── Wait for healthy ────────────────────────────────────────────────
echo ""
echo "Waiting for health check..."
for i in $(seq 1 60); do
    if curl -sf "http://localhost:${PORT}/health" > /dev/null 2>&1; then
        echo ""
        echo "Ready!  http://localhost:${PORT}"
        echo ""
        echo "  Health:  curl http://localhost:${PORT}/health"
        echo "  Search:  curl -X POST http://localhost:${PORT}/search -H 'Content-Type: application/json' -d '{\"query\":\"test\"}'"
        echo "  Stop:    docker compose --project-name $PROJECT_NAME down"
        exit 0
    fi
    sleep 2
    echo -n "."
done

echo ""
echo "ERROR: Health check timeout after 120s"
echo "  Check logs: docker compose --project-name $PROJECT_NAME logs rag"
docker compose --project-name "$PROJECT_NAME" logs --tail=30 rag
exit 1
