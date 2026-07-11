#!/usr/bin/env bash
# Запуск RAG MCP-сервера в режиме HTTP SSE streaming.
#
# Использование:
#   bash scripts/start_http_mcp.sh
#   bash scripts/start_http_mcp.sh --port 9000
#   bash scripts/start_http_mcp.sh --preload --qdrant
#   bash scripts/start_http_mcp.sh --offline --models-dir ./models
#
# Параметры (все опциональны, дефолты берутся из .env / окружения):
#   --port PORT           Порт HTTP сервера (default: 8765)
#   --store PATH          Путь к хранилищу (default: ./rag_data)
#   --provider NAME       Провайдер эмбеддингов: bge-m3 | ollama | openrouter
#   --qdrant              Использовать Qdrant Server (QDRANT_URL=http://localhost:6333)
#   --preload             Загрузить модели при старте (PRELOAD_MODELS=true)
#   --offline             Не обращаться к HuggingFace Hub (HF_HUB_OFFLINE=true)
#   --models-dir PATH     Директория с локальными моделями (MODELS_DIR)
#   --rerank              Включить CrossEncoder reranking (RERANK_ENABLED=true)
#   --expand              Включить multi-query expansion (QUERY_EXPANSION_ENABLED=true)
#   --no-dotenv           Не загружать .env файл
#   -h, --help            Показать справку
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

# ── Defaults ──────────────────────────────────────────────────────────
PORT="8765"
STORE_PATH="./rag_data"
EMBEDDING_PROVIDER=""
QDRANT_URL=""
PRELOAD="false"
HF_OFFLINE="false"
MODELS_DIR=""
RERANK_ENABLED="false"
QUERY_EXPANSION="false"
LOAD_DOTENV="true"

# ── Parse args ────────────────────────────────────────────────────────
usage() {
    sed -n '/^# Использование/,/^# *$/p' "$0" | sed 's/^# *//'
    exit 0
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --port)       PORT="$2"; shift 2 ;;
        --store)      STORE_PATH="$2"; shift 2 ;;
        --provider)   EMBEDDING_PROVIDER="$2"; shift 2 ;;
        --qdrant)     QDRANT_URL="http://localhost:6333"; shift ;;
        --preload)    PRELOAD="true"; shift ;;
        --offline)    HF_OFFLINE="true"; shift ;;
        --models-dir) MODELS_DIR="$2"; shift 2 ;;
        --rerank)     RERANK_ENABLED="true"; shift ;;
        --expand)     QUERY_EXPANSION="true"; shift ;;
        --no-dotenv)  LOAD_DOTENV="false"; shift ;;
        -h|--help)    usage ;;
        *)            echo "Unknown option: $1"; usage ;;
    esac
done

# ── Load .env if exists ──────────────────────────────────────────────
if [[ "$LOAD_DOTENV" == "true" && -f .env ]]; then
    set -a
    # shellcheck source=/dev/null
    source .env
    set +a
    echo "✓ Loaded .env"
fi

# ── Apply CLI overrides (CLI flags > .env > defaults) ────────────────
export STORE_PATH="${STORE_PATH}"
export EMBEDDING_MODEL="${EMBEDDING_PROVIDER:-${EMBEDDING_MODEL:-bge-m3}}"
[[ -n "$QDRANT_URL" ]]   && export QDRANT_URL="$QDRANT_URL"
[[ "$PRELOAD" == "true" ]]    && export PRELOAD_MODELS=true
[[ "$HF_OFFLINE" == "true" ]] && export HF_HUB_OFFLINE=true
[[ -n "$MODELS_DIR" ]]        && export MODELS_DIR="$MODELS_DIR"
[[ "$RERANK_ENABLED" == "true" ]] && export RERANK_ENABLED=true
[[ "$QUERY_EXPANSION" == "true" ]] && export QUERY_EXPANSION_ENABLED=true

# ── Print config ─────────────────────────────────────────────────────
echo "─────────────────────────────────────────"
echo "  RAG MCP Server (HTTP SSE)"
echo "─────────────────────────────────────────"
echo "  URL:        http://localhost:${PORT}"
echo "  MCP SSE:    http://localhost:${PORT}/mcp"
echo "  Docs:       http://localhost:${PORT}/docs"
echo "  Provider:   ${EMBEDDING_MODEL}"
echo "  Store:      ${STORE_PATH}"
[[ -n "$QDRANT_URL" ]] && echo "  Qdrant:     ${QDRANT_URL}"
[[ "$PRELOAD" == "true" ]] && echo "  Preload:    models will be loaded at startup"
[[ -n "$MODELS_DIR" ]] && echo "  Models dir: ${MODELS_DIR}"
[[ "$RERANK_ENABLED" == "true" ]] && echo "  Reranker:   enabled"
echo "─────────────────────────────────────────"
echo ""

# ── Start ─────────────────────────────────────────────────────────────
exec uv run python3 -m src.cli --http --port "$PORT"
