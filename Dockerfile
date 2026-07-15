FROM python:3.13-slim AS builder

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uv

WORKDIR /app

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev

# Runtime stage
FROM python:3.13-slim

WORKDIR /app

COPY --from=builder /app/.venv /app/.venv
ENV VIRTUAL_ENV=/app/.venv \
    PATH="/app/.venv/bin:$PATH"

COPY src/ src/

EXPOSE 8765

# По умолчанию HTTP MCP режим. Qdrant embedded + SQLite.
# Для production: QDRANT_URL + DATABASE_URL (см. docker-compose.yml)
CMD ["rag-server", "--http", "--port", "8765"]
