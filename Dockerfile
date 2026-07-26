FROM python:3.13-slim AS builder

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

COPY pyproject.toml uv.lock ./
COPY AGENTS.md LICENSE ./
RUN uv sync --frozen --no-dev --extra postgres

# Runtime stage
FROM python:3.13-slim

WORKDIR /app
COPY --from=builder /app/.venv /app/.venv
COPY src/ src/
ENV VIRTUAL_ENV=/app/.venv \
    PATH="/app/.venv/bin:$PATH" \
    PYTHONPATH=/app

CMD ["rag-server", "--http", "--host", "0.0.0.0", "--port", "8765"]
