FROM python:3.13-slim AS builder

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

COPY pyproject.toml uv.lock ./
COPY AGENTS.md LICENSE ./
RUN uv sync --frozen --no-dev --no-editable --extra postgres

# Runtime stage
FROM python:3.13-slim

WORKDIR /app
COPY --from=builder /app/.venv /app/.venv
COPY src/webui/ src/webui/
ENV VIRTUAL_ENV=/app/.venv \
    PATH="/app/.venv/bin:$PATH"
EXPOSE 8765

CMD ["rag-server", "--http", "--port", "8765"]
