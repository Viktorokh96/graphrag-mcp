FROM python:3.12-slim

WORKDIR /app

COPY pyproject.toml .
COPY src/ src/

# Ставим с extra [postgres]: docker-compose использует Postgres backend
# (DATABASE_URL=postgresql://...), которому нужен psycopg.
RUN pip install --no-cache-dir -e ".[postgres]"

EXPOSE 8765

CMD ["rag-server", "--http", "--port", "8765"]
