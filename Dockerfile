FROM python:3.12-slim

RUN pip install graphrag

EXPOSE 8765

CMD ["rag-server", "--http", "--port", "8765"]
