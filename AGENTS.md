# AGENTS — graphrag

MCP-сервер графовой базы знаний с гибридным поиском (семантический + BM25 + граф).

## Quick Start

```bash
cd ~/Work/graphrag

# Запуск MCP-сервера (stdio)
EMBEDDING_PROVIDER=ollama python3 -m src.mcp_server

# Тесты
python3 -m pytest tests/ -v

# CLI (если нужен)
python3 -m src.cli --help
```

## Структура

| Файл / Каталог | Описание |
|------|----------|
| `src/mcp_server.py` | MCP-сервер (JSON-RPC over stdio) |
| `src/rag.py` | RAGSystem — оркестратор поиска |
| `src/embeddings.py` | Эмбеддинги: Ollama (локально) / OpenRouter |
| `src/vector_store.py` | ChromaDB векторное хранилище |
| `src/bm25_index.py` | BM25 индекс ключевых слов |
| `src/graph_store.py` | Графовая база знаний (реляции) |
| `src/config.py` | RAGConfig (из env) |
| `src/cli.py` | CLI |
| `src/index.py` | Индексация |
| `tests/` | pytest тесты (197 шт) |
| `rag_data/` | Хранилище (gitignored) |
| `.env.example` | Пример конфигурации env |
| `pytest.ini` | Конфиг pytest |

## Провайдеры эмбеддингов

По умолчанию — **Ollama** (локально, `qwen3-embedding:8b`, 4096-мерные).
Альтернатива — OpenRouter (нужен API-ключ). См. `.env.example`.

```bash
# Ollama (по умолчанию)
export EMBEDDING_PROVIDER=ollama
export OLLAMA_BASE_URL=http://localhost:11434
export OLLAMA_MODEL=qwen3-embedding:8b

# OpenRouter
export EMBEDDING_PROVIDER=openrouter
export OPENROUTER_API_KEY=sk-or-v1-...
```

## MCP инструменты

`rag_add_document`, `rag_add_file`, `rag_search`, `rag_bm25_search`,
`rag_search_hybrid`, `rag_add_relation`, `rag_get_related`,
`rag_graph_stats`, `rag_stats`, `rag_clear`.

Поле `meta` в `rag_add_document` / `rag_add_file` опционально:
принимает dict, null, пустую строку, JSON-строку.

## Тестирование

```bash
# Полный набор
python3 -m pytest tests/ -v

# Только MCP-слой
python3 -m pytest tests/test_mcp_server.py -v
```

## Линтинг

```bash
ruff check src/ tests/
```
