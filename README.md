# GraphRAG — Графовая база знаний с гибридным поиском

Семантический поиск + BM25 (разреженные векторы) + графовые реляции между документами.

Работает как **MCP сервер** (JSON-RPC через stdio/SSE) или **HTTP REST API** (FastAPI).

---

## Установка

```bash
# Зависимости
uv sync   # или pip install -e .
```

**Зависимости:** Python ≥3.11, Qdrant (embedded), sentence-transformers (BGE-M3), FastAPI, NetworkX, SQLite.

---

## Быстрый старт (CLI)

```bash
# Добавить документ
python -m src.cli add-document --text "Python — мощный язык программирования"

# Семантический поиск
python -m src.cli search --query "язык программирования"

# Гибридный поиск (RRF alpha-dilution)
python -m src.cli hybrid-search --query "Python" --alpha 0.5

# С CrossEncoder reranking
python -m src.cli hybrid-search --query "Python" --rerank

# С LLM query expansion
python -m src.cli search --query "Python" --query-expansion

# Repomix-style индексация кода
python -m src.cli add-structured --path ./repomix-output.json

# Граф
python -m src.cli add-relation --source UUID1 --target UUID2 --relation "related_to"
python -m src.cli get-related --node UUID --max-depth 2
python -m src.cli graph-viz -o graph.html

# Статистика
python -m src.cli stats

# Запуск HTTP сервера
python -m src.cli --http --port 8765

# Очистить всё
python -m src.cli clear
```

---

## MCP сервер (Claude / Cline)

```json
{
  "mcpServers": {
    "rag-knowledge-base": {
      "command": "python",
      "args": ["-m", "src.mcp_server"],
      "env": {
        "EMBEDDING_PROVIDER": "bge-m3"
      }
    }
  }
}
```

Или standalone: `python -m src.mcp_server` (stdio) / `python -m src.http_api` (HTTP).

---

## Инструменты

### Поиск
| Инструмент | Описание |
|-----------|----------|
| `rag_search` | Dense + sparse hybrid (Qdrant). Параметры: `rerank`, `query_expansion` |
| `rag_bm25_search` | Разреженные векторы (Qdrant sparse) |
| `rag_search_hybrid` | RRF alpha-dilution + language-aware alpha |

### Индексация
| Инструмент | Описание |
|-----------|----------|
| `rag_add_document` | Текст → DocumentStore + Vector + Graph. Параметр: `extract_graph` |
| `rag_add_file` | Файл с диска |
| `rag_add_structured` | Repomix JSON → чанки + sibling связи |
| `rag_add_relation` | Ребро графа |

### Чтение и управление
| Инструмент | Описание |
|-----------|----------|
| `rag_get_document` | Полный текст с offset/limit пагинацией |
| `rag_list_documents` | Список с метаданными |
| `rag_delete_document` | Каскадное удаление (каналы + граф) |

### Граф
| Инструмент | Описание |
|-----------|----------|
| `rag_get_related` | BFS обход (out+in) с фильтром |
| `rag_graph_stats` | Статистика графа |

---

## Фичи

- **BGE-M3** — мультиязычные эмбеддинги 1024d (lazy load, sentence-transformers)
- **Qdrant** — плотные + разреженные векторы, on_disk, HNSW
- **SQLite** — DocumentStore как source of truth, синхронизация сторов
- **CrossEncoder reranker** — `BAAI/bge-reranker-v2-m3` (lazy load)
- **Query expansion** — Qwen3-1.8B multi-query + RRF слияние
- **Auto graph extraction** — LLM (Qwen3-4B) + spaCy NER fallback
- **Repomix индексация** — чанки кода с авто-sibling связями
- **HTTP REST API** — FastAPI, 15 эндпоинтов, `/docs` (OpenAPI)
- **MCP SSE** — Streamable HTTP транспорт
- **Docker** — Dockerfile + docker-compose.yml (Qdrant + Postgres)

---

## Архитектура

```
MCP Client (stdio)         HTTP Client (REST)
       │                         │
       ▼                         ▼
┌──────────────────────────────────────┐
│     MCPServer / HttpAPI             │
│  JSON-RPC stdio + SSE + FastAPI     │
└──────────┬───────────────────────────┘
           │
           ▼
┌──────────────────────────────────────┐
│          RAGSystem                  │
│  search / hybrid / rerank / expand  │
└────┬──────┬──────┬──────┬───────────┘
     │      │      │      │
     ▼      ▼      ▼      ▼
┌──────┐ ┌──────┐ ┌──────┐ ┌──────────┐
│Vector│ │Sparse│ │Graph │ │Document  │
│Store │ │      │ │Store │ │Store     │
│(Qdrt)│ │(Qdrt)│ │(Nx)  │ │(SQLite)  │
└──────┘ └──────┘ └──────┘ └──────────┘
```

| Компонент | Файл | Технология |
|-----------|------|-----------|
| Эмбеддинги | `src/embeddings.py` | BGE-M3 / Ollama / OpenRouter |
| Векторное хранилище | `src/vector_store.py` | Qdrant (dense + sparse) |
| Граф | `src/graph_store.py` | SQLite + NetworkX |
| Реестр документов | `src/vector_store.py` | SQLite (DocumentStore) |
| Оркестратор | `src/rag.py` | RRF + reranker + expansion + graph |
| MCP сервер | `src/mcp_server.py` | Python MCP SDK 1.28.1 |
| HTTP API | `src/http_api.py` | FastAPI |
| Reranker | `src/reranker.py` | CrossEncoder (lazy load) |
| Query expansion | `src/query_expander.py` | Ollama Qwen3 |
| Graph extraction | `src/graph_extractor.py` | LLM + spaCy NER |
| Структур. индексатор | `src/structured_indexer.py` | repomix JSON → чанки |
| Визуализация | `src/graph_viz.py` | vis.js + NetworkX |

---

## Переменные окружения

| Переменная | Описание | По умолчанию |
|-----------|----------|-------------|
| `EMBEDDING_PROVIDER` | Провайдер: `bge-m3`, `ollama`, `openrouter` | `bge-m3` |
| `OLLAMA_BASE_URL` | URL сервера Ollama | `http://localhost:11434` |
| `OLLAMA_MODEL` | Модель эмбеддингов | `qwen3-embedding:8b` |
| `OPENROUTER_API_KEY` | API ключ OpenRouter | — |
| `QDRANT_URL` | URL Qdrant HTTP (опционально) | — |
| `STORE_PATH` | Путь к хранилищу | `./rag_data` |
| `RAG_DEFAULT_ALPHA` | Баланс гибрида (0=BM25, 1=семантика) | `0.5` |
| `RAG_CYRILLIC_ALPHA` | Для кириллических запросов | `0.85` |
| `RERANK_ENABLED` | Включить reranker по умолч. | `false` |
| `QUERY_EXPANSION_ENABLED` | Включить expansion по умолч. | `false` |

---

## Тесты

```bash
python -m pytest tests/ -v
python -m pytest tests/test_mcp_server.py -v
python -m pytest tests/test_search_quality.py -v
```

## Docker

```bash
docker compose up --build
# HTTP API на порту 8765
```

---

## Структура

```
src/
├── cli.py              # CLI (argparse)
├── config.py           # RAGConfig (env)
├── embeddings.py       # BGE-M3 / Ollama / OpenRouter
├── graph_extractor.py  # LLM + NER → триплеты
├── graph_store.py      # SQLite + NetworkX
├── graph_viz.py        # vis.js визуализация
├── http_api.py         # FastAPI сервер
├── mcp_server.py       # MCP (stdio + SSE)
├── rag.py              # Оркестратор
├── reranker.py         # CrossEncoder reranker
├── query_expander.py   # Multi-query expansion
├── structured_indexer.py  # Repomix JSON
├── vector_store.py     # Qdrant + DocumentStore
├── _meta_filter.py     # Фильтр метаданных
tests/
├── test_rag.py, test_mcp_server.py, ...  # тесты
specifications/
├── api.md, architecture.md, cli.md
```

---

## История изменений

См. `CHANGELOG.md`.
