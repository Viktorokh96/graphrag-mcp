# GraphRAG — Графовая база знаний с гибридным поиском

Семантический поиск + BM25 (разреженные векторы) + графовые реляции между документами.

Работает как **MCP сервер** (JSON-RPC через stdio/SSE) или **HTTP REST API** (FastAPI).

---

## Установка

```bash
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
python -m src.cli hybrid-search --query "Python" --query-expansion

# Repomix-style индексация кода
python -m src.cli add-structured --content "$(cat repomix-output.json)"

# Граф
python -m src.cli add-relation --source UUID1 --target UUID2 --relation "related_to"
python -m src.cli get-related --node UUID --max-depth 2
python -m src.cli graph-viz -o graph.html

# Список документов
python -m src.cli list-documents --limit 10

# Получить документ
python -m src.cli get-document --doc-id UUID

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
        "EMBEDDING_MODEL": "bge-m3"
      }
    }
  }
}
```

Или standalone: `python -m src.mcp_server` (stdio) / `python -m src.cli --http` (HTTP).

---

## Инструменты

### Поиск
| Инструмент | Описание |
|-----------|----------|
| `rag_search` | Семантический поиск (dense vectors) |
| `rag_bm25_search` | Разреженные векторы (Qdrant sparse) |
| `rag_search_hybrid` | RRF alpha-dilution + language-aware alpha, rerank, query expansion |

### Индексация
| Инструмент | Описание |
|-----------|----------|
| `rag_add_document` | Текст → DocumentStore + Vector + Graph. `extract_graph` |
| `rag_add_file` | Файл с диска. `extract_graph` |
| `rag_add_structured` | Repomix JSON → чанки + sibling связи |
| `rag_add_relation` | Ребро графа |

### Чтение и управление
| Инструмент | Описание |
|-----------|----------|
| `rag_get_document` | Полный текст с offset/limit пагинацией |
| `rag_list_documents` | Список с фильтром и relations inline |
| `rag_update_document` | Обновить текст/метаданные. doc_id и связи сохраняются |
| `rag_delete_document` | Каскадное удаление (каналы + граф) |
| `rag_delete_relation` | Удалить конкретное ребро графа |

### Граф
| Инструмент | Описание |
|-----------|----------|
| `rag_get_related` | BFS обход (out+in) с мета-фильтром |
| `rag_graph_stats` | Статистика графа |

---

## Фичи

- **BGE-M3** — мультиязычные эмбеддинги 1024d (lazy load, sentence-transformers)
- **Qdrant** — плотные + разреженные векторы (dual-векторы в одной коллекции)
- **SQLite / Postgres** — DocumentStore как source of truth, FK CASCADE для графа
- **CrossEncoder reranker** — `BAAI/bge-reranker-v2-m3` (lazy load, ~1GB)
- **Query expansion** — Qwen3-1.8B multi-query + RRF слияние
- **Auto graph extraction** — LLM (Qwen3-4B) + spaCy NER fallback
- **Repomix индексация** — чанки кода с авто-sibling связями
- **HTTP REST API** — FastAPI, 15+ эндпоинтов, `/docs` (OpenAPI)
- **MCP SSE** — Streamable HTTP транспорт
- **Docker** — Dockerfile + docker-compose.yml (Qdrant + Postgres)
- **Graph viz** — vis.js интерактивная визуализация графа

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
| Хранилище документов | `src/document_store.py` | SQLite (source of truth) |
| Оркестратор | `src/rag.py` | RRF + reranker + expansion + graph |
| MCP сервер | `src/mcp_server.py` | Python MCP SDK ≥1.28 |
| HTTP API | `src/http_api.py` | FastAPI |
| Reranker | `src/reranker.py` | CrossEncoder (lazy load) |
| Query expansion | `src/query_expander.py` | Ollama Qwen3-1.8B |
| Graph extraction | `src/graph_extractor.py` | LLM Qwen3-4B + spaCy NER |
| Структур. индексатор | `src/structured_indexer.py` | repomix JSON → чанки |
| Визуализация | `src/graph_viz.py` | vis.js + NetworkX |
| CLI | `src/cli.py` | argparse |

---

## Переменные окружения

| Переменная | Описание | По умолчанию |
|-----------|----------|-------------|
| `EMBEDDING_MODEL` | Провайдер: `bge-m3`, `ollama`, `openrouter` | `bge-m3` |
| `EMBEDDING_DIM` | Размерность эмбеддингов | `1024` |
| `EMBEDDING_DEVICE` | Устройство (`cpu`/`cuda`) | `cpu` |
| `OLLAMA_BASE_URL` | URL сервера Ollama | `http://localhost:11434` |
| `OLLAMA_MODEL` | Модель эмбеддингов Ollama | `qwen3-embedding:8b` |
| `OLLAMA_DIMENSION` | Размерность Ollama | `4096` |
| `OPENROUTER_API_KEY` | API ключ OpenRouter | — |
| `OPENROUTER_MODEL` | Модель OpenRouter | `openai/text-embedding-3-small` |
| `OPENROUTER_DIMENSION` | Размерность OpenRouter | `1536` |
| `QDRANT_URL` | URL Qdrant HTTP (опц., иначе embedded) | — |
| `DATABASE_URL` | Postgres DSN (опц., иначе SQLite) | — |
| `STORE_PATH` | Путь к хранилищу | `./rag_data` |
| `RAG_DEFAULT_ALPHA` | Баланс гибрида (0=BM25, 1=семантика) | `0.5` |
| `RAG_CYRILLIC_ALPHA` | Для кириллических запросов | `0.85` |
| `RAG_HYBRID_EXPAND` | Candidate expansion factor | `3` |
| `RAG_HYBRID_MIN_CANDIDATES` | Мин. кандидатов из каждого канала | `20` |
| `RERANK_ENABLED` | Включить reranker по умолч. | `false` |
| `RERANK_MODEL` | Модель reranker'а | `BAAI/bge-reranker-v2-m3` |
| `RERANK_DEVICE` | Устройство reranker'а | `cpu` |
| `QUERY_EXPANSION_ENABLED` | Включить expansion по умолч. | `false` |
| `QUERY_EXPANSION_MODEL` | LLM для expansion | `qwen3:1.8b` |
| `QUERY_EXPANSION_COUNT` | Число вариантов | `3` |
| `QUERY_EXPANSION_OLLAMA_URL` | URL для LLM expansion | `http://localhost:11434` |

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
# HTTP API на порту 8765, OpenAPI: http://localhost:8765/docs
```

---

## Структура

```
src/
├── __init__.py          # init
├── __main__.py          # python -m src
├── cli.py               # CLI (argparse, console_script rag-server)
├── config.py            # RAGConfig (env → dataclass)
├── document_store.py    # SQLite/Postgres (source of truth)
├── embeddings.py        # BGE-M3 / Ollama / OpenRouter
├── graph_extractor.py   # LLM + NER → триплеты
├── graph_store.py       # SQLite + NetworkX
├── graph_viz.py         # vis.js визуализация
├── http_api.py          # FastAPI сервер
├── mcp_server.py        # MCP (stdio + SSE)
├── rag.py               # Оркестратор
├── reranker.py          # CrossEncoder reranker
├── query_expander.py    # Multi-query expansion
├── structured_indexer.py  # Repomix JSON → чанки
├── vector_store.py      # Qdrant (dense + sparse)
├── _meta_filter.py      # Фильтр метаданных
├── migrate.py           # Миграция со старой ChromaDB
├── chunker.py           # Semantic splitting (резерв)
tests/
├── test_rag.py, test_mcp_server.py, ...  # pytest тесты
specifications/
├── api.md, architecture.md, cli.md
scripts/
├── benchmark_alpha.py   # Калибровка default_alpha
```

---

## История изменений

См. `CHANGELOG.md`.
