# GraphRAG — Графовая база знаний с гибридным поиском

Семантический поиск + BM25 (разреженные векторы) + графовые реляции между документами.

Работает как **MCP сервер** (stdio + SSE) или **HTTP REST API** (FastAPI).

---

## Установка

```bash
uv sync
```

**Зависимости:** Python ≥3.12, Qdrant (embedded/HTTP), sentence-transformers (BGE-M3), FastAPI, NetworkX, SQLite, igraph, leidenalg.

---

## Быстрый старт (CLI)

```bash
# Добавить документ
rag-server add-document --text "Python — мощный язык программирования"

# Семантический поиск
rag-server search --query "язык программирования"

# Гибридный поиск (RRF alpha-dilution)
rag-server hybrid-search --query "Python" --alpha 0.5

# С CrossEncoder reranking
rag-server hybrid-search --query "Python" --rerank

# С LLM query expansion
rag-server hybrid-search --query "Python" --query-expansion

# Repomix-style индексация кода
rag-server add-structured --content "$(cat repomix-output.json)"

# Граф
rag-server add-relation --source UUID1 --target UUID2 --relation "related_to"
rag-server get-related --node UUID --max-depth 2
rag-server graph-viz -o graph.html

# Обновить документ
rag-server update-document --doc-id UUID --text "новый текст"

# Удалить документ / ребро
rag-server delete-document --doc-id UUID
rag-server delete-relation --source UUID1 --target UUID2 --relation "related_to"

# Сообщества
rag-server find-communities
rag-server set-community-names --names '{"0": "Authentication", "1": "Database"}'

# Список документов
rag-server list-documents --limit 10

# Получить документ
rag-server get-document --doc-id UUID

# Статистика
rag-server stats

# Запуск HTTP сервера (включает WebUI на /webui/)
rag-server --http --port 8765

# Очистить всё
rag-server clear
```

---

## MCP сервер (Claude / Cline)

```json
{
  "mcpServers": {
    "rag-knowledge-base": {
      "command": "uv",
      "args": ["run", "python3", "-m", "src.mcp_server"],
      "env": {
        "EMBEDDING_MODEL": "bge-m3"
      }
    }
  }
}
```

Или standalone: `uv run python3 -m src.mcp_server` (stdio) / `rag-server --http` (HTTP + WebUI).

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

### Сообщества
| Инструмент | Описание |
|-----------|----------|
| `rag_find_communities` | Leiden community detection по эмбеддингам + графу |
| `rag_set_community_names` | Задать имена сообществ (извне, LLM) |
| `rag_get_communities` | Получить сообщества с именами |

---

## Фичи

- **BGE-M3** — мультиязычные эмбеддинги 1024d (lazy load, sentence-transformers)
- **Qdrant** — плотные + разреженные векторы (dual-векторы в одной коллекции)
- **SQLite / Postgres** — DocumentStore как source of truth, FK CASCADE для графа
- **CrossEncoder reranker** — `BAAI/bge-reranker-v2-m3` (lazy load, ~1GB)
- **Query expansion** — Qwen3-1.8B multi-query + RRF слияние
- **Auto graph extraction** — LLM (Qwen3-4B) + spaCy NER fallback
- **Repomix индексация** — чанки кода с авто-sibling связями
- **Chunker** — семантическое разбиение длинных документов (>8192 токенов), интегрирован в add_document/add_file
- **Update / Delete** — `rag_update_document` (текст/мета, связи сохраняются), `rag_delete_document` (каскадное), `rag_delete_relation`
- **HTTP REST API** — FastAPI, 17+ эндпоинтов, `/docs` (OpenAPI)
- **WebUI** — визуальный интерфейс (vis.js граф + документы + поиск) на `/webui/`
- **MCP SSE** — Streamable HTTP транспорт
- **Docker** — multi-stage uv build + docker-compose.yml (Qdrant + Postgres)
- **Graph viz** — vis.js интерактивная визуализация графа
- **Leiden communities** — поиск семантических сообществ (k-NN граф эмбеддингов + Leiden)
- **Offline mode** — `HF_HUB_OFFLINE=true` для загрузки моделей без сети
- **Local models** — `MODELS_DIR=./models`, `scripts/setup_models.sh`
- **Graceful shutdown** — корректное закрытие Qdrant/SQLite при Ctrl+C
- **ChromaDB → Qdrant миграция** — `rag-server migrate [--dry-run] [--force]`

---

## Архитектура

```
MCP Client (stdio)         HTTP Client (REST)        WebUI (browser)
       │                         │                        │
       ▼                         ▼                        ▼
┌──────────────────────────────────────────────────────────────┐
│     MCPServer / HttpAPI + WebUI (/webui/)                   │
│  stdio + SSE + FastAPI + vis.js                             │
└──────────────────────────┬───────────────────────────────────┘
                           │
                           ▼
┌──────────────────────────────────────────────────────────────┐
│                      RAGSystem                              │
│  search / hybrid / rerank / expand / communities / update   │
└────┬──────┬──────┬──────┬──────┬────────────────────────────┘
     │      │      │      │      │
     ▼      ▼      ▼      ▼      ▼
┌──────┐ ┌──────┐ ┌──────┐ ┌──────────┐ ┌────────────┐
│Vector│ │Sparse│ │Graph │ │Document  │ │Community   │
│Store │ │      │ │Store │ │Store     │ │Cache       │
│(Qdrt)│ │(Qdrt)│ │(Nx)  │ │(SQLite)  │ │(JSON file) │
└──────┘ └──────┘ └──────┘ └──────────┘ └────────────┘
```

| Компонент | Файл | Технология |
|-----------|------|-----------|
| Эмбеддинги | `src/embeddings.py` | BGE-M3 / Ollama / OpenRouter |
| Векторное хранилище | `src/vector_store.py` | Qdrant (dense + sparse) |
| Граф | `src/graph_store.py` | SQLite + NetworkX |
| Хранилище документов | `src/document_store.py` | SQLite (source of truth) |
| Оркестратор | `src/rag.py` | RRF + reranker + expansion + graph |
| Сообщества | `src/rag.py` | Leiden (igraph) + k-NN эмбеддинги |
| MCP сервер | `src/mcp_server.py` | Python MCP SDK ≥1.28 |
| HTTP API | `src/http_api.py` | FastAPI |
| Reranker | `src/reranker.py` | CrossEncoder (lazy load) |
| Query expansion | `src/query_expander.py` | Ollama Qwen3-1.8B |
| Graph extraction | `src/graph_extractor.py` | LLM Qwen3-4B + spaCy NER |
| Структур. индексатор | `src/structured_indexer.py` | repomix JSON → чанки |
| Визуализация | `src/graph_viz.py` | vis.js + NetworkX |
| WebUI | `src/webui/index.html` | vis.js граф + документы |
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
| `OLLAMA_TIMEOUT` | Таймаут Ollama запросов (сек) | `120` |
| `EXTRACT_GRAPH` | Авто-извлечение графа по умолч. | `false` |
| `HF_HUB_OFFLINE` | Offline-режим (без HF Hub) | `false` |
| `MODELS_DIR` | Директория локальных моделей | — |
| `PRELOAD_MODELS` | Грузить модели при старте | `false` |

---

## Тесты

```bash
uv run python3 -m pytest tests/ -v
uv run python3 -m pytest tests/test_mcp_server.py -v
uv run python3 -m pytest tests/test_search_quality.py -v
```

## Линтинг

```bash
uv run ruff check src/ tests/
```

## Docker

```bash
# Только Qdrant (для локальной разработки)
docker compose up -d qdrant

# Полный стек: Qdrant + Postgres + RAG HTTP API
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
├── chunker.py           # Semantic splitting (интегрирован в add_document)
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
