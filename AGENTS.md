# AGENTS — graphrag

MCP-сервер графовой базы знаний с гибридным поиском (dense + sparse + граф).
Стек: BGE-M3, Qdrant, SQLite, FastAPI, CrossEncoder, NetworkX.

## Quick Start

```bash
cd ~/Work/graphrag

# Запуск MCP-сервера (stdio)
EMBEDDING_PROVIDER=bge-m3 python3 -m src.mcp_server

# HTTP API (FastAPI)
python3 -m src.http_api

# Тесты
python3 -m pytest tests/ -v

# CLI
python3 -m src.cli --help
python3 -m src.cli graph-viz -o rag_data/graph.html
```

## Структура

| Файл | Описание |
|------|----------|
| `src/mcp_server.py` | MCP-сервер (JSON-RPC stdio + SSE) |
| `src/http_api.py` | FastAPI HTTP REST API (порт 8765) |
| `src/rag.py` | RAGSystem — оркестратор поиска |
| `src/embeddings.py` | Эмбеддинги: BGE-M3 / Ollama / OpenRouter |
| `src/vector_store.py` | Qdrant (dense + sparse) + DocumentStore (SQLite) |
| `src/graph_store.py` | Граф: SQLite + NetworkX |
| `src/reranker.py` | CrossEncoder reranker (lazy load) |
| `src/query_expander.py` | Multi-query expansion (Ollama LLM) |
| `src/graph_extractor.py` | Авто-извлечение графа (LLM + NER) |
| `src/structured_indexer.py` | Repomix JSON → чанки + sibling связи |
| `src/config.py` | RAGConfig (из env) |
| `src/graph_viz.py` | Визуализация графа (vis.js) |
| `src/cli.py` | CLI (argparse) |
| `src/__init__.py` | init |
| `src/_meta_filter.py` | Фильтр метаданных |
| `scripts/benchmark_alpha.py` | Бенчмарк NDCG@k для default_alpha |
| `tests/` | pytest тесты |
| `specifications/` | api.md, architecture.md, cli.md |
| `rag_data/` | Хранилище (gitignored) |
| `.env.example` | Пример конфигурации env |
| `pytest.ini` | Конфиг pytest |
| `CHANGELOG.md` | История изменений |
| `Dockerfile`, `docker-compose.yml` | Docker |

## Провайдеры эмбеддингов

По умолчанию — **BGE-M3** (sentence-transformers, 1024d, lazy load).
Альтернативы — Ollama (4096d) или OpenRouter.

```bash
# BGE-M3 (по умолчанию)
export EMBEDDING_PROVIDER=bge-m3

# Ollama
export EMBEDDING_PROVIDER=ollama
export OLLAMA_BASE_URL=http://localhost:11434
export OLLAMA_MODEL=qwen3-embedding:8b

# OpenRouter
export EMBEDDING_PROVIDER=openrouter
export OPENROUTER_API_KEY=sk-or-v1-...
```

## MCP инструменты

Полный актуальный реестр — `src/mcp_server.py`, `TOOL_DEFS`.

### Поиск / Query

Все методы принимают: `query` (обяз.), `k=5`, `max_chars=null`, `metadata_filter`, `relations_load_depth=1`, `relations_load_type_filter`, `relations_load_meta_filter`. Возвращают `[{doc_id, text, score, metadata, links}]`.

**Новые параметры:**
- `rerank` (bool, default false) — CrossEncoder reranking
- `query_expansion` (bool, default false) — LLM multi-query expansion

| Инструмент | Особенности |
|-----------|------------|
| `rag_search` | Dense + sparse (Qdrant). `rerank`, `query_expansion` |
| `rag_bm25_search` | Разреженные векторы (Qdrant sparse) |
| `rag_search_hybrid` | RRF alpha-dilution. `alpha=null` → language-aware (кир. 0.85, иначе 0.5). `rerank`, `query_expansion` |

### Чтение / Retrieve

| Инструмент | Параметры | Описание |
|-----------|-----------|----------|
| `rag_get_document` | `doc_id`, `offset=0`, `limit=null`, `relations_load_depth=1`, ... | Полный текст с offset/limit пагинацией |

### Индексация / Store

| Инструмент | Параметры | Описание |
|-----------|-----------|----------|
| `rag_add_document` | `text`, `meta=null`, `extract_graph=false` | Добавить документ. `extract_graph=true` → LLM триплеты |
| `rag_add_file` | `filepath`, `meta=null`, `extract_graph=false` | Проиндексировать файл |
| `rag_add_structured` | `filepath`, `meta=null` | Repomix JSON: чанки кода + sibling связи |
| `rag_add_relation` | `source_id`, `target_id`, `relation`, `weight=1.0` | Ребро графа |

### Управление / Inspect

| Инструмент | Параметры | Описание |
|-----------|-----------|----------|
| `rag_list_documents` | `limit=20`, `offset=0`, `max_chars`, `metadata_filter`, relations params | Список документов |
| `rag_delete_document` | `doc_id` | Каскадное удаление (DocumentStore + Vector + Graph). Идемпотентен |
| `rag_clear` | — | ⚠️ Удалить ВСЕ данные |

### Граф / Статистика

| Инструмент | Параметры | Описание |
|-----------|-----------|----------|
| `rag_get_related` | `node_id`, `max_depth=1`, `metadata_filter` | BFS обход (out+in). `direction` в каждом ребре |
| `rag_graph_stats` | — | `{total_nodes, total_edges, relation_types}` |
| `rag_stats` | — | `{total_documents, store_path, dimension}` |

### Замечания по параметрам

- `meta` в `rag_add_document`/`rag_add_file`/`rag_add_structured`: dict/null/""/JSON-строка/строка → `_parse_meta()`
- `metadata_filter`: `dict[key, scalar|list]` — AND. VectorStore → Qdrant Filter, Graph → post-filter
- `max_chars` у всех поисков и `rag_list_documents`. Полный текст → `rag_get_document`
- **Relations inline (`links`)**: все возвращающие документы методы включают `links: {doc_id: [{relation, weight, direction}]}`. `relations_load_depth=0` → пустой `{}`
- Ошибки: неизвестный инструмент → `ValueError`, опциональные параметры → дефолты из схемы

### Гибридный поиск: RRF

```
RRF_K = 20
score = alpha/(K + rank_sem + 1) + (1-alpha)/(K + rank_bm25 + 1)  # оба канала
score = alpha/(K + rank_sem + 1)                                    # sem-only
score = (1-alpha)/(K + rank_bm25 + 1)                                # bm25-only
```

**Language-aware alpha:** `alpha=null` → кириллица → `cyrillic_alpha` (0.85), иначе `default_alpha` (0.5). Явный `alpha` приоритетен.

**Reranker:** `rerank=true` → CrossEncoder `BAAI/bge-reranker-v2-m3` (lazy load).

**Query expansion:** `query_expansion=true` → N парафразов Qwen3-1.8B → каждый search → RRF слияние.

**Candidate expansion:** `max(k*3, 20)` из каждого канала.

## Docker

```bash
docker compose up --build  # Qdrant + Postgres + RAG
# HTTP API: http://localhost:8765
# OpenAPI docs: http://localhost:8765/docs
```

## HTTP API (FastAPI)

```bash
python3 -m src.http_api
# GET  /health
# POST /search, /bm25_search, /hybrid_search
# POST /documents, /file, /structured
# GET  /documents, /document/{doc_id}
# DELETE /documents/{doc_id}
# POST /relations
# GET  /related/{node_id}
# GET  /stats, /graph_stats
# DELETE /clear
```

## Тестирование

```bash
python3 -m pytest tests/ -v
python3 -m pytest tests/test_mcp_server.py -v
python3 -m pytest tests/test_search_quality.py -v

# Новые тесты (Phase 4, 8)
python3 -m pytest tests/test_reranker.py -v
python3 -m pytest tests/test_query_expander.py -v
```

## Бенчмарк alpha

```bash
python3 -m scripts.benchmark_alpha
```

Сетка alpha ∈ [0.0, 1.0], шаг 0.05, NDCG@5 / P@5 / P@1.
Выбор: среди alpha в 1% от лучшего NDCG — ближайшее к 0.5.

## Линтинг

```bash
ruff check src/ tests/
```
