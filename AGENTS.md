# AGENTS — graphrag

MCP-сервер графовой базы знаний с гибридным поиском (dense + sparse + граф).
Стек: BGE-M3, Qdrant, SQLite, FastAPI, CrossEncoder, NetworkX.

## Quick Start

### 1. Поднять Qdrant (рекомендуется)

Embedded Qdrant не поддерживает параллельный доступ из нескольких процессов.
Для concurrent-доступа (несколько MCP-клиентов, CLI, HTTP API одновременно):

```bash
cd ~/Work/graphrag
docker compose up -d qdrant
export QDRANT_URL=http://localhost:6333
```

> Без `QDRANT_URL` используется embedded (./rag_data/qdrant) — только один процесс.

### 2. Запуск

```bash
# MCP-сервер (stdio)
EMBEDDING_MODEL=bge-m3 python3 -m src.mcp_server

# HTTP API (FastAPI)
python3 -m src.cli --http

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
| `src/vector_store.py` | Qdrant (dense + sparse) |
| `src/graph_store.py` | Граф: SQLite + NetworkX |
| `src/document_store.py` | SQLite / Postgres (source of truth) |
| `src/reranker.py` | CrossEncoder reranker (lazy load) |
| `src/query_expander.py` | Multi-query expansion (Ollama LLM) |
| `src/graph_extractor.py` | Авто-извлечение графа (LLM + NER) |
| `src/structured_indexer.py` | Repomix JSON → чанки + sibling связи |
| `src/config.py` | RAGConfig (из env) |
| `src/graph_viz.py` | Визуализация графа (vis.js) |
| `src/cli.py` | CLI (argparse, console_script `rag-server`) |
| `src/__init__.py` | init |
| `src/__main__.py` | `python -m src` entry point |
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
export EMBEDDING_MODEL=bge-m3

# Ollama
export EMBEDDING_MODEL=ollama
export OLLAMA_BASE_URL=http://localhost:11434
export OLLAMA_MODEL=qwen3-embedding:8b

# OpenRouter
export EMBEDDING_MODEL=openrouter
export OPENROUTER_API_KEY=sk-or-v1-...
```

### Offline-режим (без обращения к HuggingFace Hub)

Модели BGE-M3 и CrossEncoder скачиваются из HuggingFace Hub при первом запуске
и кешируются в `~/.cache/huggingface/hub/`. Если модель уже скачана — можно
отключить обращения к Hub:

```bash
export HF_HUB_OFFLINE=1
```

Или через `.env`:
```
HF_HUB_OFFLINE=true
```

Это передаёт `local_files_only=True` в SentenceTransformer — загрузка идёт
только из локального кеша, без сетевых запросов.

### Локальные модели (рекомендуется)

Скачайте модели скриптом — они хранятся в `./models/` и не зависят от HF Hub:

```bash
bash scripts/setup_models.sh   # ≈3 GB, один раз
```

Затем укажите в `.env`:
```
MODELS_DIR=./models
HF_HUB_OFFLINE=true
```

Код автоматически находит `./models/bge-m3/` и `./models/bge-reranker-v2-m3/`
и грузит их напрямую с диска.

### Прелоад моделей

По умолчанию модели грузятся лениво при первом запросе (~5-15 сек задержка).
Чтобы стартовал сразу с горячими моделями:

```
PRELOAD_MODELS=true
```

Старт MCP-сервера станет дольше на время загрузки моделей (~10-30 сек),
но первый запрос будет мгновенным.

## MCP инструменты

Полный актуальный реестр — `src/mcp_server.py`, `TOOL_DEFS`.

### Поиск / Query

Все методы принимают: `query` (обяз.), `k=5`, `max_chars=null`, `metadata_filter`, `relations_load_depth=1`, `relations_load_type_filter`, `relations_load_meta_filter`. Возвращают `[{doc_id, text, score, metadata, links}]`.

**Параметры:**
- `rerank` (bool, default false) — CrossEncoder reranking
- `query_expansion` (bool, default false) — LLM multi-query expansion

| Инструмент | Особенности |
|-----------|------------|
| `rag_search` | Dense (Qdrant). `rerank`, `query_expansion` |
| `rag_bm25_search` | Разреженные векторы (Qdrant sparse) |
| `rag_search_hybrid` | RRF alpha-dilution. Language-aware alpha. `rerank`, `query_expansion` |

### Индексация / Store

| Инструмент | Параметры | Описание |
|-----------|-----------|----------|
| `rag_add_document` | `text`, `meta=null`, `extract_graph=false` | Текст → три стора |
| `rag_add_file` | `filepath`, `meta=null`, `extract_graph=false` | Файл с диска |
| `rag_add_structured` | `content` (JSON строка), `meta=null`, `extract_graph=false` | Repomix JSON: чанки + sibling связи |
| `rag_add_relation` | `source_id`, `target_id`, `relation`, `weight=1.0` | Ребро графа |

### Управление / Inspect

| Инструмент | Параметры | Описание |
|-----------|-----------|----------|
| `rag_list_documents` | `limit=20`, `offset=0`, `max_chars`, `metadata_filter`, relations params | Пагинация |
| `rag_get_document` | `doc_id`, `offset=0`, `limit=null`, relations params | Полный текст |
| `rag_delete_document` | `doc_id` | Каскадное удаление. Идемпотентен |
| `rag_clear` | — | ⚠️ Удалить ВСЕ данные |

### Граф / Статистика

| Инструмент | Параметры | Описание |
|-----------|-----------|----------|
| `rag_get_related` | `node_id`, `max_depth=1`, `metadata_filter` | BFS обход (out+in) |
| `rag_graph_stats` | — | `{total_nodes, total_edges, relation_types}` |
| `rag_stats` | — | `{total_documents, store_path, dimension}` |

### Замечания по параметрам

- `meta`: dict/null/""/JSON-строка/строка → `_parse_meta()`
- `metadata_filter`: `dict[key, scalar|list]` — AND. VectorStore → Qdrant Filter, Graph → post-filter
- `max_chars` у поисков и `rag_list_documents`. Полный текст → `rag_get_document`
- **Relations inline (`links`)**: `{doc_id: [{relation, weight, direction}]}`. `relations_load_depth=0` → пустой `{}`
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

**Candidate expansion:** каждый канал возвращает `max(k*3, 20)` кандидатов перед фьюжном.

## Docker

```bash
# Только Qdrant (для локальной разработки — MCP/CLI используют QDRANT_URL)
docker compose up -d qdrant

# Полный стек: Qdrant + Postgres + RAG HTTP API
docker compose up --build
# HTTP API: http://localhost:8765
# OpenAPI docs: http://localhost:8765/docs
```

## HTTP API (FastAPI)

```bash
python3 -m src.cli --http
# GET  /health
# POST /search (mode: hybrid/bm25/semantic)
# POST /documents, /file, /structured
# GET  /documents, /document/{doc_id}
# DELETE /documents/{doc_id}
# POST /relations
# GET  /related/{node_id}, /graph-stats
# GET  /stats
# DELETE /clear
# GET  /graph-viz
# POST /reindex
```

## Тестирование

```bash
python3 -m pytest tests/ -v
python3 -m pytest tests/test_mcp_server.py -v
python3 -m pytest tests/test_search_quality.py -v
```

## Линтинг

```bash
ruff check src/ tests/
```
