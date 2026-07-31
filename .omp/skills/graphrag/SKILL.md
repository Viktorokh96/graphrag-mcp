---
name: graphrag
description: MCP-сервер графовой базы знаний с гибридным поиском (dense + sparse + граф). Стек: BGE-M3/Qdrant/SQLite/FastAPI/CrossEncoder/NetworkX. RRF-фьюжн, Leiden-сообщества, multi-query expansion, reranker. uv для управления зависимостями.
license: MIT
metadata:
  workflow: project
---

# GraphRAG — гибридная графовая база знаний

MCP-сервер, объединяющий векторный поиск (BGE-M3/Qdrant), BM25 (sparse-вектора Qdrant), графовую БД (SQLite + NetworkX) и CrossEncoder-реранкер в единый RAG-пайплайн.

## Архитектура хранилища (двухуровневая)

```
Qdrant (embedded или HTTP)  ─── dense-эмбеддинги (1024d) + sparse BM25
SQLite (WAL) или Postgres   ─── документы (source of truth) + рёбра графа
NetworkX                    ─── in-memory кеш графа для BFS-обходов
```

- **Документ** — атомарная единица. При индексации попадает сразу во все три стора (dense, sparse, graph/doc). Чанкирование — на уровне Qdrant-индекса: документ разбивается на overlapping чанки для dense/sparse, но выдаётся и хранится целиком.
- **Граф** хранится в SQLite (таблица `relations`: source_id, target_id, relation, weight) и синхронно кешируется в NetworkX для O(1)-обхода.
- **Дубликаты** определяются по SHA256 хешу нормализованного контента; add-операции идемпотентны.

## Стек

| Компонент | Технология | Назначение |
|---|---|---|
| Эмбеддинги | BGE-M3 (sentence-transformers) | Dense-вектора, 1024d, lazy load |
| Векторный стор | Qdrant (embedded / HTTP) | Dense + sparse индексы |
| Документный стор | SQLite (WAL) / Postgres | Source of truth, метаданные |
| Графовый стор | SQLite + NetworkX | Рёбра + BFS-обходы |
| Ранкер | BAAI/bge-reranker-v2-m3 | CrossEncoder, lazy load |
| Чанкер | tiktoken + overlapping | Разбиение длинных доков |
| HTTP API | FastAPI (порт 8765) | REST + OpenAPI (/docs) |
| MCP | mcp (Python SDK) | stdio + SSE транспорт |
| WebUI | vis.js | Граф + документы (/webui/) |
| Сообщества | python-igraph (Leiden) | k-NN по эмбеддингам + рёбра графа |
| Зависимости | uv | Пакетный менеджер |

## Ключевые конвенции

### uv — ГЛОБАЛЬНО
```bash
uv run python3 -m pytest tests/ -v    # тесты
uv run ruff check src/ tests/          # линтер
uv run python3 -m src.cli --http       # HTTP API
uv add <package>                       # добавить зависимость
```
**Никогда** не вызывай `pip`, `pytest`, `python3` напрямую без `uv run`.

### Переменные окружения (ключевые)
```bash
EMBEDDING_MODEL=bge-m3|ollama|openrouter  # провайдер эмбеддингов
QDRANT_URL=http://localhost:6333          # внешний Qdrant (иначе embedded)
DATABASE_URL=postgresql://...             # Postgres (иначе SQLite)
MODELS_DIR=./models                       # локальные модели (scripts/setup_models.sh)
HF_HUB_OFFLINE=true                       # оффлайн-режим
PRELOAD_MODELS=true                       # прелоад BGE-M3 + CrossEncoder при старте
RAG_DEFAULT_ALPHA=0.5                     # alpha для RRF (не-кириллица)
RAG_CYRILLIC_ALPHA=0.85                   # alpha для RRF (кириллица)
```

## Гибридный поиск (RRF)

```
RRF_K = 20
score = alpha/(K + rank_sem + 1) + (1-alpha)/(K + rank_bm25 + 1)  # оба канала
score = alpha/(K + rank_sem + 1)                                    # sem-only
score = (1-alpha)/(K + rank_bm25 + 1)                                # bm25-only
```

- **Language-aware alpha**: кириллица → cyrillic_alpha (0.85), иначе default_alpha (0.5)
- **Candidate expansion**: каждый канал возвращает `max(k*3, 20)` кандидатов перед RRF-фьюжном
- **Reranker** (опционально): CrossEncoder переранжирует top-N результатов (× `rerank_top_k_multiplier`)
- **Query expansion** (опционально): Qwen3-1.8B генерирует N парафраз → каждый → search → RRF-слияние

## Структура файлов

| Файл | Роль |
|---|---|
| `src/mcp_server.py` | MCP-сервер (stdio + SSE), TOOL_DEFS |
| `src/rag.py` | RAGSystem — оркестратор: add, search, delete |
| `src/config.py` | RAGConfig dataclass (из env) |
| `src/embeddings.py` | BGE-M3 / Ollama / OpenRouter генераторы |
| `src/vector_store.py` | Qdrant: dense + sparse индексы |
| `src/graph_store.py` | GraphStore: SQLite-рёбра + NetworkX-кеш |
| `src/document_store.py` | DocumentStore: CRUD, source of truth |
| `src/reranker.py` | CrossEncoder reranker (lazy load) |
| `src/query_expander.py` | Multi-query expansion (Ollama LLM) |
| `src/graph_extractor.py` | Авто-извлечение графа через LLM + NER |
| `src/chunker.py` | tiktoken-чанкер (overlapping) |
| `src/communities.py` | Leiden-сообщества (k-NN эмбеддингов + граф) |
| `src/http_api.py` | FastAPI REST API |
| `src/cli.py` | CLI (argparse), console_script `rag-server` |
| `src/graph_viz.py` | Визуализация графа (vis.js) |
| `src/webui/index.html` | WebUI |
| `src/structured_indexer.py` | Repomix JSON → чанки + sibling-связи |
| `src/_meta_filter.py` | Фильтр метаданных (AND, $in) |
| `tests/` | pytest-тесты |
| `specifications/` | api.md, architecture.md, cli.md |

## MCP-инструменты

### Поиск (все возвращают `[{doc_id, text, score, metadata, links}]`)
- `rag_search` — dense (семантический). `rerank`, `query_expansion`
- `rag_bm25_search` — sparse BM25 (ключевые слова)
- `rag_search_hybrid` — RRF-фьюжн. **Рекомендуемый дефолт.** `alpha`, `rerank`, `query_expansion`

### Индексация
- `rag_add_document` — текст + meta + опциональный extract_graph
- `rag_add_file` — файл с диска
- `rag_add_structured` — Repomix JSON: чанки + sibling-связи
- `rag_add_relation` — ребро графа (source_id, target_id, relation, weight)

### Чтение / управление
- `rag_list_documents` — пагинация, фильтр, max_chars
- `rag_get_document` — полный текст с offset/limit + relations
- `rag_update_document` — обновить текст/метаданные (doc_id и связи сохраняются)
- `rag_delete_document` — каскадное удаление, идемпотентен
- `rag_delete_relation` — удалить конкретное ребро

### Граф
- `rag_get_related` — BFS-обход (out+in), max_depth, metadata_filter
- `rag_graph_stats` — узлы/рёбра/типы связей
- `rag_stats` — документы/размерность/путь

### Сообщества
- `rag_find_communities` — Leiden (resolution, k_nn)
- `rag_set_community_names` — имена извне
- `rag_get_communities` — кеш последнего find

### Общие параметры поиска
- `k` (default 5), `max_chars` (default 2000, null = полный текст)
- `metadata_filter`: dict key→scalar|list, AND. Qdrant → native Filter, Graph → post-filter
- `relations_load_depth` (default 1): 0 = без связей
- `rerank` (bool), `query_expansion` (bool)

## Типичные паттерны

### Запуск
```bash
# MCP-сервер
EMBEDDING_MODEL=bge-m3 uv run python3 -m src.mcp_server

# HTTP API + WebUI
uv run python3 -m src.cli --http --host 0.0.0.0 --port 8765

# Qdrant внешний (для concurrent-доступа)
docker compose up -d qdrant
export QDRANT_URL=http://localhost:6333
```

### Добавление документа с метаданными
```python
rag_add_document(text="...", meta={"source": "spec", "project": "graphrag"})
```

### Поиск с фильтром
```python
rag_search_hybrid(query="архитектура хранилища", metadata_filter={"source": "specification"}, max_chars=2000)
```

### Чтение связей документа
```python
rag_get_document(doc_id="...", offset=0, limit=3000, relations_load_depth=2)
```

### BFS-обход графа
```python
rag_get_related(node_id="...", max_depth=2, metadata_filter={"type": "module"})
```

### Ошибки и граничные случаи
- **Zero-vector query**: запрос из неизвестных модели слов (bare identifiers типа `pytest`) → dense возвращает пустой список. Используй BM25.
- **Документ < 40 символов**: `rag_add_document` rejected с ValueError.
- **Дубликат**: возвращает `{doc_id, duplicate: true}` — существующий doc_id.
- **Qdrant embedded не поддерживает concurrent-доступ**: несколько процессов → нужен внешний Qdrant (`QDRANT_URL`).
- **Кеш сообществ инвалидируется** при clear/delete/update документа.

## Тестирование и линтинг
```bash
uv run python3 -m pytest tests/ -v
uv run python3 -m pytest tests/test_mcp_server.py -v
uv run python3 -m pytest tests/test_search_quality.py -v
uv run ruff check src/ tests/
```
