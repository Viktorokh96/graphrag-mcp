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

Полный актуальный реестр инструментов MCP-сервера (`src/mcp_server.py`, `TOOL_DEFS`).
Это канонический справочник интерфейса — при изменении `mcp_server.py` обновляйте и этот раздел.

### Поиск / Query

Все три поиска принимают `query` (обязательный), `k` (число результатов, по умолчанию 5) и `max_chars` (обрезать текст каждого результата до N символов; `null`/опущен = полный текст). Возвращают список `{doc_id, text, score, metadata}`, отсортированный по убыванию score.

| Инструмент | Параметры | Описание |
|-----------|-----------|----------|
| `rag_search` | `query`, `k=5`, `max_chars=null` | Семантический поиск через векторные эмбеддинги. Лучше для концептуальных запросов. |
| `rag_bm25_search` | `query`, `k=5`, `max_chars=null` | Ключевой поиск по алгоритму BM25 (Okapi). Лучше для точного совпадения терминов. |
| `rag_search_hybrid` | `query`, `k=5`, `alpha=0.5`, `max_chars=null` | Гибрид: `normalized_score = alpha*semantic + (1-alpha)*bm25`. `alpha=1.0` — чистая семантика, `alpha=0.0` — чистый BM25, `alpha=0.5` — баланс (рекомендуется по умолчанию). |

### Чтение / Retrieve

| Инструмент | Параметры | Описание |
|-----------|-----------|----------|
| `rag_get_document` | `doc_id` (обязательный), `offset=0`, `limit=null` | Получить один документ по ID. `offset` — смещение в символах (дефолт 0), `limit` — макс. число возвращаемых символов (`null`/опущен = весь текст начиная с offset). Используется для постраничного чтения больших документов вместо повторного поиска. |

### Индексация / Store

| Инструмент | Параметры | Описание |
|-----------|-----------|----------|
| `rag_add_document` | `text` (обязательный), `meta=null` | Добавить текстовый документ. Возвращает `{doc_id, status}`. `meta` опционален: dict / null / "" / JSON-строка / произвольная строка (оборачивается в `{"_raw": ...}`). |
| `rag_add_file` | `filepath` (обязательный), `meta=null` | Прочитать файл с диска и проиндексировать. Нормализация `meta` как у `rag_add_document`. |
| `rag_add_relation` | `source_id`, `target_id`, `relation` (все обязательные), `weight=1.0` | Создать направленное ребро в графе между двумя документами. `relation` — произвольная строка (`related_to`, `similar_to`, `prerequisite`, ...). |

### Управление / Inspect

| Инструмент | Параметры | Описание |
|-----------|-----------|----------|
| `rag_list_documents` | `limit=20`, `offset=0`, `max_chars=null` | Постраничный список документов. `max_chars` обрезает текст каждого документа. |
| `rag_delete_document` | `doc_id` (обязательный) | Удалить документ из всех хранилищ (векторное, BM25, граф). **Идемпотентен** — безопасно повторять. Возвращает `{status, doc_id, deleted}`. |
| `rag_clear` | — | ⚠️ Удалить ВСЕ данные (необратимо). Использовать с крайней осторожностью. |

### Графовый обход / Traversal

| Инструмент | Параметры | Описание |
|-----------|-----------|----------|
| `rag_get_related` | `node_id` (обязательный), `max_depth=1` | BFS-обход от узла. `max_depth` — сколько рёбер пройти (1 = прямые соседи). Возвращает `{relations: [{source, target, relation, weight}]}`. |

### Статистика

| Инструмент | Параметры | Описание |
|-----------|-----------|----------|
| `rag_stats` | — | `{total_documents, store_path, dimension}`. |
| `rag_graph_stats` | — | `{total_nodes, total_edges, relation_types}`. |

### Замечания по параметрам

- `meta` в `rag_add_document` / `rag_add_file` опционален и устойчив к типам: принимает dict, null, пустую строку, JSON-строку, любую другую строку. См. `_parse_meta()` в `src/mcp_server.py`.
- `max_chars` есть у всех поисков, `rag_list_documents` и (как `limit`) у `rag_get_document`. Передавайте конечное значение (например 1500–3000) на поисках, чтобы не переполнять контекст; полный текст забирайте через `rag_get_document` по `doc_id`.
- `k` (число результатов) есть у всех поисков, по умолчанию 5.
- Ошибок валидации нет — неизвестный инструмент бросает `ValueError`, отсутствующие опциональные параметры берут дефолты из схемы.

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
