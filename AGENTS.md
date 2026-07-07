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

# Визуализация графа
python3 -m src.cli graph-viz -o rag_data/graph.html
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
| `src/graph_viz.py` | Визуализация графа (vis.js) |
| `src/cli.py` | CLI |
| `src/index.py` | Индексация |
| `tests/` | pytest тесты (287 шт) |
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

Все три поиска принимают `query` (обязательный), `k` (число результатов, по умолчанию 5), `max_chars` (обрезать текст каждого результата до N символов; `null`/опущен = полный текст), `metadata_filter` (опциональный фильтр по метаданным — см. «Замечания по параметрам» ниже), а также опциональные параметры загрузки реляций (`relations_load_depth`, `relations_load_type_filter`, `relations_load_meta_filter`). Возвращают список `{doc_id, text, score, metadata, links}`, отсортированный по убыванию score.

| Инструмент | Параметры | Описание |
|-----------|-----------|----------|
| `rag_search` | `query`, `k=5`, `max_chars=null`, `metadata_filter=null`, `relations_load_depth=0`, `relations_load_type_filter=null`, `relations_load_meta_filter=null` | Семантический поиск через векторные эмбеддинги. Лучше для концептуальных запросов. Фильтр использует нативный `where` ChromaDB. |
| `rag_bm25_search` | `query`, `k=5`, `max_chars=null`, `metadata_filter=null`, `relations_load_depth=0`, `relations_load_type_filter=null`, `relations_load_meta_filter=null` | Ключевой поиск по алгоритму BM25 (Okapi). Лучше для точного совпадения терминов. Фильтр — post-filter результатов. |
| `rag_search_hybrid` | `query`, `k=5`, `alpha=null`, `max_chars=null`, `metadata_filter=null`, `relations_load_depth=0`, `relations_load_type_filter=null`, `relations_load_meta_filter=null` | Гибрид через **Reciprocal Rank Fusion (RRF)**. Формула (**alpha-dilution**): каждый канал взвешивается своей долей — `score = alpha/(RRF_K+rank_sem+1) + (1-alpha)/(RRF_K+rank_bm25+1)` для документов из ОБОИХ каналов; sem-only → `alpha/(RRF_K+rank_sem+1)`; bm25-only → `(1-alpha)/(RRF_K+rank_bm25+1)`. Документы с score=0 (single-channel при крайнем alpha) исключаются — поэтому `alpha=1.0` = чистая семантика, `alpha=0.0` = чистый BM25. `RRF_K=20` (не классическая 60) — даёт широкий разброс скоров для малых корпусов. **Language-aware alpha:** `alpha=null` для запросов с кириллицей → `cyrillic_alpha` (env `RAG_CYRILLIC_ALPHA`, default **0.85** — BM25 без русского стемминга шумит, поэтому семантика доминирует); для остальных → `default_alpha` (env `RAG_DEFAULT_ALPHA`, default **0.5** — выбран бенчмарком NDCG@k, см. `scripts/benchmark_alpha.py`). Явно переданный `alpha` имеет приоритет. **Candidate expansion:** из каждого канала забирается `max(k*3, 20)` кандидатов перед fusion. Фильтр применяется к обоим каналам до fusion. |

### Чтение / Retrieve

| Инструмент | Параметры | Описание |
|-----------|-----------|----------|
| `rag_get_document` | `doc_id` (обязательный), `offset=0`, `limit=null`, `relations_load_depth=0`, `relations_load_type_filter=null`, `relations_load_meta_filter=null` | Получить один документ по ID. `offset` — смещение в символах (дефолт 0), `limit` — макс. число возвращаемых символов (`null`/опущен = весь текст начиная с offset). Используется для постраничного чтения больших документов вместо повторного поиска. |

### Индексация / Store

| Инструмент | Параметры | Описание |
|-----------|-----------|----------|
| `rag_add_document` | `text` (обязательный), `meta=null` | Добавить текстовый документ. Возвращает `{doc_id, status}`. `meta` опционален: dict / null / "" / JSON-строка / произвольная строка (оборачивается в `{"_raw": ...}`). |
| `rag_add_file` | `filepath` (обязательный), `meta=null` | Прочитать файл с диска и проиндексировать. Нормализация `meta` как у `rag_add_document`. |
| `rag_add_relation` | `source_id`, `target_id`, `relation` (все обязательные), `weight=1.0` | Создать направленное ребро в графе между двумя документами. `relation` — произвольная строка (`related_to`, `similar_to`, `prerequisite`, ...). |

### Управление / Inspect

| Инструмент | Параметры | Описание |
|-----------|-----------|----------|
| `rag_list_documents` | `limit=20`, `offset=0`, `max_chars=null`, `metadata_filter=null`, `relations_load_depth=0`, `relations_load_type_filter=null`, `relations_load_meta_filter=null` | Постраничный список документов. `max_chars` обрезает текст каждого документа. `metadata_filter` — опциональный фильтр (см. «Замечания по параметрам»); при активном фильтре `total` отражает число подходящих документов. |
| `rag_delete_document` | `doc_id` (обязательный) | Удалить документ из всех хранилищ (векторное, BM25, граф). **Идемпотентен** — безопасно повторять. Возвращает `{status, doc_id, deleted}`. |
| `rag_clear` | — | ⚠️ Удалить ВСЕ данные (необратимо). Использовать с крайней осторожностью. |

### Графовый обход / Traversal

| Инструмент | Параметры | Описание |
|-----------|-----------|----------|
| `rag_get_related` | `node_id` (обязательный), `max_depth=1`, `metadata_filter=null` | BFS-обход от узла в ОБА направления (out + in). `max_depth` — сколько рёбер пройти (1 = прямые соседи). `metadata_filter` применяется к соседним (neighbor) узлам — рёбра к узлам, не проходящим фильтр, исключаются. Возвращает `{relations: [{source, target, relation, weight, direction}]}`. `direction: "out"` — исходящее ребро (source→target), `"in"` — входящее (target←source). |

### Статистика

| Инструмент | Параметры | Описание |
|-----------|-----------|----------|
| `rag_stats` | — | `{total_documents, store_path, dimension}`. |
| `rag_graph_stats` | — | `{total_nodes, total_edges, relation_types}`. |

### Замечания по параметрам

- `meta` в `rag_add_document` / `rag_add_file` опционален и устойчив к типам: принимает dict, null, пустую строку, JSON-строку, любую другую строку. См. `_parse_meta()` в `src/mcp_server.py`.
- `max_chars` есть у всех поисков, `rag_list_documents` и (как `limit`) у `rag_get_document`. Передавайте конечное значение (например 1500–3000) на поисках, чтобы не переполнять контекст; полный текст забирайте через `rag_get_document` по `doc_id`.
- `k` (число результатов) есть у всех поисков, по умолчанию 5.
- `metadata_filter` (опциональный, `null` по умолчанию) есть у `rag_search`, `rag_bm25_search`, `rag_search_hybrid`, `rag_list_documents` и `rag_get_related`. Формат — `dict[str, scalar | list[scalar]]`: каждая пара `key:value` — условие, что `metadata[key] == value`; если `value` — список, то условие `metadata[key]` входит в список (семантика `$in`). Все условия объединяются через **AND**. `null` или `{}` — фильтр отключён. Реализация: для VectorStore используется нативный `where` ChromaDB (`to_chroma_where()`); для BM25 и графа — post-filter (`matches_metadata_filter()`). В `rag_list_documents` при активном фильтре `total` отражает число подходящих документов. В `rag_get_related` фильтр применяется к соседним узлам (neighbor), рёбра к узлам, не проходящим фильтр, исключаются.
- **Relations inline (`links` field):** все методы, возвращающие документы (`rag_search*`, `rag_get_document`, `rag_list_documents`), включают поле `links` — словарь, где ключи — doc_id связанных узлов, значения — список объектов `{relation, weight, direction}`. Параметры управления:
  - `relations_load_depth` (int, default 0) — глубина BFS-обхода графа. 0 = не загружать связи (поле `links` = пустой `{}`).
  - `relations_load_type_filter` (list[str] \| null, default null) — список типов связей для фильтрации (null = все типы).
  - `relations_load_meta_filter` (dict \| null, default null) — фильтр по метаданным соседних узлов. Тот же формат, что и `metadata_filter`.
- Ошибок валидации нет — неизвестный инструмент бросает `ValueError`, отсутствующие опциональные параметры берут дефолты из схемы.

### Гибридный поиск: детали RRF

`rag_search_hybrid` использует **Reciprocal Rank Fusion (RRF)** — не линейную комбинацию скоров, а объединение через ранги. RRF устойчив к разным шкалам скоров (семантика в [0,1], BM25 — не ограничен), не требует нормализации и гарантирует отсутствие тай-оффов (ранги всегда различны).

**Формула (alpha-dilution):**

```
RRF_K = 20  # не классическая 60 — для малых корпусов даёт широкий разброс

# Документ из обоих каналов:
score = alpha / (RRF_K + rank_sem + 1) + (1 - alpha) / (RRF_K + rank_bm25 + 1)

# Только семантический канал:
score = alpha / (RRF_K + rank_sem + 1)

# Только BM25:
score = (1 - alpha) / (RRF_K + rank_bm25 + 1)
```

Документы с `score = 0` (single-channel при крайнем alpha: sem-only при `alpha=0.0`, bm25-only при `alpha=1.0`) исключаются из выдачи. Поэтому `alpha=1.0` = чистая семантика, `alpha=0.0` = чистый BM25 — без «призрачных» результатов.

**Language-aware alpha:** при `alpha=null` (по умолчанию) система выбирает alpha в зависимости от языка запроса:
- **Кириллические запросы** (русский и др.) → `cyrillic_alpha` (env `RAG_CYRILLIC_ALPHA`, default **0.85**). BM25 без русского стемминга даёт шумовый сигнал для русских запросов, поэтому семантический канал должен доминировать.
- **Остальные запросы** → `default_alpha` (env `RAG_DEFAULT_ALPHA`, default **0.5** — точка естественного баланса каналов, подтверждённая бенчмарком NDCG@k).
- Явно переданный `alpha` имеет приоритет над language-aware выбором.

## Тестирование

```bash
# Полный набор
python3 -m pytest tests/ -v

# Только MCP-слой
python3 -m pytest tests/test_mcp_server.py -v

# Только качество поиска (NDCG, релевантность, alpha)
python3 -m pytest tests/test_search_quality.py -v
```

## Бенчмарк выбора alpha

```bash
# Калибровка default_alpha по NDCG@k на детерминированном корпусе
python3 -m scripts.benchmark_alpha
```

Скрипт прогоняет `search_hybrid` по сетке alpha ∈ [0.0, 1.0] (шаг 0.05) на
корпусе с настоящей семантической структурой (`tests/semantic_mock.py`), считает
NDCG@5 / P@5 / P@1 и печатает таблицу. Среди alpha в пределах 1% от лучшего NDCG
(«хорошая область») берётся значение, ближайшее к 0.5 — точке естественного
баланса каналов. Это робастный и детерминированный выбор (в отличие от медианы
области, чья ширина колеблется из-за tie-breaking в ChromaDB). Результат должен
совпадать с `RAGConfig.default_alpha`; при расхождении — обновить конфиг.

## TODO — Relations Inline (links field)

### Что нужно сделать

Добавить во все методы, возвращающие документы (`rag_search*`, `rag_get_document`, `rag_list_documents`), поле `links` с реляциями (графовыми связями) прямо в ответе, чтобы не делать отдельный запрос `rag_get_related`.

### Формат

```python
# Каждый документ в выдаче получает поле links (пустой dict при depth=0):
{
  "doc_id": "abc-123",
  "text": "...",
  "score": 0.95,
  "metadata": {},
  "links": {
    "def-456": [
      {"relation": "related_to", "weight": 1.0, "direction": "out", "depth": 1}
    ]
  }
}
```

### Новые параметры (опциональные, все tools где есть document output)

| Параметр | Тип | Default | Описание |
|----------|-----|---------|----------|
| `relations_load_depth` | int | 0 | 0 = не загружать реляции. 1+ = BFS-глубина обхода графа |
| `relations_load_type_filter` | list[str] \| null | null | Фильтр по типу связи (null = все типы) |
| `relations_load_meta_filter` | dict \| null | null | Фильтр по метаданным соседних узлов |

### Где менять

1. **`src/graph_store.py`** — `get_edges_batch()`: массовое получение рёбер для списка node_id
2. **`src/rag.py`** — `_enrich_with_links()` + новые параметры в search/get_document/list_documents
3. **`src/mcp_server.py`** — TOOL_DEFS (schemas) + handlers (передача новых параметров + вызов _enrich_with_links)
4. **`src/cli.py`** — аргументы для новых параметров
5. **`tests/`** — тесты

**Status:** ✅ Done

## Линтинг

```bash
ruff check src/ tests/
```
