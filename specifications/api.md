# API Reference

## MCP Tools (JSON-RPC)

Сервер: `src.mcp_server.py`. Доступен через stdio и SSE.

### Поиск / Query

Параметры всех поисковых методов:
- `query` (str, обяз.) — поисковый запрос
- `k` (int, default 5) — количество результатов
- `max_chars` (int, default varies) — обрезка текста в результатах
- `metadata_filter` (dict, opt.) — фильтр по метаданным вида `{key: scalar | list}`
- `relations_load_depth` (int, default 1) — BFS глубина для `links`
- `relations_load_type_filter` (list[str], opt.) — типы рёбер для `links`
- `relations_load_meta_filter` (dict, opt.) — фильтр соседей для `links`

#### `rag_search`
Семантический поиск (dense vectors). Лучше всего подходит для концептуальных запросов. Возвращает `[{doc_id, text, score, metadata, links}]`.

#### `rag_bm25_search`
Keyword поиск (Qdrant sparse BM25). Для точных терминов/идентификаторов. Возвращает `[{doc_id, text, score, metadata, links}]`.

#### `rag_search_hybrid`
RRF alpha-dilution гибрид. Дополнительные параметры:
- `alpha` (float, default null) — баланс semantic(1.0)/BM25(0.0); null = language-aware
- `rerank` (bool, default null) — CrossEncoder reranking
- `query_expansion` (bool, default null) — multi-query expansion

### Чтение

#### `rag_get_document`
Полный текст документа с пагинацией.
- `doc_id` (str, обяз.)
- `offset` (int, default 0)
- `limit` (int, opt.) — символов от offset
- `relations_load_depth` (int, default 1)

#### `rag_list_documents`
Список документов с пагинацией и фильтром.
- `limit` (int, default 20)
- `offset` (int, default 0)
- `max_chars` (int, opt.)
- `metadata_filter` (dict, opt.)

### Индексация / Store

#### `rag_add_document`
- `text` (str, обяз.)
- `meta` (dict/null/JSON-string/string, opt.)
- `extract_graph` (bool, default false) — авто-извлечение графа через LLM
- Returns `{doc_id, duplicate}`

#### `rag_add_file`
- `filepath` (str, обяз.)
- `meta` (dict/null/JSON-string/string, opt.)
- `extract_graph` (bool, default false)

#### `rag_add_structured`
Repomix JSON индексация кода.
- `content` (str, обяз.) — JSON-строка в repomix формате
- `extract_graph` (bool, default false)
- Returns `{status, structure_doc_id, file_doc_ids, files_count, errors}`

#### `rag_add_relation`
- `source_id`, `target_id`, `relation` (str, обяз.)
- `weight` (float, default 1.0)

### Управление

#### `rag_update_document`
Обновление текста и/или метаданных документа. Сохраняет doc_id и все связи.
- `doc_id` (str, обяз.)
- `text` (str, opt.) — новый текст (content_hash пересчитывается, векторы реиндексируются)
- `meta` (dict/null/JSON-string/string, opt.) — новые метаданные (перезаписывает целиком)
- Returns `{doc_id, updated: true}`

#### `rag_delete_relation`
Удаление конкретного ребра графа. Идемпотентен.
- `source_id`, `target_id`, `relation` (str, обяз.)
- Returns `{status, deleted: bool}`

#### `rag_delete_document`
Каскадное удаление из всех сторов. Идемпотентен.
- `doc_id` (str, обяз.)
- Returns `{status, doc_id, deleted: bool}`

#### `rag_clear`
⚠️ Удалить ВСЕ данные.

#### `rag_stats`
Returns `{total_documents, store_path, dimension}`

#### `rag_graph_stats`
Returns `{total_nodes, total_edges, relation_types}`

#### `rag_get_related`
BFS обход графа от узла.
- `node_id` (str, обяз.)
- `max_depth` (int, default 1)
- `metadata_filter` (dict, opt.)
- Returns `{relations: [{source, target, relation, weight, direction}]}`

---

## HTTP REST API (FastAPI)

Сервер: `src/http_api.py`, порт 8765.

| Метод | Эндпоинт | Тело/Параметры | Описание |
|-------|----------|----------------|----------|
| GET | `/health` | — | Healthcheck |
| GET | `/stats` | — | Статистика |
| GET | `/graph/stats` | — | Статистика графа |
| POST | `/search` | `{query, k, mode, alpha, metadata_filter, max_chars, rerank, query_expansion, relations_load_*}` | Поиск (mode: semantic/bm25/hybrid) |
| POST | `/documents` | `{text, meta, extract_graph}` | Добавить документ |
| GET | `/documents` | `limit, offset, max_chars, metadata_filter` | Список документов |
| GET | `/documents/{id}` | `offset, limit, relations_load_*` | Документ по ID |
| DELETE | `/documents/{id}` | — | Удалить документ |
| PUT | `/documents/{id}` | `{text?, meta?}` | Обновить документ |
| POST | `/relations` | `{source_id, target_id, relation, weight}` | Добавить ребро |
| DELETE | `/relations` | `{source_id, target_id, relation}` | Удалить ребро |
| GET | `/relations/{id}` | `max_depth, metadata_filter` | BFS от узла |
| POST | `/clear` | — | Очистить всё |
| POST | `/reindex` | — | Пересчитать эмбеддинги |
| POST | `/structured` | `{content, extract_graph}` | Repomix индексация |
| GET | `/mcp` | — | MCP SSE handshake |
| POST | `/mcp` | JSON-RPC | MCP сообщение |

### MCP поверх HTTP

GET `/mcp` открывает SSE-соединение. POST `/mcp` отправляет JSON-RPC сообщения.
Используется MCP Python SDK `SseServerTransport`.

---

## Формат metadata_filter

```json
{"source": "specification"}
{"source": ["specification", "bug"]}
{"source": "specification", "type": "bug"}
```

AND-логика: все условия должны выполняться. Значение-список = `$in`.
