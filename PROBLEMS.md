# Problems

Date: 2026-07-15

---

## Strategic

### 1. Microsoft GraphRAG v3.1.0 вышел — догоняет по фичам

MS GraphRAG активно развивается (v3.1.0, май 2026). Добавили CosmosDB provider, parquet input, NLP streaming. MCP поддержка — вопрос времени. Наше преимущество (MCP) сужается.

### 2. LightRAG доминирует сообщество (37.5k stars)

Incremental update, multimodal, 5 storage backends — у них больше фич и сообщество. Мы выигрываем за счёт MCP, но разрыв по охвату сохраняется.

### 3. Ниша "русскоязычный MCP GraphRAG" слишком узкая

Language-aware alpha — ключевая дифференциация. Но объём рынка мал. Без выхода на английский/global рост ограничен.

---

## Product

### 4. WebUI есть, но сырой

- Граф грузит все ноды сразу, нет виртуализации/пагинации
- Нет поиска по графу (BFS/hop) — только text search
- Нет редактирования графа (drag-n-drop связи, удаление рёбер)
- Нет экспорта (PNG, SVG, JSON)
- Нет управления из WebUI (только sidebar-формы)

### 5. Community reports (текстовые саммари) — нет

Есть Leiden-кластеризация (`rag_find_communities`), но нет LLM-генерации текстовых саммари по сообществам (как map-reduce в MS GraphRAG). Внешний LLM-оркестратор может назвать сообщества, но не генерирует отчёты.

### 6. HTTP API без auth, CORS, rate limiting

FastAPI без middleware. Любой в локальной сети может читать/писать документы. Для production непригодно.

---

## Architecture

### 7. threading.Lock в async контексте

`Database._lock = threading.Lock()` блокирует event loop FastAPI. Весь сервер зависает при конкурентных запросах в HTTP API.

### 8. Неатомарные операции

`delete_document()`, `clear()` — удаляют из storages последовательно. При сбое на середине — несогласованное состояние.

### 9. sync_stores OOM

Загружает ВСЕ doc_id из SQLite и Qdrant в память для сравнения. На 1M документов — гарантированный OOM.

### 10. Embedding cache без ограничения

Кеш эмбеддингов растёт бесконечно. Ollama (4096d): 3.2GB на 100K документов. Нет LRU, нет TTL, нет вытеснения.

### 11. OpenRouter/Ollama API без batch

100 документов → 100 HTTP-запросов. API поддерживает batch, но реализация вызывает по одному.

---

## Previously Solved (archived)

Ранее бывшие проблемами, теперь решены:

| Проблема | Решение | Версия |
|----------|---------|--------|
| ~~Incremental graph updates~~ | `rag_update_document`, `rag_delete_document`, `rag_delete_relation` | 0.2.0 |
| ~~Chunking не интегрирован~~ | Семантический чанкер в `rag_add_document`/`rag_add_file` (>8192 токенов) | 0.2.0 |
| ~~Graph extraction сломана~~ | Переписан: LLM prompt + spaCy NER, `_ensure_entity()` с `_skip_length_check` | 0.2.0 |
| ~~Graceful shutdown~~ | `rag.close()` в finally-блоке MCP-сервера | 0.2.0 |
| ~~Docker непригоден~~ | Multi-stage uv build, .dockerignore | 0.2.0 |
| ~~MatchAny + None → Qdrant error~~ | Фильтрация None из MatchAny в `to_qdrant_filter` | 0.2.0 |
| ~~remove_edge → NetworkXError~~ | Проверка наличия ребра перед удалением | 0.2.0 |
| ~~add_file без extract_graph~~ | `extract_graph` параметр добавлен в add_file | 0.2.0 |
| ~~Query expansion timeout~~ | 120с через Ollama `Client(timeout=120.0)` | 0.2.0 |
| ~~Unicode нормализация~~ | SHA256 хэш нормализованного текста | 0.2.0 |

---

## Summary

| Category | Open | Key |
|----------|------|-----|
| Strategic | 3 | MS догоняет, LightRAG лидирует, ниша узка |
| Product | 3 | WebUI сырой, нет community reports, HTTP голый |
| Architecture | 5 | threading.Lock, неатомарность, OOM, cache без LRU, batch |
| Solved | 10 | см. архив выше |
