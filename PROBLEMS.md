# Problems

Date: 2026-07-10

---

## Strategic

### 1. Microsoft GraphRAG v3.1.0 вышел — догоняет по фичам

MS GraphRAG активно развивается (v3.1.0, май 2026). Добавили CosmosDB provider, parquet input, NLP streaming. MCP поддержка — вопрос времени. Наше единственное преимущество (MCP) тает.

### 2. LightRAG доминирует сообщество (37.5k stars)

Incremental update, multimodal, 5 storage backends — у них есть всё, чего нет у нас. Сообщество, контрибьюторы, PR. Мы проигрываем по охвату и скорости развития.

### 3. Ниша "русскоязычный MCP GraphRAG" слишком узкая

Language-aware alpha — единственная真正的差异化. Но объём рынка мал. Без выхода на английский/global рост ограничен.

---

## Product

### 4. WebUI есть, но сырой

- Граф грузит все 205 нод сразу, нет виртуализации/пагинации
- Нет поиска по графу (BFS/hop) — только text search
- Нет редактирования графа (drag-n-drop связи, удаление рёбер)
- Нет экспорта (PNG, SVG, JSON)
- Нет управления из WebUI (только sidebar-формы)

### 5. Incremental graph updates — нет

Любое добавление/удаление документа требует `loadGraph()` с нуля. LightRAG умеет diff-обновление графа. Без этого — проигрыш при динамических данных.

### 6. Чанкование не интегрировано

`Chunker` написан, но не вызывается. Документы >8192 токенов (BGE-M3 limit) тихо обрезаются. Хвост документа неиндексируем, непоиск.

### 7. Graph extraction сломана

- `_ensure_entity()` падает на любой сущности короче 50 символов (все реальные сущности)
- Только первые 4000 символов документа попадают в LLM
- NER загружает spaCy модель на каждый вызов (нет кеша)

### 8. Нет graceful shutdown

MCP-сервер при Ctrl+C не закрывает Qdrant (файловые locks) и SQLite (WAL). При перезапуске — locked-ошибки.

### 9. HTTP API без auth, CORS, rate limiting

FastAPI без middleware. Любой в локальной сети может читать/писать документы. Для production непригодно.

---

## Architecture

### 10. threading.Lock в async контексте

`Database._lock = threading.Lock()` блокирует event loop FastAPI. Весь сервер зависает при конкурентных запросах.

### 11. Неатомарные операции

`delete_document()`, `clear()` — удаляют из storages последовательно. При сбое на середине — несогласованное состояние (граф пуст, векторы целы, документы целы).

### 12. sync_stores OOM

Загружает ВСЕ doc_id из SQLite и Qdrant в память для сравнения. На 1M документов — гарантированный OOM.

### 13. Embedding cache без ограничения

Кеш эмбеддингов растёт бесконечно. Ollama (4096d): 3.2GB на 100K документов. Нет LRU, нет TTL, нет вытеснения.

### 14. OpenRouter/Ollama API без batch

100 документов → 100 HTTP-запросов. API поддерживает batch (`input` как массив), но реализация вызывает по одному.

### 15. Docker непригоден

- `Dockerfile`: `pip install graphrag` с PyPI, не локальный код
- `docker-compose.yml`: `rag-server` без аргументов → help → exit 1
- `:latest` теги — non-reproducible

---

## Unfixed Critical Bugs

### 16. MatchAny + None → Qdrant error

`to_qdrant_filter` пропускает `None` в `MatchAny.any`, Qdrant это не принимает.

### 17. remove_edge → NetworkXError

`GraphStore.remove_edge()` не проверяет наличие ребра в NetworkX перед удалением.

### 18. add_file не поддерживает extract_graph

В отличие от `add_document`, у `add_file` нет параметра `extract_graph`. Файл читается дважды.

### 19. Query expansion / Graph extraction без timeout

Ollama вызовы без таймаута. При недоступной Ollama — вечное зависание.

### 20. Unicode нормализация отсутствует

Один и тот же текст в NFC и NFD получает разные `doc_id`, дубликаты не находятся.

---

## Summary

| Category | Count | Key |
|----------|-------|-----|
| Strategic | 3 | MS догоняет, LightRAG лидирует, ниша узка |
| Product | 6 | WebUI сырой, нет incremental, чанкование, graph extraction сломана, нет graceful shutdown, HTTP голый |
| Architecture | 6 | threading.Lock, неатомарность, OOM, cache без LRU, batch, Docker |
| Bugs | 5 | MatchAny, NetworkXError, add_file, timeout, NFC |
