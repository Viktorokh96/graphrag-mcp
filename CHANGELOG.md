# Changelog

## [0.3.0] — 2026-07-16

### Added
- **OpenAI-compatible провайдер эмбеддингов**: `EMBEDDING_MODEL=openai-compatible` — внешние серверы с `POST /v1/embeddings` (TEI, Infinity, vLLM, самописные). Поддержка батчей, кеширования, fallback.
- Конфигурация: `EMBEDDING_BASE_URL`, `EMBEDDING_API_KEY`, `EMBEDDING_MODEL_NAME`, fallback на `OPENAI_BASE_URL`/`OPENAI_API_KEY`/`OPENAI_MODEL`.

### Changed
- `src/embeddings.py`: новый класс `OpenAICompatibleEmbeddingGenerator`

## [0.2.0] — 2026-07-09

### Added
- **HTTP REST API** (Phase 3): FastAPI сервер с 17+ эндпоинтами, запуск через `rag-server --http`
- **MCP SSE транспорт** (Phase 3): Streamable HTTP для MCP клиентов
- **Reranker** (Phase 4): CrossEncoder `BAAI/bge-reranker-v2-m3`, lazy load, `rerank=true`
- **Query expansion** (Phase 8): Qwen3-1.8B парафраз через Ollama, `query_expansion=true`
- **Repomix-style индексация** (Phase 6): `rag_add_structured` для JSON-кода, авто-sibling связи
- **Авто-извлечение графа** (Phase 7): LLM-based (Qwen3-4B) + NER fallback (spaCy), `extract_graph=true`
- **Update/Delete операции**: `rag_update_document` (текст/мета с сохранением связей), `rag_delete_document` (каскадное), `rag_delete_relation` (точечное)
- **Community Detection**: `rag_find_communities` (Leiden, igraph+leidenalg, k-NN граф из эмбеддингов), `rag_set_community_names`, `rag_get_communities` с персистентным JSON-кешем
- **WebUI**: визуальный интерфейс (vis.js граф + документы + поиск) на `/webui/`
- **Graceful shutdown**: корректное закрытие Qdrant/SQLite при Ctrl+C
- **ChromaDB → Qdrant миграция**: `rag-server migrate [--dry-run] [--force]`
- **Offline-режим**: `HF_HUB_OFFLINE=true` для загрузки моделей без сети
- **Локальные модели**: `MODELS_DIR=./models`, `scripts/setup_models.sh`
- **Прелоад моделей**: `PRELOAD_MODELS=true` для горячего старта
- **Chunker**: семантическое разбиение длинных документов (>8192 токенов) интегрировано в `rag_add_document` и `rag_add_file`
- **Docker**: multi-stage uv build, .dockerignore
- **MIT License** (Phase 10)
- CLI флаги: `--http`, `--port`, `--rerank`, `--query-expansion`
- CLI команды: `update-document`, `delete-document`, `delete-relation`, `find-communities`, `set-community-names`, `community-get`, `community-docs`

### Changed
- **Переход на Qdrant + SQLite** (Phase 2): ChromaDB → Qdrant embedded/HTTP, SQLite/Postgres
- **BGE-M3** (Phase 1): мультиязычные эмбеддинги 1024d, lazy load
- **GraphStore**: SQLite+NetworkX, двунаправленный BFS (direction="both"), каскадное удаление
- **Hybrid search**: RRF alpha-dilution (K=20), language-aware alpha (кириллица → 0.85)
- **GraphExtractor**: переписан — LLM prompt для триплетов + spaCy NER fallback, без `_ensure_entity()` length-limit
- **MIN_CONTENT_LENGTH**: 40 символов (было 50)
- **RRF_K**: 20 (а не 60 как в первоначальном плане)
- `pyproject.toml`: +`ollama`, `igraph>=0.11.0`, `leidenalg>=0.12.0` в зависимостях

### Removed
- `requirements.txt`, `src/bm25_index.py`, `debug_bm25.py` — устаревшие компоненты

### Fixed
- D1 (рассинхрон сторов): `_sync_stores()` — DocumentStore как источник правды
- D2 (фантомные узлы графа): каскадное удаление, валидация при загрузке
- D3 (однонаправленный BFS): direction="both" по умолчанию
- D5 (короткие документы): `MIN_CONTENT_LENGTH=40` с ValueError
- D6 (дубликаты): content_hash + `is_duplicate()`
- D7 (тай-оффы hybrid): RRF вместо линейной комбинации
- D9 (стоп-слова BM25): Qdrant sparse vectors + стоп-слова EN/RU
- D10 (нормализация meta): всегда dict, никогда null
- Graceful shutdown: `rag.close()` в finally-блоке
- MatchAny+None → Qdrant error: фильтрация None из MatchAny
- remove_edge → NetworkXError: проверка наличия ребра перед удалением
- add_file + extract_graph: параметр extract_graph добавлен в add_file
- Query expansion / Graph extraction timeout: 120с через Ollama Client
- Unicode нормализация: SHA256 хэш нормализованного текста

## [0.1.0] — 2026-06

### Initial
- ChromaDB векторное хранилище
- BM25 keyword search (rank-bm25)
- MCP сервер (JSON-RPC stdio)
- Гибридный поиск (RRF + alpha-dilution)
- Графовая база знаний (in-memory)
- Визуализация графа (vis.js)
- Тесты (287 шт)
