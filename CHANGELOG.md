# Changelog

## [0.2.0] — 2026-07-09

### Added
- **HTTP REST API** (Phase 3): FastAPI сервер с 12 эндпоинтами, запуск через `rag-server --http`
- **MCP SSE транспорт** (Phase 3): Streamable HTTP для MCP клиентов
- **Reranker** (Phase 4): CrossEncoder `BAAI/bge-reranker-v2-m3`, lazy load, `rerank=true`
- **Query expansion** (Phase 8): Qwen3-1.8B парафраз через Ollama, `query_expansion=true`
- **Repomix-style индексация** (Phase 6): `rag_add_structured` для JSON-кода, авто-sibling связи
- **Авто-извлечение графа** (Phase 7): LLM-based (Qwen3-4B) и NER fallback (spaCy), `extract_graph=true`
- **Docker**: Dockerfile (python:3.12-slim) + docker-compose.yml (Qdrant + Postgres)
- **MIT License** (Phase 10)
- CLI флаги: `--http`, `--port`, `--rerank`, `--query-expansion`

### Changed
- **Переход на Qdrant + SQLite** (Phase 2): ChromaDB → Qdrant embedded/HTTP, SQLite/Postgres
- **BGE-M3** (Phase 1): мультиязычные эмбеддинги 1024d, lazy load
- **GraphStore**: SQLite+NetworkX, двунаправленный BFS, каскадное удаление
- **Hybrid search**: переработан с RRF alpha-dilution, `_hybrid_search_single` + `_merge_multi_query`
- `pyproject.toml`: +`ollama` в зависимостях

### Removed
- `requirements.txt`, `src/bm25_index.py`, `debug_bm25.py` — устаревшие компоненты

### Fixed
- D1 (рассинхрон сторов): `_sync_stores()` — DocumentStore как источник правды
- D2 (фантомные узлы графа): каскадное удаление, валидация при загрузке
- D3 (однонаправленный BFS): direction="both" по умолчанию
- D5 (короткие документы): `MIN_CONTENT_LENGTH=50` с ValueError
- D6 (дубликаты): content_hash + `is_duplicate()`
- D7 (тай-оффы hybrid): RRF вместо линейной комбинации
- D9 (стоп-слова BM25): Qdrant sparse vectors (без external токенизации)
- D10 (нормализация meta): всегда dict, никогда null

## [0.1.0] — 2026-06

### Initial
- ChromaDB векторное хранилище
- BM25 keyword search (rank-bm25)
- MCP сервер (JSON-RPC stdio)
- Гибридный поиск (RRF + alpha-dilution)
- Графовая база знаний (in-memory)
- Визуализация графа (vis.js)
- Тесты (287 шт)
