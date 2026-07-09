# Архитектура graphrag

## Обзор

RAG-сервер с гибридным поиском (dense + sparse + граф). Двухуровневое хранение:
- **Qdrant** (embedded/HTTP) — dense-эмбеддинги + sparse BM25 векторы
- **SQLite** (WAL) или Postgres — документы (источник правды) + рёбра графа
- **NetworkX** — in-memory кеш графа для BFS

## Компоненты

| Компонент | Файл | Назначение |
|-----------|------|------------|
| `RAGSystem` | `src/rag.py` | Оркестратор: поиск, индексация, синхронизация сторов |
| `QdrantVectorStore` | `src/vector_store.py` | Векторное хранилище (dense + sparse BM25) |
| `DocumentStore` | `src/document_store.py` | SQL-хранилище документов (source of truth) |
| `GraphStore` | `src/graph_store.py` | Граф: SQLite + NetworkX кеш |
| `BgeM3EmbeddingGenerator` | `src/embeddings.py` | Локальные мультиязычные эмбеддинги (1024d) |
| `OllamaEmbeddingGenerator` | `src/embeddings.py` | Эмбеддинги через Ollama (4096d) |
| `OpenRouterEmbeddingGenerator` | `src/embeddings.py` | Эмбеддинги через OpenRouter API |
| `Reranker` | `src/reranker.py` | CrossEncoder BAAI/bge-reranker-v2-m3 (lazy load) |
| `QueryExpander` | `src/query_expander.py` | Multi-query expansion через Ollama LLM |
| `GraphExtractor` | `src/graph_extractor.py` | Авто-извлечение графа (LLM / spaCy NER) |
| `StructuredIndexer` | `src/structured_indexer.py` | Индексация repomix JSON |
| `Chunker` | `src/chunker.py` | Semantic splitting (не используется в production) |
| `MCP Server` | `src/mcp_server.py` | JSON-RPC stdio + SSE транспорт |
| `HTTP API` | `src/http_api.py` | FastAPI REST + MCP SSE |
| `RAGConfig` | `src/config.py` | Конфигурация из env |
| `CLI` | `src/cli.py` | Argparse CLI |

## Схема данных

### SQLite / Postgres

```
documents (doc_id PK, text, metadata JSON, content_hash, parent_doc_id, chunk_index, created_at)
graph_edges (source_id FK, target_id FK, relation PK, weight)
  └─ ON DELETE CASCADE → удаление документа удаляет рёбра
```

### Qdrant

Коллекция `rag_docs`, dual-векторы:
- `dense` — cosine distance, float[] размерности провайдера
- `bm25` — sparse vector (хэши токенов c TF-сатурацией)
- payload: `doc_id`, `metadata`, `text_preview:200`, `text_hashes[]`

## Эмбеддинги

### Провайдеры

| Провайдер | Env | Размерность | Модель по умолчанию |
|-----------|-----|-------------|---------------------|
| BGE-M3 (дефолт) | `EMBEDDING_MODEL=bge-m3` | 1024 | BAAI/bge-m3 |
| Ollama | `EMBEDDING_MODEL=ollama` | 4096 | qwen3-embedding:8b |
| OpenRouter | `EMBEDDING_MODEL=openrouter` | 1536 | openai/text-embedding-3-small |

Приоритет чтения: `EMBEDDING_MODEL` → `EMBEDDING_PROVIDER` (legacy) → `"bge-m3"`.

## Гибридный поиск (RRF)

```
RRF_K = 20

score(doc) = alpha / (RRF_K + rank_sem + 1)    — если doc найден семантикой
           + (1-alpha) / (RRF_K + rank_bm25 + 1) — если doc найден BM25
```

- **alpha = 1.0** — чистый semantic (BM25-only docs excluded)
- **alpha = 0.0** — чистый BM25 (sem-only docs excluded)
- **alpha = null** — language-aware: кириллица → `RAG_CYRILLIC_ALPHA` (0.85), иначе → `RAG_DEFAULT_ALPHA` (0.5)

Candidate expansion: `max(k * RAG_HYBRID_EXPAND, RAG_HYBRID_MIN_CANDIDATES)` из каждого канала (default: max(k*3, 20)).

### Reranker

CrossEncoder `BAAI/bge-reranker-v2-m3` (lazy load, ~1GB RAM). Включается:
- `--rerank` в CLI
- `rerank=true` в MCP/HTTP
- `RERANK_ENABLED=true` в env

### Query Expansion

Qwen3-1.8B через Ollama, генерирует N парафразов запроса. Каждый → search → RRF merge.

## Транспорты

| Транспорт | Протокол | Команда |
|-----------|----------|---------|
| MCP stdio | JSON-RPC через stdin/stdout | `python -m src.mcp_server` |
| HTTP REST | FastAPI на порту 8765 | `rag-server --http` |
| MCP SSE | Server-Sent Events через `/mcp` | часть HTTP сервера |

## Синхронизация сторов

`_sync_stores()` вызывается при старте RAGSystem:
1. `DocumentStore` — источник правды
2. Удаляет из Qdrant точки без документа (фантомы)
3. Переиндексирует документы без векторов

## Кэширование

- Эмбеддинги: in-memory dict в каждом генераторе (скидывается при clear_cache)
- Граф: NetworkX MultiDiGraph строится из SQLite при старте
- Reranker: Single-instance lazy load (model.ensure_model)
