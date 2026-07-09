# Архитектура RAG MCP Tool

## Общая схема

```
MCP Client (stdio)         HTTP Client (REST)
       │                         │
       │ JSON-RPC (stdin/stdout) │ HTTP (localhost:8765)
       ▼                         ▼
┌──────────────────────────────────────────┐
│         MCPServer (mcp_server.py)        │
│  JSON-RPC stdio + SSE (Streamable HTTP)  │
│  HttpAPI (http_api.py) FastAPI           │
└──────────┬───────────────────────────────┘
           │
           ▼
┌──────────────────────────────────────────┐
│          RAGSystem (rag.py)             │
│  Оркестратор: search / hybrid / graph   │
│  Reranker, QueryExpansion, Links        │
└────┬──────┬──────┬──────┬───────────────┘
     │      │      │      │
     ▼      ▼      ▼      ▼
┌──────┐ ┌──────┐ ┌──────┐ ┌──────────────┐
│Vector│ │Sparse│ │Graph │ │DocumentStore │
│Store │ │Search│ │Store │ │(SQLite)      │
│(Qdrt)│ │(Qdrt)│ │(Nx)  │ │              │
└──────┘ └──────┘ └──────┘ └──────────────┘
```

## Компоненты

### 1. EmbeddingGenerator (`src/embeddings.py`)
- **BGE-M3** через sentence-transformers (lazy load, 1024d)
- Fallback: Ollama (qwen3-embedding:8b, 4096d)
- OpenRouter: альтернативный провайдер

### 2. VectorStore (`src/vector_store.py`)
- **Qdrant** (embedded или HTTP) — плотные и разреженные векторы
- Коллекция `rag_docs` с on_disk=True
- Cosine distance, HNSW индексы
- Metadata фильтр через Qdrant should/must

### 3. GraphStore (`src/graph_store.py`)
- **SQLite + NetworkX** (in-memory с персистентностью)
- Трёхтабличная схема: `nodes`, `edges`, `node_metadata`
- Двунаправленный BFS (out + in)
- Каскадное удаление узлов
- `get_edges_batch()` для массовой загрузки рёбер

### 4. DocumentStore (новое, в `src/vector_store.py`)
- **SQLite** как источник правды
- `documents` + `content_hash` таблицы
- `_sync_stores()` для согласования стора

### 5. RAGSystem (`src/rag.py`)
- Оркестратор: vector + sparse + graph + store
- Hybrid search: RRF alpha-dilution (RRF_K=20)
- Language-aware alpha: кириллица → 0.85, иначе 0.5
- **Reranker**: CrossEncoder bge-reranker-v2-m3, lazy load
- **Query Expansion**: Qwen3-1.8B multi-query + RRF
- **Auto Graph Extraction**: LLM + spaCy NER
- **Structured Indexer**: repomix JSON → чанки + sibling связи

### 6. MCPServer (`src/mcp_server.py`)
- JSON-RPC stdio + SSE (Streamable HTTP)
- 17 инструментов
- `_parse_meta()`: dict/null/""/JSON-строка/строка

### 7. HttpAPI (`src/http_api.py`)
- FastAPI, порт 8765
- 12 эндпоинтов, зеркалирующих MCP
- `/health` + `/docs` (OpenAPI)

## Технологии
| Компонент | Технология |
|-----------|-----------|
| Эмбеддинги | BGE-M3 (sentence-transformers), Ollama, OpenRouter |
| Векторная БД | Qdrant (embedded/HTTP) |
| Inverted index | Qdrant sparse vectors |
| Граф | SQLite + NetworkX |
| Реестр документов | SQLite |
| HTTP API | FastAPI |
| MCP | Python SDK 1.28.1 |
| Reranker | CrossEncoder |
| LLM | Ollama (Qwen3) — расширение запроса, граф-экстракция |
| NER | spaCy (fallback для графа) |
