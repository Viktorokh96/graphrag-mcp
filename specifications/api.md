# API Specification — RAG MCP Tool

## 1. EmbeddingGenerator (`src/embeddings.py`)

### Класс: `EmbeddingGenerator` (фабрика / select_embedding)

```python
# Основной: BGE-M3 через sentence-transformers
class BGEEmbeddingGenerator:
    def __init__(self, model_name: str = "BAAI/bge-m3", device: str = "cpu")
    def get_embedding(self, text: str) -> list[float]
    def get_embeddings(self, texts: list[str]) -> list[list[float]]
    def get_dimension(self) -> int  # 1024

# Fallback: Ollama
class OllamaEmbeddingGenerator:
    def __init__(self, base_url: str = "http://localhost:11434", model: str = "qwen3-embedding:8b")
    def get_embedding(self, text: str) -> list[float]
    def get_embeddings(self, texts: list[str]) -> list[list[float]]
    def get_dimension(self) -> int  # 4096

# Альтернатива: OpenRouter
class OpenRouterEmbeddingGenerator:
    def __init__(self, api_key: str, model: str = "openai/text-embedding-3-small")
    def get_embedding(self, text: str) -> list[float]
    def get_embeddings(self, texts: list[str]) -> list[list[float]]
    def get_dimension(self) -> int
```

**Поведение:**
- Выбор провайдера по env `EMBEDDING_PROVIDER` (bge-m3 / ollama / openrouter)
- BGE-M3 lazy load (инициализируется при первом вызове)
- Кеширование результатов (dict text → embedding)

**Параметры env:**
```bash
EMBEDDING_PROVIDER=bge-m3
# Ollama альтернатива:
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=qwen3-embedding:8b
# OpenRouter:
OPENROUTER_API_KEY=sk-or-v1-...
OPENROUTER_MODEL=openai/text-embedding-3-small
```

---

## 2. VectorStore (`src/vector_store.py`)

### Класс: `VectorStore`

```python
class VectorStore:
    def __init__(self, store_path: str = "./rag_data", qdrant_url: str | None = None,
                 dimension: int = 1024, embedding_provider: str = "bge-m3")
    def add_document(self, doc_id: str, text: str, embedding: list[float],
                     metadata: dict | None = None)
    def add_documents(self, documents: dict[str, tuple[str, list[float]]],
                      metadata: dict[str, dict] | None = None)
    def search(self, query_embedding: list[float], k: int = 5,
               where: dict | None = None) -> list[tuple[str, str, float, dict]]
    def sparse_search(self, query: str, k: int = 5,
                      where: dict | None = None) -> list[tuple[str, str, float, dict]]
    def list_documents(self, limit: int = 20, offset: int = 0,
                       where: dict | None = None) -> tuple[list[tuple[str, str, dict]], int]
    def get_document(self, doc_id: str) -> tuple[str, str, dict] | None
    def clear(self)
    def stats(self) -> dict
    def remove(self, doc_id: str)
    def count(self) -> int
    def get_all_texts(self) -> list[str]
    def get_metadata(self, doc_id: str) -> dict | None
```

**Поведение:**
- Qdrant embedded (local) или HTTP (если указан `qdrant_url`)
- Две коллекции: `rag_docs` (dense vectors) + `rag_docs_sparse` (sparse vectors)
- Dense: Cosine distance, HNSW hnsw_config
- Sparse: Qdrant sparse vector (замена BM25)
- Нормализация скора: `score = clip(1 - distance²/2, 0, 1)`
- Фильтр `where` через Qdrant `Filter` (should/must)
- `_map_filter()`: преобразует `metadata_filter` в Qdrant условия
- `search()` берёт `max(k*3, 20)` кандидатов для гибридной fusion
- `stats()` → `{"total_documents": int, "store_path": str, "dimension": int}`
- `get_document()` → `(doc_id, text, metadata)` или `None`
- `list_documents()` → `([(doc_id, text, metadata)], total)`

---

## 3. DocumentStore (внутренний, `src/vector_store.py`)

### Класс: `DocumentStore`

```python
class DocumentStore:
    def __init__(self, store_path: str = "./rag_data")
    def add_document(self, doc_id: str, text: str, metadata: dict | None = None,
                     content_hash: str | None = None)
    def get_document(self, doc_id: str) -> tuple[str, str, dict, str] | None
    def get_all(self) -> list[tuple[str, str, dict]]
    def remove(self, doc_id: str)
    def count(self) -> int
    def find_by_hash(self, content_hash: str) -> str | None
    def clear(self)
```

**Поведение:**
- SQLite, таблицы `documents` (id, text, metadata_json, content_hash, created_at)
- Единый источник правды (`source of truth`) для `rag_list_documents` / `rag_get_document`
- При добавлении: проверка `content_hash` на дубликат
- При удалении: каскадное удаление из всех сторов (`remove_from_all`)
- `_sync_stores()` вызывается при старте: сверяет doc_id между Store/Vector/Graph

---

## 4. RAGSystem (`src/rag.py`)

### Класс: `RAGSystem`

```python
class RAGSystem:
    def __init__(self, store_path: str = "./rag_data",
                 config: RAGConfig | None = None)
    def add_document(self, text: str, metadata: dict | None = None,
                     extract_graph: bool = False) -> str
    def add_documents(self, texts: list[str], metadata: list[dict] | None = None) -> list[str]
    def add_file(self, filepath: str, metadata: dict | None = None) -> str
    def index_structured(self, filepath: str) -> list[str]
    def search(self, query: str, k: int = 5, metadata_filter: dict | None = None,
               rerank: bool = False, query_expansion: bool = False) -> list[dict]
    def bm25_search(self, query: str, k: int = 5,
                    metadata_filter: dict | None = None) -> list[dict]
    def search_hybrid(self, query: str, k: int = 5, alpha: float | None = None,
                      metadata_filter: dict | None = None, rerank: bool = False,
                      query_expansion: bool = False) -> list[dict]
    def get_document(self, doc_id: str) -> dict | None
    def list_documents(self, limit: int = 20, offset: int = 0,
                       metadata_filter: dict | None = None,
                       relations_load_depth: int = 1) -> dict
    def add_relation(self, source_id: str, target_id: str, relation: str,
                     weight: float = 1.0)
    def get_related(self, node_id: str, max_depth: int = 1,
                    metadata_filter: dict | None = None) -> dict
    def delete_document(self, doc_id: str) -> bool
    def clear(self)
    def stats(self) -> dict
    def graph_stats(self) -> dict
```

### Фильтрация по метаданным (`metadata_filter`)

Формат — `dict[str, scalar | list[scalar]]`:
- `key: scalar` → `metadata[key] == value`
- `key: list` → `metadata[key]` in list ($in)
- Все условия AND
- `None` или `{}` → фильтр отключён

Реализация:
- **VectorStore**: `_map_filter()` → Qdrant Filter (should/must)
- **GraphStore**: post-filter соседей в `get_related`

### Гибридный поиск (`search_hybrid`)

**Reciprocal Rank Fusion (RRF) с alpha-dilution:**
```
RRF_K = 20
score = alpha/(RRF_K + rank_sem + 1) + (1-alpha)/(RRF_K + rank_bm25 + 1)  # оба канала
score = alpha/(RRF_K + rank_sem + 1)                                       # sem-only
score = (1-alpha)/(RRF_K + rank_bm25 + 1)                                   # bm25-only
```

**Language-aware alpha:** `alpha=None`
- Кириллица → `cyrillic_alpha` (0.85)
- Остальные → `default_alpha` (0.5)
- Явный `alpha` имеет приоритет

**Candidate expansion:** `max(k * 3, 20)` из каждого канала.

**Reranker:** если `rerank=True`, после fusion все кандидаты переранжируются CrossEncoder.
**Query expansion:** если `query_expansion=True`, перед search генерируются N парафразов, каждый search независимо, затем RRF-слияние.

---

## 5. MCPServer (`src/mcp_server.py`)

### Инструменты (MCP методы)

#### Query
| Инструмент | Параметры | Описание |
|-----------|-----------|----------|
| `rag_search` | `query`, `k=5`, `max_chars=2000`, `metadata_filter`, `rerank=false`, `query_expansion=false`, relations params | Dense + sparse hybrid (Qdrant) |
| `rag_bm25_search` | `query`, `k=5`, `max_chars`, `metadata_filter`, relations params | Sparse vectors (Qdrant) |
| `rag_search_hybrid` | `query`, `k=5`, `alpha=null`, `max_chars`, `metadata_filter`, `rerank=false`, `query_expansion=false`, relations params | RRF alpha-dilution |

#### Retrieve
| Инструмент | Параметры | Описание |
|-----------|-----------|----------|
| `rag_get_document` | `doc_id`, `offset=0`, `limit=null`, relations params | Полный текст по ID |

#### Store
| Инструмент | Параметры | Описание |
|-----------|-----------|----------|
| `rag_add_document` | `text`, `meta=null`, `extract_graph=false` | Добавить документ |
| `rag_add_file` | `filepath`, `meta=null`, `extract_graph=false` | Проиндексировать файл |
| `rag_add_structured` | `filepath`, `meta=null` | Repomix JSON → чанки |
| `rag_add_relation` | `source_id`, `target_id`, `relation`, `weight=1.0` | Ребро графа |

#### Manage
| Инструмент | Параметры | Описание |
|-----------|-----------|----------|
| `rag_list_documents` | `limit=20`, `offset=0`, `max_chars`, `metadata_filter`, relations params | Список доков |
| `rag_delete_document` | `doc_id` | Удалить (каскадно) |
| `rag_clear` | — | Очистить всё |

#### Graph
| Инструмент | Параметры | Описание |
|-----------|-----------|----------|
| `rag_get_related` | `node_id`, `max_depth=1`, `metadata_filter` | BFS обход (out+in) |
| `rag_graph_stats` | — | Статистика графа |

#### Stats
| Инструмент | Параметры | Описание |
|-----------|-----------|----------|
| `rag_stats` | — | Статистика хранилища |

---

## 6. GraphStore (`src/graph_store.py`)

### Класс: `GraphStore`

```python
class GraphStore:
    def __init__(self, store_path: str = "./rag_data")
    def add_node(self, node_id: str, metadata: dict | None = None)
    def add_relation(self, source_id: str, target_id: str, relation: str,
                     weight: float = 1.0)
    def get_related(self, node_id: str, max_depth: int = 1,
                    metadata_filter: dict | None = None) -> list[tuple]
    def get_edges_batch(self, node_ids: list[str],
                        type_filter: list[str] | None = None,
                        meta_filter: dict | None = None,
                        depth: int = 1) -> dict[str, dict[str, list[dict]]]
    def remove_node(self, node_id: str)
    def stats(self) -> dict
    def clear(self)
```

**Поведение:**
- SQLite (3 таблицы: `nodes`, `edges`, `node_metadata`) + NetworkX in-memory
- `get_related()` двунаправленный BFS (out + in), возвращает `[(source, target, relation, weight, direction)]`
- `get_edges_batch()`: для списка node_id массово грузит рёбра (BFS на каждую)
- `remove_node()` удаляет узел и все инцидентные рёбра (CASCADE в SQLite)
- `stats()` → `{"total_nodes", "total_edges", "relation_types"}`

---

## 7. HttpAPI (`src/http_api.py`)

### FastAPI сервер (порт 8765, docs at /docs)

```python
app = FastAPI(title="RAG MCP HTTP API", version="0.2.0")
```

| Метод | Путь | Параметры | Описание |
|-------|------|-----------|----------|
| POST | `/search` | `query, k, max_chars, metadata_filter, rerank, query_expansion` | Гибридный поиск |
| POST | `/bm25_search` | `query, k, max_chars, metadata_filter` | Sparse vectors |
| POST | `/hybrid_search` | `query, k, alpha, max_chars, metadata_filter, rerank, query_expansion` | RRF гибрид |
| GET | `/document/{doc_id}` | `offset, limit` | Полный текст |
| POST | `/documents` | `text, meta, extract_graph` | Добавить документ |
| POST | `/file` | `filepath, meta, extract_graph` | Файл |
| POST | `/structured` | `filepath, meta` | Repomix |
| GET | `/documents` | `limit, offset` | Список |
| DELETE | `/documents/{doc_id}` | — | Удалить |
| POST | `/relations` | `source_id, target_id, relation, weight` | Ребро |
| GET | `/related/{node_id}` | `max_depth, metadata_filter` | Граф |
| GET | `/stats` | — | Статистика |
| GET | `/graph_stats` | — | Статистика графа |
| DELETE | `/clear` | — | Очистить всё |
| GET | `/health` | — | Health check |

---

## 8. Reranker (`src/reranker.py`)

```python
class Reranker:
    def __init__(self, model_name: str = "BAAI/bge-reranker-v2-m3",
                 device: str = "cpu")
    def rerank(self, query: str, documents: list[dict], top_k: int | None = None) -> list[dict]
```

**Поведение:**
- CrossEncoder, lazy load (инициализация при первом вызове)
- Принимает список `{doc_id, text, score, metadata}`, возвращает переранжированный список
- Model config: `RERANK_ENABLED`, `RERANK_MODEL`, `RERANK_DEVICE`, `RERANK_TOP_K_MULTIPLIER`
- Если CrossEncoder недоступен — fallback (возвращает как есть)

---

## 9. QueryExpander (`src/query_expander.py`)

```python
class QueryExpander:
    def __init__(self, model: str = "qwen3:1.8b",
                 ollama_base_url: str = "http://localhost:11434")
    def expand(self, query: str, n_queries: int = 3) -> list[str]
```

**Поведение:**
- Генерирует N парафразов исходного запроса через Ollama LLM
- Каждый парафраз → независимый поиск → RRF слияние
- Config: `QUERY_EXPANSION_ENABLED`, `QUERY_EXPANSION_MODEL`, `QUERY_EXPANSION_COUNT`

---

## 10. GraphExtractor (`src/graph_extractor.py`)

```python
class GraphExtractor:
    def __init__(self)
    def extract(self, text: str, doc_id: str) -> list[tuple[str, str, str, float]]
    def extract_graph(self, doc_id: str, text: str, graph_store: GraphStore)
```

**Поведение:**
- LLM (Qwen3-4B через Ollama) → триплеты `(source, relation, target)`
- spaCy NER fallback (если LLM не отвечает)
- `extract_graph()` добавляет узлы и рёбра в GraphStore

---

## 11. StructuredIndexer (`src/structured_indexer.py`)

### Класс: `StructuredIndexer`

```python
class StructuredIndexer:
    def __init__(self, rag: RAGSystem)
    def index(self, filepath: str, metadata: dict | None = None) -> list[str]
```

**Поведение:**
- Парсит repomix JSON: `{files: [{path, content, ...}]}`
- Каждый файл → чанки (по строкам, с контекстом)
- Авто-создание sibling связей между файлами в одной директории
- Возвращает список doc_id всех созданных чанков
