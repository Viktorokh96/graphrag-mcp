# API Specification — RAG MCP Tool

## 1. EmbeddingGenerator (`src/embeddings.py`)

### Класс: `OpenRouterEmbeddingGenerator`

```python
class OpenRouterEmbeddingGenerator:
    def __init__(self, api_key: str | None = None, model: str = "openai/text-embedding-3-small")
    def get_embedding(self, text: str) -> list[float]
    def get_embeddings(self, texts: list[str]) -> list[list[float]]
    def get_dimension(self) -> int
```

**Параметры:**
- `api_key`: OpenRouter API ключ (из env `OPENROUTER_API_KEY` или аргумент)
- `model`: модель для эмбеддингов (по умолчанию `openai/text-embedding-3-small`)

**Поведение:**
- При вызове `get_embedding` делает HTTP POST запрос к `https://openrouter.ai/api/v1/embeddings`
- Кеширует результаты в памяти (dict text → embedding)
- При недоступности API использует простейший fallback (TF-IDF как заглушка)

### .env / конфигурация
```bash
OPENROUTER_API_KEY=sk-or-v1-...
OPENROUTER_MODEL=openai/text-embedding-3-small
```

---

## 2. BM25Index (`src/bm25_index.py`)

### Класс: `BM25Index`

```python
class BM25Index:
    def __init__(self)
    def add_document(self, doc_id: str, text: str, metadata: dict | None = None)
    def add_documents(self, documents: dict[str, str], metadata: dict[str, dict] | None = None)
    def search(self, query: str, k: int = 5) -> list[tuple[str, str, float, dict]]
    def clear(self)
    def stats(self) -> dict
    def remove(self, doc_id: str)
```

**Поведение:**
- Использует `rank_bm25` библиотеку (`BM25Okapi`)
- Хранит тексты и метаданные отдельно
- `search()` возвращает `[(doc_id, text, score, metadata), ...]` отсортированные по убыванию score
- `stats()` возвращает `{"total_documents": int, "total_tokens": int}`

---

## 3. VectorStore (`src/vector_store.py`)

### Класс: `VectorStore`

```python
class VectorStore:
    def __init__(self, store_path: str = "./rag_data")
    def add_document(self, doc_id: str, text: str, embedding: list[float], metadata: dict | None = None)
    def add_documents(self, documents: dict[str, tuple[str, list[float]]], metadata: dict[str, dict] | None = None)
    def search(self, query_embedding: list[float], k: int = 5) -> list[tuple[str, str, float, dict]]
    def clear(self)
    def stats(self) -> dict
    def remove(self, doc_id: str)
    def count(self) -> int
    def get_all_texts(self) -> list[str]
```

**Поведение:**
- Обёртка над ChromaDB (persistent клиент)
- Коллекция называется `rag_docs`
- `search()` использует ChromaDB `query()` с `n_results=k`, берёт `max(k*3, 20)` кандидатов для гибридной fusion
- Нормализация скора: `score = clip(1 - distance²/2, 0, 1)` — корректное
  преобразование L2-расстояния в косинусную сходность для L2-нормализованных
  эмбеддингов. Даёт широкий диапазон скоров [0, 1] вместо узкого кластера
  (старая формула `1/(1+distance)`).
- `stats()` возвращает `{"total_documents": int, "store_path": str, "dimension": int}`
- `get_all_texts()` возвращает все тексты для BM25 индексации

---

## 4. RAGSystem (`src/rag.py`)

### Класс: `RAGSystem`

```python
class RAGSystem:
    def __init__(self, store_path: str = "./rag_data", api_key: str | None = None,
                 config: RAGConfig | None = None)
    def add_document(self, text: str, metadata: dict | None = None) -> str
    def add_documents(self, texts: list[str], metadata: list[dict] | None = None) -> list[str]
    def add_file(self, filepath: str, metadata: dict | None = None) -> str
    def search(self, query: str, k: int = 5) -> list[tuple[str, str, float, dict]]
    def bm25_search(self, query: str, k: int = 5) -> list[tuple[str, str, float, dict]]
    def search_hybrid(self, query: str, k: int = 5, alpha: float | None = None) -> list[tuple[str, str, float, dict]]
    def clear(self)
    def stats(self) -> dict
```

**Семантический поиск (`search`):**
- Возвращает пустой список, если: хранилище пусто, размерность эмбеддинга не
  совпадает с хранилищем, или вектор запроса нулевой (все слова неизвестны
  эмбеддинг-модели — например, чистые идентификаторы типа `pytest`/`jwt`).
  В последнем случае запрос должен обслуживаться BM25-каналом.

**Гибридный поиск (`search_hybrid`):**

Использует **Reciprocal Rank Fusion (RRF)** — объединение через ранги, а не
линейную комбинацию скоров. RRF устойчив к разным шкалам (семантика в [0,1],
BM25 не ограничен), не требует нормализации, исключает тай-оффы.

```
RRF_K = 20  # не классическая 60 — для малых корпусов даёт широкий разброс скоров

# Документ из обоих каналов:
score = alpha / (RRF_K + rank_sem + 1) + (1 - alpha) / (RRF_K + rank_bm25 + 1)

# Только семантический канал:
score = alpha / (RRF_K + rank_sem + 1)

# Только BM25:
score = (1 - alpha) / (RRF_K + rank_bm25 + 1)
```

**Alpha-dilution:** single-channel документы взвешиваются долей своего канала
(alpha для sem-only, 1-alpha для bm25-only). Документы с score=0 (single-channel
при крайнем alpha) исключаются: `alpha=1.0` = чистая семантика (BM25-only
исключены), `alpha=0.0` = чистый BM25 (sem-only исключены). Документ из обоих
каналов всегда получает score > 0 при любом alpha > 0.

**Language-aware alpha:** при `alpha=None` (по умолчанию):
- Кириллические запросы (русский и др.) → `cyrillic_alpha` (env
  `RAG_CYRILLIC_ALPHA`, default **0.85**). BM25 без русского стемминга даёт
  шумовый сигнал, поэтому семантический канал доминирует.
- Остальные запросы → `default_alpha` (env `RAG_DEFAULT_ALPHA`, default **0.5** —
  точка естественного баланса, подтверждённая бенчмарком NDCG@k).
- Явно переданный `alpha` имеет приоритет.

- `alpha` за пределами [0, 1] клиппится к границам.
- **Candidate expansion:** из каждого канала забирается
  `max(k * hybrid_expand, hybrid_min_candidates)` кандидатов (по умолчанию
  `max(k*3, 20)`) перед fusion. Это спасает документы, релевантные по одному
  каналу, но оказавшиеся за пределами top-k по другому.

**При добавлении документа:**
1. Текст → эмбеддинг-генератор → эмбеддинг
2. Эмбеддинг → VectorStore (ChromaDB)
3. Текст → BM25Index
4. Узел → GraphKnowledgeBase

---

## 5. MCPServer (`src/mcp_server.py`)

### Класс: `MCPServer`

```python
class MCPServer:
    def __init__(self, rag: RAGSystem)
    def handle_request(self, request: dict) -> dict
    def run(self)
```

**JSON-RPC Формат запроса:**
```json
{
    "jsonrpc": "2.0",
    "id": 1,
    "method": "rag_add_document",
    "params": {
        "text": "some text",
        "meta": {"source": "user"}
    }
}
```

**Ответ:**
```json
{
    "jsonrpc": "2.0",
    "id": 1,
    "result": {"doc_id": "uuid", "status": "ok"}
}
```

**Инструменты (MCP методы):**

> Канонический реестр поддерживается в `AGENTS.md` (раздел «MCP инструменты»).
> При расхождении — источник истины `AGENTS.md`. Ниже — сводка.

Поиск (все принимают `query`, `k=5`, `max_chars=null`; возвращают
`[{doc_id, text, score, metadata}]` отсортированные по убыванию score):
1. `rag_search(query, k=5, max_chars=null)` — семантический поиск (эмбеддинги).
2. `rag_bm25_search(query, k=5, max_chars=null)` — BM25 keyword-поиск.
3. `rag_search_hybrid(query, k=5, alpha=null, max_chars=null)` — гибрид (RRF);
   `alpha=null` → language-aware: кириллица → `cyrillic_alpha` (0.85), иначе
   `default_alpha` (0.5). RRF_K=20, alpha-dilution.

Чтение:
4. `rag_get_document(doc_id, offset=0, limit=null)` → `{doc_id, text, metadata, total_chars, offset, limit}` или null.

Индексация:
5. `rag_add_document(text, meta=null)` → `{doc_id}`. `meta`: dict/null/""/JSON-строка/строка.
6. `rag_add_file(filepath, meta=null)` → `{doc_id}`. `meta` как у add_document.
7. `rag_add_relation(source_id, target_id, relation, weight=1.0)` → `{status: ok}`.

Управление:
8. `rag_list_documents(limit=20, offset=0, max_chars=null)` → `{documents, total, limit, offset}`.
9. `rag_delete_document(doc_id)` → `{status, doc_id, deleted}` (идемпотентен).
10. `rag_clear()` → `{status: ok}`. ⚠️ необратимо.

Граф:
11. `rag_get_related(node_id, max_depth=1)` → `{relations: [{source, target, relation, weight}]}`.
12. `rag_graph_stats()` → `{total_nodes, total_edges, relation_types}`.

Статистика:
13. `rag_stats()` → `{total_documents, store_path, dimension}`.

---

## 6. GraphKnowledgeBase (`src/graph_store.py`)

### Класс: `GraphKnowledgeBase`

```python
class GraphKnowledgeBase:
    def __init__(self, store_path: str = "./rag_data")
    def add_node(self, node_id: str, metadata: dict | None = None)
    def add_relation(self, source_id: str, target_id: str, relation: str, weight: float = 1.0)
    def get_related(self, node_id: str, max_depth: int = 1) -> list[tuple[str, str, str, float]]
    def stats(self) -> dict
    def clear(self)
```

**Поведение:**
- Хранит граф знаний в NetworkX
- `add_node()` добавляет узел с метаданными
- `add_relation()` добавляет направленное ребро с типом отношения и весом
- `get_related()` выполняет BFS до max_depth, возвращает `[(source, target, relation, weight), ...]`
- `stats()` возвращает `{"total_nodes": int, "total_edges": int, "relation_types": list[str]}`

**Пример использования:**
```python
from src.graph_store import GraphKnowledgeBase

graph = GraphKnowledgeBase(store_path="./rag_data")

# Добавляем узлы
graph.add_node("doc1", {"title": "Python basics"})
graph.add_node("doc2", {"title": "Python advanced"})

# Добавляем отношение
graph.add_relation("doc1", "doc2", "prerequisite", weight=1.0)

# Получаем связанные узлы
related = graph.get_related("doc1", max_depth=1)
# [("doc1", "doc2", "prerequisite", 1.0)]

# Статистика
stats = graph.stats()
# {"total_nodes": 2, "total_edges": 1, "relation_types": ["prerequisite"]}
```
