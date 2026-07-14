# rag_find_communities & rag_set_community_names

## Описание

Поиск семантических сообществ в графе знаний. Расстояние между нодами — cosine distance эмбеддингов. Результат — список сообществ с ID, размером и списком документов. Имена сообществ задаются извне (LLM-оркестратор), не внутри RAG.

## Архитектура

```
┌─────────────────────────────────────────────┐
│  External LLM (opencode orchestrator)       │
│                                             │
│  1. rag_find_communities()                  │
│  2. Читает документы каждого сообщества     │
│  3. Даёт название через set_community_names │
└─────────────────────────────────────────────┘
         │                        ▲
         ▼                        │
┌─────────────────────────────────────────────┐
│  graphrag RAG system                        │
│                                             │
│  find_communities:                          │
│    embeddings → kNN graph                   │
│    + existing relations → unified graph     │
│    → Leiden → communities                   │
│                                             │
│  set_community_names:                       │
│    {id: name} → in-memory cache             │
└─────────────────────────────────────────────┘
```

## Требования

### rag_find_communities

1. Строит k-NN граф из эмбеддингов документов (cosine similarity)
2. Объединяет с существующими рёбрами графа (max вес при конфликте)
3. Запускает Louvain community detection
4. Возвращает список сообществ с членами
5. Кеширует результат (для последующего set_community_names)

**Параметры:**
- `resolution` (float, default 1.0) — управление размером сообществ (больше → мельче)
- `k_nn` (int, default 15) — количество ближайших соседей для k-NN графа

**Возврат:**
```json
{
  "communities": [
    {
      "id": 0,
      "size": 12,
      "members": ["doc_id_1", "doc_id_2", ...]
    }
  ],
  "total_communities": 8,
  "total_nodes": 238
}
```

### rag_set_community_names

1. Принимает маппинг `{community_id: name}`
2. Сохраняет в in-memory кеш
3. Возвращает количество обновлённых

**Параметры:**
- `names` (dict, обяз.) — `{int: str}` маппинг

**Возврат:**
```json
{"status": "ok", "updated": 3}
```

### rag_get_communities

1. Возвращает кешированные сообщества с именами (из последнего find)
2. Если find ещё не вызывался — возвращает пустой список

**Возврат:**
```json
{
  "communities": [
    {
      "id": 0,
      "name": "Authentication Flow",
      "size": 12,
      "members": ["doc_id_1", ...]
    }
  ]
}
```

## Implementation Details

### Шаг 1: Получение эмбеддингов

```python
# Через публичный метод VectorStore (инкапсулирует scroll-логику)
doc_embeddings = vector_store.get_all_doc_embeddings()
# → dict[str, list[list[float]]] — doc_id → [chunk_vec_1, chunk_vec_2, ...]

# Усреднение multi-chunk
emb_matrix = np.array([np.mean(doc_embeddings[did], axis=0) for did in doc_ids])
```

### Шаг 2: k-NN граф (numpy)

```python
import numpy as np

# Нормализация для cosine
norms = np.linalg.norm(emb_matrix, axis=1, keepdims=True)
emb_norm = emb_matrix / (norms + 1e-10)

# Cosine similarity matrix
sim = emb_norm @ emb_norm.T

# k-NN рёбра
for i in range(n):
    top_k = np.argsort(sim[i])[-k_nn-1:-1][::-1]
    for j in top_k:
        if sim[i][j] > 0:
            G.add_edge(doc_ids[i], doc_ids[j], weight=float(sim[i][j]))
```

### Шаг 3: Объединение с графом

```python
# Существующие рёбра из graph_edges
for source, target, relation, weight in graph_kb.get_all_edges():
    if G.has_edge(source, target):
        old = G[source][target]['weight']
        G[source][target]['weight'] = max(old, weight)
    else:
        G.add_edge(source, target, weight=weight)
```

### Шаг 4: Leiden

```python
import igraph as ig
import leidenalg

# Convert NetworkX → igraph
nx_nodes = list(G.nodes())
nx_to_ig = {n: i for i, n in enumerate(nx_nodes)}
ig_graph = ig.Graph(n=len(nx_nodes), edges=[
    (nx_to_ig[u], nx_to_ig[v]) for u, v in G.edges()
])
ig_graph.vs["name"] = nx_nodes
# Transfer edge weights
for u, v, data in G.edges(data=True):
    ig_graph.es[ig_graph.get_eid(nx_to_ig[u], nx_to_ig[v])]["weight"] = data.get("weight", 1.0)

# Leiden clustering
part = leidenalg.find_partition(
    ig_graph,
    leidenalg.ModularityVertexPartition,
    weights="weight",
    resolution_parameter=resolution,
    seed=42,
)

# Convert back
communities = []
for idx, members_ig in enumerate(part):
    communities.append({
        "id": idx,
        "size": len(members_ig),
        "members": [nx_nodes[i] for i in members_ig],
    })
```

### Шаг 5: Хранение имён

```python
# В RAGSystem.__init__:
self._communities: list[dict] = []  # кеш от find_communities
self._community_names: dict[int, str] = {}  # имена

# set_community_names:
def set_community_names(self, names: dict[int, str]) -> dict:
    self._community_names.update(names)
    return {"status": "ok", "updated": len(names)}

# get_communities — возвращает с именами:
def get_communities(self) -> list[dict]:
    result = []
    for c in self._communities:
        result.append({
            "id": c["id"],
            "name": self._community_names.get(c["id"], ""),
            "size": c["size"],
            "members": c["members"],
        })
    return result
```

## MCP Tools

### `rag_find_communities`
```json
{
  "inputSchema": {
    "properties": {
      "resolution": {"type": "number", "default": 1.0},
      "k_nn": {"type": "integer", "default": 15}
    }
  }
}
```

### `rag_set_community_names`
```json
{
  "inputSchema": {
    "properties": {
      "names": {"type": "object", "description": "{community_id: name} mapping"}
    },
    "required": ["names"]
  }
}
```

### `rag_get_communities`
```json
{
  "inputSchema": {"type": "object", "properties": {}}
}
```

## HTTP API

| Метод | Эндпоинт | Тело | Описание |
|-------|----------|------|----------|
| POST | `/communities` | `{resolution?, k_nn?}` | Найти сообщества |
| PUT | `/communities/names` | `{names: {id: name}}` | Задать имена |
| GET | `/communities` | — | Получить с именами |

## Зависимости

- `numpy` — уже есть в pyproject.toml
- `igraph>=0.11.0` — графовая библиотека (явная зависимость)
- `leidenalg>=0.12.0` — Leiden community detection
- Qdrant scroll с векторами — уже используется

## Хранение сообществ

Сообщества персистятся через JSON-файл `{store_path}/community_cache.json`.

**Формат файла:**
```json
{
  "communities": [
    {"id": 0, "size": 12, "members": ["doc_id_1", ...]}
  ],
  "names": {
    "0": "Authentication Flow",
    "1": "Database Layer"
  }
}
```

**Жизненный цикл:**
- `__init__` → `_load_community_cache()` загружает с диска (если файл есть)
- `find_communities()` → результат сохраняется в `_communities` + `_save_community_cache()`
- `set_community_names()` → имена обновляются + `_save_community_cache()`
- `clear()` / `delete_document()` / `update_document()` → `_invalidate_community_cache()` удаляет файл

**Поток использования:**
```
1. find_communities()              → кеш + JSON файл
2. LLM читает docs каждого c       → понимает тематику
3. set_community_names({0: "X"})   → имена + JSON файл
4. Перезапуск MCP-сервера          → кеш восстанавливается из JSON
5. get_communities()               → [{id:0, name:"X", ...}]
```
