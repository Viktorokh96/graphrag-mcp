# PLAN: graphrag → production-grade RAG MCP

> **Статус:** Утверждён. Переход к реализации по фазам.
>
> **Цель:** Из локального MCP-сервера с ChromaDB сделать production-ready RAG-систему с
> мультиязычными эмбеддингами, Qdrant + SQLite/Postgres, HTTP API, reranker'ом,
> чанкованием, авто-извлечением графа и деплоем в одну команду.

---

## Фаза 0: Инфраструктура — uv

**Что меняем:**
- `requirements.txt` → удалить
- `pyproject.toml` — полная конфигурация проекта (зависимости, entry point, метаданные)
- `uv.lock` — lock-файл (сгенерируется при `uv sync`)
- `pytest.ini` → можно в `pyproject.toml` секцию `[tool.pytest.ini_options]`

**Команды:**
```bash
uv sync                          # установка всех зависимостей
uv run pytest tests/             # запуск тестов
uv run rag-server                # запуск
uv add sentence-transformers     # добавить зависимость
```

**Entry point:**
```toml
[project.scripts]
rag-server = "src.cli:main"
```

---

## Фаза 1: Эмбеддинги — BGE-M3

**Модель:** `BAAI/bge-m3` через `sentence-transformers`

**Dimension:** 1024

**Новый провайдер:** `BgeM3EmbeddingGenerator` в `src/embeddings.py`

```python
class BgeM3EmbeddingGenerator:
    def __init__(self, device: str = "cpu"):
        self.model = SentenceTransformer("BAAI/bge-m3", device=device)
        self.dim = 1024

    def embed(self, texts: list[str]) -> list[list[float]]:
        return self.model.encode(texts, normalize_embeddings=True).tolist()
```

**Config (`src/config.py`):**
```python
EMBEDDING_MODEL: str = "bge-m3"          # bge-m3 | ollama | openrouter
EMBEDDING_DIM: int = 1024
EMBEDDING_DEVICE: str = "cpu"
```

- Ollama и OpenRouter остаются как альтернативные провайдеры
- Загрузка модели: lazy (при первом вызове `embed()`), ~2GB RAM
- Параметр `EMBEDDING_MODEL` переключает провайдера

**Миграция:** старые эмбеддинги (4096d) несовместимы с BGE-M3 (1024d). При смене модели
нужно переиндексировать все документы. Детекция dimension mismatch — как сейчас у ChromaDB,
только для Qdrant.

---

## Фаза 2: Хранилище — Qdrant + SQLite/Postgres

### Архитектура (двухуровневая)

| Компонент | Default (dev) | Production |
|-----------|---------------|------------|
| Vector | **Qdrant embedded** (`path=./rag_data/qdrant`) | Qdrant HTTP (`QDRANT_URL`) |
| Documents | **SQLite** (`./rag_data/store.db`, WAL) | Postgres (`DATABASE_URL`) |
| Graph | SQLite (edges table) + NetworkX кеш | Postgres (edges table) + NetworkX кеш |

**Dev:** `pip install graphrag && rag-server` — 0 внешних зависимостей.
**Production:** `QDRANT_URL=http://... DATABASE_URL=postgresql://... rag-server`

### Postgres/SQLite схемы

```sql
CREATE TABLE documents (
    doc_id TEXT PRIMARY KEY,
    text TEXT NOT NULL,
    metadata TEXT DEFAULT '{}',
    content_hash TEXT,
    parent_doc_id TEXT,
    chunk_index INTEGER,
    created_at TEXT DEFAULT (datetime('now'))
);

CREATE TABLE graph_edges (
    source_id TEXT NOT NULL,
    target_id TEXT NOT NULL,
    relation TEXT NOT NULL,
    weight REAL DEFAULT 1.0,
    PRIMARY KEY (source_id, target_id, relation),
    FOREIGN KEY (source_id) REFERENCES documents(doc_id) ON DELETE CASCADE,
    FOREIGN KEY (target_id) REFERENCES documents(doc_id) ON DELETE CASCADE
);

CREATE INDEX idx_edges_source ON graph_edges(source_id);
CREATE INDEX idx_edges_target ON graph_edges(target_id);
```

### Qdrant collection

- **Имя:** `rag_docs`
- **Dense vectors:** 1024d, cosine distance
- **Sparse vectors:** built-in BM25 (Qdrant ≥1.10)
- **Payload:** `doc_id`, `text_preview`

### Компоненты

| Файл сейчас | Что будет |
|-------------|-----------|
| `src/vector_store.py` | **Полная замена** → `QdrantVectorStore` (embedded/http) |
| `src/bm25_index.py` | **Удалить** (BM25 через Qdrant sparse) |
| `src/graph_store.py` | **Переписать** → SQLite/Postgres + NetworkX BFS |
| `src/document_store.py` | **Новый** — SQLite/Postgres documents CRUD |

### QdrantVectorStore API

```python
class QdrantVectorStore:
    def __init__(self, location: str = "./rag_data/qdrant", dimension: int = 1024):
        # location = "./rag_data/qdrant" → QdrantLocal
        # location = "http://localhost:6333" → QdrantRemote
        ...

    def add(self, doc_id: str, embedding: list[float], text: str, metadata: dict):
        ...

    def search(self, query_embedding: list[float], k: int = 5,
               metadata_filter: dict = None) -> list[dict]:
        ...

    def bm25_search(self, query: str, k: int = 5,
                    metadata_filter: dict = None) -> list[dict]:
        ...  # через sparse vectors

    def delete(self, doc_id: str):
        ...

    def count(self) -> int:
        ...
```

### GraphStore API

```python
class GraphStore:
    def __init__(self, connection: str = "./rag_data/store.db"):
        # connection = "./rag_data/store.db" → SQLite
        # connection = "postgresql://..." → Postgres
        self._build_graph()  # NetworkX из edges таблицы

    def add_edge(self, source_id: str, target_id: str, relation: str, weight: float = 1.0):
        ...

    def get_related(self, node_id: str, max_depth: int = 1,
                    metadata_filter: dict = None) -> dict:
        ...

    def get_edges_batch(self, node_ids: list[str], depth: int = 1,
                        type_filter: list[str] = None,
                        meta_filter: dict = None) -> dict:
        ...

    def delete_document(self, doc_id: str):
        ...

    def delete_all(self):
        ...
```

---

## Миграция данных (старый ChromaDB → Qdrant + SQLite)

**Проблема:** В `rag_data/` лежат старые данные:
- `rag_data/chroma_db/` — ChromaDB с 4096d векторами (Ollama qwen3-embedding)
- `rag_data/bm25_index.pkl` — pickle BM25
- `rag_data/graph_store.json` — граф в JSON

Новые модели (BGE-M3, 1024d) НЕСОВМЕСТИМЫ со старыми векторами. Переиндексация обязательна.

### Стратегия: clean start с флагом миграции

**Вариант A (рекомендуемый): `rag-server --migrate`**

При старте с флагом `--migrate`:
1. Читает старую ChromaDB (`rag_data/chroma_db/`):
   - Для каждого документа: `{doc_id, text, metadata}`
   - Векторы **НЕ** переносятся (другая размерность)
2. Читает старый граф (`rag_data/graph_store.json`):
   - Все узлы и рёбра
3. Создаёт новые стора (Qdrant embedded + SQLite)
4. Переиндексирует каждый документ:
   - Чанкует (если > chunk_size)
   - Считает embedding через BGE-M3
   - Пишет в Qdrant (dense + sparse)
   - Пишет документ в SQLite
5. Восстанавливает граф:
   - Все узлы из старого графа (если doc_id существует)
   - Все рёбра между существующими узлами
6. Переименовывает старые директории в `rag_data/chroma_db.bak`, `rag_data/bm25_index.pkl.bak`, `rag_data/graph_store.json.bak`
7. Логирует: `Migrated N documents, M edges from old store`

**Флаг:** `--migrate` / `--no-migrate` (default: `--no-migrate` — чистый старт)

**Если --migrate не указан, а старые данные есть:**
```
WARNING: Found old data in rag_data/ (chroma_db, bm25_index.pkl).
These are NOT compatible with the new store.
Run `rag-server --migrate` to re-index them, or delete rag_data/ manually.
```

**Вариант B: ручная миграция через CLI**
```bash
rag-server migrate                    # та же логика, отдельная команда
rag-server migrate --dry-run          # показать что будет перенесено
rag-server migrate --force            # не спрашивать подтверждения
```

### Обработка dimension mismatch на лету

В QdrantVectorStore добавить детекцию при старте:
```python
def _check_dimension(self):
    if self._collection_exists():
        actual_dim = self._get_collection_dim()
        if actual_dim != self.config.dimension:
            logger.warning(
                f"Collection dim={actual_dim}, config dim={self.config.dimension}. "
                f"Run `rag-server migrate` to re-index."
            )
```

**Что НЕ мигрируем:**
- Старые векторы (размерность другая) — пересчитываем через BGE-M3
- BM25 pickle — пересчитывается через Qdrant sparse при upsert

**Что теряется при миграции:**
- Только старые векторы (они бесполезны с новой моделью)
- Текст, метаданные, граф — всё сохраняется

---

## Фаза 3: HTTP REST API + MCP Streamable HTTP

### Запуск

```bash
rag-server                        # stdio MCP (как сейчас)
rag-server --http                 # stdio + HTTP (REST + MCP SSE)
rag-server --http --port 8080     # на конкретном порту
```

### REST эндпоинты (`src/http_api.py`)

| Метод | Путь | Описание |
|-------|------|----------|
| GET | `/health` | Проверка здоровья |
| GET | `/stats` | `{total_documents, dimension, store_path}` |
| GET | `/graph/stats` | `{total_nodes, total_edges, relation_types}` |
| POST | `/search` | `{query, k, mode: semantic|bm25|hybrid, alpha, ...}` |
| POST | `/documents` | `{text, meta, extract_graph}` — add |
| GET | `/documents` | `{limit, offset, metadata_filter}` — list |
| GET | `/documents/{doc_id}` | `{offset, limit, relations_load_depth}` — get |
| DELETE | `/documents/{doc_id}` | delete |
| POST | `/relations` | `{source_id, target_id, relation, weight}` |
| GET | `/relations/{node_id}` | `{max_depth, metadata_filter}` |
| POST | `/clear` | wipe all |
| POST | `/structured` | `{content, format: repomix, extract_graph}` |

### MCP HTTP транспорт

- SSE (Server-Sent Events) — MCP 2025-03-26 Streamable HTTP
- Эндпоинт: `GET /mcp` (SSE), `POST /mcp` (JSON-RPC messages)
- **stdio остаётся** как основной для Claude Desktop / Cursor

### Реализация

```python
# src/http_api.py
from fastapi import FastAPI
from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Инициализация RAGSystem
    yield
    # Cleanup

app = FastAPI(lifespan=lifespan, title="graphrag", version="0.2.0")
```

- Handlers переиспользуют те же методы `RAGSystem`
- Pydantic модели для request/response
- `src/cli.py` — флаг `--http`

---

## Фаза 3.5: WebUI — графический интерфейс

**Статус:** ❌ Не реализовано

Графический интерфейс пользователя, работающий через HTTP REST API системы.

### Архитектура

- SPA (Single Page Application) на чистом HTML/CSS/JS — Zero зависимостей (без npm/node)
- Все запросы — к REST API (`/search`, `/documents`, `/stats`, `/graph/*`, `/health`)
- Встраивается как статика в FastAPI (папка `src/webui/`)
- Единый HTML-файл с inline CSS/JS для простоты деплоя

### Функциональность

| Раздел | Что показывает | REST эндпоинт |
|--------|---------------|---------------|
| **Поиск** | Поле ввода, выбор режима (semantic/BM25/hybrid), результаты с подсветкой | `POST /search` |
| **Документы** | Список с пагинацией, поиск, удаление, добавление нового | `GET/POST/DELETE /documents` |
| **Граф** | Визуализация графа (vis.js, интерактивно, панорамирование) | `GET /graph/viz` или `GET /graph/stats` + BFS |
| **Статистика** | Количество документов, размерность, узлы/рёбра графа | `GET /stats`, `GET /graph/stats` |
| **Health** | Пинг сервера, статус компонентов | `GET /health` |

### Макет UI

```html
<!-- src/webui/index.html — единый файл -->
┌──────────────────────────────────────┐
│  🧠 graphrag · WebUI                 │
│  ┌─[Search]──[Documents]──[Graph]──┐ │
│  │  [статистика]                   │ │
│  └─────────────────────────────────┘ │
│                                      │
│  ┌──────────────────────────────────┐│
│  │ 🔍 [   поисковый запрос    ] [Go]││
│  │ Mode: [semantic ▼] k: [5]        ││
│  │ ┌──────────────────────────────┐ ││
│  │ │ 📄 Результат 1 (score: 0.92) │ ││
│  │ │   Текст результата...        │ ││
│  │ │   metadata: {...}            │ ││
│  │ └──────────────────────────────┘ ││
│  └──────────────────────────────────┘│
└──────────────────────────────────────┘
```

### Технологии

- **Чистый HTML5 + CSS3** (Flexbox/Grid, CSS variables для темы)
- **Vanilla JS** (ES6 modules) — без фреймворков
- **vis.js** — уже есть в проекте (для графа)
- **CSS Variables** — поддержка светлой и тёмной темы
- **Fetch API** — все запросы через `fetch()` к REST API
- **Адаптивность** — mobile-first, работает на телефонах

### Реализация

```python
# src/http_api.py — добавить mount статики
from fastapi.staticfiles import StaticFiles

# В lifespan / в конце
app.mount("/ui", StaticFiles(directory="src/webui", html=True), name="webui")
# Редирект с / на /ui
@app.get("/")
async def root():
    return RedirectResponse(url="/ui")
```

### Запуск

```bash
rag-server --http        # REST API на :8765, WebUI на /ui
rag-server --http --port 8765  # → http://localhost:8765/ui
```

### Без зависимостей

**Zero npm/node_modules.** Всё в одном HTML-файле с inline стилями и скриптами.
Это критично для простоты деплоя (особенно в Docker).

---

## Фаза 4: Reranker

**Модель:** `BAAI/bge-reranker-v2-m3` через `sentence-transformers` (`CrossEncoder`)

```python
from sentence_transformers import CrossEncoder

class Reranker:
    def __init__(self, device: str = "cpu"):
        self.model = CrossEncoder("BAAI/bge-reranker-v2-m3", device=device)

    def rerank(self, query: str, candidates: list[dict], top_k: int = 5) -> list[dict]:
        pairs = [(query, doc["text"]) for doc in candidates]
        scores = self.model.predict(pairs)
        for doc, score in zip(candidates, scores):
            doc["rerank_score"] = float(score)
        candidates.sort(key=lambda x: x["rerank_score"], reverse=True)
        return candidates[:top_k]
```

**Встраивание в `rag.search_hybrid()`:**

```
query → vector (k×3) ─┐
query → bm25  (k×3) ──┤→ RRF fusion (k×2) → reranker → top-k
```

**Параметры (config):**
```python
RERANK_ENABLED: bool = False
RERANK_MODEL: str = "BAAI/bge-reranker-v2-m3"
RERANK_DEVICE: str = "cpu"
RERANK_TOP_K_MULTIPLIER: int = 2  # k×2 кандидатов на вход reranker'а
```

- По умолчанию выключен (загрузка модели ~1GB, inference ~30ms/doc)
- Включается через `rerank=true` в запросе или `RERANK_ENABLED=true` в env
- Используется только на hybrid search (semantic + bm25 уже дают хорошие кандидаты)

---

## Фаза 5: Чанкование

**Стратегия:** Semantic splitting с fallback.

```python
chunk_size = 512      # токенов
chunk_overlap = 64    # токенов
split_by = "paragraph"  # paragraph → sentence → token
```

**Алгоритм:**
1. Разбить текст на параграфы (по `\n\n`)
2. Если параграф > `chunk_size` → разбить на предложения (по `. ! ?`)
3. Если предложение > `chunk_size` → разбить по токенам
4. Склеивать соседние части пока не превысят `chunk_size`

**Реализация:** `src/chunker.py`

```python
class Chunker:
    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 64):
        ...

    def chunk(self, text: str, doc_id: str) -> list[dict]:
        """Returns [{doc_id, parent_doc_id, chunk_index, text, metadata}]"""
        ...
```

- Каждый чанк → отдельная запись в Qdrant + SQLite
- `parent_doc_id` связывает чанки в исходный документ
- `rag_get_document(doc_id)` возвращает все чанки, склеенные в порядке `chunk_index`
- `metadata.parent_doc_id` для фильтрации

**Зависимость:** `tiktoken` для токенизации (можно использовать BGE-M3 tokenizer).

---

## Фаза 6: Repomix-стиль индексация

**Вместо `rag_add_file`:** новый инструмент `rag_add_structured`.

### Формат repomix

```json
{
  "repository": "graphrag",
  "structure": ["src/main.py", "src/utils.py", "tests/test_main.py"],
  "files": {
    "src/main.py": {
      "content": "def main(): pass",
      "language": "python",
      "size": 1024
    }
  }
}
```

### Пайплайн индексации

1. **Разобрать структуру** — дерево папок/файлов индексируется как отдельный документ
   с типом `structure` и метаданными `{source: repo, type: structure}`
2. **Для каждого файла:**
   - Чанковать содержимое
   - Индексировать с метаданными `{source: repo, path, language, type: file}`
3. **Опциональный LLM-шаг** (`small_llm_summary=true`):
   - Малая модель (Qwen3-1.8B) генерирует summary модуля
   - Индексируется как документ с типом `summary`
   - Связь `summarizes` с исходным файлом
4. **Автоматические связи:**
   - Файлы в одной папке → `sibling`
   - Импорты (парсинг AST для .py/.js) → `depends_on`
5. **Метаданные:** `{source: repo, language, path, type: file|summary|structure}`

### MCP инструменты

```python
rag_add_structured(content: str, format: str = "repomix",
                   extract_graph: bool = False,
                   small_llm_summary: bool = False) -> {doc_id, status, files_count}
```

- `rag_add_document(text, meta)` — остаётся как есть
- `rag_add_file` — **удаляется** (незачем, для файлов есть repomix)

---

## Фаза 7: Авто-извлечение графа

### LLM-based (основной)

```python
# src/graph_extractor.py

class GraphExtractor:
    def __init__(self, llm_provider: str = "ollama", model: str = "qwen3:4b"):
        ...

    def extract(self, text: str) -> list[tuple[str, str, str]]:
        """
        Возвращает [(entity_a, relation, entity_b), ...]
        """
        prompt = f"""Extract entity-relation triples from the text.
Format each triple as: entity_a | relation | entity_b

Text:
{text[:4000]}
"""
        response = self.llm.generate(prompt)
        return self._parse_triples(response)
```

**Механизм:**
- Запускается при `rag_add_document` / `rag_add_structured` если `extract_graph=true`
- Модель: Qwen3-4B (Ollama, ~2GB, быстрая на CPU)
- Для каждой triplet:
  - Вычислить SHA256 нормализованного имени сущности
  - Если документ-сущность не существует — создать (текст = имя, meta: `{type: entity, auto_extracted: true}`)
  - Создать ребро: `doc_id → relation → entity_doc_id`
  - Создать ребро: `entity_doc_id → extracted_from → doc_id`

### NER fallback

```python
# spaCy для быстрого извлечения без LLM
import spacy

class NerExtractor:
    def __init__(self):
        self.nlp_en = spacy.load("en_core_web_sm")
        self.nlp_ru = spacy.load("ru_core_news_sm")

    def extract_entities(self, text: str, lang: str = "en") -> list[str]:
        nlp = self.nlp_ru if lang == "ru" else self.nlp_en
        doc = nlp(text)
        return [ent.text for ent in doc.ents if ent.label_ in {"PERSON", "ORG", "PRODUCT", "GPE"}]
```

NER используется только для entities (без relations), как бюджетная альтернатива.
Выключается через `EXTRACT_GRAPH_MODE=ner` или `extract_graph=ner`.

---

## Фаза 8: Query expansion

**Опционально** (`query_expansion=true`, default `false`):

```python
# src/query_expander.py

class QueryExpander:
    def __init__(self, model: str = "qwen3:1.8b"):
        ...

    def expand(self, query: str, count: int = 3) -> list[str]:
        prompt = f"""Generate {count} alternative phrasings of this query.
Return one per line, no numbering.

Query: {query}"""
        response = self.llm.generate(prompt)
        variants = [q.strip() for q in response.split("\n") if q.strip()]
        return [query] + variants[:count]
```

**Пайплайн:**
```
query → expand → [q, q1, q2, q3]
  → parallel hybrid search (каждый)
  → RRF merge всех результатов
  → rerank (если включён)
  → top-k
```

- Малая модель (Qwen3-1.8B) — быстрая, локальная
- Параметр: `expand_count=3` (config: `QUERY_EXPANSION_COUNT`)
- Параметр в search: `query_expansion=true`

---

## Фаза 9: Graph viz — стильный визуализатор

**Что меняем:** `src/graph_viz.py` — переписать под новую архитектуру.

### Требования

- **API:** HTTP endpiont `GET /graph/viz` + возможность сохранить HTML файл через CLI
- **Все ноды — круги** разного цвета
- **Цвет:** зависит от хэша метаданных документа:
  ```python
  def node_color(metadata: dict | None, text: str) -> str:
      if metadata and isinstance(metadata, dict) and len(metadata) > 0:
          raw = json.dumps(metadata, sort_keys=True)
      else:
          raw = text[:30]
      h = hashlib.md5(raw.encode()).hexdigest()
      return f"hsl({int(h[:6], 16) % 360}, 60%, 50%)"
  ```
- **Размер ноды:** пропорционален количеству связей (degree centrality)
- **Подпись:** первые 40 символов текста или doc_id (если текст слишком длинный)
- **Рёбра:** серые полупрозрачные линии, толщина = weight
- **Легенда:** типы отношений (relation types) в углу
- **Интерактив:** vis.js (как сейчас), zoom, drag, hover — подсветка соседей

### CLI

```bash
rag-server --http graph-viz        # открыть в браузере (live)
rag-server graph-viz -o graph.html # сохранить HTML
```

### Технология

vis.js (как сейчас). HTML генерируется на сервере, отдаётся как HTML-страница.

---

## Фаза 10: Лицензия MIT

- `LICENSE` — MIT
- `pyproject.toml` — `license = "MIT"`

---

## Фаза 11: Деплой одной командой

```bash
pip install graphrag && rag-server                   # stdio (дефолт)
pip install graphrag && rag-server --http            # + REST API + MCP SSE
pip install graphrag && rag-server --http --port 8080
```

**Docker:**
```dockerfile
FROM python:3.12-slim
RUN pip install graphrag
EXPOSE 8765
CMD ["rag-server", "--http", "--port", "8765"]
```

**Docker Compose (production):**
```yaml
services:
  qdrant:
    image: qdrant/qdrant
    volumes: [./rag_data/qdrant:/storage]
  postgres:
    image: postgres:16
    environment: [POSTGRES_DB=graphrag, POSTGRES_PASSWORD=...]
  rag:
    build: .
    environment:
      QDRANT_URL: http://qdrant:6333
      DATABASE_URL: postgresql+psycopg://...@postgres/graphrag
    ports: [8765:8765]
```

---

## Фаза 12: pyproject.toml (замена requirements.txt)

```toml
[project]
name = "graphrag"
version = "0.2.0"
description = "Production-grade RAG MCP server with hybrid search, graph, and HTTP API"
license = "MIT"
requires-python = ">=3.11"

dependencies = [
    "numpy>=1.26",
    "httpx>=0.27",
    "sentence-transformers>=3.0",
    "qdrant-client>=1.10",
    "fastapi>=0.115",
    "uvicorn>=0.30",
    "tiktoken>=0.7",
    "mcp>=1.0",
    "networkx>=3.0",
]

[project.scripts]
rag-server = "src.cli:main"

[tool.pytest.ini_options]
testpaths = ["tests"]
```

`uv sync` → `uv.lock` (без `requirements.txt`).

---

## Что удаляем

| Файл / зависимость | Причина |
|--------------------|---------|
| `requirements.txt` | Заменён на `pyproject.toml` + `uv.lock` |
| `chromadb` | Заменён на Qdrant |
| `rank-bm25` | BM25 через Qdrant sparse vectors |
| `src/bm25_index.py` | Больше не нужен |
| `src/vector_store.py` | Полная замена |
| `src/graph_store.py` | Переписан |
| `debug_bm25.py` | Debug-мусор |

---

## Файлы под изменение/создание

| Файл | Действие |
|------|----------|
| `pyproject.toml` | **Новый** — uv-совместимый |
| `LICENSE` | **Новый** — MIT |
| `src/config.py` | +BGE-M3, Qdrant, SQLite, reranker, chunking, expansion |
| `src/embeddings.py` | +BgeM3EmbeddingGenerator |
| `src/vector_store.py` | **Полная замена** → QdrantVectorStore |
| `src/bm25_index.py` | **Удалить** |
| `src/graph_store.py` | **Переписать** → SQLite/Postgres + NetworkX |
| `src/document_store.py` | **Новый** — SQLite/Postgres documents CRUD |
| `src/chunker.py` | **Новый** |
| `src/graph_extractor.py` | **Новый** |
| `src/query_expander.py` | **Новый** |
| `src/http_api.py` | **Новый** — FastAPI |
| `src/webui/index.html` | **Новый** — SPA WebUI (один HTML-файл) |
| `src/graph_viz.py` | **Переписать** — новый стиль, HTTP API |
| `src/mcp_server.py` | **Рефакторинг** — новый storage, убрать rag_add_file |
| `src/rag.py` | **Рефакторинг** — новый storage, reranker, chunking |
| `src/cli.py` | +--http флаг |
| `src/index.py` | Апдейт под новый storage |
| `requirements.txt` | **Удалить** |
| `tests/*` | Апдейт под новые сторы |

---

## Миграция тестов (373 → ~450)

- Все существующие тесты переписать под Qdrant + SQLite (те же asserts, другие драйверы)
- `semantic_mock.py` — без изменений (не зависит от стора)
- `test_search_quality.py` — без изменений
- **Новые тесты:**
  - HTTP API (httpx, ~30 тестов)
  - Chunking (~10 тестов)
  - Reranker (~10 тестов)
  - Query expansion (~10 тестов)
  - Graph extraction (~10 тестов)
  - Repomix parsing (~10 тестов)
  - BGE-M3 embeddings (~10 тестов)

---

## Порядок реализации

| Фаза | Что | Зависит от |
|------|-----|------------|
| **0** | uv + pyproject.toml | — |
| **1** | BGE-M3 embedder | 0 |
| **2** | Qdrant + SQLite storage | 1 |
| **3** | HTTP REST API + MCP SSE | 2 |
| **3.5** | WebUI (графический интерфейс) | 3 |
| **4** | Reranker | 2 |
| **5** | Chunking | 2 |
| **6** | Repomix-style indexing | 2, 5 |
| **7** | Auto graph extraction | 2 |
| **8** | Query expansion | 2 |
| **9** | Graph viz (новый стиль) | 3 |
| **10** | MIT License | 0 |
| **11** | Deploy (docker, docs) | 3 |
