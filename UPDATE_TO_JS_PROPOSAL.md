# GraphRAG как фундамент journal-service

> Предложение по использованию `~/Work/graphrag` в качестве поискового и графового
> движка для journal-service системы rnd-agents.
> Дата: 2026-07-14.

---

## 1. Контекст

**journal-service** — компонент системы rnd-agents, реализующий общую память агентов:
записи о решениях (decisions), находках (findings), арбитражах (arbitrations),
proposal-ы на изменения, комментарии людей. Сейчас — **только дизайн-документ**
(`rnd-agents/docs/journal_service.md`), реализации ноль.

**GraphRAG** (`~/Work/graphrag`) — полностью рабочий графовый RAG-сервер с гибридным
поиском, MCP-транспортом и community detection.

Сопоставление показывает, что GraphRAG закрывает 60-70% технической сложности
journal-service «из коробки».

---

## 2. Что journal-service планирует vs что даёт GraphRAG

### 2.1 Поиск по записям

| Требование journal-service | Текущий план | GraphRAG |
+|---------------------------|-------------|----------|
+| Полнотекстовый поиск (`q`) | PostgreSQL `tsvector` GIN (русский) | BM25 sparse vectors в Qdrant — качество выше для технических текстов |
+| Семантический поиск (`q_semantic`) | «vector store, если включён» — без деталей | BGE-M3 1024d эмбеддинги (локально, multilang) |
+| Гибридный поиск | Нет | RRF alpha-dilution — semantic + keyword слияние |
+| Ранжирование | Нет | CrossEncoder `bge-reranker-v2-m3` |
+| Query expansion | Нет | Multi-query через Qwen3-1.8B |
+| Структурные фильтры (`type`, `status`, `scope.*`, `author`) | SQL WHERE по индексам | `metadata_filter` в Qdrant + SQLite/Postgres |

**Вывод:** GraphRAG даёт production-grade поиск без написания vector store и
BM25-индексации с нуля.

### 2.2 Связи между записями

**Сейчас:** плоские FK — `triggered_by`, `superseded_by`, `related_entry_id`.

**GraphRAG даёт:**
+ Произвольный knowledge graph с BFS-обходом (`rag_get_related`)
+ Авто-извлечение сущностей и связей из текста записи (`extract_graph=True`) —
  LLM (Qwen3-4B) или spaCy NER сами строят граф при добавлении записи
+ Типизированные рёбра (`references`, `mentions`, `contradicts`, `supports`, ...)

Пример графа, который строится автоматически:

```
entry: "ADR-024 нарушается в 12 местах"  ←──references──→  ADR-024.md
       │
       └──mentions──→ компонент auth (scope)
       │
       └──related_to──→ entry: "legacy-модуль не обновлён"
```

### 2.3 Community detection

Этого в дизайне journal-service **нет вообще**. GraphRAG даёт:

```python
rag_find_communities()
# → сообщество 0: 23 записи про authentication
# → сообщество 1: 15 записей про deployment
# → сообщество 2: 8 записей про performance
```

Leiden-алгоритм на эмбеддингах + графе связей автоматически кластеризует
записи по темам. Это критично для:

+ **digest-агента** — ежедневная сводка готова к выдаче: «за эту неделю 7
  arbitration-ов про безопасность, 3 про CI/CD, ...»
+ **journal-guardian** — дедупликация не по тексту, а по семантике
+ **code-guardian** — контекст: «в этом репо за месяц было 5 решений
  про обработку ошибок — проверь новый MR на соответствие»

### 2.4 MCP-Journal

В дизайне: написать MCP-сервер с нуля (`progress-agents/mcp/journal/`).

GraphRAG — **уже работающий MCP-сервер** с 25+ инструментами. Маппинг прямой:

| MCP-Journal (дизайн) | GraphRAG tool |
+|----------------------|---------------|
+| `search_entries(query, scope, type, ...)` | `rag_search_hybrid(query, metadata_filter={...})` |
+| `get_entry(id)` | `rag_get_document(doc_id)` |
+| `get_recent(since, limit)` | `rag_list_documents` + фильтр по дате |
+| `add_entry(type, subject, content, ...)` | `rag_add_document(text, meta={type, subject, ...})` |
+| `add_comment(entry_id, content)` | `rag_add_document(text, meta={parent: entry_id, type: "comment"})` |
+| `get_related` | `rag_get_related(node_id)` |
+| `find_communities` | `rag_find_communities()` |

Транспорты: stdio (для agent_runtime) и SSE (для Mozart).

### 2.5 HTTP REST API

В дизайне: написать FastAPI-сервер с нуля.

GraphRAG даёт работающий `src/http_api.py` — 15+ эндпоинтов, OpenAPI `/docs`,
Docker-деплой. Можно форкнуть как основу, добавив JWT-middleware и
эндпоинты для proposals / approve / reject.

---

## 3. Что GraphRAG НЕ закрывает

Это чисто бизнес-логика journal-service, которой в GraphRAG и не должно быть:

| Компонент | Что нужно написать |
+|-----------|-------------------|
+| JWT-аутентификация | Middleware, валидация токенов от auth-issuer, scope-чеки |
+| Proposals + HITL | API `/proposals` CRUD, `/approve`, `/reject`; workflow |
+| Outbox → RabbitMQ | Запись событий в БД в транзакции, асинхронная публикация |
+| Wiki-publisher | Воркер зеркалирования записей на Yandex Wiki |
+| Wiki-listener | Приём комментариев с Wiki → `journal.comments` |
+| Proposal-notifier | Постинг в Mattermost `#agent-proposals`, редактирование сообщений |
+| Expiration-checker | Cron-воркер: `valid_until < now()` → `expired` |
+| Схема `proposals`, `comments`, `status_history` | SQL-миграции (Alembic) |

---

## 4. Предлагаемая архитектура: сквозной UUID + DocumentStore как интерфейс

### 4.1 Ключевая идея

Вместо двух независимых таблиц (`journal.entries` + `graphrag.documents` с
разными UUID) — **один UUID на запись**, который связывает typed source of truth,
индексную строку в GraphRAG, вектор в Qdrant, и рёбра в графе.

Journal-service реализует typed-схему `journal.entries` и предоставляет
**JournalDocumentStore** — адаптер, реализующий интерфейс GraphRAG
`AbstractDocumentStore`. GraphRAG не знает про typed-поля — он работает
с `doc_id + text + metadata`, как и раньше.

В будущем другие typed-источники (ADR, meeting notes, code chunks) добавляются
аналогично — каждый со своей typed-таблицей и своим DocumentStore, но
индексируются и связываются через **один и тот же Qdrant + граф по UUID**.

```
                      ОДИН UUID = doc_id
                           │
      ┌────────────────────┼────────────────────┐
      ▼                    ▼                    ▼
journal.entries       graphrag index        Qdrant (векторы)
(typed source         (text + metadata      (dense + sparse)
 of truth)             JSON)
      │                    │                    │
      └────────────────────┼────────────────────┘
                           │
                  graphrag.graph_edges
                  (source_id → target_id
                   по UUID любых типов)
```

### 4.2 Интерфейс AbstractDocumentStore (добавить в GraphRAG)

```python
# graphrag/src/document_store.py — новый ABC
from abc import ABC, abstractmethod
from typing import Optional, TypedDict

class DocumentRow(TypedDict):
    doc_id: str
    text: str
    metadata: dict
    content_hash: str | None
    parent_doc_id: str | None
    chunk_index: int | None
    created_at: str

class AbstractDocumentStore(ABC):
    @abstractmethod
    def add(self, doc_id: str, text: str, metadata: Optional[dict] = None,
            content_hash: Optional[str] = None, parent_doc_id: Optional[str] = None,
            chunk_index: Optional[int] = None) -> str: ...

    @abstractmethod
    def get(self, doc_id: str) -> Optional[DocumentRow]: ...

    @abstractmethod
    def get_batch(self, doc_ids: list[str]) -> dict[str, DocumentRow]: ...

    @abstractmethod
    def get_chunks(self, parent_doc_id: str) -> list[DocumentRow]: ...

    @abstractmethod
    def delete(self, doc_id: str) -> bool: ...

    @abstractmethod
    def update_text(self, doc_id: str, text: str,
                    content_hash: Optional[str] = None) -> bool: ...

    @abstractmethod
    def update_metadata(self, doc_id: str, metadata: dict) -> bool: ...

    @abstractmethod
    def find_by_hash(self, content_hash: str) -> Optional[str]: ...

    @abstractmethod
    def list(self, limit: int = 20, offset: int = 0,
             metadata_filter: Optional[dict] = None) -> tuple[list[DocumentRow], int]: ...

    @abstractmethod
    def count(self) -> int: ...

    @abstractmethod
    def all_ids(self) -> set[str]: ...

    @abstractmethod
    def all_hashes(self) -> dict[str, str]: ...

    @abstractmethod
    def get_metadata_batch(self, doc_ids: list[str]) -> dict[str, dict]: ...

    @abstractmethod
    def clear(self) -> None: ...
```

Существующий `DocumentStore` становится реализацией этого интерфейса по умолчанию
(для standalone-использования GraphRAG без внешнего typed-источника).

### 4.3 JournalDocumentStore — адаптер поверх journal.entries

```python
# journal_service/store.py
class JournalDocumentStore(AbstractDocumentStore):
    """Реализует AbstractDocumentStore поверх typed-таблицы journal.entries."""

    def __init__(self, db: Database):
        self._db = db

    def add(self, doc_id, text, metadata=None, content_hash=None, **kwargs):
        meta = metadata or {}
        subject, content = _split_subject_content(text)

        self._db.execute("""
            INSERT INTO journal.entries
                (id, type, agent_author, subject, content, scope,
                 source_refs, status, confidence, conditions, valid_until,
                 superseded_by, triggered_by, trigger_depth)
            VALUES (%s, %s, %s, %s, %s, %s, %s, 'active', %s, %s, %s, %s, %s, %s)
        """, (
            doc_id,
            meta.get("type"),
            meta.get("agent_author"),
            subject,
            content,
            json.dumps(meta.get("scope", {})),
            json.dumps(meta.get("source_refs", [])),
            meta.get("confidence"),
            meta.get("conditions"),
            meta.get("valid_until"),
            meta.get("superseded_by"),
            meta.get("triggered_by"),
            meta.get("trigger_depth", 0),
        ))
        return doc_id

    def get(self, doc_id):
        row = self._db.execute(
            "SELECT id, type, agent_author, subject, content, scope, "
            "       source_refs, status, confidence, conditions, valid_until, "
            "       superseded_by, triggered_by, trigger_depth, created_at, updated_at "
            "FROM journal.entries WHERE id = %s",
            (doc_id,),
        )
        if not row:
            return None
        return self._to_document_row(row[0])

    def _to_document_row(self, row: tuple) -> DocumentRow:
        """Мапит typed-строку journal.entries → универсальный DocumentRow."""
        return {
            "doc_id": row[0],
            "text": f"{row[3]}\n\n{row[4]}",       # subject + content
            "metadata": {
                "type": row[1],
                "agent_author": row[2],
                "status": row[7],
                "scope": row[5],                    # JSONB → dict
                "source_refs": row[6],              # JSONB → list
                "confidence": row[8],
                "conditions": row[9],
                "valid_until": row[10].isoformat() if row[10] else None,
                "superseded_by": row[11],
                "triggered_by": row[12],
                "trigger_depth": row[13],
            },
            "content_hash": None,
            "parent_doc_id": None,
            "chunk_index": None,
            "created_at": row[14].isoformat(),
        }

    # get_batch, list, count, all_ids, delete, update_metadata —
    # аналогично, SELECT/UPDATE поверх journal.entries с обратным маппингом
```

### 4.4 Сквозной поиск по всем типам документов

```
Запрос: "auth decisions backend-api"
         ↓
Qdrant: dense + sparse по ОДНОЙ коллекции (все UUID)
         ↓
Результат:
  journal_entry#abc-123 (score 0.92) — "ADR-024 нарушается в auth-модуле"
  adr#def-456 (score 0.87)           — "ADR-024: Authentication Flow"
  code_chunk#ghi-789 (score 0.78)    — "func Login(...) в src/auth/login.go"
  meeting#jkl-012 (score 0.72)       — "Обсуждали миграцию auth на OAuth2"
         ↓
BFS по графу: doc_id → связанные узлы (любых типов)
         ↓
Полный контекст для агента: решение + ADR + код + протокол обсуждения
```

**Создание записи — одна транзакция, один UUID:**

```python
async def create_entry(entry: JournalEntry, graphrag: RAGSystem):
    doc_id = entry.id  # UUID генерируется один раз

    async with db.transaction():
        # 1. Typed source of truth (через JournalDocumentStore.add)
        graphrag.doc_store.add(
            doc_id=doc_id,
            text=f"{entry.subject}\n\n{entry.content}",
            metadata={...typed поля...}
        )
        # 2. Outbox в той же транзакции
        await db.execute("INSERT INTO journal.events_outbox ...")

    # 3. Векторы в Qdrant — тот же UUID
    graphrag.vector_store.add(doc_id=doc_id, text=..., embedding=..., metadata=...)

    # 4. Граф — тот же UUID связывает с ADR, meeting notes и т.д.
    for ref in entry.source_refs:
        graphrag.graph_kb.add_edge(doc_id, ref["id"], "references")
```

### 4.5 Полная архитектура

```
journal-service (FastAPI)
│
├── PostgreSQL (одна БД)
│   ├── journal.entries          ← typed source of truth
│   ├── journal.proposals        ← HITL proposals
│   ├── journal.comments         ← комментарии людей
│   ├── journal.status_history   ← audit log
│   ├── journal.events_outbox    ← outbox pattern
│   ├── graphrag.graph_edges     ← граф: UUID → UUID (любые типы)
│   └── (будущее) adr.entries, meetings.entries, code_chunks
│
├── JournalDocumentStore         ← адаптер: journal.entries ↔ AbstractDocumentStore
│
├── GraphRAG (embedded, получает JournalDocumentStore через RAGSystem(doc_store=...))
│   ├── Qdrant                   ← одна коллекция, dense + sparse по UUID
│   ├── NetworkX                 ← граф связей (все типы) + BFS
│   └── Leiden communities       ← тематическая кластеризация (все типы)
│
├── MCP-Journal                  ← тонкая обёртка над GraphRAG MCP tools
│   └── добавляет JWT, маппинг терминов (entry ↔ document)
│
└── Воркеры
    ├── Wiki-publisher
    ├── Wiki-listener
    ├── Proposal-notifier
    ├── Event-emitter (outbox publisher)
    └── Expiration-checker
```

---

## 5. Что нужно поменять в GraphRAG

Все изменения — в `src/document_store.py` и `src/rag.py`. **Qdrant, граф, MCP,
community detection, reranker, query expansion — не трогаем.** Они работают
с `doc_id` и уже готовы к сквозному UUID.

| Изменение | Файл | Строк | Зачем |
+|-----------|------|:-----:|-------|
+| `AbstractDocumentStore` (ABC) | `src/document_store.py` | ~20 | Интерфейс, который journal-service реализует |
+| `DocumentRow` (TypedDict) | `src/document_store.py` | ~10 | Типизированная строка документа |
+| `DocumentStore` → implements `AbstractDocumentStore` | `src/document_store.py` | ~5 | Существующий класс реализует интерфейс |
+| `RAGSystem.__init__` принимает `doc_store: AbstractDocumentStore` | `src/rag.py` | ~5 | Внедрение зависимости вместо создания |
+| `RAGSystem.__init__` — `Database` опциональна (если doc_store внешний) | `src/rag.py` | ~10 | Не создавать SQLite-схему, если store предоставлен |
+| `GraphStore` — FK на `documents.doc_id` → FK на UUID без привязки к таблице | `src/graph_store.py` | ~5 | Рёбра графа ссылаются на UUID, а не на конкретную таблицу |
+| **Итого** | | **~55** | |

Опционально:

| Изменение | Строк | Зачем |
+|-----------|:-----:|-------|
+| Инкрементальная индексация | ~50 | Не перестраивать Qdrant при старте |
+| Персистентность графа (без перестроения из БД) | ~30 | Загрузка графа из БД, а не перестроение |

---

## 6. Оценка трудозатрат

| Этап | Без GraphRAG | С GraphRAG (v1) | С GraphRAG (v2 — сквозной UUID) |
+|------|:-----------:|:----------:|:----------:|
+| Доработка GraphRAG (AbstractDocumentStore) | — | — | 1 день |
+| 1. PostgreSQL + REST API + поиск + outbox | 1 нед | 3 дня | 3 дня |
+| 2. Wiki-publisher + Wiki-listener | 1 нед | 1 нед | 1 нед |
+| 3. Proposals + Mattermost | 1 нед | 1 нед | 1 нед |
+| 4. JWT + auth-issuer | 1 нед | 1 нед | 1 нед |
+| 5. Lifecycle (expiry, метрики) | 0.5 нед | 1 день | 1 день |
+| **Итого** | **~5 нед** | **~3.5 нед** | **~3.5 нед** |

v2 требует +1 день на доработку GraphRAG, но устраняет риск расхождения схем
и открывает путь к сквозному поиску по всем типам документов.

---

## 7. Схема journal.entries — детально

Полная схема typed source of truth, которую JournalDocumentStore оборачивает
в интерфейс AbstractDocumentStore:

```sql
CREATE SCHEMA IF NOT EXISTS journal;

CREATE TABLE journal.entries (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    created_at      TIMESTAMP NOT NULL DEFAULT now(),
    updated_at      TIMESTAMP NOT NULL DEFAULT now(),

    -- Авторство: заполняется из JWT, не из payload запроса
    agent_author    TEXT NOT NULL,      -- "guardian-architecture" / "human:ivanov"

    -- Тип записи определяет смысл и дальнейшую обработку
    type            TEXT NOT NULL,      -- decision / finding / question / arbitration
                                        -- arbitration_request / decision_observed
                                        -- arch_drift / feedback / meeting_processed
                                        -- regression_test_added

    -- Содержание
    subject         TEXT NOT NULL,      -- короткий заголовок (≤200 символов)
    content         TEXT NOT NULL,      -- markdown с обоснованием

    -- Контекст: к чему относится запись
    scope           JSONB NOT NULL DEFAULT '{}'::jsonb,
                                        -- {repo, component, file, part}
    source_refs     JSONB NOT NULL DEFAULT '[]'::jsonb,
                                        -- [{type: "mr"|"task"|"adr"|"meeting", url, id}]

    -- Жизненный цикл
    status          TEXT NOT NULL DEFAULT 'active'
                    CHECK (status IN ('active', 'superseded', 'resolved',
                                      'expired', 'rejected')),
    conditions      TEXT,               -- условия отмены (свободный текст)
    valid_until     TIMESTAMP,          -- явная дата истечения
    superseded_by   UUID REFERENCES journal.entries(id),

    -- Защита от триггерных петель
    triggered_by    UUID REFERENCES journal.entries(id),
    trigger_depth   INTEGER NOT NULL DEFAULT 0
                    CHECK (trigger_depth >= 0 AND trigger_depth <= 5),

    -- Уверенность агента (0..1, только для finding/question)
    confidence      NUMERIC(3, 2)
                    CHECK (confidence IS NULL OR (confidence >= 0 AND confidence <= 1))
);

-- Индексы
CREATE INDEX idx_entries_status_type ON journal.entries (status, type);
CREATE INDEX idx_entries_scope_gin ON journal.entries USING GIN (scope);
CREATE INDEX idx_entries_created ON journal.entries (created_at DESC);
CREATE INDEX idx_entries_author ON journal.entries (agent_author);
CREATE INDEX idx_entries_valid_until ON journal.entries (valid_until)
    WHERE valid_until IS NOT NULL;
```

Маппинг `journal.entries` → `DocumentRow` (в JournalDocumentStore):

| Поле DocumentRow | Источник из journal.entries |
+|-----------------|---------------------------|
+| `doc_id` | `id` |
+| `text` | `subject + "\n\n" + content` |
+| `metadata.type` | `type` |
+| `metadata.agent_author` | `agent_author` |
+| `metadata.status` | `status` |
+| `metadata.scope` | `scope` (JSONB → dict) |
+| `metadata.source_refs` | `source_refs` (JSONB → list) |
+| `metadata.confidence` | `confidence` |
+| `metadata.conditions` | `conditions` |
+| `metadata.valid_until` | `valid_until` (datetime → ISO str) |
+| `metadata.superseded_by` | `superseded_by` |
+| `metadata.triggered_by` | `triggered_by` |
+| `metadata.trigger_depth` | `trigger_depth` |
+| `content_hash` | `None` (не используется) |
+| `parent_doc_id` | `None` |
+| `chunk_index` | `None` |
+| `created_at` | `created_at` (datetime → ISO str) |

---

## 8. Риски

| Риск | Mitigation |
+|------|-----------|
+| GraphRAG не держит нагрузку production-объёмов | Qdrant server (не embedded), PostgreSQL для графа |
+| Маппинг typed ↔ DocumentRow — накладные расходы | Одно чтение на запрос, кеширование в NetworkX, negligible |
+| GraphStore FK на произвольные UUID (нет ссылочной целостности) | Принять: граф — best-effort индекс. При удалении записи — каскад в journal-service очищает рёбра |
+| Разные DocumentStore для разных типов (ADR, meetings, code) | Один RAGSystem, разные typed-таблицы, общий Qdrant + граф. Добавление нового типа — новый адаптер |
| MCP-транспорт GraphRAG не подходит для agent_runtime | Оба используют MCP Python SDK — гарантированная совместимость |
| **Operational complexity: Qdrant + BGE-M3 + reranker + Qwen3** | Вместо одного Postgres получаем три сервиса (Postgres + Qdrant + модели). Embedded Qdrant не держит concurrent-доступ → нужен отдельный Qdrant-сервер. BGE-M3: +2.5GB RAM. Reranker bge-reranker-v2-m3: +1GB. Qwen3-1.8B для query expansion: ещё ~1.5GB. Для MVP можно отключить reranker + expansion и использовать embedded Qdrant |
| **Reindex при schema change** | При изменении typed-схемы `journal.entries` → пересборка маппинга в `JournalDocumentStore` → переиндексация Qdrant. Объём записей растёт со временем — нужно измерять время переиндексации на 10K / 100K / 1M записей до prod-деплоя |
| **Generic metadata_filter теряет typed-выразительность** | Оригинальный MCP-Journal требует фильтров `scope.repo`, `trigger_depth < 5`, `confidence > 0.7` как typed-полей. GraphRAG пропускает их через generic `metadata_filter: dict` — теряется валидация типов, автокомплит в MCP-клиентах, и документированность фильтров |
| **Reranker + expansion — latency на горячем пути** | Первый вызов: 5-15 сек (lazy load моделей). С прелоадом: задержка каждого запроса зависит от размера кандидатов. Для агентов, делающих много вызовов (code-guardian: десятки поисков на один MR), cumulative latency может быть значительной. Рекомендация: `rerank=false`, `query_expansion=false` по умолчанию для агентов, включать опционально для digest и ad-hoc поиска |
| **Тight coupling через Python import** | Journal-service импортирует `RAGSystem` напрямую, а не через API/MCP. При breaking change в GraphRAG — нужно обновлять journal-service синхронно. Mitigation: версионировать AbstractDocumentStore как публичный контракт, CI-проверка совместимости при PR в GraphRAG |

---

## 9. Рекомендация

1. **Добавить `AbstractDocumentStore` в GraphRAG** (~55 строк, 1 день).
2. **Journal-service реализует `JournalDocumentStore`** поверх `journal.entries`.
3. **UUID — сквозной:** одна запись = один UUID на всех слоях (typed таблица →
   индекс → вектор → граф).
4. **Граф — общий:** Qdrant и graph_edges индексируют UUID любых типов документов.
   Сегодня — journal entries, завтра — ADR, meeting notes, code chunks.

**Не делать:** форкать GraphRAG. Использовать как библиотеку через Python import,
передавая `JournalDocumentStore` в `RAGSystem(doc_store=...)`.

---

## 10. Порядок действий

1. Доработать GraphRAG: `AbstractDocumentStore` + `DocumentRow` + параметризация `RAGSystem` (~1 день)
2. Реализовать `JournalDocumentStore` в journal-service (~1 день)
3. Доказать концепт: journal-service с GraphRAG для CRUD + сквозного поиска (1-2 дня)
4. Если ок — полная интеграция по плану: proposals, outbox, воркеры, JWT
