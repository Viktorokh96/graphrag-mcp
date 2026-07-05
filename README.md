# RAG MCP Tool — Графовая база знаний с гибридным поиском

Семантический поиск + BM25 + графовые реляции между документами.  
Работает как **MCP сервер** (JSON-RPC через stdio) — подключается к Claude Desktop, Cline и любым MCP-клиентам.

---

## 🔧 Установка

```bash
# 1. Клонировать
git clone <url> && cd graphrag

# 2. Зависимости
pip install -r requirements.txt

# 3. Настройка провайдера эмбеддингов (см. ниже)
```

**requirements.txt:**
```
chromadb>=0.5.0
rank-bm25>=0.2.2
httpx>=0.27.0
numpy>=1.24.0
pytest>=8.0.0
scikit-learn>=1.3.0
```

---

## 🚀 Быстрый старт (CLI)

```bash
# Добавить документ
python -m src.cli add-document --text "Python — мощный язык программирования"

# Добавить файл
python -m src.cli add-file --path ./doc.txt

# Семантический поиск
python -m src.cli search --query "язык программирования"

# BM25 поиск (по ключевым словам)
python -m src.cli bm25-search --query "Python"

# Гибридный поиск (alpha=null → RAGConfig.default_alpha=0.5, см. ниже)
python -m src.cli hybrid-search --query "Python"
# Явно задать баланс: 0.0=BM25, 1.0=семантика
python -m src.cli hybrid-search --query "Python" --alpha 0.3

# Граф: добавить связь между документами
python -m src.cli add-relation --source UUID1 --target UUID2 --relation "related_to"

# Граф: получить связанные документы
python -m src.cli get-related --node UUID --max-depth 2

# Граф: статистика
python -m src.cli graph-stats

# Статистика хранилища
python -m src.cli stats

# Очистить всё
python -m src.cli clear
```

---

## 🤖 Подключение как MCP сервер к Claude

### Вариант 1: Claude Desktop

В файле конфигурации `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "rag-knowledge-base": {
      "command": "python",
      "args": [
        "-m", "src.mcp_server"
      ],
      "env": {
        "OPENROUTER_API_KEY": "sk-or-v1-..."
      }
    }
  }
}
```

### Вариант 2: Cline / VS Code Extension

В настройках MCP серверов:

```json
{
  "mcpServers": {
    "rag-knowledge-base": {
      "command": "python",
      "args": ["-m", "src.mcp_server"],
      "env": {
        "OPENROUTER_API_KEY": "sk-or-v1-..."
      }
    }
  }
}
```

### Вариант 3: Любой MCP клиент

Подключитесь к процессу: `python -m src.mcp_server`  
Протокол: JSON-RPC 2.0 через stdin/stdout

---

## 📦 MCP Инструменты (доступны Claude после подключения)

> Полный актуальный реестр инструментов поддерживается в `AGENTS.md` (раздел
> «MCP инструменты»). При расхождении — источник истины `AGENTS.md`. Ниже —
> сводка; детали поведения и edge-cases см. в AGENTS.md и `specifications/api.md`.

### Поиск / Query

Все три поиска принимают `query` (обязательный), `k` (число результатов, по умолчанию 5) и `max_chars` (обрезать текст каждого результата до N символов; `null`/опущен = полный текст). Возвращают список `{doc_id, text, score, metadata}`, отсортированных по убыванию score.

| Инструмент | Параметры | Описание |
|-----------|-----------|----------|
| `rag_search` | `query`, `k=5`, `max_chars=null` | Семантический поиск через векторные эмбеддинги. Лучше для концептуальных запросов. Нулевой вектор запроса (неизвестные идентификаторы) → пустой результат. |
| `rag_bm25_search` | `query`, `k=5`, `max_chars=null` | Ключевой поиск по алгоритму BM25 (Okapi). Лучше для точного совпадения терминов/идентификаторов. Работает офлайн. |
| `rag_search_hybrid` | `query`, `k=5`, `alpha=null`, `max_chars=null` | Гибрид: `score = alpha*semantic + (1-alpha)*bm25`. `alpha=null` → `RAGConfig.default_alpha` (env `RAG_DEFAULT_ALPHA`, default **0.5** — выбран бенчмарком NDCG@k, см. `scripts/benchmark_alpha.py`). **Candidate expansion:** из каждого канала забирается `max(k*3, 20)` кандидатов перед fusion. |

### Чтение / Retrieve

| Инструмент | Параметры | Описание |
|-----------|-----------|----------|
| `rag_get_document` | `doc_id`, `offset=0`, `limit=null` | Получить один документ по ID с посимвольной пагинацией. |

### Индексация / Store

| Инструмент | Параметры | Описание |
|-----------|-----------|----------|
| `rag_add_document` | `text`, `meta=null` | Добавить текстовый документ. `meta`: dict/null/""/JSON-строка/строка. |
| `rag_add_file` | `filepath`, `meta=null` | Прочитать файл с диска и проиндексировать. |
| `rag_add_relation` | `source_id`, `target_id`, `relation`, `weight=1.0` | Создать направленное ребро в графе между документами. |

### Управление / Inspect

| Инструмент | Параметры | Описание |
|-----------|-----------|----------|
| `rag_list_documents` | `limit=20`, `offset=0`, `max_chars=null` | Постраничный список документов. |
| `rag_delete_document` | `doc_id` | Удалить документ из всех хранилищ. Идемпотентен. |
| `rag_clear` | — | ⚠️ Удалить ВСЕ данные (необратимо). |

### Графовый обход / Traversal

| Инструмент | Параметры | Описание |
|-----------|-----------|----------|
| `rag_get_related` | `node_id`, `max_depth=1` | BFS-обход от узла. |

### Статистика

| Инструмент | Параметры | Описание |
|-----------|-----------|----------|
| `rag_stats` | — | `{total_documents, store_path, dimension}`. |
| `rag_graph_stats` | — | `{total_nodes, total_edges, relation_types}`. |

---

## 🧠 Архитектура

```
                    ┌──────────────────────┐
                    │    MCP Client         │
                    │  (Claude, Cline...)   │
                    └──────────┬───────────┘
                               │  JSON-RPC (stdin/stdout)
                    ┌──────────▼───────────┐
                    │   src/mcp_server.py   │
                    └──────────┬───────────┘
                               │
          ┌────────────────────┼────────────────────┐
          │                    │                     │
          ▼                    ▼                     ▼
   ┌──────────────┐    ┌──────────────┐    ┌──────────────────┐
   │ Semantic     │    │  BM25        │    │  GraphKnowledge  │
   │ Search       │    │  Search      │    │  Base            │
   │ (OpenRouter) │    │ (rank_bm25)  │    │  (реляции)       │
   └──────┬───────┘    └──────┬───────┘    └────────┬─────────┘
          │                   │                      │
          └───────────────────┼──────────────────────┘
                              │
                     ┌────────▼────────┐
                     │   Hybrid Search  │
                     │  (alpha blend)   │
                     └─────────────────┘
```

### Компоненты

| Компонент | Файл | Технология |
|-----------|------|-----------|
| Эмбеддинги | `src/embeddings.py` | OpenRouter API (fallback: TF-IDF) |
| Векторное хранилище | `src/vector_store.py` | ChromaDB (persistent) |
| BM25 индекс | `src/bm25_index.py` | rank_bm25 (Okapi BM25) |
| Графовая БЗ | `src/graph_store.py` | собственный in-memory граф + BFS |
| Оркестратор | `src/rag.py` | объединяет всё + гибридный поиск |
| MCP сервер | `src/mcp_server.py` | JSON-RPC 2.0 через stdio |
| CLI | `src/cli.py` | argparse |

---

## 🔍 Как работает граф

Документы становятся **узлами** графа. Между ними можно задавать **отношения**:

```python
from src.rag import RAGSystem

rag = RAGSystem()
doc1 = rag.add_document("Django — веб-фреймворк на Python")
doc2 = rag.add_document("Flask — лёгкий веб-фреймворк")

# Связываем
rag.add_relation(doc1, doc2, "similar_to")
rag.add_relation(doc1, doc2, "competitor")

# Поиск найдёт связанные документы автоматически
results = rag.search("веб-фреймворк", k=1)
# К результату добавится doc2 через отношение "similar_to"
```

**Поиск с expansion** — когда находится документ, система проверяет его связи в графе и добавляет связанные узлы в результаты (с пониженным скоринговым весом).

---

## 🧪 Тесты

```bash
# Полный набор
python3 -m pytest tests/ -v          # 267 тестов, все зелёные

# Качество поиска (NDCG, релевантность, alpha-калибровка)
python3 -m pytest tests/test_search_quality.py -v   # 27 тестов

# Только MCP-слой
python3 -m pytest tests/test_mcp_server.py -v
```

### Бенчмарк выбора alpha

```bash
# Калибровка default_alpha по NDCG@k на детерминированном корпусе
python3 -m scripts.benchmark_alpha
```

Скрипт прогоняет `search_hybrid` по сетке alpha ∈ [0.0, 1.0] (шаг 0.05) на
корпусе с настоящей семантической структурой (`tests/semantic_mock.py`), считает
NDCG@5 / P@5 / P@1 и печатает таблицу. Среди alpha в пределах 1% от лучшего NDCG
(«хорошая область») берётся значение, ближайшее к 0.5 — точке естественного
баланса каналов (робастный и детерминированный выбор). Результат должен совпадать
с `RAGConfig.default_alpha`; при расхождении — обновить конфиг.

---

## 📁 Структура проекта

```
graphrag/
├── src/
│   ├── bm25_index.py       # BM25 индекс
│   ├── cli.py              # CLI интерфейс
│   ├── config.py           # RAGConfig (из env)
│   ├── embeddings.py       # Эмбеддинги: Ollama / OpenRouter
│   ├── graph_store.py      # Графовая база знаний
│   ├── index.py            # (устаревший)
│   ├── mcp_server.py       # MCP сервер (JSON-RPC)
│   ├── rag.py              # Оркестратор RAG + гибридный поиск
│   ├── vector_store.py     # ChromaDB обёртка
│   └── __init__.py
├── scripts/
│   ├── __init__.py
│   └── benchmark_alpha.py # Бенчмарк NDCG@k для выбора default_alpha
├── tests/
│   ├── semantic_mock.py    # Детерминированный семантический mock-генератор + корпус
│   ├── test_search_quality.py # 27 тестов качества поиска (NDCG, alpha, релевантность)
│   ├── test_bm25.py        # BM25 тесты
│   ├── test_cli.py         # CLI тесты
│   ├── test_embeddings.py  # TF-IDF эмбеддинги
│   ├── test_embeddings_v2.py # OpenRouter/Ollama эмбеддинги
│   ├── test_graph_store.py # Граф тесты
│   ├── test_graph_persistence.py
│   ├── test_bm25_persistence.py
│   ├── test_dimension_mismatch.py
│   ├── test_mcp_server.py  # MCP тесты
│   ├── test_rag.py         # RAG-оркестратор тесты
│   ├── test_rag_v2.py      # RAG с моками
│   ├── test_vector_store.py # VectorStore тесты
│   ├── test_config.py      # RAGConfig тесты
│   ├── test_index.py       # (устаревший)
│   └── __init__.py
├── specifications/
│   ├── api.md              # API спецификация
│   ├── architecture.md     # Архитектура
│   └── cli.md              # CLI спецификация
├── rag_data/               # ChromaDB на диске (.gitignored)
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 🔐 Переменные окружения

| Переменная | Описание | По умолчанию |
|-----------|----------|-------------|
| `EMBEDDING_PROVIDER` | Провайдер эмбеддингов: `ollama` или `openrouter` | `ollama` |
| `OLLAMA_BASE_URL` | URL сервера Ollama | `http://localhost:11434` |
| `OLLAMA_MODEL` | Модель эмбеддингов Ollama | `qwen3-embedding:8b` |
| `OLLAMA_DIMENSION` | Размерность эмбеддингов | `4096` |
| `OPENROUTER_API_KEY` | API ключ OpenRouter | — |
| `OPENROUTER_MODEL` | Модель эмбеддингов OpenRouter | `openai/text-embedding-3-small` |
| `STORE_PATH` | Путь к хранилищу | `./rag_data` |
| `RAG_DEFAULT_ALPHA` | Баланс гибридного поиска (0=BM25, 1=семантика) — выбран бенчмарком NDCG@k | `0.5` |
| `RAG_HYBRID_EXPAND` | Candidate expansion: `max(k * EXPAND, MIN)` кандидатов из каждого канала | `3` |
| `RAG_HYBRID_MIN_CANDIDATES` | Минимум кандидатов из каждого канала при fusion | `20` |

> **Ollama** — работает сразу после установки (ollama pull qwen3-embedding).  
> **OpenRouter** — требуется API ключ. Переключиться: `EMBEDDING_PROVIDER=openrouter`

### Быстрый старт с Ollama

```bash
# 1. Установите Ollama: https://ollama.com/
# 2. Скачайте модель эмбеддингов:
ollama pull qwen3-embedding

# 3. Всё! Система использует Ollama по умолчанию
python -m src.cli add-document --text "Hello world"
python -m src.cli search --query "hello"
```

### Переключение на OpenRouter

```bash
export EMBEDDING_PROVIDER=openrouter
export OPENROUTER_API_KEY=sk-or-v1-...
```

---

## 💾 Персистентность данных

Все данные сохраняются на диск автоматически:

| Компонент | Путь | Формат |
|-----------|------|--------|
| Векторное хранилище | `rag_data/` | ChromaDB (SQLite + binary) |
| BM25 индекс | `rag_data/bm25_index.pkl` | Pickle |
| Граф знаний | `rag_data/graph_store.pkl` | Pickle |

**Важно:**
- Папка `rag_data/` уже добавлена в `.gitignore`
- Данные сохраняются автоматически при каждом изменении
- При перезапуске приложения данные загружаются из файлов

**Управление данными:**
```bash
# Очистить всё
python -m src.cli clear

# Удалить вручную
rm -rf rag_data/
```

---

## 🔧 Настройка OpenRouter API

1. Получите API ключ на https://openrouter.ai/
2. Создайте файл `.env` на основе `.env.example`:
   ```bash
   cp .env.example .env
   ```
3. Отредактируйте `.env` и вставьте ваш ключ:
   ```
   OPENROUTER_API_KEY=sk-or-v1-xxxxxxxxxxxxxxxxxxxx
   ```
4. Или установите через переменную окружения:
   ```bash
   export OPENROUTER_API_KEY=sk-or-v1-xxxxxxxxxxxxxxxxxxxx
   ```

**Без API ключа:**
- ✅ BM25 поиск работает
- ✅ Графовые функции работают
- ❌ Семантический поиск недоступен (используется TF-IDF fallback)

---

## 🤖 MCP сервер интеграция

Подключите RAG систему как MCP сервер к любым MCP-клиентам:

**Claude Desktop / Cline:**
```json
{
  "mcpServers": {
    "rag-knowledge-base": {
      "command": "python",
      "args": ["-m", "src.mcp_server"],
      "env": {
        "EMBEDDING_PROVIDER": "ollama",
        "OLLAMA_BASE_URL": "http://localhost:11434",
        "OLLAMA_MODEL": "qwen3-embedding"
      }
    }
  }
}
```

Для OpenRouter:
```json
{
  "mcpServers": {
    "rag-knowledge-base": {
      "command": "python",
      "args": ["-m", "src.mcp_server"],
      "env": {
        "EMBEDDING_PROVIDER": "openrouter",
        "OPENROUTER_API_KEY": "${OPENROUTER_API_KEY}"
      }
    }
  }
}
```

**Запуск standalone:**
```bash
python -m src.mcp_server
```

**Доступные инструменты:**
- `rag_add_document`, `rag_add_file` — индексация
- `rag_search`, `rag_bm25_search`, `rag_search_hybrid` — поиск
- `rag_add_relation`, `rag_get_related`, `rag_graph_stats` — граф
- `rag_stats`, `rag_clear` — управление

---

## 📈 Roadmap

- [x] Семантический поиск (OpenRouter embeddings)
- [x] BM25 поиск
- [x] Гибридный поиск (alpha blending)
- [x] Графовая база знаний с реляциями
- [x] BFS обход графа с глубиной
- [x] MCP сервер + CLI
- [x] Персистентность данных (BM25 + Graph на диск)
- [x] Ollama эмбеддинги (локально, по умолчанию)
- [x] OpenRouter эмбеддинги (внешние, опционально)
- [x] Конфиг через переменные окружения
- [x] Улучшенная нормализация семантических скоров (L2 → косинусная сходность, [0,1])
- [x] Candidate expansion в гибридном поиске (max(k*3, 20) кандидатов)
- [x] Параметризуемый default_alpha через RAGConfig (env RAG_DEFAULT_ALPHA)
- [x] Бенчмарк NDCG@k для калибровки alpha (`scripts/benchmark_alpha.py`)
- [x] Тесты качества поиска на детерминированном корпусе (`tests/test_search_quality.py`)
- [ ] Фильтрация по метаданным в графе
- [ ] Визуализация графа