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

# Гибридный поиск (alpha=0.5 — баланс семантики и ключевых слов)
python -m src.cli hybrid-search --query "Python" --alpha 0.5

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

### Основные

| Инструмент | Параметры | Описание |
|-----------|-----------|----------|
| `rag_add_document` | `text` (str), `meta` (object?) | Добавить документ в БЗ |
| `rag_add_file` | `filepath` (str), `meta` (object?) | Прочитать файл и проиндексировать |
| `rag_search` | `query` (str), `k` (int=5) | Семантический поиск (эмбеддинги) |
| `rag_bm25_search` | `query` (str), `k` (int=5) | Поиск по ключевым словам (BM25) |
| `rag_search_hybrid` | `query` (str), `k` (int=5), `alpha` (float=0.5) | Гибрид: semantic + BM25 |
| `rag_stats` | — | Статистика хранилища |
| `rag_clear` | — | Очистить всё |

### Графовые

| Инструмент | Параметры | Описание |
|-----------|-----------|----------|
| `rag_add_relation` | `source_id` (str), `target_id` (str), `relation` (str), `weight` (float=1.0) | Добавить отношение между документами |
| `rag_get_related` | `node_id` (str), `max_depth` (int=1) | Получить связанные документы (BFS) |
| `rag_graph_stats` | — | Статистика графа (узлы, рёбра, типы отношений) |

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
pytest           # 109 тестов, все зелёные
pytest -v        # подробно
pytest tests/test_graph_store.py -v   # только тесты графа
```

---

## 📁 Структура проекта

```
testproject_agents/
├── src/
│   ├── bm25_index.py       # BM25 индекс
│   ├── cli.py              # CLI интерфейс
│   ├── embeddings.py       # Генератор эмбеддингов
│   ├── graph_store.py      # Графовая база знаний 🔥
│   ├── index.py            # Семантический индекс (устаревший)
│   ├── mcp_server.py       # MCP сервер (JSON-RPC)
│   ├── rag.py              # Оркестратор RAG
│   ├── vector_store.py     # ChromaDB обёртка
│   └── __init__.py
├── tests/
│   ├── test_bm25.py        # 10 тестов BM25
│   ├── test_cli.py         # 12 тестов CLI
│   ├── test_embeddings.py  # 7 тестов эмбеддингов (TF-IDF)
│   ├── test_embeddings_v2.py # 5 тестов (OpenRouter)
│   ├── test_graph_store.py # 13 тестов графа 🔥
│   ├── test_index.py       # 10 тестов (устаревший)
│   ├── test_mcp_server.py  # 14 тестов MCP
│   ├── test_rag.py         # 8 тестов RAG
│   ├── test_rag_v2.py      # 15 тестов RAG (моки)
│   ├── test_vector_store.py # 8 тестов VectorStore
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
| `OLLAMA_DIMENSION` | Размерность эмбеддингов | `1024` |
| `OPENROUTER_API_KEY` | API ключ OpenRouter | — |
| `OPENROUTER_MODEL` | Модель эмбеддингов OpenRouter | `openai/text-embedding-3-small` |

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
- [ ] Фильтрация по метаданным в графе
- [ ] Визуализация графа