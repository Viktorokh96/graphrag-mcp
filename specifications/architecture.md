# Архитектура RAG MCP Tool

## Общая схема

```
                     ┌─────────────────────┐
                     │    MCP Client        │
                     │  (Claude, any MCP)   │
                     └──────────┬──────────┘
                                │ JSON-RPC (stdin/stdout)
                     ┌──────────▼──────────┐
                     │    MCP Server        │
                     │  (src/mcp_server.py) │
                     └──────────┬──────────┘
                                │
          ┌─────────────────────┼──────────────────────┐
          │                     │                      │
          ▼                     ▼                      ▼
   ┌──────────────┐    ┌──────────────┐    ┌──────────────────┐
   │ SemanticSearch│    │  BM25Search  │    │   VectorStore     │
   │ (OpenRouter) │    │ (rank_bm25) │    │   (ChromaDB)      │
   └──────────────┘    └──────────────┘    └──────────────────┘
          │                                       │
          └───────────────┬───────────────────────┘
                          ▼
                 ┌──────────────────┐
                 │  Hybrid Search   │
                 │  (RRF + alpha)   │
                 └──────────────────┘
```

## Компоненты

### 1. EmbeddingGenerator (`src/embeddings.py`)
- Вызывает OpenRouter API для получения семантических эмбеддингов
- Кеширует результаты в памяти
- Поддерживает fallback если API недоступен

### 2. BM25Index (`src/bm25_index.py`)
- Okapi BM25 для keyword-based поиска
- Использует `rank_bm25` библиотеку
- Не требует API — работает локально

### 3. VectorStore (`src/vector_store.py`)
- Обёртка над ChromaDB (persistent vector database)
- Хранит: текст, эмбеддинг, метаданные
- Встроенный косинусный поиск
- Фильтрация по метаданным
- Сохраняется на диск

### 4. RAGSystem (`src/rag.py`)
- Оркестратор, объединяющий все компоненты
- Hybrid search: Reciprocal Rank Fusion (RRF) с alpha-dilution
  - `score = alpha/(RRF_K+rank_sem+1) + (1-alpha)/(RRF_K+rank_bm25+1)`
  - RRF_K=20 (не 60 — для малых корпусов широкий разброс скоров)
  - Language-aware alpha: кириллица → 0.85, иначе 0.5
- Управляет индексацией и поиском

### 5. MCPServer (`src/mcp_server.py`)
- JSON-RPC сервер по протоколу stdio
- Обрабатывает MCP запросы от клиента

## MCP Инструменты (экспортируемые)

1. `rag_add_document(text, [store_path])` — добавить документ
2. `rag_search(query, [k=5], [store_path])` — семантический поиск
3. `rag_add_file(filepath, [store_path])` — прочитать файл и проиндексировать
4. `rag_stats([store_path])` — статистика хранилища
5. `rag_clear([store_path])` — очистить хранилище
6. `rag_bm25_search(query, [k=5], [store_path])` — BM25 поиск
7. `rag_search_hybrid(query, [k=5], [alpha=null], [store_path])` — гибридный поиск (RRF)

## Формат данных

### Документ
```python
{
    "id": str,        # UUID или хеш
    "text": str,      # Содержимое
    "metadata": dict,  # Произвольные метаданные
    "embedding": List[float],  # Вектор (семантический)
}
```

## Технологии
- **OpenRouter API** — семантические эмбеддинги (модель: `openai/text-embedding-3-small` или аналог)
- **ChromaDB** — векторная БД (persistent, Python-native)
- **rank_bm25** — BM25 алгоритм
- **numpy** — числовые операции
- **httpx** — HTTP клиент для OpenRouter
