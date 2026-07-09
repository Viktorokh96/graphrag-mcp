# CLI Reference

Сервер: `src/cli.py`. Точка входа: `rag-server` (console_script) или `python -m src.cli`.

## Глобальные флаги

| Флаг | Описание | Дефолт |
|------|----------|--------|
| `--store PATH` | Путь к хранилищу | `./rag_data` |
| `--key KEY` | OpenRouter API ключ | — |
| `--provider {bge-m3,ollama,openrouter}` | Провайдер эмбеддингов | из env |
| `--ollama-url URL` | Ollama base URL | из env |
| `--ollama-model MODEL` | Ollama модель | из env |
| `--http` | Запустить HTTP REST API + MCP SSE | — |
| `--port PORT` | HTTP порт (с --http) | 8765 |

## Команды

### `add-document`

```bash
rag-server add-document --text "content" [--meta '{"key":"val"}']
```

Параметры:
- `--text` (обяз.) — текст документа
- `--meta` — JSON-строка метаданных

### `add-file`

```bash
rag-server add-file --path /path/to/file [--meta '{"key":"val"}']
```

Параметры:
- `--path` (обяз.) — путь к файлу
- `--meta` — JSON-строка метаданных

### `add-structured`

```bash
rag-server add-structured --content '{"repository":"...","files":{...}}' [--extract-graph]
```

Параметры:
- `--content` (обяз.) — JSON-строка в repomix формате
- `--extract-graph` — извлекать граф из файлов

### `add-relation`

```bash
rag-server add-relation --source <uuid> --target <uuid> --relation related_to [--weight 1.0]
```

Параметры:
- `--source` (обяз.) — doc_id источника
- `--target` (обяз.) — doc_id цели
- `--relation` (обяз.) — тип связи
- `--weight` — вес ребра (default 1.0)

### `search`

```bash
rag-server search --query "text" [--k 5] [--meta-filter '{}'] [--relations-load-depth 1]
```

Семантический поиск.

### `bm25-search`

```bash
rag-server bm25-search --query "keywords" [--k 5] [--meta-filter '{}']
```

BM25 keyword поиск.

### `hybrid-search`

```bash
rag-server hybrid-search --query "text" [--k 5] [--alpha 0.5] [--rerank] [--query-expansion] [--meta-filter '{}']
```

Гибридный поиск (RRF alpha-dilution).
- `--alpha` — баланс (0=BM25, 1=semantic; null=language-aware)
- `--rerank` — включить CrossEncoder reranking
- `--query-expansion` — включить multi-query expansion

### `list-documents`

```bash
rag-server list-documents [--limit 20] [--offset 0] [--max-chars N] [--meta-filter '{}']
```

Список документов с пагинацией.

### `get-document`

```bash
rag-server get-document --doc-id <uuid> [--offset 0] [--limit N] [--relations-load-depth 1]
```

Получить документ по ID.

### `get-related`

```bash
rag-server get-related --node <uuid> [--max-depth 1] [--meta-filter '{}']
```

BFS обход графа от узла.

### `stats`

```bash
rag-server stats
```

Статистика хранилища.

### `graph-stats`

```bash
rag-server graph-stats
```

Статистика графа.

### `clear`

```bash
rag-server clear
```

⚠️ Удалить все данные.

### `reindex`

```bash
rag-server reindex
```

Пересчитать эмбеддинги всех документов (после смены провайдера).

### `migrate`

```bash
rag-server migrate [--dry-run] [--force]
```

Миграция старых данных (ChromaDB + graph_index.json) → Qdrant + SQLite.

### `graph-viz`

```bash
rag-server graph-viz --output graph.html [--format html|dot|json|ascii] [--max-nodes N] [--relation-type TYPE] [--focus UUID] [--max-depth 2] [--layout kamada_kawai|spring|circular|hierarchical]
```

Визуализация графа знаний.

### `serve-graph`

```bash
rag-server serve-graph --output graph.html [--port 8090] [--max-nodes N] [--relation-type TYPE] [--focus UUID] [--max-depth 2] [--layout kamada_kawai] [--no-browser]
```

Генерация HTML + live RAG API сервер (запросы документа по клику).

### `--http`

```bash
rag-server --http [--port 8765]
```

Запуск HTTP REST API + MCP SSE транспорта.
