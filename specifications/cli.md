# CLI для RAG MCP Tool

## Назначение
Консольная утилита для ручной работы с RAG системой (без MCP/HTTP сервера).

## Использование
```bash
# Базовые опции:
python -m src.cli [--store ./rag_data] [--http] [--port 8765] <command> [args]
```

## Глобальные флаги
| Флаг | Описание |
|------|----------|
| `--store PATH` | Путь к хранилищу (по умолч. `./rag_data`) |
| `--http` | Запустить HTTP сервер (также `server`, `serve`, `s`) |
| `--port PORT` | Порт HTTP сервера (по умолч. `8765`, с `--http`) |
| `--rerank` | Включить CrossEncoder reranking |
| `--query-expansion` | Включить LLM query expansion |

## Команды

### 1. add-document
```bash
python -m src.cli add-document --text "Текст" [--meta '{"source":"file"}'] [--extract-graph]
```

### 2. add-file
```bash
python -m src.cli add-file --path /path/to/file.txt [--meta '{"type":"report"}'] [--extract-graph]
```

### 3. add-structured
```bash
python -m src.cli add-structured --path /path/to/repomix.json [--meta '{"source":"repo"}']
```

### 4. search — семантический поиск
```bash
python -m src.cli search --query "что-то про python" [--k 5] [--rerank] [--query-expansion]
```

### 5. bm25-search — поиск по ключевым словам (sparse vectors)
```bash
python -m src.cli bm25-search --query "python" [--k 5]
```

### 6. hybrid-search — гибридный поиск
```bash
python -m src.cli hybrid-search --query "python" [--k 5] [--alpha 0.5] [--rerank] [--query-expansion]
```

### 7. list-documents — список документов
```bash
python -m src.cli list-documents [--limit 20] [--offset 0]
```

### 8. get-document — получить документ по ID
```bash
python -m src.cli get-document --doc-id <uuid>
```

### 9. add-relation — создать ребро графа
```bash
python -m src.cli add-relation --source-id <uuid> --target-id <uuid> --relation "uses"
```

### 10. get-related — обход графа
```bash
python -m src.cli get-related --node-id <uuid> [--max-depth 2]
```

### 11. stats — статистика
```bash
python -m src.cli stats
```

### 12. clear — очистить всё
```bash
python -m src.cli clear
```

## Архитектура
Файл: `src/cli.py`
- Использует `argparse` (из коробки)
- Создаёт `RAGSystem` и вызывает его методы
- Режим `--http`: запускает HttpAPI через uvicorn
- Выводит результаты в читаемом виде (с разделителями)
- При ошибках выводит stderr и exit code 1

## Пример работы
```bash
$ python -m src.cli add-document --text "Python крутой язык"
✅ Добавлен документ: a1b2c3d4

$ python -m src.cli search --query "язык"
┌─────────────────────────────────────────┐
│ Результаты поиска (k=5):                │
├─────────────────────────────────────────┤
│ 1. Python крутой язык [0.92]            │
└─────────────────────────────────────────┘
```
