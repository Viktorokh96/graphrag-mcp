# CLI для RAG MCP Tool

## Назначение
Консольная утилита для ручной работы с RAG системой (без MCP сервера).

## Использование
```bash
# Базовые опции:
python -m src.cli --store ./my_rag --key openrouter-api-key <command> [args]

# Если не указывать --key, читает из OPENROUTER_API_KEY
```

## Команды

### 1. add-document
```bash
python -m src.cli add-document --text "Текст документа" [--meta '{"source":"file"}']
```

### 2. add-file
```bash
python -m src.cli add-file --path /path/to/file.txt [--meta '{"type":"report"}']
```

### 3. search — семантический поиск
```bash
python -m src.cli search --query "что-то про python" [--k 5]
```

### 4. bm25-search — поиск по ключевым словам
```bash
python -m src.cli bm25-search --query "python" [--k 5]
```

### 5. hybrid-search — гибридный поиск
```bash
python -m src.cli hybrid-search --query "python" [--k 5] [--alpha 0.5]
```

### 6. stats — статистика
```bash
python -m src.cli stats
```

### 7. clear — очистить всё
```bash
python -m src.cli clear
```

## Архитектура
Файл: `src/cli.py`
- Использует `argparse` (из коробки)
- Создаёт `RAGSystem` и вызывает его методы
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
