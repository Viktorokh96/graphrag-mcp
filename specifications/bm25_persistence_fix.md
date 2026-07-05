# Спецификация: Персистентность BM25 индекса

## Проблема
BM25Index — полностью in-memory. При перезапуске CLI или MCP-сервера BM25 индекс пуст,
хотя ChromaDB (векторное хранилище) сохраняет данные на диск.

**Симптомы:**
- `python -m src.cli bm25-search --query "python"` → "результатов нет"
- `python -m src.cli hybrid-search --query "python"` → "результатов нет"
- Семантический поиск работает (данные в ChromaDB есть)
- Тесты (109 шт.) проходят — они используют свежие инстансы

## Требования

### 1. Сохранение BM25 индекса на диск
- При `add_document` / `add_documents` BM25 индекс должен сохраняться на диск
- При инициализации `RAGSystem` BM25 индекс должен загружаться с диска
- Формат хранения: JSON (простой, человекочитаемый) или pickle
- Путь хранения: `{store_path}/bm25_index.json` или `{store_path}/bm25_index.pkl`

### 2. Восстановление BM25 при старте
- `BM25Index.__init__()` принимает опциональный `store_path`
- Если файл существует — загружает данные и перестраивает BM25Okapi
- Если файла нет — создаёт пустой индекс

### 3. Токенизация
- Заменить `text.split()` на `re.findall(r'\b\w+\b', text.lower())`
- Это улучшит качество поиска (убирает пунктуацию, приводит к нижнему регистру)

### 4. debug_bm25.py
- Исправить ошибку: `index._documents` — список, у него нет `.values()`
- Скрипт должен корректно демонстрировать работу BM25

### 5. Совместимость
- Все существующие тесты (109 шт.) должны проходить
- CLI должен работать: add-document → bm25-search после перезапуска

## API изменения

### BM25Index
```python
class BM25Index:
    def __init__(self, store_path: Optional[str] = None):
        ...
    
    def save(self) -> None:
        """Сохранить индекс на диск."""
    
    def load(self) -> bool:
        """Загрузить индекс с диска. Возвращает True если успешно."""
```

### RAGSystem
```python
class RAGSystem:
    def __init__(self, store_path: str = "./rag_data", ...):
        # BM25Index получает store_path
        self.bm25_index = BM25Index(store_path=store_path)
        # При инициализации BM25 загружается с диска
```

## Файлы для изменения
- `src/bm25_index.py` — добавить save/load, store_path, улучшить токенизацию
- `src/rag.py` — передавать store_path в BM25Index
- `debug_bm25.py` — исправить ошибку
