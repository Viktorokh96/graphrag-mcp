# rag_update_document & rag_delete_relation

## Описание

Добавление операций редактирования и точечного удаления в MCP/HTTP API:
- `rag_update_document` — обновление текста и/или метаданных документа с сохранением doc_id и всех связей
- `rag_delete_relation` — удаление конкретного ребра графа

## Требования

1. **rag_update_document**:
   - Принимает `doc_id` (обяз.) + хотя бы один из `text` / `meta`
   - Обновляет текст → пересчитывает content_hash → ре-индексирует в Qdrant (dense + sparse)
   - Обновляет метаданные без ре-индексации векторов
   - Сохраняет doc_id и все рёбра графа (связи не ломаются)
   - Идемпотентен: повторный вызов с тем же текстом не создаёт дубликат

2. **rag_delete_relation**:
   - Принимает `source_id`, `target_id`, `relation` (все обяз.)
   - Удаляет конкретное ребро из SQL + NetworkX кеша
   - Идемпотентен: удаление несуществующего ребра безопасно

## API

### MCP Tools

#### `rag_update_document`

```json
{
  "name": "rag_update_document",
  "inputSchema": {
    "type": "object",
    "properties": {
      "doc_id": {"type": "string", "description": "doc_id документа для обновления"},
      "text": {"type": "string", "description": "Новый текст (опционально). Если передан, content_hash пересчитывается и векторы реиндексируются."},
      "meta": {
        "description": "Новые метаданные (опционально). Тот же формат что rag_add_document.meta. Перезаписывает целиком."
      }
    },
    "required": ["doc_id"]
  }
}
```

**Возврат:**
```json
{"doc_id": "uuid", "updated": true}
```

**Ошибки:**
- `ValueError` если doc_id не найден
- `ValueError` если переданы ни text, ни meta

#### `rag_delete_relation`

```json
{
  "name": "rag_delete_relation",
  "inputSchema": {
    "type": "object",
    "properties": {
      "source_id": {"type": "string", "description": "doc_id исходного узла"},
      "target_id": {"type": "string", "description": "doc_id целевого узла"},
      "relation": {"type": "string", "description": "Тип отношения (ключевое поле)"}
    },
    "required": ["source_id", "target_id", "relation"]
  }
}
```

**Возврат:**
```json
{"status": "ok", "deleted": true}
```

### HTTP REST API

| Метод | Эндпоинт | Тело | Описание |
|-------|----------|------|----------|
| PUT | `/documents/{doc_id}` | `{text?, meta?}` | Обновить документ |
| DELETE | `/relations` | `{source_id, target_id, relation}` | Удалить ребро |

## Implementation Details

### document_store.py

Добавить метод:

```python
def update_text(self, doc_id: str, text: str, content_hash: Optional[str] = None) -> bool:
    """Обновить текст и content_hash документа. Возвращает True если документ существовал."""
    existed = self.get(doc_id) is not None
    self._db.execute(
        "UPDATE documents SET text = ?, content_hash = ? WHERE doc_id = ?",
        (text, content_hash, doc_id),
    )
    return existed

def update_metadata(self, doc_id: str, metadata: dict) -> bool:
    """Обновить метаданные документа. Возвращает True если документ существовал."""
    existed = self.get(doc_id) is not None
    meta_json = json.dumps(metadata, ensure_ascii=False)
    self._db.execute(
        "UPDATE documents SET metadata = ? WHERE doc_id = ?",
        (meta_json, doc_id),
    )
    return existed
```

### vector_store.py

Добавить метод:

```python
def replace(self, doc_id: str, text: str, embedding: list[float], metadata: Optional[dict] = None) -> None:
    """Полная замена точек документа: удалить старые + добавить новые."""
    self.remove(doc_id)
    self.add(doc_id, text, embedding, metadata)
```

Метод `remove()` уже обрабатывает чанки (удаляет все точки с payload.doc_id == doc_id).

### rag.py

Добавить метод:

```python
def update_document(self, doc_id: str, text: Optional[str] = None, meta: Optional[dict] = None) -> dict:
    """Обновить документ. Возвращает {doc_id, updated: True}."""
    if text is None and meta is None:
        raise ValueError("At least one of 'text' or 'meta' must be provided")
    
    record = self.doc_store.get(doc_id)
    if record is None:
        raise ValueError(f"Document not found: {doc_id}")
    
    if text is not None:
        if len(text.strip()) < self.MIN_CONTENT_LENGTH:
            raise ValueError(f"Document too short ({len(text.strip())} chars)")
        content_hash = self._compute_hash(text)
        self.doc_store.update_text(doc_id, text, content_hash)
        self._index_vector(doc_id, text, meta or record["metadata"])
    elif meta is not None:
        # Только метаданные — обновляем без реиндексации векторов
        pass
    
    if meta is not None:
        self.doc_store.update_metadata(doc_id, meta)
    
    return {"doc_id": doc_id, "updated": True}

def delete_relation(self, source_id: str, target_id: str, relation: str) -> dict:
    """Удалить конкретное ребро. Идемпотентен."""
    self.graph_kb.remove_edge(source_id, target_id, relation)
    return {"status": "ok", "deleted": True}
```

### mcp_server.py

Добавить TOOL_DEFS и handlers для `rag_update_document` и `rag_delete_relation`.

### http_api.py

Добавить эндпоинты `PUT /documents/{doc_id}` и `DELETE /relations`.

## Связи с существующим кодом

- `graph_store.py:remove_edge()` — уже реализован, просто не экспортирован через MCP/HTTP
- `vector_store.py:remove()` — уже обрабатывает чанки
- `_index_vector()` — переиспользуется как есть для реиндексации
- `_parse_meta()` — переиспользуется для нормализации meta
