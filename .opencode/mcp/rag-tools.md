# RAG MCP Server Tools

Доступные инструменты для агентов opencode через MCP сервер.

## Поиск

### `rag_search` — Семантический поиск
Поиск по смыслу запроса с помощью векторных эмбеддингов.

```json
{
  "query": "запрос для поиска",
  "k": 5,
  "filter": {}
}
```

**Пример:**
```python
# Поиск документов о Python
rag_search(query="Python программирование", k=3)
```

---

### `rag_bm25_search` — Поиск по ключевым словам
Точный поиск по ключевым словам с использованием BM25 алгоритма.

```json
{
  "query": "ключевые слова",
  "k": 5
}
```

**Пример:**
```python
# Поиск документов с точными словами
rag_bm25_search(query="machine learning алгоритмы", k=3)
```

---

### `rag_search_hybrid` — Гибридный поиск
Комбинация семантического и BM25 поиска для лучших результатов.

```json
{
  "query": "запрос",
  "k": 5,
  "alpha": 0.5  # 0.0 = только BM25, 1.0 = только semantic
}
```

**Пример:**
```python
# Гибридный поиск (рекомендуется)
rag_search_hybrid(query="RAG системы", k=5, alpha=0.5)
```

---

## Управление документами

### `rag_add_document` — Добавить документ
Добавить текст в базу знаний.

```json
{
  "text": "текст документа",
  "meta": {"source": "источник", "date": "2024-01-01"}
}
```

**Пример:**
```python
rag_add_document(
  text="Python - это язык программирования",
  meta={"source": "user_input"}
)
```

---

### `rag_add_file` — Добавить файл
Добавить содержимое файла в базу знаний.

```json
{
  "filepath": "/path/to/file.txt",
  "meta": {"source": "file"}
}
```

**Пример:**
```python
rag_add_file(filepath="./docs/readme.md", meta={"source": "documentation"})
```

---

## Граф знаний

### `rag_add_relation` — Добавить отношение
Создать связь между двумя документами.

```json
{
  "source_id": "doc_id_1",
  "target_id": "doc_id_2",
  "relation": "related_to",
  "weight": 1.0
}
```

**Пример:**
```python
rag_add_relation(
  source_id="5c73e7d0-c98d-47c8-b2f6-5c456a49a6b9",
  target_id="fa403834-0b5b-41ed-a1cc-f56141b61c71",
  relation="related_to",
  weight=0.8
)
```

---

### `rag_get_related` — Получить связанные документы
Найти документы, связанные с указанным.

```json
{
  "node_id": "doc_id",
  "max_depth": 2
}
```

**Пример:**
```python
rag_get_related(node_id="5c73e7d0-c98d-47c8-b2f6-5c456a49a6b9", max_depth=2)
```

---

## Статистика

### `rag_stats` — Статистика базы знаний
Получить общую статистику RAG системы.

```json
{}
```

**Пример:**
```python
stats = rag_stats()
# Возвращает: total_documents, store_path, dimension
```

---

### `rag_graph_stats` — Статистика графа
Получить статистику графа знаний.

```json
{}
```

**Пример:**
```python
graph_stats = rag_graph_stats()
# Возвращает: total_nodes, total_edges, relation_types
```

---

## Очистка

### `rag_clear` — Очистить всё
Удалить все документы, векторы и граф.

```json
{}
```

**Внимание:** Опасная операция! Удаляет все данные безвозвратно.

---

## Рекомендации по использованию

1. **Для точного поиска** используйте `rag_bm25_search`
2. **Для семантического поиска** используйте `rag_search`
3. **Для лучших результатов** используйте `rag_search_hybrid` с `alpha=0.5`
4. **Для навигации по знаниям** используйте `rag_get_related`
5. **Перед очисткой** убедитесь что данные не нужны

## Пример workflow

```python
# 1. Добавить документ
doc_id = rag_add_document(text="Машинное обучение — это...", meta={"source": "wiki"})

# 2. Поиск информации
results = rag_search_hybrid(query="machine learning", k=5, alpha=0.5)

# 3. Найти связанные документы
related = rag_get_related(node_id=doc_id, max_depth=1)

# 4. Проверить статистику
stats = rag_stats()
```
