"""Тесты для BM25Index."""

import pytest


class TestBM25Index:
    """Тесты для BM25 индекса."""

    def test_import(self):
        """Модуль импортируется без ошибок."""
        from src.bm25_index import BM25Index
        assert BM25Index is not None

    def test_init_empty(self):
        """Пустой индекс создаётся без ошибок."""
        from src.bm25_index import BM25Index
        index = BM25Index()
        assert index is not None

    def test_add_and_search(self):
        """После добавления документа поиск возвращает его."""
        from src.bm25_index import BM25Index

        index = BM25Index()
        index.add_document("doc1", "python programming language")
        results = index.search("python", k=1)

        assert len(results) == 1
        assert results[0][0] == "doc1"
        assert "python" in results[0][1].lower()

    def test_search_returns_sorted(self):
        """Результаты поиска отсортированы по убыванию релевантности."""
        from src.bm25_index import BM25Index

        index = BM25Index()
        index.add_document("doc1", "python programming")
        index.add_document("doc2", "java programming")
        index.add_document("doc3", "python python python")

        results = index.search("python", k=3)
        scores = [r[2] for r in results]
        for i in range(len(scores) - 1):
            assert scores[i] >= scores[i + 1], "Оценки должны убывать"

    def test_top_k_limit(self):
        """Параметр k ограничивает количество результатов."""
        from src.bm25_index import BM25Index

        index = BM25Index()
        for i in range(10):
            index.add_document(f"doc{i}", f"text number {i} python")

        results = index.search("python", k=3)
        assert len(results) == 3

    def test_search_empty_index(self):
        """Поиск в пустом индексе возвращает пустой список."""
        from src.bm25_index import BM25Index

        index = BM25Index()
        results = index.search("anything", k=5)
        assert results == []

    def test_clear(self):
        """Очистка индекса удаляет все документы."""
        from src.bm25_index import BM25Index

        index = BM25Index()
        index.add_document("doc1", "some text")
        index.clear()
        results = index.search("text", k=5)
        assert results == []

    def test_stats(self):
        """stats() возвращает корректную статистику."""
        from src.bm25_index import BM25Index

        index = BM25Index()
        assert index.stats()["total_documents"] == 0

        index.add_document("doc1", "hello world")
        stats = index.stats()
        assert stats["total_documents"] == 1
        assert stats["total_tokens"] > 0

        index.add_document("doc2", "another document with more text")
        stats = index.stats()
        assert stats["total_documents"] == 2

    def test_add_documents_batch(self):
        """add_documents принимает словарь и метаданные."""
        from src.bm25_index import BM25Index

        index = BM25Index()
        docs = {
            "doc1": "python is great",
            "doc2": "java python is also good",
        }
        meta = {
            "doc1": {"lang": "python"},
            "doc2": {"lang": "java"},
        }
        index.add_documents(docs, meta)

        results = index.search("python", k=5)
        assert len(results) == 2

        # Проверяем метаданные
        for doc_id, text, score, metadata in results:
            assert "lang" in metadata

    def test_remove_document(self):
        """Удаление документа из индекса."""
        from src.bm25_index import BM25Index

        index = BM25Index()
        index.add_document("doc1", "python programming")
        index.add_document("doc2", "java programming")

        index.remove("doc1")
        results = index.search("python", k=5)
        # doc1 должен быть удалён, doc2 может быть в результатах (содержит "programming")
        # но doc1 точно не должен быть
        doc_ids = [r[0] for r in results]
        assert "doc1" not in doc_ids

        # java всё ещё есть
        results = index.search("java", k=5)
        assert len(results) == 1
        assert results[0][0] == "doc2"

    def test_metadata_preserved(self):
        """Метаданные возвращаются при поиске."""
        from src.bm25_index import BM25Index

        index = BM25Index()
        index.add_document("doc1", "python text", metadata={"source": "test", "page": 1})
        results = index.search("python", k=1)

        assert len(results) == 1
        doc_id, text, score, meta = results[0]
        assert meta["source"] == "test"
        assert meta["page"] == 1