"""Тесты для модуля SemanticIndex (семантический индекс/поиск)"""

import pytest
import numpy as np
from src.index import SemanticIndex


class TestSemanticIndex:
    """Класс тестов для семантического индекса."""

    @pytest.fixture
    def index(self):
        """Фикстура: пустой индекс."""
        return SemanticIndex()

    def test_add_and_search(self, index):
        """Должен добавить вектор и найти его по похожему запросу."""
        vec = np.array([1.0, 0.0, 0.0])
        index.add("doc1", vec)
        query = np.array([0.9, 0.1, 0.0])
        results = index.search(query, top_k=1)
        assert len(results) == 1
        assert results[0][0] == "doc1"

    def test_search_returns_sorted_by_similarity(self, index):
        """Результаты поиска должны быть отсортированы по убыванию схожести."""
        index.add("doc1", np.array([1.0, 0.0]))
        index.add("doc2", np.array([0.0, 1.0]))
        index.add("doc3", np.array([0.7, 0.3]))
        query = np.array([1.0, 0.0])
        results = index.search(query, top_k=3)
        # doc1 самый похожий, doc3 второе место, doc2 наименее похож
        assert results[0][0] == "doc1"
        assert results[2][0] == "doc2"
        # Схожести должны убывать
        scores = [r[1] for r in results]
        for i in range(len(scores) - 1):
            assert scores[i] >= scores[i + 1], "Схожести должны убывать"

    def test_top_k_respects_limit(self, index):
        """Параметр top_k должен ограничивать количество результатов."""
        for i in range(10):
            vec = np.array([float(i), 0.0])
            index.add(f"doc{i}", vec)
        query = np.array([5.0, 0.0])
        results = index.search(query, top_k=3)
        assert len(results) == 3

    def test_search_in_empty_index(self, index):
        """Поиск в пустом индексе должен возвращать пустой список."""
        query = np.array([1.0, 2.0])
        results = index.search(query, top_k=5)
        assert results == []

    def test_cosine_similarity_identical(self, index):
        """Идентичные векторы должны иметь косинусную близость 1.0."""
        vec = np.array([3.0, 4.0])
        index.add("doc1", vec)
        results = index.search(vec, top_k=1)
        assert abs(results[0][1] - 1.0) < 1e-6

    def test_cosine_similarity_orthogonal(self, index):
        """Ортогональные векторы должны иметь близость 0."""
        index.add("doc1", np.array([1.0, 0.0]))
        query = np.array([0.0, 1.0])
        results = index.search(query, top_k=1)
        assert abs(results[0][1]) < 1e-6

    def test_add_multiple_documents(self, index):
        """Должен корректно хранить множество документов с метаданными."""
        docs = [
            ("doc_a", np.array([1.0, 0.0])),
            ("doc_b", np.array([0.0, 1.0])),
            ("doc_c", np.array([1.0, 1.0])),
        ]
        for doc_id, vec in docs:
            index.add(doc_id, vec, metadata={"source": doc_id})
        query = np.array([1.0, 0.0])
        results = index.search(query, top_k=3)
        assert len(results) == 3
        # Проверяем метаданные у лучшего результата
        best_doc_id, _, meta = results[0]
        assert meta["source"] == best_doc_id

    def test_update_vector(self, index):
        """Должен обновлять вектор существующего документа."""
        index.add("doc1", np.array([1.0, 0.0]))
        index.add("doc1", np.array([0.0, 1.0]))  # обновление
        query = np.array([0.0, 1.0])
        results = index.search(query, top_k=1)
        assert results[0][0] == "doc1"
        assert abs(results[0][1] - 1.0) < 1e-6
