"""Регрессионные тесты для обработки несовпадения размерности эмбеддингов.

Сценарий: ChromaDB коллекция создана с размерностью A (например, 1536 от
OpenRouter text-embedding-3-small), но текущий embedding-провайдер выдаёт
размерность B (например, 4096 от Ollama qwen3-embedding:8b).  Без защиты это
приводит к падению ChromaDB с ошибкой:
    "Collection expecting embedding with dimension of X, got Y"

Тестируемые защитные механизмы:
1. VectorStore.get_dimension() — корректное чтение размерности коллекции
2. RAGSystem._ensure_dimension_compatibility() — авто-переиндексация при несовпадении
3. RAGSystem.search() — graceful возврат [] вместо падения
4. RAGSystem.search_hybrid() — fallback на BM25 при недоступности семантики
"""

import os
import shutil
import tempfile
from unittest.mock import MagicMock, patch

import pytest


class TestVectorStoreGetDimension:
    """Тесты для VectorStore.get_dimension()."""

    @pytest.fixture(autouse=True)
    def setup_temp_dir(self):
        self.temp_dir = tempfile.mkdtemp()
        self.store_path = os.path.join(self.temp_dir, "rag_data")
        yield
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def test_get_dimension_empty_store(self):
        """Пустое хранилище возвращает размерность 0."""
        from src.vector_store import VectorStore

        store = VectorStore(store_path=self.store_path)
        assert store.get_dimension() == 0

    def test_get_dimension_after_add(self):
        """После добавления вектора размерность соответствует."""
        from src.vector_store import VectorStore

        store = VectorStore(store_path=self.store_path)
        store.add("doc1", "text", [0.1, 0.2, 0.3])
        assert store.get_dimension() == 3

    def test_get_dimension_large_vector(self):
        """Работает с векторами большой размерности."""
        from src.vector_store import VectorStore

        store = VectorStore(store_path=self.store_path)
        store.add("doc1", "text", [0.1] * 1536)
        assert store.get_dimension() == 1536

    def test_get_dimension_after_recreate(self):
        """После recreate_collection размерность сбрасывается в 0."""
        from src.vector_store import VectorStore

        store = VectorStore(store_path=self.store_path)
        store.add("doc1", "text", [0.1, 0.2, 0.3])
        assert store.get_dimension() == 3

        store.recreate_collection()
        assert store.get_dimension() == 0

    def test_get_dimension_persistence_across_reload(self):
        """Размерность сохраняется между перезагрузками."""
        from src.vector_store import VectorStore

        store1 = VectorStore(store_path=self.store_path)
        store1.add("doc1", "text", [0.1, 0.2, 0.3, 0.4])

        store2 = VectorStore(store_path=self.store_path)
        assert store2.get_dimension() == 4

    def test_stats_uses_get_dimension(self):
        """stats() возвращает ту же размерность что и get_dimension()."""
        from src.vector_store import VectorStore

        store = VectorStore(store_path=self.store_path)
        store.add("doc1", "text", [0.1, 0.2])

        stats = store.stats()
        assert stats["dimension"] == store.get_dimension() == 2


class TestSearchDimensionMismatch:
    """Тесты для graceful обработки несовпадения размерности при поиске."""

    @pytest.fixture(autouse=True)
    def setup_temp_dir(self):
        self.temp_dir = tempfile.mkdtemp()
        self.store_path = os.path.join(self.temp_dir, "rag_data")
        yield
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def _make_mock_httpx(self, dim=3):
        """Создать мок httpx.Client, возвращающий эмбеддинги заданной размерности."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "data": [{"embedding": [0.1 * (i + 1) for i in range(dim)]}]
        }
        mock_client = MagicMock()
        mock_client.__enter__.return_value = mock_client
        mock_client.post.return_value = mock_response
        return mock_client

    @patch("src.embeddings.httpx.Client")
    def test_search_returns_empty_on_dimension_mismatch(self, mock_httpx):
        """search() возвращает [] вместо падения при несовпадении размерности."""
        from src.rag import RAGSystem

        mock_httpx.return_value = self._make_mock_httpx(dim=3)

        rag = RAGSystem(store_path=self.store_path, api_key="test-key")
        rag.add_document("test document about python programming language and scripting")

        # Меняем размерность генератора: store=3-dim, новый запрос=5-dim
        rag.embedding_generator._fallback_dimension = 5
        rag.embedding_generator.clear_cache()
        mock_httpx.return_value = self._make_mock_httpx(dim=5)

        results = rag.search("python")
        assert results == []

    @patch("src.embeddings.httpx.Client")
    def test_search_works_when_dimensions_match(self, mock_httpx):
        """search() работает корректно когда размерности совпадают."""
        from src.rag import RAGSystem

        mock_httpx.return_value = self._make_mock_httpx(dim=3)

        rag = RAGSystem(store_path=self.store_path, api_key="test-key")
        rag.add_document("python programming language for scripting and automation tasks")
        rag.add_document("java programming language for enterprise software development")

        results = rag.search("programming", k=2)
        assert len(results) == 2

    @patch("src.embeddings.httpx.Client")
    def test_bm25_search_unaffected_by_dimension_mismatch(self, mock_httpx):
        """BM25 поиск не зависит от размерности эмбеддингов."""
        from src.rag import RAGSystem

        mock_httpx.return_value = self._make_mock_httpx(dim=3)

        rag = RAGSystem(store_path=self.store_path, api_key="test-key")
        rag.add_document("python programming language for general purpose scripting and automation")

        # Меняем размерность — BM25 не должен пострадать
        rag.embedding_generator._fallback_dimension = 5
        rag.embedding_generator.clear_cache()
        mock_httpx.return_value = self._make_mock_httpx(dim=5)

        results = rag.bm25_search("python", k=1)
        assert len(results) == 1
        assert "python" in results[0][1].lower()

    @patch("src.embeddings.httpx.Client")
    def test_hybrid_search_falls_back_to_bm25_on_mismatch(self, mock_httpx):
        """Hybrid search использует BM25 при недоступности семантического поиска."""
        from src.rag import RAGSystem

        mock_httpx.return_value = self._make_mock_httpx(dim=3)

        rag = RAGSystem(store_path=self.store_path, api_key="test-key")
        rag.add_document("python programming language for scripting and automation tasks")
        rag.add_document("java programming language for enterprise software development")

        # Меняем размерность — семантика упадёт, BM25 должен спасти
        rag.embedding_generator._fallback_dimension = 5
        rag.embedding_generator.clear_cache()
        mock_httpx.return_value = self._make_mock_httpx(dim=5)

        results = rag.search_hybrid("python", k=2, alpha=0.5)
        assert len(results) >= 1
        assert "python" in results[0][1].lower()


class TestAutoReindexOnDimensionMismatch:
    """Тесты для автоматической переиндексации при инициализации."""

    @pytest.fixture(autouse=True)
    def setup_temp_dir(self):
        self.temp_dir = tempfile.mkdtemp()
        self.store_path = os.path.join(self.temp_dir, "rag_data")
        yield
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def _make_mock_httpx(self, dim=3):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "data": [{"embedding": [0.1 * (i + 1) for i in range(dim)]}]
        }
        mock_client = MagicMock()
        mock_client.__enter__.return_value = mock_client
        mock_client.post.return_value = mock_response
        return mock_client

    @patch("src.embeddings.httpx.Client")
    def test_auto_reindex_on_dimension_mismatch(self, mock_httpx):
        """RAGSystem.__init__ вызывает reindex при несовпадении размерности."""
        from src.rag import RAGSystem

        # Шаг 1: создаём store с 3-dim эмбеддингами
        mock_httpx.return_value = self._make_mock_httpx(dim=3)
        rag1 = RAGSystem(store_path=self.store_path, api_key="test-key")
        rag1.add_document("first document for testing dimension mismatch and reindex")
        rag1.add_document("second document for testing dimension mismatch and reindex")
        assert rag1.vector_store.get_dimension() == 3
        assert rag1.vector_store.count() == 2

        # Шаг 2: меняем размерность на 5
        mock_httpx.return_value = self._make_mock_httpx(dim=5)

        # Шаг 3: новый RAGSystem с тем же store — должен авто-переиндексировать
        rag2 = RAGSystem(store_path=self.store_path, api_key="test-key")
        assert rag2.vector_store.get_dimension() == 5
        assert rag2.vector_store.count() == 2

    @patch("src.embeddings.httpx.Client")
    def test_auto_reindex_preserves_document_count(self, mock_httpx):
        """Переиндексация сохраняет количество документов."""
        from src.rag import RAGSystem

        mock_httpx.return_value = self._make_mock_httpx(dim=3)
        rag1 = RAGSystem(store_path=self.store_path, api_key="test-key")
        for i in range(5):
            rag1.add_document(f"document number {i} for testing reindex and dimension changes")
        assert rag1.vector_store.count() == 5

        # Меняем размерность
        mock_httpx.return_value = self._make_mock_httpx(dim=8)
        rag2 = RAGSystem(store_path=self.store_path, api_key="test-key")

        assert rag2.vector_store.count() == 5
        assert rag2.vector_store.get_dimension() == 8

    @patch("src.embeddings.httpx.Client")
    def test_no_reindex_on_matching_dimension(self, mock_httpx):
        """При совпадении размерности reindex не вызывается."""
        from src.rag import RAGSystem

        mock_httpx.return_value = self._make_mock_httpx(dim=3)
        rag1 = RAGSystem(store_path=self.store_path, api_key="test-key")
        rag1.add_document("test document for checking no reindex on matching dimensions")
        assert rag1.vector_store.get_dimension() == 3

        # Создаём новый RAGSystem — размерности совпадают, reindex не нужен
        rag2 = RAGSystem(store_path=self.store_path, api_key="test-key")
        assert rag2.vector_store.get_dimension() == 3
        assert rag2.vector_store.count() == 1

    @patch("src.embeddings.httpx.Client")
    def test_no_reindex_on_empty_store(self, mock_httpx):
        """Пустое хранилище не требует переиндексации."""
        from src.rag import RAGSystem

        mock_httpx.return_value = self._make_mock_httpx(dim=3)
        rag = RAGSystem(store_path=self.store_path, api_key="test-key")
        assert rag.vector_store.count() == 0
        assert rag.vector_store.get_dimension() == 0

        # Не должно вызвать ошибку
        rag._ensure_dimension_compatibility()

    @patch("src.embeddings.httpx.Client")
    def test_search_works_after_auto_reindex(self, mock_httpx):
        """После авто-переиндексации поиск работает с новой размерностью."""
        from src.rag import RAGSystem

        # Создаём store с 3-dim
        mock_httpx.return_value = self._make_mock_httpx(dim=3)
        rag1 = RAGSystem(store_path=self.store_path, api_key="test-key")
        rag1.add_document("python programming language for scripting and automation tasks")
        rag1.add_document("java programming language for enterprise software development")

        # Меняем размерность на 5 и пересоздаём RAGSystem
        mock_httpx.return_value = self._make_mock_httpx(dim=5)
        rag2 = RAGSystem(store_path=self.store_path, api_key="test-key")

        # Поиск должен работать
        results = rag2.search("programming", k=2)
        assert len(results) == 2

    @patch("src.embeddings.httpx.Client")
    def test_bm25_search_works_after_auto_reindex(self, mock_httpx):
        """BM25 индекс не теряется при переиндексации векторного хранилища."""
        from src.rag import RAGSystem

        mock_httpx.return_value = self._make_mock_httpx(dim=3)
        rag1 = RAGSystem(store_path=self.store_path, api_key="test-key")
        rag1.add_document("python programming language for scripting and automation tasks")
        rag1.add_document("java programming language for enterprise software development")

        # Меняем размерность
        mock_httpx.return_value = self._make_mock_httpx(dim=5)
        rag2 = RAGSystem(store_path=self.store_path, api_key="test-key")

        # BM25 должен найти документы
        results = rag2.bm25_search("python", k=1)
        assert len(results) == 1
        assert "python" in results[0][1].lower()


class TestReindexChangesDimension:
    """Тесты для метода reindex() при смене размерности."""

    @pytest.fixture(autouse=True)
    def setup_temp_dir(self):
        self.temp_dir = tempfile.mkdtemp()
        self.store_path = os.path.join(self.temp_dir, "rag_data")
        yield
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def _make_mock_httpx(self, dim=3):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "data": [{"embedding": [0.1 * (i + 1) for i in range(dim)]}]
        }
        mock_client = MagicMock()
        mock_client.__enter__.return_value = mock_client
        mock_client.post.return_value = mock_response
        return mock_client

    @patch("src.embeddings.httpx.Client")
    def test_reindex_changes_dimension(self, mock_httpx):
        """reindex() меняет размерность коллекции."""
        from src.rag import RAGSystem

        mock_httpx.return_value = self._make_mock_httpx(dim=3)
        rag = RAGSystem(store_path=self.store_path, api_key="test-key")
        rag.add_document("doc one for testing reindex with dimension change scenario")
        rag.add_document("doc two for testing reindex with dimension change scenario")
        assert rag.vector_store.get_dimension() == 3

        # Меняем мок на 7-dim и переиндексируем
        mock_httpx.return_value = self._make_mock_httpx(dim=7)
        rag.embedding_generator._fallback_dimension = 7
        rag.embedding_generator.clear_cache()
        count = rag.reindex()

        assert count == 2
        assert rag.vector_store.get_dimension() == 7
        assert rag.vector_store.count() == 2

    @patch("src.embeddings.httpx.Client")
    def test_reindex_empty_store_returns_zero(self, mock_httpx):
        """reindex() на пустом хранилище возвращает 0."""
        from src.rag import RAGSystem

        mock_httpx.return_value = self._make_mock_httpx(dim=3)
        rag = RAGSystem(store_path=self.store_path, api_key="test-key")
        assert rag.reindex() == 0

    @patch("src.embeddings.httpx.Client")
    def test_reindex_same_dimension_no_recreate(self, mock_httpx):
        """reindex() при совпадении размерности не пересоздаёт коллекцию."""
        from src.rag import RAGSystem

        mock_httpx.return_value = self._make_mock_httpx(dim=3)
        rag = RAGSystem(store_path=self.store_path, api_key="test-key")
        rag.add_document("doc one for testing reindex with same dimension scenario")
        doc_ids_before = set(d[0] for d in rag.vector_store.get_all())

        count = rag.reindex()

        assert count == 1
        assert rag.vector_store.get_dimension() == 3
        doc_ids_after = set(d[0] for d in rag.vector_store.get_all())
        assert doc_ids_before == doc_ids_after
