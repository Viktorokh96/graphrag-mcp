"""Тесты для RAGSystem (оркестратор с гибридным поиском)."""

import pytest
import os
import shutil
import tempfile
from unittest.mock import patch, MagicMock


class TestRAGSystem:
    """Тесты для RAG системы."""

    @pytest.fixture(autouse=True)
    def setup_temp_dir(self):
        self.temp_dir = tempfile.mkdtemp()
        self.store_path = os.path.join(self.temp_dir, "rag_data")
        yield
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    @patch("src.embeddings.httpx.Client")
    def test_add_document(self, mock_httpx):
        from src.rag import RAGSystem

        # Мокаем OpenRouter
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "data": [{"embedding": [0.1, 0.2, 0.3]}]
        }
        mock_client = MagicMock()
        mock_client.__enter__.return_value = mock_client
        mock_client.post.return_value = mock_response
        mock_httpx.return_value = mock_client

        rag = RAGSystem(store_path=self.store_path, api_key="test-key")
        doc_id = rag.add_document("Hello world, this is a test document")

        assert doc_id is not None
        assert isinstance(doc_id, str)
        assert len(doc_id) > 0

    @patch("src.embeddings.httpx.Client")
    def test_add_and_search(self, mock_httpx):
        from src.rag import RAGSystem

        def mock_post_side_effect(url, *args, **kwargs):
            mock_resp = MagicMock()
            mock_resp.status_code = 200
            data = kwargs.get("json", {})
            texts = data.get("input", [""])
            if isinstance(texts, str):
                texts = [texts]
            mock_resp.json.return_value = {
                "data": [
                    {"embedding": [0.1 + i * 0.01, 0.2 + i * 0.01, 0.3 + i * 0.01]}
                    for i in range(len(texts))
                ]
            }
            return mock_resp

        mock_client = MagicMock()
        mock_client.__enter__.return_value = mock_client
        mock_client.post.side_effect = mock_post_side_effect
        mock_httpx.return_value = mock_client

        rag = RAGSystem(store_path=self.store_path, api_key="test-key")
        rag.add_document("Python is a programming language")
        rag.add_document("Java runs on a virtual machine")
        rag.add_document("Python is great for machine learning")

        results = rag.search("Python programming", k=2)
        # Мокаем эмбеддинги - semantic search не работает корректно, проверяем только наличие результатов
        assert len(results) == 2

    @patch("src.embeddings.httpx.Client")
    def test_bm25_search(self, mock_httpx):
        from src.rag import RAGSystem

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "data": [{"embedding": [0.1, 0.2, 0.3]}]
        }
        mock_client = MagicMock()
        mock_client.__enter__.return_value = mock_client
        mock_client.post.return_value = mock_response
        mock_httpx.return_value = mock_client

        rag = RAGSystem(store_path=self.store_path, api_key="test-key")
        rag.add_document("python programming language")
        rag.add_document("java programming language")

        results = rag.bm25_search("python", k=1)
        assert len(results) == 1
        assert "python" in results[0][1].lower()

    @patch("src.embeddings.httpx.Client")
    def test_search_hybrid(self, mock_httpx):
        from src.rag import RAGSystem

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "data": [{"embedding": [0.1, 0.2, 0.3]}]
        }
        mock_client = MagicMock()
        mock_client.__enter__.return_value = mock_client
        mock_client.post.return_value = mock_response
        mock_httpx.return_value = mock_client

        rag = RAGSystem(store_path=self.store_path, api_key="test-key")
        rag.add_document("python programming language")
        rag.add_document("java programming language")

        # pure semantic
        results_sem = rag.search_hybrid("python", k=2, alpha=1.0)
        assert len(results_sem) == 2

        # pure bm25
        results_bm = rag.search_hybrid("python", k=2, alpha=0.0)
        assert len(results_bm) == 2

        # balanced
        results_mix = rag.search_hybrid("python", k=2, alpha=0.5)
        assert len(results_mix) == 2

    @patch("src.embeddings.httpx.Client")
    def test_clear(self, mock_httpx):
        from src.rag import RAGSystem

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "data": [{"embedding": [0.1, 0.2, 0.3]}]
        }
        mock_client = MagicMock()
        mock_client.__enter__.return_value = mock_client
        mock_client.post.return_value = mock_response
        mock_httpx.return_value = mock_client

        rag = RAGSystem(store_path=self.store_path, api_key="test-key")
        rag.add_document("some text")
        stats_before = rag.stats()
        assert stats_before["total_documents"] > 0

        rag.clear()
        stats = rag.stats()
        total = stats["total_documents"]
        assert total == 0

    @patch("src.embeddings.httpx.Client")
    def test_stats(self, mock_httpx):
        from src.rag import RAGSystem

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "data": [{"embedding": [0.1, 0.2, 0.3]}]
        }
        mock_client = MagicMock()
        mock_client.__enter__.return_value = mock_client
        mock_client.post.return_value = mock_response
        mock_httpx.return_value = mock_client

        rag = RAGSystem(store_path=self.store_path, api_key="test-key")
        stats = rag.stats()
        assert "total_documents" in stats
        assert "store_path" in stats

    @patch("src.embeddings.httpx.Client")
    def test_add_file(self, mock_httpx):
        from src.rag import RAGSystem

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "data": [{"embedding": [0.1, 0.2, 0.3]}]
        }
        mock_client = MagicMock()
        mock_client.__enter__.return_value = mock_client
        mock_client.post.return_value = mock_response
        mock_httpx.return_value = mock_client

        # Создаём временный файл
        test_file = os.path.join(self.temp_dir, "test_doc.txt")
        with open(test_file, "w") as f:
            f.write("This is a test file content for indexing")

        rag = RAGSystem(store_path=self.store_path, api_key="test-key")
        doc_id = rag.add_file(test_file)

        assert doc_id is not None
        assert isinstance(doc_id, str)

    @patch("src.embeddings.httpx.Client")
    def test_search_empty_index(self, mock_httpx):
        from src.rag import RAGSystem

        rag = RAGSystem(store_path=self.store_path, api_key="test-key")
        results = rag.search("anything")
        assert results == []

    @patch("src.embeddings.httpx.Client")
    def test_search_empty_bm25(self, mock_httpx):
        from src.rag import RAGSystem

        rag = RAGSystem(store_path=self.store_path, api_key="test-key")
        results = rag.bm25_search("anything")
        assert results == []

    @patch("src.embeddings.httpx.Client")
    def test_search_empty_hybrid(self, mock_httpx):
        from src.rag import RAGSystem

        rag = RAGSystem(store_path=self.store_path, api_key="test-key")
        results = rag.search_hybrid("anything")
        assert results == []

    @patch("src.embeddings.httpx.Client")
    def test_hybrid_with_custom_alpha(self, mock_httpx):
        from src.rag import RAGSystem

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "data": [{"embedding": [0.1, 0.2, 0.3]}]
        }
        mock_client = MagicMock()
        mock_client.__enter__.return_value = mock_client
        mock_client.post.return_value = mock_response
        mock_httpx.return_value = mock_client

        rag = RAGSystem(store_path=self.store_path, api_key="test-key")
        rag.add_document("python programming language")

        # alpha=0.75
        results = rag.search_hybrid("python", k=1, alpha=0.75)
        assert len(results) == 1

    @patch("src.embeddings.httpx.Client")
    def test_add_documents_batch(self, mock_httpx):
        from src.rag import RAGSystem

        def mock_post_side_effect(url, *args, **kwargs):
            mock_resp = MagicMock()
            mock_resp.status_code = 200
            data = kwargs.get("json", {})
            texts = data.get("input", [""])
            if isinstance(texts, str):
                texts = [texts]
            mock_resp.json.return_value = {
                "data": [
                    {"embedding": [0.1, 0.2, 0.3]}
                    for _ in range(len(texts))
                ]
            }
            return mock_resp

        mock_client = MagicMock()
        mock_client.__enter__.return_value = mock_client
        mock_client.post.side_effect = mock_post_side_effect
        mock_httpx.return_value = mock_client

        rag = RAGSystem(store_path=self.store_path, api_key="test-key")
        doc_ids = rag.add_documents(
            ["First document", "Second document", "Third document"],
            metadata=[{"idx": 1}, {"idx": 2}, {"idx": 3}]
        )
        assert len(doc_ids) == 3

        stats = rag.stats()
        assert stats["total_documents"] == 3