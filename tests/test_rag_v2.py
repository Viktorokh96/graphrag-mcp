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
        doc_id = rag.add_document("Hello world, this is a test document for the RAG system.")

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
        rag.add_document("Python is a programming language used for various applications.")
        rag.add_document("Java runs on a virtual machine and is used for enterprise apps.")
        rag.add_document("Python is great for machine learning and data science tasks.")

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
        rag.add_document("python programming language is widely used for many purposes.")
        rag.add_document("java programming language is widely used for many purposes.")

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
        rag.add_document("python programming language is widely used for many purposes.")
        rag.add_document("java programming language is widely used for many purposes.")

        # pure semantic (alpha=1.0): both docs have same mock embedding → both found
        results_sem = rag.search_hybrid("python", k=2, alpha=1.0)
        assert len(results_sem) == 2

        # pure bm25 (alpha=0.0): only "python" doc matches BM25 (no "python" in java doc);
        # sem-only docs are excluded at alpha=0.0 (alpha-dilution)
        results_bm = rag.search_hybrid("python", k=2, alpha=0.0)
        assert len(results_bm) == 1
        assert "python" in results_bm[0][1].lower()

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
        rag.add_document("some text content that is sufficiently long for the test to pass.")
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
            f.write("This is a test file content for indexing purposes in our RAG system.")

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
        rag.add_document("python programming language is widely used for many purposes.")

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
                    {"embedding": [0.1 + i * 0.1, 0.2 + i * 0.05, 0.3 + i * 0.03]}
                    for i in range(len(texts))
                ]
            }
            return mock_resp

        mock_client = MagicMock()
        mock_client.__enter__.return_value = mock_client
        mock_client.post.side_effect = mock_post_side_effect
        mock_httpx.return_value = mock_client

        rag = RAGSystem(store_path=self.store_path, api_key="test-key")
        doc_ids = rag.add_documents(
            [
                "First document for testing the RAG system batch indexing feature.",
                "Second document for testing the RAG system batch indexing feature.",
                "Third document for testing the RAG system batch indexing feature.",
            ],
            metadata=[{"idx": 1}, {"idx": 2}, {"idx": 3}]
        )
        assert len(doc_ids) == 3

        stats = rag.stats()
        assert stats["total_documents"] == 3

    @patch("src.embeddings.httpx.Client")
    def test_add_document_too_short(self, mock_httpx):
        """D5: Документ короче MIN_CONTENT_LENGTH вызывает ValueError."""
        from src.rag import RAGSystem
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"data": [{"embedding": [0.1, 0.2, 0.3]}]}
        mock_client = MagicMock()
        mock_client.__enter__.return_value = mock_client
        mock_client.post.return_value = mock_response
        mock_httpx.return_value = mock_client

        rag = RAGSystem(store_path=self.store_path, api_key="test-key")
        with pytest.raises(ValueError, match="too short"):
            rag.add_document("short")

    @patch("src.embeddings.httpx.Client")
    def test_add_document_duplicate(self, mock_httpx):
        """D6: Добавление дубликата возвращает существующий doc_id."""
        from src.rag import RAGSystem
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"data": [{"embedding": [0.1, 0.2, 0.3]}]}
        mock_client = MagicMock()
        mock_client.__enter__.return_value = mock_client
        mock_client.post.return_value = mock_response
        mock_httpx.return_value = mock_client

        rag = RAGSystem(store_path=self.store_path, api_key="test-key")
        text = "This is a sufficiently long document for testing deduplication in the RAG system."
        assert rag.is_duplicate(text) is None, "Before adding, should not be a duplicate"
        doc_id1 = rag.add_document(text)
        assert rag.is_duplicate(text) == doc_id1, "is_duplicate should return existing doc_id"
        doc_id2 = rag.add_document(text)
        assert doc_id1 == doc_id2, "Duplicate documents should return the same doc_id"

    @patch("src.embeddings.httpx.Client")
    def test_store_sync_on_init(self, mock_httpx):
        """D1: _sync_stores() при инициализации удаляет фантомные документы из
        BM25 и графа, которых нет в vector store (источник истины)."""
        from src.rag import RAGSystem

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"data": [{"embedding": [0.1, 0.2, 0.3]}]}
        mock_client = MagicMock()
        mock_client.__enter__.return_value = mock_client
        mock_client.post.return_value = mock_response
        mock_httpx.return_value = mock_client

        rag = RAGSystem(store_path=self.store_path, api_key="test-key")
        rag.add_document("This is a valid document that is long enough to pass the length check.")

        # Прямо добавляем фантомный документ в BM25, которого нет в vector store
        rag.bm25_index.add_document("phantom-bm25", "This phantom doc is only in BM25")
        assert "phantom-bm25" in rag.bm25_index.get_all_doc_ids()

        # Прямо добавляем фантомный узел в граф
        rag.graph_kb.add_node("phantom-graph", "This phantom node is only in graph")
        assert "phantom-graph" in rag.graph_kb.get_all_node_ids()

        # Создаём новый RAGSystem на том же store_path — _sync_stores() подчистит
        rag2 = RAGSystem(store_path=self.store_path, api_key="test-key")

        # Фантомные документы должны быть удалены из BM25
        assert "phantom-bm25" not in rag2.bm25_index.get_all_doc_ids(), (
            "Фантомный BM25 документ должен быть удалён при sync"
        )
        # Фантомные узлы должны быть удалены из графа
        assert "phantom-graph" not in rag2.graph_kb.get_all_node_ids(), (
            "Фантомный граф-узел должен быть удалён при sync"
        )
        # Реальный документ должен сохраниться
        vector_ids = rag2.vector_store.get_all_ids()
        assert len(vector_ids) >= 1

    @patch("src.embeddings.httpx.Client")
    def test_metadata_always_dict_in_get_document(self, mock_httpx):
        """D10: metadata в get_document всегда dict, не None."""
        from src.rag import RAGSystem

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"data": [{"embedding": [0.1, 0.2, 0.3]}]}
        mock_client = MagicMock()
        mock_client.__enter__.return_value = mock_client
        mock_client.post.return_value = mock_response
        mock_httpx.return_value = mock_client

        rag = RAGSystem(store_path=self.store_path, api_key="test-key")
        doc_id = rag.add_document("A sufficiently long document for testing metadata normalization.")

        # get_document с None meta
        doc = rag.get_document(doc_id)
        assert doc is not None
        assert isinstance(doc["metadata"], dict), f"metadata should be dict, got {type(doc['metadata'])}"
        # Не None и не null
        assert doc["metadata"] is not None

    @patch("src.embeddings.httpx.Client")
    def test_metadata_always_dict_in_list_documents(self, mock_httpx):
        """D10: metadata в list_documents всегда dict, не None."""
        from src.rag import RAGSystem

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"data": [{"embedding": [0.1, 0.2, 0.3]}]}
        mock_client = MagicMock()
        mock_client.__enter__.return_value = mock_client
        mock_client.post.return_value = mock_response
        mock_httpx.return_value = mock_client

        rag = RAGSystem(store_path=self.store_path, api_key="test-key")
        rag.add_document("A sufficiently long document for testing metadata normalization.")
        rag.add_document("Another sufficiently long document with metadata for testing purposes.", metadata={"key": "val"})

        result = rag.list_documents(limit=10)
        for doc in result["documents"]:
            assert isinstance(doc["metadata"], dict), (
                f"metadata should be dict, got {type(doc['metadata'])} for {doc['doc_id']}"
            )
            assert doc["metadata"] is not None

    @patch("src.embeddings.httpx.Client")
    def test_metadata_always_dict_in_search(self, mock_httpx):
        """D10: metadata в search/bm25_search/search_hybrid всегда dict, не None."""
        from src.rag import RAGSystem

        def mock_post_side_effect(url, *args, **kwargs):
            mock_resp = MagicMock()
            mock_resp.status_code = 200
            data = kwargs.get("json", {})
            texts = data.get("input", [""])
            if isinstance(texts, str):
                texts = [texts]
            mock_resp.json.return_value = {
                "data": [{"embedding": [0.1, 0.2, 0.3]} for _ in range(len(texts))]
            }
            return mock_resp

        mock_client = MagicMock()
        mock_client.__enter__.return_value = mock_client
        mock_client.post.side_effect = mock_post_side_effect
        mock_httpx.return_value = mock_client

        rag = RAGSystem(store_path=self.store_path, api_key="test-key")
        rag.add_document("A sufficiently long document for testing metadata normalization in searches.")

        for search_fn, query in [
            (rag.search, "document"),
            (rag.bm25_search, "document"),
            (lambda q: rag.search_hybrid(q, alpha=0.5), "document"),
        ]:
            results = search_fn(query)
            for r in results:
                meta = r[3]  # metadata — четвёртый элемент кортежа
                assert isinstance(meta, dict), (
                    f"metadata should be dict in {search_fn.__name__}, got {type(meta)}"
                )
                assert meta is not None

    @patch("src.embeddings.httpx.Client")
    def test_rrf_no_ties(self, mock_httpx):
        """D7: RRF не даёт тай-оффов в топ-3."""
        from src.rag import RAGSystem
        counter = [0]

        def mock_post_side_effect(url, *args, **kwargs):
            mock_resp = MagicMock()
            mock_resp.status_code = 200
            data = kwargs.get("json", {})
            texts = data.get("input", [""])
            if isinstance(texts, str):
                texts = [texts]
            n = len(texts)
            mock_resp.json.return_value = {
                "data": [
                    {"embedding": [0.1 + (counter[0] + j) * 0.1, 0.2 + (counter[0] + j) * 0.05, 0.3 + (counter[0] + j) * 0.03]}
                    for j in range(n)
                ]
            }
            counter[0] += n
            return mock_resp

        mock_client = MagicMock()
        mock_client.__enter__.return_value = mock_client
        mock_client.post.side_effect = mock_post_side_effect
        mock_httpx.return_value = mock_client

        rag = RAGSystem(store_path=self.store_path, api_key="test-key")
        rag.add_document("Python programming language for building web applications and data science.")
        rag.add_document("Java programming language for enterprise development and mobile apps.")
        rag.add_document("JavaScript programming language for frontend and backend development.")
        rag.add_document("C++ programming language for systems programming and game development.")
        rag.add_document("Ruby programming language for web development with Rails framework.")

        results = rag.search_hybrid("programming language", k=3, alpha=0.5)
        assert len(results) >= 2
        scores = [r[2] for r in results[:3]]
        # All scores should be distinct (RRF eliminates ties)
        assert len(set(scores)) == len(scores), f"RRF scores should be distinct: {scores}"