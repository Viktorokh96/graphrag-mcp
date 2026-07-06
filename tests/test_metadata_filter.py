"""Тесты для фильтрации по метаданным (metadata_filter).

Покрывают:
- src/_meta_filter.py: matches_metadata_filter(), to_chroma_where()
- src/vector_store.py: search(where=...), list_documents(where=...)
- src/bm25_index.py: search(metadata_filter=...)
- src/graph_store.py: get_related(metadata_filter=...)
- src/rag.py: search/bm25_search/search_hybrid/list_documents/get_related с фильтром
- src/mcp_server.py: handle_tool_call пробрасывает metadata_filter
- src/cli.py: флаг --meta-filter парсится
"""

import os
import shutil
import tempfile
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mock_embedding_client():
    """Мок OpenRouter/OpenAI embeddings API (как в test_mcp_server.py)."""
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"data": [{"embedding": [0.1, 0.2, 0.3]}]}
    mock_client = MagicMock()
    mock_client.__enter__.return_value = mock_client
    mock_client.post.return_value = mock_response
    return mock_client


# ===========================================================================
# src/_meta_filter.py
# ===========================================================================

class TestMatchesMetadataFilter:
    """Unit-тесты для matches_metadata_filter()."""

    def test_none_filter_returns_true(self):
        from src._meta_filter import matches_metadata_filter
        assert matches_metadata_filter({"a": 1}, None) is True

    def test_empty_filter_returns_true(self):
        from src._meta_filter import matches_metadata_filter
        assert matches_metadata_filter({"a": 1}, {}) is True

    def test_scalar_exact_match(self):
        from src._meta_filter import matches_metadata_filter
        assert matches_metadata_filter({"source": "spec"}, {"source": "spec"}) is True

    def test_scalar_no_match(self):
        from src._meta_filter import matches_metadata_filter
        assert matches_metadata_filter({"source": "spec"}, {"source": "other"}) is False

    def test_list_in_membership(self):
        from src._meta_filter import matches_metadata_filter
        assert matches_metadata_filter({"type": "bug"}, {"type": ["bug", "feature"]}) is True

    def test_list_not_in_membership(self):
        from src._meta_filter import matches_metadata_filter
        assert matches_metadata_filter({"type": "doc"}, {"type": ["bug", "feature"]}) is False

    def test_multiple_keys_and(self):
        from src._meta_filter import matches_metadata_filter
        meta = {"source": "spec", "type": "bug", "idx": 5}
        assert matches_metadata_filter(meta, {"source": "spec", "type": "bug"}) is True
        assert matches_metadata_filter(meta, {"source": "spec", "type": "feature"}) is False
        assert matches_metadata_filter(meta, {"source": "spec", "type": ["bug", "feature"]}) is True

    def test_missing_key_returns_false(self):
        from src._meta_filter import matches_metadata_filter
        assert matches_metadata_filter({"a": 1}, {"b": 2}) is False

    def test_none_meta_with_active_filter(self):
        from src._meta_filter import matches_metadata_filter
        assert matches_metadata_filter(None, {"a": 1}) is False

    def test_empty_meta_with_active_filter(self):
        from src._meta_filter import matches_metadata_filter
        assert matches_metadata_filter({}, {"a": 1}) is False


class TestNormalizeMetadataFilter:
    """Unit-тесты для normalize_metadata_filter()."""

    def test_none_returns_none(self):
        from src._meta_filter import normalize_metadata_filter
        assert normalize_metadata_filter(None) is None

    def test_empty_string_returns_none(self):
        from src._meta_filter import normalize_metadata_filter
        assert normalize_metadata_filter("") is None

    def test_empty_dict_returns_none(self):
        from src._meta_filter import normalize_metadata_filter
        assert normalize_metadata_filter({}) is None

    def test_dict_passthrough(self):
        from src._meta_filter import normalize_metadata_filter
        result = normalize_metadata_filter({"source": "spec"})
        assert result == {"source": "spec"}

    def test_json_string_parsed(self):
        from src._meta_filter import normalize_metadata_filter
        result = normalize_metadata_filter('{"source": "spec", "type": ["bug", "feature"]}')
        assert result == {"source": "spec", "type": ["bug", "feature"]}

    def test_empty_json_string_returns_none(self):
        from src._meta_filter import normalize_metadata_filter
        assert normalize_metadata_filter('{}') is None

    def test_invalid_json_string_returns_none(self):
        from src._meta_filter import normalize_metadata_filter
        assert normalize_metadata_filter("not a json") is None

    def test_json_array_string_returns_none(self):
        from src._meta_filter import normalize_metadata_filter
        assert normalize_metadata_filter('[1, 2, 3]') is None

    def test_non_string_non_dict_returns_none(self):
        from src._meta_filter import normalize_metadata_filter
        assert normalize_metadata_filter(42) is None
        assert normalize_metadata_filter(["a", "b"]) is None


class TestToChromaWhere:
    """Unit-тесты для to_chroma_where()."""

    def test_none_returns_none(self):
        from src._meta_filter import to_chroma_where
        assert to_chroma_where(None) is None

    def test_empty_returns_none(self):
        from src._meta_filter import to_chroma_where
        assert to_chroma_where({}) is None

    def test_single_scalar(self):
        from src._meta_filter import to_chroma_where
        assert to_chroma_where({"source": "spec"}) == {"source": "spec"}

    def test_single_list_in(self):
        from src._meta_filter import to_chroma_where
        assert to_chroma_where({"type": ["bug", "feature"]}) == {"type": {"$in": ["bug", "feature"]}}

    def test_multiple_keys_and(self):
        from src._meta_filter import to_chroma_where
        result = to_chroma_where({"source": "spec", "type": "bug"})
        assert result == {"$and": [{"source": "spec"}, {"type": "bug"}]}

    def test_non_scalar_values_ignored(self):
        from src._meta_filter import to_chroma_where
        # dict-значения не поддерживаются ChromaDB → игнорируются
        result = to_chroma_where({"nested": {"a": 1}, "source": "spec"})
        assert result == {"source": "spec"}

    def test_list_with_non_scalar_elements_cleaned(self):
        from src._meta_filter import to_chroma_where
        result = to_chroma_where({"type": ["bug", {"x": 1}, "feature"]})
        assert result == {"type": {"$in": ["bug", "feature"]}}


# ===========================================================================
# src/vector_store.py
# ===========================================================================

class TestVectorStoreMetadataFilter:
    """VectorStore.search / list_documents с where-клаузой ChromaDB."""

    @pytest.fixture(autouse=True)
    def setup_temp_dir(self):
        self.temp_dir = tempfile.mkdtemp()
        self.store_path = os.path.join(self.temp_dir, "rag_data")
        yield
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def _make_store(self):
        from src.vector_store import VectorStore
        return VectorStore(store_path=self.store_path)

    def test_search_with_where_filters_results(self):
        store = self._make_store()
        store.add("d1", "python programming language", [0.1, 0.2, 0.3], {"source": "spec"})
        store.add("d2", "java programming language", [0.1, 0.2, 0.3], {"source": "blog"})
        store.add("d3", "python machine learning", [0.1, 0.2, 0.3], {"source": "spec"})

        results = store.search([0.1, 0.2, 0.3], k=10, where={"source": "spec"})
        ids = {r[0] for r in results}
        assert ids == {"d1", "d3"}

    def test_search_without_where_returns_all(self):
        store = self._make_store()
        store.add("d1", "python programming language", [0.1, 0.2, 0.3], {"source": "spec"})
        store.add("d2", "java programming language", [0.1, 0.2, 0.3], {"source": "blog"})

        results = store.search([0.1, 0.2, 0.3], k=10)
        assert len(results) == 2

    def test_list_documents_with_where(self):
        store = self._make_store()
        store.add("d1", "python doc one", [0.1, 0.2, 0.3], {"type": "bug"})
        store.add("d2", "java doc two", [0.1, 0.2, 0.3], {"type": "feature"})
        store.add("d3", "python doc three", [0.1, 0.2, 0.3], {"type": "bug"})

        items, total = store.list_documents(limit=20, offset=0, where={"type": "bug"})
        ids = {i[0] for i in items}
        assert ids == {"d1", "d3"}
        assert total == 2, "total должен отражать число отфильтрованных документов"

    def test_list_documents_with_in_filter(self):
        store = self._make_store()
        store.add("d1", "doc one", [0.1, 0.2, 0.3], {"type": "bug"})
        store.add("d2", "doc two", [0.1, 0.2, 0.3], {"type": "feature"})
        store.add("d3", "doc three", [0.1, 0.2, 0.3], {"type": "doc"})

        where = {"type": {"$in": ["bug", "feature"]}}
        items, total = store.list_documents(limit=20, offset=0, where=where)
        ids = {i[0] for i in items}
        assert ids == {"d1", "d2"}
        assert total == 2


# ===========================================================================
# src/bm25_index.py
# ===========================================================================

class TestBM25MetadataFilter:
    """BM25Index.search с metadata_filter (post-filter)."""

    def test_search_with_metadata_filter(self):
        from src.bm25_index import BM25Index
        idx = BM25Index(store_path=None)
        idx.add_document("d1", "python programming language tutorial", {"source": "spec"})
        idx.add_document("d2", "python programming guide", {"source": "blog"})
        idx.add_document("d3", "java programming reference", {"source": "spec"})

        results = idx.search("python", k=10, metadata_filter={"source": "spec"})
        ids = {r[0] for r in results}
        # d2 (blog) отфильтрован; d1 входит (python, spec); d3 не содержит "python"
        assert "d1" in ids
        assert "d2" not in ids

    def test_search_with_in_list_filter(self):
        from src.bm25_index import BM25Index
        idx = BM25Index(store_path=None)
        idx.add_document("d1", "python tutorial here", {"type": "bug"})
        idx.add_document("d2", "python guide here", {"type": "feature"})
        idx.add_document("d3", "python reference here", {"type": "doc"})

        results = idx.search("python", k=10, metadata_filter={"type": ["bug", "feature"]})
        ids = {r[0] for r in results}
        assert ids == {"d1", "d2"}

    def test_search_without_filter_returns_all_matches(self):
        from src.bm25_index import BM25Index
        idx = BM25Index(store_path=None)
        idx.add_document("d1", "python tutorial here", {"source": "spec"})
        idx.add_document("d2", "python guide here", {"source": "blog"})

        results = idx.search("python", k=10)
        assert len(results) == 2

    def test_search_filter_returns_matching(self):
        from src.bm25_index import BM25Index
        idx = BM25Index(store_path=None)
        idx.add_document("d1", "python python python tutorial guide introduction", {"src": "a"})
        idx.add_document("d2", "python tutorial guide for beginners here", {"src": "a"})
        idx.add_document("d3", "python tutorial guide for beginners here", {"src": "b"})

        results = idx.search("python", k=10, metadata_filter={"src": "a"})
        ids = {r[0] for r in results}
        # d3 (src=b) отфильтрован; d1 и d2 возвращаются
        assert ids == {"d1", "d2"}


# ===========================================================================
# src/graph_store.py
# ===========================================================================

class TestGraphMetadataFilter:
    """GraphKnowledgeBase.get_related с metadata_filter."""

    def _build_graph(self):
        from src.graph_store import GraphKnowledgeBase
        g = GraphKnowledgeBase(store_path=None)
        g.add_node("n1", "node one text", {"type": "spec"})
        g.add_node("n2", "node two text", {"type": "bug"})
        g.add_node("n3", "node three text", {"type": "spec"})
        g.add_node("n4", "node four text", {"type": "doc"})
        g.add_edge("n1", "n2", "related_to")
        g.add_edge("n1", "n3", "related_to")
        g.add_edge("n1", "n4", "related_to")
        return g

    def test_get_related_without_filter(self):
        g = self._build_graph()
        rels = g.get_related("n1", max_depth=1)
        targets = {r[1] for r in rels}
        assert targets == {"n2", "n3", "n4"}

    def test_get_related_with_filter_keeps_matching(self):
        g = self._build_graph()
        rels = g.get_related("n1", max_depth=1, metadata_filter={"type": "spec"})
        targets = {r[1] for r in rels}
        # Только n3 имеет type=spec среди соседей n1
        assert targets == {"n3"}

    def test_get_related_with_in_list_filter(self):
        g = self._build_graph()
        rels = g.get_related("n1", max_depth=1, metadata_filter={"type": ["spec", "bug"]})
        targets = {r[1] for r in rels}
        assert targets == {"n2", "n3"}

    def test_get_related_filter_excludes_all(self):
        g = self._build_graph()
        rels = g.get_related("n1", max_depth=1, metadata_filter={"type": "nonexistent"})
        assert rels == []

    def test_get_related_empty_filter_no_filtering(self):
        g = self._build_graph()
        rels = g.get_related("n1", max_depth=1, metadata_filter={})
        targets = {r[1] for r in rels}
        assert targets == {"n2", "n3", "n4"}


# ===========================================================================
# src/rag.py — RAGSystem с мок-эмбеддингами
# ===========================================================================

class TestRAGMetadataFilter:
    """RAGSystem: search/bm25_search/search_hybrid/list_documents/get_related."""

    @pytest.fixture(autouse=True)
    def setup_temp_dir(self):
        self.temp_dir = tempfile.mkdtemp()
        self.store_path = os.path.join(self.temp_dir, "rag_data")
        yield
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def _make_rag(self, mock_httpx):
        from src.rag import RAGSystem
        mock_httpx.return_value = _mock_embedding_client()
        return RAGSystem(store_path=self.store_path, api_key="test-key")

    def _seed(self, rag):
        """Добавить документы с разной метаданной."""
        rag.add_document("python programming language tutorial introduction for beginners", {"source": "spec", "type": "doc"})
        rag.add_document("java programming language enterprise guide for backend developers", {"source": "blog", "type": "doc"})
        rag.add_document("python machine learning neural network deep learning tutorial", {"source": "spec", "type": "tutorial"})
        rag.add_document("cooking recipe pasta italian food delicious meal preparation", {"source": "blog", "type": "tutorial"})

    @patch("src.embeddings.httpx.Client")
    def test_search_with_metadata_filter(self, mock_httpx):
        rag = self._make_rag(mock_httpx)
        self._seed(rag)
        results = rag.search("programming", k=10, metadata_filter={"source": "spec"})
        for _doc_id, _text, _score, meta in results:
            assert meta.get("source") == "spec"

    @patch("src.embeddings.httpx.Client")
    def test_search_with_in_list_filter(self, mock_httpx):
        rag = self._make_rag(mock_httpx)
        self._seed(rag)
        results = rag.search("programming", k=10, metadata_filter={"type": ["doc", "tutorial"]})
        for _doc_id, _text, _score, meta in results:
            assert meta.get("type") in ("doc", "tutorial")

    @patch("src.embeddings.httpx.Client")
    def test_search_without_filter_returns_all(self, mock_httpx):
        rag = self._make_rag(mock_httpx)
        self._seed(rag)
        results = rag.search("programming", k=10)
        # Без фильтра возвращаются все документы (мок-эмбеддинги одинаковые)
        assert len(results) == 4

    @patch("src.embeddings.httpx.Client")
    def test_bm25_search_with_metadata_filter(self, mock_httpx):
        rag = self._make_rag(mock_httpx)
        self._seed(rag)
        results = rag.bm25_search("python", k=10, metadata_filter={"source": "spec"})
        # Только python+spec документы
        for _doc_id, text, _score, meta in results:
            assert meta.get("source") == "spec"
            assert "python" in text.lower()

    @patch("src.embeddings.httpx.Client")
    def test_bm25_search_filter_excludes_non_matching(self, mock_httpx):
        rag = self._make_rag(mock_httpx)
        self._seed(rag)
        results = rag.bm25_search("programming", k=10, metadata_filter={"source": "blog"})
        # Только java doc (blog) содержит "programming" и source=blog
        for _doc_id, _text, _score, meta in results:
            assert meta.get("source") == "blog"

    @patch("src.embeddings.httpx.Client")
    def test_search_hybrid_with_metadata_filter(self, mock_httpx):
        rag = self._make_rag(mock_httpx)
        self._seed(rag)
        results = rag.search_hybrid("python", k=10, alpha=0.5, metadata_filter={"source": "spec"})
        for _doc_id, _text, _score, meta in results:
            assert meta.get("source") == "spec"

    @patch("src.embeddings.httpx.Client")
    def test_search_hybrid_filter_applies_both_channels(self, mock_httpx):
        """Гибридный поиск: фильтр применяется к обоим каналам."""
        rag = self._make_rag(mock_httpx)
        self._seed(rag)
        # alpha=0.0 → чистый BM25; фильтр должен сработать
        results_bm = rag.search_hybrid("programming", k=10, alpha=0.0, metadata_filter={"source": "spec"})
        for _doc_id, _text, _score, meta in results_bm:
            assert meta.get("source") == "spec"
        # alpha=1.0 → чистая семантика; фильтр должен сработать
        results_sem = rag.search_hybrid("programming", k=10, alpha=1.0, metadata_filter={"source": "blog"})
        for _doc_id, _text, _score, meta in results_sem:
            assert meta.get("source") == "blog"

    @patch("src.embeddings.httpx.Client")
    def test_list_documents_with_metadata_filter(self, mock_httpx):
        rag = self._make_rag(mock_httpx)
        self._seed(rag)
        out = rag.list_documents(limit=20, offset=0, metadata_filter={"source": "spec"})
        # 2 документа с source=spec
        assert out["total"] == 2
        for d in out["documents"]:
            assert d["metadata"].get("source") == "spec"

    @patch("src.embeddings.httpx.Client")
    def test_list_documents_without_filter(self, mock_httpx):
        rag = self._make_rag(mock_httpx)
        self._seed(rag)
        out = rag.list_documents(limit=20, offset=0)
        assert out["total"] == 4

    @patch("src.embeddings.httpx.Client")
    def test_list_documents_empty_filter_no_filtering(self, mock_httpx):
        rag = self._make_rag(mock_httpx)
        self._seed(rag)
        out = rag.list_documents(limit=20, offset=0, metadata_filter={})
        assert out["total"] == 4

    @patch("src.embeddings.httpx.Client")
    def test_get_related_with_metadata_filter(self, mock_httpx):
        rag = self._make_rag(mock_httpx)
        d1 = rag.add_document("first node document with sufficient length for the test here", {"type": "spec"})
        d2 = rag.add_document("second node document with sufficient length for the test here", {"type": "bug"})
        d3 = rag.add_document("third node document with sufficient length for the test here", {"type": "spec"})
        rag.add_relation(d1, d2, "related_to")
        rag.add_relation(d1, d3, "related_to")

        rels = rag.get_related(d1, max_depth=1, metadata_filter={"type": "spec"})
        targets = {r[1] for r in rels}
        assert targets == {d3}

    @patch("src.embeddings.httpx.Client")
    def test_get_related_without_filter(self, mock_httpx):
        rag = self._make_rag(mock_httpx)
        d1 = rag.add_document("first node document with sufficient length for the test here", {"type": "spec"})
        d2 = rag.add_document("second node document with sufficient length for the test here", {"type": "bug"})
        rag.add_relation(d1, d2, "related_to")

        rels = rag.get_related(d1, max_depth=1)
        targets = {r[1] for r in rels}
        assert targets == {d2}

    @patch("src.embeddings.httpx.Client")
    def test_doc_without_metadata_excluded_by_filter(self, mock_httpx):
        """Документ без metadata (None/{}) не проходит активный фильтр."""
        rag = self._make_rag(mock_httpx)
        rag.add_document("python programming language tutorial introduction for beginners", None)
        rag.add_document("java programming language enterprise guide for backend developers", {"source": "spec"})

        results = rag.bm25_search("programming", k=10, metadata_filter={"source": "spec"})
        for _doc_id, _text, _score, meta in results:
            assert meta.get("source") == "spec"
        # Документ с None-meta не должен попасть в выдачу
        assert len(results) == 1


# ===========================================================================
# src/mcp_server.py — handle_tool_call
# ===========================================================================

class TestMCPMetadataFilter:
    """handle_tool_call пробрасывает metadata_filter в RAGSystem."""

    @pytest.fixture(autouse=True)
    def setup_temp_dir(self):
        self.temp_dir = tempfile.mkdtemp()
        self.store_path = os.path.join(self.temp_dir, "rag_data")
        yield
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def _make_rag(self, mock_httpx):
        from src.rag import RAGSystem
        mock_httpx.return_value = _mock_embedding_client()
        return RAGSystem(store_path=self.store_path, api_key="test-key")

    def _seed(self, rag):
        rag.add_document("python programming language tutorial introduction for beginners here", {"source": "spec"})
        rag.add_document("java programming language enterprise guide for developers here", {"source": "blog"})
        rag.add_document("python machine learning neural network deep learning tutorial here", {"source": "spec"})

    @patch("src.embeddings.httpx.Client")
    def test_rag_search_passes_metadata_filter(self, mock_httpx):
        from src.mcp_server import handle_tool_call
        rag = self._make_rag(mock_httpx)
        self._seed(rag)
        result = handle_tool_call(rag, "rag_search", {
            "query": "programming", "k": 10, "metadata_filter": {"source": "spec"},
        })
        for r in result:
            assert r["metadata"].get("source") == "spec"

    @patch("src.embeddings.httpx.Client")
    def test_rag_bm25_search_passes_metadata_filter(self, mock_httpx):
        from src.mcp_server import handle_tool_call
        rag = self._make_rag(mock_httpx)
        self._seed(rag)
        result = handle_tool_call(rag, "rag_bm25_search", {
            "query": "python", "k": 10, "metadata_filter": {"source": "spec"},
        })
        for r in result:
            assert r["metadata"].get("source") == "spec"

    @patch("src.embeddings.httpx.Client")
    def test_rag_search_hybrid_passes_metadata_filter(self, mock_httpx):
        from src.mcp_server import handle_tool_call
        rag = self._make_rag(mock_httpx)
        self._seed(rag)
        result = handle_tool_call(rag, "rag_search_hybrid", {
            "query": "python", "k": 10, "metadata_filter": {"source": "spec"},
        })
        for r in result:
            assert r["metadata"].get("source") == "spec"

    @patch("src.embeddings.httpx.Client")
    def test_rag_list_documents_passes_metadata_filter(self, mock_httpx):
        from src.mcp_server import handle_tool_call
        rag = self._make_rag(mock_httpx)
        self._seed(rag)
        result = handle_tool_call(rag, "rag_list_documents", {
            "limit": 20, "metadata_filter": {"source": "spec"},
        })
        assert result["total"] == 2
        for d in result["documents"]:
            assert d["metadata"].get("source") == "spec"

    @patch("src.embeddings.httpx.Client")
    def test_rag_get_related_passes_metadata_filter(self, mock_httpx):
        from src.mcp_server import handle_tool_call
        rag = self._make_rag(mock_httpx)
        d1 = rag.add_document("first node document with sufficient length for the test here", {"type": "spec"})
        d2 = rag.add_document("second node document with sufficient length for the test here", {"type": "bug"})
        d3 = rag.add_document("third node document with sufficient length for the test here", {"type": "spec"})
        rag.add_relation(d1, d2, "related_to")
        rag.add_relation(d1, d3, "related_to")

        result = handle_tool_call(rag, "rag_get_related", {
            "node_id": d1, "max_depth": 1, "metadata_filter": {"type": "spec"},
        })
        targets = {r["target"] for r in result["relations"]}
        assert targets == {d3}

    @patch("src.embeddings.httpx.Client")
    def test_rag_search_without_metadata_filter_backward_compat(self, mock_httpx):
        """Без metadata_filter — обратная совместимость (возвращаются все)."""
        from src.mcp_server import handle_tool_call
        rag = self._make_rag(mock_httpx)
        self._seed(rag)
        result = handle_tool_call(rag, "rag_search", {"query": "programming", "k": 10})
        assert len(result) == 3

    @patch("src.embeddings.httpx.Client")
    def test_rag_search_metadata_filter_as_json_string(self, mock_httpx):
        """Регрессия: metadata_filter передан как JSON-строка (как делает MCP SDK).
        Не должен падать с AttributeError — должен распарситься в dict."""
        from src.mcp_server import handle_tool_call
        rag = self._make_rag(mock_httpx)
        self._seed(rag)
        result = handle_tool_call(rag, "rag_search", {
            "query": "programming", "k": 10,
            "metadata_filter": '{"source": "spec"}',
        })
        for r in result:
            assert r["metadata"].get("source") == "spec"

    @patch("src.embeddings.httpx.Client")
    def test_rag_list_documents_metadata_filter_as_json_string(self, mock_httpx):
        """Регрессия: metadata_filter как JSON-строка для list_documents."""
        from src.mcp_server import handle_tool_call
        rag = self._make_rag(mock_httpx)
        self._seed(rag)
        result = handle_tool_call(rag, "rag_list_documents", {
            "limit": 20, "metadata_filter": '{"source": "spec"}',
        })
        assert result["total"] == 2
        for d in result["documents"]:
            assert d["metadata"].get("source") == "spec"

    @patch("src.embeddings.httpx.Client")
    def test_rag_search_hybrid_metadata_filter_as_json_string(self, mock_httpx):
        """Регрессия: metadata_filter как JSON-строка для search_hybrid."""
        from src.mcp_server import handle_tool_call
        rag = self._make_rag(mock_httpx)
        self._seed(rag)
        result = handle_tool_call(rag, "rag_search_hybrid", {
            "query": "python", "k": 10, "metadata_filter": '{"source": "spec"}',
        })
        for r in result:
            assert r["metadata"].get("source") == "spec"

    @patch("src.embeddings.httpx.Client")
    def test_rag_search_invalid_json_metadata_filter_ignored(self, mock_httpx):
        """Невалидная JSON-строка в metadata_filter → фильтр отключается (не падает)."""
        from src.mcp_server import handle_tool_call
        rag = self._make_rag(mock_httpx)
        self._seed(rag)
        result = handle_tool_call(rag, "rag_search", {
            "query": "programming", "k": 10, "metadata_filter": "not a json",
        })
        # Фильтр отключен → все документы возвращаются
        assert len(result) == 3


# ===========================================================================
# src/cli.py — флаг --meta-filter
# ===========================================================================

class TestCLIMetaFilter:
    """CLI: --meta-filter парсится и передаётся в RAGSystem."""

    @pytest.fixture(autouse=True)
    def setup_temp_dir(self):
        self.temp_dir = tempfile.mkdtemp()
        self.store_path = os.path.join(self.temp_dir, "rag_data")
        yield
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    @patch("src.embeddings.httpx.Client")
    def test_search_with_meta_filter_flag(self, mock_httpx):
        from src.cli import main
        from src.rag import RAGSystem

        mock_httpx.return_value = _mock_embedding_client()
        rag = RAGSystem(store_path=self.store_path, api_key="test-key")
        rag.add_document("python programming language tutorial introduction for beginners here", {"source": "spec"})
        rag.add_document("java programming language enterprise guide for developers here", {"source": "blog"})
        del rag  # закрываем, CLI пересоздаст

        exit_code = main([
            "--store", self.store_path, "--key", "test-key",
            "search", "--query", "programming", "--k", "10",
            "--meta-filter", '{"source": "spec"}',
        ])
        assert exit_code == 0

    @patch("src.embeddings.httpx.Client")
    def test_hybrid_search_with_meta_filter_flag(self, mock_httpx):
        from src.cli import main
        from src.rag import RAGSystem

        mock_httpx.return_value = _mock_embedding_client()
        rag = RAGSystem(store_path=self.store_path, api_key="test-key")
        rag.add_document("python programming language tutorial introduction for beginners here", {"source": "spec"})
        del rag

        exit_code = main([
            "--store", self.store_path, "--key", "test-key",
            "hybrid-search", "--query", "python", "--meta-filter", '{"source": "spec"}',
        ])
        assert exit_code == 0

    @patch("src.embeddings.httpx.Client")
    def test_search_without_meta_filter_flag(self, mock_httpx):
        from src.cli import main
        from src.rag import RAGSystem

        mock_httpx.return_value = _mock_embedding_client()
        rag = RAGSystem(store_path=self.store_path, api_key="test-key")
        rag.add_document("python programming language tutorial introduction for beginners here", {"source": "spec"})
        del rag

        exit_code = main([
            "--store", self.store_path, "--key", "test-key",
            "search", "--query", "python",
        ])
        assert exit_code == 0
