"""Тесты для фильтрации по метаданным (metadata_filter) на стеке Qdrant + SQLite.

Покрывают:
- src/_meta_filter.py: matches_metadata_filter(), normalize_metadata_filter()
- src/vector_store.py: to_qdrant_filter() — преобразование в нативный Qdrant Filter
- src/rag.py: search/bm25_search/search_hybrid/list_documents/get_related с фильтром
- src/mcp_server.py: handle_tool_call пробрасывает metadata_filter

Интеграционные тесты используют фикстуры rag/make_rag из conftest
(RAGSystem + HashEmbeddingGenerator, tmp store, авто-close для Windows).
"""



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


# ===========================================================================
# src/vector_store.py — to_qdrant_filter (замена to_chroma_where)
# ===========================================================================

class TestToQdrantFilter:
    """Unit-тесты для to_qdrant_filter(): metadata_filter → qdrant Filter."""

    def test_none_returns_none(self):
        from src.vector_store import to_qdrant_filter
        assert to_qdrant_filter(None) is None

    def test_empty_returns_none(self):
        from src.vector_store import to_qdrant_filter
        assert to_qdrant_filter({}) is None

    def test_single_scalar_string(self):
        from qdrant_client import models
        from src.vector_store import to_qdrant_filter
        f = to_qdrant_filter({"source": "spec"})
        assert isinstance(f, models.Filter)
        assert len(f.must) == 1
        cond = f.must[0]
        assert isinstance(cond, models.FieldCondition)
        assert cond.key == "metadata.source"
        assert cond.match == models.MatchValue(value="spec")

    def test_single_scalar_int(self):
        from qdrant_client import models
        from src.vector_store import to_qdrant_filter
        f = to_qdrant_filter({"idx": 5})
        assert len(f.must) == 1
        assert f.must[0].key == "metadata.idx"
        assert f.must[0].match == models.MatchValue(value=5)

    def test_single_scalar_bool(self):
        from qdrant_client import models
        from src.vector_store import to_qdrant_filter
        f = to_qdrant_filter({"active": True})
        assert len(f.must) == 1
        assert f.must[0].match == models.MatchValue(value=True)

    def test_float_scalar_becomes_range(self):
        """float не поддерживается MatchValue → Range(gte=lte=v)."""
        from src.vector_store import to_qdrant_filter
        f = to_qdrant_filter({"score": 2.5})
        assert len(f.must) == 1
        cond = f.must[0]
        assert cond.key == "metadata.score"
        assert cond.match is None
        assert cond.range.gte == 2.5
        assert cond.range.lte == 2.5

    def test_list_becomes_match_any(self):
        from qdrant_client import models
        from src.vector_store import to_qdrant_filter
        f = to_qdrant_filter({"type": ["bug", "feature"]})
        assert len(f.must) == 1
        cond = f.must[0]
        assert cond.key == "metadata.type"
        assert cond.match == models.MatchAny(any=["bug", "feature"])

    def test_list_floats_dropped(self):
        """float-элементы списков отбрасываются (MatchAny их не принимает)."""
        from qdrant_client import models
        from src.vector_store import to_qdrant_filter
        f = to_qdrant_filter({"type": ["bug", 1.5, "feature"]})
        assert len(f.must) == 1
        assert f.must[0].match == models.MatchAny(any=["bug", "feature"])

    def test_list_only_floats_gives_none(self):
        """Список из одних float → условий нет → фильтр отключён."""
        from src.vector_store import to_qdrant_filter
        assert to_qdrant_filter({"score": [1.5, 2.5]}) is None

    def test_multiple_keys_and(self):
        from src.vector_store import to_qdrant_filter
        f = to_qdrant_filter({"source": "spec", "type": "bug"})
        assert len(f.must) == 2
        keys = {c.key for c in f.must}
        assert keys == {"metadata.source", "metadata.type"}

    def test_dict_values_ignored(self):
        """dict-значения не поддерживаются → игнорируются."""
        from src.vector_store import to_qdrant_filter
        f = to_qdrant_filter({"nested": {"a": 1}, "source": "spec"})
        assert len(f.must) == 1
        assert f.must[0].key == "metadata.source"

    def test_only_ignored_values_gives_none(self):
        from src.vector_store import to_qdrant_filter
        assert to_qdrant_filter({"nested": {"a": 1}}) is None

    def test_list_with_dict_elements_cleaned(self):
        from qdrant_client import models
        from src.vector_store import to_qdrant_filter
        f = to_qdrant_filter({"type": ["bug", {"x": 1}, "feature"]})
        assert len(f.must) == 1
        assert f.must[0].match == models.MatchAny(any=["bug", "feature"])


# ===========================================================================
# src/rag.py — RAGSystem (интеграция через Qdrant embedded)
# ===========================================================================

def _seed(rag):
    """Добавить документы с разной метаданной."""
    rag.add_document("python programming language tutorial introduction for beginners", {"source": "spec", "type": "doc"})
    rag.add_document("java programming language enterprise guide for backend developers", {"source": "blog", "type": "doc"})
    rag.add_document("python machine learning neural network deep learning tutorial", {"source": "spec", "type": "tutorial"})
    rag.add_document("cooking recipe pasta italian food delicious meal preparation", {"source": "blog", "type": "tutorial"})


class TestRAGMetadataFilter:
    """RAGSystem: search/bm25_search/search_hybrid/list_documents/get_related."""

    def test_search_with_metadata_filter(self, rag):
        _seed(rag)
        results = rag.search("programming", k=10, metadata_filter={"source": "spec"})
        assert len(results) == 2
        for _doc_id, _text, _score, meta in results:
            assert meta.get("source") == "spec"

    def test_search_with_in_list_filter(self, rag):
        _seed(rag)
        results = rag.search("programming", k=10, metadata_filter={"type": ["doc", "tutorial"]})
        assert len(results) == 4
        for _doc_id, _text, _score, meta in results:
            assert meta.get("type") in ("doc", "tutorial")

    def test_search_without_filter_returns_all(self, rag):
        _seed(rag)
        results = rag.search("programming", k=10)
        # Без фильтра dense-поиск возвращает все документы (k=10 > корпуса)
        assert len(results) == 4

    def test_bm25_search_with_metadata_filter(self, rag):
        _seed(rag)
        results = rag.bm25_search("python", k=10, metadata_filter={"source": "spec"})
        # Только python+spec документы
        assert len(results) == 2
        for _doc_id, text, _score, meta in results:
            assert meta.get("source") == "spec"
            assert "python" in text.lower()

    def test_bm25_search_filter_excludes_non_matching(self, rag):
        _seed(rag)
        results = rag.bm25_search("programming", k=10, metadata_filter={"source": "blog"})
        # Только java doc (blog) содержит "programming" и source=blog
        assert len(results) >= 1
        for _doc_id, _text, _score, meta in results:
            assert meta.get("source") == "blog"

    def test_search_hybrid_with_metadata_filter(self, rag):
        _seed(rag)
        results = rag.search_hybrid("python", k=10, alpha=0.5, metadata_filter={"source": "spec"})
        assert results
        for _doc_id, _text, _score, meta in results:
            assert meta.get("source") == "spec"

    def test_search_hybrid_filter_applies_both_channels(self, rag):
        """Гибридный поиск: фильтр применяется к обоим каналам."""
        _seed(rag)
        # alpha=0.0 → чистый BM25; фильтр должен сработать
        results_bm = rag.search_hybrid("programming", k=10, alpha=0.0, metadata_filter={"source": "spec"})
        for _doc_id, _text, _score, meta in results_bm:
            assert meta.get("source") == "spec"
        # alpha=1.0 → чистая семантика; фильтр должен сработать
        results_sem = rag.search_hybrid("programming", k=10, alpha=1.0, metadata_filter={"source": "blog"})
        for _doc_id, _text, _score, meta in results_sem:
            assert meta.get("source") == "blog"

    def test_list_documents_with_metadata_filter(self, rag):
        _seed(rag)
        out = rag.list_documents(limit=20, offset=0, metadata_filter={"source": "spec"})
        # 2 документа с source=spec
        assert out["total"] == 2
        for d in out["documents"]:
            assert d["metadata"].get("source") == "spec"

    def test_list_documents_without_filter(self, rag):
        _seed(rag)
        out = rag.list_documents(limit=20, offset=0)
        assert out["total"] == 4

    def test_list_documents_empty_filter_no_filtering(self, rag):
        _seed(rag)
        out = rag.list_documents(limit=20, offset=0, metadata_filter={})
        assert out["total"] == 4

    def test_get_related_with_metadata_filter(self, rag):
        d1 = rag.add_document("first node document with sufficient length for the test here", {"type": "spec"})
        d2 = rag.add_document("second node document with sufficient length for the test here", {"type": "bug"})
        d3 = rag.add_document("third node document with sufficient length for the test here", {"type": "spec"})
        rag.add_relation(d1, d2, "related_to")
        rag.add_relation(d1, d3, "related_to")

        rels = rag.get_related(d1, max_depth=1, metadata_filter={"type": "spec"})
        targets = {r[1] for r in rels}
        assert targets == {d3}

    def test_get_related_without_filter(self, rag):
        d1 = rag.add_document("first node document with sufficient length for the test here", {"type": "spec"})
        d2 = rag.add_document("second node document with sufficient length for the test here", {"type": "bug"})
        rag.add_relation(d1, d2, "related_to")

        rels = rag.get_related(d1, max_depth=1)
        targets = {r[1] for r in rels}
        assert targets == {d2}

    def test_doc_without_metadata_excluded_by_filter(self, rag):
        """Документ без metadata (None/{}) не проходит активный фильтр."""
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

def _seed_mcp(rag):
    rag.add_document("python programming language tutorial introduction for beginners here", {"source": "spec"})
    rag.add_document("java programming language enterprise guide for developers here", {"source": "blog"})
    rag.add_document("python machine learning neural network deep learning tutorial here", {"source": "spec"})


class TestMCPMetadataFilter:
    """handle_tool_call пробрасывает metadata_filter в RAGSystem."""

    def test_rag_search_passes_metadata_filter(self, rag):
        from src.mcp_server import handle_tool_call
        _seed_mcp(rag)
        result = handle_tool_call(rag, "rag_search", {
            "query": "programming", "k": 10, "metadata_filter": {"source": "spec"},
        })
        assert len(result) == 2
        for r in result:
            assert r["metadata"].get("source") == "spec"

    def test_rag_bm25_search_passes_metadata_filter(self, rag):
        from src.mcp_server import handle_tool_call
        _seed_mcp(rag)
        result = handle_tool_call(rag, "rag_bm25_search", {
            "query": "python", "k": 10, "metadata_filter": {"source": "spec"},
        })
        assert result
        for r in result:
            assert r["metadata"].get("source") == "spec"

    def test_rag_search_hybrid_passes_metadata_filter(self, rag):
        from src.mcp_server import handle_tool_call
        _seed_mcp(rag)
        result = handle_tool_call(rag, "rag_search_hybrid", {
            "query": "python", "k": 10, "metadata_filter": {"source": "spec"},
        })
        assert result
        for r in result:
            assert r["metadata"].get("source") == "spec"

    def test_rag_list_documents_passes_metadata_filter(self, rag):
        from src.mcp_server import handle_tool_call
        _seed_mcp(rag)
        result = handle_tool_call(rag, "rag_list_documents", {
            "limit": 20, "metadata_filter": {"source": "spec"},
        })
        assert result["total"] == 2
        for d in result["documents"]:
            assert d["metadata"].get("source") == "spec"

    def test_rag_get_related_passes_metadata_filter(self, rag):
        from src.mcp_server import handle_tool_call
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

    def test_rag_search_without_metadata_filter_backward_compat(self, rag):
        """Без metadata_filter — обратная совместимость (возвращаются все)."""
        from src.mcp_server import handle_tool_call
        _seed_mcp(rag)
        result = handle_tool_call(rag, "rag_search", {"query": "programming", "k": 10})
        assert len(result) == 3

    def test_rag_search_metadata_filter_as_json_string(self, rag):
        """Регрессия: metadata_filter передан как JSON-строка (как делает MCP SDK).
        Не должен падать с AttributeError — должен распарситься в dict."""
        from src.mcp_server import handle_tool_call
        _seed_mcp(rag)
        result = handle_tool_call(rag, "rag_search", {
            "query": "programming", "k": 10,
            "metadata_filter": '{"source": "spec"}',
        })
        assert len(result) == 2
        for r in result:
            assert r["metadata"].get("source") == "spec"

    def test_rag_list_documents_metadata_filter_as_json_string(self, rag):
        """Регрессия: metadata_filter как JSON-строка для list_documents."""
        from src.mcp_server import handle_tool_call
        _seed_mcp(rag)
        result = handle_tool_call(rag, "rag_list_documents", {
            "limit": 20, "metadata_filter": '{"source": "spec"}',
        })
        assert result["total"] == 2
        for d in result["documents"]:
            assert d["metadata"].get("source") == "spec"

    def test_rag_search_hybrid_metadata_filter_as_json_string(self, rag):
        """Регрессия: metadata_filter как JSON-строка для search_hybrid."""
        from src.mcp_server import handle_tool_call
        _seed_mcp(rag)
        result = handle_tool_call(rag, "rag_search_hybrid", {
            "query": "python", "k": 10, "metadata_filter": '{"source": "spec"}',
        })
        assert result
        for r in result:
            assert r["metadata"].get("source") == "spec"

    def test_rag_search_invalid_json_metadata_filter_ignored(self, rag):
        """Невалидная JSON-строка в metadata_filter → фильтр отключается (не падает)."""
        from src.mcp_server import handle_tool_call
        _seed_mcp(rag)
        result = handle_tool_call(rag, "rag_search", {
            "query": "programming", "k": 10, "metadata_filter": "not a json",
        })
        # Фильтр отключен → все документы возвращаются
        assert len(result) == 3
