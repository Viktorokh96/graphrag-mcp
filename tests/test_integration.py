"""Интеграционные тесты для RAG MCP server.

Покрывают полные сценарии через handle_tool_call с временным хранилищем.
Используется фикстура `rag` из conftest.py: RAGSystem на Qdrant + SQLite во
временном каталоге с HashEmbeddingGenerator и автоматическим close().

Сценарии:
- Полный lifecycle: add → list → search → delete
- Пагинация list_documents
- Графовые связи: add_relation → get_related → delete с рёбрами
- Идемпотентное удаление
- Удаление очищает vector store (dense + BM25) и граф одновременно
- Совместное использование search (semantic, BM25, hybrid) после удаления
"""


def _call(rag, tool_name, **kwargs):
    """Вызвать MCP-инструмент через handle_tool_call."""
    from src.mcp_server import handle_tool_call
    return handle_tool_call(rag, tool_name, kwargs)


def _add(rag, text, meta=None):
    """Добавить документ и вернуть doc_id (строка)."""
    kwargs = {"text": text}
    if meta is not None:
        kwargs["meta"] = meta
    return _call(rag, "rag_add_document", **kwargs)["doc_id"]


class TestFullLifecycle:
    """Полный lifecycle: add → list → search → delete."""

    def test_add_list_search_delete(self, rag):
        doc_id = _add(rag, "Python programming language for general purpose scripting")

        listed = _call(rag, "rag_list_documents", limit=10, offset=0)
        assert listed["total"] == 1
        assert listed["documents"][0]["doc_id"] == doc_id

        result = _call(rag, "rag_delete_document", doc_id=doc_id)
        assert result["status"] == "ok"
        assert result["deleted"] is True

        assert _call(rag, "rag_stats")["total_documents"] == 0
        listed = _call(rag, "rag_list_documents")
        assert listed["total"] == 0
        assert listed["documents"] == []


class TestListDocumentsPagination:
    """Пагинация rag_list_documents."""

    def test_pagination_disjoint_pages(self, rag):
        for i in range(10):
            _add(rag, f"sample document number {i} for testing pagination and analysis purposes")

        page1 = _call(rag, "rag_list_documents", limit=3, offset=0)
        page2 = _call(rag, "rag_list_documents", limit=3, offset=3)

        assert page1["total"] == 10
        assert page2["total"] == 10
        assert len(page1["documents"]) == 3
        assert len(page2["documents"]) == 3

        ids1 = {d["doc_id"] for d in page1["documents"]}
        ids2 = {d["doc_id"] for d in page2["documents"]}
        assert ids1.isdisjoint(ids2)

    def test_offset_beyond_total_returns_empty(self, rag):
        _add(rag, "only one document in the entire storage system right now")
        result = _call(rag, "rag_list_documents", limit=10, offset=100)
        assert result["total"] == 1
        assert result["documents"] == []

    def test_default_limit_and_offset(self, rag):
        _add(rag, "single document in storage for testing default limit and offset")
        result = _call(rag, "rag_list_documents")
        assert result["limit"] == 20
        assert result["offset"] == 0

    def test_empty_store(self, rag):
        result = _call(rag, "rag_list_documents")
        assert result["documents"] == []
        assert result["total"] == 0
        assert result["limit"] == 20
        assert result["offset"] == 0


class TestDeleteDocument:
    """Удаление документов — все хранилища, идемпотентность, рёбра графа."""

    def test_delete_removes_from_all_stores(self, rag):
        doc_id = _add(rag, "this document is going to be deleted from all stores and cleaned up")
        assert _call(rag, "rag_stats")["total_documents"] == 1

        _call(rag, "rag_delete_document", doc_id=doc_id)

        assert _call(rag, "rag_stats")["total_documents"] == 0
        stats = _call(rag, "rag_graph_stats")
        assert stats["total_nodes"] == 0
        assert stats["total_edges"] == 0

    def test_delete_idempotent(self, rag):
        result = _call(rag, "rag_delete_document", doc_id="nonexistent-uuid")
        assert result["status"] == "ok"
        assert result["deleted"] is False

    def test_delete_removes_graph_edges(self, rag):
        doc1 = _add(rag, "first document for testing graph edge removal and cascading cleanup")
        doc2 = _add(rag, "second document related to the first one via graph edge for tests")
        _call(rag, "rag_add_relation",
              source_id=doc1, target_id=doc2, relation="related_to", weight=1.0)

        assert _call(rag, "rag_graph_stats")["total_edges"] == 1

        _call(rag, "rag_delete_document", doc_id=doc1)

        stats = _call(rag, "rag_graph_stats")
        assert stats["total_nodes"] == 1
        assert stats["total_edges"] == 0

        related = _call(rag, "rag_get_related", node_id=doc1, max_depth=1)
        assert related["relations"] == []

    def test_delete_does_not_affect_other_docs(self, rag):
        doc1 = _add(rag, "this document should be kept and remain in the storage system")
        doc2 = _add(rag, "this document will be deleted from the storage system entirely")

        _call(rag, "rag_delete_document", doc_id=doc2)

        listed = _call(rag, "rag_list_documents")
        assert listed["total"] == 1
        assert listed["documents"][0]["doc_id"] == doc1


class TestSearchAfterDelete:
    """Поиск (semantic, BM25, hybrid) после удаления."""

    def test_bm25_search_excludes_deleted(self, rag):
        _add(rag, "python programming language for scripting and automation tasks")
        doc2 = _add(rag, "java programming language for enterprise software development")

        _call(rag, "rag_delete_document", doc_id=doc2)

        results = _call(rag, "rag_bm25_search", query="programming", k=5)
        assert len(results) == 1
        assert "python" in results[0]["text"].lower()

    def test_semantic_search_excludes_deleted(self, rag):
        keep_id = _add(rag, "python programming language for scripting and automation")
        delete_id = _add(rag, "java programming language for enterprise software systems")

        _call(rag, "rag_delete_document", doc_id=delete_id)

        results = _call(rag, "rag_search", query="python", k=5)
        assert len(results) >= 1
        assert results[0]["doc_id"] == keep_id

    def test_hybrid_search_excludes_deleted(self, rag):
        _add(rag, "python programming language for scripting and automation")
        delete_id = _add(rag, "java programming language for enterprise software systems")

        _call(rag, "rag_delete_document", doc_id=delete_id)

        results = _call(rag, "rag_search_hybrid", query="python", k=5, alpha=0.5)
        assert len(results) >= 1
        deleted_ids = [r["doc_id"] for r in results if r["doc_id"] == delete_id]
        assert deleted_ids == []


class TestGraphWorkflow:
    """Полный графовый сценарий: add → relate → get_related → delete."""

    def test_full_graph_workflow(self, rag):
        doc1 = _add(rag, "orchestrator document for dispatching events and managing workflows")
        doc2 = _add(rag, "agent runtime document for executing tasks and processing data")
        doc3 = _add(rag, "tests agent document for generating and running automated test suites")

        _call(rag, "rag_add_relation",
              source_id=doc1, target_id=doc2, relation="dispatches_to", weight=0.9)
        _call(rag, "rag_add_relation",
              source_id=doc3, target_id=doc2, relation="runs_on", weight=0.8)

        related = _call(rag, "rag_get_related", node_id=doc1, max_depth=1)
        assert len(related["relations"]) == 1
        assert related["relations"][0]["target"] == doc2

        _call(rag, "rag_delete_document", doc_id=doc1)

        related = _call(rag, "rag_get_related", node_id=doc3, max_depth=1)
        assert len(related["relations"]) == 1
        assert related["relations"][0]["source"] == doc3
        assert related["relations"][0]["target"] == doc2

        stats = _call(rag, "rag_graph_stats")
        assert stats["total_nodes"] == 2
        assert stats["total_edges"] == 1


class TestStatsConsistency:
    """Консистентность stats после операций."""

    def test_stats_after_add_and_delete(self, rag):
        ids = [_add(rag, f"sample document number {i} for testing stats and consistency") for i in range(5)]

        assert _call(rag, "rag_stats")["total_documents"] == 5
        assert _call(rag, "rag_graph_stats")["total_nodes"] == 5

        _call(rag, "rag_delete_document", doc_id=ids[0])
        _call(rag, "rag_delete_document", doc_id=ids[1])

        assert _call(rag, "rag_stats")["total_documents"] == 3
        assert _call(rag, "rag_graph_stats")["total_nodes"] == 3
