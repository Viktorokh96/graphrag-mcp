"""Тесты для rag_update_document и rag_delete_relation (MCP + нижние слои).

Покрывают:
- rag_update_document: обновление текста, метаданных, обоих, ошибки
- rag_delete_relation: удаление рёбер, идемпотентность
- handle_tool_call: MCP-хэндлеры для обоих инструментов
- TOOL_DEFS: наличие новых инструментов в реестре
- DocumentStore.update_text / update_metadata: нижнеуровневые методы
- VectorStore.replace: замена точек

Используются фикстуры rag/make_rag из conftest.
"""

import pytest


# ── DocumentStore: update_text / update_metadata ──────────────────────────────


class TestDocumentStoreUpdate:
    """Unit-тесты методов обновления DocumentStore."""

    def test_update_text(self, rag):
        doc_id = rag.add_document("hello world this is a test document for RAG storage system")
        record = rag.doc_store.get(doc_id)
        old_hash = record["content_hash"]

        new_text = "updated content for this test document to verify text update works"
        result = rag.doc_store.update_text(doc_id, new_text, "new_hash")
        assert result is True

        updated = rag.doc_store.get(doc_id)
        assert updated["text"] == new_text
        assert updated["content_hash"] == "new_hash"

    def test_update_text_nonexistent(self, rag):
        result = rag.doc_store.update_text("nonexistent-id", "text", "hash")
        assert result is False

    def test_update_metadata(self, rag):
        doc_id = rag.add_document("hello world this is a test document for RAG storage system",
                                  meta={"source": "test"})
        result = rag.doc_store.update_metadata(doc_id, {"source": "updated", "tag": "v2"})
        assert result is True

        updated = rag.doc_store.get(doc_id)
        assert updated["metadata"] == {"source": "updated", "tag": "v2"}

    def test_update_metadata_nonexistent(self, rag):
        result = rag.doc_store.update_metadata("nonexistent-id", {"key": "val"})
        assert result is False


# ── VectorStore: replace ─────────────────────────────────────────────────────


class TestVectorStoreReplace:
    """Unit-тест метода replace — замена точек документа."""

    def test_replace_simple(self, rag):
        doc_id = rag.add_document("hello world this is a test document for RAG storage system")
        assert rag.vector_store.has(doc_id)

        new_text = "completely new text content for vector store replacement test case"
        embedding = rag.embedding_generator.get_embedding(new_text)
        rag.vector_store.replace(doc_id, new_text, embedding, {"source": "replaced"})

        assert rag.vector_store.has(doc_id)
        # BM25-поиск по новому тексту должен найти документ
        hits = rag.vector_store.bm25_search("replacement test case", k=5)
        assert any(h["doc_id"] == doc_id for h in hits)

    def test_replace_removes_old_bm25_tokens(self, rag):
        doc_id = rag.add_document("hello world this is a test document for RAG storage system")
        # Старый текст содержит "storage" — ищем
        hits_old = rag.vector_store.bm25_search("storage", k=5)
        assert any(h["doc_id"] == doc_id for h in hits_old)

        new_text = "completely new content with different keywords for verification only here"
        embedding = rag.embedding_generator.get_embedding(new_text)
        rag.vector_store.replace(doc_id, new_text, embedding)

        # "storage" больше не должен находиться в этом документе
        hits_new = rag.vector_store.bm25_search("storage", k=10)
        assert not any(h["doc_id"] == doc_id for h in hits_new)


# ── RAGSystem: update_document ───────────────────────────────────────────────


class TestRAGUpdateDocument:
    """Интеграционные тесты RAGSystem.update_document."""

    def test_update_text_only(self, rag):
        doc_id = rag.add_document("hello world this is a test document for RAG storage system")
        old_hash = rag.doc_store.get(doc_id)["content_hash"]

        result = rag.update_document(doc_id, text="brand new text content for the updated document")
        assert result == {"doc_id": doc_id, "updated": True}

        record = rag.doc_store.get(doc_id)
        assert record["text"] == "brand new text content for the updated document"
        assert record["content_hash"] != old_hash

    def test_update_meta_only(self, rag):
        doc_id = rag.add_document("hello world this is a test document for RAG storage system",
                                  meta={"source": "test"})
        result = rag.update_document(doc_id, meta={"source": "updated"})
        assert result == {"doc_id": doc_id, "updated": True}

        record = rag.doc_store.get(doc_id)
        assert record["metadata"] == {"source": "updated"}
        # Текст не изменился
        assert "test document" in record["text"]

    def test_update_text_and_meta(self, rag):
        doc_id = rag.add_document("hello world this is a test document for RAG storage system",
                                  meta={"source": "test"})
        result = rag.update_document(doc_id, text="updated text for combined metadata update test",
                                     meta={"source": "updated", "version": 2})
        assert result["updated"] is True

        record = rag.doc_store.get(doc_id)
        assert record["text"] == "updated text for combined metadata update test"
        assert record["metadata"]["source"] == "updated"
        assert record["metadata"]["version"] == 2

    def test_update_preserves_doc_id(self, rag):
        doc_id = rag.add_document("hello world this is a test document for RAG storage system")
        rag.update_document(doc_id, text="completely different text content for doc_id preservation test")
        # doc_id не изменился
        record = rag.doc_store.get(doc_id)
        assert record is not None
        assert record["doc_id"] == doc_id

    def test_update_preserves_relations(self, rag):
        d1 = rag.add_document("hello world this is a test document for RAG storage system")
        d2 = rag.add_document("second document for relation preservation testing purposes")
        rag.add_relation(d1, d2, "related_to")

        # Обновляем d1
        rag.update_document(d1, text="updated first document for relation preservation testing")

        # Связь должна сохраниться
        related = rag.get_related(d1)
        assert len(related) >= 1
        assert any(r[1] == d2 for r in related)

    def test_update_short_text_raises(self, rag):
        doc_id = rag.add_document("hello world this is a test document for RAG storage system")
        with pytest.raises(ValueError, match="too short"):
            rag.update_document(doc_id, text="short")

    def test_update_nothing_raises(self, rag):
        doc_id = rag.add_document("hello world this is a test document for RAG storage system")
        with pytest.raises(ValueError, match="at least one"):
            rag.update_document(doc_id)

    def test_update_nonexistent_raises(self, rag):
        with pytest.raises(ValueError, match="not found"):
            rag.update_document("nonexistent-id", text="new text for nonexistent document test")

    def test_update_reindexes_vector_store(self, rag):
        doc_id = rag.add_document("hello world this is a test document for RAG storage system")
        # BM25 поиск по старому тексту
        hits = rag.vector_store.bm25_search("storage system", k=5)
        assert any(h["doc_id"] == doc_id for h in hits)

        rag.update_document(doc_id, text="brand new content with unique keyword zzzzzzz for search test")
        # BM25 поиск по новому тексту
        hits_new = rag.vector_store.bm25_search("zzzzzzz", k=5)
        assert any(h["doc_id"] == doc_id for h in hits_new)

    def test_update_idempotent_same_text(self, rag):
        """Повторное обновление с тем же текстом не создаёт дубликат."""
        doc_id = rag.add_document("hello world this is a test document for RAG storage system")
        text = "brand new text content for idempotent update verification test case"
        rag.update_document(doc_id, text=text)
        rag.update_document(doc_id, text=text)

        # В сторе ровно один документ с этим content_hash
        content_hash = rag._compute_hash(text)
        found = rag.doc_store.find_by_hash(content_hash)
        assert found == doc_id


# ── RAGSystem: delete_relation ───────────────────────────────────────────────


class TestRAGDeleteRelation:
    """Интеграционные тесты RAGSystem.delete_relation."""

    def test_delete_relation(self, rag):
        d1 = rag.add_document("hello world this is a test document for RAG storage system")
        d2 = rag.add_document("second document for relation deletion testing purposes")
        rag.add_relation(d1, d2, "related_to")

        # Проверяем что ребро есть
        related = rag.get_related(d1)
        assert len(related) >= 1

        result = rag.delete_relation(d1, d2, "related_to")
        assert result == {"status": "ok", "deleted": True}

        # Ребра больше нет
        related_after = rag.get_related(d1)
        assert not any(r[1] == d2 and r[2] == "related_to" for r in related_after)

    def test_delete_relation_idempotent(self, rag):
        d1 = rag.add_document("hello world this is a test document for RAG storage system")
        d2 = rag.add_document("second document for idempotent deletion testing")
        rag.add_relation(d1, d2, "related_to")

        rag.delete_relation(d1, d2, "related_to")
        # Повторное удаление — не ошибка
        result = rag.delete_relation(d1, d2, "related_to")
        assert result == {"status": "ok", "deleted": True}

    def test_delete_relation_nonexistent(self, rag):
        d1 = rag.add_document("hello world this is a test document for RAG storage system")
        d2 = rag.add_document("second document for nonexistent edge deletion test")
        # Удаление несуществующего ребра — не ошибка
        result = rag.delete_relation(d1, d2, "nonexistent")
        assert result == {"status": "ok", "deleted": True}


# ── MCP handle_tool_call ─────────────────────────────────────────────────────


class TestMCPUpdateDocument:
    """Тесты MCP-хэндлера rag_update_document."""

    def test_update_text(self, rag):
        from src.mcp_server import handle_tool_call
        doc_id = handle_tool_call(rag, "rag_add_document", {
            "text": "hello world this is a test document for RAG storage system"
        })["doc_id"]

        result = handle_tool_call(rag, "rag_update_document", {
            "doc_id": doc_id,
            "text": "updated text content for MCP handler verification test",
        })
        assert result["doc_id"] == doc_id
        assert result["updated"] is True

    def test_update_meta(self, rag):
        from src.mcp_server import handle_tool_call
        doc_id = handle_tool_call(rag, "rag_add_document", {
            "text": "hello world this is a test document for RAG storage system"
        })["doc_id"]

        result = handle_tool_call(rag, "rag_update_document", {
            "doc_id": doc_id,
            "meta": {"source": "mcp-test"},
        })
        assert result["updated"] is True

        record = rag.doc_store.get(doc_id)
        assert record["metadata"] == {"source": "mcp-test"}

    def test_update_with_json_string_meta(self, rag):
        """Метаданные в виде JSON-строки парсятся через _parse_meta."""
        from src.mcp_server import handle_tool_call
        doc_id = handle_tool_call(rag, "rag_add_document", {
            "text": "hello world this is a test document for RAG storage system"
        })["doc_id"]

        result = handle_tool_call(rag, "rag_update_document", {
            "doc_id": doc_id,
            "meta": '{"source": "json-string-update"}',
        })
        assert result["updated"] is True

        record = rag.doc_store.get(doc_id)
        assert record["metadata"] == {"source": "json-string-update"}


class TestMCPDeleteRelation:
    """Тесты MCP-хэндлера rag_delete_relation."""

    def test_delete_relation(self, rag):
        from src.mcp_server import handle_tool_call
        d1 = handle_tool_call(rag, "rag_add_document", {
            "text": "hello world this is a test document for RAG storage system"
        })["doc_id"]
        d2 = handle_tool_call(rag, "rag_add_document", {
            "text": "second document for MCP relation deletion testing purposes"
        })["doc_id"]
        handle_tool_call(rag, "rag_add_relation", {
            "source_id": d1, "target_id": d2, "relation": "related_to"
        })

        result = handle_tool_call(rag, "rag_delete_relation", {
            "source_id": d1, "target_id": d2, "relation": "related_to"
        })
        assert result["status"] == "ok"
        assert result["deleted"] is True

        # Проверяем через get_related
        related = handle_tool_call(rag, "rag_get_related", {"node_id": d1})
        assert not any(r["target"] == d2 and r["relation"] == "related_to"
                       for r in related["relations"])


# ── TOOL_DEFS ────────────────────────────────────────────────────────────────


class TestNewToolDefs:
    """Проверяем наличие новых инструментов в TOOL_DEFS."""

    def test_update_document_tool_exists(self):
        from src.mcp_server import TOOL_DEFS
        names = {t.name for t in TOOL_DEFS}
        assert "rag_update_document" in names

    def test_delete_relation_tool_exists(self):
        from src.mcp_server import TOOL_DEFS
        names = {t.name for t in TOOL_DEFS}
        assert "rag_delete_relation" in names

    def test_update_document_schema(self):
        from src.mcp_server import TOOL_DEFS
        tool = next(t for t in TOOL_DEFS if t.name == "rag_update_document")
        schema = tool.inputSchema
        assert "doc_id" in schema["properties"]
        assert "text" in schema["properties"]
        assert "meta" in schema["properties"]
        assert "doc_id" in schema["required"]

    def test_delete_relation_schema(self):
        from src.mcp_server import TOOL_DEFS
        tool = next(t for t in TOOL_DEFS if t.name == "rag_delete_relation")
        schema = tool.inputSchema
        assert "source_id" in schema["properties"]
        assert "target_id" in schema["properties"]
        assert "relation" in schema["properties"]
        assert set(schema["required"]) == {"source_id", "target_id", "relation"}
