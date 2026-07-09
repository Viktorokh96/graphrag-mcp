"""Тесты для MCP сервера RAG (стек Qdrant embedded + SQLite).

Покрывают:
- handle_tool_call: каждый инструмент (add_document, add_file, search, stats, clear, relations, graph)
- _parse_meta: нормализация поля meta во всех формах (dict, None, "", JSON-строка, raw-строка)
- TOOL_DEFS: корректность JSON-схем, в частности поле meta не имеет type-ограничения
- Регрессия на баг: meta="" не должен отбиваться валидацией

Используются фикстуры rag/make_rag из conftest (RAGSystem + HashEmbeddingGenerator,
временный стор, автоматический close() до удаления tmp-директории — важно для Windows).
"""

from unittest.mock import MagicMock

import pytest


class TestParseMeta:
    """Регрессионные тесты для _parse_meta — нормализация поля meta."""

    def test_none_returns_none(self):
        from src.mcp_server import _parse_meta
        assert _parse_meta(None) is None

    def test_empty_string_returns_none(self):
        """Регрессия: пустая строка не должна вызывать ошибку валидации."""
        from src.mcp_server import _parse_meta
        assert _parse_meta("") is None

    def test_dict_passthrough(self):
        from src.mcp_server import _parse_meta
        result = _parse_meta({"source": "test", "idx": 1})
        assert result == {"source": "test", "idx": 1}

    def test_json_string_parsed(self):
        from src.mcp_server import _parse_meta
        result = _parse_meta('{"k": "v", "n": 42}')
        assert result == {"k": "v", "n": 42}

    def test_json_array_string_wrapped(self):
        """JSON-массив (не dict) оборачивается в {"_raw": value}."""
        from src.mcp_server import _parse_meta
        result = _parse_meta('[1, 2, 3]')
        assert result == {"_raw": "[1, 2, 3]"}

    def test_raw_string_wrapped(self):
        """Произвольная строка без JSON оборачивается в {"_raw": value}."""
        from src.mcp_server import _parse_meta
        result = _parse_meta("just text")
        assert result == {"_raw": "just text"}

    def test_number_wrapped(self):
        """Не-dict, не-string значение оборачивается в {"_raw": value}."""
        from src.mcp_server import _parse_meta
        result = _parse_meta(42)
        assert result == {"_raw": 42}


class TestToolDefs:
    """Регрессионные тесты для TOOL_DEFS — корректность JSON-схем."""

    def test_all_tools_have_required_fields(self):
        from src.mcp_server import TOOL_DEFS
        for tool in TOOL_DEFS:
            assert tool.name
            assert tool.description
            assert tool.inputSchema["type"] == "object"
            assert "properties" in tool.inputSchema

    def test_meta_field_has_no_type_constraint(self):
        """Регрессия: поле meta не должно иметь type-ограничение,
        иначе пустая строка или null отбиваются валидацией MCP SDK.
        См. _parse_meta — он сам нормализует значение."""
        from src.mcp_server import TOOL_DEFS
        tools_by_name = {t.name: t for t in TOOL_DEFS}
        for name in ("rag_add_document", "rag_add_file"):
            tool = tools_by_name[name]
            assert "meta" in tool.inputSchema["properties"]
            meta_schema = tool.inputSchema["properties"]["meta"]
            assert "type" not in meta_schema, (
                f"{name}: meta field has type={meta_schema.get('type')!r} — "
                "это ломает валидацию при meta='' или meta=null"
            )

    def test_meta_not_in_required(self):
        """Поле meta не должно быть обязательным."""
        from src.mcp_server import TOOL_DEFS
        tools_by_name = {t.name: t for t in TOOL_DEFS}
        for name in ("rag_add_document", "rag_add_file"):
            tool = tools_by_name[name]
            assert "meta" not in tool.inputSchema.get("required", [])


class TestHandleToolCall:
    """Тесты для handle_tool_call — каждая MCP-команда тестируется напрямую
    без поднятия stdio-сервера. RAGSystem — фикстура rag из conftest."""

    def test_add_document_without_meta(self, rag):
        from src.mcp_server import handle_tool_call
        result = handle_tool_call(rag, "rag_add_document", {"text": "hello world this is a test document for RAG storage system"})
        assert "doc_id" in result
        assert isinstance(result["doc_id"], str)
        assert result["duplicate"] is False, "first add should not be a duplicate"

    def test_add_document_duplicate_flag(self, rag):
        """D6: MCP возвращает duplicate=true при повторном добавлении того же контента."""
        from src.mcp_server import handle_tool_call
        text = "hello world this is a test document for RAG storage system"
        r1 = handle_tool_call(rag, "rag_add_document", {"text": text})
        assert r1["duplicate"] is False, "first add should not be a duplicate"
        r2 = handle_tool_call(rag, "rag_add_document", {"text": text})
        assert r2["duplicate"] is True, "second add should be flagged as duplicate"
        assert r2["doc_id"] == r1["doc_id"], "duplicate should return the same doc_id"

    def test_add_document_with_dict_meta(self, rag):
        from src.mcp_server import handle_tool_call
        result = handle_tool_call(rag, "rag_add_document", {
            "text": "hello world this is a test document for the RAG storage system",
            "meta": {"source": "test", "idx": 1},
        })
        doc_id = result["doc_id"]
        # Проверяем что метаданные дошли до хранилища документов
        record = rag.doc_store.get(doc_id)
        assert record is not None, "документ не найден в хранилище"
        assert record["metadata"] == {"source": "test", "idx": 1}

    def test_add_document_with_empty_string_meta(self, rag):
        """Регрессия: meta='' не должна ломать добавление документа."""
        from src.mcp_server import handle_tool_call
        result = handle_tool_call(rag, "rag_add_document", {"text": "hello world this is a test document for RAG storage", "meta": ""})
        assert "doc_id" in result

    def test_add_document_with_null_meta(self, rag):
        """Регрессия: meta=null не должна ломать добавление документа."""
        from src.mcp_server import handle_tool_call
        result = handle_tool_call(rag, "rag_add_document", {"text": "hello world this is a test document for RAG storage", "meta": None})
        assert "doc_id" in result

    def test_add_document_with_json_string_meta(self, rag):
        """Регрессия: meta='{"k":"v"}' должна распарситься в dict."""
        from src.mcp_server import handle_tool_call
        result = handle_tool_call(rag, "rag_add_document", {
            "text": "hello world this is a test for JSON string metadata parsing",
            "meta": '{"source": "json-string"}',
        })
        doc_id = result["doc_id"]
        record = rag.doc_store.get(doc_id)
        assert record is not None
        assert record["metadata"] == {"source": "json-string"}

    def test_add_file_without_meta(self, rag, tmp_path):
        from src.mcp_server import handle_tool_call
        test_file = tmp_path / "test.txt"
        test_file.write_text("File content for RAG system indexing and storage testing purposes", encoding="utf-8")
        result = handle_tool_call(rag, "rag_add_file", {"filepath": str(test_file)})
        assert "doc_id" in result

    def test_add_file_with_dict_meta(self, rag, tmp_path):
        from src.mcp_server import handle_tool_call
        test_file = tmp_path / "test.txt"
        test_file.write_text("File content for RAG system with metadata dict test purposes", encoding="utf-8")
        result = handle_tool_call(rag, "rag_add_file", {
            "filepath": str(test_file),
            "meta": {"source": "file", "path": str(test_file)},
        })
        doc_id = result["doc_id"]
        record = rag.doc_store.get(doc_id)
        assert record is not None
        assert record["metadata"] == {"source": "file", "path": str(test_file)}

    def test_add_file_with_empty_string_meta(self, rag, tmp_path):
        """Регрессия: meta='' для add_file тоже не должна ломаться."""
        from src.mcp_server import handle_tool_call
        test_file = tmp_path / "test.txt"
        test_file.write_text("file content for RAG indexing with empty string meta", encoding="utf-8")
        result = handle_tool_call(rag, "rag_add_file", {"filepath": str(test_file), "meta": ""})
        assert "doc_id" in result

    def test_search(self, rag):
        from src.mcp_server import handle_tool_call
        rag.add_document("Python programming language for general purpose scripting and automation")
        result = handle_tool_call(rag, "rag_search", {"query": "Python", "k": 5})
        assert isinstance(result, list)
        assert len(result) >= 1
        assert "doc_id" in result[0]
        assert "text" in result[0]
        assert "score" in result[0]

    def test_bm25_search(self, rag):
        from src.mcp_server import handle_tool_call
        rag.add_document("python programming language for general purpose scripting and automation")
        rag.add_document("java programming language for enterprise software development and scaling")
        result = handle_tool_call(rag, "rag_bm25_search", {"query": "python", "k": 1})
        assert isinstance(result, list)
        assert len(result) == 1
        assert "python" in result[0]["text"].lower()

    def test_search_hybrid(self, rag):
        from src.mcp_server import handle_tool_call
        rag.add_document("python programming language for general purpose scripting and automation")
        result = handle_tool_call(rag, "rag_search_hybrid", {
            "query": "python", "k": 1, "alpha": 0.5,
        })
        assert isinstance(result, list)
        assert len(result) == 1

    def test_stats(self, rag):
        from src.mcp_server import handle_tool_call
        result = handle_tool_call(rag, "rag_stats", {})
        assert "total_documents" in result
        assert "store_path" in result
        assert "dimension" in result
        # dimension коллекции Qdrant = размерность HashEmbeddingGenerator
        assert result["dimension"] == 64

    def test_clear(self, rag):
        from src.mcp_server import handle_tool_call
        rag.add_document("some text for the RAG system to index and store permanently")
        assert handle_tool_call(rag, "rag_stats", {})["total_documents"] > 0
        result = handle_tool_call(rag, "rag_clear", {})
        assert result == {"status": "ok"}
        assert handle_tool_call(rag, "rag_stats", {})["total_documents"] == 0

    def test_add_relation(self, rag):
        from src.mcp_server import handle_tool_call
        doc1 = rag.add_document("First document for graph and relation testing in RAG system")
        doc2 = rag.add_document("Second document for graph and relation testing in RAG system")
        result = handle_tool_call(rag, "rag_add_relation", {
            "source_id": doc1, "target_id": doc2,
            "relation": "related_to", "weight": 1.0,
        })
        assert result == {"status": "ok"}

    def test_get_related(self, rag):
        from src.mcp_server import handle_tool_call
        doc1 = rag.add_document("First document for graph and relation testing in RAG system")
        doc2 = rag.add_document("Second document for graph and relation testing in RAG system")
        rag.add_relation(doc1, doc2, "related_to", 1.0)
        result = handle_tool_call(rag, "rag_get_related", {"node_id": doc1, "max_depth": 1})
        assert "relations" in result
        assert len(result["relations"]) >= 1
        rel = result["relations"][0]
        assert rel["source"] == doc1
        assert rel["target"] == doc2
        assert rel["relation"] == "related_to"

    def test_graph_stats(self, rag):
        from src.mcp_server import handle_tool_call
        doc1 = rag.add_document("First document for graph and relation testing in RAG system")
        doc2 = rag.add_document("Second document for graph and relation testing in RAG system")
        rag.add_relation(doc1, doc2, "related_to", 1.0)
        result = handle_tool_call(rag, "rag_graph_stats", {})
        assert "total_nodes" in result
        assert "total_edges" in result
        assert "relation_types" in result
        assert result["total_edges"] >= 1

    def test_delete_document(self, rag):
        """rag_delete_document удаляет документ из всех хранилищ."""
        from src.mcp_server import handle_tool_call
        doc_id = rag.add_document("this document is going to be deleted and removed from storage")
        assert handle_tool_call(rag, "rag_stats", {})["total_documents"] == 1
        result = handle_tool_call(rag, "rag_delete_document", {"doc_id": doc_id})
        assert result["status"] == "ok"
        assert result["doc_id"] == doc_id
        assert result["deleted"] is True
        assert handle_tool_call(rag, "rag_stats", {})["total_documents"] == 0
        assert rag.doc_store.get(doc_id) is None
        assert rag.doc_store.count() == 0

    def test_delete_document_idempotent(self, rag):
        """Удаление несуществующего doc_id не вызывает ошибку."""
        from src.mcp_server import handle_tool_call
        result = handle_tool_call(rag, "rag_delete_document", {"doc_id": "nonexistent-uuid"})
        assert result["status"] == "ok"
        assert result["deleted"] is False

    def test_delete_document_removes_from_graph(self, rag):
        """Удаление документа удаляет также рёбра графа."""
        from src.mcp_server import handle_tool_call
        doc1 = rag.add_document("first document for testing graph deletion and cascading cleanup")
        doc2 = rag.add_document("second document related to first via graph edges for testing")
        rag.add_relation(doc1, doc2, "related_to", 1.0)
        assert handle_tool_call(rag, "rag_graph_stats", {})["total_nodes"] == 2
        handle_tool_call(rag, "rag_delete_document", {"doc_id": doc1})
        stats = handle_tool_call(rag, "rag_graph_stats", {})
        assert stats["total_nodes"] == 1
        assert stats["total_edges"] == 0

    def test_list_documents_empty(self, rag):
        """rag_list_documents на пустом хранилище возвращает пустой список."""
        from src.mcp_server import handle_tool_call
        result = handle_tool_call(rag, "rag_list_documents", {})
        assert result["documents"] == []
        assert result["total"] == 0
        assert result["limit"] == 20
        assert result["offset"] == 0

    # ── Relations inline (links field) ───────────────────────────────────

    def test_search_relations_depth_zero_has_empty_links(self, rag):
        """rag_search при relations_load_depth=0 возвращает links={}."""
        from src.mcp_server import handle_tool_call
        rag.add_document("python programming language for testing purposes example text content here")
        result = handle_tool_call(rag, "rag_search", {
            "query": "python", "k": 1, "relations_load_depth": 0,
        })
        assert "links" in result[0]
        assert result[0]["links"] == {}

    def test_search_relations_depth_one(self, rag):
        """rag_search с relations_load_depth=1 загружает соседей."""
        from src.mcp_server import handle_tool_call
        doc_a = rag.add_document("AAA test document for python relations link testing example")
        doc_b = rag.add_document("BBB test document for java relations link testing example content")
        rag.add_relation(doc_a, doc_b, "related_to", 1.0)

        result = handle_tool_call(rag, "rag_search", {
            "query": "AAA", "k": 5, "relations_load_depth": 1,
        })
        found = [r for r in result if r["doc_id"] == doc_a]
        assert found
        assert doc_b in found[0]["links"]

    def test_bm25_search_relations_depth_one(self, rag):
        """rag_bm25_search с relations_load_depth загружает соседей."""
        from src.mcp_server import handle_tool_call
        doc_a = rag.add_document("AAA test document for python relations link testing example")
        doc_b = rag.add_document("BBB test document for java relations link testing example content")
        rag.add_relation(doc_a, doc_b, "related_to", 1.0)

        result = handle_tool_call(rag, "rag_bm25_search", {
            "query": "AAA", "k": 5, "relations_load_depth": 1,
        })
        found = [r for r in result if r["doc_id"] == doc_a]
        assert found
        assert doc_b in found[0]["links"]

    def test_hybrid_search_relations_depth_one(self, rag):
        """rag_search_hybrid с relations_load_depth загружает соседей."""
        from src.mcp_server import handle_tool_call
        doc_a = rag.add_document("AAA test document for python relations link testing example")
        doc_b = rag.add_document("BBB test document for java relations link testing example content")
        rag.add_relation(doc_a, doc_b, "related_to", 1.0)

        result = handle_tool_call(rag, "rag_search_hybrid", {
            "query": "AAA", "k": 5, "relations_load_depth": 1, "alpha": 0.5,
        })
        found = [r for r in result if r["doc_id"] == doc_a]
        assert found
        assert doc_b in found[0]["links"]

    def test_get_document_with_relations_via_mcp(self, rag):
        """rag_get_document с relations_load_depth возвращает links."""
        from src.mcp_server import handle_tool_call
        doc_a = rag.add_document("AAA test document for python relations link testing example")
        doc_b = rag.add_document("BBB test document for java relations link testing example content")
        rag.add_relation(doc_a, doc_b, "related_to", 1.0)

        result = handle_tool_call(rag, "rag_get_document", {
            "doc_id": doc_a, "relations_load_depth": 1,
        })
        assert result is not None
        assert "links" in result
        assert doc_b in result["links"]

    def test_get_document_with_relations_depth_zero_via_mcp(self, rag):
        """rag_get_document при depth=0 возвращает links={}."""
        from src.mcp_server import handle_tool_call
        doc_a = rag.add_document("AAA test document for python relations link testing example")
        doc_b = rag.add_document("BBB test document for java relations link testing example content")
        rag.add_relation(doc_a, doc_b, "related_to", 1.0)

        result = handle_tool_call(rag, "rag_get_document", {
            "doc_id": doc_a, "relations_load_depth": 0,
        })
        assert result is not None
        assert result["links"] == {}

    def test_list_documents_with_relations_via_mcp(self, rag):
        """rag_list_documents с relations_load_depth возвращает links."""
        from src.mcp_server import handle_tool_call
        doc_a = rag.add_document("AAA test document for enrichment link testing purposes example text")
        doc_b = rag.add_document("BBB test document for enrichment link testing purposes example text")
        rag.add_relation(doc_a, doc_b, "related_to", 1.0)

        result = handle_tool_call(rag, "rag_list_documents", {
            "limit": 10, "relations_load_depth": 1,
        })
        for doc in result["documents"]:
            assert "links" in doc

    def test_list_documents_basic_fields(self, rag):
        """rag_list_documents возвращает doc_id/text/metadata и корректный total."""
        from src.mcp_server import handle_tool_call
        for i in range(5):
            rag.add_document(f"sample document number {i} for testing and analysis purposes")
        result = handle_tool_call(rag, "rag_list_documents", {"limit": 3, "offset": 0})
        assert result["total"] == 5
        assert len(result["documents"]) == 3
        assert "doc_id" in result["documents"][0]
        assert "text" in result["documents"][0]
        assert "metadata" in result["documents"][0]

    def test_list_documents_pagination(self, rag):
        """Пагинация: offset сдвигает страницу."""
        from src.mcp_server import handle_tool_call
        for i in range(10):
            rag.add_document(f"sample document number {i} for testing pagination and listing")
        page1 = handle_tool_call(rag, "rag_list_documents", {"limit": 3, "offset": 0})
        page2 = handle_tool_call(rag, "rag_list_documents", {"limit": 3, "offset": 3})
        assert page1["total"] == 10
        assert page2["total"] == 10
        assert len(page1["documents"]) == 3
        assert len(page2["documents"]) == 3
        page1_ids = {d["doc_id"] for d in page1["documents"]}
        page2_ids = {d["doc_id"] for d in page2["documents"]}
        assert page1_ids.isdisjoint(page2_ids)

    def test_list_documents_full_text_by_default(self, rag):
        """По умолчанию list_documents отдаёт полный текст без усечения."""
        from src.mcp_server import handle_tool_call
        long_text = "A" * 2000
        rag.add_document(long_text)
        result = handle_tool_call(rag, "rag_list_documents", {"limit": 1})
        assert result["documents"][0]["text"] == long_text

    def test_list_documents_max_chars_truncates(self, rag):
        """max_chars обрезает текст в list_documents."""
        from src.mcp_server import handle_tool_call
        rag.add_document("A" * 2000)
        result = handle_tool_call(rag, "rag_list_documents", {"limit": 1, "max_chars": 100})
        assert len(result["documents"][0]["text"]) == 100

    def test_search_full_text_by_default(self, rag):
        """По умолчанию rag_search отдаёт полный текст без усечения."""
        from src.mcp_server import handle_tool_call
        long_text = "Python " + "B" * 2000
        rag.add_document(long_text)
        result = handle_tool_call(rag, "rag_search", {"query": "Python", "k": 1})
        assert result[0]["text"] == long_text

    def test_search_max_chars_truncates(self, rag):
        """max_chars обрезает текст в результатах rag_search."""
        from src.mcp_server import handle_tool_call
        rag.add_document("Python " + "B" * 2000)
        result = handle_tool_call(rag, "rag_search", {"query": "Python", "k": 1, "max_chars": 50})
        assert len(result[0]["text"]) == 50

    def test_bm25_search_max_chars_truncates(self, rag):
        """max_chars обрезает текст в результатах rag_bm25_search."""
        from src.mcp_server import handle_tool_call
        rag.add_document("python " + "C" * 2000)
        result = handle_tool_call(rag, "rag_bm25_search", {"query": "python", "k": 1, "max_chars": 80})
        assert len(result[0]["text"]) == 80

    def test_hybrid_search_max_chars_truncates(self, rag):
        """max_chars обрезает текст в результатах rag_search_hybrid."""
        from src.mcp_server import handle_tool_call
        rag.add_document("python " + "D" * 2000)
        result = handle_tool_call(rag, "rag_search_hybrid", {
            "query": "python", "k": 1, "max_chars": 60,
        })
        assert len(result[0]["text"]) == 60

    def test_get_document_full_text(self, rag):
        """rag_get_document возвращает полный текст по умолчанию."""
        from src.mcp_server import handle_tool_call
        full_text = "Hello world " + "E" * 2000
        doc_id = rag.add_document(full_text)
        result = handle_tool_call(rag, "rag_get_document", {"doc_id": doc_id})
        assert result["doc_id"] == doc_id
        assert result["text"] == full_text
        assert result["total_chars"] == len(full_text)
        assert result["offset"] == 0
        assert result["limit"] is None

    def test_get_document_with_limit(self, rag):
        """rag_get_document с limit обрезает текст с начала."""
        from src.mcp_server import handle_tool_call
        full_text = "0123456789" * 100  # 1000 символов
        doc_id = rag.add_document(full_text)
        result = handle_tool_call(rag, "rag_get_document", {"doc_id": doc_id, "limit": 100})
        assert result["text"] == full_text[:100]
        assert result["total_chars"] == 1000
        assert result["offset"] == 0
        assert result["limit"] == 100

    def test_get_document_with_offset(self, rag):
        """rag_get_document с offset начинает чтение с середины."""
        from src.mcp_server import handle_tool_call
        full_text = "0123456789" * 100  # 1000 символов
        doc_id = rag.add_document(full_text)
        result = handle_tool_call(rag, "rag_get_document", {"doc_id": doc_id, "offset": 500})
        assert result["text"] == full_text[500:]
        assert result["offset"] == 500
        assert result["limit"] is None

    def test_get_document_with_offset_and_limit(self, rag):
        """rag_get_document с offset+limit читает страницу из середины."""
        from src.mcp_server import handle_tool_call
        full_text = "0123456789" * 100  # 1000 символов
        doc_id = rag.add_document(full_text)
        result = handle_tool_call(rag, "rag_get_document", {
            "doc_id": doc_id, "offset": 200, "limit": 100,
        })
        assert result["text"] == full_text[200:300]
        assert result["total_chars"] == 1000
        assert result["offset"] == 200
        assert result["limit"] == 100

    def test_get_document_pagination_covers_whole_text(self, rag):
        """Постраничное чтение: несколько вызовов с offset+limit собирают весь текст."""
        from src.mcp_server import handle_tool_call
        full_text = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789abcdefghijklmnopqrstuv"
        doc_id = rag.add_document(full_text)
        collected = ""
        offset = 0
        page_size = 10
        while offset < len(full_text):
            page = handle_tool_call(rag, "rag_get_document", {
                "doc_id": doc_id, "offset": offset, "limit": page_size,
            })
            collected += page["text"]
            offset += page_size
        assert collected == full_text

    def test_get_document_nonexistent_returns_none(self, rag):
        """rag_get_document для несуществующего ID возвращает None."""
        from src.mcp_server import handle_tool_call
        result = handle_tool_call(rag, "rag_get_document", {"doc_id": "nonexistent-uuid"})
        assert result is None

    def test_unknown_tool_raises(self):
        from src.mcp_server import handle_tool_call
        rag = MagicMock()
        with pytest.raises(ValueError, match="Unknown tool"):
            handle_tool_call(rag, "nonexistent_tool", {})

    def test_none_arguments_handled(self):
        from src.mcp_server import handle_tool_call
        rag = MagicMock()
        rag.stats.return_value = {
            "total_documents": 0, "store_path": "/tmp", "dimension": 1,
            "total_nodes": 0, "total_edges": 0, "relation_types": [],
        }
        # arguments=None не должен падать с AttributeError
        result = handle_tool_call(rag, "rag_stats", None)  # type: ignore[arg-type]
        assert "total_documents" in result


class TestMCPServerModuleImport:
    """Проверка что модуль импортируется без ошибок."""

    def test_module_imports(self):
        import src.mcp_server as mod
        assert hasattr(mod, "TOOL_DEFS")
        assert hasattr(mod, "handle_tool_call")
        assert hasattr(mod, "_parse_meta")
        assert hasattr(mod, "main")
        assert len(mod.TOOL_DEFS) == 13

    def test_main_callable(self):
        """main() должен быть вызываемым (но не вызываем — он запускает stdio)."""
        import src.mcp_server as mod
        assert callable(mod.main)
