"""Тесты для MCP сервера RAG.

Покрывают:
- handle_tool_call: каждый инструмент (add_document, add_file, search, stats, clear, relations, graph)
- _parse_meta: нормализация поля meta во всех формах (dict, None, "", JSON-строка, raw-строка)
- TOOL_DEFS: корректность JSON-схем, в частности поле meta не имеет type-ограничения
- Регрессия на баг: meta="" не должен отбиваться валидацией
"""

import json
import os
import shutil
import tempfile
from unittest.mock import MagicMock, patch

import pytest


def _mock_embedding_response():
    """Мок ответа OpenRouter/OpenAI embeddings API."""
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"data": [{"embedding": [0.1, 0.2, 0.3]}]}
    mock_client = MagicMock()
    mock_client.__enter__.return_value = mock_client
    mock_client.post.return_value = mock_response
    return mock_client


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
    без поднятия stdio-сервера."""

    @pytest.fixture(autouse=True)
    def setup_temp_dir(self):
        self.temp_dir = tempfile.mkdtemp()
        self.store_path = os.path.join(self.temp_dir, "rag_data")
        yield
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def _make_rag(self, mock_httpx):
        from src.rag import RAGSystem
        mock_httpx.return_value = _mock_embedding_response()
        return RAGSystem(store_path=self.store_path, api_key="test-key")

    @patch("src.embeddings.httpx.Client")
    def test_add_document_without_meta(self, mock_httpx):
        from src.mcp_server import handle_tool_call
        rag = self._make_rag(mock_httpx)
        result = handle_tool_call(rag, "rag_add_document", {"text": "hello world"})
        assert "doc_id" in result
        assert isinstance(result["doc_id"], str)

    @patch("src.embeddings.httpx.Client")
    def test_add_document_with_dict_meta(self, mock_httpx):
        from src.mcp_server import handle_tool_call
        rag = self._make_rag(mock_httpx)
        result = handle_tool_call(rag, "rag_add_document", {
            "text": "hello",
            "meta": {"source": "test", "idx": 1},
        })
        doc_id = result["doc_id"]
        # Проверяем что метаданные дошли до хранилища
        all_docs = rag.vector_store.get_all()
        found = [d for d in all_docs if d[0] == doc_id]
        assert found, "документ не найден в хранилище"
        assert found[0][2] == {"source": "test", "idx": 1}

    @patch("src.embeddings.httpx.Client")
    def test_add_document_with_empty_string_meta(self, mock_httpx):
        """Регрессия: meta='' не должна ломать добавление документа."""
        from src.mcp_server import handle_tool_call
        rag = self._make_rag(mock_httpx)
        result = handle_tool_call(rag, "rag_add_document", {"text": "hello", "meta": ""})
        assert "doc_id" in result

    @patch("src.embeddings.httpx.Client")
    def test_add_document_with_null_meta(self, mock_httpx):
        """Регрессия: meta=null не должна ломать добавление документа."""
        from src.mcp_server import handle_tool_call
        rag = self._make_rag(mock_httpx)
        result = handle_tool_call(rag, "rag_add_document", {"text": "hello", "meta": None})
        assert "doc_id" in result

    @patch("src.embeddings.httpx.Client")
    def test_add_document_with_json_string_meta(self, mock_httpx):
        """Регрессия: meta='{"k":"v"}' должна распарситься в dict."""
        from src.mcp_server import handle_tool_call
        rag = self._make_rag(mock_httpx)
        result = handle_tool_call(rag, "rag_add_document", {
            "text": "hello",
            "meta": '{"source": "json-string"}',
        })
        doc_id = result["doc_id"]
        all_docs = rag.vector_store.get_all()
        found = [d for d in all_docs if d[0] == doc_id]
        assert found
        assert found[0][2] == {"source": "json-string"}

    @patch("src.embeddings.httpx.Client")
    def test_add_file_without_meta(self, mock_httpx):
        from src.mcp_server import handle_tool_call
        rag = self._make_rag(mock_httpx)
        test_file = os.path.join(self.temp_dir, "test.txt")
        with open(test_file, "w") as f:
            f.write("File content for RAG")
        result = handle_tool_call(rag, "rag_add_file", {"filepath": test_file})
        assert "doc_id" in result

    @patch("src.embeddings.httpx.Client")
    def test_add_file_with_dict_meta(self, mock_httpx):
        from src.mcp_server import handle_tool_call
        rag = self._make_rag(mock_httpx)
        test_file = os.path.join(self.temp_dir, "test.txt")
        with open(test_file, "w") as f:
            f.write("File content")
        result = handle_tool_call(rag, "rag_add_file", {
            "filepath": test_file,
            "meta": {"source": "file", "path": test_file},
        })
        doc_id = result["doc_id"]
        all_docs = rag.vector_store.get_all()
        found = [d for d in all_docs if d[0] == doc_id]
        assert found
        assert found[0][2] == {"source": "file", "path": test_file}

    @patch("src.embeddings.httpx.Client")
    def test_add_file_with_empty_string_meta(self, mock_httpx):
        """Регрессия: meta='' для add_file тоже не должна ломаться."""
        from src.mcp_server import handle_tool_call
        rag = self._make_rag(mock_httpx)
        test_file = os.path.join(self.temp_dir, "test.txt")
        with open(test_file, "w") as f:
            f.write("content")
        result = handle_tool_call(rag, "rag_add_file", {"filepath": test_file, "meta": ""})
        assert "doc_id" in result

    @patch("src.embeddings.httpx.Client")
    def test_search(self, mock_httpx):
        from src.mcp_server import handle_tool_call
        rag = self._make_rag(mock_httpx)
        rag.add_document("Python programming language")
        result = handle_tool_call(rag, "rag_search", {"query": "Python", "k": 5})
        assert isinstance(result, list)
        assert len(result) >= 1
        assert "doc_id" in result[0]
        assert "text" in result[0]
        assert "score" in result[0]

    @patch("src.embeddings.httpx.Client")
    def test_bm25_search(self, mock_httpx):
        from src.mcp_server import handle_tool_call
        rag = self._make_rag(mock_httpx)
        rag.add_document("python programming")
        rag.add_document("java programming")
        result = handle_tool_call(rag, "rag_bm25_search", {"query": "python", "k": 1})
        assert isinstance(result, list)
        assert len(result) == 1
        assert "python" in result[0]["text"].lower()

    @patch("src.embeddings.httpx.Client")
    def test_search_hybrid(self, mock_httpx):
        from src.mcp_server import handle_tool_call
        rag = self._make_rag(mock_httpx)
        rag.add_document("python programming")
        result = handle_tool_call(rag, "rag_search_hybrid", {
            "query": "python", "k": 1, "alpha": 0.5,
        })
        assert isinstance(result, list)
        assert len(result) == 1

    @patch("src.embeddings.httpx.Client")
    def test_stats(self, mock_httpx):
        from src.mcp_server import handle_tool_call
        rag = self._make_rag(mock_httpx)
        result = handle_tool_call(rag, "rag_stats", {})
        assert "total_documents" in result
        assert "store_path" in result
        assert "dimension" in result

    @patch("src.embeddings.httpx.Client")
    def test_clear(self, mock_httpx):
        from src.mcp_server import handle_tool_call
        rag = self._make_rag(mock_httpx)
        rag.add_document("some text")
        assert handle_tool_call(rag, "rag_stats", {})["total_documents"] > 0
        result = handle_tool_call(rag, "rag_clear", {})
        assert result == {"status": "ok"}
        assert handle_tool_call(rag, "rag_stats", {})["total_documents"] == 0

    @patch("src.embeddings.httpx.Client")
    def test_add_relation(self, mock_httpx):
        from src.mcp_server import handle_tool_call
        rag = self._make_rag(mock_httpx)
        doc1 = rag.add_document("First document")
        doc2 = rag.add_document("Second document")
        result = handle_tool_call(rag, "rag_add_relation", {
            "source_id": doc1, "target_id": doc2,
            "relation": "related_to", "weight": 1.0,
        })
        assert result == {"status": "ok"}

    @patch("src.embeddings.httpx.Client")
    def test_get_related(self, mock_httpx):
        from src.mcp_server import handle_tool_call
        rag = self._make_rag(mock_httpx)
        doc1 = rag.add_document("First document")
        doc2 = rag.add_document("Second document")
        rag.add_relation(doc1, doc2, "related_to", 1.0)
        result = handle_tool_call(rag, "rag_get_related", {"node_id": doc1, "max_depth": 1})
        assert "relations" in result
        assert len(result["relations"]) >= 1
        rel = result["relations"][0]
        assert rel["source"] == doc1
        assert rel["target"] == doc2
        assert rel["relation"] == "related_to"

    @patch("src.embeddings.httpx.Client")
    def test_graph_stats(self, mock_httpx):
        from src.mcp_server import handle_tool_call
        rag = self._make_rag(mock_httpx)
        doc1 = rag.add_document("First document")
        doc2 = rag.add_document("Second document")
        rag.add_relation(doc1, doc2, "related_to", 1.0)
        result = handle_tool_call(rag, "rag_graph_stats", {})
        assert "total_nodes" in result
        assert "total_edges" in result
        assert "relation_types" in result
        assert result["total_edges"] >= 1

    @patch("src.embeddings.httpx.Client")
    def test_delete_document(self, mock_httpx):
        """rag_delete_document удаляет документ из всех хранилищ."""
        from src.mcp_server import handle_tool_call
        rag = self._make_rag(mock_httpx)
        doc_id = rag.add_document("to be deleted")
        assert handle_tool_call(rag, "rag_stats", {})["total_documents"] == 1
        result = handle_tool_call(rag, "rag_delete_document", {"doc_id": doc_id})
        assert result["status"] == "ok"
        assert result["doc_id"] == doc_id
        assert result["deleted"] is True
        assert handle_tool_call(rag, "rag_stats", {})["total_documents"] == 0

    @patch("src.embeddings.httpx.Client")
    def test_delete_document_idempotent(self, mock_httpx):
        """Удаление несуществующего doc_id не вызывает ошибку."""
        from src.mcp_server import handle_tool_call
        rag = self._make_rag(mock_httpx)
        result = handle_tool_call(rag, "rag_delete_document", {"doc_id": "nonexistent-uuid"})
        assert result["status"] == "ok"
        assert result["deleted"] is True

    @patch("src.embeddings.httpx.Client")
    def test_delete_document_removes_from_graph(self, mock_httpx):
        """Удаление документа удаляет также рёбра графа."""
        from src.mcp_server import handle_tool_call
        rag = self._make_rag(mock_httpx)
        doc1 = rag.add_document("first")
        doc2 = rag.add_document("second")
        rag.add_relation(doc1, doc2, "related_to", 1.0)
        assert handle_tool_call(rag, "rag_graph_stats", {})["total_nodes"] == 2
        handle_tool_call(rag, "rag_delete_document", {"doc_id": doc1})
        stats = handle_tool_call(rag, "rag_graph_stats", {})
        assert stats["total_nodes"] == 1
        assert stats["total_edges"] == 0

    @patch("src.embeddings.httpx.Client")
    def test_list_documents_empty(self, mock_httpx):
        """rag_list_documents на пустом хранилище возвращает пустой список."""
        from src.mcp_server import handle_tool_call
        rag = self._make_rag(mock_httpx)
        result = handle_tool_call(rag, "rag_list_documents", {})
        assert result["documents"] == []
        assert result["total"] == 0
        assert result["limit"] == 20
        assert result["offset"] == 0

    @patch("src.embeddings.httpx.Client")
    def test_list_documents_with_data(self, mock_httpx):
        """rag_list_documents возвращает документы с пагинацией."""
        from src.mcp_server import handle_tool_call
        rag = self._make_rag(mock_httpx)
        for i in range(5):
            rag.add_document(f"document number {i}")
        result = handle_tool_call(rag, "rag_list_documents", {"limit": 3, "offset": 0})
        assert result["total"] == 5
        assert len(result["documents"]) == 3
        assert "doc_id" in result["documents"][0]
        assert "text" in result["documents"][0]
        assert "metadata" in result["documents"][0]

    @patch("src.embeddings.httpx.Client")
    def test_list_documents_pagination(self, mock_httpx):
        """Пагинация: offset сдвигает страницу."""
        from src.mcp_server import handle_tool_call
        rag = self._make_rag(mock_httpx)
        for i in range(10):
            rag.add_document(f"doc {i}")
        page1 = handle_tool_call(rag, "rag_list_documents", {"limit": 3, "offset": 0})
        page2 = handle_tool_call(rag, "rag_list_documents", {"limit": 3, "offset": 3})
        assert page1["total"] == 10
        assert page2["total"] == 10
        assert len(page1["documents"]) == 3
        assert len(page2["documents"]) == 3
        page1_ids = {d["doc_id"] for d in page1["documents"]}
        page2_ids = {d["doc_id"] for d in page2["documents"]}
        assert page1_ids.isdisjoint(page2_ids)

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
        assert len(mod.TOOL_DEFS) == 12

    def test_main_callable(self):
        """main() должен быть вызываемым (но не вызываем — он запускает stdio)."""
        import src.mcp_server as mod
        assert callable(mod.main)
