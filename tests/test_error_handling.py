"""Тесты распространения ошибок: ничего не должно теряться молча."""

import json

import pytest
from fastapi.testclient import TestClient

from src import http_api
from src.config import RAGConfig
from src.mcp_server import handle_tool_call
from src.rag import RAGSystem
from src.structured_indexer import StructuredIndexer
from tests.conftest import HashEmbeddingGenerator

LONG = "This is a sufficiently long document about Python programming and testing. " * 2


@pytest.fixture
def client(tmp_path, monkeypatch):
    cfg = RAGConfig()
    cfg.store_path = str(tmp_path / "store")
    rag = RAGSystem(config=cfg, embedding_generator=HashEmbeddingGenerator())

    def fake_init(self, config=None):
        self.instance = rag
        return rag

    monkeypatch.setattr(http_api.RAGHolder, "init", fake_init)
    with TestClient(http_api.app) as c:
        yield c
    rag.close()


class TestHttpApiErrors:
    def test_relation_to_unknown_node_is_400(self, client):
        r = client.post("/relations", json={"source_id": "nope", "target_id": "also-nope", "relation": "rel"})
        assert r.status_code == 400
        assert "does not exist" in r.json()["detail"]

    def test_structured_invalid_json_is_400(self, client):
        r = client.post("/structured", json={"content": "{not json"})
        assert r.status_code == 400
        assert "valid JSON" in r.json()["detail"]

    def test_community_name_with_non_numeric_id_is_400(self, client):
        r = client.put("/communities/names", json={"names": {"main": "Auth"}})
        assert r.status_code == 400
        assert "integer" in r.json()["detail"]

    def test_short_document_is_400(self, client):
        r = client.post("/documents", json={"text": "short"})
        assert r.status_code == 400
        assert "too short" in r.json()["detail"].lower()


class TestMcpToolErrors:
    def test_missing_required_argument_names_it(self, rag):
        with pytest.raises(ValueError, match="requires argument 'doc_id'"):
            handle_tool_call(rag, "rag_get_document", {})

    def test_unknown_tool_lists_available(self, rag):
        with pytest.raises(ValueError, match="rag_search"):
            handle_tool_call(rag, "rag_nonexistent", {})

    def test_missing_file_reports_path(self, rag, tmp_path):
        missing = str(tmp_path / "absent.md")
        with pytest.raises(ValueError, match="Cannot read file"):
            handle_tool_call(rag, "rag_add_file", {"filepath": missing})


class TestStructuredIndexerErrors:
    def test_invalid_json_raises_value_error(self, rag):
        with pytest.raises(ValueError, match="valid JSON"):
            StructuredIndexer(rag).index("{")

    def test_per_file_failures_are_reported(self, rag):
        content = json.dumps({
            "repository": "demo",
            "structure": ["a.py", "b.py"],
            "files": {"a.py": {"content": LONG}, "b.py": "not-an-object"},
        })
        result = StructuredIndexer(rag).index(content)
        assert result["files_count"] == 1
        assert result["errors"] == 1
        assert result["error_details"][0]["path"] == "b.py"


class TestGraphExtractionErrors:
    def test_unknown_mode_raises(self, rag):
        with pytest.raises(RuntimeError, match="graph extraction"):
            rag.add_document(LONG, extract_graph=True, extract_graph_mode="magic")

    def test_document_survives_failed_extraction(self, rag):
        with pytest.raises(RuntimeError):
            rag.add_document(LONG, extract_graph=True, extract_graph_mode="magic")
        assert rag.is_duplicate(LONG) is not None


class TestCliExitCodes:
    def test_help_exits_zero(self, capsys):
        from src.cli import main

        assert main(["--help"]) == 0

    def test_bad_arguments_exit_nonzero(self, capsys):
        from src.cli import main

        assert main(["definitely-not-a-command"]) == 1


class TestSyncStoresErrors:
    def test_unreachable_vector_store_propagates(self, rag):
        def boom():
            raise ConnectionError("qdrant is down")

        rag.vector_store.get_all_ids = boom
        with pytest.raises(ConnectionError):
            rag._sync_stores()

    def test_single_document_reindex_failure_is_not_fatal(self, rag, caplog):
        doc_id = rag.add_document(LONG)
        rag.vector_store.remove(doc_id)

        def boom(*_args, **_kwargs):
            raise RuntimeError("embedding provider unavailable")

        rag._index_vector = boom
        with caplog.at_level("ERROR"):
            rag._sync_stores()
        assert "embedding provider unavailable" in caplog.text
