"""Тесты для src/migrate.py — миграция ChromaDB + graph_index.json → новые стора."""

import json
import sys
import types
from unittest.mock import MagicMock, patch

import pytest

from src import migrate


class FakeDocStore:
    def __init__(self, ids):
        self._ids = set(ids)

    def all_ids(self):
        return self._ids


class FakeRAG:
    """Стаб RAGSystem: собирает добавленные документы и рёбра."""

    def __init__(self, store_path, valid_ids=None, reject=()):
        self.store_path = str(store_path)
        self.added: list[tuple] = []
        self.relations: list[tuple] = []
        self._reject = set(reject)
        self.doc_store = FakeDocStore(valid_ids if valid_ids is not None else [])

    def add_document(self, text, metadata=None, doc_id=None):
        if doc_id in self._reject:
            raise ValueError("document too short")
        self.added.append((doc_id, text, metadata))
        return doc_id

    def add_relation(self, source, target, relation, weight):
        self.relations.append((source, target, relation, weight))


def _write_graph(store_path, nodes=None, edges=None):
    (store_path / migrate._OLD_GRAPH).write_text(
        json.dumps({"nodes": nodes or {}, "edges": edges or {}}), encoding="utf-8"
    )


class TestDetectOldData:
    def test_empty_dir(self, tmp_path):
        assert migrate.detect_old_data(str(tmp_path)) == {
            "chroma": False, "bm25": False, "graph": False
        }
        assert migrate.has_old_data(str(tmp_path)) is False

    def test_missing_dir(self, tmp_path):
        assert migrate.has_old_data(str(tmp_path / "nope")) is False

    @pytest.mark.parametrize(
        "filename,key",
        [(migrate._OLD_CHROMA, "chroma"), (migrate._OLD_BM25, "bm25"), (migrate._OLD_GRAPH, "graph")],
    )
    def test_single_artifact_detected(self, tmp_path, filename, key):
        (tmp_path / filename).write_text("x")
        found = migrate.detect_old_data(str(tmp_path))
        assert found[key] is True
        assert sum(found.values()) == 1
        assert migrate.has_old_data(str(tmp_path)) is True


class TestReadOldGraph:
    def test_missing_file_returns_empty(self, tmp_path):
        assert migrate._read_old_graph(str(tmp_path)) == ({}, {})

    def test_reads_nodes_and_edges(self, tmp_path):
        _write_graph(tmp_path, nodes={"a": {}}, edges={"e1": {"source": "a", "target": "b"}})
        nodes, edges = migrate._read_old_graph(str(tmp_path))
        assert list(nodes) == ["a"]
        assert edges["e1"]["target"] == "b"

    def test_missing_keys_default_to_empty(self, tmp_path):
        (tmp_path / migrate._OLD_GRAPH).write_text("{}", encoding="utf-8")
        assert migrate._read_old_graph(str(tmp_path)) == ({}, {})


class TestReadChromaDocuments:
    def test_raises_when_chromadb_missing(self, tmp_path):
        with patch.dict(sys.modules, {"chromadb": None}):
            with pytest.raises(RuntimeError, match="chromadb"):
                migrate._read_chroma_documents(str(tmp_path))

    def _fake_chromadb(self, collection):
        client = MagicMock()
        if isinstance(collection, Exception):
            client.get_collection.side_effect = collection
        else:
            client.get_collection.return_value = collection
        module = types.ModuleType("chromadb")
        module.PersistentClient = MagicMock(return_value=client)
        return module

    def test_missing_collection_returns_empty(self, tmp_path):
        module = self._fake_chromadb(Exception("no collection"))
        with patch.dict(sys.modules, {"chromadb": module}):
            assert migrate._read_chroma_documents(str(tmp_path)) == []

    def test_maps_ids_texts_and_metadata(self, tmp_path):
        collection = MagicMock()
        collection.get.return_value = {
            "ids": ["d1", "d2", "d3"],
            "documents": ["one", "two"],
            "metadatas": [{"k": 1}, "not-a-dict"],
        }
        module = self._fake_chromadb(collection)
        with patch.dict(sys.modules, {"chromadb": module}):
            docs = migrate._read_chroma_documents(str(tmp_path))

        assert docs == [
            {"doc_id": "d1", "text": "one", "metadata": {"k": 1}},
            {"doc_id": "d2", "text": "two", "metadata": {}},
            {"doc_id": "d3", "text": "", "metadata": {}},
        ]

    def test_empty_result_keys(self, tmp_path):
        collection = MagicMock()
        collection.get.return_value = {}
        module = self._fake_chromadb(collection)
        with patch.dict(sys.modules, {"chromadb": module}):
            assert migrate._read_chroma_documents(str(tmp_path)) == []


class TestBackupOldFiles:
    def test_renames_existing_files_only(self, tmp_path):
        (tmp_path / migrate._OLD_CHROMA).write_text("db")
        (tmp_path / migrate._OLD_GRAPH).write_text("{}")
        migrate._backup_old_files(str(tmp_path))

        assert (tmp_path / f"{migrate._OLD_CHROMA}.bak").exists()
        assert (tmp_path / f"{migrate._OLD_GRAPH}.bak").exists()
        assert not (tmp_path / migrate._OLD_CHROMA).exists()
        assert not (tmp_path / f"{migrate._OLD_BM25}.bak").exists()


class TestRunMigration:
    def test_no_old_data_returns_none(self, tmp_path):
        assert migrate.run_migration(FakeRAG(tmp_path)) is None

    def test_dry_run_reports_counts_without_writing(self, tmp_path, capsys):
        _write_graph(tmp_path, edges={"e1": {"source": "a", "target": "b"}})
        rag = FakeRAG(tmp_path)
        stats = migrate.run_migration(rag, dry_run=True)

        assert stats == {"documents": 0, "edges": 1, "skipped_documents": 0, "skipped_edges": 0}
        assert rag.added == [] and rag.relations == []
        assert (tmp_path / migrate._OLD_GRAPH).exists()

    def test_dry_run_lists_first_ten_documents(self, tmp_path, capsys):
        (tmp_path / migrate._OLD_CHROMA).write_text("db")
        documents = [{"doc_id": f"doc{i:02d}", "text": f"text {i}", "metadata": {}} for i in range(12)]
        with patch.object(migrate, "_read_chroma_documents", return_value=documents):
            stats = migrate.run_migration(FakeRAG(tmp_path), dry_run=True)

        out = capsys.readouterr().out
        assert stats["documents"] == 12
        assert "и ещё 2" in out
        assert "doc10" not in out

    def test_declined_confirmation_aborts(self, tmp_path):
        _write_graph(tmp_path, edges={"e1": {"source": "a", "target": "b"}})
        rag = FakeRAG(tmp_path, valid_ids=["a", "b"])
        with patch("builtins.input", return_value="n"):
            assert migrate.run_migration(rag) is None
        assert rag.relations == []
        assert (tmp_path / migrate._OLD_GRAPH).exists()

    def test_eof_on_confirmation_aborts(self, tmp_path):
        _write_graph(tmp_path, edges={})
        with patch("builtins.input", side_effect=EOFError):
            assert migrate.run_migration(FakeRAG(tmp_path)) is None

    def test_accepted_confirmation_migrates(self, tmp_path):
        _write_graph(tmp_path, edges={"e1": {"source": "a", "target": "b", "relation": "links", "weight": 2.0}})
        rag = FakeRAG(tmp_path, valid_ids=["a", "b"])
        with patch("builtins.input", return_value="Yes"):
            stats = migrate.run_migration(rag)

        assert stats["edges"] == 1
        assert rag.relations == [("a", "b", "links", 2.0)]

    def test_force_migrates_documents_and_edges(self, tmp_path):
        (tmp_path / migrate._OLD_CHROMA).write_text("db")
        _write_graph(tmp_path, edges={
            "e1": {"source": "d1", "target": "d2"},
            "e2": {"source": "d1", "target": "gone"},
        })
        documents = [
            {"doc_id": "d1", "text": "first", "metadata": {"k": 1}},
            {"doc_id": "d2", "text": "second", "metadata": {}},
        ]
        rag = FakeRAG(tmp_path, valid_ids=["d1", "d2"])
        with patch.object(migrate, "_read_chroma_documents", return_value=documents):
            stats = migrate.run_migration(rag, force=True)

        assert stats == {"documents": 2, "edges": 1, "skipped_documents": 0, "skipped_edges": 1}
        assert rag.added == [("d1", "first", {"k": 1}), ("d2", "second", None)]
        assert rag.relations == [("d1", "d2", "related_to", 1.0)]
        assert (tmp_path / f"{migrate._OLD_CHROMA}.bak").exists()
        assert (tmp_path / f"{migrate._OLD_GRAPH}.bak").exists()

    def test_rejected_documents_are_skipped(self, tmp_path, capsys):
        (tmp_path / migrate._OLD_CHROMA).write_text("db")
        documents = [
            {"doc_id": "short-doc", "text": "x", "metadata": {}},
            {"doc_id": "ok-doc", "text": "long enough", "metadata": {}},
        ]
        rag = FakeRAG(tmp_path, valid_ids=["ok-doc"], reject=["short-doc"])
        with patch.object(migrate, "_read_chroma_documents", return_value=documents):
            stats = migrate.run_migration(rag, force=True)

        assert stats["documents"] == 1
        assert stats["skipped_documents"] == 1
        assert [doc_id for doc_id, _, _ in rag.added] == ["ok-doc"]
        assert "skip short-do" in capsys.readouterr().err

    def test_graph_only_store_skips_chroma_read(self, tmp_path):
        _write_graph(tmp_path, edges={})
        with patch.object(migrate, "_read_chroma_documents") as reader:
            stats = migrate.run_migration(FakeRAG(tmp_path), force=True)
        reader.assert_not_called()
        assert stats == {"documents": 0, "edges": 0, "skipped_documents": 0, "skipped_edges": 0}
