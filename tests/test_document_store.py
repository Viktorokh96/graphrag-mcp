"""Тесты DocumentStore — SQLite-хранилища документов (таблица documents)."""

import pytest

from src.document_store import Database, DocumentStore, is_postgres_url
from src.graph_store import GraphStore


@pytest.fixture
def db(tmp_path):
    """SQLite Database во временной директории; close() в teardown (Windows)."""
    database = Database(str(tmp_path / "store.db"))
    yield database
    database.close()


@pytest.fixture
def docs(db):
    return DocumentStore(db)


class TestAddGet:
    """add / get / upsert."""

    def test_add_and_get_roundtrip(self, docs):
        docs.add(
            "doc1",
            "Python is a programming language",
            metadata={"lang": "python", "level": 1},
            content_hash="hash1",
            parent_doc_id="parent1",
            chunk_index=3,
        )

        doc = docs.get("doc1")
        assert doc == {
            "doc_id": "doc1",
            "text": "Python is a programming language",
            "metadata": {"lang": "python", "level": 1},
            "content_hash": "hash1",
            "parent_doc_id": "parent1",
            "chunk_index": 3,
        }

    def test_add_defaults(self, docs):
        """Необязательные поля по умолчанию: metadata={}, остальные None."""
        docs.add("doc1", "Plain text")
        doc = docs.get("doc1")
        assert doc["metadata"] == {}
        assert doc["content_hash"] is None
        assert doc["parent_doc_id"] is None
        assert doc["chunk_index"] is None

    def test_get_missing_returns_none(self, docs):
        assert docs.get("nonexistent") is None

    def test_add_returns_doc_id(self, docs):
        assert docs.add("doc1", "Text") == "doc1"

    def test_add_upserts_existing(self, docs):
        """Повторный add с тем же doc_id обновляет содержимое."""
        docs.add("doc1", "Original text", {"key": "value1"}, content_hash="h1")
        docs.add("doc1", "Updated text", {"key": "value2"}, content_hash="h2")

        doc = docs.get("doc1")
        assert doc["text"] == "Updated text"
        assert doc["metadata"]["key"] == "value2"
        assert doc["content_hash"] == "h2"
        assert docs.count() == 1

    def test_unicode_text_and_metadata(self, docs):
        """Кириллица в тексте и метаданных сохраняется без искажений."""
        docs.add("doc1", "Питон — язык программирования", {"тема": "языки"})
        doc = docs.get("doc1")
        assert doc["text"] == "Питон — язык программирования"
        assert doc["metadata"]["тема"] == "языки"


class TestBatchAndChunks:
    """get_batch / get_chunks / get_metadata_batch."""

    def test_get_batch(self, docs):
        docs.add("doc1", "Text 1")
        docs.add("doc2", "Text 2")
        docs.add("doc3", "Text 3")

        batch = docs.get_batch(["doc1", "doc3", "missing"])
        assert set(batch) == {"doc1", "doc3"}
        assert batch["doc1"]["text"] == "Text 1"
        assert batch["doc3"]["text"] == "Text 3"

    def test_get_batch_empty(self, docs):
        assert docs.get_batch([]) == {}

    def test_get_chunks_ordered(self, docs):
        """Чанки родителя возвращаются в порядке chunk_index."""
        docs.add("chunk_b", "Chunk B", parent_doc_id="parent", chunk_index=1)
        docs.add("chunk_c", "Chunk C", parent_doc_id="parent", chunk_index=2)
        docs.add("chunk_a", "Chunk A", parent_doc_id="parent", chunk_index=0)
        docs.add("other", "Other doc", parent_doc_id="another_parent", chunk_index=0)

        chunks = docs.get_chunks("parent")
        assert [c["doc_id"] for c in chunks] == ["chunk_a", "chunk_b", "chunk_c"]
        assert [c["chunk_index"] for c in chunks] == [0, 1, 2]

    def test_get_chunks_empty(self, docs):
        assert docs.get_chunks("no_such_parent") == []

    def test_get_metadata_batch(self, docs):
        docs.add("doc1", "Text 1", {"type": "a"})
        docs.add("doc2", "Text 2")

        meta = docs.get_metadata_batch(["doc1", "doc2", "missing"])
        assert meta == {"doc1": {"type": "a"}, "doc2": {}}
        assert docs.get_metadata_batch([]) == {}


class TestDelete:
    """delete / clear / каскад рёбер."""

    def test_delete_existing_returns_true(self, docs):
        docs.add("doc1", "Text")
        assert docs.delete("doc1") is True
        assert docs.get("doc1") is None
        assert docs.count() == 0

    def test_delete_missing_returns_false(self, docs):
        assert docs.delete("nonexistent") is False

    def test_delete_cascades_graph_edges(self, docs, db):
        """Удаление документа удаляет его рёбра в graph_edges (FK CASCADE)."""
        graph = GraphStore(db, docs)
        docs.add("doc1", "Text 1")
        docs.add("doc2", "Text 2")
        docs.add("doc3", "Text 3")
        graph.add_edge("doc1", "doc2", "related_to", 1.0)
        graph.add_edge("doc3", "doc1", "similar_to", 0.8)
        graph.add_edge("doc2", "doc3", "related_to", 0.5)

        assert docs.delete("doc1") is True

        # В SQL остаётся только ребро, не касающееся doc1
        rows = db.execute("SELECT source_id, target_id, relation FROM graph_edges")
        assert rows == [("doc2", "doc3", "related_to")]

    def test_clear(self, docs):
        docs.add("doc1", "Text 1")
        docs.add("doc2", "Text 2")
        docs.clear()
        assert docs.count() == 0
        assert docs.all_ids() == set()


class TestHashes:
    """find_by_hash / all_hashes."""

    def test_find_by_hash(self, docs):
        docs.add("doc1", "Text 1", content_hash="hash1")
        docs.add("doc2", "Text 2", content_hash="hash2")

        assert docs.find_by_hash("hash1") == "doc1"
        assert docs.find_by_hash("hash2") == "doc2"
        assert docs.find_by_hash("unknown") is None

    def test_all_hashes(self, docs):
        docs.add("doc1", "Text 1", content_hash="hash1")
        docs.add("doc2", "Text 2", content_hash="hash2")
        docs.add("doc3", "Text 3")  # без хеша — не попадает в словарь

        assert docs.all_hashes() == {"hash1": "doc1", "hash2": "doc2"}

    def test_all_hashes_empty(self, docs):
        assert docs.all_hashes() == {}


class TestList:
    """list: пагинация, total, metadata_filter."""

    def test_list_empty(self, docs):
        items, total = docs.list()
        assert items == []
        assert total == 0

    def test_list_pagination(self, docs):
        for i in range(5):
            docs.add(f"doc{i}", f"Text {i}")

        items, total = docs.list(limit=2, offset=0)
        assert total == 5
        assert len(items) == 2

        items2, total2 = docs.list(limit=2, offset=4)
        assert total2 == 5
        assert len(items2) == 1

        # Страницы не пересекаются и покрывают все документы
        all_ids = set()
        for offset in (0, 2, 4):
            page, _ = docs.list(limit=2, offset=offset)
            all_ids.update(d["doc_id"] for d in page)
        assert all_ids == {f"doc{i}" for i in range(5)}

    def test_list_metadata_filter_scalar(self, docs):
        docs.add("doc1", "Text 1", {"type": "person"})
        docs.add("doc2", "Text 2", {"type": "place"})
        docs.add("doc3", "Text 3", {"type": "person"})

        items, total = docs.list(metadata_filter={"type": "person"})
        assert total == 2
        assert {d["doc_id"] for d in items} == {"doc1", "doc3"}

    def test_list_metadata_filter_in_list(self, docs):
        """Значение-список — семантика $in."""
        docs.add("doc1", "Text 1", {"type": "person"})
        docs.add("doc2", "Text 2", {"type": "place"})
        docs.add("doc3", "Text 3", {"type": "event"})

        items, total = docs.list(metadata_filter={"type": ["person", "place"]})
        assert total == 2
        assert {d["doc_id"] for d in items} == {"doc1", "doc2"}

    def test_list_metadata_filter_and(self, docs):
        """Несколько ключей объединяются через AND."""
        docs.add("doc1", "Text 1", {"type": "person", "lang": "ru"})
        docs.add("doc2", "Text 2", {"type": "person", "lang": "en"})
        docs.add("doc3", "Text 3", {"type": "place", "lang": "ru"})

        items, total = docs.list(metadata_filter={"type": "person", "lang": "ru"})
        assert total == 1
        assert items[0]["doc_id"] == "doc1"

    def test_list_metadata_filter_no_match(self, docs):
        docs.add("doc1", "Text 1", {"type": "person"})
        items, total = docs.list(metadata_filter={"type": "robot"})
        assert items == []
        assert total == 0

    def test_list_metadata_filter_pagination(self, docs):
        """offset/limit применяются после фильтрации, total — по фильтру."""
        for i in range(4):
            docs.add(f"match{i}", f"Text {i}", {"kind": "match"})
        docs.add("other", "Other", {"kind": "other"})

        items, total = docs.list(limit=2, offset=2, metadata_filter={"kind": "match"})
        assert total == 4
        assert len(items) == 2
        assert all(d["metadata"]["kind"] == "match" for d in items)


class TestCounters:
    """count / all_ids."""

    def test_count_and_all_ids(self, docs):
        assert docs.count() == 0
        assert docs.all_ids() == set()

        docs.add("doc1", "Text 1")
        docs.add("doc2", "Text 2")
        assert docs.count() == 2
        assert docs.all_ids() == {"doc1", "doc2"}


class TestPersistence:
    """Документы переживают переоткрытие Database."""

    def test_documents_persist_across_reopen(self, tmp_path):
        db_path = str(tmp_path / "store.db")
        db = Database(db_path)
        docs = DocumentStore(db)
        docs.add("doc1", "Persistent text", {"key": "value"}, content_hash="h1")
        db.close()

        db2 = Database(db_path)
        try:
            docs2 = DocumentStore(db2)
            doc = docs2.get("doc1")
            assert doc is not None
            assert doc["text"] == "Persistent text"
            assert doc["metadata"] == {"key": "value"}
            assert doc["content_hash"] == "h1"
        finally:
            db2.close()


class TestConnectionString:
    """Определение backend'а по строке подключения."""

    def test_is_postgres_url(self):
        assert is_postgres_url("postgresql://user@host/db") is True
        assert is_postgres_url("postgres://user@host/db") is True
        assert is_postgres_url("postgresql+psycopg://user@host/db") is True
        assert is_postgres_url("./rag_data/store.db") is False
        assert is_postgres_url("C:/data/store.db") is False
