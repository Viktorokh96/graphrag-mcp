"""Тесты для QdrantVectorStore (Qdrant embedded, dense-поиск)."""

import sys

import pytest

DIM = 8

# Баг Windows + embedded Qdrant: recreate_collection()/clear() не удаляют данные.
# QdrantLocal.delete_collection не закрывает sqlite-хэндл коллекции (полагается
# на GC, но LocalCollection в reference-цикле), shutil.rmtree(ignore_errors=True)
# молча падает на залоченном файле, и create_collection поднимает старые точки.
# См. src/vector_store.py:137-140 (recreate_collection) и :181-182 (clear).
XFAIL_CLEAR_WINDOWS = pytest.mark.xfail(
    sys.platform == "win32",
    reason="clear()/recreate_collection() не очищают embedded Qdrant на Windows "
    "(file lock: QdrantLocal.delete_collection не закрывает sqlite коллекции)",
    strict=False,
)


def unit(i: int, dim: int = DIM) -> list[float]:
    """Единичный вектор по оси i."""
    v = [0.0] * dim
    v[i] = 1.0
    return v


@pytest.fixture
def make_store(tmp_path):
    """Фабрика сторов: каждый созданный стор гарантированно закрывается
    в teardown (Windows: embedded Qdrant держит файловый лок)."""
    from src.vector_store import QdrantVectorStore

    stores = []

    def _make(subdir: str = "qdrant", dimension: int = DIM) -> QdrantVectorStore:
        store = QdrantVectorStore(location=str(tmp_path / subdir), dimension=dimension)
        stores.append(store)
        return store

    yield _make
    for s in stores:
        try:
            s.close()
        except Exception:
            pass


class TestQdrantVectorStore:
    """Тесты для векторного хранилища на Qdrant."""

    def test_import(self):
        from src.vector_store import QdrantVectorStore

        assert QdrantVectorStore is not None

    def test_init_creates_collection(self, make_store):
        store = make_store()
        assert store is not None
        # Коллекция создана с нужной размерностью
        assert store.get_dimension() == DIM
        assert store.count() == 0

    def test_add_and_search(self, make_store):
        store = make_store()

        doc_id = store.add("doc1", "hello world", unit(0))
        assert doc_id == "doc1"

        results = store.search(unit(0), k=1)
        assert len(results) == 1
        assert results[0]["doc_id"] == "doc1"

    def test_search_by_similarity(self, make_store):
        store = make_store()

        store.add("cat", "cat animal", unit(0))
        store.add("dog", "dog animal", unit(1))
        store.add("car", "car vehicle", unit(2))

        # Запрос почти коллинеарен вектору кота
        query = [0.0] * DIM
        query[0] = 0.85
        query[1] = 0.15
        results = store.search(query, k=2)

        assert results[0]["doc_id"] == "cat"
        assert results[0]["score"] > results[1]["score"]

    def test_search_scores_in_unit_interval(self, make_store):
        store = make_store()
        store.add("a", "text a", unit(0))
        store.add("b", "text b", unit(1))

        results = store.search(unit(0), k=5)
        for r in results:
            assert 0.0 <= r["score"] <= 1.0

    def test_top_k(self, make_store):
        store = make_store()

        for i in range(10):
            emb = [0.0] * DIM
            emb[i % DIM] = 1.0
            emb[(i + 1) % DIM] = float(i) / 10
            store.add(f"doc{i}", f"text {i}", emb)

        results = store.search(unit(0), k=3)
        assert len(results) == 3

    def test_search_empty_store(self, make_store):
        store = make_store()
        assert store.search(unit(0), k=5) == []

    def test_clear(self, make_store):
        store = make_store()

        store.add("doc1", "some text", unit(0))
        assert store.count() == 1

        store.clear()
        assert store.count() == 0

    def test_stats(self, make_store, tmp_path):
        store = make_store()

        stats = store.stats()
        assert stats["total_documents"] == 0
        assert stats["store_path"] == str(tmp_path / "qdrant")
        assert stats["dimension"] == DIM

        store.add("doc1", "text1", unit(0))
        store.add("doc2", "text2", unit(1))
        stats = store.stats()
        assert stats["total_documents"] == 2

    def test_remove(self, make_store):
        store = make_store()

        store.add("doc1", "text1", unit(0))
        store.add("doc2", "text2", unit(1))

        store.remove("doc1")
        assert store.count() == 1
        assert not store.has("doc1")
        assert store.has("doc2")

    def test_count(self, make_store):
        store = make_store()

        assert store.count() == 0
        store.add("doc1", "text", unit(0))
        assert store.count() == 1

    def test_add_same_id_upserts(self, make_store):
        store = make_store()

        store.add("doc1", "old text", unit(0))
        store.add("doc1", "new text", unit(1))
        assert store.count() == 1

        results = store.search(unit(1), k=1)
        assert results[0]["doc_id"] == "doc1"

    def test_get_all_ids(self, make_store):
        store = make_store()

        store.add("doc1", "hello world", unit(0))
        store.add("doc2", "foo bar", unit(1))

        ids = store.get_all_ids()
        assert ids == {"doc1", "doc2"}

    def test_get_all_ids_empty(self, make_store):
        store = make_store()
        assert store.get_all_ids() == set()

    def test_has(self, make_store):
        store = make_store()

        assert not store.has("doc1")
        store.add("doc1", "text", unit(0))
        assert store.has("doc1")
        assert not store.has("nonexistent")

    def test_search_result_contains_no_text(self, make_store):
        """Тексты не хранятся в Qdrant — результат поиска без текста."""
        store = make_store()
        store.add("doc1", "hello world", unit(0))

        results = store.search(unit(0), k=1)
        assert set(results[0].keys()) == {"doc_id", "score", "metadata"}

    def test_persistence_across_reload(self, make_store):
        """Данные сохраняются между перезагрузками (embedded хранит на диске)."""
        store1 = make_store("persist")
        store1.add("persist_doc", "persistent text", unit(3))
        assert store1.count() == 1
        # Один embedded path нельзя открыть двумя клиентами — сначала close()
        store1.close()

        store2 = make_store("persist")
        assert store2.count() == 1

        results = store2.search(unit(3), k=1)
        assert len(results) == 1
        assert results[0]["doc_id"] == "persist_doc"

    def test_metadata_support(self, make_store):
        store = make_store()

        store.add("doc1", "text with meta", unit(0), metadata={"category": "test", "page": 5})
        results = store.search(unit(0), k=1)

        assert len(results) == 1
        meta = results[0]["metadata"]
        assert meta["category"] == "test"
        assert meta["page"] == 5

    def test_metadata_default_empty(self, make_store):
        store = make_store()
        store.add("doc1", "no meta", unit(0))

        results = store.search(unit(0), k=1)
        assert results[0]["metadata"] == {}

    def test_metadata_filter_scalar(self, make_store):
        store = make_store()

        store.add("a1", "text a1", unit(0), metadata={"category": "a"})
        store.add("b1", "text b1", unit(0), metadata={"category": "b"})

        results = store.search(unit(0), k=5, metadata_filter={"category": "a"})
        assert [r["doc_id"] for r in results] == ["a1"]

    def test_metadata_filter_list_is_in(self, make_store):
        """Список в фильтре — семантика $in."""
        store = make_store()

        store.add("a1", "text", unit(0), metadata={"category": "a"})
        store.add("b1", "text", unit(0), metadata={"category": "b"})
        store.add("c1", "text", unit(0), metadata={"category": "c"})

        results = store.search(unit(0), k=5, metadata_filter={"category": ["a", "c"]})
        assert {r["doc_id"] for r in results} == {"a1", "c1"}

    def test_metadata_filter_and_semantics(self, make_store):
        """Несколько ключей объединяются по AND."""
        store = make_store()

        store.add("d1", "text", unit(0), metadata={"category": "a", "page": 1})
        store.add("d2", "text", unit(0), metadata={"category": "a", "page": 2})
        store.add("d3", "text", unit(0), metadata={"category": "b", "page": 1})

        results = store.search(unit(0), k=5, metadata_filter={"category": "a", "page": 1})
        assert [r["doc_id"] for r in results] == ["d1"]

    def test_metadata_filter_no_match(self, make_store):
        store = make_store()
        store.add("doc1", "text", unit(0), metadata={"category": "a"})

        results = store.search(unit(0), k=5, metadata_filter={"category": "zzz"})
        assert results == []

    def test_update_embedding(self, make_store):
        store = make_store()

        store.add("doc1", "text", unit(0))
        store.add("doc2", "text2", unit(1))

        store.update_embedding("doc1", unit(5))
        results = store.search(unit(5), k=1)
        assert len(results) == 1
        assert results[0]["doc_id"] == "doc1"

    def test_recreate_collection(self, make_store):
        store = make_store()

        store.add("doc1", "text", unit(0))
        assert store.count() == 1

        store.recreate_collection()
        assert store.count() == 0
        # Размерность коллекции восстановлена из конфига стора
        assert store.get_dimension() == DIM

        store.add("doc2", "text2", unit(1))
        assert store.count() == 1
        assert store.stats()["dimension"] == DIM

    def test_get_dimension(self, make_store):
        store = make_store("dim4", dimension=4)
        assert store.get_dimension() == 4
        assert store.stats()["dimension"] == 4

    def test_add_wrong_dimension_raises(self, make_store):
        store = make_store()

        with pytest.raises(Exception):
            store.add("doc1", "text", [0.1, 0.2])  # dim=2 вместо 8
