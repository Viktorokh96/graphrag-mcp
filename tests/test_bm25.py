"""Тесты BM25-поиска: sparse-векторы в Qdrant (QdrantVectorStore.bm25_search)."""

import pytest

DIM = 8


def unit(i: int, dim: int = DIM) -> list[float]:
    v = [0.0] * dim
    v[i % dim] = 1.0
    return v


@pytest.fixture
def make_store(tmp_path):
    """Фабрика сторов с гарантированным close() в teardown (Windows file lock)."""
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


class TestBM25Search:
    """Тесты BM25-поиска через sparse-векторы Qdrant."""

    def test_import(self):
        from src.vector_store import QdrantVectorStore

        assert hasattr(QdrantVectorStore, "bm25_search")

    def test_add_and_search(self, make_store):
        """После добавления документа поиск по его токену возвращает его."""
        store = make_store()
        store.add("doc1", "python programming language", unit(0))

        results = store.bm25_search("python", k=1)
        assert len(results) == 1
        assert results[0]["doc_id"] == "doc1"
        assert results[0]["score"] > 0

    def test_exact_token_match_only(self, make_store):
        """Совпадение по точным токенам: отсутствующий токен не находит ничего."""
        store = make_store()
        store.add("doc1", "python programming", unit(0))
        store.add("doc2", "java programming", unit(1))

        results = store.bm25_search("golang", k=5)
        assert results == []

        # А по существующему токену — находит только нужный документ
        results = store.bm25_search("java", k=5)
        assert [r["doc_id"] for r in results] == ["doc2"]

    def test_search_returns_sorted(self, make_store):
        """Результаты отсортированы по убыванию score."""
        store = make_store()
        store.add("doc1", "python programming", unit(0))
        store.add("doc2", "java programming", unit(1))
        store.add("doc3", "python python python", unit(2))

        results = store.bm25_search("python", k=3)
        scores = [r["score"] for r in results]
        for i in range(len(scores) - 1):
            assert scores[i] >= scores[i + 1], "Оценки должны убывать"

    def test_ranking_tf_and_length(self, make_store):
        """Документ с бОльшим числом вхождений запроса и меньшей длиной — выше."""
        store = make_store()
        store.add("dense_doc", "python python python", unit(0))
        store.add("sparse_doc", "python appears once in this considerably longer document text", unit(1))

        results = store.bm25_search("python", k=2)
        assert len(results) == 2
        assert results[0]["doc_id"] == "dense_doc"
        assert results[0]["score"] > results[1]["score"]

    def test_ranking_idf(self, make_store):
        """IDF: слово, встречающееся во всех документах, даёт меньший вклад,
        чем редкое (при равной длине документов)."""
        store = make_store()
        # 'shared' — во всех документах, 'rareword' — только в одном
        store.add("common1", "shared", unit(0))
        store.add("common2", "shared", unit(1))
        store.add("common3", "shared", unit(2))
        store.add("rare_doc", "rareword", unit(3))

        results = store.bm25_search("shared rareword", k=4)
        assert len(results) == 4
        # Документ с редким словом ранжируется выше документов с частым
        assert results[0]["doc_id"] == "rare_doc"
        common_scores = [r["score"] for r in results if r["doc_id"] != "rare_doc"]
        assert all(results[0]["score"] > s for s in common_scores)

    def test_top_k_limit(self, make_store):
        """Параметр k ограничивает количество результатов."""
        store = make_store()
        for i in range(10):
            store.add(f"doc{i}", f"text number token{i} python", unit(i))

        results = store.bm25_search("python", k=3)
        assert len(results) == 3

    def test_search_empty_store(self, make_store):
        """Поиск в пустой коллекции возвращает пустой список."""
        store = make_store()
        assert store.bm25_search("anything", k=5) == []

    def test_empty_query_returns_empty(self, make_store):
        """Запрос без токенов (пунктуация/пробелы) — пустой результат."""
        store = make_store()
        store.add("doc1", "python programming", unit(0))

        assert store.bm25_search("", k=5) == []
        assert store.bm25_search("!!! --- ...", k=5) == []

    def test_clear(self, make_store):
        """Очистка стора удаляет BM25-индекс."""
        store = make_store()
        store.add("doc1", "some text", unit(0))
        store.clear()

        assert store.bm25_search("text", k=5) == []

    def test_remove_document(self, make_store):
        """Удалённый документ не участвует в BM25-поиске."""
        store = make_store()
        store.add("doc1", "python programming", unit(0))
        store.add("doc2", "java programming", unit(1))

        store.remove("doc1")
        results = store.bm25_search("python programming", k=5)
        doc_ids = [r["doc_id"] for r in results]
        assert "doc1" not in doc_ids

        results = store.bm25_search("java", k=5)
        assert [r["doc_id"] for r in results] == ["doc2"]

    def test_metadata_preserved(self, make_store):
        """Метаданные возвращаются в результатах BM25-поиска."""
        store = make_store()
        store.add("doc1", "python text", unit(0), metadata={"source": "test", "page": 1})

        results = store.bm25_search("python", k=1)
        assert len(results) == 1
        meta = results[0]["metadata"]
        assert meta["source"] == "test"
        assert meta["page"] == 1

    def test_metadata_filter(self, make_store):
        """metadata_filter применяется нативно и в BM25-поиске."""
        store = make_store()
        store.add("doc1", "python guide", unit(0), metadata={"lang": "python"})
        store.add("doc2", "python and java compared", unit(1), metadata={"lang": "java"})

        results = store.bm25_search("python", k=5, metadata_filter={"lang": "java"})
        assert [r["doc_id"] for r in results] == ["doc2"]

        results = store.bm25_search("python", k=5, metadata_filter={"lang": ["python", "java"]})
        assert {r["doc_id"] for r in results} == {"doc1", "doc2"}

    def test_case_insensitive(self, make_store):
        """Поиск нечувствителен к регистру."""
        store = make_store()
        store.add("doc1", "Python Programming Language", unit(0))

        results_lower = store.bm25_search("python", k=5)
        results_upper = store.bm25_search("PYTHON", k=5)

        assert len(results_lower) == 1
        assert len(results_upper) == 1
        assert results_lower[0]["doc_id"] == "doc1"
        assert results_upper[0]["doc_id"] == "doc1"

    def test_cyrillic_tokens(self, make_store):
        """Токенизатор поддерживает кириллицу."""
        store = make_store()
        store.add("ru_doc", "Python — мощный язык программирования!", unit(0))
        store.add("en_doc", "python is a powerful language", unit(1))

        results = store.bm25_search("язык", k=5)
        assert [r["doc_id"] for r in results] == ["ru_doc"]

        results = store.bm25_search("ЯЗЫК ПРОГРАММИРОВАНИЯ", k=5)
        assert [r["doc_id"] for r in results] == ["ru_doc"]

    def test_digits_and_underscore_tokens(self, make_store):
        """Токенизатор поддерживает цифры и underscore."""
        store = make_store()
        store.add("doc1", "variable my_var_1 equals 42", unit(0))
        store.add("doc2", "another document entirely", unit(1))

        results = store.bm25_search("my_var_1", k=5)
        assert [r["doc_id"] for r in results] == ["doc1"]

        results = store.bm25_search("42", k=5)
        assert [r["doc_id"] for r in results] == ["doc1"]


class TestBM25ModuleFunctions:
    """Тесты модульных функций BM25 (без Qdrant-клиента)."""

    def test_hash_token_stable_and_positive(self):
        from src.vector_store import hash_token

        assert hash_token("python") == hash_token("python")
        assert hash_token("python") >= 0
        assert hash_token("python") != hash_token("java")
        # Кириллица хэшируется без ошибок
        assert hash_token("язык") >= 0

    def test_bm25_sparse_vector_document(self):
        from src.vector_store import bm25_sparse_vector, hash_token

        vec = bm25_sparse_vector("python python java")
        assert len(vec.indices) == 2
        assert set(vec.indices) == {hash_token("python"), hash_token("java")}
        weights = dict(zip(vec.indices, vec.values))
        # TF-сатурация: больше вхождений — больше вес, но все веса > 0
        assert weights[hash_token("python")] > weights[hash_token("java")] > 0

    def test_bm25_sparse_vector_query_weights_are_one(self):
        from src.vector_store import bm25_sparse_vector

        vec = bm25_sparse_vector("python python java", is_query=True)
        assert len(vec.indices) == 2
        assert all(v == 1.0 for v in vec.values)

    def test_bm25_sparse_vector_empty_text(self):
        from src.vector_store import bm25_sparse_vector

        for text in ("", "   ", "!!! ---"):
            vec = bm25_sparse_vector(text)
            assert vec.indices == []
            assert vec.values == []

    def test_bm25_sparse_vector_tf_saturation(self):
        """TF растёт с числом вхождений, но с насыщением (вклад убывает)."""
        from src.vector_store import bm25_sparse_vector

        w1 = bm25_sparse_vector("python").values[0]
        w3 = bm25_sparse_vector("python python python").values[0]
        w9 = bm25_sparse_vector(" ".join(["python"] * 9)).values[0]
        assert w1 < w3 < w9
        assert (w3 - w1) > (w9 - w3) / 3  # прирост замедляется

    def test_to_qdrant_filter_none_and_empty(self):
        from src.vector_store import to_qdrant_filter

        assert to_qdrant_filter(None) is None
        assert to_qdrant_filter({}) is None

    def test_to_qdrant_filter_scalar_and_list(self):
        from qdrant_client import models

        from src.vector_store import to_qdrant_filter

        filt = to_qdrant_filter({"category": "a", "tags": ["x", "y"]})
        assert isinstance(filt, models.Filter)
        assert len(filt.must) == 2
        by_key = {c.key: c for c in filt.must}
        assert by_key["metadata.category"].match == models.MatchValue(value="a")
        assert by_key["metadata.tags"].match == models.MatchAny(any=["x", "y"])
