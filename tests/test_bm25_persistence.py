"""Тесты персистентности BM25: sparse-векторы хранятся в embedded Qdrant
на диске и переживают переоткрытие QdrantVectorStore."""

import pytest

DIM = 8


def unit(i: int, dim: int = DIM) -> list[float]:
    v = [0.0] * dim
    v[i % dim] = 1.0
    return v


@pytest.fixture
def make_store(tmp_path):
    """Фабрика сторов с гарантированным close() в teardown (Windows file lock).

    Один embedded path нельзя открыть двумя клиентами одновременно —
    тесты обязаны close()'ить стор перед переоткрытием того же пути.
    """
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


class TestBM25Persistence:
    """Сохранение и восстановление BM25-индекса через переоткрытие стора."""

    def test_reopen_empty(self, make_store):
        """Пустой стор переоткрывается пустым."""
        store = make_store()
        store.close()

        store2 = make_store()
        assert store2.count() == 0
        assert store2.bm25_search("anything", k=5) == []

    def test_bm25_search_after_reopen(self, make_store):
        """BM25-поиск работает после переоткрытия стора."""
        store = make_store()
        store.add("doc1", "python programming language", unit(0))
        store.add("doc2", "machine learning algorithms", unit(1))
        store.close()

        store2 = make_store()
        assert store2.count() == 2

        results = store2.bm25_search("python", k=5)
        assert len(results) == 1
        assert results[0]["doc_id"] == "doc1"
        assert results[0]["score"] > 0

    def test_ranking_preserved_after_reopen(self, make_store):
        """Ранжирование (TF/длина документа) сохраняется после перезагрузки."""
        store = make_store()
        store.add("dense_doc", "python python python", unit(0))
        store.add("sparse_doc", "python appears once in this considerably longer document text", unit(1))
        store.close()

        store2 = make_store()
        results = store2.bm25_search("python", k=2)
        assert len(results) == 2
        assert results[0]["doc_id"] == "dense_doc"
        assert results[0]["score"] > results[1]["score"]

    def test_add_after_reopen(self, make_store):
        """Добавление документов после переоткрытия работает."""
        store = make_store()
        store.add("doc1", "python programming", unit(0))
        store.close()

        store2 = make_store()
        store2.add("doc2", "java programming", unit(1))
        store2.close()

        store3 = make_store()
        assert store3.count() == 2
        results = store3.bm25_search("java", k=5)
        assert len(results) == 1
        assert results[0]["doc_id"] == "doc2"

    def test_metadata_preserved_after_reopen(self, make_store):
        """Метаданные переживают переоткрытие и доступны фильтрам."""
        store = make_store()
        store.add("doc1", "python guide", unit(0), metadata={"lang": "python", "page": 3})
        store.add("doc2", "python and java", unit(1), metadata={"lang": "java", "page": 7})
        store.close()

        store2 = make_store()
        results = store2.bm25_search("python", k=5, metadata_filter={"lang": "java"})
        assert len(results) == 1
        assert results[0]["doc_id"] == "doc2"
        assert results[0]["metadata"] == {"lang": "java", "page": 7}

    def test_clear_persists(self, make_store):
        """Очистка стора персистентна: после переоткрытия индекс пуст."""
        store = make_store()
        store.add("doc1", "python programming", unit(0))
        assert store.bm25_search("python", k=5)

        store.clear()
        store.close()

        store2 = make_store()
        assert store2.count() == 0
        assert store2.bm25_search("python", k=5) == []

    def test_remove_persists(self, make_store):
        """Удаление документа персистентно."""
        store = make_store()
        store.add("doc1", "python programming", unit(0))
        store.add("doc2", "java programming", unit(1))
        store.remove("doc1")
        store.close()

        store2 = make_store()
        assert store2.count() == 1
        results = store2.bm25_search("python programming", k=5)
        assert "doc1" not in [r["doc_id"] for r in results]

    def test_tokenization_regex_lowercase(self, make_store):
        """Токенизация — regex + lower case: пунктуация не индексируется,
        регистр и кириллица переживают перезагрузку."""
        store = make_store()
        store.add("doc1", "Python — мощный язык программирования!", unit(0))
        store.close()

        store2 = make_store()
        # Слова находятся независимо от регистра
        assert store2.bm25_search("python", k=5)[0]["doc_id"] == "doc1"
        assert store2.bm25_search("мощный", k=5)[0]["doc_id"] == "doc1"
        # Пунктуация не токенизируется — запрос без токенов пуст
        assert store2.bm25_search("—", k=5) == []
        assert store2.bm25_search("!", k=5) == []

    def test_search_case_insensitive_after_reopen(self, make_store):
        """Поиск нечувствителен к регистру после перезагрузки."""
        store = make_store()
        store.add("doc1", "Python Programming Language", unit(0))
        store.close()

        store2 = make_store()
        results_lower = store2.bm25_search("python", k=5)
        results_upper = store2.bm25_search("PYTHON", k=5)

        assert len(results_lower) == 1
        assert len(results_upper) == 1
        assert results_lower[0]["doc_id"] == "doc1"
        assert results_upper[0]["doc_id"] == "doc1"
