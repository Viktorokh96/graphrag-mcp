"""Тесты обработки несовпадения размерности эмбеддингов (Qdrant + SQLite).

Новая семантика (после миграции с ChromaDB):
  • авто-reindex при инициализации УДАЛЁН;
  • при несовпадении размерности генератора и коллекции QdrantVectorStore
    логирует warning при старте — система не падает;
  • rag.search() возвращает [] (размерность запроса != размерности коллекции);
  • BM25 (sparse-канал) работает независимо от размерности dense-векторов;
  • явный rag.reindex() пересоздаёт коллекцию под новую размерность и
    переиндексирует все документы из doc_store (источника правды).
"""

import logging

import pytest

from src.vector_store import QdrantVectorStore
from tests.conftest import HashEmbeddingGenerator


@pytest.fixture
def make_vector_store(tmp_path):
    """Фабрика QdrantVectorStore с автоматическим close() (Windows-локи)."""
    created: list[QdrantVectorStore] = []

    def _make(subdir: str = "qdrant", dimension: int = 4) -> QdrantVectorStore:
        store = QdrantVectorStore(location=str(tmp_path / subdir), dimension=dimension)
        created.append(store)
        return store

    yield _make
    for store in created:
        try:
            store.close()
        except Exception:
            pass


class TestQdrantGetDimension:
    """QdrantVectorStore.get_dimension() — размерность dense-векторов коллекции."""

    def test_get_dimension_matches_configured(self, make_vector_store):
        """Коллекция создаётся с размерностью из конфига — get_dimension() совпадает."""
        store = make_vector_store(dimension=8)
        assert store.get_dimension() == 8

    def test_get_dimension_large_vector(self, make_vector_store):
        """Работает с большой размерностью."""
        store = make_vector_store(subdir="large", dimension=1536)
        store.add("doc1", "text", [0.1] * 1536)
        assert store.get_dimension() == 1536

    def test_recreate_collection_applies_new_dimension(self, make_vector_store):
        """recreate_collection() пересоздаёт коллекцию под новую размерность."""
        store = make_vector_store(dimension=3)
        store.add("doc1", "text", [0.1, 0.2, 0.3])
        assert store.get_dimension() == 3

        store.dimension = 16
        store.recreate_collection()
        assert store.get_dimension() == 16
        assert store.count() == 0

    def test_get_dimension_persists_across_reload(self, make_vector_store):
        """Размерность коллекции сохраняется между перезагрузками."""
        store1 = make_vector_store(dimension=4)
        store1.add("doc1", "text", [0.1, 0.2, 0.3, 0.4])
        store1.close()

        store2 = make_vector_store(dimension=4)
        assert store2.get_dimension() == 4

    def test_reload_with_other_dimension_keeps_collection_and_warns(
        self, make_vector_store, caplog
    ):
        """Открытие коллекции с другой конфиг-размерностью не падает: warning в лог,
        коллекция сохраняет исходную размерность."""
        store1 = make_vector_store(dimension=4)
        store1.add("doc1", "text", [0.1, 0.2, 0.3, 0.4])
        store1.close()

        with caplog.at_level(logging.WARNING, logger="src.vector_store"):
            store2 = make_vector_store(dimension=9)
        assert store2.get_dimension() == 4  # размерность коллекции, не конфига
        assert any(
            "dim" in rec.getMessage().lower() for rec in caplog.records
        ), "Ожидается warning о несовпадении размерности"

    def test_stats_uses_get_dimension(self, make_vector_store):
        """stats() возвращает ту же размерность, что и get_dimension()."""
        store = make_vector_store(dimension=2)
        store.add("doc1", "text", [0.1, 0.2])
        stats = store.stats()
        assert stats["dimension"] == store.get_dimension() == 2


class TestOpenStoreWithMismatchedEmbedder:
    """Открытие существующего стора эмбеддером другой размерности."""

    def test_open_does_not_crash_and_logs_warning(self, make_rag, caplog):
        """(a) Открытие стора с эмбеддером другой размерности не падает и логирует warning."""
        rag1 = make_rag(embedder=HashEmbeddingGenerator(dimension=64))
        rag1.add_document("python programming language for scripting and automation tasks")
        rag1.close()

        with caplog.at_level(logging.WARNING, logger="src.vector_store"):
            rag2 = make_rag(embedder=HashEmbeddingGenerator(dimension=32))

        # Система жива, коллекция осталась на старой размерности
        assert rag2.vector_store.get_dimension() == 64
        assert rag2.doc_store.count() == 1
        assert any("dim" in rec.getMessage().lower() for rec in caplog.records), (
            "Ожидается warning о несовпадении размерности при старте"
        )

    def test_search_returns_empty_on_mismatch(self, make_rag):
        """(b) search() возвращает [] вместо падения при несовпадении размерности."""
        rag1 = make_rag(embedder=HashEmbeddingGenerator(dimension=64))
        rag1.add_document("test document about python programming language and scripting")
        rag1.close()

        rag2 = make_rag(embedder=HashEmbeddingGenerator(dimension=32))
        assert rag2.search("python") == []

    def test_search_mismatch_after_generator_swap(self, rag):
        """search() возвращает [] и при подмене генератора на живой системе."""
        rag.add_document("test document about python programming language and scripting")
        assert len(rag.search("python")) >= 1

        rag.embedding_generator = HashEmbeddingGenerator(dimension=32)
        assert rag.search("python") == []

    def test_search_works_when_dimensions_match(self, rag):
        """search() работает корректно, когда размерности совпадают."""
        rag.add_document("python programming language for scripting and automation tasks")
        rag.add_document("java programming language for enterprise software development")

        results = rag.search("programming", k=2)
        assert len(results) == 2

    def test_bm25_search_unaffected_by_dimension_mismatch(self, make_rag):
        """(b) BM25 поиск не зависит от размерности dense-эмбеддингов."""
        rag1 = make_rag(embedder=HashEmbeddingGenerator(dimension=64))
        rag1.add_document("python programming language for general purpose scripting and automation")
        rag1.close()

        rag2 = make_rag(embedder=HashEmbeddingGenerator(dimension=32))
        results = rag2.bm25_search("python", k=1)
        assert len(results) == 1
        assert "python" in results[0][1].lower()

    def test_hybrid_search_falls_back_to_bm25_on_mismatch(self, make_rag):
        """Hybrid search опирается на BM25, когда семантический канал недоступен."""
        rag1 = make_rag(embedder=HashEmbeddingGenerator(dimension=64))
        rag1.add_document("python programming language for scripting and automation tasks")
        rag1.add_document("java programming language for enterprise software development")
        rag1.close()

        rag2 = make_rag(embedder=HashEmbeddingGenerator(dimension=32))
        results = rag2.search_hybrid("python", k=2, alpha=0.5)
        assert len(results) >= 1
        assert "python" in results[0][1].lower()


class TestExplicitReindexFixesMismatch:
    """(c)/(d) Явный reindex() чинит несовпадение размерности."""

    def test_reindex_recreates_collection_with_new_dimension(self, make_rag):
        """После reindex() search работает и get_dimension() = новая размерность."""
        rag1 = make_rag(embedder=HashEmbeddingGenerator(dimension=64))
        rag1.add_document("python programming language for scripting and automation tasks")
        rag1.add_document("java programming language for enterprise software development")
        rag1.close()

        rag2 = make_rag(embedder=HashEmbeddingGenerator(dimension=32))
        assert rag2.search("python") == []  # до reindex семантика мертва

        count = rag2.reindex()

        assert count == 2
        assert rag2.vector_store.get_dimension() == 32
        assert rag2.vector_store.count() == 2
        results = rag2.search("python", k=2)
        assert len(results) == 2

    def test_reindex_in_place_dimension_change(self, rag):
        """reindex() чинит и подмену генератора на живой системе."""
        rag.add_document("doc one for testing reindex with dimension change scenario")
        rag.add_document("doc two for testing reindex with dimension change scenario")
        assert rag.vector_store.get_dimension() == 64

        rag.embedding_generator = HashEmbeddingGenerator(dimension=7)
        count = rag.reindex()

        assert count == 2
        assert rag.vector_store.get_dimension() == 7
        assert rag.vector_store.count() == 2
        assert len(rag.search("doc one testing", k=2)) == 2

    def test_reindex_returns_document_count(self, rag):
        """(d) reindex() возвращает число переиндексированных документов."""
        for i in range(5):
            rag.add_document(f"document number {i} for testing reindex and dimension changes")

        rag.embedding_generator = HashEmbeddingGenerator(dimension=16)
        assert rag.reindex() == 5
        assert rag.vector_store.count() == 5
        assert rag.vector_store.get_dimension() == 16

    def test_reindex_empty_store_returns_zero(self, rag):
        """reindex() на пустом хранилище возвращает 0."""
        assert rag.reindex() == 0

    def test_reindex_same_dimension_preserves_ids(self, rag):
        """reindex() при совпадении размерности сохраняет все точки."""
        rag.add_document("doc one for testing reindex with same dimension scenario")
        ids_before = rag.vector_store.get_all_ids()

        count = rag.reindex()

        assert count == 1
        assert rag.vector_store.get_dimension() == 64
        assert rag.vector_store.get_all_ids() == ids_before

    def test_bm25_still_works_after_reindex(self, make_rag):
        """BM25-канал жив после reindex (sparse-векторы пересозданы вместе с dense)."""
        rag1 = make_rag(embedder=HashEmbeddingGenerator(dimension=64))
        rag1.add_document("python programming language for scripting and automation tasks")
        rag1.add_document("java programming language for enterprise software development")
        rag1.close()

        rag2 = make_rag(embedder=HashEmbeddingGenerator(dimension=32))
        rag2.reindex()

        results = rag2.bm25_search("python", k=1)
        assert len(results) == 1
        assert "python" in results[0][1].lower()
