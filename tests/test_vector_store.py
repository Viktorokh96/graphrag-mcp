"""Тесты для VectorStore (ChromaDB)."""

import pytest
import os
import shutil
import tempfile


class TestVectorStore:
    """Тесты для хранилища векторов на ChromaDB."""

    @pytest.fixture(autouse=True)
    def setup_temp_dir(self):
        """Каждый тест использует временную директорию."""
        self.temp_dir = tempfile.mkdtemp()
        self.store_path = os.path.join(self.temp_dir, "rag_data")
        yield
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def test_import(self):
        from src.vector_store import VectorStore
        assert VectorStore is not None

    def test_init_creates_collection(self):
        from src.vector_store import VectorStore
        store = VectorStore(store_path=self.store_path)
        assert store is not None
        # Коллекция должна существовать
        assert store.collection is not None

    def test_add_and_search(self):
        from src.vector_store import VectorStore
        store = VectorStore(store_path=self.store_path)

        doc_id = store.add("doc1", "hello world", [0.1, 0.2, 0.3])
        assert doc_id == "doc1"

        results = store.search([0.1, 0.2, 0.3], k=1)
        assert len(results) == 1
        assert results[0][0] == "doc1"  # doc_id

    def test_search_by_similarity(self):
        from src.vector_store import VectorStore
        store = VectorStore(store_path=self.store_path)

        store.add("cat", "cat animal", [0.9, 0.1, 0.0])
        store.add("dog", "dog animal", [0.1, 0.9, 0.0])
        store.add("car", "car vehicle", [0.0, 0.0, 1.0])

        # Ищем похожее на кота
        results = store.search([0.85, 0.15, 0.0], k=2)
        assert results[0][0] == "cat"
        assert results[0][2] > results[1][2]  # score cat > score dog

    def test_top_k(self):
        from src.vector_store import VectorStore
        store = VectorStore(store_path=self.store_path)

        for i in range(10):
            store.add(f"doc{i}", f"text {i}", [float(i) / 10, 0.0, 0.0])

        results = store.search([0.5, 0.0, 0.0], k=3)
        assert len(results) == 3

    def test_clear(self):
        from src.vector_store import VectorStore
        store = VectorStore(store_path=self.store_path)

        store.add("doc1", "some text", [0.1, 0.2])
        assert store.count() == 1

        store.clear()
        assert store.count() == 0

    def test_stats(self):
        from src.vector_store import VectorStore
        store = VectorStore(store_path=self.store_path)

        stats = store.stats()
        assert stats["total_documents"] == 0
        assert stats["store_path"] == self.store_path

        store.add("doc1", "text1", [0.1, 0.2])
        store.add("doc2", "text2", [0.3, 0.4])
        stats = store.stats()
        assert stats["total_documents"] == 2

    def test_remove(self):
        from src.vector_store import VectorStore
        store = VectorStore(store_path=self.store_path)

        store.add("doc1", "text1", [0.1, 0.2])
        store.add("doc2", "text2", [0.3, 0.4])

        store.remove("doc1")
        assert store.count() == 1

    def test_count(self):
        from src.vector_store import VectorStore
        store = VectorStore(store_path=self.store_path)

        assert store.count() == 0
        store.add("doc1", "text", [0.1, 0.2])
        assert store.count() == 1

    def test_get_all_texts(self):
        from src.vector_store import VectorStore
        store = VectorStore(store_path=self.store_path)

        store.add("doc1", "hello world", [0.1, 0.2])
        store.add("doc2", "foo bar", [0.3, 0.4])

        texts = store.get_all_texts()
        assert len(texts) == 2
        assert "hello world" in texts
        assert "foo bar" in texts

    def test_persistence_across_reload(self):
        """Данные сохраняются между перезагрузками."""
        from src.vector_store import VectorStore

        # Первая сессия
        store1 = VectorStore(store_path=self.store_path)
        store1.add("persist_doc", "persistent text", [0.5, 0.5])
        assert store1.count() == 1

        # Вторая сессия (та же директория)
        store2 = VectorStore(store_path=self.store_path)
        assert store2.count() == 1

        results = store2.search([0.5, 0.5], k=1)
        assert len(results) == 1
        assert results[0][0] == "persist_doc"

    def test_metadata_support(self):
        from src.vector_store import VectorStore
        store = VectorStore(store_path=self.store_path)

        store.add("doc1", "text with meta", [0.1, 0.2], metadata={"category": "test", "page": 5})
        results = store.search([0.1, 0.2], k=1)

        assert len(results) == 1
        doc_id, text, score, meta = results[0]
        assert meta["category"] == "test"
        assert meta["page"] == 5

    def test_get_all_returns_ids_texts_metadata(self):
        from src.vector_store import VectorStore
        store = VectorStore(store_path=self.store_path)

        store.add("doc1", "hello", [0.1, 0.2], metadata={"x": 1})
        store.add("doc2", "world", [0.3, 0.4])

        all_docs = store.get_all()
        assert len(all_docs) == 2
        ids = [d[0] for d in all_docs]
        assert "doc1" in ids
        assert "doc2" in ids
        for did, text, meta in all_docs:
            if did == "doc1":
                assert text == "hello"
                assert meta == {"x": 1}
            if did == "doc2":
                assert text == "world"
                assert meta == {}

    def test_get_all_empty(self):
        from src.vector_store import VectorStore
        store = VectorStore(store_path=self.store_path)
        assert store.get_all() == []

    def test_update_embedding(self):
        from src.vector_store import VectorStore
        store = VectorStore(store_path=self.store_path)

        store.add("doc1", "text", [0.1, 0.2])
        store.update_embedding("doc1", [0.9, 0.8])
        results = store.search([0.9, 0.8], k=1)
        assert len(results) == 1
        assert results[0][0] == "doc1"

    def test_recreate_collection(self):
        from src.vector_store import VectorStore
        store = VectorStore(store_path=self.store_path)

        store.add("doc1", "text", [0.1, 0.2])
        assert store.count() == 1

        store.recreate_collection()
        assert store.count() == 0

        # Можно добавлять с новой размерностью
        store.add("doc2", "text2", [0.1, 0.2, 0.3, 0.4])
        assert store.count() == 1
        assert store.stats()["dimension"] == 4

    def test_update_embedding_wrong_dimension_raises(self):
        from src.vector_store import VectorStore
        store = VectorStore(store_path=self.store_path)

        store.add("doc1", "text", [0.1, 0.2])
        with pytest.raises(Exception):
            store.update_embedding("doc1", [0.1, 0.2, 0.3, 0.4])