"""Тесты для RAGSystem (оркестратор с гибридным поиском) на стеке Qdrant + SQLite.

Все тесты используют фикстуры conftest (rag / make_rag) с mock-эмбеддерами —
реальные модели не загружаются, сети нет. Фикстуры сами вызывают rag.close().
"""

import os

import pytest


class TestRAGSystem:
    """Тесты для RAG системы."""

    def test_add_document(self, rag):
        doc_id = rag.add_document("Hello world, this is a test document for the RAG system.")

        assert doc_id is not None
        assert isinstance(doc_id, str)
        assert len(doc_id) > 0

    def test_add_and_search(self, rag):
        rag.add_document("Python is a programming language used for various applications.")
        rag.add_document("Java runs on a virtual machine and is used for enterprise apps.")
        rag.add_document("Python is great for machine learning and data science tasks.")

        results = rag.search("Python programming", k=2)
        assert len(results) == 2

    def test_bm25_search(self, rag):
        rag.add_document("python programming language is widely used for many purposes.")
        rag.add_document("java programming language is widely used for many purposes.")

        results = rag.bm25_search("python", k=1)
        assert len(results) == 1
        assert "python" in results[0][1].lower()

    def test_search_hybrid(self, rag):
        rag.add_document("python programming language is widely used for many purposes.")
        rag.add_document("java programming language is widely used for many purposes.")

        # pure semantic (alpha=1.0): оба документа возвращаются dense-каналом
        results_sem = rag.search_hybrid("python", k=2, alpha=1.0)
        assert len(results_sem) == 2

        # pure bm25 (alpha=0.0): токен "python" есть только в одном документе;
        # sem-only документы исключаются при крайнем alpha (alpha-dilution)
        results_bm = rag.search_hybrid("python", k=2, alpha=0.0)
        assert len(results_bm) == 1
        assert "python" in results_bm[0][1].lower()

        # balanced
        results_mix = rag.search_hybrid("python", k=2, alpha=0.5)
        assert len(results_mix) == 2

    def test_clear(self, rag):
        rag.add_document("some text content that is sufficiently long for the test to pass.")
        stats_before = rag.stats()
        assert stats_before["total_documents"] > 0

        rag.clear()
        stats = rag.stats()
        assert stats["total_documents"] == 0

    def test_stats(self, rag):
        stats = rag.stats()
        assert "total_documents" in stats
        assert "store_path" in stats
        assert "dimension" in stats
        assert "total_nodes" in stats
        assert "total_edges" in stats
        assert "relation_types" in stats

    def test_add_file(self, rag, tmp_path):
        test_file = os.path.join(str(tmp_path), "test_doc.txt")
        with open(test_file, "w", encoding="utf-8") as f:
            f.write("This is a test file content for indexing purposes in our RAG system.")

        doc_id = rag.add_file(test_file)

        assert doc_id is not None
        assert isinstance(doc_id, str)

    def test_search_empty_index(self, rag):
        results = rag.search("anything")
        assert results == []

    def test_search_empty_bm25(self, rag):
        results = rag.bm25_search("anything")
        assert results == []

    def test_search_empty_hybrid(self, rag):
        results = rag.search_hybrid("anything")
        assert results == []

    def test_hybrid_with_custom_alpha(self, rag):
        rag.add_document("python programming language is widely used for many purposes.")

        results = rag.search_hybrid("python", k=1, alpha=0.75)
        assert len(results) == 1

    def test_add_documents_batch(self, rag):
        doc_ids = rag.add_documents(
            [
                "First document for testing the RAG system batch indexing feature.",
                "Second document for testing the RAG system batch indexing feature.",
                "Third document for testing the RAG system batch indexing feature.",
            ],
            metadata=[{"idx": 1}, {"idx": 2}, {"idx": 3}],
        )
        assert len(doc_ids) == 3

        stats = rag.stats()
        assert stats["total_documents"] == 3

    def test_add_document_too_short(self, rag):
        """D5: Документ короче MIN_CONTENT_LENGTH вызывает ValueError."""
        with pytest.raises(ValueError, match="too short"):
            rag.add_document("short")

    def test_add_document_duplicate(self, rag):
        """D6: Добавление дубликата возвращает существующий doc_id."""
        text = "This is a sufficiently long document for testing deduplication in the RAG system."
        assert rag.is_duplicate(text) is None, "Before adding, should not be a duplicate"
        doc_id1 = rag.add_document(text)
        assert rag.is_duplicate(text) == doc_id1, "is_duplicate should return existing doc_id"
        doc_id2 = rag.add_document(text)
        assert doc_id1 == doc_id2, "Duplicate documents should return the same doc_id"

    def test_dedup_unicode_nfc_nfd(self, rag):
        """Один и тот же текст в NFC и NFD должен дедуплицироваться (NFC-нормализация)."""
        import unicodedata

        base = "Café details for the naïve résumé of Zoë — long enough to pass min length."
        nfc = unicodedata.normalize("NFC", base)
        nfd = unicodedata.normalize("NFD", base)
        assert nfc != nfd, "тестовые строки должны различаться на уровне байт"
        doc_id1 = rag.add_document(nfc)
        doc_id2 = rag.add_document(nfd)
        assert doc_id1 == doc_id2, "NFC и NFD варианты одного текста должны совпадать"
        assert rag.stats()["total_documents"] == 1

    def test_store_sync_on_init(self, make_rag):
        """D1: _sync_stores() при инициализации: DocumentStore — источник правды.

        Фантомные точки Qdrant (без документа) удаляются; документы,
        отсутствующие в Qdrant, переиндексируются.
        """
        rag1 = make_rag()
        doc_id = rag1.add_document(
            "This is a valid document that is long enough to pass the length check."
        )

        # Фантомная точка в Qdrant — документа с таким id нет в DocumentStore
        phantom_text = "This phantom point exists only in the Qdrant vector store"
        rag1.vector_store.add(
            doc_id="phantom-vector",
            text=phantom_text,
            embedding=rag1.embedding_generator.get_embedding(phantom_text),
        )
        assert "phantom-vector" in rag1.vector_store.get_all_ids()

        # Документ только в DocumentStore — в Qdrant отсутствует
        missing_text = "This document exists only in the SQLite document store for now."
        rag1.doc_store.add(
            "only-in-doc-store", missing_text, None,
            content_hash=rag1._compute_hash(missing_text),
        )
        assert not rag1.vector_store.has("only-in-doc-store")

        # Освобождаем локи Qdrant/SQLite перед повторным открытием того же стора
        rag1.close()

        # Новый RAGSystem на том же store_path — _sync_stores() подчистит
        rag2 = make_rag()

        assert "phantom-vector" not in rag2.vector_store.get_all_ids(), (
            "Фантомная точка Qdrant должна быть удалена при sync"
        )
        assert rag2.vector_store.has("only-in-doc-store"), (
            "Документ из DocumentStore должен быть переиндексирован в Qdrant"
        )
        # Реальный документ сохранился в обоих хранилищах
        assert rag2.doc_store.get(doc_id) is not None
        assert rag2.vector_store.has(doc_id)

    def test_metadata_always_dict_in_get_document(self, rag):
        """D10: metadata в get_document всегда dict, не None."""
        doc_id = rag.add_document("A sufficiently long document for testing metadata normalization.")

        doc = rag.get_document(doc_id)
        assert doc is not None
        assert isinstance(doc["metadata"], dict), f"metadata should be dict, got {type(doc['metadata'])}"
        assert doc["metadata"] is not None

    def test_metadata_always_dict_in_list_documents(self, rag):
        """D10: metadata в list_documents всегда dict, не None."""
        rag.add_document("A sufficiently long document for testing metadata normalization.")
        rag.add_document(
            "Another sufficiently long document with metadata for testing purposes.",
            metadata={"key": "val"},
        )

        result = rag.list_documents(limit=10)
        for doc in result["documents"]:
            assert isinstance(doc["metadata"], dict), (
                f"metadata should be dict, got {type(doc['metadata'])} for {doc['doc_id']}"
            )
            assert doc["metadata"] is not None

    def test_metadata_always_dict_in_search(self, rag):
        """D10: metadata в search/bm25_search/search_hybrid всегда dict, не None."""
        rag.add_document("A sufficiently long document for testing metadata normalization in searches.")

        for name, search_fn in [
            ("search", rag.search),
            ("bm25_search", rag.bm25_search),
            ("search_hybrid", lambda q: rag.search_hybrid(q, alpha=0.5)),
        ]:
            results = search_fn("document")
            assert results, f"{name} should return results"
            for r in results:
                meta = r[3]  # metadata — четвёртый элемент кортежа
                assert isinstance(meta, dict), (
                    f"metadata should be dict in {name}, got {type(meta)}"
                )
                assert meta is not None

    def test_rrf_no_ties(self, rag):
        """D7: RRF не даёт тай-оффов в топ-3."""
        rag.add_document("Python programming language for building web applications and data science.")
        rag.add_document("Java enterprise development platform for large-scale mobile and cloud apps.")
        rag.add_document("JavaScript frontend framework for building interactive user interfaces.")
        rag.add_document("C++ systems programming for game engines and real-time graphics rendering.")
        rag.add_document("Ruby on Rails web framework for rapid application development.")

        results = rag.search_hybrid("programming language", k=3, alpha=0.5)
        assert len(results) >= 2
        scores = [r[2] for r in results[:3]]
        assert len(set(scores)) == len(scores), f"RRF scores should be distinct: {scores}"
