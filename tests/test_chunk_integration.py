"""Интеграция чанкования в RAGSystem.add_document (Фаза 5b).

Большой документ индексируется несколькими chunk-точками в Qdrant под общим
doc_id, но в DocumentStore остаётся цельным. Поиск схлопывает чанки обратно в
один doc_id; get/list/delete/dedup работают на уровне документа.
"""

import pytest

from src.config import RAGConfig
from tests.conftest import HashEmbeddingGenerator


@pytest.fixture
def chunk_rag(make_rag):
    """RAGSystem с маленьким chunk_size, чтобы средние документы чанковались."""
    cfg = RAGConfig(chunk_size=32, chunk_overlap=8)
    return make_rag(embedder=HashEmbeddingGenerator(), config=cfg)


def _big_text(n_sentences: int = 40) -> str:
    return " ".join(f"Sentence {i} about distinct subject matter numbered {i}." for i in range(n_sentences))


class TestChunkIntegration:
    def test_small_doc_single_point(self, chunk_rag):
        text = "Short enough document that stays a single vector point in the store."
        doc_id = chunk_rag.add_document(text)
        # одна точка на документ
        assert chunk_rag.vector_store.count() == 1
        assert chunk_rag.doc_store.count() == 1
        assert chunk_rag.vector_store.has(doc_id)

    def test_large_doc_multiple_points_single_record(self, chunk_rag):
        text = _big_text()
        doc_id = chunk_rag.add_document(text)
        # документ цельный в DocumentStore
        assert chunk_rag.doc_store.count() == 1
        assert chunk_rag.stats()["total_documents"] == 1
        # но в Qdrant несколько chunk-точек
        assert chunk_rag.vector_store.count() > 1
        # get_document возвращает ПОЛНЫЙ текст
        doc = chunk_rag.get_document(doc_id)
        assert doc["text"] == text
        assert doc["total_chars"] == len(text)

    def test_search_dedups_to_single_doc(self, chunk_rag):
        text = _big_text()
        doc_id = chunk_rag.add_document(text)
        results = chunk_rag.bm25_search("distinct subject matter", k=5)
        ids = [r[0] for r in results]
        # документ встречается ровно один раз, несмотря на несколько чанков
        assert ids.count(doc_id) == 1

    def test_tail_content_searchable(self, chunk_rag):
        # уникальный токен только в самом конце длинного документа
        text = _big_text(40) + " zzzuniquetail marker at the very end of the document."
        doc_id = chunk_rag.add_document(text)
        results = chunk_rag.bm25_search("zzzuniquetail", k=5)
        assert results and results[0][0] == doc_id

    def test_list_shows_one_document(self, chunk_rag):
        chunk_rag.add_document(_big_text())
        listing = chunk_rag.list_documents(limit=10)
        assert listing["total"] == 1
        assert len(listing["documents"]) == 1

    def test_delete_removes_all_chunks(self, chunk_rag):
        doc_id = chunk_rag.add_document(_big_text())
        assert chunk_rag.vector_store.count() > 1
        assert chunk_rag.delete_document(doc_id) is True
        assert chunk_rag.vector_store.count() == 0
        assert chunk_rag.doc_store.count() == 0
        assert not chunk_rag.vector_store.has(doc_id)

    def test_metadata_filter_on_chunked_doc(self, chunk_rag):
        chunk_rag.add_document(_big_text(), metadata={"kind": "big"})
        chunk_rag.add_document("Small document that does not get chunked at all in this test case.", metadata={"kind": "small"})
        results = chunk_rag.bm25_search("subject matter", k=5, metadata_filter={"kind": "big"})
        assert all(r[3].get("kind") == "big" for r in results)

    def test_dedup_still_works_for_chunked(self, chunk_rag):
        text = _big_text()
        id1 = chunk_rag.add_document(text)
        id2 = chunk_rag.add_document(text)
        assert id1 == id2
        assert chunk_rag.doc_store.count() == 1

    def test_reindex_preserves_chunking(self, chunk_rag):
        doc_id = chunk_rag.add_document(_big_text())
        points_before = chunk_rag.vector_store.count()
        assert points_before > 1
        chunk_rag.reindex()
        assert chunk_rag.vector_store.count() == points_before
        assert chunk_rag.bm25_search("distinct subject", k=5)[0][0] == doc_id
