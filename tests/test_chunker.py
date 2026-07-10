"""Тесты чанкования (Фаза 5): paragraph → sentence → token splitting."""

import pytest

from src.chunker import Chunker


@pytest.fixture(scope="module")
def chunker():
    return Chunker(chunk_size=64, chunk_overlap=16)


def _make_paragraphs(n: int, words_per_paragraph: int = 40) -> str:
    return "\n\n".join(
        " ".join(f"word{p}x{w}" for w in range(words_per_paragraph))
        for p in range(n)
    )


class TestChunker:
    def test_invalid_overlap_raises(self):
        with pytest.raises(ValueError):
            Chunker(chunk_size=100, chunk_overlap=100)

    def test_short_text_single_chunk(self, chunker):
        chunks = chunker.chunk("Short text.", doc_id="doc-1")
        assert len(chunks) == 1
        assert chunks[0]["doc_id"] == "doc-1"
        assert chunks[0]["parent_doc_id"] is None
        assert chunks[0]["chunk_index"] is None
        assert chunks[0]["text"] == "Short text."

    def test_long_text_multiple_chunks(self, chunker):
        text = _make_paragraphs(6)
        chunks = chunker.chunk(text, doc_id="parent-1")
        assert len(chunks) > 1
        for i, ch in enumerate(chunks):
            assert ch["parent_doc_id"] == "parent-1"
            assert ch["chunk_index"] == i
            assert ch["doc_id"] == f"parent-1#{i}"

    def test_chunk_token_budget_respected(self, chunker):
        text = _make_paragraphs(8)
        for ch in chunker.chunk(text):
            # строгий инвариант: chunk_size включает overlap-хвост и разделитель
            assert chunker.count_tokens(ch["text"]) <= chunker.chunk_size

    def test_all_content_preserved(self, chunker):
        # Предложения с точками → нарезка по границам предложений (не hard-split
        # по токенам, который может разорвать слово). Так проверяется, что ни одно
        # слово не теряется при склейке единиц в чанки.
        sentences = [f"This is sentence number {i} describing topic {i}." for i in range(30)]
        text = " ".join(sentences)
        chunks = chunker.chunk(text)
        assert len(chunks) > 1
        combined = " ".join(ch["text"] for ch in chunks)
        for i in range(30):
            assert f"number {i} " in combined or f"topic {i}." in combined

    def test_overlap_between_chunks(self, chunker):
        text = _make_paragraphs(6)
        chunks = chunker.chunk(text)
        assert len(chunks) >= 2
        # начало следующего чанка повторяет хвост предыдущего
        for prev, nxt in zip(chunks, chunks[1:]):
            tail = chunker._tail_tokens(prev["text"])
            if tail:
                assert nxt["text"].startswith(tail)

    def test_metadata_propagated_with_parent_fields(self, chunker):
        text = _make_paragraphs(4)
        chunks = chunker.chunk(text, doc_id="p1", metadata={"source": "wiki"})
        for ch in chunks:
            assert ch["metadata"]["source"] == "wiki"
            assert ch["metadata"]["parent_doc_id"] == "p1"
            assert ch["metadata"]["chunk_index"] == ch["chunk_index"]

    def test_single_chunk_metadata_untouched(self, chunker):
        chunks = chunker.chunk("tiny", metadata={"a": 1})
        assert chunks[0]["metadata"] == {"a": 1}
        assert "parent_doc_id" not in chunks[0]["metadata"]

    def test_giant_sentence_hard_split(self, chunker):
        # одно "предложение" без точек и параграфов длиннее chunk_size
        text = " ".join(f"tok{i}" for i in range(400))
        chunks = chunker.chunk(text)
        assert len(chunks) > 1

    def test_sentences_split_before_tokens(self, chunker):
        # параграф из многих предложений: нарезка должна идти по границам предложений
        sentences = [f"Sentence number {i} about topic {i}." for i in range(40)]
        text = " ".join(sentences)
        chunks = chunker.chunk(text)
        assert len(chunks) > 1
        # каждый чанк (кроме overlap-хвоста) начинается с начала предложения
        for ch in chunks[1:]:
            assert "Sentence" in ch["text"]

    def test_generated_doc_id_when_missing(self, chunker):
        chunks = chunker.chunk("hello world")
        assert chunks[0]["doc_id"]

    def test_empty_paragraphs_ignored(self, chunker):
        text = "First paragraph.\n\n\n\n\n\nSecond paragraph."
        chunks = chunker.chunk(text)
        assert len(chunks) == 1

    def test_needs_chunking(self, chunker):
        assert not chunker.needs_chunking("short")
        assert chunker.needs_chunking(_make_paragraphs(6))

    def test_cyrillic_text(self, chunker):
        text = "\n\n".join(
            " ".join(f"слово{p}номер{w}" for w in range(30))
            for p in range(6)
        )
        chunks = chunker.chunk(text)
        assert len(chunks) > 1
        combined = " ".join(ch["text"] for ch in chunks)
        assert "слово0номер0" in combined
        assert "слово5номер29" in combined
