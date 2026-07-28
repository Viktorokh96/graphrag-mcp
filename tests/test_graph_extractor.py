"""Тесты для src/graph_extractor.py — извлечение графа (LLM-триплеты + NER)."""

import sys
import types
from unittest.mock import MagicMock, patch

import pytest

from src.graph_extractor import GraphExtractor


class FakeRAG:
    """Минимальный стаб RAGSystem: помнит документы и рёбра."""

    def __init__(self, existing: dict | None = None):
        self.documents: dict[str, dict] = dict(existing or {})
        self.relations: list[tuple] = []

    def get_document(self, doc_id, limit=None, **kwargs):
        return self.documents.get(doc_id)

    def add_document(self, text, metadata=None, doc_id=None, **kwargs):
        self.documents[doc_id] = {"text": text, "metadata": metadata}
        return doc_id

    def add_relation(self, source_id, target_id, relation, weight=1.0):
        self.relations.append((source_id, target_id, relation, weight))


def _extractor(rag=None, response=""):
    ex = GraphExtractor(rag or FakeRAG(), model="mock", base_url="http://localhost:1")
    client = MagicMock()
    client.generate.return_value = {"response": response}
    ex._client = client
    return ex


class TestEntityDocId:
    def test_stable_and_prefixed(self):
        ex = _extractor()
        assert ex._entity_doc_id("Qdrant").startswith("entity-")
        assert ex._entity_doc_id("Qdrant") == ex._entity_doc_id("Qdrant")

    def test_case_and_whitespace_insensitive(self):
        ex = _extractor()
        assert ex._entity_doc_id("  QDRANT ") == ex._entity_doc_id("qdrant")

    def test_different_names_differ(self):
        ex = _extractor()
        assert ex._entity_doc_id("a") != ex._entity_doc_id("b")


class TestEnsureClient:
    def test_creates_ollama_client_once(self):
        ex = GraphExtractor(FakeRAG(), model="mock", base_url="http://host:11434")
        fake_client = MagicMock()
        fake_module = types.ModuleType("ollama")
        fake_module.Client = MagicMock(return_value=fake_client)
        with patch.dict(sys.modules, {"ollama": fake_module}):
            assert ex._ensure_client() is fake_client
            assert ex._ensure_client() is fake_client
        fake_module.Client.assert_called_once_with(host="http://host:11434", timeout=120.0)


class TestEnsureEntity:
    def test_creates_missing_entity_document(self):
        rag = FakeRAG()
        ex = _extractor(rag)
        eid = ex._ensure_entity("Qdrant")
        assert eid in rag.documents
        assert rag.documents[eid]["text"] == "Qdrant"
        assert rag.documents[eid]["metadata"] == {
            "type": "entity",
            "auto_extracted": True,
            "name": "Qdrant",
        }

    def test_reuses_existing_entity_document(self):
        rag = FakeRAG()
        ex = _extractor(rag)
        eid = ex._ensure_entity("Qdrant")
        rag.documents[eid]["text"] = "untouched"
        assert ex._ensure_entity("Qdrant") == eid
        assert rag.documents[eid]["text"] == "untouched"


class TestExtractLlm:
    def test_creates_entities_and_edges_for_triple(self):
        rag = FakeRAG()
        ex = _extractor(rag, "Qdrant | stores | vectors")
        results = ex.extract_and_link("doc-1", "text")

        assert len(results) == 1
        assert results[0]["triple"] == ("Qdrant", "stores", "vectors")
        assert results[0]["source"] == "doc-1"
        id_a, id_b = results[0]["entity_a_id"], results[0]["entity_b_id"]
        assert rag.relations == [
            ("doc-1", id_a, "mentions", 0.9),
            ("doc-1", id_b, "mentions", 0.9),
            (id_a, id_b, "stores", 1.0),
        ]

    def test_skips_lines_without_separator(self):
        ex = _extractor(response="just prose\n\nA | rel | B\nincomplete | line")
        results = ex.extract_and_link("doc-1", "text")
        assert [r["triple"] for r in results] == [("A", "rel", "B")]

    def test_extra_separators_join_into_target(self):
        ex = _extractor(response="A | rel | B | C")
        results = ex.extract_and_link("doc-1", "text")
        assert results[0]["triple"] == ("A", "rel", "B | C")

    def test_empty_response_gives_no_results(self):
        rag = FakeRAG()
        ex = _extractor(rag, "")
        assert ex.extract_and_link("doc-1", "text") == []
        assert rag.relations == []

    def test_prompt_truncates_long_text_and_uses_model(self):
        ex = _extractor(response="")
        ex.extract_and_link("doc-1", "x" * 5000)
        kwargs = ex._client.generate.call_args[1]
        assert kwargs["model"] == "mock"
        assert "x" * 4000 in kwargs["prompt"]
        assert "x" * 4001 not in kwargs["prompt"]

    def test_default_mode_is_llm(self):
        ex = _extractor(response="A | rel | B")
        assert ex.extract_and_link("doc-1", "text") == ex.extract_and_link("doc-1", "text", mode="llm")


class TestExtractNer:
    def test_returns_error_when_spacy_missing(self):
        ex = _extractor()
        with patch("importlib.util.find_spec", return_value=None):
            with pytest.raises(RuntimeError, match="spaCy is not installed"):
                ex.extract_and_link("doc-1", "text", mode="ner")

    def test_returns_error_when_model_not_downloaded(self):
        ex = _extractor()
        fake_spacy = types.ModuleType("spacy")
        fake_spacy.load = MagicMock(side_effect=OSError("missing"))
        with patch("importlib.util.find_spec", return_value=object()), \
                patch.dict(sys.modules, {"spacy": fake_spacy}):
            with pytest.raises(RuntimeError, match="en_core_web_sm"):
                ex.extract_and_link("doc-1", "text", mode="ner")
    def test_links_supported_entities_and_deduplicates(self):
        rag = FakeRAG()
        ex = _extractor(rag)

        def _ent(text, label):
            e = MagicMock()
            e.text = text
            e.label_ = label
            return e

        doc = MagicMock()
        doc.ents = [
            _ent("Alice ", "PERSON"),
            _ent("alice", "PERSON"),
            _ent("Berlin", "GPE"),
            _ent("Tuesday", "DATE"),
            _ent("  ", "ORG"),
        ]
        fake_spacy = types.ModuleType("spacy")
        fake_spacy.load = MagicMock(return_value=MagicMock(return_value=doc))

        with patch("importlib.util.find_spec", return_value=object()), \
                patch.dict(sys.modules, {"spacy": fake_spacy}):
            result = ex.extract_and_link("doc-1", "text", mode="ner")

        assert [r["entity"] for r in result] == ["Alice", "Berlin"]
        assert [r["label"] for r in result] == ["PERSON", "GPE"]
        assert all(rel == ("doc-1", r["doc_id"], "mentions", 0.8)
                   for rel, r in zip(rag.relations, result))
        assert len(rag.relations) == 2

    def test_truncates_text_to_10k_chars(self):
        ex = _extractor()
        nlp = MagicMock()
        nlp.return_value = MagicMock(ents=[])
        fake_spacy = types.ModuleType("spacy")
        fake_spacy.load = MagicMock(return_value=nlp)

        with patch("importlib.util.find_spec", return_value=object()), \
                patch.dict(sys.modules, {"spacy": fake_spacy}):
            assert ex.extract_and_link("doc-1", "y" * 20000, mode="ner") == []
        assert len(nlp.call_args[0][0]) == 10000


@pytest.mark.parametrize("mode", ["llm", "ner"])
def test_extract_and_link_dispatches_by_mode(mode):
    ex = _extractor(response="")
    with patch.object(ex, "_extract_llm", return_value=["llm"]) as llm, \
            patch.object(ex, "_extract_ner", return_value=["ner"]) as ner:
        assert ex.extract_and_link("doc-1", "text", mode=mode) == [mode]
    assert (llm.called, ner.called) == (mode == "llm", mode == "ner")
