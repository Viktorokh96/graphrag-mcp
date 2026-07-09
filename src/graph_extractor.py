"""Авто-извлечение графа из текста документа.

Два режима:
1. LLM-based (основной): Qwen3-4B через Ollama, извлекает entity-relation triples
2. NER fallback: spaCy NER для имён сущностей (без отношений)

Включается параметром extract_graph=true в rag_add_document / rag_add_structured.
"""

import hashlib

from src.rag import RAGSystem


class GraphExtractor:
    """Извлечение entity-relation triples из текста через LLM."""

    def __init__(self, rag: RAGSystem, model: str = "qwen3:4b", base_url: str = "http://localhost:11434"):
        self.rag = rag
        self.model = model
        self.base_url = base_url
        self._client = None

    def _ensure_client(self):
        if self._client is None:
            from ollama import Client
            self._client = Client(host=self.base_url, timeout=120.0)
        return self._client

    def _entity_doc_id(self, name: str) -> str:
        """Стабильный doc_id для сущности на основе хэша имени."""
        h = hashlib.sha256(name.lower().strip().encode()).hexdigest()[:32]
        return f"entity-{h}"

    def extract_and_link(
        self,
        doc_id: str,
        text: str,
        mode: str = "llm",
    ) -> list[dict]:
        """Извлечь граф из текста и создать связи в базе.

        Args:
            doc_id: документ, к которому привязываем сущности
            text: текст документа
            mode: "llm" — Qwen3-4B (триплеты), "ner" — spaCy (только сущности)

        Returns:
            [{triple: (entity_a, relation, entity_b) | entity, status, doc_id}]
        """
        if mode == "ner":
            return self._extract_ner(doc_id, text)
        return self._extract_llm(doc_id, text)

    def _extract_llm(self, doc_id: str, text: str) -> list[dict]:
        """LLM-based извлечение триплетов."""
        prompt = (
            "Extract entity-relation triples from the text below. "
            "Format each triple as: entity_a | relation | entity_b\n"
            "Return one triple per line, no numbering, no explanation.\n\n"
            f"Text:\n{text[:4000]}"
        )
        client = self._ensure_client()
        response = client.generate(model=self.model, prompt=prompt)
        raw = response.get("response", "")
        results = []
        for line in raw.split("\n"):
            line = line.strip()
            if not line or "|" not in line:
                continue
            parts = [p.strip() for p in line.split("|")]
            if len(parts) >= 3:
                entity_a, relation, entity_b = parts[0], parts[1], " | ".join(parts[2:])
                results.append(self._link_triple(doc_id, entity_a, relation, entity_b))
        return results

    def _link_triple(self, doc_id: str, entity_a: str, relation: str, entity_b: str) -> dict:
        """Создать сущности и рёбра для одного триплета."""
        id_a = self._ensure_entity(entity_a)
        id_b = self._ensure_entity(entity_b)

        self.rag.add_relation(doc_id, id_a, "mentions", weight=0.9)
        self.rag.add_relation(doc_id, id_b, "mentions", weight=0.9)
        self.rag.add_relation(id_a, id_b, relation, weight=1.0)

        return {
            "triple": (entity_a, relation, entity_b),
            "source": doc_id,
            "entity_a_id": id_a,
            "entity_b_id": id_b,
        }

    def _ensure_entity(self, name: str) -> str:
        """Создать документ-сущность, если не существует."""
        doc_id = self._entity_doc_id(name)
        existing = self.rag.get_document(doc_id, limit=1)
        if existing is not None:
            return doc_id

        self.rag.add_document(
            name,
            metadata={"type": "entity", "auto_extracted": True, "name": name},
            doc_id=doc_id,
            _skip_length_check=True,
        )
        return doc_id

    def _extract_ner(self, doc_id: str, text: str) -> list[dict]:
        """NER-based (spaCy) — только сущности, без отношений."""
        import importlib
        if importlib.util.find_spec("spacy") is None:
            return [{"error": "spaCy not installed, run: pip install spacy && python -m spacy download en_core_web_sm"}]

        import spacy
        try:
            nlp_en = spacy.load("en_core_web_sm")
        except OSError:
            return [{"error": "spaCy model 'en_core_web_sm' not downloaded. Run: python -m spacy download en_core_web_sm"}]
        doc = nlp_en(text[:10000])
        results = []
        seen = set()
        for ent in doc.ents:
            if ent.label_ not in {"PERSON", "ORG", "PRODUCT", "GPE"}:
                continue
            name = ent.text.strip()
            if not name or name.lower() in seen:
                continue
            seen.add(name.lower())
            eid = self._ensure_entity(name)
            self.rag.add_relation(doc_id, eid, "mentions", weight=0.8)
            results.append({"entity": name, "label": ent.label_, "doc_id": eid})
        return results
