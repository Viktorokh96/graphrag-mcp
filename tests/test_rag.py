"""Тесты для модуля RAGSystem (оркестратор RAG) на новом стеке Qdrant + SQLite.

Используются фикстуры conftest: rag (HashEmbeddingGenerator), semantic_rag
(SemanticMockEmbeddingGenerator), make_rag (фабрика). Все они делают rag.close()
в teardown — это критично для файловых локов Qdrant/SQLite на Windows.
"""

import pytest

from src.config import RAGConfig
from src.embeddings import OllamaEmbeddingGenerator, OpenRouterEmbeddingGenerator
from src.rag import RAGSystem
from tests.conftest import HashEmbeddingGenerator


@pytest.fixture
def make_raw_rag(tmp_path):
    """Фабрика RAGSystem без mock-эмбеддера — для тестов выбора провайдера.

    Провайдеры Ollama/OpenRouter не делают сетевых вызовов при init
    (эмбеддинги нужны только при add/search), поэтому реальная модель
    не загружается. Teardown закрывает все созданные системы.
    """
    created: list[RAGSystem] = []

    def _make(**kwargs):
        rag = RAGSystem(**kwargs)
        created.append(rag)
        return rag

    yield _make
    for rag in created:
        try:
            rag.close()
        except Exception:
            pass


class TestSearch:
    """Индексация и поиск."""

    def test_ingest_and_query_basic(self, semantic_rag):
        """После индексации документов запрос должен возвращать результаты."""
        texts = [
            "Python is a programming language for general purpose",
            "Java runs on a virtual machine and is statically typed",
            "Python is great for machine learning and data science",
        ]
        doc_ids = semantic_rag.add_documents(texts)
        results = semantic_rag.search("Python programming", k=2)
        assert len(results) == 2
        result_doc_ids = [r[0] for r in results]
        assert len(set(result_doc_ids) & set(doc_ids)) == 2

    def test_ingest_empty(self, rag):
        """add_documents с пустым списком не должен вызывать ошибок."""
        rag.add_documents([])
        results = rag.search("test", k=5)
        assert results == []

    def test_query_before_ingest(self, rag):
        """Запрос до индексации должен возвращать пустой список."""
        results = rag.search("anything")
        assert results == []

    def test_query_returns_sorted_by_relevance(self, semantic_rag):
        """Результаты запроса должны быть отсортированы по релевантности."""
        texts = [
            "machine learning deep learning neural networks artificial intelligence",
            "deep learning concepts overview and fundamental principles explained",
            "cooking recipes for pasta with tomato sauce and fresh vegetables",
        ]
        semantic_rag.add_documents(texts)
        results = semantic_rag.search("machine learning deep learning", k=3)
        assert len(results) >= 2
        scores = [r[2] for r in results]
        for i in range(len(scores) - 1):
            assert scores[i] >= scores[i + 1], "Результаты должны быть отсортированы по релевантности"

    def test_query_with_metadata(self, rag):
        """Запрос должен возвращать метаданные, если указаны при добавлении."""
        texts = [
            "artificial intelligence and deep learning concepts overview",
            "machine learning with neural networks and data analysis",
        ]
        metadata = [{"category": "AI"}, {"category": "ML"}]
        rag.add_documents(texts, metadata)
        results = rag.search("AI", k=2)
        assert len(results) == 2
        for doc_id, text, score, meta in results:
            assert "category" in meta, f"У {doc_id} нет метаданных"

    def test_reingest_updates_index(self, semantic_rag):
        """После удаления и добавления нового документа индекс актуален."""
        doc_id_1 = semantic_rag.add_document(
            "Python programming language for scripting and automation"
        )
        assert semantic_rag.delete_document(doc_id_1) is True

        doc_id_2 = semantic_rag.add_document(
            "Java programming language for enterprise applications"
        )
        results = semantic_rag.search("Java", k=1)
        assert results[0][0] == doc_id_2
        # Python больше не должен быть релевантнее Java
        java_score = semantic_rag.search("Java", k=1)[0][2]
        python_results = semantic_rag.search("Python", k=1)
        python_score = python_results[0][2] if python_results else 0.0
        assert java_score >= python_score, (
            f"Java ({java_score:.3f}) должна быть >= Python ({python_score:.3f})"
        )

    def test_large_top_k(self, rag):
        """k больше числа документов не должно вызывать ошибок."""
        texts = [f"sample text document number {i} for testing and analysis" for i in range(5)]
        rag.add_documents(texts)
        results = rag.search("text", k=100)
        assert len(results) <= 5

    def test_full_pipeline_integration(self, rag):
        """Полный pipeline: индексация → поиск → форматирование ответа."""
        texts = [
            "To reset your password go to settings page and follow the instructions",
            "Password must be at least 8 characters long for security reasons",
        ]
        doc_ids = rag.add_documents(texts)
        results = rag.search("How to reset password?", k=2)
        assert len(results) == 2
        context = "\n\n".join([f"[{doc_id}] {text}" for doc_id, text, _, _ in results])
        assert doc_ids[0] in context or doc_ids[1] in context
        assert "password" in context.lower()


class TestGetDocument:
    """get_document: пагинация текста."""

    def test_get_document_full_text(self, rag):
        """get_document возвращает полный текст по умолчанию."""
        full_text = "0123456789" * 100
        doc_id = rag.add_document(full_text)
        result = rag.get_document(doc_id)
        assert result["doc_id"] == doc_id
        assert result["text"] == full_text
        assert result["total_chars"] == len(full_text)
        assert result["offset"] == 0
        assert result["limit"] is None

    def test_get_document_with_limit(self, rag):
        """get_document с limit обрезает текст с начала."""
        full_text = "0123456789" * 100
        doc_id = rag.add_document(full_text)
        result = rag.get_document(doc_id, limit=100)
        assert result["text"] == full_text[:100]
        assert result["total_chars"] == 1000
        assert result["offset"] == 0
        assert result["limit"] == 100

    def test_get_document_with_offset(self, rag):
        """get_document с offset начинает чтение с середины."""
        full_text = "0123456789" * 100
        doc_id = rag.add_document(full_text)
        result = rag.get_document(doc_id, offset=500)
        assert result["text"] == full_text[500:]
        assert result["offset"] == 500
        assert result["limit"] is None

    def test_get_document_with_offset_and_limit(self, rag):
        """get_document с offset+limit читает страницу из середины."""
        full_text = "0123456789" * 100
        doc_id = rag.add_document(full_text)
        result = rag.get_document(doc_id, offset=200, limit=100)
        assert result["text"] == full_text[200:300]
        assert result["total_chars"] == 1000
        assert result["offset"] == 200
        assert result["limit"] == 100

    def test_get_document_nonexistent_returns_none(self, rag):
        """get_document для несуществующего ID возвращает None."""
        assert rag.get_document("nonexistent-uuid") is None

    def test_get_document_pagination_covers_whole_text(self, rag):
        """Постраничное чтение через offset+limit собирает весь текст."""
        full_text = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789abcdefghijklmnopqrstuv"
        doc_id = rag.add_document(full_text)
        collected = ""
        offset = 0
        while offset < len(full_text):
            page = rag.get_document(doc_id, offset=offset, limit=10)
            collected += page["text"]
            offset += 10
        assert collected == full_text


class TestListDocuments:
    """list_documents: усечение текста."""

    def test_list_documents_max_chars_truncates(self, rag):
        """list_documents с max_chars обрезает текст каждого документа."""
        rag.add_document("A" * 2000)
        rag.add_document("B" * 2000)
        result = rag.list_documents(limit=2, max_chars=100)
        for doc in result["documents"]:
            assert len(doc["text"]) == 100

    def test_list_documents_full_text_by_default(self, rag):
        """list_documents без max_chars отдаёт полный текст."""
        long_text = "A" * 2000
        rag.add_document(long_text)
        result = rag.list_documents(limit=1)
        assert result["documents"][0]["text"] == long_text


class TestReindex:
    """reindex(): пересчёт эмбеддингов."""

    def test_reindex_updates_embeddings(self, rag):
        rag.add_document("Python programming for scripting and automation tasks")
        rag.add_document("Java programming for enterprise software development")
        count = rag.reindex()
        assert count == 2
        results = rag.search("Python", k=1)
        assert len(results) == 1

    def test_reindex_empty_returns_zero(self, rag):
        count = rag.reindex()
        assert count == 0

    def test_reindex_with_dimension_change(self, rag):
        """Reindex должен пересоздать коллекцию при смене размерности."""
        rag.add_document("Python programming for scripting and automation tasks")
        rag.add_document("Java programming for enterprise software development")
        assert rag.stats()["dimension"] == 64  # HashEmbeddingGenerator по умолчанию

        # Меняем генератор на 128-dim и переиндексируем
        rag.embedding_generator = HashEmbeddingGenerator(dimension=128)
        count = rag.reindex()
        assert count == 2
        assert rag.stats()["dimension"] == 128

        # Поиск должен работать после reindex
        results = rag.search("Python", k=1)
        assert len(results) == 1


class TestProviderSelection:
    """Выбор провайдера эмбеддингов (без загрузки реальных моделей)."""

    def test_default_provider_is_bge_m3(self):
        """Дефолтный провайдер конфигурации — bge-m3 (локальная модель)."""
        assert RAGConfig().embedding_provider == "bge-m3"

    def test_provider_selection_openrouter_with_api_key(self, tmp_path, make_raw_rag):
        """При передаче api_key — принудительно OpenRouter."""
        rag = make_raw_rag(store_path=str(tmp_path / "s1"), api_key="test-key", config=RAGConfig())
        assert isinstance(rag.embedding_generator, OpenRouterEmbeddingGenerator)

    def test_provider_selection_openrouter_from_config(self, tmp_path, make_raw_rag):
        """При embedding_provider=openrouter + api_key в конфиге — OpenRouter."""
        cfg = RAGConfig(embedding_provider="openrouter", openrouter_api_key="cfg-key")
        rag = make_raw_rag(store_path=str(tmp_path / "s2"), config=cfg)
        assert isinstance(rag.embedding_generator, OpenRouterEmbeddingGenerator)

    def test_provider_selection_ollama_from_config(self, tmp_path, make_raw_rag):
        """При embedding_provider=ollama — Ollama, даже если openrouter_api_key задан."""
        cfg = RAGConfig(embedding_provider="ollama", openrouter_api_key="some-key")
        rag = make_raw_rag(store_path=str(tmp_path / "s3"), config=cfg)
        assert isinstance(rag.embedding_generator, OllamaEmbeddingGenerator)

    def test_api_key_overrides_config_provider(self, tmp_path, make_raw_rag):
        """api_key приоритетнее config.embedding_provider."""
        cfg = RAGConfig(embedding_provider="ollama")
        rag = make_raw_rag(store_path=str(tmp_path / "s4"), api_key="force-openrouter", config=cfg)
        assert isinstance(rag.embedding_generator, OpenRouterEmbeddingGenerator)

    def test_embedding_generator_overrides_everything(self, tmp_path, make_raw_rag):
        """Явный embedding_generator имеет приоритет над api_key и config."""
        embedder = HashEmbeddingGenerator()
        cfg = RAGConfig(embedding_provider="ollama")
        rag = make_raw_rag(
            store_path=str(tmp_path / "s5"), api_key="test-key",
            config=cfg, embedding_generator=embedder,
        )
        assert rag.embedding_generator is embedder


class TestStorePath:
    """Разрешение store_path."""

    def test_store_path_from_config(self, tmp_path, make_raw_rag):
        """store_path берётся из config, если не передан явно."""
        custom_path = str(tmp_path / "custom_store")
        cfg = RAGConfig(store_path=custom_path)
        rag = make_raw_rag(config=cfg, embedding_generator=HashEmbeddingGenerator())
        assert rag.store_path == custom_path

    def test_store_path_explicit_overrides_config(self, tmp_path, make_raw_rag):
        """Явный store_path приоритетнее config.store_path."""
        explicit = str(tmp_path / "explicit_store")
        cfg = RAGConfig(store_path=str(tmp_path / "from_config"))
        rag = make_raw_rag(
            store_path=explicit, config=cfg, embedding_generator=HashEmbeddingGenerator()
        )
        assert rag.store_path == explicit


class TestEnrichWithLinks:
    """_enrich_with_links и загрузка связей."""

    def test_enrich_with_links_depth_zero(self, rag):
        """_enrich_with_links при depth=0 добавляет пустой links."""
        doc_id_a = rag.add_document("AAA test document for enrichment link testing purposes example text")
        doc_id_b = rag.add_document("BBB test document for enrichment link testing purposes example text")
        rag.add_relation(doc_id_a, doc_id_b, "related_to", 1.0)

        docs = [{"doc_id": doc_id_a, "text": "aaa"}]
        rag._enrich_with_links(docs, relations_load_depth=0)
        assert docs[0].get("links") == {}

    def test_enrich_with_links_depth_one(self, rag):
        """_enrich_with_links при depth=1 добавляет соседние узлы."""
        doc_id_a = rag.add_document("AAA test document for enrichment link testing purposes example text")
        doc_id_b = rag.add_document("BBB test document for enrichment link testing purposes example text")
        rag.add_relation(doc_id_a, doc_id_b, "related_to", 1.0)

        docs = [{"doc_id": doc_id_a, "text": "aaa"}]
        rag._enrich_with_links(docs, relations_load_depth=1)
        links = docs[0].get("links", {})
        assert doc_id_b in links
        assert len(links[doc_id_b]) == 1
        assert links[doc_id_b][0]["relation"] == "related_to"
        assert links[doc_id_b][0]["weight"] == 1.0
        assert links[doc_id_b][0]["direction"] == "out"

    def test_enrich_with_links_type_filter(self, rag):
        """_enrich_with_links фильтрует по типу связи."""
        doc_id_a = rag.add_document("AAA test document for enrichment link testing purposes example text")
        doc_id_b = rag.add_document("BBB test document for enrichment link testing purposes example text")
        doc_id_c = rag.add_document("CCC test document for enrichment link testing purposes example text")
        rag.add_relation(doc_id_a, doc_id_b, "related_to", 1.0)
        rag.add_relation(doc_id_a, doc_id_c, "similar_to", 0.8)

        docs = [{"doc_id": doc_id_a, "text": "aaa"}]
        rag._enrich_with_links(docs, relations_load_depth=1,
                               relations_load_type_filter=["related_to"])
        links = docs[0].get("links", {})
        assert doc_id_b in links
        assert doc_id_c not in links

    def test_enrich_with_links_no_relations(self, rag):
        """_enrich_with_links для узла без связей."""
        doc_id = rag.add_document("AAA test document for enrichment link testing purposes example text")
        docs = [{"doc_id": doc_id, "text": "aaa"}]
        rag._enrich_with_links(docs, relations_load_depth=1)
        assert docs[0].get("links") == {}

    def test_enrich_with_links_multiple_docs(self, rag):
        """_enrich_with_links обогащает несколько документов."""
        doc_a = rag.add_document("AAA test document for enrichment link testing purposes example text")
        doc_b = rag.add_document("BBB test document for enrichment link testing purposes example text")
        doc_c = rag.add_document("CCC test document for enrichment link testing purposes example text")
        rag.add_relation(doc_a, doc_b, "related_to", 1.0)
        rag.add_relation(doc_a, doc_c, "similar_to", 0.8)

        docs = [{"doc_id": doc_a}, {"doc_id": doc_b}]
        rag._enrich_with_links(docs, relations_load_depth=1)
        # doc_a связан с b и c
        assert len(docs[0]["links"]) == 2
        assert doc_b in docs[0]["links"]
        assert doc_c in docs[0]["links"]
        # doc_b имеет входящее ребро от doc_a (direction="in")
        assert doc_a in docs[1]["links"]
        assert docs[1]["links"][doc_a][0]["direction"] == "in"

    def test_get_document_with_relations(self, rag):
        """get_document с relations_load_depth возвращает links."""
        doc_a = rag.add_document("AAA test document for enrichment link testing purposes example text")
        doc_b = rag.add_document("BBB test document for enrichment link testing purposes example text")
        rag.add_relation(doc_a, doc_b, "related_to", 1.0)

        result = rag.get_document(doc_a, relations_load_depth=1)
        assert result is not None
        assert "links" in result
        assert doc_b in result["links"]

    def test_get_document_with_relations_depth_zero(self, rag):
        """get_document с relations_load_depth=0 не загружает связи."""
        doc_a = rag.add_document("AAA test document for enrichment link testing purposes example text")
        doc_b = rag.add_document("BBB test document for enrichment link testing purposes example text")
        rag.add_relation(doc_a, doc_b, "related_to", 1.0)

        result = rag.get_document(doc_a, relations_load_depth=0)
        assert result is not None
        assert result["links"] == {}

    def test_list_documents_with_relations(self, rag):
        """list_documents с relations_load_depth возвращает links."""
        doc_a = rag.add_document("AAA test document for enrichment link testing purposes example text")
        doc_b = rag.add_document("BBB test document for enrichment link testing purposes example text")
        rag.add_relation(doc_a, doc_b, "related_to", 1.0)

        result = rag.list_documents(limit=10, relations_load_depth=1)
        for doc in result["documents"]:
            assert "links" in doc
