"""Тесты для модуля RAGSystem (оркестратор RAG)"""

import pytest
import os
import shutil
import tempfile
from unittest.mock import patch, MagicMock
from src.rag import RAGSystem
from src.config import RAGConfig
from src.embeddings import OpenRouterEmbeddingGenerator, OllamaEmbeddingGenerator


class TestRAGSystem:
    """Класс тестов для RAG-системы."""

    @pytest.fixture(autouse=True)
    def setup_temp_dir(self):
        """Фикстура: временная директория для каждого теста."""
        self.temp_dir = tempfile.mkdtemp()
        self.store_path = os.path.join(self.temp_dir, "rag_data")
        yield
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    @pytest.fixture(autouse=True)
    def mock_httpx(self):
        """Мокаем httpx.Client для всех тестов."""
        with patch("src.embeddings.httpx.Client") as mock_client_class:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "data": [{"embedding": [0.1] * 128}]
            }
            mock_client = MagicMock()
            mock_client.__enter__.return_value = mock_client
            mock_client.post.return_value = mock_response
            mock_client_class.return_value = mock_client
            yield

    @pytest.fixture
    def rag(self):
        """Фикстура: пустая RAG-система с временным хранилищем."""
        return RAGSystem(store_path=self.store_path, api_key="test-key")

    def test_ingest_and_query_basic(self, rag):
        """После индексации документов запрос должен возвращать результаты."""
        # Используем новый API: add_documents вместо ingest
        texts = [
            "Python is a programming language for general purpose",
            "Java runs on a virtual machine and is statically typed",
            "Python is great for machine learning and data science",
        ]
        doc_ids = rag.add_documents(texts)
        results = rag.search("Python programming", k=2)
        assert len(results) == 2
        # Должен вернуть 2 результата из 3 добавленных документов
        result_doc_ids = [r[0] for r in results]
        assert len(set(result_doc_ids) & set(doc_ids)) == 2

    def test_ingest_empty(self, rag):
        """Ingest с пустым списком не должен вызывать ошибок."""
        rag.add_documents([])
        results = rag.search("test", k=5)
        assert results == []

    def test_query_before_ingest(self, rag):
        """Запрос до индексации должен возвращать пустой список."""
        results = rag.search("anything")
        assert results == []

    def test_query_returns_sorted_by_relevance(self, rag):
        """Результаты запроса должны быть отсортированы по релевантности."""
        texts = [
            "machine learning deep learning neural networks artificial intelligence",
            "deep learning concepts overview and fundamental principles explained",
            "cooking recipes for pasta with tomato sauce and fresh vegetables",
        ]
        rag.add_documents(texts)
        results = rag.search("machine learning deep learning", k=3)
        scores = [r[2] for r in results]
        for i in range(len(scores) - 1):
            assert scores[i] >= scores[i + 1], "Результаты должны быть отсортированы по релевантности"

    def test_query_with_metadata(self, rag):
        """Запрос должен возвращать метаданные if указаны при добавлении."""
        texts = ["artificial intelligence and deep learning concepts overview", "machine learning with neural networks and data analysis"]
        metadata = [{"category": "AI"}, {"category": "ML"}]
        rag.add_documents(texts, metadata)
        results = rag.search("AI", k=2)
        # Проверяем, что метаданные сохранились
        for doc_id, text, score, meta in results:
            assert "category" in meta, f"У {doc_id} нет метаданных"

    def test_reingest_updates_index(self, rag):
        """Повторное добавление должно обновлять существующие документы."""
        doc_id_1 = rag.add_document("Python programming language for scripting and automation")
        # Удаляем старый и добавляем новый с тем же текстом
        rag.vector_store.remove(doc_id_1)
        rag.bm25_index.remove(doc_id_1)
        rag.graph_kb.remove_node(doc_id_1)
        doc_id_2 = rag.add_document("Java programming language for enterprise applications")
        results = rag.search("Java", k=1)
        assert results[0][0] == doc_id_2
        # Python больше не должен быть ассоциирован с doc_id_2
        results_java = rag.search("Java", k=1)
        java_score = results_java[0][2]
        results_python = rag.search("Python", k=1)
        python_score = results_python[0][2]
        assert java_score >= python_score, (
            f"Java ({java_score:.3f}) должна быть >= Python ({python_score:.3f})"
        )

    def test_large_top_k(self, rag):
        """top_k больше числа документов не должно вызывать ошибок."""
        texts = [f"sample text document number {i} for testing and analysis" for i in range(5)]
        rag.add_documents(texts)
        results = rag.search("text", k=100)
        assert len(results) <= 5  # Не больше, чем есть документов

    def test_full_pipeline_integration(self, rag):
        """Полный pipeline: индексация → поиск → форматирование ответа."""
        texts = [
            "To reset your password go to settings page and follow the instructions",
            "Password must be at least 8 characters long for security reasons",
        ]
        doc_ids = rag.add_documents(texts)
        results = rag.search("How to reset password?", k=2)
        assert len(results) == 2
        # Форматируем как контекст для LLM
        context = "\n\n".join([f"[{doc_id}] {text}" for doc_id, text, _, _ in results])
        assert doc_ids[0] in context or doc_ids[1] in context
        assert "password" in context.lower()

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
        # Добавляем с текущим моком (128-dim от OpenRouter)
        rag.add_document("Python programming for scripting and automation tasks")
        rag.add_document("Java programming for enterprise software development")
        assert rag.stats()["dimension"] == 128

        # Меняем мок на 256-dim
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "data": [{"embedding": [0.1] * 256}]
        }
        mock_client = MagicMock()
        mock_client.__enter__.return_value = mock_client
        mock_client.post.return_value = mock_response
        with patch("src.embeddings.httpx.Client", return_value=mock_client):
            count = rag.reindex()
            assert count == 2
            assert rag.stats()["dimension"] == 256

            # Поиск должен работать после reindex (под тем же моком)
            results = rag.search("Python", k=1)
            assert len(results) == 1

    def test_provider_selection_ollama_by_default(self):
        """Без api_key и без EMBEDDING_PROVIDER — Ollama."""
        with patch.dict(os.environ, {}, clear=False):
            for key in ["OPENROUTER_API_KEY", "EMBEDDING_PROVIDER"]:
                os.environ.pop(key, None)
            rag = RAGSystem(store_path=self.store_path)
            assert isinstance(rag.embedding_generator, OllamaEmbeddingGenerator)

    def test_provider_selection_openrouter_with_api_key(self):
        """При передаче api_key — принудительно OpenRouter."""
        rag = RAGSystem(store_path=self.store_path, api_key="test-key")
        assert isinstance(rag.embedding_generator, OpenRouterEmbeddingGenerator)

    def test_provider_selection_openrouter_from_config(self):
        """При EMBEDDING_PROVIDER=openrouter + api_key в конфиге — OpenRouter."""
        cfg = RAGConfig(
            embedding_provider="openrouter",
            openrouter_api_key="cfg-key",
        )
        rag = RAGSystem(store_path=self.store_path, config=cfg)
        assert isinstance(rag.embedding_generator, OpenRouterEmbeddingGenerator)

    def test_provider_selection_ollama_from_config(self):
        """При EMBEDDING_PROVIDER=ollama — Ollama, даже если openrouter_api_key задан."""
        cfg = RAGConfig(
            embedding_provider="ollama",
            openrouter_api_key="some-key",
        )
        rag = RAGSystem(store_path=self.store_path, config=cfg)
        assert isinstance(rag.embedding_generator, OllamaEmbeddingGenerator)

    def test_api_key_overrides_config_provider(self):
        """api_key приоритетнее config.embedding_provider."""
        cfg = RAGConfig(embedding_provider="ollama")
        rag = RAGSystem(store_path=self.store_path, api_key="force-openrouter", config=cfg)
        assert isinstance(rag.embedding_generator, OpenRouterEmbeddingGenerator)

    def test_store_path_from_config(self):
        """store_path берётся из config если не передан явно."""
        custom_path = os.path.join(self.temp_dir, "custom_store")
        cfg = RAGConfig(store_path=custom_path)
        rag = RAGSystem(config=cfg)
        assert rag.store_path == custom_path

    def test_store_path_explicit_overrides_config(self):
        """Явный store_path приоритетнее config.store_path."""
        cfg = RAGConfig(store_path="/from/config")
        rag = RAGSystem(store_path=self.store_path, config=cfg)
        assert rag.store_path == self.store_path