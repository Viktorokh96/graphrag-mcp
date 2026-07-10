"""Тесты для OpenRouterEmbeddingGenerator и OllamaEmbeddingGenerator."""

from unittest.mock import patch, MagicMock


class TestEmbeddingGenerator:
    """Тесты для генератора эмбеддингов через OpenRouter."""

    def test_import(self):
        """Модуль импортируется без ошибок."""
        from src.embeddings import OpenRouterEmbeddingGenerator
        assert OpenRouterEmbeddingGenerator is not None

    def test_init_with_defaults(self):
        """Инициализация с параметрами по умолчанию."""
        from src.embeddings import OpenRouterEmbeddingGenerator

        gen = OpenRouterEmbeddingGenerator()
        assert gen.model == "openai/text-embedding-3-small"
        assert gen.api_key is None or len(gen.api_key) > 0

    def test_init_with_custom_model(self):
        """Инициализация с кастомной моделью."""
        from src.embeddings import OpenRouterEmbeddingGenerator

        gen = OpenRouterEmbeddingGenerator(model="custom/model")
        assert gen.model == "custom/model"

    @patch("src.embeddings.httpx.Client")
    def test_get_embedding(self, mock_httpx):
        """get_embedding возвращает список float."""
        from src.embeddings import OpenRouterEmbeddingGenerator

        # Мокаем ответ API
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "data": [{"embedding": [0.1, 0.2, 0.3]}]
        }
        mock_client = MagicMock()
        mock_client.__enter__.return_value = mock_client
        mock_client.post.return_value = mock_response
        mock_httpx.return_value = mock_client

        gen = OpenRouterEmbeddingGenerator(api_key="test-key")
        result = gen.get_embedding("hello world")

        assert isinstance(result, list)
        assert len(result) == 3
        assert result == [0.1, 0.2, 0.3]

    @patch("src.embeddings.httpx.Client")
    def test_get_embeddings_batch(self, mock_httpx):
        """get_embeddings обрабатывает несколько текстов."""
        from src.embeddings import OpenRouterEmbeddingGenerator

        def mock_post_side_effect(url, *args, **kwargs):
            mock_resp = MagicMock()
            mock_resp.status_code = 200
            data = kwargs.get("json", {})
            text = data.get("input", "")
            # Возвращаем разные эмбеддинги для разных текстов
            if text == "hello":
                mock_resp.json.return_value = {"data": [{"embedding": [0.1, 0.2, 0.3]}]}
            elif text == "world":
                mock_resp.json.return_value = {"data": [{"embedding": [0.4, 0.5, 0.6]}]}
            else:
                mock_resp.json.return_value = {"data": [{"embedding": [0.0, 0.0, 0.0]}]}
            return mock_resp

        mock_client = MagicMock()
        mock_client.__enter__.return_value = mock_client
        mock_client.post.side_effect = mock_post_side_effect
        mock_httpx.return_value = mock_client

        gen = OpenRouterEmbeddingGenerator(api_key="test-key")
        results = gen.get_embeddings(["hello", "world"])

        assert len(results) == 2
        assert results[0] == [0.1, 0.2, 0.3]
        assert results[1] == [0.4, 0.5, 0.6]

    @patch("src.embeddings.httpx.Client")
    def test_caching(self, mock_httpx):
        """Повторный запрос того же текста использует кеш."""
        from src.embeddings import OpenRouterEmbeddingGenerator

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "data": [{"embedding": [0.1, 0.2, 0.3]}]
        }
        mock_client = MagicMock()
        mock_client.__enter__.return_value = mock_client
        mock_client.post.return_value = mock_response
        mock_httpx.return_value = mock_client

        gen = OpenRouterEmbeddingGenerator(api_key="test-key")

        # Первый вызов — идёт в API
        result1 = gen.get_embedding("hello")
        assert mock_client.post.call_count == 1

        # Второй вызов того же текста — из кеша
        result2 = gen.get_embedding("hello")
        assert mock_client.post.call_count == 1  # Не увеличился!

        assert result1 == result2

    @patch("src.embeddings.httpx.Client")
    def test_api_error_fallback(self, mock_httpx):
        """При ошибке API использует fallback (TF-IDF заглушка)."""
        from src.embeddings import OpenRouterEmbeddingGenerator
        import httpx as real_httpx

        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.raise_for_status.side_effect = real_httpx.HTTPStatusError("Server error", request=None, response=None)
        mock_client = MagicMock()
        mock_client.__enter__.return_value = mock_client
        mock_client.post.return_value = mock_response
        mock_httpx.return_value = mock_client

        gen = OpenRouterEmbeddingGenerator(api_key="test-key")
        # Должен вернуть какой-то вектор даже при ошибке API
        result = gen.get_embedding("fallback test")
        assert isinstance(result, list)
        assert len(result) > 0
        # Проверяем, что это fallback-эмбеддинг (размерность 1536 — как у text-embedding-3-small)
        assert len(result) == 1536

    def test_dimension_consistency(self):
        """Все эмбеддинги должны иметь одинаковую размерность."""
        # Тест консистентности (без API, на fallback)
        from src.embeddings import OpenRouterEmbeddingGenerator

        gen = OpenRouterEmbeddingGenerator(api_key="test-key")
        # Принудительно используем fallback
        e1 = gen._fallback_embedding("hello")
        e2 = gen._fallback_embedding("world longer text here")

        assert len(e1) == len(e2), "Размерность fallback эмбеддингов должна совпадать"

    def test_fallback_embedding_shape(self):
        """Fallback эмбеддинг имеет разумную размерность."""
        from src.embeddings import OpenRouterEmbeddingGenerator

        gen = OpenRouterEmbeddingGenerator(api_key="test-key")
        emb = gen._fallback_embedding("test text for embedding")
        assert len(emb) > 0
        assert all(isinstance(v, float) for v in emb)

    def test_get_dimension(self):
        """get_dimension возвращает число."""
        from src.embeddings import OpenRouterEmbeddingGenerator

        gen = OpenRouterEmbeddingGenerator(api_key="test-key")
        dim = gen.get_dimension()
        assert isinstance(dim, int)
        assert dim > 0

    def test_cache_clear(self):
        """Кеш должен очищаться."""
        from src.embeddings import OpenRouterEmbeddingGenerator

        gen = OpenRouterEmbeddingGenerator(api_key="test-key")
        gen._cache["test"] = [0.1, 0.2]
        gen.clear_cache()
        assert len(gen._cache) == 0


class TestOllamaEmbeddingGenerator:
    """Тесты для OllamaEmbeddingGenerator."""

    def test_import(self):
        from src.embeddings import OllamaEmbeddingGenerator
        assert OllamaEmbeddingGenerator is not None

    def test_init_with_defaults(self):
        from src.embeddings import OllamaEmbeddingGenerator

        gen = OllamaEmbeddingGenerator()
        assert gen.model == "qwen3-embedding:8b"
        assert gen.base_url == "http://localhost:11434"

    def test_init_with_custom_params(self):
        from src.embeddings import OllamaEmbeddingGenerator

        gen = OllamaEmbeddingGenerator(base_url="http://custom:8080", model="nomic-embed-text", dimension=768)
        assert gen.base_url == "http://custom:8080"
        assert gen.model == "nomic-embed-text"
        assert gen._dimension == 768

    def test_base_url_trailing_slash_stripped(self):
        from src.embeddings import OllamaEmbeddingGenerator

        gen = OllamaEmbeddingGenerator(base_url="http://localhost:11434/")
        assert gen.base_url == "http://localhost:11434"

    @patch("src.embeddings.httpx.Client")
    def test_get_embedding(self, mock_httpx):
        from src.embeddings import OllamaEmbeddingGenerator

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "embeddings": [[0.1, 0.2, 0.3]]
        }
        mock_client = MagicMock()
        mock_client.__enter__.return_value = mock_client
        mock_client.post.return_value = mock_response
        mock_httpx.return_value = mock_client

        gen = OllamaEmbeddingGenerator()
        result = gen.get_embedding("hello world")

        assert isinstance(result, list)
        assert len(result) == 3
        assert result == [0.1, 0.2, 0.3]

    @patch("src.embeddings.httpx.Client")
    def test_get_embeddings_batch(self, mock_httpx):
        from src.embeddings import OllamaEmbeddingGenerator

        vecs = {"hello": [0.1, 0.2, 0.3], "world": [0.4, 0.5, 0.6]}

        def mock_post_side_effect(url, *args, **kwargs):
            mock_resp = MagicMock()
            mock_resp.status_code = 200
            data = kwargs.get("json", {})
            # Ollama /api/embed принимает `input` как строку или список строк
            # (batch) и возвращает по одному эмбеддингу на каждый вход.
            inp = data.get("input", "")
            texts = inp if isinstance(inp, list) else [inp]
            mock_resp.json.return_value = {
                "embeddings": [vecs.get(t, [0.0, 0.0, 0.0]) for t in texts]
            }
            return mock_resp

        mock_client = MagicMock()
        mock_client.__enter__.return_value = mock_client
        mock_client.post.side_effect = mock_post_side_effect
        mock_httpx.return_value = mock_client

        gen = OllamaEmbeddingGenerator()
        results = gen.get_embeddings(["hello", "world"])

        assert len(results) == 2
        assert results[0] == [0.1, 0.2, 0.3]
        assert results[1] == [0.4, 0.5, 0.6]

    @patch("src.embeddings.httpx.Client")
    def test_caching(self, mock_httpx):
        from src.embeddings import OllamaEmbeddingGenerator

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "embeddings": [[0.1, 0.2, 0.3]]
        }
        mock_client = MagicMock()
        mock_client.__enter__.return_value = mock_client
        mock_client.post.return_value = mock_response
        mock_httpx.return_value = mock_client

        gen = OllamaEmbeddingGenerator()

        result1 = gen.get_embedding("hello")
        assert mock_client.post.call_count == 1

        result2 = gen.get_embedding("hello")
        assert mock_client.post.call_count == 1

        assert result1 == result2

    @patch("src.embeddings.httpx.Client")
    def test_api_error_fallback(self, mock_httpx):
        from src.embeddings import OllamaEmbeddingGenerator
        import httpx as real_httpx

        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.raise_for_status.side_effect = real_httpx.HTTPStatusError("Server error", request=None, response=None)
        mock_client = MagicMock()
        mock_client.__enter__.return_value = mock_client
        mock_client.post.return_value = mock_response
        mock_httpx.return_value = mock_client

        gen = OllamaEmbeddingGenerator()
        result = gen.get_embedding("fallback test")
        assert isinstance(result, list)
        assert len(result) > 0
        assert len(result) == 4096

    def test_fallback_embedding_shape(self):
        from src.embeddings import OllamaEmbeddingGenerator

        gen = OllamaEmbeddingGenerator()
        emb = gen._fallback_embedding("test text for embedding")
        assert len(emb) == 4096
        assert all(isinstance(v, float) for v in emb)

    def test_get_dimension(self):
        from src.embeddings import OllamaEmbeddingGenerator

        gen = OllamaEmbeddingGenerator()
        dim = gen.get_dimension()
        assert isinstance(dim, int)
        assert dim == 4096

    def test_cache_clear(self):
        from src.embeddings import OllamaEmbeddingGenerator

        gen = OllamaEmbeddingGenerator()
        gen._cache["test"] = [0.1, 0.2]
        gen.clear_cache()
        assert len(gen._cache) == 0