"""Тесты провайдеров эмбеддингов: sentence_transformer / Ollama / OpenAI-compatible.

Сеть и модели не используются — httpx.Client и SentenceTransformer мокаются.
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.embeddings import (
    OllamaEmbeddingGenerator,
    OpenAICompatibleEmbeddingGenerator,
    SentenceTransformerEmbeddingGenerator,
)


def _mock_client(json_payload):
    client = MagicMock()
    response = MagicMock()
    response.json.return_value = json_payload
    client.post.return_value = response
    return client


class TestSentenceTransformerGenerator:
    def _generator(self, **kwargs):
        return SentenceTransformerEmbeddingGenerator(model_name="mock", **kwargs)

    def _patch_model(self, vectors):
        model = MagicMock()
        model.encode.return_value = np.asarray(vectors, dtype=np.float32)
        return model

    def test_lazy_load_happens_once(self):
        gen = self._generator()
        model = self._patch_model([[0.1, 0.2]])
        with patch("sentence_transformers.SentenceTransformer", return_value=model) as ctor:
            gen.get_embedding("hello")
            gen.get_embedding("world")
            assert gen._ensure_model() is model
        assert ctor.call_count == 1

    def test_constructor_kwargs_passed_to_model(self):
        gen = self._generator(device="cuda", local_files_only=True, token="hf_x")
        model = self._patch_model([[0.1]])
        with patch("sentence_transformers.SentenceTransformer", return_value=model) as ctor:
            gen.get_embeddings(["a"])
        assert ctor.call_args[0][0] == "mock"
        assert ctor.call_args[1] == {"device": "cuda", "local_files_only": True, "token": "hf_x"}

    def test_token_omitted_when_empty(self):
        gen = self._generator(token="")
        model = self._patch_model([[0.1]])
        with patch("sentence_transformers.SentenceTransformer", return_value=model) as ctor:
            gen.get_embeddings(["a"])
        assert "token" not in ctor.call_args[1]

    def test_cache_avoids_recomputation_and_preserves_order(self):
        gen = self._generator()
        model = self._patch_model([[0.1, 0.2], [0.3, 0.4]])
        with patch("sentence_transformers.SentenceTransformer", return_value=model):
            first = gen.get_embeddings(["a", "b"])
            model.encode.return_value = np.asarray([[0.5, 0.6]], dtype=np.float32)
            second = gen.get_embeddings(["b", "a", "c"])

        assert first == [[pytest.approx(0.1), pytest.approx(0.2)], [pytest.approx(0.3), pytest.approx(0.4)]]
        assert second[0] == first[1] and second[1] == first[0]
        assert model.encode.call_args[0][0] == ["c"]

    def test_get_embedding_uses_cache_without_model(self):
        gen = self._generator()
        gen._cache["cached"] = [1.0, 2.0]
        assert gen.get_embedding("cached") == [1.0, 2.0]

    def test_get_dimension_defaults_then_follows_cache(self):
        gen = self._generator(dimension=1024)
        assert gen.get_dimension() == 1024
        gen._cache["a"] = [0.0] * 7
        assert gen.get_dimension() == 7

    def test_clear_cache(self):
        gen = self._generator(dimension=3)
        gen._cache["a"] = [0.0, 0.0]
        gen.clear_cache()
        assert gen._cache == {}
        assert gen.get_dimension() == 3


class TestOllamaGenerator:
    def test_posts_to_embed_endpoint(self):
        with patch("httpx.Client", return_value=_mock_client({"embeddings": [[0.1, 0.2]]})) as ctor:
            gen = OllamaEmbeddingGenerator(base_url="http://host:11434/", model="m")
            assert gen.get_embedding("hello") == [0.1, 0.2]

        client = ctor.return_value
        assert client.post.call_args[0][0] == "http://host:11434/api/embed"
        assert client.post.call_args[1]["json"] == {"model": "m", "input": ["hello"]}

    def test_duplicates_deduplicated_in_request(self):
        with patch("httpx.Client", return_value=_mock_client({"embeddings": [[0.1], [0.2]]})) as ctor:
            gen = OllamaEmbeddingGenerator()
            result = gen.get_embeddings(["a", "b", "a"])

        assert ctor.return_value.post.call_args[1]["json"]["input"] == ["a", "b"]
        assert result == [[0.1], [0.2], [0.1]]

    def test_cached_texts_are_not_requested_again(self):
        with patch("httpx.Client", return_value=_mock_client({"embeddings": [[0.1]]})) as ctor:
            gen = OllamaEmbeddingGenerator()
            gen.get_embeddings(["a"])
            gen.get_embeddings(["a"])
        assert ctor.return_value.post.call_count == 1

    def test_count_mismatch_raises(self):
        with patch("httpx.Client", return_value=_mock_client({"embeddings": [[0.1]]})):
            gen = OllamaEmbeddingGenerator()
            with pytest.raises(ValueError, match="1 embeddings for 2 inputs"):
                gen.get_embeddings(["a", "b"])

    def test_http_error_propagates(self):
        client = _mock_client({})
        client.post.return_value.raise_for_status.side_effect = RuntimeError("500")
        with patch("httpx.Client", return_value=client):
            with pytest.raises(RuntimeError, match="500"):
                OllamaEmbeddingGenerator().get_embeddings(["a"])

    def test_get_embedding_uses_cache(self):
        with patch("httpx.Client", return_value=_mock_client({"embeddings": [[0.1]]})) as ctor:
            gen = OllamaEmbeddingGenerator()
            assert gen.get_embedding("a") == [0.1]
            assert gen.get_embedding("a") == [0.1]
        assert ctor.return_value.post.call_count == 1

    def test_dimension_and_clear_cache(self):
        with patch("httpx.Client", return_value=_mock_client({"embeddings": [[0.1, 0.2, 0.3]]})):
            gen = OllamaEmbeddingGenerator(dimension=4096)
            assert gen.get_dimension() == 4096
            gen.get_embeddings(["a"])
            assert gen.get_dimension() == 3
            gen.clear_cache()
            assert gen.get_dimension() == 4096


class TestOpenAICompatibleGenerator:
    def _payload(self, *vectors):
        return {"data": [{"index": i, "embedding": list(v)} for i, v in enumerate(vectors)]}

    def test_single_text_sent_as_string(self):
        with patch("httpx.Client", return_value=_mock_client(self._payload([0.1, 0.2]))) as ctor:
            gen = OpenAICompatibleEmbeddingGenerator(base_url="http://host:8080/v1/", model="m")
            assert gen.get_embedding("hello") == [0.1, 0.2]

        call = ctor.return_value.post.call_args
        assert call[0][0] == "http://host:8080/v1/embeddings"
        assert call[1]["json"] == {"model": "m", "input": "hello"}
        assert "Authorization" not in call[1]["headers"]

    def test_batch_sent_as_list_with_auth_header(self):
        with patch("httpx.Client", return_value=_mock_client(self._payload([0.1], [0.2]))) as ctor:
            gen = OpenAICompatibleEmbeddingGenerator(api_key="sk-test")
            assert gen.get_embeddings(["a", "b"]) == [[0.1], [0.2]]

        call = ctor.return_value.post.call_args
        assert call[1]["json"]["input"] == ["a", "b"]
        assert call[1]["headers"]["Authorization"] == "Bearer sk-test"

    def test_out_of_order_response_sorted_by_index(self):
        payload = {"data": [{"index": 1, "embedding": [0.2]}, {"index": 0, "embedding": [0.1]}]}
        with patch("httpx.Client", return_value=_mock_client(payload)):
            gen = OpenAICompatibleEmbeddingGenerator()
            assert gen.get_embeddings(["a", "b"]) == [[0.1], [0.2]]

    def test_duplicates_deduplicated(self):
        with patch("httpx.Client", return_value=_mock_client(self._payload([0.1], [0.2]))) as ctor:
            gen = OpenAICompatibleEmbeddingGenerator()
            assert gen.get_embeddings(["a", "b", "a"]) == [[0.1], [0.2], [0.1]]
        assert ctor.return_value.post.call_args[1]["json"]["input"] == ["a", "b"]

    def test_count_mismatch_raises(self):
        with patch("httpx.Client", return_value=_mock_client(self._payload([0.1]))):
            gen = OpenAICompatibleEmbeddingGenerator()
            with pytest.raises(ValueError, match="1 embeddings for 2 inputs"):
                gen.get_embeddings(["a", "b"])

    def test_get_embedding_uses_cache(self):
        with patch("httpx.Client", return_value=_mock_client(self._payload([0.1]))) as ctor:
            gen = OpenAICompatibleEmbeddingGenerator()
            gen.get_embedding("a")
            gen.get_embedding("a")
        assert ctor.return_value.post.call_count == 1

    def test_dimension_clear_cache_and_close(self):
        with patch("httpx.Client", return_value=_mock_client(self._payload([0.1, 0.2]))) as ctor:
            gen = OpenAICompatibleEmbeddingGenerator(dimension=1024)
            assert gen.get_dimension() == 1024
            gen.get_embeddings(["a"])
            assert gen.get_dimension() == 2
            gen.clear_cache()
            assert gen.get_dimension() == 1024
            gen.close()
        ctor.return_value.close.assert_called_once()
