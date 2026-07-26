"""Тесты для Reranker (CrossEncoder, BGE-reranker-v2-m3).

Модель не загружается — мокаем sentence_transformers.CrossEncoder.
"""

from unittest.mock import patch, MagicMock

import pytest


class TestReranker:
    def test_import(self):
        from src.reranker import Reranker
        assert Reranker is not None

    @patch("sentence_transformers.CrossEncoder")
    def test_rerank_empty(self, mock_ce):
        from src.reranker import Reranker

        r = Reranker(model_name="mock", device="cpu")
        result = r.rerank("query", [])
        assert result == []

    @patch("sentence_transformers.CrossEncoder")
    def test_rerank_sorts_by_score(self, mock_ce):
        from src.reranker import Reranker

        mock_model = MagicMock()
        mock_model.predict.return_value = [0.1, 0.9, 0.5]
        mock_ce.return_value = mock_model

        r = Reranker(model_name="mock", device="cpu")
        candidates = [
            {"doc_id": "a", "text": "low relevance"},
            {"doc_id": "b", "text": "high relevance"},
            {"doc_id": "c", "text": "medium relevance"},
        ]
        result = r.rerank("test query", candidates, top_k=2)
        assert len(result) == 2
        assert result[0]["doc_id"] == "b"
        assert result[1]["doc_id"] == "c"
        for d in result:
            assert "rerank_score" in d

    @patch("sentence_transformers.CrossEncoder")
    def test_rerank_top_k(self, mock_ce):
        from src.reranker import Reranker

        mock_model = MagicMock()
        mock_model.predict.return_value = [0.3, 0.7, 0.1]
        mock_ce.return_value = mock_model

        r = Reranker(model_name="mock", device="cpu")
        candidates = [
            {"doc_id": "a", "text": "one"},
            {"doc_id": "b", "text": "two"},
            {"doc_id": "c", "text": "three"},
        ]
        result = r.rerank("query", candidates, top_k=1)
        assert len(result) == 1
        assert result[0]["doc_id"] == "b"

    @patch("sentence_transformers.CrossEncoder")
    def test_rerank_preserves_full_list_when_no_top_k(self, mock_ce):
        from src.reranker import Reranker

        mock_model = MagicMock()
        mock_model.predict.return_value = [0.2, 0.8]
        mock_ce.return_value = mock_model

        r = Reranker(model_name="mock", device="cpu")
        candidates = [
            {"doc_id": "a", "text": "low"},
            {"doc_id": "b", "text": "high"},
        ]
        result = r.rerank("query", candidates, top_k=None)
        assert len(result) == 2
        assert result[0]["doc_id"] == "b"


class TestRerankerProviderSelection:
    """Валидация провайдера в конструкторе и диспетчеризация в rerank()."""

    def test_unknown_provider_raises(self):
        from src.reranker import Reranker

        with pytest.raises(ValueError, match="Unknown reranker provider"):
            Reranker(provider="magic")

    @pytest.mark.parametrize("provider", ["anthropic", "ollama"])
    def test_unimplemented_providers_raise(self, provider):
        from src.reranker import Reranker

        with pytest.raises(NotImplementedError, match=provider):
            Reranker(provider=provider)

    @patch("sentence_transformers.CrossEncoder")
    def test_local_model_loaded_lazily_once(self, mock_ce):
        from src.reranker import Reranker

        mock_model = MagicMock()
        mock_model.predict.return_value = [0.5]
        mock_ce.return_value = mock_model

        r = Reranker(model_name="mock", device="cpu")
        assert r._model is None
        r.rerank("q", [{"doc_id": "a", "text": "one"}])
        r.rerank("q", [{"doc_id": "a", "text": "one"}])
        mock_ce.assert_called_once_with("mock", device="cpu")

    @patch("sentence_transformers.CrossEncoder")
    def test_empty_candidates_skip_model_load(self, mock_ce):
        from src.reranker import Reranker

        assert Reranker(model_name="mock").rerank("q", []) == []
        mock_ce.assert_not_called()

    @patch("sentence_transformers.CrossEncoder")
    def test_predict_receives_query_text_pairs(self, mock_ce):
        from src.reranker import Reranker

        mock_model = MagicMock()
        mock_model.predict.return_value = [0.1, 0.2]
        mock_ce.return_value = mock_model

        Reranker(model_name="mock").rerank("query", [{"text": "one"}, {"text": "two"}])
        assert mock_model.predict.call_args[0][0] == [("query", "one"), ("query", "two")]

    @patch("httpx.Client")
    def test_rerank_dispatch_to_unknown_provider(self, mock_client_class):
        from src.reranker import Reranker

        r = Reranker(provider="openai-compatible", base_url="http://host/v1")
        r.provider = "changed-later"
        with pytest.raises(NotImplementedError, match="changed-later"):
            r.rerank("q", [{"text": "one"}])


class TestRerankerApiProvider:
    """openai-compatible: POST {base_url}/rerank."""

    def _reranker(self, mock_client_class, payload, api_key=""):
        client = MagicMock()
        response = MagicMock()
        response.json.return_value = payload
        client.post.return_value = response
        mock_client_class.return_value = client
        from src.reranker import Reranker

        return Reranker(
            provider="openai-compatible",
            model_name="bge",
            base_url="http://host/v1/",
            api_key=api_key,
        ), client

    @patch("httpx.Client")
    def test_request_shape_and_sorting(self, mock_client_class):
        r, client = self._reranker(
            mock_client_class,
            {"results": [{"index": 0, "relevance_score": 0.1}, {"index": 1, "relevance_score": 0.9}]},
            api_key="sk-1",
        )
        candidates = [{"doc_id": "a", "text": "one"}, {"doc_id": "b", "text": "two"}]
        result = r.rerank("query", candidates)

        assert [d["doc_id"] for d in result] == ["b", "a"]
        assert result[0]["rerank_score"] == 0.9
        call = client.post.call_args
        assert call[0][0] == "http://host/v1/rerank"
        assert call[1]["headers"]["Authorization"] == "Bearer sk-1"
        assert call[1]["json"] == {"model": "bge", "query": "query", "documents": ["one", "two"]}

    @patch("httpx.Client")
    def test_top_k_sent_and_applied(self, mock_client_class):
        r, client = self._reranker(
            mock_client_class,
            {"results": [{"index": 0, "relevance_score": 0.2}, {"index": 1, "relevance_score": 0.8}]},
        )
        result = r.rerank("query", [{"doc_id": "a", "text": "one"}, {"doc_id": "b", "text": "two"}], top_k=1)

        assert [d["doc_id"] for d in result] == ["b"]
        assert client.post.call_args[1]["json"]["top_n"] == 1
        assert "Authorization" not in client.post.call_args[1]["headers"]

    @patch("httpx.Client")
    def test_missing_scores_default_to_zero(self, mock_client_class):
        r, _ = self._reranker(mock_client_class, {"results": [{"index": 1, "relevance_score": 0.5}]})
        result = r.rerank("query", [{"doc_id": "a", "text": "one"}, {"doc_id": "b", "text": "two"}])

        assert [d["doc_id"] for d in result] == ["b", "a"]
        assert "rerank_score" not in result[1]

    @patch("httpx.Client")
    def test_empty_results_keep_candidates(self, mock_client_class):
        r, _ = self._reranker(mock_client_class, {})
        candidates = [{"doc_id": "a", "text": "one"}, {"doc_id": "b", "text": "two"}]
        assert len(r.rerank("query", candidates)) == 2

    @patch("httpx.Client")
    def test_http_error_propagates(self, mock_client_class):
        r, client = self._reranker(mock_client_class, {})
        client.post.return_value.raise_for_status.side_effect = RuntimeError("500")
        with pytest.raises(RuntimeError, match="500"):
            r.rerank("query", [{"text": "one"}])

    @patch("httpx.Client")
    def test_close_closes_client(self, mock_client_class):
        r, client = self._reranker(mock_client_class, {})
        r.close()
        client.close.assert_called_once()

    @patch("sentence_transformers.CrossEncoder")
    def test_close_without_client_is_noop(self, mock_ce):
        from src.reranker import Reranker

        Reranker(model_name="mock").close()
