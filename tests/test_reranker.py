"""Тесты для Reranker (CrossEncoder, BGE-reranker-v2-m3).

Модель не загружается — мокаем sentence_transformers.CrossEncoder.
"""

from unittest.mock import patch, MagicMock


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
