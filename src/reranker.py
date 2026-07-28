"""Reranker — переранжирование результатов поиска.

Провайдеры:
  - sentence_transformer: локальный CrossEncoder (BAAI/bge-reranker-v2-m3)
  - openai-compatible:   любой сервер с POST /v1/rerank
  - anthropic, ollama:   NotImplementedError (пока нет API)
"""

import logging
from typing import Optional

import httpx

from src.http_utils import json_headers

logger = logging.getLogger(__name__)


class Reranker:
    """Переранжирование топ-кандидатов."""

    def __init__(
        self,
        provider: str = "sentence_transformer",
        model_name: str = "BAAI/bge-reranker-v2-m3",
        device: str = "cpu",
        base_url: str = "",
        api_key: str = "",
    ):
        self.provider = provider
        self.model_name = model_name
        self.device = device
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self._model = None
        self._client = None

        if provider == "sentence_transformer":
            pass  # lazy load CrossEncoder
        elif provider == "openai-compatible":
            self._client = httpx.Client(timeout=30.0)
        elif provider in ("anthropic", "ollama"):
            raise NotImplementedError(f"Reranker provider {provider!r} is not implemented yet")
        else:
            raise ValueError(f"Unknown reranker provider: {provider!r}")

    def _ensure_model(self):
        if self._model is None:
            from sentence_transformers import CrossEncoder
            self._model = CrossEncoder(self.model_name, device=self.device)
        return self._model

    def rerank(
        self,
        query: str,
        candidates: list[dict],
        top_k: Optional[int] = None,
    ) -> list[dict]:
        if not candidates:
            return []

        if self.provider == "sentence_transformer":
            return self._rerank_local(query, candidates, top_k)
        elif self.provider == "openai-compatible":
            return self._rerank_api(query, candidates, top_k)
        else:
            raise NotImplementedError(f"Reranker provider {self.provider!r}")

    def _rerank_local(
        self, query: str, candidates: list[dict], top_k: Optional[int]
    ) -> list[dict]:
        model = self._ensure_model()
        pairs = [(query, d["text"]) for d in candidates]
        scores = model.predict(pairs)
        for doc, score in zip(candidates, scores):
            doc["rerank_score"] = float(score)
        candidates.sort(key=lambda x: x["rerank_score"], reverse=True)
        if top_k is not None:
            return candidates[:top_k]
        return candidates

    def _rerank_api(
        self, query: str, candidates: list[dict], top_k: Optional[int]
    ) -> list[dict]:
        url = f"{self.base_url}/rerank"
        payload = {
            "model": self.model_name,
            "query": query,
            "documents": [d["text"] for d in candidates],
        }
        if top_k is not None:
            payload["top_n"] = top_k

        response = self._client.post(url, headers=json_headers(self.api_key), json=payload)
        response.raise_for_status()
        data = response.json()

        # OpenAI-compatible rerank response: {"results": [{"index": 0, "relevance_score": 0.9}, ...]}
        results = data.get("results", [])
        for r in results:
            idx = r["index"]
            candidates[idx]["rerank_score"] = float(r["relevance_score"])

        candidates.sort(key=lambda x: x.get("rerank_score", 0.0), reverse=True)
        if top_k is not None:
            return candidates[:top_k]
        return candidates

    def close(self):
        if self._client:
            self._client.close()
