"""Reranker для переранжирования результатов поиска.

Модель: BAAI/bge-reranker-v2-m3 (CrossEncoder, ~1GB).
Ленивая загрузка — модель поднимается только при первом вызове rerank().

По умолчанию выключен (RERANK_ENABLED=False). Включается через
`rerank=True` в запросе поиска или env RERANK_ENABLED=true.
"""

from typing import Optional


class Reranker:
    """Cross-encoder reranker для уточнения топ-кандидатов.

    Используется в `search_hybrid()` после RRF-слияния: берёт k×2 кандидатов,
    вычисляет релевантность каждой пары (query, doc) и пересортировывает.
    """

    def __init__(self, model_name: str = "BAAI/bge-reranker-v2-m3", device: str = "cpu"):
        self.model_name = model_name
        self.device = device
        self._model = None

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
        """Переранжировать кандидатов по релевантности к запросу.

        Args:
            query: исходный поисковый запрос
            candidates: список dict с полем `text`
            top_k: сколько результатов вернуть (None = все)

        Returns:
            Тот же список, отсортированный по убыванию rerank_score,
            с добавленным полем `rerank_score`.
        """
        if not candidates:
            return []
        model = self._ensure_model()
        pairs = [(query, d["text"]) for d in candidates]
        scores = model.predict(pairs)
        for doc, score in zip(candidates, scores):
            doc["rerank_score"] = float(score)
        candidates.sort(key=lambda x: x["rerank_score"], reverse=True)
        if top_k is not None:
            return candidates[:top_k]
        return candidates
