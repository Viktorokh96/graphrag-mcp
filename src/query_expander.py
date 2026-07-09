"""Query expansion — генерация альтернативных формулировок запроса.

Использует малую LLM (Qwen3-1.8B через Ollama) для парафраза.
По умолчанию выключен (QUERY_EXPANSION_ENABLED=False).
"""

from typing import Optional


class QueryExpander:
    """Генерация альтернативных формулировок для улучшения recall.

    Пайплайн: query → expand → [q, q1, q2, q3]
      → parallel hybrid search (каждый)
      → RRF merge всех результатов
      → rerank (если включён)
      → top-k
    """

    def __init__(
        self,
        model: str = "qwen3:1.8b",
        base_url: str = "http://localhost:11434",
        count: int = 3,
    ):
        self.model = model
        self.base_url = base_url
        self.count = count
        self._client = None

    def _ensure_client(self):
        if self._client is None:
            from ollama import Client
            self._client = Client(host=self.base_url)
        return self._client

    def expand(self, query: str, count: Optional[int] = None) -> list[str]:
        """Сгенерировать альтернативные формулировки запроса.

        Args:
            query: исходный запрос
            count: число вариантов (None = self.count)

        Returns:
            [исходный_запрос, вариант1, вариант2, ...]
        """
        n = count if count is not None else self.count
        if n < 1:
            return [query]

        prompt = (
            f"Generate {n} alternative phrasings of this query. "
            "Return one per line, no numbering, no extra text.\n\n"
            f"Query: {query}"
        )

        client = self._ensure_client()
        response = client.generate(model=self.model, prompt=prompt)
        text = response.get("response", "")
        variants = [q.strip() for q in text.split("\n") if q.strip()]
        return [query] + variants[:n]
