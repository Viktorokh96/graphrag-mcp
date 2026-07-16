"""Query expansion — генерация альтернативных формулировок запроса.

Провайдеры:
  - ollama:             Ollama /api/generate
  - openai-compatible:  POST /v1/chat/completions (OpenAI, vLLM, Ollama, ...)
  - anthropic:          Anthropic Messages API
  - sentence_transformer: NotImplementedError (не поддерживает генерацию текста)
"""

import logging
from typing import Optional

import httpx

logger = logging.getLogger(__name__)


class QueryExpander:
    """Генерация альтернативных формулировок для улучшения recall."""

    def __init__(
        self,
        provider: str = "ollama",
        model: str = "qwen3:1.8b",
        base_url: str = "http://localhost:11434",
        api_key: str = "",
        count: int = 3,
    ):
        self.provider = provider
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.count = count
        self._client = None

        if provider == "ollama":
            self._client = httpx.Client(timeout=120.0)
        elif provider == "openai-compatible":
            self._client = httpx.Client(timeout=120.0)
        elif provider == "anthropic":
            self._client = httpx.Client(timeout=120.0)
        elif provider == "sentence_transformer":
            raise NotImplementedError(
                "Query expansion is not supported for sentence_transformer provider"
            )
        else:
            raise ValueError(f"Unknown expansion provider: {provider!r}")

    def expand(self, query: str, count: Optional[int] = None) -> list[str]:
        n = count if count is not None else self.count
        if n < 1:
            return [query]

        if self.provider == "ollama":
            return self._expand_ollama(query, n)
        elif self.provider == "openai-compatible":
            return self._expand_openai(query, n)
        elif self.provider == "anthropic":
            return self._expand_anthropic(query, n)
        else:
            raise NotImplementedError(f"Expansion provider {self.provider!r}")

    # -- ollama ----------------------------------------------------------------

    def _expand_ollama(self, query: str, n: int) -> list[str]:
        url = f"{self.base_url}/api/generate"
        prompt = self._build_prompt(query, n)
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
        }
        response = self._client.post(url, json=payload)
        response.raise_for_status()
        text = response.json().get("response", "")
        variants = [q.strip() for q in text.split("\n") if q.strip()]
        return [query] + variants[:n]

    # -- openai-compatible -----------------------------------------------------

    def _expand_openai(self, query: str, n: int) -> list[str]:
        url = f"{self.base_url}/chat/completions"
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        prompt = self._build_prompt(query, n)
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.7,
            "max_tokens": 256,
        }
        response = self._client.post(url, headers=headers, json=payload)
        response.raise_for_status()
        text = response.json()["choices"][0]["message"]["content"]
        variants = [q.strip() for q in text.split("\n") if q.strip()]
        return [query] + variants[:n]

    # -- anthropic -------------------------------------------------------------

    def _expand_anthropic(self, query: str, n: int) -> list[str]:
        url = f"{self.base_url}/messages"
        headers = {
            "Content-Type": "application/json",
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
        }

        prompt = self._build_prompt(query, n)
        payload = {
            "model": self.model,
            "max_tokens": 256,
            "messages": [{"role": "user", "content": prompt}],
        }
        response = self._client.post(url, headers=headers, json=payload)
        response.raise_for_status()
        text = response.json()["content"][0]["text"]
        variants = [q.strip() for q in text.split("\n") if q.strip()]
        return [query] + variants[:n]

    # -- prompt ----------------------------------------------------------------

    def _build_prompt(self, query: str, n: int) -> str:
        return (
            f"Generate {n} alternative phrasings of this query. "
            "Return one per line, no numbering, no extra text.\n\n"
            f"Query: {query}"
        )

    def close(self):
        if self._client:
            self._client.close()
