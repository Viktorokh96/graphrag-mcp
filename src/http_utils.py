"""Общие helpers для HTTP-клиентов провайдеров (эмбеддинги, reranker, expansion)."""


def json_headers(api_key: str = "") -> dict[str, str]:
    """JSON-заголовки с опциональной Bearer-авторизацией."""
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    return headers


def split_lines(text: str) -> list[str]:
    """Разбить ответ LLM на непустые строки без пробельных краёв."""
    return [line.strip() for line in text.split("\n") if line.strip()]
