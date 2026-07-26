"""Общие helpers для формирования выдачи поиска.

Используются всеми фронтендами (MCP-сервер, HTTP API, CLI), чтобы формат
результата и загрузка графовых связей были одинаковыми.
"""

from typing import Optional

from src._meta_filter import normalize_metadata_filter


def format_results(
    results: list[tuple[str, str, float, dict]],
    max_chars: Optional[int] = None,
) -> list[dict]:
    """Кортежи поиска → dict-результаты {doc_id, text, score, metadata}.

    Args:
        results: [(doc_id, text, score, metadata), ...]
        max_chars: обрезать текст до N символов (None — полный текст)
    """
    return [
        {
            "doc_id": doc_id,
            "text": text[:max_chars] if max_chars is not None else text,
            "score": round(score, 4),
            "metadata": metadata if isinstance(metadata, dict) else {},
        }
        for doc_id, text, score, metadata in results
    ]


def enrich_with_links(rag, docs: list[dict], params: dict) -> list[dict]:
    """Догрузить графовые связи в поле `links` по параметрам вызова.

    Args:
        rag: экземпляр RAGSystem
        docs: dict-результаты (изменяются на месте)
        params: словарь с ключами relations_load_depth / relations_load_type_filter /
                relations_load_meta_filter (значения в форме, приходящей от клиента)
    """
    if not docs:
        return docs
    return rag._enrich_with_links(
        docs,
        relations_load_depth=params.get("relations_load_depth", 1),
        relations_load_type_filter=params.get("relations_load_type_filter"),
        relations_load_meta_filter=normalize_metadata_filter(
            params.get("relations_load_meta_filter")
        ),
    )
