"""Общие helpers для фильтрации документов по метаданным.

Используется в BM25-индексе (post-filter), графовой базе (post-filter соседей)
и векторном хранилище (преобразование в `where`-клаузу ChromaDB).

Формат фильтра — dict[str, scalar | list[scalar]]:
  • каждая пара key:value означает: документ подходит, если metadata[key] == value;
  • если value — список: metadata[key] входит в список (семантика $in);
  • условия объединяются через AND (должны выполняться все);
  • None или {} — фильтр отключён (текущее поведение).
"""

from typing import Any, Optional


def matches_metadata_filter(meta: Optional[dict], filt: Optional[dict]) -> bool:
    """Проверить, проходит ли документ с метаданными `meta` фильтр `filt`.

    Args:
        meta: метаданные документа (dict или None).
        filt: фильтр в формате {key: scalar | list[scalar]} или None.

    Returns:
        True если фильтр отключён (None/{}) и документ проходит,
        либо все условия фильтра выполняются.
    """
    if not filt:
        return True
    if not isinstance(meta, dict):
        return False
    for key, expected in filt.items():
        actual = meta.get(key)
        if isinstance(expected, list):
            if actual not in expected:
                return False
        else:
            if actual != expected:
                return False
    return True


def to_chroma_where(filt: Optional[dict]) -> Optional[dict]:
    """Преобразовать metadata_filter в `where`-клаузу ChromaDB.

    ChromaDB поддерживает прямое равенство {key: value} и оператор $in:
      • {"key": "value"}              → {"key": "value"}
      • {"key": ["a", "b"]}           → {"key": {"$in": ["a", "b"]}}
      • несколько ключей (AND)        → {"$and": [{...}, {...}]}

    Нескалярные значения (кроме list) в фильтре игнорируются, чтобы не
    ломать валидацию ChromaDB (она принимает только примитивы).

    Args:
        filt: фильтр в формате {key: scalar | list[scalar]} или None.

    Returns:
        where-клауза для ChromaDB или None (фильтр отключён).
    """
    if not filt:
        return None
    clauses: list[dict[str, Any]] = []
    for key, expected in filt.items():
        # Скалярные значения: str/int/float/bool/None
        if isinstance(expected, (str, int, float, bool)) or expected is None:
            clauses.append({key: expected})
        elif isinstance(expected, list):
            # $in-семантика; пропускаем элементы-контейнеры внутри списка
            cleaned = [
                v for v in expected
                if isinstance(v, (str, int, float, bool)) or v is None
            ]
            if cleaned:
                clauses.append({key: {"$in": cleaned}})
        # Прочие типы (dict и т.п.) игнорируем — ChromaDB их не поддержит
    if not clauses:
        return None
    if len(clauses) == 1:
        return clauses[0]
    return {"$and": clauses}
