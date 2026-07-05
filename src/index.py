"""Модуль семантического индекса для хранения и поиска векторов."""

from typing import Dict, List, Optional, Tuple
import numpy as np


class SemanticIndex:
    """Семантический индекс.

    Хранит векторы документов и выполняет поиск ближайших
    соседей по косинусной близости.
    """

    def __init__(self):
        self.vectors: Dict[str, np.ndarray] = {}
        self.texts: Dict[str, str] = {}
        self.metadata: Dict[str, dict] = {}

    def add(
        self,
        doc_id: str,
        vector: np.ndarray,
        text: str = "",
        metadata: Optional[dict] = None,
    ) -> None:
        """Добавляет документ в индекс.

        Если doc_id уже существует — обновляет запись.

        Args:
            doc_id: Уникальный идентификатор документа.
            vector: Векторное представление документа.
            text: Исходный текст документа.
            metadata: Произвольные метаданные документа.
        """
        self.vectors[doc_id] = vector
        self.texts[doc_id] = text
        self.metadata[doc_id] = metadata if metadata is not None else {}

    def search(
        self, query_vector: np.ndarray, top_k: int = 5
    ) -> List[Tuple[str, float, dict]]:
        """Ищет top_k ближайших документов по косинусной близости.

        Args:
            query_vector: Вектор запроса.
            top_k: Количество результатов.

        Returns:
            Список кортежей (doc_id, score, metadata),
            отсортированных по убыванию score.
        """
        if not self.vectors:
            return []

        scores = []
        for doc_id, vector in self.vectors.items():
            score = self._cosine_similarity(query_vector, vector)
            scores.append((doc_id, score, self.metadata[doc_id]))

        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]

    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Вычисляет косинусную близость между двумя векторами.

        Args:
            a: Первый вектор.
            b: Второй вектор.

        Returns:
            Косинусная близость в диапазоне [0, 1].
        """
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)

        if norm_a == 0 or norm_b == 0:
            return 0.0

        return float(np.dot(a, b) / (norm_a * norm_b))