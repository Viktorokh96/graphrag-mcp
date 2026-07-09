"""QdrantVectorStore — векторное хранилище на Qdrant (embedded или HTTP).

Заменяет ChromaDB. Хранит dense-вектор (эмбеддинги, cosine) и sparse-вектор
BM25 (токены с TF-сатурацией; IDF считает Qdrant через Modifier.IDF).

Режимы:
  • location = путь (``./rag_data/qdrant``) → embedded QdrantLocal, 0 внешних зависимостей
  • location = URL (``http://host:6333``)   → QdrantRemote (production)

Тексты документов здесь НЕ хранятся (источник правды — DocumentStore);
payload содержит doc_id, metadata (для нативных фильтров) и text_preview.
"""

import re
import uuid
from typing import Optional

from qdrant_client import QdrantClient, models

COLLECTION = "rag_docs"
# Namespace для детерминированного uuid5(doc_id) → point_id: Qdrant принимает
# только UUID/uint как id точки, а doc_id бывает произвольной строкой
# (например, SHA256-имя сущности из авто-извлечения графа).
_POINT_NS = uuid.UUID("6ba7b810-9dad-11d1-80b4-00c04fd430c8")

# BM25 TF-сатурация (IDF применяет Qdrant на своей стороне)
_BM25_K1 = 1.2
_BM25_B = 0.75
_BM25_AVG_LEN = 256.0

_TOKEN_RE = re.compile(r"[a-zа-яё0-9_]+", re.IGNORECASE)


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


def _point_id(doc_id: str) -> str:
    return str(uuid.uuid5(_POINT_NS, doc_id))


def bm25_sparse_vector(text: str, is_query: bool = False) -> models.SparseVector:
    """Sparse-вектор BM25: индекс = хэш токена, значение = насыщенный TF.

    Для запроса значения = 1.0 (классический BM25 взвешивает только документы).
    """
    tokens = _tokenize(text)
    if not tokens:
        return models.SparseVector(indices=[], values=[])
    tf: dict[int, float] = {}
    for tok in tokens:
        idx = hash_token(tok)
        tf[idx] = tf.get(idx, 0.0) + 1.0
    if is_query:
        return models.SparseVector(indices=list(tf.keys()), values=[1.0] * len(tf))
    doc_len = float(len(tokens))
    norm = _BM25_K1 * (1.0 - _BM25_B + _BM25_B * doc_len / _BM25_AVG_LEN)
    indices, values = [], []
    for idx, freq in tf.items():
        indices.append(idx)
        values.append(freq * (_BM25_K1 + 1.0) / (freq + norm))
    return models.SparseVector(indices=indices, values=values)


def hash_token(token: str) -> int:
    """Стабильный 32-битный положительный хэш токена (не зависит от PYTHONHASHSEED)."""
    import zlib

    return zlib.crc32(token.encode("utf-8"))


def to_qdrant_filter(filt: Optional[dict]) -> Optional[models.Filter]:
    """metadata_filter ({key: scalar | list}) → нативный Qdrant Filter (AND)."""
    if not filt:
        return None
    conditions: list[models.FieldCondition] = []
    for key, expected in filt.items():
        field = f"metadata.{key}"
        if isinstance(expected, list):
            cleaned = [v for v in expected if isinstance(v, (str, int, bool)) or v is None]
            # MatchAny не принимает float/None — они сравниваются по одному через should
            if cleaned:
                conditions.append(
                    models.FieldCondition(key=field, match=models.MatchAny(any=cleaned))
                )
        elif isinstance(expected, (str, int, bool)):
            conditions.append(models.FieldCondition(key=field, match=models.MatchValue(value=expected)))
        elif isinstance(expected, float):
            conditions.append(
                models.FieldCondition(key=field, range=models.Range(gte=expected, lte=expected))
            )
        # dict и прочее — игнорируем (как в to_chroma_where)
    if not conditions:
        return None
    return models.Filter(must=conditions)


class QdrantVectorStore:
    """Хранилище dense+sparse векторов на Qdrant."""

    def __init__(self, location: str = "./rag_data/qdrant", dimension: int = 1024):
        self.location = location
        self.dimension = dimension
        if location.startswith(("http://", "https://")):
            self._client = QdrantClient(url=location)
        else:
            self._client = QdrantClient(path=location)
        self._ensure_collection()

    # -- collection lifecycle -------------------------------------------------

    def _ensure_collection(self) -> None:
        if not self._client.collection_exists(COLLECTION):
            self._create_collection()
            return
        actual = self.get_dimension()
        if actual and actual != self.dimension:
            import logging

            logging.getLogger(__name__).warning(
                "Collection dim=%s, config dim=%s. Run `rag-server migrate` to re-index.",
                actual,
                self.dimension,
            )

    def _create_collection(self) -> None:
        self._client.create_collection(
            collection_name=COLLECTION,
            vectors_config={
                "dense": models.VectorParams(size=self.dimension, distance=models.Distance.COSINE),
            },
            sparse_vectors_config={
                "bm25": models.SparseVectorParams(modifier=models.Modifier.IDF),
            },
        )

    def recreate_collection(self) -> None:
        if self._client.collection_exists(COLLECTION):
            self._client.delete_collection(COLLECTION)
        self._create_collection()

    def get_dimension(self) -> int:
        """Размерность dense-векторов коллекции (из конфига коллекции)."""
        try:
            info = self._client.get_collection(COLLECTION)
            params = info.config.params.vectors
            if isinstance(params, dict) and "dense" in params:
                return params["dense"].size
        except Exception:
            pass
        return 0

    # -- CRUD -----------------------------------------------------------------

    def add(self, doc_id: str, text: str, embedding: list[float], metadata: Optional[dict] = None) -> str:
        self._client.upsert(
            collection_name=COLLECTION,
            points=[
                models.PointStruct(
                    id=_point_id(doc_id),
                    vector={
                        "dense": embedding,
                        "bm25": bm25_sparse_vector(text),
                    },
                    payload={
                        "doc_id": doc_id,
                        "metadata": metadata or {},
                        "text_preview": text[:200],
                    },
                )
            ],
        )
        return doc_id

    def remove(self, doc_id: str) -> None:
        self._client.delete(
            collection_name=COLLECTION,
            points_selector=models.PointIdsList(points=[_point_id(doc_id)]),
        )

    def clear(self) -> None:
        self.recreate_collection()

    def count(self) -> int:
        return self._client.count(COLLECTION, exact=True).count

    def get_all_ids(self) -> set[str]:
        ids: set[str] = set()
        offset = None
        while True:
            points, offset = self._client.scroll(
                COLLECTION, limit=1024, offset=offset, with_payload=["doc_id"], with_vectors=False
            )
            ids.update(p.payload["doc_id"] for p in points)
            if offset is None:
                return ids

    def has(self, doc_id: str) -> bool:
        return bool(self._client.retrieve(COLLECTION, ids=[_point_id(doc_id)], with_payload=False))

    def update_embedding(self, doc_id: str, embedding: list[float]) -> None:
        self._client.update_vectors(
            collection_name=COLLECTION,
            points=[models.PointVectors(id=_point_id(doc_id), vector={"dense": embedding})],
        )

    # -- search ---------------------------------------------------------------

    def search(
        self,
        query_embedding: list[float],
        k: int = 5,
        metadata_filter: Optional[dict] = None,
    ) -> list[dict]:
        """Dense-поиск. Возвращает [{doc_id, score, metadata}] по убыванию score.

        Score — cosine similarity из Qdrant, клипнутый в [0, 1].
        """
        if self.count() == 0:
            return []
        hits = self._client.query_points(
            collection_name=COLLECTION,
            query=query_embedding,
            using="dense",
            limit=k,
            query_filter=to_qdrant_filter(metadata_filter),
            with_payload=True,
        ).points
        return [
            {
                "doc_id": h.payload["doc_id"],
                "score": max(0.0, min(1.0, float(h.score))),
                "metadata": h.payload.get("metadata") or {},
            }
            for h in hits
        ]

    def bm25_search(self, query: str, k: int = 5, metadata_filter: Optional[dict] = None) -> list[dict]:
        """Sparse BM25-поиск. Возвращает [{doc_id, score, metadata}]."""
        if self.count() == 0:
            return []
        sparse = bm25_sparse_vector(query, is_query=True)
        if not sparse.indices:
            return []
        hits = self._client.query_points(
            collection_name=COLLECTION,
            query=sparse,
            using="bm25",
            limit=k,
            query_filter=to_qdrant_filter(metadata_filter),
            with_payload=True,
        ).points
        return [
            {
                "doc_id": h.payload["doc_id"],
                "score": float(h.score),
                "metadata": h.payload.get("metadata") or {},
            }
            for h in hits
            if h.score > 0
        ]

    def stats(self) -> dict:
        return {
            "total_documents": self.count(),
            "store_path": self.location,
            "dimension": self.get_dimension() or self.dimension,
        }

    def close(self) -> None:
        self._client.close()
