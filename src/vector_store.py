"""QdrantVectorStore — векторное хранилище на Qdrant (embedded или HTTP).

Заменяет ChromaDB. Хранит dense-вектор (эмбеддинги, cosine) и sparse-вектор
BM25 (токены с TF-сатурацией; IDF считает Qdrant через Modifier.IDF).

Режимы:
  • location = путь (``./rag_data/qdrant``) → embedded QdrantLocal, 0 внешних зависимостей
  • location = URL (``http://host:6333``)   → QdrantRemote (production)

Тексты документов здесь НЕ хранятся (источник правды — DocumentStore);
payload содержит doc_id, metadata (для нативных фильтров) и text_preview.
"""

import logging
import math
import re
import uuid
from typing import Optional

logger = logging.getLogger(__name__)


from qdrant_client import QdrantClient, models

COLLECTION = "rag_docs"
# Namespace для детерминированного uuid5(doc_id) → point_id: Qdrant принимает
# только UUID/uint как id точки, а doc_id бывает произвольной строкой
# (например, SHA256-имя сущности из авто-извлечения графа).
_POINT_NS = uuid.UUID("6ba7b810-9dad-11d1-80b4-00c04fd430c8")

# BM25 TF-сатурация
_BM25_K1 = 1.2
_BM25_B = 0.75
_BM25_AVG_LEN = 256.0
_BM25_EPS = 0.25  # сглаживание IDF: log((N - df + 0.5) / (df + 0.5))

_TOKEN_RE = re.compile(r"[a-zа-яё0-9_]+", re.IGNORECASE)


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


def _point_id(doc_id: str, chunk_index: Optional[int] = None) -> str:
    key = doc_id if chunk_index is None else f"{doc_id}#{chunk_index}"
    return str(uuid.uuid5(_POINT_NS, key))


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
            cleaned = [v for v in expected if isinstance(v, (str, int, bool))]
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
        self._is_remote = location.startswith(("http://", "https://"))
        logger.info("Connecting to Qdrant: %s (dim=%d, remote=%s) …", location, dimension, self._is_remote)
        t0 = __import__("time").monotonic()
        self._connect()
        logger.info("Qdrant client connected in %.1fs", __import__("time").monotonic() - t0)
        self._token_df: dict[int, int] = {}
        self._did_load_token_df = False
        t0 = __import__("time").monotonic()
        self._ensure_collection()
        logger.info("Qdrant collection '%s' ready in %.1fs", COLLECTION, __import__("time").monotonic() - t0)

    def _connect(self) -> None:
        if self._is_remote:
            self._client = QdrantClient(url=self.location)
        else:
            self._client = QdrantClient(path=self.location)

    def _ensure_collection(self) -> None:
        if not self._client.collection_exists(COLLECTION):
            logger.info("Creating Qdrant collection '%s' …", COLLECTION)
            self._create_collection()
            return
        actual = self.get_dimension()
        if actual and actual != self.dimension:
            logger.warning(
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
                "bm25": models.SparseVectorParams(),
            },
        )

    def recreate_collection(self) -> None:
        """Полностью пересоздать коллекцию (например, при смене размерности).

        В embedded (local) режиме qdrant-client содержит баг: delete_collection +
        create_collection с тем же именем не сбрасывает персистентное хранилище на
        диске — старые точки и размерность остаются. Поэтому для local-режима
        физически удаляем директорию хранилища и переоткрываем клиент. Для
        remote-сервера обычного delete + create достаточно.
        """
        if self._is_remote:
            if self._client.collection_exists(COLLECTION):
                self._client.delete_collection(COLLECTION)
            self._create_collection()
        else:
            import shutil
            from pathlib import Path

            self._client.close()
            path = Path(self.location)
            if path.exists():
                shutil.rmtree(path, ignore_errors=True)
            self._connect()
            self._create_collection()
        self._token_df.clear()
        self._did_load_token_df = True

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

    def _load_token_df(self) -> None:
        if self._did_load_token_df:
            return
        self._token_df.clear()
        offset = None
        while True:
            points, offset = self._client.scroll(
                COLLECTION, limit=1024, offset=offset,
                with_payload=["text_hashes"], with_vectors=False,
            )
            for p in points:
                hashes = p.payload.get("text_hashes") or []
                for h in hashes:
                    self._token_df[h] = self._token_df.get(h, 0) + 1
            if offset is None:
                break
        self._did_load_token_df = True

    def _update_token_df(self, text: str, delta: int) -> None:
        tokens = set(_tokenize(text))
        for tok in tokens:
            h = hash_token(tok)
            self._token_df[h] = max(0, self._token_df.get(h, 0) + delta)

    def _idf_for_tokens(self, tokens: list[str]) -> dict[int, float]:
        self._load_token_df()
        n = max(self.count(), 1)
        idf: dict[int, float] = {}
        seen: set[int] = set()
        for tok in tokens:
            h = hash_token(tok)
            if h in seen:
                continue
            seen.add(h)
            df = self._token_df.get(h, 0)
            idf[h] = math.log((n - df + 0.5) / (df + 0.5) + 1.0)
        return idf

    # -- CRUD -----------------------------------------------------------------

    def _point(self, doc_id: str, text: str, embedding: list[float],
               metadata: Optional[dict], chunk_index: Optional[int] = None) -> models.PointStruct:
        payload = {
            "doc_id": doc_id,
            "metadata": metadata or {},
            "text_preview": text[:200],
            "text_hashes": list(set(hash_token(t) for t in _tokenize(text))),
        }
        if chunk_index is not None:
            payload["chunk_index"] = chunk_index
        return models.PointStruct(
            id=_point_id(doc_id, chunk_index),
            vector={"dense": embedding, "bm25": bm25_sparse_vector(text)},
            payload=payload,
        )

    def replace(self, doc_id: str, text: str, embedding: list[float], metadata: Optional[dict] = None) -> None:
        """Полная замена точек документа: удалить старые + добавить новые."""
        self.remove(doc_id)
        self.add(doc_id, text, embedding, metadata)

    def add(self, doc_id: str, text: str, embedding: list[float], metadata: Optional[dict] = None) -> str:
        self._client.upsert(collection_name=COLLECTION, points=[self._point(doc_id, text, embedding, metadata)])
        self._update_token_df(text, delta=1)
        return doc_id

    def add_chunks(
        self,
        doc_id: str,
        chunks: list[tuple[str, list[float]]],
        metadata: Optional[dict] = None,
    ) -> str:
        """Индексировать документ как несколько chunk-точек с общим payload.doc_id.

        Каждый чанк — отдельная точка Qdrant (id = uuid5(doc_id#i)), но поиск
        схлопывает их обратно в один doc_id (см. _dedup_hits). Один чанк идёт
        обычным одно-точечным путём (обратная совместимость, id = uuid5(doc_id)).
        """
        if len(chunks) == 1:
            return self.add(doc_id, chunks[0][0], chunks[0][1], metadata)
        points = [
            self._point(doc_id, text, emb, metadata, chunk_index=i)
            for i, (text, emb) in enumerate(chunks)
        ]
        self._client.upsert(collection_name=COLLECTION, points=points)
        for text, _emb in chunks:
            self._update_token_df(text, delta=1)
        return doc_id

    def remove(self, doc_id: str) -> None:
        # Документ может быть представлен несколькими chunk-точками — удаляем все
        # по payload.doc_id, попутно уменьшая token_df на каждую.
        selector = models.Filter(
            must=[models.FieldCondition(key="doc_id", match=models.MatchValue(value=doc_id))]
        )
        offset = None
        while True:
            points, offset = self._client.scroll(
                COLLECTION, scroll_filter=selector, limit=256, offset=offset,
                with_payload=["text_hashes"], with_vectors=False,
            )
            for p in points:
                for h in (p.payload.get("text_hashes") or []):
                    self._token_df[h] = max(0, self._token_df.get(h, 0) - 1)
            if offset is None:
                break
        self._client.delete(
            collection_name=COLLECTION,
            points_selector=models.FilterSelector(filter=selector),
        )

    def clear(self) -> None:
        """Удалить все точки, сохранив конфигурацию коллекции.

        Удаление точек по пустому фильтру работает одинаково в local и remote
        режимах (в отличие от recreate_collection, который меняет и структуру).
        """
        self._client.delete(
            collection_name=COLLECTION,
            points_selector=models.FilterSelector(filter=models.Filter()),
        )
        self._token_df.clear()
        self._did_load_token_df = True

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
        # Ищем по payload.doc_id (документ может быть набором chunk-точек).
        count = self._client.count(
            COLLECTION,
            count_filter=models.Filter(
                must=[models.FieldCondition(key="doc_id", match=models.MatchValue(value=doc_id))]
            ),
            exact=True,
        ).count
        return count > 0

    def get_all_doc_embeddings(self) -> dict[str, list[list[float]]]:
        """Вернуть все эмбеддинги, сгруппированные по doc_id.

        Multi-chunk документы содержат несколько векторов (усредняются вызывающим
        кодом при необходимости).
        """
        doc_embeddings: dict[str, list[list[float]]] = {}
        offset = None
        while True:
            points, offset = self._client.scroll(
                COLLECTION,
                limit=1024,
                offset=offset,
                with_vectors=True,
                with_payload=["doc_id"],
            )
            for p in points:
                doc_id = p.payload.get("doc_id") if p.payload else None
                if doc_id and p.vector is not None:
                    vec = p.vector.get("dense") if isinstance(p.vector, dict) else p.vector
                    if vec is not None:
                        doc_embeddings.setdefault(doc_id, []).append(vec)
            if offset is None:
                break
        return doc_embeddings

    @staticmethod
    def _dedup_hits(hits: list, k: int, clip: bool = False) -> list[dict]:
        """Схлопнуть chunk-точки одного документа в один результат (макс. score).

        hits уже отсортированы Qdrant по убыванию score, поэтому первый
        встреченный chunk документа несёт максимальный score. Берём первые k
        уникальных документов.
        """
        best: dict[str, dict] = {}
        for h in hits:
            doc_id = h.payload["doc_id"]
            if doc_id in best:
                continue
            score = float(h.score)
            if clip:
                score = max(0.0, min(1.0, score))
            best[doc_id] = {
                "doc_id": doc_id,
                "score": score,
                "metadata": h.payload.get("metadata") or {},
            }
            if len(best) >= k:
                break
        return list(best.values())

    def update_embedding(self, doc_id: str, embedding: list[float]) -> None:
        self._client.update_vectors(
            collection_name=COLLECTION,
            points=[models.PointVectors(id=_point_id(doc_id), vector={"dense": embedding})],
        )


    def get_embeddings(self, doc_ids: list[str]) -> dict[str, list[float]]:
        """Batch-получение dense-эмбеддингов по doc_id.

        Для multi-chunk документов возвращает эмбеддинг первого чанка.
        doc_id без эмбеддинга в Qdrant не включаются в результат.
        """
        if not doc_ids:
            return {}

        result: dict[str, list[float]] = {}
        BATCH = 500
        for i in range(0, len(doc_ids), BATCH):
            batch = doc_ids[i:i + BATCH]
            # Пробуем оба варианта point_id: single-chunk (_point_id(did))
            # и multi-chunk first chunk (_point_id(did, 0))
            point_ids = []
            for did in batch:
                point_ids.append(_point_id(did))
                point_ids.append(_point_id(did, 0))
            points = self._client.retrieve(
                collection_name=COLLECTION,
                ids=point_ids,
                with_vectors=True,
                with_payload=["doc_id"],
            )
            for p in points:
                did = p.payload.get("doc_id") if p.payload else None
                if did is None or did in result:
                    continue
                vec = p.vector.get("dense") if isinstance(p.vector, dict) else p.vector
                if vec is not None:
                    result[did] = vec
        return result

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
        # Запрашиваем с запасом: несколько chunk-точек одного документа схлопнутся
        # в _dedup_hits, поэтому нужно больше кандидатов, чтобы добрать k уникальных.
        hits = self._client.query_points(
            collection_name=COLLECTION,
            query=query_embedding,
            using="dense",
            limit=max(k * 4, k),
            query_filter=to_qdrant_filter(metadata_filter),
            with_payload=True,
        ).points
        return self._dedup_hits(hits, k, clip=True)

    def bm25_search(self, query: str, k: int = 5, metadata_filter: Optional[dict] = None) -> list[dict]:
        """Sparse BM25-поиск. Возвращает [{doc_id, score, metadata}]."""
        if self.count() == 0:
            return []
        query_tokens = _tokenize(query)
        if not query_tokens:
            return []
        idf = self._idf_for_tokens(query_tokens)
        sparse = models.SparseVector(
            indices=list(idf.keys()),
            values=list(idf.values()),
        )
        if not sparse.indices:
            return []
        hits = self._client.query_points(
            collection_name=COLLECTION,
            query=sparse,
            using="bm25",
            limit=max(k * 4, k),
            query_filter=to_qdrant_filter(metadata_filter),
            with_payload=True,
        ).points
        positive = [h for h in hits if h.score > 0]
        return self._dedup_hits(positive, k)

    def stats(self) -> dict:
        return {
            "total_documents": self.count(),
            "store_path": self.location,
            "dimension": self.get_dimension() or self.dimension,
        }

    def close(self) -> None:
        self._client.close()
