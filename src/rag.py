"""RAG System — оркестратор гибридного поиска (semantic + BM25 + граф).

Архитектура хранения (двухуровневая):
  • Qdrant (embedded или HTTP)     — dense-эмбеддинги + sparse BM25
  • SQLite (WAL) или Postgres      — документы (источник правды) + рёбра графа
  • NetworkX                       — in-memory кеш графа для BFS
"""

import hashlib
import re
import uuid
from typing import Optional

from src.config import RAGConfig
from src.document_store import Database, DocumentStore
from src.embeddings import (
    BgeM3EmbeddingGenerator,
    OllamaEmbeddingGenerator,
    OpenRouterEmbeddingGenerator,
)
from src.graph_store import GraphStore
from src.vector_store import QdrantVectorStore


class RAGSystem:
    """RAG система с гибридным поиском (семантический + BM25)."""

    MIN_CONTENT_LENGTH = 50

    def __init__(
        self,
        store_path: Optional[str] = None,
        api_key: Optional[str] = None,
        config: Optional[RAGConfig] = None,
        embedding_generator=None,
    ):
        """
        Args:
            store_path: путь к директории хранилища (по умолчанию — из config);
                        внутри создаются qdrant/ и store.db, если не заданы
                        QDRANT_URL / DATABASE_URL
            api_key: OpenRouter API ключ (если передан — принудительно OpenRouter)
            config: опциональная конфигурация (если None — читается из env)
            embedding_generator: явный генератор эмбеддингов (тесты/кастомные
                        провайдеры); имеет приоритет над config
        """
        cfg = config or RAGConfig.from_env()
        if store_path:
            cfg.store_path = store_path
        self.config = cfg
        self.store_path = cfg.store_path

        if embedding_generator is not None:
            self.embedding_generator = embedding_generator
        elif api_key is not None:
            self.embedding_generator = OpenRouterEmbeddingGenerator(
                api_key=api_key, dimension=cfg.openrouter_dimension,
            )
        elif cfg.embedding_provider == "openrouter" and cfg.openrouter_api_key:
            self.embedding_generator = OpenRouterEmbeddingGenerator(
                api_key=cfg.openrouter_api_key,
                model=cfg.openrouter_model,
                dimension=cfg.openrouter_dimension,
            )
        elif cfg.embedding_provider == "ollama":
            self.embedding_generator = OllamaEmbeddingGenerator(
                base_url=cfg.ollama_base_url,
                model=cfg.ollama_model,
                dimension=cfg.ollama_dimension,
            )
        elif cfg.embedding_provider == "bge-m3":
            self.embedding_generator = BgeM3EmbeddingGenerator(
                model_name=cfg.bge_model_name,
                device=cfg.embedding_device,
                dimension=cfg.embedding_dim,
            )
        else:
            raise ValueError(
                f"Unknown embedding provider: {cfg.embedding_provider!r}. "
                f"Supported: bge-m3, ollama, openrouter."
            )

        self.db = Database(cfg.resolve_database_url())
        self.doc_store = DocumentStore(self.db)
        self.vector_store = QdrantVectorStore(
            location=cfg.resolve_qdrant_location(),
            dimension=self.embedding_generator.get_dimension(),
        )
        self.graph_kb = GraphStore(self.db, self.doc_store)

        self._default_alpha = cfg.default_alpha
        self._cyrillic_alpha = cfg.cyrillic_alpha
        self._hybrid_expand = cfg.hybrid_expand
        self._hybrid_min_candidates = cfg.hybrid_min_candidates
        self._reranker_enabled = cfg.rerank_enabled
        self._reranker_model = cfg.rerank_model
        self._reranker_device = cfg.rerank_device
        self._reranker_top_k_multiplier = cfg.rerank_top_k_multiplier
        self._reranker = None
        self._query_expansion_enabled = cfg.query_expansion_enabled
        self._query_expansion_model = cfg.query_expansion_model
        self._query_expansion_count = cfg.query_expansion_count
        self._query_expansion_ollama_url = cfg.query_expansion_ollama_url
        self._query_expander = None
        self._sync_stores()

    # -- text utils -----------------------------------------------------------

    def _normalize_text(self, text: str) -> str:
        """Normalize text for dedup: lowercase, strip, collapse whitespace."""
        return re.sub(r"\s+", " ", text.lower().strip())

    def _compute_hash(self, text: str) -> str:
        """Compute SHA-256 hash of normalized text."""
        return hashlib.sha256(self._normalize_text(text).encode()).hexdigest()

    def _get_reranker(self):
        """Lazy load reranker (CrossEncoder, ~1GB)."""
        if self._reranker is None:
            from src.reranker import Reranker
            self._reranker = Reranker(
                model_name=self._reranker_model,
                device=self._reranker_device,
            )
        return self._reranker

    def _get_query_expander(self):
        """Lazy load query expander (Ollama LLM)."""
        if self._query_expander is None:
            from src.query_expander import QueryExpander
            self._query_expander = QueryExpander(
                model=self._query_expansion_model,
                base_url=self._query_expansion_ollama_url,
                count=self._query_expansion_count,
            )
        return self._query_expander

    # -- indexing ---------------------------------------------------------------

    def add_document(
        self,
        text: str,
        metadata: Optional[dict] = None,
        doc_id: Optional[str] = None,
        extract_graph: bool = False,
        extract_graph_mode: str = "llm",
        _skip_length_check: bool = False,
    ) -> str:
        """
        Добавить документ в систему.

        Args:
            text: текст документа
            metadata: метаданные документа
            doc_id: явный идентификатор (используется миграцией для сохранения
                    старых id); по умолчанию генерируется uuid4
            extract_graph: извлечь граф (entity-relation triples) через LLM
            extract_graph_mode: "llm" (Qwen3-4B) или "ner" (spaCy)
            _skip_length_check: внутренний — пропустить проверку MIN_CONTENT_LENGTH
                                (для документов-сущностей из GraphExtractor)

        Returns:
            doc_id (или существующий id при дубликате)

        Raises:
            ValueError: если текст короче MIN_CONTENT_LENGTH
        """
        if not _skip_length_check and len(text.strip()) < self.MIN_CONTENT_LENGTH:
            raise ValueError(
                f"Document too short ({len(text.strip())} chars). "
                f"Minimum content length is {self.MIN_CONTENT_LENGTH} characters."
            )

        content_hash = self._compute_hash(text)
        existing = self.doc_store.find_by_hash(content_hash)
        if existing is not None:
            return existing

        doc_id = doc_id or str(uuid.uuid4())
        embedding = self.embedding_generator.get_embedding(text)
        self.doc_store.add(doc_id, text, metadata, content_hash=content_hash)
        self.vector_store.add(doc_id=doc_id, text=text, embedding=embedding, metadata=metadata)

        if extract_graph:
            from src.graph_extractor import GraphExtractor
            extractor = GraphExtractor(self)
            extractor.extract_and_link(doc_id, text, mode=extract_graph_mode)

        return doc_id

    def is_duplicate(self, text: str) -> Optional[str]:
        """Вернуть doc_id существующего документа с тем же content hash, иначе None."""
        return self.doc_store.find_by_hash(self._compute_hash(text))

    def add_documents(self, texts: list[str], metadata: Optional[list[dict]] = None) -> list[str]:
        doc_ids = []
        for i, text in enumerate(texts):
            meta = metadata[i] if metadata and i < len(metadata) else None
            doc_ids.append(self.add_document(text, meta))
        return doc_ids

    def add_file(
        self, filepath: str, metadata: Optional[dict] = None,
        extract_graph: bool = False, extract_graph_mode: str = "llm",
    ) -> str:
        with open(filepath, "r", encoding="utf-8-sig") as f:
            text = f.read()
        return self.add_document(text, metadata, extract_graph=extract_graph, extract_graph_mode=extract_graph_mode)

    # -- search -----------------------------------------------------------------

    def _join_texts(self, hits: list[dict]) -> list[tuple[str, str, float, dict]]:
        """Подтянуть тексты из DocumentStore к результатам поиска Qdrant."""
        docs = self.doc_store.get_batch([h["doc_id"] for h in hits])
        results = []
        for h in hits:
            doc = docs.get(h["doc_id"])
            if doc is None:
                continue  # фантом в Qdrant — уберёт следующий _sync_stores
            meta = doc["metadata"] if isinstance(doc["metadata"], dict) else {}
            results.append((h["doc_id"], doc["text"], h["score"], meta))
        return results

    def search(
        self, query: str, k: int = 5, metadata_filter: Optional[dict] = None
    ) -> list[tuple[str, str, float, dict]]:
        """Семантический поиск. Возвращает [(doc_id, text, score, metadata)]."""
        if self.vector_store.count() == 0:
            return []
        query_embedding = self.embedding_generator.get_embedding(query)
        store_dim = self.vector_store.get_dimension()
        if store_dim and len(query_embedding) != store_dim:
            return []
        # Нулевой вектор запроса не несёт семантического сигнала — пусто,
        # гибридный поиск обопрётся на BM25 (запросы-идентификаторы).
        if all(abs(v) < 1e-12 for v in query_embedding):
            return []
        hits = self.vector_store.search(query_embedding, k=k, metadata_filter=metadata_filter)
        return self._join_texts(hits)

    def bm25_search(
        self, query: str, k: int = 5, metadata_filter: Optional[dict] = None
    ) -> list[tuple[str, str, float, dict]]:
        """BM25 (sparse) поиск. Возвращает [(doc_id, text, score, metadata)]."""
        if self.vector_store.count() == 0:
            return []
        hits = self.vector_store.bm25_search(query, k=k, metadata_filter=metadata_filter)
        return self._join_texts(hits)

    def _alpha_for_query(self, query: str) -> float:
        """Language-aware alpha: кириллица → cyrillic_alpha, иначе default_alpha."""
        if any('Ѐ' <= ch <= 'ӿ' for ch in query):
            return self._cyrillic_alpha
        return self._default_alpha

    def _hybrid_search_single(
        self,
        query: str,
        k: int = 5,
        alpha: Optional[float] = None,
        metadata_filter: Optional[dict] = None,
    ) -> list[tuple[str, str, float, dict]]:
        """Одно-запросный гибридный поиск (RRF alpha-dilution).

        Возвращает [(doc_id, text, score, metadata)] — k результатов.
        """
        if alpha is None:
            alpha = self._alpha_for_query(query)
        alpha = min(1.0, max(0.0, alpha))

        if self.vector_store.count() == 0:
            return []

        candidate_k = max(k * self._hybrid_expand, self._hybrid_min_candidates)
        semantic_results = self.search(query, k=candidate_k, metadata_filter=metadata_filter)
        bm25_results = self.bm25_search(query, k=candidate_k, metadata_filter=metadata_filter)

        if not semantic_results and not bm25_results:
            return []

        RRF_K = 20
        text_by_id: dict[str, str] = {}
        meta_by_id: dict[str, dict] = {}
        rank_sem: dict[str, int] = {}
        rank_bm25: dict[str, int] = {}

        for i, r in enumerate(semantic_results):
            text_by_id.setdefault(r[0], r[1])
            meta_by_id.setdefault(r[0], r[3])
            rank_sem[r[0]] = i

        for i, r in enumerate(bm25_results):
            text_by_id.setdefault(r[0], r[1])
            meta_by_id.setdefault(r[0], r[3])
            rank_bm25[r[0]] = i

        rrf_scores: list[tuple[str, float]] = []
        for doc_id in set(rank_sem) | set(rank_bm25):
            score = 0.0
            if doc_id in rank_sem:
                score += alpha * (1.0 / (RRF_K + rank_sem[doc_id] + 1))
            if doc_id in rank_bm25:
                score += (1 - alpha) * (1.0 / (RRF_K + rank_bm25[doc_id] + 1))
            if score > 0.0:
                rrf_scores.append((doc_id, score))

        rrf_scores.sort(key=lambda x: x[1], reverse=True)
        return [
            (doc_id, text_by_id.get(doc_id, ""), score, meta_by_id.get(doc_id, {}))
            for doc_id, score in rrf_scores[:k]
        ]

    def _merge_multi_query(
        self,
        results: list[list[tuple[str, str, float, dict]]],
        k: int,
    ) -> list[tuple[str, str, float, dict]]:
        """Объединить результаты нескольких запросов через RRF.

        Каждый результат [(doc_id, text, score, metadata), ...] от одного запроса.
        """
        if not results:
            return []
        if len(results) == 1:
            return results[0][:k]

        RRF_K = 20
        text_by_id: dict[str, str] = {}
        meta_by_id: dict[str, dict] = {}
        rrf_score: dict[str, float] = {}

        for q_results in results:
            for rank, (doc_id, text, score, meta) in enumerate(q_results):
                text_by_id.setdefault(doc_id, text)
                meta_by_id.setdefault(doc_id, meta)
                rrf_score[doc_id] = rrf_score.get(doc_id, 0.0) + 1.0 / (RRF_K + rank + 1)

        merged = sorted(rrf_score.items(), key=lambda x: x[1], reverse=True)
        return [
            (doc_id, text_by_id.get(doc_id, ""), score, meta_by_id.get(doc_id, {}))
            for doc_id, score in merged[:k]
        ]

    def search_hybrid(
        self,
        query: str,
        k: int = 5,
        alpha: Optional[float] = None,
        metadata_filter: Optional[dict] = None,
        rerank: Optional[bool] = None,
        query_expansion: Optional[bool] = None,
    ) -> list[tuple[str, str, float, dict]]:
        """Гибридный поиск: RRF (alpha-dilution) семантического и BM25 каналов.

        Args:
            rerank: переранжировать финальную выдачу через CrossEncoder.
                    None = использовать config.rerank_enabled.
            query_expansion: генерировать альтернативные формулировки запроса
                             и объединять результаты через RRF.
                             None = использовать config.query_expansion_enabled.
        """
        if self.vector_store.count() == 0:
            return []

        do_expansion = (
            query_expansion if query_expansion is not None
            else self._query_expansion_enabled
        )

        if do_expansion:
            variants = self._get_query_expander().expand(query)
            all_results = []
            for v in variants:
                qr = self._hybrid_search_single(v, k=k, alpha=alpha, metadata_filter=metadata_filter)
                if qr:
                    all_results.append(qr)
            merged = self._merge_multi_query(all_results, k=k)
        else:
            merged = self._hybrid_search_single(query, k=k, alpha=alpha, metadata_filter=metadata_filter)

        if not merged:
            return []

        do_rerank = rerank if rerank is not None else self._reranker_enabled
        if do_rerank and len(merged) > 1:
            top_k = k
            candidate_k = min(len(merged), k * self._reranker_top_k_multiplier)
            candidates = [
                {"doc_id": doc_id, "text": text, "score": score, "metadata": meta}
                for doc_id, text, score, meta in merged[:candidate_k]
            ]
            reranked = self._get_reranker().rerank(query, candidates, top_k=top_k)
            merged = [
                (d["doc_id"], d["text"], d["rerank_score"], d["metadata"])
                for d in reranked
            ]

        return merged

    # -- retrieval / management ---------------------------------------------------

    def delete_document(self, doc_id: str) -> bool:
        """Удалить документ из всех хранилищ. Идемпотентен."""
        existed = self.doc_store.delete(doc_id)  # FK CASCADE удаляет рёбра
        self.vector_store.remove(doc_id)
        self.graph_kb.delete_document(doc_id)
        return existed

    def get_document(
        self,
        doc_id: str,
        offset: int = 0,
        limit: Optional[int] = None,
        relations_load_depth: int = 1,
        relations_load_type_filter: Optional[list[str]] = None,
        relations_load_meta_filter: Optional[dict] = None,
    ) -> Optional[dict]:
        """Документ по ID с пагинацией текста и загрузкой links."""
        record = self.doc_store.get(doc_id)
        if record is None:
            return None
        full_text = record["text"]
        total_chars = len(full_text)
        text = full_text[offset:offset + limit] if limit is not None else full_text[offset:]
        doc = {
            "doc_id": doc_id,
            "text": text,
            "metadata": record["metadata"],
            "total_chars": total_chars,
            "offset": offset,
            "limit": limit,
        }
        self._enrich_with_links(
            [doc],
            relations_load_depth=relations_load_depth,
            relations_load_type_filter=relations_load_type_filter,
            relations_load_meta_filter=relations_load_meta_filter,
        )
        return doc

    def list_documents(
        self,
        limit: int = 20,
        offset: int = 0,
        max_chars: Optional[int] = None,
        metadata_filter: Optional[dict] = None,
        relations_load_depth: int = 1,
        relations_load_type_filter: Optional[list[str]] = None,
        relations_load_meta_filter: Optional[dict] = None,
    ) -> dict:
        """Постраничный список документов (см. прежний контракт)."""
        items, total = self.doc_store.list(limit=limit, offset=offset, metadata_filter=metadata_filter)
        documents = []
        for record in items:
            text = record["text"][:max_chars] if max_chars is not None else record["text"]
            documents.append({"doc_id": record["doc_id"], "text": text, "metadata": record["metadata"]})
        self._enrich_with_links(
            documents,
            relations_load_depth=relations_load_depth,
            relations_load_type_filter=relations_load_type_filter,
            relations_load_meta_filter=relations_load_meta_filter,
        )
        return {"documents": documents, "total": total, "limit": limit, "offset": offset}

    def _enrich_with_links(
        self,
        docs: list[dict],
        relations_load_depth: int = 1,
        relations_load_type_filter: Optional[list[str]] = None,
        relations_load_meta_filter: Optional[dict] = None,
    ) -> list[dict]:
        """Attach graph relations as `links` field to each document dict."""
        links_field: dict[str, dict[str, list[dict]]] = {}
        if relations_load_depth > 0:
            doc_ids = {d["doc_id"] for d in docs if "doc_id" in d}
            links_field = self.graph_kb.get_edges_batch(
                doc_ids,
                max_depth=relations_load_depth,
                relation_type_filter=relations_load_type_filter,
                metadata_filter=relations_load_meta_filter,
            )
        for d in docs:
            d["links"] = links_field.get(d.get("doc_id", ""), {})
        return docs

    def clear(self) -> None:
        """Очистить все хранилища."""
        self.graph_kb.delete_all()
        self.doc_store.clear()
        self.vector_store.clear()
        self.embedding_generator.clear_cache()

    def add_relation(self, source_id: str, target_id: str, relation: str, weight: float = 1.0) -> None:
        self.graph_kb.add_edge(source_id, target_id, relation, weight)

    def get_related(
        self,
        node_id: str,
        max_depth: int = 1,
        direction: str = "both",
        metadata_filter: Optional[dict] = None,
    ) -> list[tuple[str, str, str, float, str]]:
        return self.graph_kb.get_related(node_id, max_depth, direction, metadata_filter=metadata_filter)

    def index_structured(self, content: str, extract_graph: bool = False) -> dict:
        """Индексация repomix-формата (структурированный код).

        Args:
            content: JSON-строка в формате repomix
            extract_graph: извлекать граф из каждого файла

        Returns:
            {status, structure_doc_id, file_doc_ids: dict[path→doc_id],
             files_count, errors}
        """
        from src.structured_indexer import StructuredIndexer
        return StructuredIndexer(self, extract_graph=extract_graph).index(content)

    # -- maintenance ---------------------------------------------------------------

    def _sync_stores(self) -> None:
        """Синхронизировать хранилища: DocumentStore — источник правды.

        Удаляет фантомные точки из Qdrant (нет документа) и переиндексирует
        документы, отсутствующие в Qdrant (например, после сбоя записи).
        """
        try:
            doc_ids = self.doc_store.all_ids()
            vector_ids = self.vector_store.get_all_ids()

            for phantom in vector_ids - doc_ids:
                self.vector_store.remove(phantom)

            missing = doc_ids - vector_ids
            for doc_id in missing:
                record = self.doc_store.get(doc_id)
                if record is None:
                    continue
                embedding = self.embedding_generator.get_embedding(record["text"])
                self.vector_store.add(
                    doc_id=doc_id, text=record["text"],
                    embedding=embedding, metadata=record["metadata"],
                )

            phantom_count = len(vector_ids - doc_ids)
            if phantom_count or missing:
                print(f"[sync] Removed {phantom_count} phantom vectors, reindexed {len(missing)} docs")
        except Exception as e:
            print(f"[sync] Warning: store sync failed: {e}")

    def reindex(self) -> int:
        """Пересчитать эмбеддинги всех документов (смена провайдера/модели)."""
        items, total = self.doc_store.list(limit=self.doc_store.count() or 1, offset=0)
        if not items:
            return 0

        self.embedding_generator.clear_cache()
        first = items[0]
        first_embedding = self.embedding_generator.get_embedding(first["text"])
        new_dim = len(first_embedding)
        old_dim = self.vector_store.get_dimension()

        if old_dim and old_dim != new_dim:
            self.vector_store.dimension = new_dim
            self.vector_store.recreate_collection()

        self.vector_store.add(
            doc_id=first["doc_id"], text=first["text"],
            embedding=first_embedding, metadata=first["metadata"],
        )
        for record in items[1:]:
            embedding = self.embedding_generator.get_embedding(record["text"])
            self.vector_store.add(
                doc_id=record["doc_id"], text=record["text"],
                embedding=embedding, metadata=record["metadata"],
            )
        return len(items)

    def stats(self) -> dict:
        graph_stats = self.graph_kb.stats()
        return {
            "total_documents": self.doc_store.count(),
            "store_path": self.store_path,
            "dimension": self.vector_store.get_dimension() or self.vector_store.dimension,
            "total_nodes": graph_stats["total_nodes"],
            "total_edges": graph_stats["total_edges"],
            "relation_types": graph_stats["relation_types"],
        }

    def close(self) -> None:
        """Освободить ресурсы (файловые локи Qdrant embedded и SQLite)."""
        try:
            self.vector_store.close()
        finally:
            try:
                close_gen = getattr(self.embedding_generator, "close", None)
                if close_gen:
                    close_gen()
            finally:
                self.db.close()
