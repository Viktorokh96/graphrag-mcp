"""RAG System - оркестратор для семантического и BM25 поиска."""

import hashlib
import re
import uuid
from typing import Optional
from src.config import RAGConfig
from src.embeddings import OpenRouterEmbeddingGenerator, OllamaEmbeddingGenerator
from src.vector_store import VectorStore
from src.bm25_index import BM25Index
from src.graph_store import GraphKnowledgeBase
from src._meta_filter import to_chroma_where


class RAGSystem:
    """RAG система с гибридным поиском (семантический + BM25)."""

    MIN_CONTENT_LENGTH = 50

    def __init__(
        self,
        store_path: Optional[str] = None,
        api_key: Optional[str] = None,
        graph: Optional[GraphKnowledgeBase] = None,
        config: Optional[RAGConfig] = None,
    ):
        """
        Инициализация RAG системы.

        Args:
            store_path: путь к хранилищу векторов (по умолчанию — из config)
            api_key: OpenRouter API ключ (если передан — принудительно используется OpenRouter,
                     даже если config говорит ollama)
            graph: опциональная графовая база знаний
            config: опциональная конфигурация (если None — читается из env)
        """
        cfg = config or RAGConfig.from_env()
        self.store_path = store_path or cfg.store_path

        if api_key is not None:
            self.embedding_generator = OpenRouterEmbeddingGenerator(api_key=api_key)
        elif cfg.embedding_provider == "openrouter" and cfg.openrouter_api_key:
            self.embedding_generator = OpenRouterEmbeddingGenerator(
                api_key=cfg.openrouter_api_key,
                model=cfg.openrouter_model,
            )
        else:
            self.embedding_generator = OllamaEmbeddingGenerator(
                base_url=cfg.ollama_base_url,
                model=cfg.ollama_model,
                dimension=cfg.ollama_dimension,
            )

        self.vector_store = VectorStore(store_path=self.store_path)
        self.bm25_index = BM25Index(store_path=self.store_path)
        self.graph_kb = graph or GraphKnowledgeBase(store_path=self.store_path)
        self._default_alpha = cfg.default_alpha
        self._cyrillic_alpha = cfg.cyrillic_alpha
        self._hybrid_expand = cfg.hybrid_expand
        self._hybrid_min_candidates = cfg.hybrid_min_candidates
        self._content_hashes: dict[str, str] = {}
        self._ensure_dimension_compatibility()
        self._sync_stores()

    def _ensure_dimension_compatibility(self) -> None:
        """Проверить совместимость размерности эмбеддингов и хранилища.

        Если размерность генератора не совпадает с хранилищем (например,
        провайдер сменился с OpenRouter 1536 → Ollama 4096) — запускает
        reindex() чтобы пересоздать коллекцию с правильной размерностью.
        """
        store_dim = self.vector_store.get_dimension()
        if store_dim == 0:
            return
        gen_dim = self.embedding_generator.get_dimension()
        if gen_dim != store_dim:
            self.reindex()

    def _normalize_text(self, text: str) -> str:
        """Normalize text for dedup: lowercase, strip, collapse whitespace."""
        return re.sub(r'\s+', ' ', text.lower().strip())

    def _compute_hash(self, text: str) -> str:
        """Compute SHA-256 hash of normalized text."""
        return hashlib.sha256(self._normalize_text(text).encode()).hexdigest()

    def _rebuild_content_hashes(self) -> None:
        """Rebuild _content_hashes from all documents in vector store."""
        self._content_hashes.clear()
        all_docs = self.vector_store.get_all()
        for doc_id, text, _meta in all_docs:
            h = self._compute_hash(text)
            self._content_hashes[h] = doc_id

    def add_document(self, text: str, metadata: Optional[dict] = None) -> str:
        """
        Добавить документ в систему.

        Args:
            text: текст документа
            metadata: метаданные документа

        Returns:
            doc_id: сгенерированный UUID (или существующий при дубликате)

        Raises:
            ValueError: если текст короче MIN_CONTENT_LENGTH
        """
        # D5: Проверка минимальной длины
        if len(text.strip()) < self.MIN_CONTENT_LENGTH:
            raise ValueError(
                f"Document too short ({len(text.strip())} chars). "
                f"Minimum content length is {self.MIN_CONTENT_LENGTH} characters."
            )

        # D6: Проверка дубликатов по хэшу содержимого
        content_hash = self._compute_hash(text)
        if content_hash in self._content_hashes:
            return self._content_hashes[content_hash]

        doc_id = str(uuid.uuid4())

        # Получаем эмбеддинг
        embedding = self.embedding_generator.get_embedding(text)

        # Добавляем в VectorStore
        self.vector_store.add(doc_id=doc_id, text=text, embedding=embedding, metadata=metadata)

        # Добавляем в BM25Index
        self.bm25_index.add_document(doc_id=doc_id, text=text, metadata=metadata)

        # Добавляем в GraphKnowledgeBase
        self.graph_kb.add_node(doc_id, text, metadata)

        # D6: Сохраняем хэш
        self._content_hashes[content_hash] = doc_id

        return doc_id

    def is_duplicate(self, text: str) -> Optional[str]:
        """Check if a document with the same content hash already exists.

        Returns the existing doc_id if duplicate, None otherwise.
        """
        content_hash = self._compute_hash(text)
        return self._content_hashes.get(content_hash)

    def add_documents(self, texts: list[str], metadata: Optional[list[dict]] = None) -> list[str]:
        """
        Добавить несколько документов.

        Args:
            texts: список текстов
            metadata: список метаданных (или None)

        Returns:
            список doc_id
        """
        doc_ids = []
        for i, text in enumerate(texts):
            meta = metadata[i] if metadata and i < len(metadata) else None
            doc_id = self.add_document(text, meta)
            doc_ids.append(doc_id)
        return doc_ids

    def add_file(self, filepath: str, metadata: Optional[dict] = None) -> str:
        """
        Прочитать файл и добавить его содержимое как документ.

        Args:
            filepath: путь к файлу
            metadata: метаданные

        Returns:
            doc_id
        """
        with open(filepath, 'r', encoding='utf-8') as f:
            text = f.read()
        return self.add_document(text, metadata)

    def search(self, query: str, k: int = 5, metadata_filter: Optional[dict] = None) -> list[tuple[str, str, float, dict]]:
        """
        Семантический поиск по эмбеддингу.

        Args:
            query: поисковый запрос
            k: количество результатов
            metadata_filter: опциональный фильтр по метаданным
                ({key: scalar | list[scalar]}, AND-комбинация). None/{} — без фильтра.

        Returns:
            список кортежей (doc_id, text, score, metadata)
        """
        # Проверка на пустое хранилище
        if self.vector_store.count() == 0:
            return []
        query_embedding = self.embedding_generator.get_embedding(query)
        # Проверка размерности: если не совпадает — возвращаем пустой результат
        # (BM25 поиск продолжает работать независимо)
        store_dim = self.vector_store.get_dimension()
        if store_dim and len(query_embedding) != store_dim:
            return []
        # Нулевой вектор запроса (все слова неизвестны эмбеддинг-модели) не несёт
        # семантического сигнала: все документы равноудалены. Возвращаем пусто,
        # чтобы гибридный поиск опирался только на BM25 — это спасает запросы по
        # идентификаторам (pytest, jwt, bm25), которые семантика не видит.
        if all(abs(v) < 1e-12 for v in query_embedding):
            return []
        where = to_chroma_where(metadata_filter)
        results = self.vector_store.search(query_embedding, k=k, where=where)
        # D10: нормализация metadata — всегда dict
        return [
            (doc_id, text, score, meta if isinstance(meta, dict) else {})
            for doc_id, text, score, meta in results
        ]

    def bm25_search(self, query: str, k: int = 5, metadata_filter: Optional[dict] = None) -> list[tuple[str, str, float, dict]]:
        """
        BM25 поиск по ключевым словам.

        Args:
            query: поисковый запрос
            k: количество результатов
            metadata_filter: опциональный фильтр по метаданным
                ({key: scalar | list[scalar]}, AND-комбинация). None/{} — без фильтра.

        Returns:
            список кортежей (doc_id, text, score, metadata)
        """
        # Проверка на пустой индекс BM25
        stats = self.bm25_index.stats()
        if stats["total_documents"] == 0:
            return []
        results = self.bm25_index.search(query, k=k, metadata_filter=metadata_filter)
        # D10: нормализация metadata — всегда dict
        return [
            (doc_id, text, score, meta if isinstance(meta, dict) else {})
            for doc_id, text, score, meta in results
        ]

    def _alpha_for_query(self, query: str) -> float:
        """Выбрать alpha в зависимости от языка запроса.

        Для запросов с кириллицей используется cyrillic_alpha (0.85): BM25 без
        русского стемминга даёт шумовый сигнал, поэтому семантический канал
        должен доминировать. Для остальных — default_alpha (0.5, баланс каналов).
        """
        if any('\u0400' <= ch <= '\u04ff' for ch in query):
            return self._cyrillic_alpha
        return self._default_alpha

    def search_hybrid(self, query: str, k: int = 5, alpha: Optional[float] = None,
                      metadata_filter: Optional[dict] = None) -> list[tuple[str, str, float, dict]]:
        """Гибридный поиск: Reciprocal Rank Fusion (RRF) семантического и BM25.

        Метод RRF (Reciprocal Rank Fusion) заменяет старую min-max нормализацию
        с линейной комбинацией. RRF устойчив к разным шкалам скоров, не требует
        нормализации и гарантирует отсутствие тай-оффов (ранги всегда различны).

        Формула (alpha-dilution): каждый канал взвешивается своей долей alpha.
          • both-channels:  score = alpha/(RRF_K+rank_sem+1) + (1-alpha)/(RRF_K+rank_bm25+1)
          • sem-only:       score = alpha/(RRF_K+rank_sem+1)         (вес alpha)
          • bm25-only:      score = (1-alpha)/(RRF_K+rank_bm25+1)    (вес 1-alpha)
        Это означает, что alpha=1.0 → чистая семантика (BM25-only доки исключаются),
        alpha=0.0 → чистый BM25 (sem-only доки исключаются). Документ, найденный
        обоими каналами, получает сумму взвешенных reciprocal ranks и потому
        ранжируется выше документа, найденного только одним каналом — это
        предотвращает засорение выдачи шумом слабого канала.

        Улучшения:
          • candidate expansion — из каждого источника забирается `max(k*3, 20)`
            кандидатов (а не k), что спасает документы, релевантные по одному
            каналу, но оказавшиеся за пределами top-k по другому.
          • RRF_K=20 (вместо классической 60) — для малых корпусов даёт широкий
            разброс скоров (дискриминация рангов 1..5), тогда как k=60 сжимает
            все скоры в узкую полосу ~7% и лишает выдачу различительной силы.
          • alpha по умолчанию берётся из RAGConfig.default_alpha (его значение
            выбрано по результатам бенчмарка NDCG@k, см. scripts/benchmark_alpha.py).
          • Language-aware alpha: для запросов с кириллицей используется
            cyrillic_alpha (0.85 по умолчанию) вместо default_alpha (0.5).
            BM25 без русского стемминга даёт шумовый сигнал для русских запросов,
            поэтому семантический канал должен доминировать. Явно переданный alpha
            имеет приоритет над language-aware выбором.

        Args:
            query: поисковый запрос
            k: количество результатов в выдаче
            alpha: баланс (None=language-aware default, 0.0 = чистый BM25, 1.0 = чистый семантический)
            metadata_filter: опциональный фильтр по метаданным
                ({key: scalar | list[scalar]}, AND-комбинация). None/{} — без фильтра.
                Применяется к обоим каналам (semantic + BM25) до fusion.

        Returns:
            список кортежей (doc_id, text, score, metadata), отсортированных по убыванию score
        """
        if alpha is None:
            alpha = self._alpha_for_query(query)
        # Защита от невалидных значений alpha
        if alpha < 0.0:
            alpha = 0.0
        elif alpha > 1.0:
            alpha = 1.0

        # Проверка на пустые хранилища
        if self.vector_store.count() == 0:
            return []
        stats = self.bm25_index.stats()
        if stats["total_documents"] == 0:
            return []

        # Candidate expansion: забираем больше кандидатов из каждого канала,
        # чтобы при fusion не потерять документ, релевантный по одному каналу,
        # но оказавшийся за пределами top-k по другому.
        candidate_k = max(k * self._hybrid_expand, self._hybrid_min_candidates)

        semantic_results = self.search(query, k=candidate_k, metadata_filter=metadata_filter)
        bm25_results = self.bm25_search(query, k=candidate_k, metadata_filter=metadata_filter)

        if not semantic_results and not bm25_results:
            return []

        # Reciprocal Rank Fusion с alpha-dilution.
        # RRF_K=20: для типичных корпусов (десятки-сотни доков) даёт разброс
        # скоров ~0.02..0.05 между рангами 1 и 5, обеспечивая различимость
        # результатов. Классическая k=60 сжимает разброс до <1%, делая скоры
        # неразличимыми. Alpha-dilution: каждый канал взвешивается своей долей,
        # поэтому docs из обоих каналов ранжируются выше single-channel docs.
        RRF_K = 20

        # Словари текст/мета для быстрого доступа
        text_by_id: dict[str, str] = {}
        meta_by_id: dict[str, dict] = {}
        rank_sem: dict[str, int] = {}  # 0-based rank
        rank_bm25: dict[str, int] = {}

        for i, r in enumerate(semantic_results):
            text_by_id.setdefault(r[0], r[1])
            meta_by_id.setdefault(r[0], r[3])
            rank_sem[r[0]] = i  # 0 = best

        for i, r in enumerate(bm25_results):
            text_by_id.setdefault(r[0], r[1])
            meta_by_id.setdefault(r[0], r[3])
            rank_bm25[r[0]] = i

        all_doc_ids = set(rank_sem.keys()) | set(rank_bm25.keys())

        # Вычисляем RRF scores с alpha-dilution
        rrf_scores: list[tuple[str, float]] = []
        for doc_id in all_doc_ids:
            in_sem = doc_id in rank_sem
            in_bm25 = doc_id in rank_bm25
            score = 0.0
            if in_sem:
                # семантический канал всегда взвешивается alpha
                score += alpha * (1.0 / (RRF_K + rank_sem[doc_id] + 1))
            if in_bm25:
                # BM25 канал всегда взвешивается (1-alpha)
                score += (1 - alpha) * (1.0 / (RRF_K + rank_bm25[doc_id] + 1))
            # Документы с нулевым score (single-channel при крайнем alpha)
            # исключаются — это делает alpha=1.0 чистой семантикой, alpha=0.0 —
            # чистым BM25, без «призрачных» результатов с score=0.
            if score > 0.0:
                rrf_scores.append((doc_id, score))

        # Сортируем по убыванию score
        rrf_scores.sort(key=lambda x: x[1], reverse=True)

        # Формируем выдачу
        result = []
        for doc_id, score in rrf_scores[:k]:
            result.append((doc_id, text_by_id.get(doc_id, ""), score, meta_by_id.get(doc_id, {})))

        return result

    def delete_document(self, doc_id: str) -> bool:
        """
        Удалить документ из всех хранилищ (vector, BM25, graph).

        Идемпотентная операция: удаление несуществующего doc_id не вызывает ошибку.

        Args:
            doc_id: идентификатор документа

        Returns:
            True если документ существовал и был удалён, иначе False
        """
        # Проверяем, существует ли документ
        vector_doc = self.vector_store.get_by_id(doc_id)
        existed = vector_doc is not None

        self.vector_store.remove(doc_id)
        self.bm25_index.remove(doc_id)
        self.graph_kb.remove_node(doc_id)

        # D6: Удаляем хэш содержимого
        if existed:
            h = self._compute_hash(vector_doc[1])  # type: ignore[union-attr]
            self._content_hashes.pop(h, None)

        return existed

    def get_document(self, doc_id: str, offset: int = 0, limit: Optional[int] = None) -> Optional[dict]:
        """Получить один документ по ID с пагинацией текста.

        Args:
            doc_id: идентификатор документа
            offset: символьный сдвиг начала текста (по умолчанию 0)
            limit: максимальное количество символов текста (None = весь остаток)

        Returns:
            dict с ключами: doc_id, text, metadata, total_chars, offset, limit
            None если документ не найден
        """
        result = self.vector_store.get_by_id(doc_id)
        if result is None:
            return None
        _, full_text, meta = result
        total_chars = len(full_text)
        if limit is not None:
            text = full_text[offset:offset + limit]
        else:
            text = full_text[offset:]
        # D10: metadata всегда словарь
        meta = meta if isinstance(meta, dict) else {}
        return {
            "doc_id": doc_id,
            "text": text,
            "metadata": meta,
            "total_chars": total_chars,
            "offset": offset,
            "limit": limit,
        }

    def list_documents(self, limit: int = 20, offset: int = 0, max_chars: Optional[int] = None,
                       metadata_filter: Optional[dict] = None) -> dict:
        """
        Получить список документов с пагинацией.

        Args:
            limit: количество документов на странице (по умолчанию 20)
            offset: сдвиг от начала (по умолчанию 0)
            max_chars: ограничение длины текста каждого документа
                       (None = полный текст, иначе обрезается до max_chars символов)
            metadata_filter: опциональный фильтр по метаданным
                ({key: scalar | list[scalar]}, AND-комбинация). None/{} — без фильтра.
                `total` при активном фильтре отражает число подходящих документов.

        Returns:
            dict с ключами:
                documents: список {doc_id, text, metadata}
                total: общее количество документов (с учётом фильтра)
                limit: текущий limit
                offset: текущий offset
        """
        where = to_chroma_where(metadata_filter)
        items, total = self.vector_store.list_documents(limit=limit, offset=offset, where=where)
        documents = []
        for doc_id, text, meta in items:
            if max_chars is not None:
                text = text[:max_chars]
            # D10: metadata всегда словарь
            meta = meta if isinstance(meta, dict) else {}
            documents.append({"doc_id": doc_id, "text": text, "metadata": meta})
        return {
            "documents": documents,
            "total": total,
            "limit": limit,
            "offset": offset,
        }

    def clear(self) -> None:
        """Очистить все хранилища."""
        self.vector_store.clear()
        self.bm25_index.clear()
        self.graph_kb.clear()
        self.embedding_generator.clear_cache()
        self._content_hashes.clear()

    def add_relation(self, source_id: str, target_id: str, relation: str, weight: float = 1.0) -> None:
        """
        Добавить отношение между двумя документами.

        Args:
            source_id: идентификатор исходного документа
            target_id: идентификатор целевого документа
            relation: тип отношения
            weight: вес отношения
        """
        self.graph_kb.add_edge(source_id, target_id, relation, weight)

    def get_related(self, node_id: str, max_depth: int = 1, direction: str = "both",
                    metadata_filter: Optional[dict] = None) -> list[tuple[str, str, str, float, str]]:
        """
        Получить связанные документы (двунаправленный BFS).

        Args:
            node_id: идентификатор документа
            max_depth: максимальная глубина обхода
            direction: "out" | "in" | "both" (по умолчанию "both")
            metadata_filter: опциональный фильтр по метаданным соседних узлов
                ({key: scalar | list[scalar]}, AND-комбинация). None/{} — без фильтра.
                Соседние узлы, не проходящие фильтр, исключаются из выдачи.

        Returns:
            список кортежей (source_id, target_id, relation, weight, direction)
        """
        return self.graph_kb.get_related(node_id, max_depth, direction, metadata_filter=metadata_filter)

    def _sync_stores(self) -> None:
        """Synchronise all stores to have exactly the same set of doc_ids.

        Vector store (ChromaDB) is the source of truth.
        Removes phantom entries from BM25 and graph stores.
        """
        try:
            # Get all doc_ids from vector store (source of truth)
            vector_ids = self.vector_store.get_all_ids()

            # Remove phantom BM25 docs (not in vector store)
            bm25_ids = self.bm25_index.get_all_doc_ids()
            phantom_bm25 = bm25_ids - vector_ids
            for doc_id in phantom_bm25:
                self.bm25_index.remove(doc_id)

            # Remove phantom graph nodes (not in vector store)
            graph_ids = self.graph_kb.get_all_node_ids()
            phantom_graph = graph_ids - vector_ids
            for node_id in phantom_graph:
                self.graph_kb.remove_node(node_id)

            # Remove phantom edges (edges referencing non-existent nodes)
            if vector_ids:
                self.graph_kb.remove_phantom_edges(vector_ids)

            # Rebuild content hashes for dedup
            self._rebuild_content_hashes()

            if phantom_bm25 or phantom_graph:
                print(
                    f"[sync] Removed {len(phantom_bm25)} phantom BM25 docs, "
                    f"{len(phantom_graph)} phantom graph nodes"
                )
        except Exception as e:
            print(f"[sync] Warning: store sync failed: {e}")

    def reindex(self) -> int:
        docs = self.vector_store.get_all()
        if not docs:
            return 0

        # Очищаем кеш эмбеддингов — старые векторы могут иметь другую размерность
        self.embedding_generator.clear_cache()

        # Генерируем первый эмбеддинг чтобы узнать РЕАЛЬНУЮ размерность от API
        # (а не из конфига — может не совпадать)
        first_doc = docs[0]
        first_embedding = self.embedding_generator.get_embedding(first_doc[1])
        new_dim = len(first_embedding)
        old_dim = self.vector_store.get_dimension()

        if old_dim and old_dim != new_dim:
            self.vector_store.recreate_collection()

        # Сохраняем первый эмбеддинг
        if old_dim and old_dim != new_dim:
            self.vector_store.add(doc_id=first_doc[0], text=first_doc[1], embedding=first_embedding, metadata=first_doc[2])
        else:
            self.vector_store.update_embedding(first_doc[0], first_embedding)

        # Остальные документы
        for doc_id, text, meta in docs[1:]:
            embedding = self.embedding_generator.get_embedding(text)
            if old_dim and old_dim != new_dim:
                self.vector_store.add(doc_id=doc_id, text=text, embedding=embedding, metadata=meta)
            else:
                self.vector_store.update_embedding(doc_id, embedding)

        # D6: rebuild content hashes after reindex
        self._rebuild_content_hashes()

        return len(docs)

    def stats(self) -> dict:
        """
        Получить статистику системы.

        Returns:
            dict с total_documents, store_path, dimension, total_nodes, total_edges, relation_types
        """
        vector_stats = self.vector_store.stats()
        graph_stats = self.graph_kb.stats()
        vector_stats["total_nodes"] = graph_stats["total_nodes"]
        vector_stats["total_edges"] = graph_stats["total_edges"]
        vector_stats["relation_types"] = graph_stats["relation_types"]
        return vector_stats
