"""RAG System - оркестратор для семантического и BM25 поиска."""

import uuid
from typing import Optional
from src.config import RAGConfig
from src.embeddings import OpenRouterEmbeddingGenerator, OllamaEmbeddingGenerator
from src.vector_store import VectorStore
from src.bm25_index import BM25Index
from src.graph_store import GraphKnowledgeBase


class RAGSystem:
    """RAG система с гибридным поиском (семантический + BM25)."""

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
        self._ensure_dimension_compatibility()

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

    def add_document(self, text: str, metadata: Optional[dict] = None) -> str:
        """
        Добавить документ в систему.

        Args:
            text: текст документа
            metadata: метаданные документа

        Returns:
            doc_id: сгенерированный UUID
        """
        doc_id = str(uuid.uuid4())
        
        # Получаем эмбеддинг
        embedding = self.embedding_generator.get_embedding(text)
        
        # Добавляем в VectorStore
        self.vector_store.add(doc_id=doc_id, text=text, embedding=embedding, metadata=metadata)
        
        # Добавляем в BM25Index
        self.bm25_index.add_document(doc_id=doc_id, text=text, metadata=metadata)
        
        # Добавляем в GraphKnowledgeBase
        self.graph_kb.add_node(doc_id, text, metadata)
        
        return doc_id

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

    def search(self, query: str, k: int = 5) -> list[tuple[str, str, float, dict]]:
        """
        Семантический поиск по эмбеддингу.

        Args:
            query: поисковый запрос
            k: количество результатов

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
        return self.vector_store.search(query_embedding, k=k)

    def bm25_search(self, query: str, k: int = 5) -> list[tuple[str, str, float, dict]]:
        """
        BM25 поиск по ключевым словам.

        Args:
            query: поисковый запрос
            k: количество результатов

        Returns:
            список кортежей (doc_id, text, score, metadata)
        """
        # Проверка на пустой индекс BM25
        stats = self.bm25_index.stats()
        if stats["total_documents"] == 0:
            return []
        return self.bm25_index.search(query, k=k)

    def search_hybrid(self, query: str, k: int = 5, alpha: float = 0.5) -> list[tuple[str, str, float, dict]]:
        """
        Гибридный поиск: комбинация семантического и BM25.

        score = alpha * semantic_score + (1 - alpha) * bm25_score

        Args:
            query: поисковый запрос
            k: количество результатов
            alpha: баланс (0.0 = чистый BM25, 1.0 = чистый семантический)

        Returns:
            список кортежей (doc_id, text, score, metadata)
        """
        # Проверка на пустые хранилища
        if self.vector_store.count() == 0:
            return []
        stats = self.bm25_index.stats()
        if stats["total_documents"] == 0:
            return []
        # Получаем результаты обоих поисков
        semantic_results = self.search(query, k=k)
        bm25_results = self.bm25_search(query, k=k)

        if not semantic_results and not bm25_results:
            return []

        # Нормализуем scores
        def normalize_scores(results):
            if not results:
                return {}
            scores = [r[2] for r in results]
            min_score = min(scores)
            max_score = max(scores)
            if max_score == min_score:
                return {r[0]: 1.0 for r in results}
            return {r[0]: (r[2] - min_score) / (max_score - min_score) for r in results}

        semantic_normalized = normalize_scores(semantic_results)
        bm25_normalized = normalize_scores(bm25_results)

        # Объединяем результаты
        all_doc_ids = set(semantic_normalized.keys()) | set(bm25_normalized.keys())

        # Вычисляем гибридные scores
        hybrid_scores = []
        for doc_id in all_doc_ids:
            sem_score = semantic_normalized.get(doc_id, 0.0)
            bm25_score = bm25_normalized.get(doc_id, 0.0)
            hybrid_score = alpha * sem_score + (1 - alpha) * bm25_score
            hybrid_scores.append((doc_id, hybrid_score))

        # Сортируем по убыванию score
        hybrid_scores.sort(key=lambda x: x[1], reverse=True)

        # Формируем результат с текстом и метаданными
        result = []
        for doc_id, score in hybrid_scores[:k]:
            # Ищем текст и метаданные из semantic или bm25 результатов
            text = ""
            meta = {}
            for r in semantic_results:
                if r[0] == doc_id:
                    text = r[1]
                    meta = r[3]
                    break
            if not text:
                for r in bm25_results:
                    if r[0] == doc_id:
                        text = r[1]
                        meta = r[3]
                        break
            result.append((doc_id, text, score, meta))

        return result

    def clear(self) -> None:
        """Очистить все хранилища."""
        self.vector_store.clear()
        self.bm25_index.clear()
        self.graph_kb.clear()
        self.embedding_generator.clear_cache()

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

    def get_related(self, node_id: str, max_depth: int = 1) -> list[tuple[str, str, str, float]]:
        """
        Получить связанные документы.

        Args:
            node_id: идентификатор документа
            max_depth: максимальная глубина обхода

        Returns:
            список кортежей (source_id, target_id, relation, weight)
        """
        return self.graph_kb.get_related(node_id, max_depth)

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
