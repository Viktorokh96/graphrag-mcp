"""VectorStore на базе ChromaDB."""

from typing import Optional
import chromadb


class VectorStore:
    """Хранилище векторов на базе ChromaDB."""

    def __init__(self, store_path: str = "./rag_data"):
        self.store_path = store_path
        # Создаем persistent клиент ChromaDB
        self._client = chromadb.PersistentClient(path=store_path)
        # Создаем или получаем коллекцию
        self.collection = self._client.get_or_create_collection(name="rag_docs")

    def add(self, doc_id: str, text: str, embedding: list[float], metadata: Optional[dict] = None) -> str:
        """
        Добавить документ в хранилище.

        Args:
            doc_id: идентификатор документа
            text: текст документа
            embedding: вектор эмбеддинга
            metadata: метаданные

        Returns:
            doc_id
        """
        self.collection.add(
            ids=[doc_id],
            documents=[text],
            embeddings=[embedding],
            metadatas=[metadata] if metadata else None
        )
        return doc_id

    def search(self, query_embedding: list[float], k: int = 5) -> list[tuple[str, str, float, dict]]:
        """
        Поиск документов по эмбеддингу.

        Args:
            query_embedding: вектор запроса
            k: количество результатов

        Returns:
            Список кортежей (doc_id, text, score, metadata), отсортированных по убыванию score
        """
        # Проверка на пустую коллекцию
        if self.count() == 0:
            return []
        result = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=k,
            include=["documents", "distances", "metadatas"]
        )

        # ChromaDB возвращает distances как расстояния, преобразуем в scores (чем меньше расстояние, тем лучше)
        # Используем 1 / (1 + distance) для нормализации
        results = []
        if result["ids"] and result["ids"][0]:
            for i, doc_id in enumerate(result["ids"][0]):
                text = result["documents"][0][i] if result["documents"] and result["documents"][0] else ""
                distance = result["distances"][0][i] if result["distances"] and result["distances"][0] else 0.0
                score = 1.0 / (1.0 + distance)
                meta = result["metadatas"][0][i] if result["metadatas"] and result["metadatas"][0] else {}
                results.append((doc_id, text, score, meta))

        return results

    def clear(self) -> None:
        """Очистить хранилище."""
        self.collection.delete(where={"__id": "dummy"})  # Удаляем все документы
        # Альтернатива: удалить и создать заново
        self._client.delete_collection("rag_docs")
        self.collection = self._client.create_collection(name="rag_docs")

    def get_dimension(self) -> int:
        """Получить размерность эмбеддингов в коллекции (0 если пусто)."""
        if self.count() == 0:
            return 0
        result = self.collection.get(limit=1, include=["embeddings"])
        # Явная проверка: ChromaDB может вернуть numpy array, и проверка `if array:` падает с ValueError
        if result.get("embeddings") is not None and len(result["embeddings"]) > 0 and result["embeddings"][0] is not None:
            return len(result["embeddings"][0])
        return 0

    def stats(self) -> dict:
        """Получить статистику хранилища."""
        return {
            "total_documents": self.count(),
            "store_path": self.store_path,
            "dimension": self.get_dimension()
        }

    def remove(self, doc_id: str) -> None:
        """Удалить документ из хранилища."""
        self.collection.delete(ids=[doc_id])

    def count(self) -> int:
        """Получить количество документов."""
        return self.collection.count()

    def list_documents(self, limit: int = 20, offset: int = 0) -> tuple[list[tuple[str, str, dict]], int]:
        """
        Получить страницу документов с пагинацией.

        Args:
            limit: количество документов на странице
            offset: сдвиг от начала

        Returns:
            Кортеж (список кортежей (doc_id, text, metadata), total_count)
        """
        total = self.count()
        result = self.collection.get(
            limit=limit,
            offset=offset,
            include=["documents", "metadatas"],
        )
        ids = result.get("ids") or []
        docs = result.get("documents") or []
        metas = result.get("metadatas") or []
        items = [
            (ids[i], docs[i], (metas[i] if metas and i < len(metas) and metas[i] is not None else {}))
            for i in range(len(ids))
        ]
        return items, total

    def get_all_texts(self) -> list[str]:
        """Получить все тексты документов."""
        result = self.collection.get(include=["documents"])
        return result["documents"] if result["documents"] else []

    def get_all(self) -> list[tuple[str, str, dict]]:
        result = self.collection.get(include=["documents", "metadatas"])
        ids = result.get("ids") or []
        docs = result.get("documents") or []
        metas = result.get("metadatas") or []
        return [
            (ids[i], docs[i], (metas[i] if metas and i < len(metas) and metas[i] is not None else {}))
            for i in range(len(ids))
        ]

    def update_embedding(self, doc_id: str, embedding: list[float]) -> None:
        self.collection.update(ids=[doc_id], embeddings=[embedding])

    def recreate_collection(self) -> None:
        try:
            self._client.delete_collection("rag_docs")
        except Exception:
            pass
        self.collection = self._client.create_collection(name="rag_docs")
