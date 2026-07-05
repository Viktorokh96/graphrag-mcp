"""BM25 индекс для keyword-based поиска."""

import json
import re
from pathlib import Path
from typing import Optional
from rank_bm25 import BM25Okapi


class BM25Index:
    """BM25 индекс для поиска по ключевым словам с поддержкой персистентности."""

    def __init__(self, store_path: Optional[str] = None):
        """
        Инициализация BM25 индекса.

        Args:
            store_path: путь к директории для сохранения индекса (опционально)
        """
        self._bm25: Optional[BM25Okapi] = None
        self._documents: list[list[str]] = []
        self._texts: dict[str, str] = {}
        self._metadata: dict[str, dict] = {}
        self._store_path = store_path
        
        # Загружаем индекс с диска если store_path указан
        if store_path:
            self.load()

    def _tokenize(self, text: str) -> list[str]:
        """
        Токенизация текста: нижний регистр, только слова.

        Args:
            text: текст для токенизации

        Returns:
            список токенов
        """
        text = text.lower()
        words = re.findall(r'\b\w+\b', text)
        return words

    def add_document(self, doc_id: str, text: str, metadata: Optional[dict] = None) -> None:
        """
        Добавить один документ в индекс.

        Args:
            doc_id: идентификатор документа
            text: текст документа
            metadata: метаданные документа
        """
        tokens = self._tokenize(text)
        self._texts[doc_id] = text
        self._metadata[doc_id] = metadata or {}
        self._documents.append(tokens)
        self._bm25 = BM25Okapi(self._documents)
        
        # Сохраняем индекс на диск если store_path указан
        if self._store_path:
            self.save()

    def add_documents(self, documents: dict[str, str], metadata: Optional[dict[str, dict]] = None) -> None:
        """
        Добавить несколько документов в индекс.

        Args:
            documents: словарь doc_id -> text
            metadata: словарь doc_id -> metadata
        """
        meta = metadata or {}
        for doc_id, text in documents.items():
            tokens = self._tokenize(text)
            self._texts[doc_id] = text
            self._metadata[doc_id] = meta.get(doc_id, {})
            self._documents.append(tokens)
        self._bm25 = BM25Okapi(self._documents)
        
        # Сохраняем индекс на диск если store_path указан
        if self._store_path:
            self.save()

    def search(self, query: str, k: int = 5) -> list[tuple[str, str, float, dict]]:
        """
        Поиск документов по запросу.

        Args:
            query: поисковый запрос
            k: количество результатов

        Returns:
            Список кортежей (doc_id, text, score, metadata), отсортированных по убыванию score
        """
        if not self._documents or self._bm25 is None:
            return []

        query_tokens = self._tokenize(query)
        if not query_tokens:
            return []
        
        scores = self._bm25.get_scores(query_tokens)

        # Собираем результаты с doc_id
        doc_ids = list(self._texts.keys())
        results = []
        for idx, score in enumerate(scores):
            doc_id = doc_ids[idx]
            text = self._texts[doc_id]
            meta = self._metadata[doc_id]
            
            # Вычисляем количество совпадений query tokens в документе
            doc_tokens = set(self._documents[idx])
            query_set = set(query_tokens)
            token_overlap = len(doc_tokens & query_set)
            
            results.append((doc_id, text, score, meta, token_overlap))

        # Сортируем по (score, token_overlap) - оба по убыванию
        # Это обеспечивает стабильное ранжирование при одинаковых scores
        results.sort(key=lambda x: (x[2], x[4]), reverse=True)

        # Фильтруем и возвращаем результаты
        # BM25 может возвращать отрицательные scores для нерелевантных документов
        if results:
            # Фильтруем документы с score > 0
            filtered = [r for r in results if r[2] > 0]
            if filtered:
                # Возвращаем без token_overlap (5-й элемент)
                return [(r[0], r[1], r[2], r[3]) for r in filtered[:k]]
            # Если нет результатов с score > 0, фильтруем по token_overlap > 0
            overlap_filtered = [r for r in results if r[4] > 0]
            if overlap_filtered:
                return [(r[0], r[1], r[2], r[3]) for r in overlap_filtered[:k]]
            # Если всё ещё нет результатов, возвращаем top-k
            return [(r[0], r[1], r[2], r[3]) for r in results[:k]]

        return []

    def clear(self) -> None:
        """
        Очистить индекс.
        
        Если store_path указан — удаляет файл с диска.
        """
        self._bm25 = None
        self._documents = []
        self._texts = {}
        self._metadata = {}
        
        # Удаляем файл с диска если store_path указан
        if self._store_path:
            index_file = Path(self._store_path) / "bm25_index.json"
            if index_file.exists():
                index_file.unlink()

    def stats(self) -> dict:
        """Получить статистику индекса."""
        total_tokens = sum(len(doc) for doc in self._documents)
        return {
            "total_documents": len(self._texts),
            "total_tokens": total_tokens
        }

    def remove(self, doc_id: str) -> None:
        """
        Удалить документ из индекса.

        Args:
            doc_id: идентификатор документа
        """
        if doc_id not in self._texts:
            return

        # Находим индекс документа
        doc_ids = list(self._texts.keys())
        idx = doc_ids.index(doc_id)

        # Удаляем из всех структур
        del self._texts[doc_id]
        del self._metadata[doc_id]
        del self._documents[idx]

        # Перестраиваем BM25 индекс
        if self._documents:
            self._bm25 = BM25Okapi(self._documents)
        else:
            self._bm25 = None
        
        # Сохраняем индекс на диск если store_path указан
        if self._store_path:
            self.save()

    def get_all_metadata(self) -> dict[str, dict]:
        return dict(self._metadata)

    def save(self) -> None:
        """
        Сохранить индекс на диск в формате JSON.

        Сохраняет:
        - _texts: тексты документов
        - _metadata: метаданные
        - _documents: токенизированные документы
        """
        if not self._store_path:
            return
        
        # Создаем директорию если не существует
        store_path = Path(self._store_path)
        store_path.mkdir(parents=True, exist_ok=True)
        
        # Путь к файлу индекса
        index_file = store_path / "bm25_index.json"
        
        # Данные для сохранения
        data = {
            "texts": self._texts,
            "metadata": self._metadata,
            "documents": self._documents
        }
        
        # Сохраняем в JSON
        with open(index_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def load(self) -> bool:
        """
        Загрузить индекс с диска.

        Returns:
            True если успешно загружено, False если файла нет
        """
        if not self._store_path:
            return False
        
        # Путь к файлу индекса
        index_file = Path(self._store_path) / "bm25_index.json"
        
        if not index_file.exists():
            return False
        
        # Загружаем из JSON
        with open(index_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        self._texts = data.get("texts", {})
        self._metadata = data.get("metadata", {})
        self._documents = data.get("documents", [])
        
        # Перестраиваем BM25 индекс
        if self._documents:
            self._bm25 = BM25Okapi(self._documents)
        else:
            self._bm25 = None
        
        return True
