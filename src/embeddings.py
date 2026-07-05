"""Embedding Generator for RAG system."""

import re
from typing import Optional
import httpx
import os
import numpy as np


class EmbeddingGenerator:
    """TF-IDF векторизатор для эмбеддингов."""

    def __init__(self):
        """Инициализация векторизатора."""
        self._vocabulary: dict[str, int] = {}
        self._idf: dict[str, float] = {}
        self._vocab_size: int = 0

    def fit(self, texts: list[str]) -> "EmbeddingGenerator":
        """
        Построить словарь и вычислить IDF для документов.

        Args:
            texts: Список текстов для обучения

        Returns:
            self
        """
        # Токенизация всех документов
        doc_tokens = [self._tokenize(text) for text in texts]

        # Сбор уникальных слов (vocabulary)
        all_words = set()
        for tokens in doc_tokens:
            all_words.update(tokens)

        self._vocabulary = {word: idx for idx, word in enumerate(sorted(all_words))}
        self._vocab_size = len(self._vocabulary)

        # Вычисление IDF
        doc_freq = {word: 0 for word in self._vocabulary}
        for tokens in doc_tokens:
            unique_tokens = set(tokens)
            for word in unique_tokens:
                if word in doc_freq:
                    doc_freq[word] += 1

        n_docs = len(texts)
        for word, freq in doc_freq.items():
            self._idf[word] = np.log((n_docs + 1) / (freq + 1)) + 1

        return self

    def transform(self, texts: list[str]) -> list[list[float]]:
        """
        Преобразовать тексты в векторы TF-IDF.

        Args:
            texts: Список текстов для векторизации

        Returns:
            Список списков float (векторов)
        """
        vectors = []
        for text in texts:
            vector = self._transform_single(text)
            vectors.append(vector.tolist())
        return vectors

    def fit_transform(self, texts: list[str]) -> list[list[float]]:
        """
        fit + transform в одном вызове.

        Args:
            texts: Список текстов

        Returns:
            Список списков float
        """
        self.fit(texts)
        return self.transform(texts)

    def _tokenize(self, text: str) -> list[str]:
        """Токенизация текста: нижний регистр, только слова."""
        text = text.lower()
        words = re.findall(r'\b[a-z]+\b', text)
        return words

    def _transform_single(self, text: str) -> np.ndarray:
        """Преобразовать один текст в вектор."""
        tokens = self._tokenize(text)

        # TF (term frequency)
        tf = {}
        for word in tokens:
            tf[word] = tf.get(word, 0) + 1

        # Нормализация TF по длине документа
        if tokens:
            tf_len = len(tokens)
            for word in tf:
                tf[word] /= tf_len

        # TF-IDF вектор
        vector = np.zeros(self._vocab_size, dtype=np.float64)

        for word, idx in self._vocabulary.items():
            if word in tf:
                vector[idx] = tf[word] * self._idf.get(word, 0.0)

        # L2 нормализация
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector = vector / norm

        return vector


class OpenRouterEmbeddingGenerator:
    """Генератор эмбеддингов через OpenRouter API."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "openai/text-embedding-3-small"
    ):
        """
        Инициализация генератора эмбеддингов.

        Args:
            api_key: OpenRouter API ключ (из аргумента или env OPENROUTER_API_KEY)
            model: Модель для эмбеддингов (по умолчанию openai/text-embedding-3-small)
        """
        self.api_key = api_key or os.environ.get("OPENROUTER_API_KEY")
        self.model = model
        self._cache: dict[str, list[float]] = {}
        # Размерность fallback эмбеддингов.
        # openai/text-embedding-3-small выдаёт 1536, поэтому fallback должен совпадать,
        # иначе ChromaDB коллекция сломается при смене источника.
        self._fallback_dimension = 1536

    def get_embedding(self, text: str) -> list[float]:
        """
        Получить эмбеддинг для текста через OpenRouter API.

        Args:
            text: Текст для эмбеддинга

        Returns:
            Список float значений (эмбеддинг)
        """
        # Проверка кеша
        if text in self._cache:
            return self._cache[text]

        # Попытка получить через API
        try:
            embedding = self._call_api(text)
            self._cache[text] = embedding
            return embedding
        except Exception:
            # Fallback при ошибке API
            embedding = self._fallback_embedding(text)
            self._cache[text] = embedding
            return embedding

    def get_embeddings(self, texts: list[str]) -> list[list[float]]:
        """
        Получить эмбеддинги для нескольких текстов (батч).

        Args:
            texts: Список текстов для эмбеддинга

        Returns:
            Список списков float значений
        """
        results = []
        for text in texts:
            results.append(self.get_embedding(text))
        return results

    def get_dimension(self) -> int:
        """
        Получить размерность эмбеддинга.

        Если есть кешированный эмбеддинг — берёт из него,
        иначе возвращает размерность fallback'а (1536).
        """
        # Берём любой эмбеддинг из кеша чтобы узнать размерность
        for emb in self._cache.values():
            return len(emb)
        return self._fallback_dimension

    def clear_cache(self):
        """Очистить кеш эмбеддингов."""
        self._cache.clear()

    def _call_api(self, text: str) -> list[float]:
        """
        Вызов OpenRouter API для получения эмбеддинга.

        Args:
            text: Текст для эмбеддинга

        Returns:
            Список float значений

        Raises:
            Exception: При ошибке API
        """
        url = "https://openrouter.ai/api/v1/embeddings"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": self.model,
            "input": text
        }

        with httpx.Client() as client:
            response = client.post(url, headers=headers, json=payload, timeout=30.0)
            response.raise_for_status()
            data = response.json()
            embedding = data["data"][0]["embedding"]
            return embedding

    def _fallback_embedding(self, text: str) -> list[float]:
        """
        Упрощённый TF-IDF-like вектор как fallback при ошибке API.

        Args:
            text: Текст для эмбеддинга

        Returns:
            Список float значений фиксированной размерности
        """
        # Простая реализация: хешируем слова и создаём вектор
        import hashlib

        embedding = [0.0] * self._fallback_dimension

        # Разбиваем текст на слова
        words = text.lower().split()

        for word in words:
            # Удаляем пунктуацию
            word = word.strip(".,!?;:'\"()[]{}")
            if not word:
                continue

            # Хешируем слово для получения индекса
            hash_val = int(hashlib.md5(word.encode()).hexdigest(), 16)
            index = hash_val % self._fallback_dimension

            # Добавляем вклад слова (простой TF-like)
            embedding[index] += 1.0 / len(words)

        # Нормализация (L2)
        norm = sum(v * v for v in embedding) ** 0.5
        if norm > 0:
            embedding = [v / norm for v in embedding]

        return embedding


class OllamaEmbeddingGenerator:
    """Генератор эмбеддингов через Ollama API (локально)."""

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model: str = "qwen3-embedding:8b",
        dimension: int = 4096,
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self._dimension = dimension
        self._cache: dict[str, list[float]] = {}

    def get_embedding(self, text: str) -> list[float]:
        if text in self._cache:
            return self._cache[text]

        try:
            embedding = self._call_api(text)
            self._cache[text] = embedding
            return embedding
        except Exception:
            embedding = self._fallback_embedding(text)
            self._cache[text] = embedding
            return embedding

    def get_embeddings(self, texts: list[str]) -> list[list[float]]:
        results = []
        for text in texts:
            results.append(self.get_embedding(text))
        return results

    def get_dimension(self) -> int:
        for emb in self._cache.values():
            return len(emb)
        return self._dimension

    def clear_cache(self):
        self._cache.clear()

    def _call_api(self, text: str) -> list[float]:
        url = f"{self.base_url}/api/embeddings"
        payload = {
            "model": self.model,
            "prompt": text,
        }

        with httpx.Client() as client:
            response = client.post(url, json=payload, timeout=30.0)
            response.raise_for_status()
            data = response.json()
            embedding = data["embedding"]
            return embedding

    def _fallback_embedding(self, text: str) -> list[float]:
        import hashlib

        embedding = [0.0] * self._dimension
        words = text.lower().split()

        for word in words:
            word = word.strip(".,!?;:'\"()[]{}")
            if not word:
                continue
            hash_val = int(hashlib.md5(word.encode()).hexdigest(), 16)
            index = hash_val % self._dimension
            embedding[index] += 1.0 / len(words)

        norm = sum(v * v for v in embedding) ** 0.5
        if norm > 0:
            embedding = [v / norm for v in embedding]

        return embedding
