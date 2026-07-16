"""Embedding Generator for RAG system."""
import logging
import re
import httpx
import os
import numpy as np

logger = logging.getLogger(__name__)


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


class SentenceTransformerEmbeddingGenerator:
    """Локальные эмбеддинги через sentence-transformers (BGE-M3, all-MiniLM-L6-v2 и др.).

    Модель загружается лениво при первом вызове get_embedding/get_embeddings.
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-m3",
        device: str = "cpu",
        dimension: int = 1024,
        local_files_only: bool = False,
        token: str = "",
    ):
        self.model_name = model_name
        self.device = device
        self._dimension = dimension
        self._local_files_only = local_files_only
        self._token = token or None
        self._model = None
        self._cache: dict[str, list[float]] = {}

    def _ensure_model(self):
        if self._model is None:
            import time as _time
            t0 = _time.monotonic()
            logger.info(
                "Loading embedding model %s (device=%s, local_files_only=%s) …",
                self.model_name, self.device, self._local_files_only,
            )
            from sentence_transformers import SentenceTransformer
            kwargs = dict(
                device=self.device,
                local_files_only=self._local_files_only,
            )
            if self._token:
                kwargs["token"] = self._token
            self._model = SentenceTransformer(self.model_name, **kwargs)
            logger.info("Embedding model loaded in %.1fs", _time.monotonic() - t0)
        return self._model

    def get_embedding(self, text: str) -> list[float]:
        if text in self._cache:
            return self._cache[text]
        embedding = self.get_embeddings([text])[0]
        return embedding

    def get_embeddings(self, texts: list[str]) -> list[list[float]]:
        missing = [t for t in texts if t not in self._cache]
        if missing:
            model = self._ensure_model()
            vectors = model.encode(missing, normalize_embeddings=True)
            for text, vec in zip(missing, vectors):
                self._cache[text] = vec.tolist()
        return [self._cache[t] for t in texts]

    def get_dimension(self) -> int:
        for emb in self._cache.values():
            return len(emb)
        return self._dimension

    def clear_cache(self):
        self._cache.clear()


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
        self._client = httpx.Client(timeout=120.0)

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
        # Дедуплицируем missing, сохраняя порядок: батч в Ollama возвращает по
        # одному эмбеддингу на каждый элемент input, поэтому дубликаты сместили бы
        # соответствие text↔embedding.
        missing = list(dict.fromkeys(t for t in texts if t not in self._cache))
        if missing:
            try:
                url = f"{self.base_url}/api/embed"
                payload = {"model": self.model, "input": missing}
                response = self._client.post(url, json=payload, timeout=120.0)
                response.raise_for_status()
                embeddings = response.json().get("embeddings", [])
                if len(embeddings) != len(missing):
                    raise ValueError(
                        f"Ollama returned {len(embeddings)} embeddings for {len(missing)} inputs"
                    )
                for text, vec in zip(missing, embeddings):
                    self._cache[text] = vec
            except Exception:
                for text in missing:
                    self._cache[text] = self._fallback_embedding(text)
        return [self._cache[t] for t in texts]

    def get_dimension(self) -> int:
        for emb in self._cache.values():
            return len(emb)
        return self._dimension

    def clear_cache(self):
        self._cache.clear()

    def close(self):
        self._client.close()

    def _call_api(self, text: str) -> list[float]:
        url = f"{self.base_url}/api/embed"
        payload = {
            "model": self.model,
            "input": text,
        }

        response = self._client.post(url, json=payload)
        response.raise_for_status()
        data = response.json()
        embedding = data["embeddings"][0]
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


class OpenAICompatibleEmbeddingGenerator:
    """Генератор эмбеддингов через OpenAI-compatible API (TEI, Infinity, vLLM и др.).
    
    Поддерживает POST /v1/embeddings с телом {model, input}.
    Совместим с HuggingFace TEI, Infinity, vLLM (≥0.6.0) и самописными серверами.
    """
    
    def __init__(
        self,
        base_url: str = "http://localhost:8080/v1",
        api_key: str = "",
        model: str = "BAAI/bge-m3",
        dimension: int = 1024,
    ):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self._dimension = dimension
        self._cache: dict[str, list[float]] = {}
        self._client = httpx.Client(timeout=120.0)
    
    def get_embedding(self, text: str) -> list[float]:
        if text in self._cache:
            return self._cache[text]
        embedding = self.get_embeddings([text])[0]
        return embedding
    
    def get_embeddings(self, texts: list[str]) -> list[list[float]]:
        missing = list(dict.fromkeys(t for t in texts if t not in self._cache))
        if missing:
            try:
                url = f"{self.base_url}/embeddings"
                headers = {"Content-Type": "application/json"}
                if self.api_key:
                    headers["Authorization"] = f"Bearer {self.api_key}"
                payload = {"model": self.model, "input": missing if len(missing) > 1 else missing[0]}
                response = self._client.post(url, headers=headers, json=payload, timeout=120.0)
                response.raise_for_status()
                data = response.json()
                embeddings = [item["embedding"] for item in sorted(data["data"], key=lambda x: x["index"])]
                if len(embeddings) != len(missing):
                    raise ValueError(
                        f"Server returned {len(embeddings)} embeddings for {len(missing)} inputs"
                    )
                for text, vec in zip(missing, embeddings):
                    self._cache[text] = vec
            except Exception:
                for text in missing:
                    self._cache[text] = self._fallback_embedding(text)
        return [self._cache[t] for t in texts]
    
    def get_dimension(self) -> int:
        for emb in self._cache.values():
            return len(emb)
        return self._dimension
    
    def clear_cache(self):
        self._cache.clear()
    
    def close(self):
        self._client.close()
    
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
