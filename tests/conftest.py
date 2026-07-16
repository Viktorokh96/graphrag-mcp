"""Общие фикстуры."""

import os

import pytest

_CONFIG_ENV_VARS = [
    "EMBEDDING_PROVIDER",
    "EMBEDDING_MODEL",
    "EMBEDDING_MODEL_NAME",
    "EMBEDDING_BASE_URL",
    "EMBEDDING_API_KEY",
    "EMBEDDING_DIM",
    "EMBEDDING_DEVICE",
    "OPENAI_MODEL",
    "OPENAI_BASE_URL",
    "OPENAI_API_KEY",
    "HF_OFFLINE",
    "HF_TOKEN",
    "MODELS_DIR",
    "PRELOAD_MODELS",
    "STORE_PATH",
    "QDRANT_URL",
    "DATABASE_URL",
    "RERANK_ENABLED",
    "RERANK_PROVIDER",
    "RERANK_MODEL",
    "RERANK_BASE_URL",
    "RERANK_API_KEY",
    "RERANK_DEVICE",
    "QUERY_EXPANSION_ENABLED",
    "EXPANSION_PROVIDER",
    "EXPANSION_MODEL",
    "EXPANSION_BASE_URL",
    "EXPANSION_API_KEY",
    "EXPANSION_COUNT",
    "QUERY_EXPANSION_OLLAMA_URL",
    "QUERY_EXPANSION_MODEL",
    "OLLAMA_BASE_URL",
]


@pytest.fixture(autouse=True)
def clean_config_env(monkeypatch):
    """Очистка env-переменных конфига перед каждым тестом."""
    for key in _CONFIG_ENV_VARS:
        monkeypatch.delenv(key, raising=False)

import numpy as np

from src.config import RAGConfig
from src.rag import RAGSystem
from tests.semantic_mock import SemanticMockEmbeddingGenerator


class HashEmbeddingGenerator:
    """Детерминированный мок-эмбеддер: вектор из PRNG, сидированного текстом.

    Семантики нет (тексты почти ортогональны) — подходит для тестов CRUD,
    персистентности и BM25-канала. Для тестов качества поиска используйте
    SemanticMockEmbeddingGenerator из tests/semantic_mock.py.
    """

    def __init__(self, dimension: int = 64):
        self._dimension = dimension
        self._cache: dict[str, list[float]] = {}

    def get_embedding(self, text: str) -> list[float]:
        if text not in self._cache:
            seed = sum(ord(c) * (i + 1) for i, c in enumerate(text[:100])) % (2**32)
            v = np.random.default_rng(seed).standard_normal(self._dimension)
            self._cache[text] = (v / np.linalg.norm(v)).tolist()
        return self._cache[text]

    def get_embeddings(self, texts: list[str]) -> list[list[float]]:
        return [self.get_embedding(t) for t in texts]

    def get_dimension(self) -> int:
        return self._dimension

    def clear_cache(self) -> None:
        self._cache.clear()


@pytest.fixture
def make_rag(tmp_path):
    """Фабрика RAGSystem на новом стеке с автоматическим close() в teardown.

    Использование:
        rag = make_rag()                       # HashEmbeddingGenerator
        rag = make_rag(embedder=SemanticMockEmbeddingGenerator())
        rag = make_rag(subdir="other")         # второй стор в том же тесте
    """
    created: list[RAGSystem] = []

    def _make(embedder=None, subdir: str = "store", config: RAGConfig = None):
        cfg = config or RAGConfig()
        cfg.store_path = str(tmp_path / subdir)
        rag = RAGSystem(config=cfg, embedding_generator=embedder or HashEmbeddingGenerator())
        created.append(rag)
        return rag

    yield _make
    for rag in created:
        try:
            rag.close()
        except Exception:
            pass


@pytest.fixture
def rag(make_rag):
    """Готовый RAGSystem с HashEmbeddingGenerator."""
    return make_rag()


@pytest.fixture
def semantic_rag(make_rag):
    """RAGSystem с семантическим мок-эмбеддером (для тестов качества)."""
    return make_rag(embedder=SemanticMockEmbeddingGenerator())
