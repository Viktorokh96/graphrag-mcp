"""Тесты BGE-M3 провайдера эмбеддингов (Фаза 1).

Модель мокается через sys.modules — тесты не скачивают веса и не тянут torch.
Реальная интеграция — в test_real_model_smoke (включается RAG_REAL_MODELS=1).
"""

import os
import sys
import types
from unittest.mock import MagicMock

import numpy as np
import pytest

from src.config import RAGConfig
from src.embeddings import BgeM3EmbeddingGenerator


@pytest.fixture
def fake_st(monkeypatch):
    """Подменяет sentence_transformers фейковым модулем.

    Возвращает MagicMock класса SentenceTransformer; его инстанс кодирует
    детерминированные нормализованные векторы 1024d.
    """
    instance = MagicMock()

    def encode(texts, normalize_embeddings=False):
        vectors = []
        for t in texts:
            rng = np.random.default_rng(abs(hash(t)) % (2**32))
            v = rng.standard_normal(1024)
            v = v / np.linalg.norm(v)
            vectors.append(v)
        return np.array(vectors)

    instance.encode = MagicMock(side_effect=encode)
    st_class = MagicMock(return_value=instance)
    module = types.ModuleType("sentence_transformers")
    module.SentenceTransformer = st_class
    monkeypatch.setitem(sys.modules, "sentence_transformers", module)
    return st_class, instance


class TestBgeM3EmbeddingGenerator:
    def test_lazy_load_no_model_on_init(self, fake_st):
        st_class, _ = fake_st
        gen = BgeM3EmbeddingGenerator()
        assert gen._model is None
        st_class.assert_not_called()

    def test_model_loaded_on_first_embed(self, fake_st):
        st_class, _ = fake_st
        gen = BgeM3EmbeddingGenerator(device="cpu")
        gen.get_embedding("hello world")
        st_class.assert_called_once_with("BAAI/bge-m3", device="cpu")

    def test_custom_model_name_and_device(self, fake_st):
        st_class, _ = fake_st
        gen = BgeM3EmbeddingGenerator(model_name="BAAI/bge-m3-custom", device="cuda")
        gen.get_embedding("text")
        st_class.assert_called_once_with("BAAI/bge-m3-custom", device="cuda")

    def test_default_dimension_before_load(self, fake_st):
        gen = BgeM3EmbeddingGenerator()
        assert gen.get_dimension() == 1024

    def test_dimension_from_cache_after_embed(self, fake_st):
        gen = BgeM3EmbeddingGenerator(dimension=999)
        gen.get_embedding("hello")
        assert gen.get_dimension() == 1024

    def test_embedding_is_normalized(self, fake_st):
        gen = BgeM3EmbeddingGenerator()
        emb = gen.get_embedding("hello world")
        assert len(emb) == 1024
        assert abs(np.linalg.norm(emb) - 1.0) < 1e-6

    def test_encode_called_with_normalization(self, fake_st):
        _, instance = fake_st
        gen = BgeM3EmbeddingGenerator()
        gen.get_embedding("hello")
        _, kwargs = instance.encode.call_args
        assert kwargs.get("normalize_embeddings") is True

    def test_caching_no_reencode(self, fake_st):
        _, instance = fake_st
        gen = BgeM3EmbeddingGenerator()
        first = gen.get_embedding("hello")
        second = gen.get_embedding("hello")
        assert first == second
        assert instance.encode.call_count == 1

    def test_batch_encodes_only_missing(self, fake_st):
        _, instance = fake_st
        gen = BgeM3EmbeddingGenerator()
        gen.get_embedding("a")
        instance.encode.reset_mock()
        result = gen.get_embeddings(["a", "b", "c"])
        assert len(result) == 3
        (texts_arg,), _ = instance.encode.call_args
        assert texts_arg == ["b", "c"]

    def test_clear_cache(self, fake_st):
        _, instance = fake_st
        gen = BgeM3EmbeddingGenerator()
        gen.get_embedding("hello")
        gen.clear_cache()
        gen.get_embedding("hello")
        assert instance.encode.call_count == 2

    def test_deterministic_same_text_same_vector(self, fake_st):
        gen1 = BgeM3EmbeddingGenerator()
        gen2 = BgeM3EmbeddingGenerator()
        assert gen1.get_embedding("same text") == gen2.get_embedding("same text")


class TestConfigBgeM3:
    def test_default_provider_is_bge_m3(self, monkeypatch):
        for var in ("EMBEDDING_MODEL", "EMBEDDING_PROVIDER"):
            monkeypatch.delenv(var, raising=False)
        cfg = RAGConfig.from_env()
        assert cfg.embedding_provider == "bge-m3"
        assert cfg.embedding_dim == 1024
        assert cfg.embedding_device == "cpu"
        assert cfg.bge_model_name == "BAAI/bge-m3"

    def test_embedding_model_env(self, monkeypatch):
        monkeypatch.setenv("EMBEDDING_MODEL", "ollama")
        cfg = RAGConfig.from_env()
        assert cfg.embedding_provider == "ollama"

    def test_legacy_embedding_provider_env(self, monkeypatch):
        monkeypatch.delenv("EMBEDDING_MODEL", raising=False)
        monkeypatch.setenv("EMBEDDING_PROVIDER", "openrouter")
        cfg = RAGConfig.from_env()
        assert cfg.embedding_provider == "openrouter"

    def test_embedding_model_priority_over_provider(self, monkeypatch):
        monkeypatch.setenv("EMBEDDING_MODEL", "bge-m3")
        monkeypatch.setenv("EMBEDDING_PROVIDER", "ollama")
        cfg = RAGConfig.from_env()
        assert cfg.embedding_provider == "bge-m3"

    def test_dim_and_device_env(self, monkeypatch):
        monkeypatch.setenv("EMBEDDING_DIM", "512")
        monkeypatch.setenv("EMBEDDING_DEVICE", "cuda")
        cfg = RAGConfig.from_env()
        assert cfg.embedding_dim == 512
        assert cfg.embedding_device == "cuda"

    def test_env_preview_contains_new_vars(self):
        preview = RAGConfig().to_env_preview()
        assert "EMBEDDING_MODEL=bge-m3" in preview
        assert "BGE_MODEL_NAME=BAAI/bge-m3" in preview
        assert "EMBEDDING_DIM=1024" in preview


@pytest.mark.skipif(
    os.environ.get("RAG_REAL_MODELS") != "1",
    reason="Реальная модель BGE-M3 (~2GB): включается RAG_REAL_MODELS=1",
)
class TestRealBgeM3:
    """Смоук-тест на реальной модели: качество мультиязычных эмбеддингов."""

    def test_real_model_smoke(self):
        gen = BgeM3EmbeddingGenerator()
        emb_en = gen.get_embedding("The cat sits on the mat")
        emb_ru = gen.get_embedding("Кот сидит на коврике")
        emb_off = gen.get_embedding("Quarterly financial report Q3 revenue")

        assert len(emb_en) == 1024
        # Кросс-язычная пара должна быть ближе, чем нерелевантный текст,
        # с заметным отрывом. Абсолютный порог 0.6: реальное значение ~0.68
        # для коротких фраз (косинусы BGE-M3 на коротких текстах ниже, чем
        # на абзацах — это норма).
        sim_pair = float(np.dot(emb_en, emb_ru))
        sim_off = float(np.dot(emb_en, emb_off))
        assert sim_pair > sim_off + 0.15
        assert sim_pair > 0.6
