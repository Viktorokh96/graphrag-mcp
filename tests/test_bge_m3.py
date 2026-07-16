"""Тесты конфигурации эмбеддингов."""
from src.config import RAGConfig


class TestConfigBgeM3:
    def test_default_provider_is_openai_compatible(self, monkeypatch):
        monkeypatch.delenv("EMBEDDING_PROVIDER", raising=False)
        monkeypatch.delenv("EMBEDDING_MODEL", raising=False)
        cfg = RAGConfig()
        assert cfg.embedding_provider == "openai-compatible"
        assert cfg.embedding_model_name == "BAAI/bge-m3"

    def test_embedding_provider_env(self, monkeypatch):
        monkeypatch.setenv("EMBEDDING_PROVIDER", "ollama")
        monkeypatch.setenv("EMBEDDING_MODEL", "qwen3-embedding:8b")
        cfg = RAGConfig()
        assert cfg.embedding_provider == "ollama"
        assert cfg.embedding_model_name == "qwen3-embedding:8b"

    def test_embedding_model_via_OPENAI_MODEL_fallback(self, monkeypatch):
        monkeypatch.delenv("EMBEDDING_MODEL", raising=False)
        monkeypatch.setenv("OPENAI_MODEL", "text-embedding-3-small")
        cfg = RAGConfig()
        assert cfg.embedding_model_name == "text-embedding-3-small"

    def test_embedding_base_url_fallback(self, monkeypatch):
        monkeypatch.delenv("EMBEDDING_BASE_URL", raising=False)
        monkeypatch.setenv("OPENAI_BASE_URL", "http://custom:8080/v1")
        cfg = RAGConfig()
        assert cfg.embedding_base_url == "http://custom:8080/v1"

    def test_env_preview_contains_new_vars(self):
        preview = RAGConfig().to_env_preview()
        assert "EMBEDDING_PROVIDER=openai-compatible" in preview
        assert "EMBEDDING_MODEL=BAAI/bge-m3" in preview
