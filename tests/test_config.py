"""Тесты для модуля конфигурации RAG системы."""

from src.config import RAGConfig


class TestRAGConfig:
    """Тесты для RAGConfig."""

    def test_defaults(self):
        cfg = RAGConfig()
        assert cfg.embedding_provider == "bge-m3"
        assert cfg.bge_model_name == "BAAI/bge-m3"
        assert cfg.embedding_dim == 1024
        assert cfg.embedding_device == "cpu"
        assert cfg.ollama_base_url == "http://localhost:11434"
        assert cfg.ollama_model == "qwen3-embedding:8b"
        assert cfg.ollama_dimension == 4096
        assert cfg.openrouter_api_key is None
        assert cfg.openrouter_model == "openai/text-embedding-3-small"
        assert cfg.store_path == "./rag_data"
        assert cfg.qdrant_url == ""
        assert cfg.database_url == ""

    def test_from_env_defaults(self, monkeypatch):
        for key in ["EMBEDDING_MODEL", "EMBEDDING_PROVIDER", "OLLAMA_BASE_URL", "OLLAMA_MODEL",
                     "OLLAMA_DIMENSION", "OPENROUTER_API_KEY", "OPENROUTER_MODEL", "STORE_PATH",
                     "QDRANT_URL", "DATABASE_URL"]:
            monkeypatch.delenv(key, raising=False)

        cfg = RAGConfig.from_env()
        assert cfg.embedding_provider == "bge-m3"
        assert cfg.ollama_base_url == "http://localhost:11434"

    def test_resolve_storage_locations(self):
        cfg = RAGConfig(store_path="./data")
        assert cfg.resolve_qdrant_location() == "./data/qdrant"
        assert cfg.resolve_database_url() == "./data/store.db"
        cfg_prod = RAGConfig(qdrant_url="http://q:6333", database_url="postgresql://u@h/db")
        assert cfg_prod.resolve_qdrant_location() == "http://q:6333"
        assert cfg_prod.resolve_database_url() == "postgresql://u@h/db"

    def test_from_env_custom(self, monkeypatch):
        monkeypatch.setenv("EMBEDDING_PROVIDER", "openrouter")
        monkeypatch.setenv("OLLAMA_BASE_URL", "http://custom:11434")
        monkeypatch.setenv("OLLAMA_MODEL", "nomic-embed-text")
        monkeypatch.setenv("OLLAMA_DIMENSION", "768")
        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test")
        monkeypatch.setenv("OPENROUTER_MODEL", "custom/model")
        monkeypatch.setenv("STORE_PATH", "/tmp/test_rag")

        cfg = RAGConfig.from_env()
        assert cfg.embedding_provider == "openrouter"
        assert cfg.ollama_base_url == "http://custom:11434"
        assert cfg.ollama_model == "nomic-embed-text"
        assert cfg.ollama_dimension == 768
        assert cfg.openrouter_api_key == "sk-test"
        assert cfg.openrouter_model == "custom/model"
        assert cfg.store_path == "/tmp/test_rag"

    def test_custom_values(self):
        cfg = RAGConfig(
            embedding_provider="openrouter",
            ollama_base_url="http://custom:11434",
            ollama_model="nomic-embed-text",
            ollama_dimension=768,
            openrouter_api_key="sk-test",
            openrouter_model="custom/model",
            store_path="/tmp/test_rag",
        )
        assert cfg.embedding_provider == "openrouter"
        assert cfg.ollama_base_url == "http://custom:11434"
        assert cfg.openrouter_api_key == "sk-test"

    def test_to_env_preview(self):
        cfg = RAGConfig()
        preview = cfg.to_env_preview()
        assert "EMBEDDING_MODEL=bge-m3" in preview
        assert "OLLAMA_BASE_URL=http://localhost:11434" in preview