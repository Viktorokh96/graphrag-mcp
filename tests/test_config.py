"""Тесты для RAGConfig."""

from src.config import RAGConfig


class TestRAGConfig:
    """Тесты для RAGConfig."""

    def test_defaults(self):
        cfg = RAGConfig()
        assert cfg.embedding_provider == "openai-compatible"
        assert cfg.embedding_model_name == "BAAI/bge-m3"
        assert cfg.embedding_dim == 1024
        assert cfg.embedding_base_url == ""
        assert cfg.embedding_api_key == ""
        assert cfg.store_path == "./rag_data"
        assert cfg.qdrant_url == ""
        assert cfg.database_url == ""

    def test_from_env_defaults(self, monkeypatch):
        for key in ["EMBEDDING_PROVIDER", "EMBEDDING_MODEL", "EMBEDDING_BASE_URL",
                     "EMBEDDING_API_KEY", "STORE_PATH", "QDRANT_URL", "DATABASE_URL"]:
            monkeypatch.delenv(key, raising=False)

        cfg = RAGConfig()
        assert cfg.embedding_provider == "openai-compatible"
        assert cfg.embedding_model_name == "BAAI/bge-m3"

    def test_resolve_storage_locations(self):
        cfg = RAGConfig(store_path="./data")
        assert cfg.resolve_qdrant_location() == "./data/qdrant"
        assert cfg.resolve_database_url() == "./data/store.db"
        cfg_prod = RAGConfig(qdrant_url="http://q:6333", database_url="postgresql://u@h/db")
        assert cfg_prod.resolve_qdrant_location() == "http://q:6333"
        assert cfg_prod.resolve_database_url() == "postgresql://u@h/db"

    def test_from_env_custom(self, monkeypatch):
        monkeypatch.setenv("EMBEDDING_PROVIDER", "ollama")
        monkeypatch.setenv("EMBEDDING_MODEL_NAME", "qwen3-embedding:8b")
        monkeypatch.setenv("EMBEDDING_BASE_URL", "http://ollama:11434")
        monkeypatch.setenv("STORE_PATH", "/tmp/test_rag")

        cfg = RAGConfig()
        assert cfg.embedding_provider == "ollama"
        assert cfg.embedding_model_name == "qwen3-embedding:8b"
        assert cfg.embedding_base_url == "http://ollama:11434"
        assert cfg.store_path == "/tmp/test_rag"

    def test_custom_values(self):
        cfg = RAGConfig(
            embedding_provider="ollama",
            embedding_model_name="nomic-embed-text",
            embedding_base_url="http://ollama:11434",
            store_path="/tmp/test_rag",
        )
        assert cfg.embedding_provider == "ollama"
        assert cfg.embedding_model_name == "nomic-embed-text"
        assert cfg.embedding_base_url == "http://ollama:11434"

    def test_to_env_preview(self):
        cfg = RAGConfig()
        preview = cfg.to_env_preview()
        assert "EMBEDDING_PROVIDER=openai-compatible" in preview
        assert "EMBEDDING_MODEL=BAAI/bge-m3" in preview
