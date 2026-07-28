import os
from typing import Any

from pydantic import model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


def _mask_secret(value: str) -> str:
    """Показать только последние 4 символа секрета (или пусто)."""
    if not value:
        return ""
    return f"***{value[-4:]}" if len(value) > 4 else "***"


class RAGConfig(BaseSettings):
    model_config = SettingsConfigDict(extra="ignore")

    # -- Embedding ---------------------------------------------------------
    embedding_provider: str = "openai-compatible"
    embedding_model_name: str = "BAAI/bge-m3"
    embedding_base_url: str = ""
    embedding_api_key: str = ""
    embedding_dim: int = 1024
    embedding_device: str = "cpu"
    hf_offline: bool = False
    hf_token: str = ""
    models_dir: str = ""
    preload_models: bool = False

    # -- Storage -----------------------------------------------------------
    store_path: str = "./rag_data"
    qdrant_url: str = ""
    database_url: str = ""

    # -- Hybrid search -----------------------------------------------------
    default_alpha: float = 0.5
    cyrillic_alpha: float = 0.85
    hybrid_expand: int = 3
    hybrid_min_candidates: int = 20

    # -- Chunking ----------------------------------------------------------
    chunk_size: int = 512
    chunk_overlap: int = 64

    # -- Reranker ----------------------------------------------------------
    rerank_enabled: bool = False
    rerank_provider: str = "sentence_transformer"
    rerank_model: str = "BAAI/bge-reranker-v2-m3"
    rerank_base_url: str = ""
    rerank_api_key: str = ""
    rerank_device: str = "cpu"
    rerank_top_k_multiplier: int = 2

    # -- HTTP API security -------------------------------------------------
    # Bearer-токен для REST/MCP-эндпоинтов. Пустая строка — аутентификация
    # выключена (допустимо только для loopback-биндинга).
    api_token: str = ""
    # Разрешённые CORS-origin'ы (через запятую). Пусто — cross-origin запрещён,
    # работает только same-origin UI. "*" разрешает любой origin.
    cors_allow_origins: str = ""

    # -- Query expansion ---------------------------------------------------
    query_expansion_enabled: bool = False
    expansion_provider: str = "ollama"
    expansion_model: str = "qwen3:1.8b"
    expansion_base_url: str = "http://localhost:11434"
    expansion_api_key: str = ""
    expansion_count: int = 3

    # -- Validators --------------------------------------------------------

    @model_validator(mode="before")
    @classmethod
    def _apply_fallback_env_vars(cls, data: Any) -> Any:
        """Map legacy env var names to field names."""
        if not isinstance(data, dict):
            return data
        def _pick(*keys: str) -> str | None:
            for k in keys:
                if k in os.environ:
                    return os.environ[k]
            return None


        # EMBEDDING_MODEL / OPENAI_MODEL → embedding_model_name
        if "embedding_model_name" not in data:
            v = _pick("EMBEDDING_MODEL", "OPENAI_MODEL")
            if v is not None:
                data["embedding_model_name"] = v

        # EMBEDDING_BASE_URL / OPENAI_BASE_URL → embedding_base_url
        if "embedding_base_url" not in data:
            v = _pick("OPENAI_BASE_URL")
            if v is not None:
                data["embedding_base_url"] = v

        # EMBEDDING_API_KEY / OPENAI_API_KEY → embedding_api_key
        if "embedding_api_key" not in data:
            v = _pick("OPENAI_API_KEY")
            if v is not None:
                data["embedding_api_key"] = v

        # EXPANSION_MODEL / QUERY_EXPANSION_MODEL → expansion_model
        if "expansion_model" not in data:
            v = _pick("QUERY_EXPANSION_MODEL")
            if v is not None:
                data["expansion_model"] = v

        # EXPANSION_BASE_URL / QUERY_EXPANSION_OLLAMA_URL / OLLAMA_BASE_URL → expansion_base_url
        if "expansion_base_url" not in data:
            v = _pick("QUERY_EXPANSION_OLLAMA_URL", "OLLAMA_BASE_URL")
            if v is not None:
                data["expansion_base_url"] = v

        return data

    # -- Methods -----------------------------------------------------------

    @classmethod
    def from_env(cls) -> "RAGConfig":
        """Load from os.environ + .env file."""
        return cls(_env_file=".env")

    def to_env_preview(self) -> str:
        return "\n".join([
            f"EMBEDDING_PROVIDER={self.embedding_provider}",
            f"EMBEDDING_MODEL={self.embedding_model_name}",
            f"EMBEDDING_BASE_URL={self.embedding_base_url}",
            f"EMBEDDING_API_KEY={_mask_secret(self.embedding_api_key)}",
            f"EMBEDDING_DIM={self.embedding_dim}",
            f"EMBEDDING_DEVICE={self.embedding_device}",
            f"STORE_PATH={self.store_path}",
            f"QDRANT_URL={self.qdrant_url}",
            f"DATABASE_URL={self.database_url}",
            f"RERANK_ENABLED={'true' if self.rerank_enabled else 'false'}",
            f"RERANK_PROVIDER={self.rerank_provider}",
            f"RERANK_MODEL={self.rerank_model}",
            f"QUERY_EXPANSION_ENABLED={'true' if self.query_expansion_enabled else 'false'}",
            f"EXPANSION_PROVIDER={self.expansion_provider}",
            f"EXPANSION_MODEL={self.expansion_model}",
            f"EXPANSION_BASE_URL={self.expansion_base_url}",
            f"PRELOAD_MODELS={'true' if self.preload_models else 'false'}",
        ])

    def resolve_cors_origins(self) -> list[str]:
        return [o.strip() for o in self.cors_allow_origins.split(",") if o.strip()]

    def resolve_qdrant_location(self) -> str:
        return self.qdrant_url or f"{self.store_path}/qdrant"

    def resolve_database_url(self) -> str:
        return self.database_url or f"{self.store_path}/store.db"

    def resolve_model_path(self, model_name: str) -> str:
        if self.models_dir:
            short_name = model_name.rsplit("/", 1)[-1]
            local = os.path.join(self.models_dir, short_name)
            if os.path.isdir(local):
                return local
        return model_name
