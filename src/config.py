import os
from dataclasses import dataclass
from typing import Optional


def _env_int(key: str, default: str) -> int:
    val = os.environ.get(key)
    if val is None or val.strip() == "":
        return int(default)
    return int(val)


def _env_float(key: str, default: str) -> float:
    val = os.environ.get(key)
    if val is None or val.strip() == "":
        return float(default)
    return float(val)


@dataclass
class RAGConfig:
    # Тип провайдера: openai-compatible | anthropic | ollama | sentence_transformer
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
    store_path: str = "./rag_data"
    # Хранилища (Фаза 2). Пустые значения → embedded-режим внутри store_path:
    #   qdrant_url:   URL Qdrant-сервера (prod) | "" → embedded {store_path}/qdrant
    #   database_url: postgresql://... (prod)   | "" → SQLite {store_path}/store.db
    qdrant_url: str = ""
    database_url: str = ""
    # Баланс гибридного поиска: 0.0 = чистый BM25, 1.0 = чистый семантический.
    default_alpha: float = 0.5
    cyrillic_alpha: float = 0.85
    hybrid_expand: int = 3
    hybrid_min_candidates: int = 20
    # Чанкование больших документов (в токенах)
    chunk_size: int = 512
    chunk_overlap: int = 64
    rerank_enabled: bool = False
    rerank_model: str = "BAAI/bge-reranker-v2-m3"
    rerank_device: str = "cpu"
    rerank_top_k_multiplier: int = 2
    query_expansion_enabled: bool = False
    query_expansion_model: str = "qwen3:1.8b"
    query_expansion_count: int = 3
    query_expansion_ollama_url: str = "http://localhost:11434"

    @classmethod
    def from_env(cls) -> "RAGConfig":
        return cls(
            embedding_provider=os.environ.get("EMBEDDING_PROVIDER", "openai-compatible"),
            embedding_model_name=os.environ.get(
                "EMBEDDING_MODEL", os.environ.get("OPENAI_MODEL", "BAAI/bge-m3")
            ),
            embedding_base_url=os.environ.get(
                "EMBEDDING_BASE_URL", os.environ.get("OPENAI_BASE_URL", "")
            ),
            embedding_api_key=os.environ.get(
                "EMBEDDING_API_KEY", os.environ.get("OPENAI_API_KEY", "")
            ),
            embedding_dim=_env_int("EMBEDDING_DIM", "1024"),
            embedding_device=os.environ.get("EMBEDDING_DEVICE", "cpu"),
            hf_offline=os.environ.get("HF_HUB_OFFLINE", "").lower() in ("1", "true", "yes"),
            hf_token=os.environ.get("HF_TOKEN", ""),
            preload_models=os.environ.get("PRELOAD_MODELS", "").lower() in ("1", "true", "yes"),
            models_dir=os.environ.get("MODELS_DIR", ""),
            store_path=os.environ.get("STORE_PATH", "./rag_data"),
            qdrant_url=os.environ.get("QDRANT_URL", ""),
            database_url=os.environ.get("DATABASE_URL", ""),
            default_alpha=_env_float("RAG_DEFAULT_ALPHA", "0.5"),
            cyrillic_alpha=_env_float("RAG_CYRILLIC_ALPHA", "0.85"),
            hybrid_expand=_env_int("RAG_HYBRID_EXPAND", "3"),
            hybrid_min_candidates=_env_int("RAG_HYBRID_MIN_CANDIDATES", "20"),
            chunk_size=_env_int("CHUNK_SIZE", "512"),
            chunk_overlap=_env_int("CHUNK_OVERLAP", "64"),
            rerank_enabled=os.environ.get("RERANK_ENABLED", "").lower() in ("1", "true", "yes"),
            rerank_model=os.environ.get("RERANK_MODEL", "BAAI/bge-reranker-v2-m3"),
            rerank_device=os.environ.get("RERANK_DEVICE", "cpu"),
            rerank_top_k_multiplier=_env_int("RERANK_TOP_K_MULTIPLIER", "2"),
            query_expansion_enabled=os.environ.get("QUERY_EXPANSION_ENABLED", "").lower() in ("1", "true", "yes"),
            query_expansion_model=os.environ.get("QUERY_EXPANSION_MODEL", "qwen3:1.8b"),
            query_expansion_count=_env_int("QUERY_EXPANSION_COUNT", "3"),
            query_expansion_ollama_url=os.environ.get(
                "QUERY_EXPANSION_OLLAMA_URL",
                os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434"),
            ),
        )


    def to_env_preview(self) -> str:
        lines = [
            "# Тип провайдера: openai-compatible | anthropic | ollama | sentence_transformer",
            f"EMBEDDING_PROVIDER={self.embedding_provider}",
            "",
            "# Имя модели эмбеддингов",
            f"EMBEDDING_MODEL={self.embedding_model_name}",
            "",
            "# Базовый URL API (для openai-compatible, anthropic, ollama)",
            f"EMBEDDING_BASE_URL={self.embedding_base_url}",
            "# API ключ (если требуется)",
            f"EMBEDDING_API_KEY={self.embedding_api_key or ''}",
            "# Размерность эмбеддингов",
            f"EMBEDDING_DIM={self.embedding_dim}",
            "# Устройство (для sentence_transformer)",
            f"EMBEDDING_DEVICE={self.embedding_device}",
            "",
            "# Путь к хранилищу",
            f"STORE_PATH={self.store_path}",
            "",
            "# Production-хранилища (пусто = embedded внутри STORE_PATH)",
            f"QDRANT_URL={self.qdrant_url}",
            f"DATABASE_URL={self.database_url}",
            "",
            "# Гибридный поиск",
            f"RAG_DEFAULT_ALPHA={self.default_alpha}",
            f"RAG_CYRILLIC_ALPHA={self.cyrillic_alpha}",
            f"RAG_HYBRID_EXPAND={self.hybrid_expand}",
            f"RAG_HYBRID_MIN_CANDIDATES={self.hybrid_min_candidates}",
            "",
            "# Чанкование больших документов (в токенах)",
            f"CHUNK_SIZE={self.chunk_size}",
            f"CHUNK_OVERLAP={self.chunk_overlap}",
            "",
            "# Reranker (Cross-encoder)",
            f"RERANK_ENABLED={'true' if self.rerank_enabled else 'false'}",
            f"RERANK_MODEL={self.rerank_model}",
            f"RERANK_DEVICE={self.rerank_device}",
            f"RERANK_TOP_K_MULTIPLIER={self.rerank_top_k_multiplier}",
            "",
            "# Query expansion (малая LLM для парафраза запросов)",
            f"QUERY_EXPANSION_ENABLED={'true' if self.query_expansion_enabled else 'false'}",
            f"QUERY_EXPANSION_MODEL={self.query_expansion_model}",
            f"QUERY_EXPANSION_COUNT={self.query_expansion_count}",
            "",
            "# Preload моделей при старте (только для sentence_transformer)",
            f"PRELOAD_MODELS={'true' if self.preload_models else 'false'}",
        ]
        return "\n".join(lines)

    def resolve_qdrant_location(self) -> str:
        """URL Qdrant-сервера или путь к embedded-хранилищу внутри store_path."""
        return self.qdrant_url or f"{self.store_path}/qdrant"

    def resolve_database_url(self) -> str:
        """DSN Postgres или путь к SQLite-файлу внутри store_path."""
        return self.database_url or f"{self.store_path}/store.db"

    def resolve_model_path(self, model_name: str) -> str:
        """Путь к модели: локальная директория (models/) или имя в HF Hub."""
        if self.models_dir:
            short_name = model_name.rsplit("/", 1)[-1]
            local = os.path.join(self.models_dir, short_name)
            if os.path.isdir(local):
                return local
        return model_name