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
    # Провайдер эмбеддингов: bge-m3 (локально, дефолт) | ollama | openrouter.
    # Env: EMBEDDING_MODEL (приоритет) или EMBEDDING_PROVIDER (legacy-алиас).
    embedding_provider: str = "bge-m3"
    # BGE-M3: локальная мультиязычная модель (sentence-transformers)
    bge_model_name: str = "BAAI/bge-m3"
    embedding_dim: int = 1024
    embedding_device: str = "cpu"
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "qwen3-embedding:8b"
    ollama_dimension: int = 4096
    openrouter_api_key: Optional[str] = None
    openrouter_model: str = "openai/text-embedding-3-small"
    openrouter_dimension: int = 1536
    store_path: str = "./rag_data"
    # Хранилища (Фаза 2). Пустые значения → embedded-режим внутри store_path:
    #   qdrant_url:   URL Qdrant-сервера (prod) | "" → embedded {store_path}/qdrant
    #   database_url: postgresql://... (prod)   | "" → SQLite {store_path}/store.db
    qdrant_url: str = ""
    database_url: str = ""
    # Баланс гибридного поиска: 0.0 = чистый BM25, 1.0 = чистый семантический.
    # Значение по умолчанию (0.5 — истинный баланс) выбрано по результатам бенчмарка
    # NDCG@k на детерминированном корпусе с настоящей семантической структурой
    # (см. scripts/benchmark_alpha.py, tests/semantic_mock.py). После перехода на
    # RRF с alpha-dilution (RRF_K=20) бенчмарк выявил широкую «хорошую область»
    # alpha ∈ [0.05, 0.75] с NDCG@5=0.8241 и P@1=0.8929; за её пределами качество
    # падает: pure BM25 (alpha=0.0) даёт NDCG≈0.69 (провал на концептуальных/
    # cross-lingual запросах), pure semantic (alpha=1.0) даёт NDCG≈0.80 (провал
    # на идентификаторах). Дефолт = значение в хорошей области, ближайшее к 0.5
    # (точке естественного баланса каналов) — робастный и детерминированный выбор.
    default_alpha: float = 0.5
    # Alpha для запросов с кириллицей (русский и др.). BM25 без русского стемминга
    # даёт шумовый сигнал для русских запросов (морфология, отсутствие лемматизации),
    # поэтому семантический канал должен доминировать. Бенчмарк NDCG@k на mock-корпусе
    # с идеальными cross-lingual эмбеддингами показывает широкое плато alpha ∈ [0.05, 0.75]
    # (см. scripts/benchmark_alpha.py) — 0.85 лежит за краем, но это оправдано для реальных
    # (не идеальных) мультиязычных эмбеддингов Ollama, где BM25-канал для русских
    # концептуальных запросов вносит больше шума, чем сигнала. Проверено эмпирически:
    # alpha=0.85 поднимает Tests Agent в топ-2 для запроса «агент тестирования кода»
    # (при alpha=0.5 документ отсутствует в топ-5).
    cyrillic_alpha: float = 0.85
    # Candidate expansion для гибридного поиска: из каждого канала забирается
    # max(k * hybrid_expand, hybrid_min_candidates) кандидатов перед fusion.
    hybrid_expand: int = 3
    hybrid_min_candidates: int = 20
    # Чанкование больших документов: текст длиннее chunk_size токенов режется на
    # чанки (paragraph→sentence→token) и индексируется несколькими точками в
    # Qdrant под общим doc_id. Решает обрезку хвоста длинных документов эмбеддером
    # (BGE-M3 max ~8192 токенов). Документ в DocumentStore остаётся цельным.
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
            embedding_provider=os.environ.get(
                "EMBEDDING_MODEL", os.environ.get("EMBEDDING_PROVIDER", "bge-m3")
            ),
            bge_model_name=os.environ.get("BGE_MODEL_NAME", "BAAI/bge-m3"),
            embedding_dim=_env_int("EMBEDDING_DIM", "1024"),
            embedding_device=os.environ.get("EMBEDDING_DEVICE", "cpu"),
            ollama_base_url=os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434"),
            ollama_model=os.environ.get("OLLAMA_MODEL", "qwen3-embedding:8b"),
            ollama_dimension=_env_int("OLLAMA_DIMENSION", "4096"),
            openrouter_api_key=os.environ.get("OPENROUTER_API_KEY"),
            openrouter_model=os.environ.get("OPENROUTER_MODEL", "openai/text-embedding-3-small"),
            openrouter_dimension=_env_int("OPENROUTER_DIMENSION", "1536"),
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

    def resolve_qdrant_location(self) -> str:
        """URL Qdrant-сервера или путь к embedded-хранилищу внутри store_path."""
        return self.qdrant_url or f"{self.store_path}/qdrant"

    def resolve_database_url(self) -> str:
        """DSN Postgres или путь к SQLite-файлу внутри store_path."""
        return self.database_url or f"{self.store_path}/store.db"

    def to_env_preview(self) -> str:
        lines = [
            "# Выбор провайдера эмбеддингов: bge-m3 (по умолчанию), ollama или openrouter",
            f"EMBEDDING_MODEL={self.embedding_provider}",
            "",
            "# BGE-M3 настройки (локальная модель, sentence-transformers)",
            f"BGE_MODEL_NAME={self.bge_model_name}",
            f"EMBEDDING_DIM={self.embedding_dim}",
            f"EMBEDDING_DEVICE={self.embedding_device}",
            "",
            "# Ollama настройки",
            f"OLLAMA_BASE_URL={self.ollama_base_url}",
            f"OLLAMA_MODEL={self.ollama_model}",
            "",
            "# OpenRouter настройки (нужен API ключ)",
            f"OPENROUTER_API_KEY={self.openrouter_api_key or ''}",
            f"OPENROUTER_MODEL={self.openrouter_model}",
            f"OPENROUTER_DIMENSION={self.openrouter_dimension}",
            "",
            "# Путь к хранилищу (embedded-режим: qdrant/ и store.db внутри)",
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
            "# Reranker (Cross-encoder, ~1GB)",
            f"RERANK_ENABLED={'true' if self.rerank_enabled else 'false'}",
            f"RERANK_MODEL={self.rerank_model}",
            f"RERANK_DEVICE={self.rerank_device}",
            f"RERANK_TOP_K_MULTIPLIER={self.rerank_top_k_multiplier}",
            "",
            "# Query expansion (малая LLM для парафраза запросов)",
            f"QUERY_EXPANSION_ENABLED={'true' if self.query_expansion_enabled else 'false'}",
            f"QUERY_EXPANSION_MODEL={self.query_expansion_model}",
            f"QUERY_EXPANSION_COUNT={self.query_expansion_count}",
        ]
        return "\n".join(lines)