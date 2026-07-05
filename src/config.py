import os
from dataclasses import dataclass
from typing import Optional


@dataclass
class RAGConfig:
    embedding_provider: str = "ollama"
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "qwen3-embedding:8b"
    ollama_dimension: int = 4096
    openrouter_api_key: Optional[str] = None
    openrouter_model: str = "openai/text-embedding-3-small"
    store_path: str = "./rag_data"
    # Баланс гибридного поиска: 0.0 = чистый BM25, 1.0 = чистый семантический.
    # Значение по умолчанию (0.5 — истинный баланс) выбрано по результатам бенчмарка
    # NDCG@k на детерминированном корпусе с настоящей семантической структурой
    # (см. scripts/benchmark_alpha.py, tests/semantic_mock.py). Бенчмарк выявил
    # широкую «хорошую область» alpha ∈ [0.05, 0.95] с одинаково высоким NDCG@5;
    # за её пределами качество падает: pure BM25 (alpha=0.0) даёт NDCG≈0.36
    # (провал на концептуальных/cross-lingual запросах), pure semantic (alpha=1.0)
    # даёт NDCG≈0.80 (провал на идентификаторах). Ширина плато достигнута за счёт
    # двух улучшений: (1) candidate expansion в search_hybrid, (2) graceful
    # отключение семантического канала для запросов с нулевым эмбеддингом (неизвестные
    # идентификаторы) — такие запросы целиком обслуживает BM25. Дефолт = значение в
    # хорошей области, ближайшее к 0.5 (точке естественного баланса каналов) —
    # робастный и детерминированный выбор.
    default_alpha: float = 0.5
    # Candidate expansion для гибридного поиска: из каждого канала забирается
    # max(k * hybrid_expand, hybrid_min_candidates) кандидатов перед fusion.
    hybrid_expand: int = 3
    hybrid_min_candidates: int = 20

    @classmethod
    def from_env(cls) -> "RAGConfig":
        return cls(
            embedding_provider=os.environ.get("EMBEDDING_PROVIDER", "ollama"),
            ollama_base_url=os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434"),
            ollama_model=os.environ.get("OLLAMA_MODEL", "qwen3-embedding:8b"),
            ollama_dimension=int(os.environ.get("OLLAMA_DIMENSION", "4096")),
            openrouter_api_key=os.environ.get("OPENROUTER_API_KEY"),
            openrouter_model=os.environ.get("OPENROUTER_MODEL", "openai/text-embedding-3-small"),
            store_path=os.environ.get("STORE_PATH", "./rag_data"),
            default_alpha=float(os.environ.get("RAG_DEFAULT_ALPHA", "0.5")),
            hybrid_expand=int(os.environ.get("RAG_HYBRID_EXPAND", "3")),
            hybrid_min_candidates=int(os.environ.get("RAG_HYBRID_MIN_CANDIDATES", "20")),
        )

    def to_env_preview(self) -> str:
        lines = [
            "# Выбор провайдера эмбеддингов: ollama (по умолчанию) или openrouter",
            f"EMBEDDING_PROVIDER={self.embedding_provider}",
            "",
            "# Ollama настройки",
            f"OLLAMA_BASE_URL={self.ollama_base_url}",
            f"OLLAMA_MODEL={self.ollama_model}",
            "",
            "# OpenRouter настройки (нужен API ключ)",
            f"OPENROUTER_API_KEY={self.openrouter_api_key or ''}",
            f"OPENROUTER_MODEL={self.openrouter_model}",
            "",
            "# Путь к хранилищу",
            f"STORE_PATH={self.store_path}",
            "",
            "# Гибридный поиск",
            f"RAG_DEFAULT_ALPHA={self.default_alpha}",
            f"RAG_HYBRID_EXPAND={self.hybrid_expand}",
            f"RAG_HYBRID_MIN_CANDIDATES={self.hybrid_min_candidates}",
        ]
        return "\n".join(lines)