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
        ]
        return "\n".join(lines)