#!/usr/bin/env python3
"""Reindex all documents — regenerate embeddings after switching embedding provider/model.

Usage:
    uv run python3 scripts/reindex.py

Env:
    Uses EMBEDDING_PROVIDER, EMBEDDING_MODEL, EMBEDDING_BASE_URL etc. from .env
    or the current environment (same as the main RAG server).
"""

import logging
import sys
import time

from src.config import RAGConfig
from src.rag import RAGSystem

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("reindex")


def main() -> int:
    cfg = RAGConfig.from_env()
    logger.info("Reindex config: provider=%s, model=%s, dim=%d, store=%s",
                cfg.embedding_provider, cfg.embedding_model_name,
                cfg.embedding_dim, cfg.store_path)

    rag = RAGSystem(config=cfg)
    try:
        doc_count = rag.doc_store.count()
        logger.info("Documents in store: %d", doc_count)

        if doc_count == 0:
            logger.info("Nothing to reindex — store is empty")
            return 0

        t0 = time.monotonic()
        reindexed = rag.reindex()
        elapsed = time.monotonic() - t0

        logger.info("Reindex complete: %d documents in %.1fs (%.1f doc/s)",
                    reindexed, elapsed, reindexed / elapsed if elapsed else 0)
        return 0
    finally:
        rag.close()


if __name__ == "__main__":
    sys.exit(main())
