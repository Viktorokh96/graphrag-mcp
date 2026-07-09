"""DocumentStore — хранилище документов на SQLite (dev) / Postgres (production).

Источник правды для текстов и метаданных. Векторы живут в Qdrant
(см. vector_store.py), рёбра графа — в таблице graph_edges (см. graph_store.py)
в этой же базе.

Выбор backend'а — по строке подключения:
  • путь к файлу (``./rag_data/store.db``) → SQLite (WAL)
  • ``postgresql://...`` → Postgres (требует extra ``graphrag[postgres]``)
"""

from __future__ import annotations

import json
import sqlite3
import threading
from pathlib import Path
from typing import Any, Optional

from src._meta_filter import matches_metadata_filter

_SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    doc_id TEXT PRIMARY KEY,
    text TEXT NOT NULL,
    metadata TEXT DEFAULT '{}',
    content_hash TEXT,
    parent_doc_id TEXT,
    chunk_index INTEGER,
    created_at TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS graph_edges (
    source_id TEXT NOT NULL,
    target_id TEXT NOT NULL,
    relation TEXT NOT NULL,
    weight REAL DEFAULT 1.0,
    PRIMARY KEY (source_id, target_id, relation),
    FOREIGN KEY (source_id) REFERENCES documents(doc_id) ON DELETE CASCADE,
    FOREIGN KEY (target_id) REFERENCES documents(doc_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_edges_source ON graph_edges(source_id);
CREATE INDEX IF NOT EXISTS idx_edges_target ON graph_edges(target_id);
CREATE INDEX IF NOT EXISTS idx_documents_hash ON documents(content_hash);
CREATE INDEX IF NOT EXISTS idx_documents_parent ON documents(parent_doc_id);
"""

_SCHEMA_PG = _SCHEMA.replace("datetime('now')", "now()::text")


def is_postgres_url(connection: str) -> bool:
    return connection.startswith(("postgresql://", "postgres://", "postgresql+psycopg://"))


class Database:
    """Тонкая обёртка над sqlite3/psycopg с единым placeholder-стилем.

    SQL пишется с ``?`` (стиль sqlite); для Postgres placeholder'ы
    транслируются в ``%s``. Одно соединение, сериализованное локом —
    для MCP-сервера (однопоточный) и FastAPI (пул не нужен на нашем масштабе).
    """

    def __init__(self, connection: str):
        self.connection_string = connection
        self.is_postgres = is_postgres_url(connection)
        self._lock = threading.Lock()
        if self.is_postgres:
            import psycopg

            dsn = connection.replace("postgresql+psycopg://", "postgresql://")
            self._conn = psycopg.connect(dsn, autocommit=True)
            with self._lock, self._conn.cursor() as cur:
                cur.execute(_SCHEMA_PG)
        else:
            path = Path(connection)
            path.parent.mkdir(parents=True, exist_ok=True)
            self._conn = sqlite3.connect(str(path), check_same_thread=False)
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA foreign_keys=ON")
            self._conn.executescript(_SCHEMA)
            self._conn.commit()

    def execute(self, sql: str, params: tuple = ()) -> list[tuple]:
        if self.is_postgres:
            sql = sql.replace("?", "%s")
        with self._lock:
            if self.is_postgres:
                with self._conn.cursor() as cur:
                    cur.execute(sql, params)
                    if cur.description:
                        return cur.fetchall()
                    return []
            cur = self._conn.execute(sql, params)
            rows = cur.fetchall() if cur.description else []
            self._conn.commit()
            return rows

    def executemany(self, sql: str, seq_params: list[tuple]) -> None:
        if self.is_postgres:
            sql = sql.replace("?", "%s")
        with self._lock:
            if self.is_postgres:
                with self._conn.cursor() as cur:
                    cur.executemany(sql, seq_params)
            else:
                self._conn.executemany(sql, seq_params)
                self._conn.commit()

    def close(self) -> None:
        self._conn.close()


class DocumentStore:
    """CRUD документов поверх Database (таблица documents)."""

    def __init__(self, db: Database):
        self._db = db

    def add(
        self,
        doc_id: str,
        text: str,
        metadata: Optional[dict] = None,
        content_hash: Optional[str] = None,
        parent_doc_id: Optional[str] = None,
        chunk_index: Optional[int] = None,
    ) -> str:
        meta_json = json.dumps(metadata or {}, ensure_ascii=False)
        # Upsert: повторный add с тем же doc_id обновляет содержимое
        self._db.execute(
            "INSERT INTO documents (doc_id, text, metadata, content_hash, parent_doc_id, chunk_index) "
            "VALUES (?, ?, ?, ?, ?, ?) "
            "ON CONFLICT (doc_id) DO UPDATE SET "
            "text=excluded.text, metadata=excluded.metadata, content_hash=excluded.content_hash, "
            "parent_doc_id=excluded.parent_doc_id, chunk_index=excluded.chunk_index",
            (doc_id, text, meta_json, content_hash, parent_doc_id, chunk_index),
        )
        return doc_id

    def get(self, doc_id: str) -> Optional[dict]:
        rows = self._db.execute(
            "SELECT doc_id, text, metadata, content_hash, parent_doc_id, chunk_index "
            "FROM documents WHERE doc_id = ?",
            (doc_id,),
        )
        if not rows:
            return None
        return self._row_to_dict(rows[0])

    def get_batch(self, doc_ids: list[str]) -> dict[str, dict]:
        if not doc_ids:
            return {}
        placeholders = ",".join("?" for _ in doc_ids)
        rows = self._db.execute(
            f"SELECT doc_id, text, metadata, content_hash, parent_doc_id, chunk_index "
            f"FROM documents WHERE doc_id IN ({placeholders})",
            tuple(doc_ids),
        )
        return {r[0]: self._row_to_dict(r) for r in rows}

    def get_chunks(self, parent_doc_id: str) -> list[dict]:
        """Все чанки родительского документа в порядке chunk_index."""
        rows = self._db.execute(
            "SELECT doc_id, text, metadata, content_hash, parent_doc_id, chunk_index "
            "FROM documents WHERE parent_doc_id = ? ORDER BY chunk_index",
            (parent_doc_id,),
        )
        return [self._row_to_dict(r) for r in rows]

    def delete(self, doc_id: str) -> bool:
        existed = self.get(doc_id) is not None
        # FK CASCADE удалит рёбра graph_edges автоматически
        self._db.execute("DELETE FROM documents WHERE doc_id = ?", (doc_id,))
        return existed

    def find_by_hash(self, content_hash: str) -> Optional[str]:
        rows = self._db.execute(
            "SELECT doc_id FROM documents WHERE content_hash = ? LIMIT 1",
            (content_hash,),
        )
        return rows[0][0] if rows else None

    def list(
        self,
        limit: int = 20,
        offset: int = 0,
        metadata_filter: Optional[dict] = None,
    ) -> tuple[list[dict], int]:
        """Страница документов + total (с учётом фильтра).

        Фильтр по метаданным — post-filter в Python (устойчив к типам,
        одинаково работает на SQLite и Postgres).
        """
        if not metadata_filter:
            total_rows = self._db.execute("SELECT COUNT(*) FROM documents")
            total = total_rows[0][0]
            rows = self._db.execute(
                "SELECT doc_id, text, metadata, content_hash, parent_doc_id, chunk_index "
                "FROM documents ORDER BY created_at, doc_id LIMIT ? OFFSET ?",
                (limit, offset),
            )
            return [self._row_to_dict(r) for r in rows], total

        rows = self._db.execute(
            "SELECT doc_id, text, metadata, content_hash, parent_doc_id, chunk_index "
            "FROM documents ORDER BY created_at, doc_id"
        )
        matched = [
            d for d in (self._row_to_dict(r) for r in rows)
            if matches_metadata_filter(d["metadata"], metadata_filter)
        ]
        return matched[offset:offset + limit], len(matched)

    def count(self) -> int:
        return self._db.execute("SELECT COUNT(*) FROM documents")[0][0]

    def all_ids(self) -> set[str]:
        return {r[0] for r in self._db.execute("SELECT doc_id FROM documents")}

    def all_hashes(self) -> dict[str, str]:
        """{content_hash: doc_id} для дедупликации при старте."""
        rows = self._db.execute(
            "SELECT content_hash, doc_id FROM documents WHERE content_hash IS NOT NULL"
        )
        return {r[0]: r[1] for r in rows}

    def get_metadata_batch(self, doc_ids: list[str]) -> dict[str, dict]:
        if not doc_ids:
            return {}
        placeholders = ",".join("?" for _ in doc_ids)
        rows = self._db.execute(
            f"SELECT doc_id, metadata FROM documents WHERE doc_id IN ({placeholders})",
            tuple(doc_ids),
        )
        return {r[0]: self._parse_meta(r[1]) for r in rows}

    def clear(self) -> None:
        self._db.execute("DELETE FROM documents")

    @staticmethod
    def _parse_meta(raw: Any) -> dict:
        if isinstance(raw, dict):
            return raw
        try:
            parsed = json.loads(raw) if raw else {}
            return parsed if isinstance(parsed, dict) else {}
        except (json.JSONDecodeError, TypeError):
            return {}

    @classmethod
    def _row_to_dict(cls, row: tuple) -> dict:
        return {
            "doc_id": row[0],
            "text": row[1],
            "metadata": cls._parse_meta(row[2]),
            "content_hash": row[3],
            "parent_doc_id": row[4],
            "chunk_index": row[5],
        }
