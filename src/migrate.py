"""Миграция старых данных (ChromaDB + graph_index.json) → Qdrant + SQLite.

Старый формат (до v0.2):
  • ChromaDB               — {store_path}/chroma.sqlite3 (+ сегментные директории)
  • BM25 pickle/json       — {store_path}/bm25_index.json (пересчитывается, не переносится)
  • Граф                   — {store_path}/graph_index.json (nodes + edges)

Старые векторы НЕ переносятся: размерность старой модели (Ollama 4096d)
несовместима с BGE-M3 (1024d). Каждый документ переиндексируется заново.
Текст, метаданные и рёбра графа сохраняются полностью; doc_id остаются прежними.
"""

import json
import sys
from pathlib import Path
from typing import Optional

_OLD_CHROMA = "chroma.sqlite3"
_OLD_BM25 = "bm25_index.json"
_OLD_GRAPH = "graph_index.json"


def detect_old_data(store_path: str) -> dict:
    """Какие артефакты старого формата есть в store_path."""
    base = Path(store_path)
    return {
        "chroma": (base / _OLD_CHROMA).exists(),
        "bm25": (base / _OLD_BM25).exists(),
        "graph": (base / _OLD_GRAPH).exists(),
    }


def has_old_data(store_path: str) -> bool:
    return any(detect_old_data(store_path).values())


def _read_chroma_documents(store_path: str) -> list[dict]:
    """Прочитать все документы из старой ChromaDB: [{doc_id, text, metadata}]."""
    try:
        import chromadb
    except ImportError:
        raise RuntimeError(
            "chromadb не установлен. Для миграции: uv sync --extra migrate "
            "(или pip install 'graphrag[migrate]')"
        )
    client = chromadb.PersistentClient(path=store_path)
    try:
        collection = client.get_collection("rag_docs")
    except Exception:
        return []
    result = collection.get(include=["documents", "metadatas"])
    ids = result.get("ids") or []
    docs = result.get("documents") or []
    metas = result.get("metadatas") or []
    return [
        {
            "doc_id": ids[i],
            "text": docs[i] if i < len(docs) else "",
            "metadata": metas[i] if i < len(metas) and isinstance(metas[i], dict) else {},
        }
        for i in range(len(ids))
    ]


def _read_old_graph(store_path: str) -> tuple[dict, dict]:
    """Прочитать nodes/edges из graph_index.json старого формата."""
    graph_file = Path(store_path) / _OLD_GRAPH
    if not graph_file.exists():
        return {}, {}
    with open(graph_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("nodes", {}), data.get("edges", {})


def _backup_old_files(store_path: str) -> None:
    base = Path(store_path)
    for name in (_OLD_CHROMA, _OLD_BM25, _OLD_GRAPH):
        path = base / name
        if path.exists():
            path.rename(base / f"{name}.bak")


def run_migration(rag, dry_run: bool = False, force: bool = False) -> Optional[dict]:
    """Перенести старые данные в новые стора через переиндексацию.

    Args:
        rag: инициализированный RAGSystem (новые стора)
        dry_run: только показать, что будет перенесено
        force: не спрашивать подтверждения

    Returns:
        Статистика {documents, edges, skipped_documents, skipped_edges}
        или None при отказе/отсутствии данных.
    """
    store_path = rag.store_path
    found = detect_old_data(store_path)
    if not any(found.values()):
        print(f"Старых данных в {store_path} не найдено — мигрировать нечего.")
        return None

    documents = _read_chroma_documents(store_path) if found["chroma"] else []
    nodes, edges = _read_old_graph(store_path)

    print(f"Найдено: {len(documents)} документов (ChromaDB), "
          f"{len(nodes)} узлов и {len(edges)} рёбер (graph_index.json)")

    if dry_run:
        for doc in documents[:10]:
            print(f"  doc {doc['doc_id'][:8]}...  {doc['text'][:60]!r}")
        if len(documents) > 10:
            print(f"  ... и ещё {len(documents) - 10}")
        return {"documents": len(documents), "edges": len(edges),
                "skipped_documents": 0, "skipped_edges": 0}

    if not force:
        try:
            answer = input("Переиндексировать в новые стора? Старые файлы будут переименованы в *.bak [y/N]: ")
        except EOFError:
            answer = ""
        if answer.strip().lower() not in ("y", "yes"):
            print("Миграция отменена. Запустите с --force чтобы пропустить подтверждение.")
            return None

    migrated_docs = 0
    skipped_docs = 0
    for doc in documents:
        try:
            rag.add_document(doc["text"], doc["metadata"] or None, doc_id=doc["doc_id"])
            migrated_docs += 1
        except ValueError as e:
            # Слишком короткие документы (< MIN_CONTENT_LENGTH) не переносятся
            print(f"  skip {doc['doc_id'][:8]}...: {e}", file=sys.stderr)
            skipped_docs += 1

    valid_ids = rag.doc_store.all_ids()
    migrated_edges = 0
    skipped_edges = 0
    for edge in edges.values():
        source, target = edge.get("source"), edge.get("target")
        if source in valid_ids and target in valid_ids:
            rag.add_relation(source, target, edge.get("relation", "related_to"), edge.get("weight", 1.0))
            migrated_edges += 1
        else:
            skipped_edges += 1

    _backup_old_files(store_path)
    return {
        "documents": migrated_docs,
        "edges": migrated_edges,
        "skipped_documents": skipped_docs,
        "skipped_edges": skipped_edges,
    }
