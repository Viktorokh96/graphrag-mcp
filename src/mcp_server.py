"""MCP Server for RAG system using MCP Python SDK."""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mcp.server import Server, NotificationOptions
from mcp.server.models import InitializationOptions
from mcp.types import TextContent, Tool


TOOL_DEFS = [
    Tool(
        name="rag_add_document",
        description=(
            "Add a text document to the knowledge base. The document is indexed in all "
            "three stores: vector (ChromaDB, via embeddings), BM25 (keyword index), and "
            "graph (as a new node). Returns the generated doc_id. Use this to store any "
            "textual knowledge — architectural decisions, discovered patterns, bug notes, "
            "specifications, summaries — that future searches should retrieve. Optional "
            "`meta` accepts a dict, null, an empty string, a JSON string, or any plain "
            "string (which is wrapped as {'_raw': value}); it is stored verbatim and "
            "echoed back in search results."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "text": {"type": "string", "description": "Text content to add"},
                "meta": {
                    "description": "Optional metadata",
                    "default": None,
                },
            },
            "required": ["text"],
        },
    ),
    Tool(
        name="rag_add_file",
        description=(
            "Read a file from disk and index it as a single document in the knowledge base "
            "(vector + BM25 + graph stores). Useful for bulk-importing existing Markdown, "
            "specifications, notes, or source files. Returns the generated doc_id and the "
            "filepath. Optional `meta` follows the same flexible conventions as "
            "rag_add_document (dict / null / empty / JSON string / plain string)."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "filepath": {"type": "string", "description": "Path to the file"},
                "meta": {
                    "description": "Optional metadata",
                    "default": None,
                },
            },
            "required": ["filepath"],
        },
    ),
    Tool(
        name="rag_search",
        description=(
            "Semantic search over the knowledge base using vector embeddings. Best for "
            "conceptual, meaning-based queries where exact wording may differ (e.g. "
            "'how does auth work' matches a doc titled 'authentication flow'). Returns "
            "the top-k documents ranked by embedding similarity to `query`. Each result "
            "is {doc_id, text, score, metadata}. Pass `max_chars` to truncate each result's "
            "text (recommended to control context size, e.g. 1500-3000); omit it or pass "
            "null for full text. Use `k` to set the number of results (default 5). "
            "Requires an embedding provider (Ollama by default, or OpenRouter); if "
            "unavailable, falls back to TF-IDF heuristics."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query"},
                "k": {"type": "integer", "description": "Number of results", "default": 5},
                "max_chars": {"type": "integer", "description": "Truncate document text to this many characters. null or omitted = full text", "default": None},
            },
            "required": ["query"],
        },
    ),
    Tool(
        name="rag_bm25_search",
        description=(
            "Keyword search using the BM25 (Okapi) algorithm over tokenized document text. "
            "Best for queries that rely on exact terminology, identifiers, names, or short "
            "technical phrases (e.g. 'Journal Service', 'PROGRESS', 'rag_search_hybrid'). "
            "Returns top-k {doc_id, text, score, metadata} sorted by BM25 relevance. Does "
            "not require an embedding provider and works fully offline. `max_chars` "
            "truncates each result's text; omit/null for full text. `k` sets the result "
            "count (default 5)."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search keywords"},
                "k": {"type": "integer", "description": "Number of results", "default": 5},
                "max_chars": {"type": "integer", "description": "Truncate document text to this many characters. null or omitted = full text", "default": None},
            },
            "required": ["query"],
        },
    ),
    Tool(
        name="rag_search_hybrid",
        description=(
            "Hybrid search combining semantic (vector) and BM25 (keyword) signals into a "
            "single ranked list. Recommended default for most queries — it captures both "
            "meaning and exact terms. The blend is controlled by `alpha`: "
            "alpha=0.0 = pure BM25, alpha=1.0 = pure semantic, alpha=0.5 (default) = "
            "balanced. Formula: normalized_score = alpha * semantic_score + (1 - alpha) * "
            "bm25_score. Returns top-k {doc_id, text, score, metadata}. Pass `max_chars` "
            "to truncate each result's text (recommended for context management); omit "
            "or pass null for full text. `k` sets the number of results (default 5)."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query"},
                "k": {"type": "integer", "description": "Number of results", "default": 5},
                "alpha": {"type": "number", "description": "Balance 0=BM25 only, 1=semantic only", "default": 0.5},
                "max_chars": {"type": "integer", "description": "Truncate document text to this many characters. null or omitted = full text", "default": None},
            },
            "required": ["query"],
        },
    ),
    Tool(
        name="rag_add_relation",
        description=(
            "Create a directed, typed, weighted edge between two documents in the "
            "knowledge graph. Both endpoints must already exist as documents (create them "
            "first via rag_add_document / rag_add_file). `relation` is an arbitrary "
            "string describing the link, e.g. 'related_to', 'similar_to', "
            "'prerequisite', 'supersedes', 'competitor'. `weight` (default 1.0) can bias "
            "graph expansion and is preserved in get_related output. Multiple edges between "
            "the same pair with different relation types are allowed. Returns {status: ok}."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "source_id": {"type": "string"},
                "target_id": {"type": "string"},
                "relation": {"type": "string", "description": "Relation type (e.g. related_to, similar_to)"},
                "weight": {"type": "number", "default": 1.0},
            },
            "required": ["source_id", "target_id", "relation"],
        },
    ),
    Tool(
        name="rag_get_related",
        description=(
            "Find documents connected to a given node via breadth-first search (BFS) over "
            "the knowledge graph. `max_depth` controls how many hops to traverse: 1 "
            "(default) returns direct neighbours, 2 returns neighbours-of-neighbours, etc. "
            "Returns {relations: [{source, target, relation, weight}, ...]} covering all "
            "edges traversed. Useful for discovering related documents that do not "
            "textually match a query but are linked semantically through explicit "
            "relations. Returns empty `relations` if the node is unknown or isolated."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "node_id": {"type": "string"},
                "max_depth": {"type": "integer", "default": 1},
            },
            "required": ["node_id"],
        },
    ),
    Tool(
        name="rag_graph_stats",
        description=(
            "Return statistics about the knowledge graph: total node count, total edge "
            "count, and the list of distinct relation types currently in use. Use this to "
            "inspect graph health, audit which relation labels have been applied, or "
            "verify that expected relations exist. No parameters."
        ),
        inputSchema={"type": "object", "properties": {}},
    ),
    Tool(
        name="rag_stats",
        description=(
            "Return storage statistics for the knowledge base: total document count, the "
            "on-disk store path, and the embedding dimension in use. Useful for sanity "
            "checks (e.g. 'is the store empty?', 'which provider dimension is active?'). "
            "No parameters."
        ),
        inputSchema={"type": "object", "properties": {}},
    ),
    Tool(
        name="rag_clear",
        description=(
            "DANGEROUS — irreversibly delete ALL data from every store (vector, BM25, and "
            "graph). The on-disk files under rag_data/ are wiped. There is no undo and no "
            "confirmation prompt. Use only when you intend to fully reset the knowledge "
            "base (e.g. fresh reindex). Returns {status: ok}."
        ),
        inputSchema={"type": "object", "properties": {}},
    ),
    Tool(
        name="rag_delete_document",
        description=(
            "Delete a single document by its ID from all stores (vector, BM25, graph). "
            "Idempotent — calling with an unknown or already-deleted doc_id is safe and "
            "returns deleted=false. Prefer this over rag_clear when removing individual "
            "stale or erroneous entries. Returns {status, doc_id, deleted} where `deleted` "
            "is a boolean indicating whether the document was actually removed."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "doc_id": {"type": "string", "description": "Document ID to delete"},
            },
            "required": ["doc_id"],
        },
    ),
    Tool(
        name="rag_get_document",
        description=(
            "Retrieve a single document by its ID, including the full text and metadata. "
            "Supports character-level pagination for large documents via `offset` (start "
            "position in characters, default 0) and `limit` (maximum characters to return; "
            "null/omitted returns the full text from offset onwards). Use this instead of "
            "re-running a search with a larger max_chars when you need more of a known "
            "document — it is cheaper and deterministic. Returns the document record or "
            "an indication if not found."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "doc_id": {"type": "string", "description": "Document ID"},
                "offset": {"type": "integer", "description": "Character offset to start reading from (default 0)", "default": 0},
                "limit": {"type": "integer", "description": "Maximum characters to return. null or omitted = full text from offset", "default": None},
            },
            "required": ["doc_id"],
        },
    ),
    Tool(
        name="rag_list_documents",
        description=(
            "List documents in the knowledge base with pagination. `limit` is the page "
            "size (default 20), `offset` is the starting index (default 0). `max_chars` "
            "optionally truncates each returned document's text to keep responses compact; "
            "omit or pass null for full text. Use this to browse the corpus, audit what has "
            "been indexed, or discover doc_ids for subsequent get_document / delete / "
            "relation calls. Returns a list of document records."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "limit": {"type": "integer", "description": "Page size (default 20)", "default": 20},
                "offset": {"type": "integer", "description": "Offset from start (default 0)", "default": 0},
                "max_chars": {"type": "integer", "description": "Truncate document text to this many characters. null or omitted = full text", "default": None},
            },
        },
    ),
]


def _fmt(results, max_chars=None):
    if max_chars is not None:
        return [{"doc_id": r[0], "text": r[1][:max_chars], "score": round(r[2], 4), "metadata": r[3]} for r in results]
    return [{"doc_id": r[0], "text": r[1], "score": round(r[2], 4), "metadata": r[3]} for r in results]


def _parse_meta(value):
    """Нормализовать значение поля `meta` из аргументов MCP-вызова.

    Допускает: dict (проходит как есть), None/"", строку с JSON (парсится),
    любую строку без JSON (оборачивается в {"_raw": value}).
    Возвращает None для отсутствующего/пустого значения.
    """
    if value is None or value == "":
        return None
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            return parsed if isinstance(parsed, dict) else {"_raw": value}
        except Exception:
            return {"_raw": value}
    return {"_raw": value}


def handle_tool_call(rag, name: str, arguments: dict) -> dict:
    """Синхронный обработчик MCP tool call. Тестируется напрямую без stdio.

    Args:
        rag: экземпляр RAGSystem
        name: имя инструмента
        arguments: словарь аргументов (значения уже отвалидированы схемой)

    Returns:
        dict-результат, который будет сериализован в TextContent
    """
    if arguments is None:
        arguments = {}

    handlers = {
        "rag_add_document": lambda p: {"doc_id": rag.add_document(p["text"], _parse_meta(p.get("meta")))},
        "rag_add_file": lambda p: {"doc_id": rag.add_file(p["filepath"], _parse_meta(p.get("meta")))},
        "rag_search": lambda p: _fmt(rag.search(p.get("query", ""), k=p.get("k", 5)), max_chars=p.get("max_chars")),
        "rag_bm25_search": lambda p: _fmt(rag.bm25_search(p.get("query", ""), k=p.get("k", 5)), max_chars=p.get("max_chars")),
        "rag_search_hybrid": lambda p: _fmt(
            rag.search_hybrid(p.get("query", ""), k=p.get("k", 5), alpha=p.get("alpha", 0.5)), max_chars=p.get("max_chars")
        ),
        "rag_add_relation": lambda p: (
            rag.add_relation(p["source_id"], p["target_id"], p["relation"], p.get("weight", 1.0)),
            {"status": "ok"},
        )[1],
        "rag_get_related": lambda p: {
            "relations": [
                {"source": r[0], "target": r[1], "relation": r[2], "weight": r[3]}
                for r in rag.get_related(p["node_id"], p.get("max_depth", 1))
            ]
        },
        "rag_get_document": lambda p: rag.get_document(
            p["doc_id"], offset=p.get("offset", 0), limit=p.get("limit")
        ),
        "rag_graph_stats": lambda p: (
            s := rag.stats(),
            {
                "total_nodes": s.get("total_nodes", 0),
                "total_edges": s.get("total_edges", 0),
                "relation_types": s.get("relation_types", []),
            },
        )[1],
        "rag_stats": lambda p: (
            s := rag.stats(),
            {"total_documents": s["total_documents"], "store_path": s["store_path"], "dimension": s["dimension"]},
        )[1],
        "rag_clear": lambda p: (rag.clear(), {"status": "ok"})[1],
        "rag_delete_document": lambda p: {"status": "ok", "doc_id": p["doc_id"], "deleted": rag.delete_document(p["doc_id"])},
        "rag_list_documents": lambda p: rag.list_documents(
            limit=p.get("limit", 20), offset=p.get("offset", 0), max_chars=p.get("max_chars")
        ),
    }
    fn = handlers.get(name)
    if not fn:
        raise ValueError(f"Unknown tool: {name}")
    return fn(arguments)


def main():
    from src.rag import RAGSystem
    from src.config import RAGConfig

    config = RAGConfig.from_env()
    rag = RAGSystem(config=config)
    server = Server("rag-knowledge-base")

    @server.list_tools()
    async def list_tools() -> list[Tool]:
        return TOOL_DEFS

    @server.call_tool()
    async def call_tool(name: str, arguments: dict) -> list[TextContent]:
        result = handle_tool_call(rag, name, arguments)
        return [TextContent(type="text", text=json.dumps(result, ensure_ascii=False))]

    import asyncio
    from mcp.server.stdio import stdio_server

    async def _run():
        async with stdio_server() as (read_stream, write_stream):
            await server.run(
                read_stream,
                write_stream,
                InitializationOptions(
                    server_name="rag-knowledge-base",
                    server_version="1.0.0",
                    capabilities=server.get_capabilities(
                        notification_options=NotificationOptions(),
                        experimental_capabilities={},
                    ),
                ),
            )

    asyncio.run(_run())


if __name__ == "__main__":
    main()