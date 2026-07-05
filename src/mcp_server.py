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
        description="Add a document to the knowledge base",
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
        description="Read and index a file",
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
        description="Semantic search using vector embeddings",
        inputSchema={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query"},
                "k": {"type": "integer", "description": "Number of results", "default": 5},
            },
            "required": ["query"],
        },
    ),
    Tool(
        name="rag_bm25_search",
        description="Keyword search using BM25 algorithm",
        inputSchema={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search keywords"},
                "k": {"type": "integer", "description": "Number of results", "default": 5},
            },
            "required": ["query"],
        },
    ),
    Tool(
        name="rag_search_hybrid",
        description="Hybrid search combining semantic and BM25",
        inputSchema={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query"},
                "k": {"type": "integer", "description": "Number of results", "default": 5},
                "alpha": {"type": "number", "description": "Balance 0=BM25 only, 1=semantic only", "default": 0.5},
            },
            "required": ["query"],
        },
    ),
    Tool(
        name="rag_add_relation",
        description="Create a relation between two documents in the graph",
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
        description="Find documents related to a given node in the graph (BFS)",
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
        description="Get graph statistics",
        inputSchema={"type": "object", "properties": {}},
    ),
    Tool(
        name="rag_stats",
        description="Get storage statistics",
        inputSchema={"type": "object", "properties": {}},
    ),
    Tool(
        name="rag_clear",
        description="Clear all data (DANGEROUS)",
        inputSchema={"type": "object", "properties": {}},
    ),
]


def _fmt(results):
    return [{"doc_id": r[0], "text": r[1][:500], "score": round(r[2], 4), "metadata": r[3]} for r in results]


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
        "rag_search": lambda p: _fmt(rag.search(p.get("query", ""), k=p.get("k", 5))),
        "rag_bm25_search": lambda p: _fmt(rag.bm25_search(p.get("query", ""), k=p.get("k", 5))),
        "rag_search_hybrid": lambda p: _fmt(
            rag.search_hybrid(p.get("query", ""), k=p.get("k", 5), alpha=p.get("alpha", 0.5))
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