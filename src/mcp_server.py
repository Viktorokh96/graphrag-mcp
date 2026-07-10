"""MCP Server for RAG system using MCP Python SDK."""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mcp.server import Server, NotificationOptions
from mcp.server.models import InitializationOptions
from mcp.types import TextContent, Tool

from src._meta_filter import normalize_metadata_filter


TOOL_DEFS = [
    Tool(
        name="rag_add_document",
        description=(
            "Add a text document to the knowledge base. The document is indexed in all "
            "stores: vector (Qdrant, dense embeddings), BM25 (Qdrant sparse vectors), and "
            "the document/graph database. Use this to store any textual knowledge — architectural "
            "decisions, discovered patterns, bug notes, specifications, summaries — that "
            "future searches should retrieve. `meta` is optional and persists verbatim: it "
            "is echoed back unchanged in search/get/list results, so callers can later "
            "filter or annotate results by source, type, project, etc. `meta` accepts a "
            "dict, null, an empty string, a JSON-encoded string (parsed to dict), or any "
            "plain string (wrapped as {'_raw': value}). Returns {doc_id: <uuid4 string>, "
            "duplicate: <bool>}. The `duplicate` flag is true when a document with the same "
            "normalized content hash already exists; in that case `doc_id` is the existing "
            "document's id and no new entry is created. Documents shorter than 50 chars "
            "(after stripping) are rejected with a ValueError."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "text": {
                    "type": "string",
                    "description": (
                        "Text content to index. Stored verbatim in the document store; large "
                        "documents are additionally split into overlapping chunks for vector "
                        "indexing (retrieval still returns the whole document by its doc_id)."
                    ),
                },
                "meta": {
                    "description": (
                        "Optional metadata stored alongside the document and returned verbatim "
                        "in search/get/list results. Accepts: a dict (passed through), null/empty "
                        "string (stored as None), a JSON-encoded string (parsed to dict), or any "
                        "plain string (wrapped as {'_raw': value}). Useful for filtering/annotating "
                        "results by source, type, project, etc."
                    ),
                    "default": None,
                },
                "extract_graph": {
                    "type": "boolean",
                    "description": "Auto-extract entity-relation graph from text via Ollama LLM (Qwen3-4B). Default false.",
                    "default": False,
                },
            },
            "required": ["text"],
        },
    ),
    Tool(
        name="rag_add_file",
        description=(
            "Read a file from disk (UTF-8) and index its full contents as a single document "
            "in the knowledge base (vector + BM25 + graph stores). Useful for bulk-importing "
            "existing Markdown, specifications, notes, or source files. The file is read as one "
            "document — no chunking is performed. `meta` follows the same flexible conventions "
            "as rag_add_document (dict / null / empty / JSON string / plain string). Returns "
            "{doc_id: <uuid4 string>, duplicate: <bool>}; see rag_add_document for the "
            "duplicate semantics. Raises if the file cannot be read (missing path, "
            "permissions, non-UTF-8) or if the content is shorter than 50 chars."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "filepath": {
                    "type": "string",
                    "description": "Absolute or relative path to the file to read and index.",
                },
                "meta": {
                    "description": (
                        "Optional metadata, same conventions as rag_add_document.meta "
                        "(dict / null / empty / JSON string / plain string)."
                    ),
                    "default": None,
                },
                "extract_graph": {
                    "type": "boolean",
                    "description": "If true, extract entity-relation graph via LLM.",
                    "default": False,
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
            "'how does auth work' matches a doc titled 'authentication flow'), including "
            "cross-lingual cases (Russian query matching English docs) when the embedding "
            "model aligns both. Returns the top-k documents ranked by embedding similarity "
            "to `query`. Each result is {doc_id, text, score, metadata, links}, sorted by descending "
            "score. Scores are cosine-similarity-derived (range [0, 1]): score = "
            "clip(1 - L2_distance^2 / 2, 0, 1). Edge cases: returns an empty list if the "
            "store is empty, the query embedding dimension mismatches the store, or the "
            "query embedding is a zero vector (i.e. every word in the query is unknown to "
            "the embedding model — common for bare identifiers like 'pytest' or 'jwt'). "
            "For such identifier-only queries, use rag_bm25_search instead. Pass `max_chars` "
            "to truncate each result's text (recommended to control context size, e.g. "
            "1500-3000); omit it or pass null for full text. Use `k` to set the number of "
            "results (default 5). Pass `metadata_filter` to restrict results to documents "
            "whose metadata matches all given key->value pairs (AND); value may be a scalar "
            "(exact match) or a list ($in). Pass `relations_load_depth` (default 1) to "
            "load graph relations into each result's `links` field (depth 1 = direct "
            "neighbours); `relations_load_type_filter` filters by relation type; "
            "`relations_load_meta_filter` filters neighbour metadata. "
            "Requires an embedding provider (Ollama by default, or OpenRouter via "
            "OPENROUTER_API_KEY)."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query text (natural language)."},
                "k": {"type": "integer", "description": "Maximum number of results to return (1..N). Default 5.", "default": 5},
                "max_chars": {
                    "type": "integer",
                    "description": "Truncate each result's text to at most this many characters. Default 2000. Pass null for full text.",
                    "default": 2000,
                },
                "metadata_filter": {
                    "description": (
                        "Optional filter on document metadata. A dict of key->value pairs; "
                        "ALL must match (AND). Value may be a scalar (exact match) or a list "
                        "(membership/$in). null/omitted = no filter. Example: "
                        "{\"source\": \"specification\", \"type\": [\"bug\",\"feature\"]}."
                    ),
                    "default": None,
                },
                "relations_load_depth": {
                    "type": "integer",
                    "description": "BFS depth for loading graph relations into each result's `links` field. 0 = no relations loaded. 1 (default) = direct neighbours.",
                    "default": 1,
                },
                "relations_load_type_filter": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Only load relations of these types. null/omitted = all types.",
                    "default": None,
                },
                "relations_load_meta_filter": {
                    "description": (
                        "Optional filter on neighbour node metadata, applied when loading relations. "
                        "Same format as metadata_filter. null/omitted = no filter."
                    ),
                    "default": None,
                },
            },
            "required": ["query"],
        },
    ),
    Tool(
        name="rag_bm25_search",
        description=(
            "Keyword search using the BM25 (Okapi) algorithm over tokenized document text. "
            "Best for queries that rely on exact terminology, identifiers, names, or short "
            "technical phrases (e.g. 'Journal Service', 'PROGRESS', 'rag_search_hybrid', "
            "'pytest', 'jwt'). Returns top-k {doc_id, text, score, metadata, links} sorted by "
            "descending BM25 score; results with score <= 0 are filtered out, falling back "
            "to token-overlap ranking if no positive scores exist. Does not require an "
            "embedding provider and works fully offline — making it the reliable channel for "
            "identifier-only queries that semantic search cannot resolve. `max_chars` "
            "truncates each result's text; omit/null for full text. `k` sets the result count "
            "(default 5). Pass `metadata_filter` to restrict results to documents whose "
            "metadata matches all given key->value pairs (AND); value may be a scalar "
            "(exact match) or a list ($in). Pass `relations_load_depth` (default 1) to "
            "load graph relations into each result's `links` field; "
            "`relations_load_type_filter` and `relations_load_meta_filter` further "
            "control which relations are loaded."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search keywords / identifiers / exact terms."},
                "k": {"type": "integer", "description": "Maximum number of results to return. Default 5.", "default": 5},
                "max_chars": {
                    "type": "integer",
                    "description": "Truncate each result's text to at most this many characters. null or omitted = full text.",
                    "default": None,
                },
                "metadata_filter": {
                    "description": (
                        "Optional filter on document metadata. A dict of key->value pairs; "
                        "ALL must match (AND). Value may be a scalar (exact match) or a list "
                        "(membership/$in). null/omitted = no filter. Example: "
                        "{\"source\": \"specification\", \"type\": [\"bug\",\"feature\"]}."
                    ),
                    "default": None,
                },
                "relations_load_depth": {
                    "type": "integer",
                    "description": "BFS depth for loading graph relations into each result's `links` field. 0 = no relations loaded. 1 (default) = direct neighbours.",
                    "default": 1,
                },
                "relations_load_type_filter": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Only load relations of these types. null/omitted = all types.",
                    "default": None,
                },
                "relations_load_meta_filter": {
                    "description": "Optional filter on neighbour node metadata for relations. null/omitted = no filter.",
                    "default": None,
                },
            },
            "required": ["query"],
        },
    ),
    Tool(
        name="rag_search_hybrid",
        description=(
            "Hybrid search combining semantic (vector) and BM25 (keyword) signals via "
            "Reciprocal Rank Fusion (RRF). Recommended default for most queries — it "
            "captures both meaning and exact terms, and degrades gracefully: if one "
            "channel returns nothing (e.g. semantic search for an unknown identifier), "
            "the other channel still ranks candidates. The blend is controlled by `alpha` "
            "(alpha-dilution): each channel is weighted by its share — "
            "RRF score = alpha/(RRF_K+rank_sem+1) + (1-alpha)/(RRF_K+rank_bm25+1) for docs "
            "found by BOTH channels; sem-only docs get alpha/(RRF_K+rank_sem+1); bm25-only "
            "docs get (1-alpha)/(RRF_K+rank_bm25+1). Thus alpha=1.0 = pure semantic "
            "(BM25-only docs excluded), alpha=0.0 = pure BM25 (sem-only docs excluded). "
            "RRF_K=20 (not classic 60) gives wider score spread for small corpora. "
            "RRF is robust to different score scales and eliminates ties. "
            "If `alpha` is omitted/null, language-aware selection applies: queries "
            "containing Cyrillic use cyrillic_alpha (env RAG_CYRILLIC_ALPHA, default 0.85) "
            "because BM25 without Russian stemming is noisy for Russian queries; other "
            "queries use default_alpha (env RAG_DEFAULT_ALPHA, default 0.5 — calibrated by "
            "NDCG@k benchmark, see scripts/benchmark_alpha.py). alpha outside [0, 1] is "
            "clamped to the nearest bound. Candidate expansion: each channel retrieves "
            "max(k * RAG_HYBRID_EXPAND, RAG_HYBRID_MIN_CANDIDATES) candidates (defaults: "
            "max(k*3, 20)) before fusion, so documents relevant by one channel but ranked "
            "beyond top-k in the other are not lost. Returns top-k {doc_id, text, score, "
            "metadata, links}. Pass `max_chars` to truncate each result's text (recommended for "
            "context management); omit or pass null for full text. `k` sets the number of "
            "results (default 5). Pass `metadata_filter` to restrict results to documents "
            "whose metadata matches all given key->value pairs (AND); value may be a scalar "
            "(exact match) or a list ($in). The filter applies to BOTH channels before fusion. "
            "Pass `relations_load_depth` (default 1) to load graph relations into each "
            "result's `links` field; `relations_load_type_filter` and "
            "`relations_load_meta_filter` further control relation loading."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query (natural language, keywords, or mixed)."},
                "k": {"type": "integer", "description": "Maximum number of results to return. Default 5.", "default": 5},
                "alpha": {
                    "type": "number",
                    "description": (
                        "Balance between semantic (1.0) and BM25 (0.0). If omitted/null, "
                        "language-aware: Cyrillic queries use cyrillic_alpha (0.85), others "
                        "use default_alpha (0.5). Clamped to [0, 1]."
                    ),
                    "default": None,
                },
                "max_chars": {
                    "type": "integer",
                    "description": "Truncate each result's text to at most this many characters. null or omitted = full text.",
                    "default": None,
                },
                "metadata_filter": {
                    "description": (
                        "Optional filter on document metadata. A dict of key->value pairs; "
                        "ALL must match (AND). Value may be a scalar (exact match) or a list "
                        "(membership/$in). null/omitted = no filter. Example: "
                        "{\"source\": \"specification\", \"type\": [\"bug\",\"feature\"]}."
                    ),
                    "default": None,
                },
                "relations_load_depth": {
                    "type": "integer",
                    "description": "BFS depth for loading graph relations into each result's `links` field. 0 = no relations loaded. 1 (default) = direct neighbours.",
                    "default": 1,
                },
                "relations_load_type_filter": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Only load relations of these types. null/omitted = all types.",
                    "default": None,
                },
                "relations_load_meta_filter": {
                    "description": "Optional filter on neighbour node metadata for relations. null/omitted = no filter.",
                    "default": None,
                },
                "rerank": {
                    "type": "boolean",
                    "description": "Re-rank final candidates via CrossEncoder (BGE-reranker-v2-m3). null/omitted = use RERANK_ENABLED env.",
                    "default": None,
                },
                "query_expansion": {
                    "type": "boolean",
                    "description": "Generate alternative query phrasings via a small LLM and merge results via RRF. null/omitted = use QUERY_EXPANSION_ENABLED env.",
                    "default": None,
                },
            },
            "required": ["query"],
        },
    ),
    Tool(
        name="rag_add_relation",
        description=(
            "Create a directed, typed, weighted edge between two documents in the "
            "knowledge graph. Both endpoints must already exist as documents (create them "
            "first via rag_add_document / rag_add_file). `relation` is an arbitrary string "
            "describing the link, e.g. 'related_to', 'similar_to', 'prerequisite', "
            "'supersedes', 'competitor'. `weight` (default 1.0) can bias graph expansion and "
            "is preserved verbatim in rag_get_related output. Multiple edges between the "
            "same pair with different relation types are allowed; creating the same "
            "(source, target, relation) edge again overwrites the weight. Returns "
            "{status: 'ok'}."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "source_id": {"type": "string", "description": "doc_id of the source document (must already exist)."},
                "target_id": {"type": "string", "description": "doc_id of the target document (must already exist)."},
                "relation": {"type": "string", "description": "Relation type label (arbitrary string, e.g. 'related_to', 'prerequisite', 'supersedes')."},
                "weight": {"type": "number", "default": 1.0, "description": "Edge weight preserved in rag_get_related output (default 1.0)."},
            },
            "required": ["source_id", "target_id", "relation"],
        },
    ),
    Tool(
        name="rag_get_related",
        description=(
            "Find documents connected to a given node via bidirectional breadth-first "
            "search (BFS) over the knowledge graph. `max_depth` controls how many hops to "
            "traverse: 1 (default) returns direct neighbours, 2 returns neighbours-of-"
            "neighbours, etc. Returns {relations: [{source, target, relation, weight, "
            "direction}, ...]} covering all edges traversed (each edge's source/target are "
            "doc_ids). `direction` is 'out' for source->target or 'in' for target->source. "
            "Useful for discovering related documents that do not textually match a query "
            "but are linked semantically through explicit relations. Returns an empty "
            "`relations` list if the node is unknown or isolated. Pass `metadata_filter` "
            "to keep only edges whose neighbour node's metadata matches all given key->value "
            "pairs (AND); value may be a scalar (exact match) or a list ($in)."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "node_id": {"type": "string", "description": "doc_id of the node to start BFS from."},
                "max_depth": {"type": "integer", "default": 1, "description": "Maximum BFS hop count (1 = direct neighbours, 2 = neighbours-of-neighbours, ...)."},
                "metadata_filter": {
                    "description": (
                        "Optional filter on neighbour metadata. A dict of key->value pairs; "
                        "ALL must match (AND). Value may be a scalar (exact match) or a list "
                        "(membership/$in). null/omitted = no filter. Edges whose neighbour "
                        "node does not pass the filter are excluded."
                    ),
                    "default": None,
                },
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
            "verify that expected relations exist. No parameters. Returns "
            "{total_nodes: int, total_edges: int, relation_types: [str, ...]}."
        ),
        inputSchema={"type": "object", "properties": {}},
    ),
    Tool(
        name="rag_stats",
        description=(
            "Return storage statistics for the knowledge base: total document count, the "
            "on-disk store path, and the embedding dimension in use. Useful for sanity "
            "checks (e.g. 'is the store empty?', 'which provider dimension is active?'). "
            "No parameters. Returns {total_documents: int, store_path: str, dimension: int}."
        ),
        inputSchema={"type": "object", "properties": {}},
    ),
    Tool(
        name="rag_clear",
        description=(
            "DANGEROUS — irreversibly delete ALL data from every store (vector, BM25, and "
            "graph). The on-disk files under rag_data/ are wiped. There is no undo and no "
            "confirmation prompt. Use only when you intend to fully reset the knowledge "
            "base (e.g. fresh reindex). Returns {status: 'ok'}."
        ),
        inputSchema={"type": "object", "properties": {}},
    ),
    Tool(
        name="rag_delete_document",
        description=(
            "Delete a single document by its ID from all stores (vector, BM25, graph). "
            "Idempotent — calling with an unknown or already-deleted doc_id is safe and "
            "returns deleted=false. Prefer this over rag_clear when removing individual "
            "stale or erroneous entries. Returns {status: 'ok', doc_id: <id>, deleted: bool} "
            "where `deleted` is true only if the document actually existed and was removed."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "doc_id": {"type": "string", "description": "doc_id (UUID4) of the document to delete."},
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
            "document — it is cheaper and deterministic. Returns the document record "
            "{doc_id, text, metadata, total_chars, offset, limit, links} on success, or null if "
            "the document is not found. To page through a long doc: call with offset=0, "
            "limit=N; then offset=N, limit=N; etc., until offset >= total_chars. "
            "Pass `relations_load_depth` (default 1) to load graph relations into the "
            "result's `links` field; `relations_load_type_filter` and "
            "`relations_load_meta_filter` further control relation loading."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "doc_id": {"type": "string", "description": "doc_id (UUID4) of the document to retrieve."},
                "offset": {"type": "integer", "description": "Character offset to start reading from (default 0).", "default": 0},
                "limit": {"type": "integer", "description": "Maximum characters to return from offset. null or omitted = full text from offset to end.", "default": None},
                "relations_load_depth": {
                    "type": "integer",
                    "description": "BFS depth for loading graph relations into each result's `links` field. 0 = no relations loaded. 1 (default) = direct neighbours.",
                    "default": 1,
                },
                "relations_load_type_filter": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Only load relations of these types. null/omitted = all types.",
                    "default": None,
                },
                "relations_load_meta_filter": {
                    "description": "Optional filter on neighbour node metadata for relations. null/omitted = no filter.",
                    "default": None,
                },
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
            "omit or pass null for full text. Use this to browse the corpus, audit what "
            "has been indexed, or discover doc_ids for subsequent get_document / delete / "
            "relation calls. Pass `metadata_filter` to restrict the listing to documents "
            "whose metadata matches all given key->value pairs (AND); value may be a scalar "
            "(exact match) or a list ($in). When a filter is active, `total` reflects the "
            "number of matching documents. Returns {documents: [{doc_id, text, metadata, links}, "
            "...], total: int, limit: int, offset: int}. Pass `relations_load_depth` "
            "(default 1) to load graph relations into each document's `links` field; "
            "`relations_load_type_filter` and `relations_load_meta_filter` further "
            "control relation loading."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "limit": {"type": "integer", "description": "Page size (number of documents per page). Default 20.", "default": 20},
                "offset": {"type": "integer", "description": "Index of the first document to return (for pagination). Default 0.", "default": 0},
                "max_chars": {
                    "type": "integer",
                    "description": "Truncate each result's text to at most this many characters. null or omitted = return full text. Recommended for context budget control.",
                    "default": None,
                },
                "metadata_filter": {
                    "description": (
                        "Optional filter on document metadata. A dict of key->value pairs; "
                        "ALL must match (AND). Value may be a scalar (exact match) or a list "
                        "(membership/$in). null/omitted = no filter. Example: "
                        "{\"source\": \"specification\", \"type\": [\"bug\",\"feature\"]}."
                    ),
                    "default": None,
                },
                "relations_load_depth": {
                    "type": "integer",
                    "description": "BFS depth for loading graph relations into each result's `links` field. 0 = no relations loaded. 1 (default) = direct neighbours.",
                    "default": 1,
                },
                "relations_load_type_filter": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Only load relations of these types. null/omitted = all types.",
                    "default": None,
                },
                "relations_load_meta_filter": {
                    "description": "Optional filter on neighbour node metadata for relations. null/omitted = no filter.",
                    "default": None,
                },
            },
        },
    ),
    Tool(
        name="rag_add_structured",
        description=(
            "Index a structured code repository in repomix JSON format. "
            "The JSON must contain 'repository' (string), 'structure' (list of file paths), "
            "and 'files' (dict of path -> {content, language?, size?}). "
            "Each file is indexed as a separate document with metadata "
            "{source, path, language, type: file}. The directory tree is indexed as "
            "a 'structure' document. Files in the same directory are auto-linked "
            "with 'sibling' relations. Returns {status, structure_doc_id, "
            "file_doc_ids: dict[path->doc_id], files_count, errors}."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "content": {
                    "type": "string",
                    "description": "JSON string in repomix format.",
                },
                "extract_graph": {
                    "type": "boolean",
                    "description": "Extract entity-relation graph from each file (requires Ollama). Default false.",
                    "default": False,
                },
            },
            "required": ["content"],
        },
    ),
]


def _fmt(results, max_chars=None):
    if max_chars is not None:
        return [
            {
                "doc_id": r[0],
                "text": r[1][:max_chars],
                "score": round(r[2], 4),
                "metadata": r[3] if isinstance(r[3], dict) else {},
            }
            for r in results
        ]
    return [
        {
            "doc_id": r[0],
            "text": r[1],
            "score": round(r[2], 4),
            "metadata": r[3] if isinstance(r[3], dict) else {},
        }
        for r in results
    ]


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
        except (json.JSONDecodeError, TypeError):
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

    def _add_document_handler(p):
        text = p["text"]
        meta = _parse_meta(p.get("meta"))
        existing = rag.is_duplicate(text)
        doc_id = rag.add_document(text, meta, extract_graph=p.get("extract_graph", False))
        return {"doc_id": doc_id, "duplicate": existing is not None}

    def _add_file_handler(p):
        filepath = p["filepath"]
        meta = _parse_meta(p.get("meta"))
        with open(filepath, "r", encoding="utf-8-sig") as f:
            text = f.read()
        existing = rag.is_duplicate(text)
        doc_id = rag.add_document(text, meta, extract_graph=p.get("extract_graph", False))
        return {"doc_id": doc_id, "duplicate": existing is not None}

    def _enrich(p, docs):
        return rag._enrich_with_links(
            docs,
            relations_load_depth=p.get("relations_load_depth", 1),
            relations_load_type_filter=p.get("relations_load_type_filter"),
            relations_load_meta_filter=normalize_metadata_filter(p.get("relations_load_meta_filter")),
        )

    handlers = {
        "rag_add_document": _add_document_handler,
        "rag_add_file": _add_file_handler,
        "rag_search": lambda p: _enrich(p, _fmt(
            rag.search(p.get("query", ""), k=p.get("k", 5), metadata_filter=normalize_metadata_filter(p.get("metadata_filter"))),
            max_chars=p.get("max_chars"),
        )),
        "rag_bm25_search": lambda p: _enrich(p, _fmt(
            rag.bm25_search(p.get("query", ""), k=p.get("k", 5), metadata_filter=normalize_metadata_filter(p.get("metadata_filter"))),
            max_chars=p.get("max_chars"),
        )),
        "rag_search_hybrid": lambda p: _enrich(p, _fmt(
            rag.search_hybrid(
                p.get("query", ""), k=p.get("k", 5), alpha=p.get("alpha"),
                metadata_filter=normalize_metadata_filter(p.get("metadata_filter")),
                rerank=p.get("rerank"),
                query_expansion=p.get("query_expansion"),
            ),
            max_chars=p.get("max_chars"),
        )),
        "rag_add_relation": lambda p: (
            rag.add_relation(p["source_id"], p["target_id"], p["relation"], p.get("weight", 1.0)),
            {"status": "ok"},
        )[1],
        "rag_get_related": lambda p: {
            "relations": [
                {"source": r[0], "target": r[1], "relation": r[2], "weight": r[3], "direction": r[4]}
                for r in rag.get_related(
                    p["node_id"], p.get("max_depth", 1),
                    metadata_filter=normalize_metadata_filter(p.get("metadata_filter")),
                )
            ]
        },
        "rag_get_document": lambda p: rag.get_document(
            p["doc_id"],
            offset=p.get("offset", 0), limit=p.get("limit"),
            relations_load_depth=p.get("relations_load_depth", 1),
            relations_load_type_filter=p.get("relations_load_type_filter"),
            relations_load_meta_filter=normalize_metadata_filter(p.get("relations_load_meta_filter")),
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
            limit=p.get("limit", 20), offset=p.get("offset", 0),
            max_chars=p.get("max_chars"),
            metadata_filter=normalize_metadata_filter(p.get("metadata_filter")),
            relations_load_depth=p.get("relations_load_depth", 1),
            relations_load_type_filter=p.get("relations_load_type_filter"),
            relations_load_meta_filter=normalize_metadata_filter(p.get("relations_load_meta_filter")),
        ),
        "rag_add_structured": lambda p: rag.index_structured(
            p["content"], extract_graph=p.get("extract_graph", False),
        ),
    }
    fn = handlers.get(name)
    if not fn:
        raise ValueError(f"Unknown tool: {name}")
    return fn(arguments)


def main():
    from src.rag import RAGSystem
    from src.config import RAGConfig
    from src.migrate import has_old_data

    config = RAGConfig.from_env()
    if has_old_data(config.store_path):
        print(
            f"WARNING: Found old data in {config.store_path} (chroma.sqlite3 / graph_index.json).\n"
            "These are NOT compatible with the new store (Qdrant + SQLite).\n"
            "Run `rag-server migrate` to re-index them, or delete them manually.",
            file=sys.stderr,
        )
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

    # Graceful shutdown: закрываем Qdrant (файловые locks) и SQLite (WAL) при
    # выходе — иначе при перезапуске возможны locked-ошибки. KeyboardInterrupt
    # (Ctrl+C) и штатное завершение обрабатываются одинаково.
    try:
        asyncio.run(_run())
    except KeyboardInterrupt:
        pass
    finally:
        rag.close()


if __name__ == "__main__":
    main()