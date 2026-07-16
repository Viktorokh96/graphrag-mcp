"""CLI интерфейс для RAG системы."""

import argparse
import json
import os
import sys
from typing import Optional

from src.rag import RAGSystem


def main(argv: Optional[list[str]] = None) -> int:
    """
    CLI entry point для RAG системы.

    Args:
        argv: список аргументов командной строки (без имени программы);
              None — взять из sys.argv (режим console script `rag-server`)

    Returns:
        int: exit code (0 — успех, 1 — ошибка)
    """
    import logging as _logging
    _log_level = os.environ.get("LOG_LEVEL", "INFO").upper()
    _logging.basicConfig(
        level=getattr(_logging, _log_level, _logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        stream=sys.stderr,
    )
    for _lib in ("httpx", "huggingface_hub", "sentence_transformers", "httpcore"):
        _logging.getLogger(_lib).setLevel(_logging.WARNING)

    if argv is None:
        argv = sys.argv[1:]

    parser = argparse.ArgumentParser(description="RAG System CLI")
    parser.add_argument("--store", type=str, default="./rag_data", help="Path to vector store")
    parser.add_argument("--key", type=str, default=None, help="API key for embedding provider")
    parser.add_argument("--provider", type=str, default=None, choices=["openai-compatible", "anthropic", "ollama"], help="Embedding provider type")
    parser.add_argument("--model", type=str, default=None, help="Embedding model name (e.g. BAAI/bge-m3)")
    parser.add_argument("--ollama-url", type=str, default=None, help="Ollama base URL")
    parser.add_argument("--http", action="store_true", help="Start HTTP REST API + MCP SSE server")
    parser.add_argument("--port", type=int, default=8765, help="HTTP server port (default: 8765)")

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # add-document
    parser_add_doc = subparsers.add_parser("add-document", help="Add a document")
    parser_add_doc.add_argument("--text", type=str, required=True, help="Document text")
    parser_add_doc.add_argument("--meta", type=str, default=None, help="Metadata JSON")

    # add-file
    parser_add_file = subparsers.add_parser("add-file", help="Add a file")
    parser_add_file.add_argument("--path", type=str, required=True, help="File path")
    parser_add_file.add_argument("--meta", type=str, default=None, help="Metadata JSON")

    # search
    parser_search = subparsers.add_parser("search", help="Semantic search")
    parser_search.add_argument("--query", type=str, required=True, help="Search query")
    parser_search.add_argument("--k", type=int, default=5, help="Number of results")
    parser_search.add_argument("--meta-filter", type=str, default=None, help='Metadata filter JSON, e.g. \'{"source":"spec"}\'')
    parser_search.add_argument("--relations-load-depth", type=int, default=1, help="BFS depth for graph relations (0=off, 1=direct neighbours)")
    parser_search.add_argument("--relations-load-type-filter", type=str, default=None, help='Comma-separated relation types, e.g. "related_to,similar_to"')
    parser_search.add_argument("--relations-load-meta-filter", type=str, default=None, help='Neighbour metadata filter JSON')

    # bm25-search
    parser_bm25 = subparsers.add_parser("bm25-search", help="BM25 keyword search")
    parser_bm25.add_argument("--query", type=str, required=True, help="Search query")
    parser_bm25.add_argument("--k", type=int, default=5, help="Number of results")
    parser_bm25.add_argument("--meta-filter", type=str, default=None, help='Metadata filter JSON, e.g. \'{"source":"spec"}\'')
    parser_bm25.add_argument("--relations-load-depth", type=int, default=1, help="BFS depth for graph relations (0=off, 1=direct neighbours)")
    parser_bm25.add_argument("--relations-load-type-filter", type=str, default=None, help='Comma-separated relation types')
    parser_bm25.add_argument("--relations-load-meta-filter", type=str, default=None, help='Neighbour metadata filter JSON')

    # hybrid-search
    parser_hybrid = subparsers.add_parser("hybrid-search", help="Hybrid search")
    parser_hybrid.add_argument("--query", type=str, required=True, help="Search query")
    parser_hybrid.add_argument("--k", type=int, default=5, help="Number of results")
    parser_hybrid.add_argument("--alpha", type=float, default=None, help="Hybrid alpha (0=BM25, 1=semantic; default=language-aware)")
    parser_hybrid.add_argument("--rerank", action="store_true", default=None, help="Enable CrossEncoder reranker")
    parser_hybrid.add_argument("--query-expansion", action="store_true", default=None, help="Enable query expansion via LLM")
    parser_hybrid.add_argument("--meta-filter", type=str, default=None, help='Metadata filter JSON, e.g. \'{"source":"spec"}\'')
    parser_hybrid.add_argument("--relations-load-depth", type=int, default=1, help="BFS depth for graph relations (0=off, 1=direct neighbours)")
    parser_hybrid.add_argument("--relations-load-type-filter", type=str, default=None, help='Comma-separated relation types')
    parser_hybrid.add_argument("--relations-load-meta-filter", type=str, default=None, help='Neighbour metadata filter JSON')

    # add-structured
    parser_add_structured = subparsers.add_parser("add-structured", help="Index structured code (repomix JSON)")
    parser_add_structured.add_argument("--content", type=str, required=True, help="JSON string in repomix format")
    parser_add_structured.add_argument("--extract-graph", action="store_true", help="Extract entity-relation graph from files")

    # list-documents
    parser_list = subparsers.add_parser("list-documents", help="List documents with pagination")
    parser_list.add_argument("--limit", type=int, default=20, help="Max documents to return (default: 20)")
    parser_list.add_argument("--offset", type=int, default=0, help="Start offset (default: 0)")
    parser_list.add_argument("--max-chars", type=int, default=None, help="Truncate text to N chars")
    parser_list.add_argument("--meta-filter", type=str, default=None, help='Metadata filter JSON, e.g. \'{"source":"spec"}\'')
    parser_list.add_argument("--relations-load-depth", type=int, default=1, help="BFS depth for graph relations")

    # get-document
    parser_get = subparsers.add_parser("get-document", help="Get document by ID")
    parser_get.add_argument("--doc-id", type=str, required=True, help="Document UUID")
    parser_get.add_argument("--offset", type=int, default=0, help="Character offset (default: 0)")
    parser_get.add_argument("--limit", type=int, default=None, help="Max characters to return")
    parser_get.add_argument("--relations-load-depth", type=int, default=1, help="BFS depth for graph relations")

    # stats
    subparsers.add_parser("stats", help="Show statistics")

    # clear
    subparsers.add_parser("clear", help="Clear all data")

    # add-relation
    parser_add_relation = subparsers.add_parser("add-relation", help="Add a relation between documents")
    parser_add_relation.add_argument("--source", type=str, required=True, help="Source document UUID")
    parser_add_relation.add_argument("--target", type=str, required=True, help="Target document UUID")
    parser_add_relation.add_argument("--relation", type=str, required=True, help="Relation type")
    parser_add_relation.add_argument("--weight", type=float, default=1.0, help="Relation weight")

    # get-related
    parser_get_related = subparsers.add_parser("get-related", help="Get related documents")
    parser_get_related.add_argument("--node", type=str, required=True, help="Node UUID")
    parser_get_related.add_argument("--max-depth", type=int, default=1, help="Max depth for traversal")
    parser_get_related.add_argument("--meta-filter", type=str, default=None, help='Metadata filter JSON, e.g. \'{"source":"spec"}\'')

    # graph-stats
    subparsers.add_parser("graph-stats", help="Show graph statistics")

    # reindex
    subparsers.add_parser("reindex", help="Re-generate embeddings for all documents (when switching embedding model)")

    # migrate (старый ChromaDB + graph_index.json → Qdrant + SQLite)
    parser_migrate = subparsers.add_parser("migrate", help="Migrate old ChromaDB/BM25/JSON-graph data to Qdrant + SQLite")
    parser_migrate.add_argument("--dry-run", action="store_true", help="Show what would be migrated without writing")
    parser_migrate.add_argument("--force", action="store_true", help="Do not ask for confirmation")

    # graph-viz
    parser_graph_viz = subparsers.add_parser("graph-viz", help="Visualize knowledge graph")
    parser_graph_viz.add_argument("--output", "-o", type=str, default="graph_viz.html", help="Output file path")
    parser_graph_viz.add_argument("--format", "-f", type=str, default="html",
                                  choices=["html", "dot", "json", "ascii"],
                                  help="Output format (default: html)")
    parser_graph_viz.add_argument("--max-nodes", "-n", type=int, default=None,
                                  help="Limit to top-N nodes by degree")
    parser_graph_viz.add_argument("--relation-type", "-r", type=str, action="append", default=None,
                                  help="Filter by relation type (can be repeated)")
    parser_graph_viz.add_argument("--focus", type=str, default=None,
                                  help="Show subgraph around this doc_id")
    parser_graph_viz.add_argument("--max-depth", type=int, default=2,
                                  help="BFS depth when using --focus (default: 2)")
    parser_graph_viz.add_argument("--layout", type=str, default="kamada_kawai",
                                  choices=["kamada_kawai", "spring", "circular", "hierarchical"],
                                  help="Layout algorithm (html/dot only)")

    # serve-graph
    parser_serve = subparsers.add_parser("serve-graph", help="Serve graph with live RAG API")
    parser_serve.add_argument("--output", "-o", type=str, default="graph_viz.html", help="Output file path")
    parser_serve.add_argument("--port", "-p", type=int, default=8090, help="HTTP port")
    parser_serve.add_argument("--max-nodes", "-n", type=int, default=None,
                              help="Limit to top-N nodes by degree")
    parser_serve.add_argument("--relation-type", "-r", type=str, action="append", default=None,
                              help="Filter by relation type (can be repeated)")
    parser_serve.add_argument("--focus", type=str, default=None,
                              help="Show subgraph around this doc_id")
    parser_serve.add_argument("--max-depth", type=int, default=2,
                              help="BFS depth when using --focus (default: 2)")
    parser_serve.add_argument("--layout", type=str, default="kamada_kawai",
                              choices=["kamada_kawai", "spring", "circular", "hierarchical"],
                              help="Layout algorithm")
    parser_serve.add_argument("--no-browser", action="store_true", help="Don't open browser")

    try:
        args = parser.parse_args(argv)
    except SystemExit:
        return 1

    if args.http:
        return _start_http(args)

    if not args.command:
        parser.print_help()
        return 1

    try:
        from src.config import RAGConfig
        cfg = RAGConfig.from_env()
        if args.store:
            cfg.store_path = args.store
        if args.provider:
            cfg.embedding_provider = args.provider
        if args.model:
            cfg.embedding_model_name = args.model
        if args.ollama_url:
            cfg.embedding_base_url = args.ollama_url
        if args.key:
            cfg.embedding_api_key = args.key

        rag = RAGSystem(store_path=args.store, config=cfg)

        if args.command == "add-document":
            meta = json.loads(args.meta) if args.meta else None
            doc_id = rag.add_document(args.text, meta)
            print(f"✅ Добавлен документ: {doc_id}")
            return 0

        elif args.command == "add-file":
            meta = json.loads(args.meta) if args.meta else None
            doc_id = rag.add_file(args.path, meta)
            print(f"✅ Добавлен файл: {args.path} -> {doc_id}")
            return 0

        elif args.command == "search":
            meta_filter = json.loads(args.meta_filter) if args.meta_filter else None
            rtype = args.relations_load_type_filter.split(",") if args.relations_load_type_filter else None
            rmeta = json.loads(args.relations_load_meta_filter) if args.relations_load_meta_filter else None
            results = rag.search(args.query, args.k, metadata_filter=meta_filter)
            docs = [{"doc_id": r[0], "text": r[1], "score": r[2], "metadata": r[3]} for r in results]
            rag._enrich_with_links(docs, relations_load_depth=args.relations_load_depth, relations_load_type_filter=rtype, relations_load_meta_filter=rmeta)
            _print_dict_results("Семантический поиск", docs)
            return 0

        elif args.command == "bm25-search":
            meta_filter = json.loads(args.meta_filter) if args.meta_filter else None
            rtype = args.relations_load_type_filter.split(",") if args.relations_load_type_filter else None
            rmeta = json.loads(args.relations_load_meta_filter) if args.relations_load_meta_filter else None
            results = rag.bm25_search(args.query, args.k, metadata_filter=meta_filter)
            docs = [{"doc_id": r[0], "text": r[1], "score": r[2], "metadata": r[3]} for r in results]
            rag._enrich_with_links(docs, relations_load_depth=args.relations_load_depth, relations_load_type_filter=rtype, relations_load_meta_filter=rmeta)
            _print_dict_results("BM25 поиск", docs)
            return 0

        elif args.command == "hybrid-search":
            meta_filter = json.loads(args.meta_filter) if args.meta_filter else None
            rtype = args.relations_load_type_filter.split(",") if args.relations_load_type_filter else None
            rmeta = json.loads(args.relations_load_meta_filter) if args.relations_load_meta_filter else None
            results = rag.search_hybrid(args.query, args.k, args.alpha, metadata_filter=meta_filter, rerank=args.rerank, query_expansion=args.query_expansion)
            docs = [{"doc_id": r[0], "text": r[1], "score": r[2], "metadata": r[3]} for r in results]
            rag._enrich_with_links(docs, relations_load_depth=args.relations_load_depth, relations_load_type_filter=rtype, relations_load_meta_filter=rmeta)
            _print_dict_results("Гибридный поиск", docs)
            return 0

        elif args.command == "add-structured":
            result = rag.index_structured(args.content, extract_graph=args.extract_graph)
            print(json.dumps(result, ensure_ascii=False, indent=2))
            return 0

        elif args.command == "list-documents":
            meta_filter = json.loads(args.meta_filter) if args.meta_filter else None
            result = rag.list_documents(
                limit=args.limit, offset=args.offset, max_chars=args.max_chars,
                metadata_filter=meta_filter, relations_load_depth=args.relations_load_depth,
            )
            print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
            return 0

        elif args.command == "get-document":
            doc = rag.get_document(args.doc_id, offset=args.offset, limit=args.limit, relations_load_depth=args.relations_load_depth)
            if doc is None:
                print(f"❌ Document not found: {args.doc_id}")
                return 1
            print(json.dumps(doc, ensure_ascii=False, indent=2, default=str))
            return 0

        elif args.command == "stats":
            stats = rag.stats()
            print("📊 Статистика:")
            print(f"   Всего документов: {stats.get('total_documents', 'N/A')}")
            print(f"   Путь хранилища: {stats.get('store_path', 'N/A')}")
            print(f"   Размерность: {stats.get('dimension', 'N/A')}")
            return 0

        elif args.command == "clear":
            rag.clear()
            print("✅ Все данные очищены")
            return 0

        elif args.command == "add-relation":
            rag.add_relation(args.source, args.target, args.relation, args.weight)
            print(f"✅ Отношение добавлено: {args.source} --[{args.relation}]--> {args.target}")
            return 0

        elif args.command == "get-related":
            meta_filter = json.loads(args.meta_filter) if args.meta_filter else None
            relations = rag.get_related(args.node, args.max_depth, **({"metadata_filter": meta_filter} if meta_filter else {}))
            if not relations:
                print(f"📭 Нет связанных узлов для {args.node}")
            else:
                print(f"🔗 Связанные узлы для {args.node[:8]}... (depth={args.max_depth}):")
                for source, target, relation, weight, _direction in relations:
                    print(f"   {source[:8]}... --[{relation}] (w={weight:.2f})--> {target[:8]}...")
            return 0

        elif args.command == "graph-stats":
            stats = rag.stats()
            print("📊 Статистика графа:")
            print(f"   Всего узлов: {stats.get('total_nodes', 'N/A')}")
            print(f"   Всего рёбер: {stats.get('total_edges', 'N/A')}")
            print(f"   Типы отношений: {stats.get('relation_types', 'N/A')}")
            return 0

        elif args.command == "reindex":
            count = rag.reindex()
            print(f"♻️ Переиндексировано документов: {count}")
            return 0

        elif args.command == "migrate":
            from src.migrate import run_migration
            stats = run_migration(rag, dry_run=args.dry_run, force=args.force)
            if stats is None:
                return 1
            print(
                f"✅ Migrated {stats['documents']} documents, {stats['edges']} edges "
                f"(skipped: {stats['skipped_documents']} docs, {stats['skipped_edges']} edges)"
            )
            return 0

        elif args.command in ("graph-viz", "serve-graph"):
            from src.graph_viz import render_graph_viz, serve_graph
            if args.command == "serve-graph":
                serve_graph(
                    rag.graph_kb,
                    rag=rag,
                    output_path=args.output,
                    port=args.port,
                    max_nodes=args.max_nodes,
                    relation_type=args.relation_type,
                    focus_node=args.focus,
                    max_depth=args.max_depth,
                    layout=args.layout,
                    open_browser=not getattr(args, 'no_browser', False),
                )
            else:
                render_graph_viz(
                    rag.graph_kb,
                    output_path=args.output,
                    output_format=args.format,
                    max_nodes=args.max_nodes,
                    relation_type=args.relation_type,
                    focus_node=args.focus,
                    max_depth=args.max_depth,
                    layout=args.layout,
                )
            return 0

    except Exception as e:
        print(f"❌ Ошибка: {e}", file=sys.stderr)
        return 1

    return 0


def _start_http(args) -> int:
    """Запустить HTTP REST API + MCP SSE сервер."""
    import uvicorn
    os.environ.setdefault("STORE_PATH", args.store or "./rag_data")
    if args.provider:
        os.environ["EMBEDDING_PROVIDER"] = args.provider
    if args.model:
        os.environ["EMBEDDING_MODEL"] = args.model
    if args.ollama_url:
        os.environ["EMBEDDING_BASE_URL"] = args.ollama_url
    if args.key:
        os.environ["EMBEDDING_API_KEY"] = args.key
    port = args.port or 8765
    print(f"🌐 Starting HTTP server on http://localhost:{port}", file=sys.stderr)
    print(f"   REST API: http://localhost:{port}/docs", file=sys.stderr)
    print(f"   MCP SSE:  http://localhost:{port}/mcp", file=sys.stderr)
    uvicorn.run("src.http_api:app", host="0.0.0.0", port=port, log_level="info")
    return 0


def _print_results(title: str, results: list) -> None:
    """Вывести результаты поиска в читаемом виде (формат кортежей)."""
    if not results:
        print(f"📭 {title}: результатов нет")
        return

    print(f"🔍 {title} (k={len(results)}):")
    print("─" * 60)
    for i, (doc_id, text, score, metadata) in enumerate(results, 1):
        text_preview = text[:80] + "..." if len(text) > 80 else text
        print(f"  {i}. [{doc_id[:8]}...] {text_preview}")
        print(f"     Score: {score:.4f}")
        if metadata:
            print(f"     Meta: {metadata}")
        print()


def _print_dict_results(title: str, docs: list[dict]) -> None:
    """Вывести результаты поиска в читаемом виде (формат dict с links)."""
    if not docs:
        print(f"📭 {title}: результатов нет")
        return

    print(f"🔍 {title} (k={len(docs)}):")
    print("─" * 60)
    for i, d in enumerate(docs, 1):
        doc_id = d.get("doc_id", "?")
        text = d.get("text", "")
        score = d.get("score", 0.0)
        metadata = d.get("metadata", {})
        links = d.get("links", {})
        text_preview = text[:80] + "..." if len(text) > 80 else text
        print(f"  {i}. [{doc_id[:8]}...] {text_preview}")
        print(f"     Score: {score:.4f}")
        if metadata:
            print(f"     Meta: {metadata}")
        if links:
            linked_ids = list(links.keys())
            print(f"     Links: {', '.join(did[:8] + '...' for did in linked_ids[:3])}{'...' if len(linked_ids) > 3 else ''}")
        print()


if __name__ == "__main__":
    sys.exit(main())