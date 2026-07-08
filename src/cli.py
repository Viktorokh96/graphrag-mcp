"""CLI интерфейс для RAG системы."""

import argparse
import json
import sys
from src.rag import RAGSystem


def main(argv: list[str]) -> int:
    """
    CLI entry point для RAG системы.

    Args:
        argv: список аргументов командной строки (без имени программы)

    Returns:
        int: exit code (0 — успех, 1 — ошибка)
    """

    parser = argparse.ArgumentParser(description="RAG System CLI")
    parser.add_argument("--store", type=str, default="./rag_data", help="Path to vector store")
    parser.add_argument("--key", type=str, default=None, help="OpenRouter API key")
    parser.add_argument("--provider", type=str, default=None, choices=["ollama", "openrouter"], help="Embedding provider")
    parser.add_argument("--ollama-url", type=str, default=None, help="Ollama base URL")
    parser.add_argument("--ollama-model", type=str, default=None, help="Ollama model name")

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
    parser_hybrid.add_argument("--alpha", type=float, default=0.5, help="Hybrid alpha (0=BM25, 1=semantic)")
    parser_hybrid.add_argument("--meta-filter", type=str, default=None, help='Metadata filter JSON, e.g. \'{"source":"spec"}\'')
    parser_hybrid.add_argument("--relations-load-depth", type=int, default=1, help="BFS depth for graph relations (0=off, 1=direct neighbours)")
    parser_hybrid.add_argument("--relations-load-type-filter", type=str, default=None, help='Comma-separated relation types')
    parser_hybrid.add_argument("--relations-load-meta-filter", type=str, default=None, help='Neighbour metadata filter JSON')

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
    subparsers.add_parser("reindex", help="Re-generate embeddings for all documents (when switching embedding provider)")

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
        if args.ollama_url:
            cfg.ollama_base_url = args.ollama_url
        if args.ollama_model:
            cfg.ollama_model = args.ollama_model

        rag = RAGSystem(store_path=args.store, api_key=args.key, config=cfg)

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
            results = rag.search_hybrid(args.query, args.k, args.alpha, metadata_filter=meta_filter)
            docs = [{"doc_id": r[0], "text": r[1], "score": r[2], "metadata": r[3]} for r in results]
            rag._enrich_with_links(docs, relations_load_depth=args.relations_load_depth, relations_load_type_filter=rtype, relations_load_meta_filter=rmeta)
            _print_dict_results("Гибридный поиск", docs)
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

        elif args.command == "graph-viz":
            from src.graph_viz import render_graph_viz
            from src.graph_store import GraphKnowledgeBase
            g = GraphKnowledgeBase(store_path=args.store)
            render_graph_viz(
                g,
                output_path=args.output,
                output_format=args.format,
                max_nodes=args.max_nodes,
                relation_type=args.relation_type,
                focus_node=args.focus,
                max_depth=args.max_depth,
                layout=args.layout,
            )
            return 0

        elif args.command == "serve-graph":
            from src.graph_viz import serve_graph
            from src.graph_store import GraphKnowledgeBase
            g = GraphKnowledgeBase(store_path=args.store)
            serve_graph(
                g,
                rag=rag,
                output_path=args.output,
                port=args.port,
                max_nodes=args.max_nodes,
                relation_type=args.relation_type,
                focus_node=args.focus,
                max_depth=args.max_depth,
                layout=args.layout,
                open_browser=not args.no_browser,
            )
            return 0

    except Exception as e:
        print(f"❌ Ошибка: {e}", file=sys.stderr)
        return 1

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
    sys.exit(main(sys.argv[1:]))