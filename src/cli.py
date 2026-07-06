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

    # bm25-search
    parser_bm25 = subparsers.add_parser("bm25-search", help="BM25 keyword search")
    parser_bm25.add_argument("--query", type=str, required=True, help="Search query")
    parser_bm25.add_argument("--k", type=int, default=5, help="Number of results")
    parser_bm25.add_argument("--meta-filter", type=str, default=None, help='Metadata filter JSON, e.g. \'{"source":"spec"}\'')

    # hybrid-search
    parser_hybrid = subparsers.add_parser("hybrid-search", help="Hybrid search")
    parser_hybrid.add_argument("--query", type=str, required=True, help="Search query")
    parser_hybrid.add_argument("--k", type=int, default=5, help="Number of results")
    parser_hybrid.add_argument("--alpha", type=float, default=0.5, help="Hybrid alpha (0=BM25, 1=semantic)")
    parser_hybrid.add_argument("--meta-filter", type=str, default=None, help='Metadata filter JSON, e.g. \'{"source":"spec"}\'')

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
            results = rag.search(args.query, args.k, **({"metadata_filter": meta_filter} if meta_filter else {}))
            _print_results("Семантический поиск", results)
            return 0

        elif args.command == "bm25-search":
            meta_filter = json.loads(args.meta_filter) if args.meta_filter else None
            results = rag.bm25_search(args.query, args.k, **({"metadata_filter": meta_filter} if meta_filter else {}))
            _print_results("BM25 поиск", results)
            return 0

        elif args.command == "hybrid-search":
            meta_filter = json.loads(args.meta_filter) if args.meta_filter else None
            results = rag.search_hybrid(args.query, args.k, args.alpha, **({"metadata_filter": meta_filter} if meta_filter else {}))
            _print_results("Гибридный поиск", results)
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

    except Exception as e:
        print(f"❌ Ошибка: {e}", file=sys.stderr)
        return 1

    return 0


def _print_results(title: str, results: list) -> None:
    """Вывести результаты поиска в читаемом виде."""
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


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))