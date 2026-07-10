#!/usr/bin/env python3
"""Бенчмарк выбора оптимального alpha для гибридного поиска.

Запуск:
    cd ~/Work/graphrag
    python3 -m scripts.benchmark_alpha

Методология:
  1. Строится детерминированный тестовый корпус (tests.semantic_mock.make_test_corpus)
     с настоящей семантической структурой через SemanticMockEmbeddingGenerator.
  2. Для каждого запроса из QUERY_SET вычисляются ground-truth суждения
     релевантности (бинарные) через overlap понятий запроса и документа.
  3. Для каждого alpha в сетке [0.0, 0.1, ..., 1.0] выполняется search_hybrid,
     считается NDCG@k (k=5) и precision@k по всем запросам.
  4. Выбирается alpha с максимальным средним NDCG@k; результат печатается в виде
     таблицы. Значение-победитель должно совпадать с RAGConfig.default_alpha
     (этот скрипт используется для его калибровки).
"""

from __future__ import annotations

import math
import os
import shutil
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.rag import RAGSystem
from src.config import RAGConfig
from tests.semantic_mock import (
    SemanticMockEmbeddingGenerator,
    make_test_corpus,
    relevance_for_query,
)


# Запросы: смесь keyword-specific (точные идентификаторы), концептуальных
# (cross-wording) и cross-lingual. Идентификаторные запросы (pytest, jwt, bm25,
# chromadb) семантический канал не видит — их находит только BM25. Концептуальные
# запросы с синонимами/cross-lingual BM25 обрабатывает хуже из-за отсутствия
# точных совпадений. Это создаёт настоящую комплементарность каналов.
QUERY_SET: list[str] = [
    # Концептуальные (семантика сильнее)
    "python programming",
    "machine learning neural networks",
    "authentication and authorization",
    "orchestrator dispatch events",
    "architecture guardian rules",
    "documentation freshness",
    "decisions and findings storage",
    "meeting transcripts processing",
    "vector retrieval embeddings",
    "knowledge graph relations",
    "italian food recipe",
    "java virtual machine",
    "code requirements checking",
    "deep learning computer vision",
    # Cross-lingual (семантика сильнее: BM25 не matching русские vs английские)
    "аутентификация и авторизация",
    "оркестратор событий агент",
    "машинное обучение нейронная сеть",
    "генерация тестов",
    "граф знаний рёбра",
    "готовка рецепт паста",
    # Идентификаторные (BM25 сильнее: семантика не видит имена)
    "pytest",
    "jwt",
    "bm25",
    "chromadb",
    # Смешанные (нужен blend)
    "pytest tests agent",
    "jwt authentication",
    "bm25 keyword search",
    "chromadb graph storage",
]

K = 5


def dcg(rels: list[int]) -> float:
    return sum(rel / math.log2(i + 2) for i, rel in enumerate(rels))


def ndcg_at_k(ranked_doc_ids: list[str], judgments: dict[str, int], k: int) -> float:
    rels = [judgments.get(did, 0) for did in ranked_doc_ids[:k]]
    ideal = sorted(judgments.values(), reverse=True)[:k]
    idcg = dcg(ideal)
    if idcg == 0:
        return 0.0
    return dcg(rels) / idcg


def precision_at_k(ranked_doc_ids: list[str], judgments: dict[str, int], k: int) -> float:
    top = ranked_doc_ids[:k]
    if not top:
        return 0.0
    hits = sum(1 for did in top if judgments.get(did, 0) > 0)
    return hits / len(top)


def build_rag(store_path: str) -> RAGSystem:
    """Собрать RAGSystem (Qdrant embedded + SQLite) на семантическом mock-генераторе.

    Генератор передаётся в конструктор явно: дефолтный провайдер (BGE-M3)
    тянет реальную модель, а мок должен быть подключён до создания коллекции
    Qdrant (размерность dense-векторов фиксируется при инициализации).
    """
    return RAGSystem(
        config=RAGConfig(store_path=store_path),
        embedding_generator=SemanticMockEmbeddingGenerator(),
    )


def index_corpus(rag: RAGSystem) -> dict[str, str]:
    """Индексировать корпус; вернуть mapping doc_id_hint → реальный UUID."""
    corpus = make_test_corpus()
    hint_to_uuid: dict[str, str] = {}
    for hint, text, _concepts in corpus:
        uuid = rag.add_document(text, metadata={"hint": hint})
        hint_to_uuid[hint] = uuid
    return hint_to_uuid


def evaluate_alpha(rag: RAGSystem, hint_to_uuid: dict[str, str], alpha: float) -> tuple[float, float]:
    """Вернуть (avg_ndcg, avg_precision) по всем запросам для данного alpha."""
    corpus = make_test_corpus()

    total_ndcg = 0.0
    total_prec = 0.0
    for query in QUERY_SET:
        judgments_hint = relevance_for_query(query, corpus)
        judgments_uuid = {hint_to_uuid[h]: v for h, v in judgments_hint.items()}
        results = rag.search_hybrid(query, k=K, alpha=alpha)
        ranked = [r[0] for r in results]
        total_ndcg += ndcg_at_k(ranked, judgments_uuid, K)
        total_prec += precision_at_k(ranked, judgments_uuid, K)
    n = len(QUERY_SET)
    return total_ndcg / n, total_prec / n


def main() -> None:
    temp_dir = tempfile.mkdtemp(prefix="rag_bench_")
    store_path = os.path.join(temp_dir, "rag_data")
    rag = None
    try:
        rag = build_rag(store_path)
        hint_to_uuid = index_corpus(rag)

        print(f"Corpus: {len(hint_to_uuid)} docs | Queries: {len(QUERY_SET)} | k={K}")
        print("-" * 60)
        print(f"{'alpha':>6} | {'NDCG@5':>8} | {'P@5':>6} | {'P@1':>6}")
        print("-" * 60)

        rows: list[tuple[float, float, float, float]] = []
        step = 0.05
        alpha = 0.0
        while alpha <= 1.0 + 1e-9:
            ndcg, prec = evaluate_alpha(rag, hint_to_uuid, round(alpha, 4))
            # Дополнительная чувствительная метрика: precision@1 — попадает ли
            # самый релевантный документ на первое место.
            p1_total = 0.0
            corpus = make_test_corpus()
            for query in QUERY_SET:
                judgments_hint = relevance_for_query(query, corpus)
                judgments_uuid = {hint_to_uuid[h]: v for h, v in judgments_hint.items()}
                results = rag.search_hybrid(query, k=K, alpha=round(alpha, 4))
                ranked = [r[0] for r in results]
                p1_total += precision_at_k(ranked, judgments_uuid, 1)
            p1 = p1_total / len(QUERY_SET)
            rows.append((round(alpha, 4), ndcg, prec, p1))
            print(f"{alpha:6.2f} | {ndcg:8.4f} | {prec:6.4f} | {p1:6.4f}")
            alpha += step

        print("-" * 60)

        best_ndcg = max(r[1] for r in rows)
        # Робастный выбор: среди alpha в пределах 1% от лучшего NDCG берём то,
        # что ближе всего к 0.5 — естественной точке баланса каналов. Это
        # детерминированно (в отличие от медианы хорошей области, чья ширина
        # колеблется из-за tie-breaking в векторном сторе) и философски оправдано:
        # баланс по умолчанию, если только данные не говорят обратного.
        threshold = best_ndcg * 0.99
        good_alphas = [r[0] for r in rows if r[1] >= threshold]
        good_alphas.sort()
        best_alpha = min(good_alphas, key=lambda a: abs(a - 0.5))
        best_row = next(r for r in rows if r[0] == best_alpha)

        print(f"Best NDCG@5 = {best_ndcg:.4f}")
        # ASCII-only вывод: Windows-консоль (cp1252) не кодирует символ "∈".
        print(f"Good region (within 1% of best): alpha in [{good_alphas[0]:.2f}, {good_alphas[-1]:.2f}]")
        print(f"Selected default alpha = {best_alpha:.2f}  "
              f"(closest-to-0.5 in good region; NDCG={best_row[1]:.4f}, "
              f"P@5={best_row[2]:.4f}, P@1={best_row[3]:.4f})")
        print(f"Current RAGConfig.default_alpha = {RAGConfig().default_alpha}")
        if abs(best_alpha - RAGConfig().default_alpha) < 1e-6:
            print("OK: config default matches benchmark winner.")
        else:
            print("MISMATCH: update RAGConfig.default_alpha to match.")
    finally:
        # Windows: Qdrant embedded и SQLite держат файловые локи —
        # без close() rmtree не сможет удалить временную директорию.
        if rag is not None:
            rag.close()
        shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
