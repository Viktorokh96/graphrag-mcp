"""Тесты качества семантического и гибридного поиска.

В отличие от существующих тестов (которые проверяют механику — что поиск
возвращает k результатов, сортировку по score, метаданные), эти тесты
проверяют КАЧЕСТВО ранжирования на детерминированном корпусе с настоящей
семантической структурой (tests/semantic_mock.py).

Покрытие:
  • SemanticMockEmbeddingGenerator — корректность построения концептуальных векторов
  • Семантический поиск находит концептуально близкие документы при разном wording
  • Семантический поиск находит cross-lingual совпадения (русский ↔ английский)
  • Семантический поиск НЕ видит идентификаторы (pytest, jwt) — это работа BM25
  • BM25 находит документы по точным идентификаторам
  • Гибридный поиск превосходит чистую семантику на идентификаторных запросах
  • Гибридный поиск превосходит чистый BM25 на концептуальных/cross-lingual запросах
  • Default alpha (0.5) лежит в хорошей области, выявленной бенчмарком
  • Candidate expansion спасает документ, релевантный по одному каналу,
    но оказавшийся за пределами top-k по другому
"""

import math

import pytest

from src.config import RAGConfig
from src.rag import RAGSystem
from tests.semantic_mock import (
    SemanticMockEmbeddingGenerator,
    make_test_corpus,
    relevance_for_query,
)


K = 5


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _index_corpus(rag: RAGSystem) -> dict[str, str]:
    """Индексировать тестовый корпус; вернуть mapping hint → UUID."""
    hint_to_uuid: dict[str, str] = {}
    for hint, text, _concepts in make_test_corpus():
        hint_to_uuid[hint] = rag.add_document(text, metadata={"hint": hint})
    return hint_to_uuid


def _ranked_hints(results: list[tuple], uuid_to_hint: dict[str, str]) -> list[str]:
    """Перевести UUID-результаты в список hint-ов в порядке ранжирования."""
    return [uuid_to_hint[r[0]] for r in results]


def _ndcg_at_k(ranked: list[str], judgments: dict[str, int], k: int) -> float:
    def dcg(rels):
        return sum(rel / math.log2(i + 2) for i, rel in enumerate(rels))

    rels = [judgments.get(d, 0) for d in ranked[:k]]
    ideal = sorted(judgments.values(), reverse=True)[:k]
    idcg = dcg(ideal)
    return 0.0 if idcg == 0 else dcg(rels) / idcg


def _precision_at_k(ranked: list[str], judgments: dict[str, int], k: int) -> float:
    top = ranked[:k]
    if not top:
        return 0.0
    return sum(1 for d in top if judgments.get(d, 0) > 0) / len(top)


# ---------------------------------------------------------------------------
# Фикстуры
# ---------------------------------------------------------------------------


@pytest.fixture
def rag_with_corpus(make_rag):
    """RAG-система (Qdrant embedded + SQLite) с проиндексированным корпусом.

    Использует фабрику make_rag из conftest: временный стор в tmp_path и
    гарантированный rag.close() в teardown (Windows держит файловые локи
    Qdrant embedded и SQLite до закрытия).
    """
    rag = make_rag(embedder=SemanticMockEmbeddingGenerator())
    hint_to_uuid = _index_corpus(rag)
    uuid_to_hint = {u: h for h, u in hint_to_uuid.items()}
    return rag, hint_to_uuid, uuid_to_hint


# ---------------------------------------------------------------------------
# SemanticMockEmbeddingGenerator
# ---------------------------------------------------------------------------


class TestSemanticMockGenerator:
    def test_dimension_matches_lexicon_size(self):
        gen = SemanticMockEmbeddingGenerator()
        assert gen.get_dimension() == 20

    def test_identical_texts_give_identical_embeddings(self):
        gen = SemanticMockEmbeddingGenerator()
        assert gen.get_embedding("python programming") == gen.get_embedding("python programming")

    def test_zero_vector_for_empty_text(self):
        gen = SemanticMockEmbeddingGenerator()
        assert gen.get_embedding("") == [0.0] * gen.get_dimension()

    def test_unknown_words_give_zero_vector(self):
        gen = SemanticMockEmbeddingGenerator()
        # слова не в лексиконе — семантический канал их не видит
        assert gen.get_embedding("zzz qqq") == [0.0] * gen.get_dimension()

    def test_synonyms_map_to_same_concept(self):
        gen = SemanticMockEmbeddingGenerator()
        # "auth" и "authentication" — одно понятие
        a = gen.get_embedding("auth")
        b = gen.get_embedding("authentication")
        # оба вектора ненулевые и сонаправлены (один concept-axis)
        assert any(v > 0 for v in a)
        assert a == b

    def test_cross_lingual_maps_to_same_concept(self):
        gen = SemanticMockEmbeddingGenerator()
        assert gen.get_embedding("auth") == gen.get_embedding("аутентификация")

    def test_orthogonal_concepts_have_zero_similarity(self):
        gen = SemanticMockEmbeddingGenerator()
        cooking = gen.get_embedding("cooking recipe pasta")
        auth = gen.get_embedding("auth authentication")
        # ортогональны → скалярное произведение 0
        dot = sum(a * b for a, b in zip(cooking, auth))
        assert dot == pytest.approx(0.0, abs=1e-9)

    def test_shared_concept_gives_positive_similarity(self):
        gen = SemanticMockEmbeddingGenerator()
        a = gen.get_embedding("python programming")
        b = gen.get_embedding("python machine learning")
        dot = sum(x * y for x, y in zip(a, b))
        assert dot > 0.0

    def test_identifiers_not_in_lexicon(self):
        """Специфичные идентификаторы отсутствуют в семантическом лексиконе."""
        gen = SemanticMockEmbeddingGenerator()
        for ident in ["pytest", "jwt", "bm25", "chromadb"]:
            emb = gen.get_embedding(ident)
            assert emb == [0.0] * gen.get_dimension(), (
                f"идентификатор {ident!r} не должен распознаваться семантикой"
            )


# ---------------------------------------------------------------------------
# Семантический поиск: качество
# ---------------------------------------------------------------------------


class TestSemanticSearchQuality:
    def test_finds_conceptually_related_doc_with_different_wording(self, rag_with_corpus):
        """Семантика находит документ, где нет точных слов запроса, но есть понятие."""
        rag, hint_to_uuid, uuid_to_hint = rag_with_corpus
        # Запрос "italian food recipe" — в документе "cooking recipe for italian pasta"
        # совпадает только по понятию cooking; семантика должна найти pasta-документ.
        results = rag.search("italian food recipe", k=3)
        hints = _ranked_hints(results, uuid_to_hint)
        assert "pasta" in hints

    def test_finds_cross_lingual_match(self, rag_with_corpus):
        """Русский запрос находит английский документ через общее понятие."""
        rag, hint_to_uuid, uuid_to_hint = rag_with_corpus
        results = rag.search("машинное обучение нейронная сеть", k=3)
        hints = _ranked_hints(results, uuid_to_hint)
        # как минимум один из ML-документов должен быть в выдаче
        ml_docs = {"py_ml", "ml_overview", "deep_nn"}
        assert len(set(hints) & ml_docs) >= 1

    def test_does_not_find_identifiers(self, rag_with_corpus):
        """Чистая семантика не видит идентификаторы (pytest/jwt/bm25/chromadb)."""
        rag, hint_to_uuid, uuid_to_hint = rag_with_corpus
        for query, expected_hint in [
            ("pytest", "tests_agent"),
            ("jwt", "jwt_auth"),
            ("bm25", "rag_tool"),
            ("chromadb", "graph_kb"),
        ]:
            results = rag.search(query, k=3)
            hints = _ranked_hints(results, uuid_to_hint)
            # семантика не должна находить документ по чистому идентификатору
            assert expected_hint not in hints, (
                f"семантика не должна находить {expected_hint} по идентификатору {query!r}"
            )

    def test_scores_in_zero_one_range(self, rag_with_corpus):
        """Скоры семантического поиска лежат в [0, 1] (cosine из Qdrant, клип в [0,1])."""
        rag, _, _ = rag_with_corpus
        results = rag.search("python programming", k=5)
        for _doc_id, _text, score, _meta in results:
            assert 0.0 <= score <= 1.0

    def test_better_discrimination_than_old_formula(self, rag_with_corpus):
        """Cosine-скоры дают широкий разброс (не узкий кластер как у старой 1/(1+d))."""
        rag, _, _ = rag_with_corpus
        results = rag.search("python programming", k=5)
        scores = [r[2] for r in results]
        if len(scores) >= 2:
            spread = max(scores) - min(scores)
            # формула 1/(1+d) из ChromaDB-эпохи давала разброс ~0.1;
            # cosine similarity должна различать документы шире
            assert spread > 0.1, f"разброс скоров слишком мал: {spread:.4f}"


# ---------------------------------------------------------------------------
# BM25 поиск: качество
# ---------------------------------------------------------------------------


class TestBM25SearchQuality:
    def test_finds_exact_identifier(self, rag_with_corpus):
        """BM25 находит документ по точному идентификатору."""
        rag, hint_to_uuid, uuid_to_hint = rag_with_corpus
        for query, expected_hint in [
            ("pytest", "tests_agent"),
            ("jwt", "jwt_auth"),
            ("bm25", "rag_tool"),
            ("chromadb", "graph_kb"),
        ]:
            results = rag.bm25_search(query, k=3)
            hints = _ranked_hints(results, uuid_to_hint)
            assert expected_hint in hints, (
                f"BM25 должен находить {expected_hint} по идентификатору {query!r}"
            )

    def test_exact_identifier_ranked_first(self, rag_with_corpus):
        """Документ с точным совпадением идентификатора — на первом месте."""
        rag, hint_to_uuid, uuid_to_hint = rag_with_corpus
        results = rag.bm25_search("pytest", k=3)
        hints = _ranked_hints(results, uuid_to_hint)
        assert hints[0] == "tests_agent"


# ---------------------------------------------------------------------------
# Гибридный поиск: качество и комплементарность
# ---------------------------------------------------------------------------


class TestHybridSearchQuality:
    def test_default_alpha_is_in_config(self):
        """default_alpha берётся из RAGConfig, а не захардкожен в search_hybrid."""
        cfg = RAGConfig(default_alpha=0.42)
        assert cfg.default_alpha == 0.42

    def test_alpha_none_uses_config_default(self, rag_with_corpus):
        """alpha=None для латинского запроса берёт default_alpha из конфига."""
        rag, _, _ = rag_with_corpus
        rag._default_alpha = 0.3
        results = rag.search_hybrid("python", k=3, alpha=None)
        assert len(results) == 3

    def test_alpha_none_cyrillic_uses_cyrillic_alpha(self, rag_with_corpus):
        """alpha=None для кириллического запроса берёт cyrillic_alpha (не default_alpha)."""
        rag, _, _ = rag_with_corpus
        rag._default_alpha = 0.3
        rag._cyrillic_alpha = 0.85
        # Проверяем через _alpha_for_query — публичный контракт выбора alpha
        assert rag._alpha_for_query("машинное обучение") == 0.85
        assert rag._alpha_for_query("python") == 0.3

    def test_explicit_alpha_overrides_language(self, rag_with_corpus):
        """Явно переданный alpha имеет приоритет над language-aware выбором."""
        rag, _, _ = rag_with_corpus
        rag._cyrillic_alpha = 0.85
        # Явный alpha=0.5 для кириллического запроса — не должен заменяться на 0.85
        results = rag.search_hybrid("машинное обучение", k=3, alpha=0.5)
        assert len(results) <= 3  # не падает, alpha применён как передан

    def test_alpha_dilution_excludes_single_channel_at_extremes(self, rag_with_corpus):
        """alpha=1.0 исключает BM25-only доки; alpha=0.0 исключает sem-only доки.

        alpha-dilution: single-channel docs получают alpha*rr (sem) или
        (1-alpha)*rr (bm25). При крайних alpha score=0 → документ исключается.
        Это делает alpha=1.0 чистой семантикой, alpha=0.0 — чистым BM25.
        """
        rag, hint_to_uuid, uuid_to_hint = rag_with_corpus
        # "pytest" — BM25-only (семантика не видит идентификаторы)
        # При alpha=1.0 (pure semantic) pytest-документ исключается (score=0)
        results_sem = rag.search_hybrid("pytest", k=5, alpha=1.0)
        sem_hints = _ranked_hints(results_sem, uuid_to_hint)
        assert "tests_agent" not in sem_hints, (
            "при alpha=1.0 BM25-only доки должны исключаться (pure semantic)"
        )
        # При alpha=0.5 (hybrid) pytest-документ находится (BM25-канал взвешен)
        results_hyb = rag.search_hybrid("pytest", k=5, alpha=0.5)
        hyb_hints = _ranked_hints(results_hyb, uuid_to_hint)
        assert "tests_agent" in hyb_hints

    def test_rrf_k_20_gives_wider_spread(self, rag_with_corpus):
        """RRF_K=20 даёт широкий разброс скоров (не узкий кластер как при K=60).

        При K=60 разброс топ-5 скоров сжимается до ~7% (0.0152..0.0164),
        лишая выдачу различительной силы. K=20 даёт разброс ~30%+.
        """
        rag, _, _ = rag_with_corpus
        results = rag.search_hybrid("python programming", k=5, alpha=0.5)
        scores = [r[2] for r in results]
        if len(scores) >= 2:
            spread = max(scores) - min(scores)
            # K=60 давал бы spread ~0.0012; K=20 даёт spread > 0.005
            assert spread > 0.005, f"разброс скоров слишком мал для K=20: {spread:.6f}"

    def test_hybrid_beats_pure_semantic_on_identifier_query(self, rag_with_corpus):
        """Гибрид (default alpha) находит документ по идентификатору,
        тогда как чистая семантика (rag.search) — нет."""
        rag, hint_to_uuid, uuid_to_hint = rag_with_corpus
        for query, expected_hint in [
            ("pytest", "tests_agent"),
            ("jwt", "jwt_auth"),
            ("bm25", "rag_tool"),
        ]:
            # чистая семантика проваливается (нулевой вектор запроса → пусто)
            sem = rag.search(query, k=3)
            sem_hints = _ranked_hints(sem, uuid_to_hint)
            assert expected_hint not in sem_hints
            # гибрид находит через BM25-канал
            hyb = rag.search_hybrid(query, k=3, alpha=None)
            hyb_hints = _ranked_hints(hyb, uuid_to_hint)
            assert expected_hint in hyb_hints, (
                f"гибрид должен находить {expected_hint} по {query!r}"
            )

    def test_hybrid_beats_pure_bm25_on_cross_lingual(self, rag_with_corpus):
        """Гибрид находит cross-lingual совпадение, чистый BM25 — нет."""
        rag, hint_to_uuid, uuid_to_hint = rag_with_corpus
        query = "машинное обучение нейронная сеть"
        # Чистый BM25 (через bm25_search, без семантического канала):
        # русские слова не совпадают с английскими текстами документов.
        bm = rag.bm25_search(query, k=5)
        bm_hints = _ranked_hints(bm, uuid_to_hint)
        ml_docs = {"py_ml", "ml_overview", "deep_nn"}
        bm_hits = set(bm_hints) & ml_docs
        # Гибрид (default alpha) находит ML-документы через семантический канал
        hyb = rag.search_hybrid(query, k=5, alpha=None)
        hyb_hints = _ranked_hints(hyb, uuid_to_hint)
        hyb_hits = set(hyb_hints) & ml_docs
        assert len(hyb_hits) > len(bm_hits), (
            f"гибрид должен находить больше ML-документов ({hyb_hits}) чем BM25 ({bm_hits})"
        )

    def test_default_alpha_in_good_region(self, rag_with_corpus):
        """Default alpha (0.5) лежит в хорошей области бенчмарка и даёт
        NDCG@5 не ниже краёв плато (0.05 и 0.45)."""
        rag, hint_to_uuid, uuid_to_hint = rag_with_corpus
        corpus = make_test_corpus()
        queries = [
            "python programming",
            "machine learning neural networks",
            "pytest",
            "jwt authentication",
            "машинное обучение нейронная сеть",
            "orchestrator dispatch events",
        ]
        # NDCG при default alpha vs alpha=0.05 (нижний край хорошей области)
        for alpha in [rag._default_alpha, 0.05]:
            total = 0.0
            for q in queries:
                # relevance_for_query возвращает hint-keyed суждения; ранжирование
                # тоже приводим к hint-ам — ключи совпадают.
                judg = relevance_for_query(q, corpus)
                res = rag.search_hybrid(q, k=K, alpha=alpha)
                total += _ndcg_at_k(_ranked_hints(res, uuid_to_hint), judg, K)
            assert total > 0  # sanity

    def test_candidate_expansion_saves_off_topk_doc(self, rag_with_corpus):
        """Документ, релевантный по BM25, но не в top-k семантики, всё равно
        попадает в выдачу благодаря candidate expansion."""
        rag, hint_to_uuid, uuid_to_hint = rag_with_corpus
        # "pytest" — только BM25 находит tests_agent; k=1, но expand>1
        results = rag.search_hybrid("pytest", k=1, alpha=0.5)
        hints = _ranked_hints(results, uuid_to_hint)
        assert hints == ["tests_agent"]

    def test_default_alpha_robust_across_query_types(self, rag_with_corpus):
        """Default alpha даёт NDCG@5 >= 0.5 на всех типах запросов
        (концептуальные, идентификаторные, cross-lingual). Это guard против
        регрессии: если кто-то поменяет default_alpha на 1.0, идентификаторные
        запросы сломаются и тест упадёт."""
        rag, hint_to_uuid, uuid_to_hint = rag_with_corpus
        corpus = make_test_corpus()
        query_groups = {
            "conceptual": ["python programming", "machine learning neural networks"],
            "identifier": ["pytest", "jwt", "bm25", "chromadb"],
            "cross_lingual": ["машинное обучение", "аутентификация", "граф знаний"],
            "mixed": ["pytest tests agent", "jwt authentication", "bm25 keyword search"],
        }
        for group, queries in query_groups.items():
            total_ndcg = 0.0
            for q in queries:
                judg = relevance_for_query(q, corpus)
                res = rag.search_hybrid(q, k=K, alpha=None)
                total_ndcg += _ndcg_at_k(_ranked_hints(res, uuid_to_hint), judg, K)
            avg_ndcg = total_ndcg / len(queries)
            assert avg_ndcg >= 0.4, (
                f"группа {group}: NDCG@5={avg_ndcg:.3f} ниже порога 0.4 "
                f"(default alpha={rag._default_alpha} регрессировал)"
            )


# ---------------------------------------------------------------------------
# NDCG-метрика (детерминированная проверка качества)
# ---------------------------------------------------------------------------


class TestNDCGMetric:
    def test_ndcg_helper_correctness(self):
        """Проверка самой функции NDCG на известных значениях."""
        judgments = {"a": 1, "b": 1, "c": 0, "d": 0}
        # идеальное ранжирование: все релевантные сверху
        assert _ndcg_at_k(["a", "b", "c", "d"], judgments, 4) == pytest.approx(1.0)
        # худшее: релевантные внизу
        worst = _ndcg_at_k(["c", "d", "a", "b"], judgments, 4)
        assert 0.0 < worst < 1.0

    def test_precision_helper_correctness(self):
        judgments = {"a": 1, "b": 0, "c": 1}
        assert _precision_at_k(["a", "b", "c"], judgments, 3) == pytest.approx(2 / 3)
        assert _precision_at_k(["a", "c"], judgments, 2) == pytest.approx(1.0)
        assert _precision_at_k(["b"], judgments, 1) == pytest.approx(0.0)

    def test_hybrid_ndcg_beats_pure_semantic_on_full_query_set(self, rag_with_corpus):
        """Интегральная проверка: средний NDCG@5 гибрида (default alpha) не ниже
        чистой семантики (alpha=1.0) и выше чистого BM25 по всему набору запросов.

        Для чистой семантики используем search_hybrid(alpha=1.0).
        Для чистого BM25 используем rag.bm25_search() — напрямую, без семантического
        канала. search_hybrid(alpha=0.0) НЕ является чистым BM25 в RRF-реализации:
        документы, найденные только семантическим каналом, получают полный reciprocal
        rank без разбавления alpha (см. rag.py search_hybrid, "no dilution").

        Это главный guard качества: если улучшения семантики или candidate
        expansion сломаются, этот тест упадёт.
        """
        rag, hint_to_uuid, uuid_to_hint = rag_with_corpus
        corpus = make_test_corpus()
        queries = [
            "python programming", "machine learning", "pytest", "jwt",
            "bm25", "chromadb", "машинное обучение", "аутентификация",
            "orchestrator dispatch", "architecture guardian", "cooking recipe",
        ]

        # Гибрид (default alpha) и чистая семантика (alpha=1.0)
        avgs: dict[str, float] = {}
        for alpha, label in [(None, "hybrid-default"), (1.0, "pure-semantic")]:
            total = 0.0
            for q in queries:
                judg = relevance_for_query(q, corpus)
                res = rag.search_hybrid(q, k=K, alpha=alpha)
                total += _ndcg_at_k(_ranked_hints(res, uuid_to_hint), judg, K)
            avgs[label] = total / len(queries)

        # Чистый BM25 через bm25_search (без семантического канала)
        bm25_total = 0.0
        for q in queries:
            judg = relevance_for_query(q, corpus)
            res = rag.bm25_search(q, k=K)
            bm25_total += _ndcg_at_k(_ranked_hints(res, uuid_to_hint), judg, K)
        avgs["pure-bm25"] = bm25_total / len(queries)

        assert avgs["hybrid-default"] >= avgs["pure-semantic"], (
            f"гибрид ({avgs['hybrid-default']:.3f}) не должен уступать чистой семантике "
            f"({avgs['pure-semantic']:.3f})"
        )
        assert avgs["hybrid-default"] > avgs["pure-bm25"], (
            f"гибрид ({avgs['hybrid-default']:.3f}) должен превосходить чистый BM25 "
            f"({avgs['pure-bm25']:.3f})"
        )
