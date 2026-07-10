"""Детерминированный генератор семантических эмбеддингов для тестов качества.

Реальная семантическая структура без внешних API: каждое понятие (concept)
отображается в свой базисный вектор (one-hot), текст → нормализованная сумма
векторов понятий, встреченных в нём. Поддержаны синонимы (например
"authentication" → понятие "auth"), что позволяет тестировать семантическое
сходство при разном wording, включая cross-lingual (русские термины тоже
разложены по тем же понятиям).

Этот генератор используется в scripts/benchmark_alpha.py и в
tests/test_search_quality.py — он даёт воспроизводимые, осмысленные векторы,
на которых можно объективно измерить качество поиска (NDCG@k).
"""

from __future__ import annotations

import re
import math


# Понятие → список слов/фраз, которые на него отображаются (lowercase).
# Включены английские и русские формы чтобы тестировать cross-lingual сходство.
#
# ВАЖНО: специфичные идентификаторы (pytest, jwt, bm25, chromadb, uuid и т.п.)
# намеренно ОТСУТСТВУЮТ в лексиконе. Это моделирует реальное поведение эмбеддингов:
# они улавливают концепции (auth, tests, ml), но не точные имена инструментов/
# протоколов. Такие идентификаторы находятся только через BM25 (точное совпадение
# токенов), что создаёт настоящую комплементарность каналов и делает выбор alpha
# содержательным.
_CONCEPT_LEXICON: dict[str, list[str]] = {
    "python": ["python", "питон", "пайтон"],
    "java": ["java", "ява"],
    "programming": ["programming", "программирование", "language", "язык"],
    "machine_learning": ["machine", "learning", "ml", "машинное", "обучение", "нейросеть"],
    "neural": ["neural", "network", "deep", "нейронная", "сеть"],
    "cooking": ["cooking", "recipe", "pasta", "food", "готовка", "рецепт", "еда"],
    "auth": ["auth", "authentication", "login", "аутентификация", "вход"],
    "orchestrator": ["orchestrator", "dispatch", "event", "оркестратор", "диспетчер", "событие"],
    "agent": ["agent", "агент"],
    "architecture": ["architecture", "guardian", "архитектура", "хранитель"],
    "code": ["code", "ast", "код"],
    "tests": ["test", "tests", "тест", "тесты"],
    "wiki": ["wiki", "documentation", "docs", "вики", "документация"],
    "journal": ["journal", "журнал", "лог"],
    "meeting": ["meeting", "transcript", "встреча", "транскрипт"],
    "rag": ["rag", "retrieval", "эмбеддинг", "вектор"],
    "graph": ["graph", "knowledge", "граф", "знание"],
    "keyword": ["keyword", "ключевое"],
    "security": ["security", "безопасность"],
    "database": ["database", "storage", "база", "хранилище"],
}


def _build_word_to_concept() -> dict[str, str]:
    """Слово → понятие (первое совпадение выигрывает)."""
    mapping: dict[str, str] = {}
    for concept, words in _CONCEPT_LEXICON.items():
        for w in words:
            mapping.setdefault(w, concept)
    return mapping


_WORD_TO_CONCEPT = _build_word_to_concept()
_DIMENSION = len(_CONCEPT_LEXICON)
_CONCEPT_INDEX = {concept: i for i, concept in enumerate(sorted(_CONCEPT_LEXICON))}


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-zа-яё]+", text.lower())


class SemanticMockEmbeddingGenerator:
    """Генератор эмбеддингов с настоящей семантической структурой.

    Каждый текст превращается в вектор размерности len(_CONCEPT_LEXICON):
    на позицию понятия кладётся взвешенный по TF вклад, затем L2-нормализация.
    Тексты, разделяющие понятия, получаются близкими по косинусу; без общих
    понятий — ортогональными. Синонимы (включая русские) расширяют покрытие.
    """

    def __init__(self, dimension: int = _DIMENSION):
        self._dimension = dimension

    def get_embedding(self, text: str) -> list[float]:
        tokens = _tokenize(text)
        if not tokens:
            return [0.0] * self._dimension

        tf: dict[str, float] = {}
        for tok in tokens:
            tf[tok] = tf.get(tok, 0.0) + 1.0
        total = sum(tf.values())
        for tok in tf:
            tf[tok] /= total

        vec = [0.0] * self._dimension
        for word, weight in tf.items():
            concept = _WORD_TO_CONCEPT.get(word)
            if concept is None:
                continue
            idx = _CONCEPT_INDEX[concept]
            vec[idx] += weight

        norm = math.sqrt(sum(v * v for v in vec))
        if norm > 0:
            vec = [v / norm for v in vec]
        return vec

    def get_embeddings(self, texts: list[str]) -> list[list[float]]:
        return [self.get_embedding(t) for t in texts]

    def get_dimension(self) -> int:
        return self._dimension

    def clear_cache(self) -> None:
        pass


def make_test_corpus() -> list[tuple[str, str, list[str]]]:
    """Вернуть тестовый корпус: (doc_id_hint, text, relevant_concepts).

    relevant_concepts — понятия, которые запрос может "зацепить" семантически.
    Документы со специфичными идентификаторами (pytest, jwt, bm25, chromadb)
    релевантны запросам по этим идентификаторам, но семантический канал их не
    видит (идентификаторов нет в лексиконе) — их находит только BM25. Это
    моделирует реальную комплементарность каналов.
    """
    return [
        ("py_lang", "Python is a programming language with simple syntax", ["python", "programming"]),
        ("py_ml", "Python is great for machine learning and neural networks", ["python", "machine_learning", "neural"]),
        ("java_vm", "Java runs on a virtual machine and is statically typed", ["java", "programming"]),
        ("ml_overview", "Machine learning uses algorithms to learn from data", ["machine_learning"]),
        ("deep_nn", "Deep learning neural networks for computer vision tasks", ["neural", "machine_learning"]),
        ("pasta", "Cooking recipe for italian pasta with tomato sauce", ["cooking"]),
        # jwt — идентификатор, виден только BM25; auth — концепция, видна семантике.
        ("jwt_auth", "JWT tokens for authentication and authorization flows", ["auth", "security", "_id:jwt"]),
        ("orchestrator", "Orchestrator dispatches events and launches agents for tasks", ["orchestrator", "agent"]),
        ("arch_guardian", "Architecture guardian enforces architectural rules and standards", ["architecture"]),
        ("code_guardian", "Code guardian checks source code against project requirements", ["code", "architecture"]),
        # pytest — идентификатор, виден только BM25; tests — концепция.
        ("tests_agent", "Tests agent generates pytest unit tests automatically", ["tests", "agent", "_id:pytest"]),
        ("wiki_guardian", "Wiki guardian keeps documentation fresh and consistent", ["wiki", "architecture"]),
        ("journal_service", "Journal service stores project decisions and research findings", ["journal", "database"]),
        ("meeting_agent", "Meeting agent processes transcripts and extracts decisions", ["meeting", "agent"]),
        # bm25/chromadb — идентификаторы, видны только BM25.
        ("rag_tool", "RAG tool with vector embeddings and bm25 keyword search", ["rag", "graph", "_id:bm25"]),
        ("graph_kb", "Knowledge graph with typed relations stored in chromadb", ["graph", "database", "_id:chromadb"]),
    ]


# Список понятий по индексу в порядке _CONCEPT_INDEX (sorted).
_CONCEPT_BY_INDEX = sorted(_CONCEPT_LEXICON)


def relevance_for_query(query: str, corpus: list[tuple[str, str, list[str]]]) -> dict[str, int]:
    """Построить бинарные суждения релевантности (1/0) для запроса.

    Документ релевантен, если выполняется хотя бы одно из:
      • пересечение понятий запроса и relevant_concepts документа непусто
        (семантическое соответствие, включая синонимы и cross-lingual);
      • запрос содержит слово-идентификатор, отмеченное в relevant_concepts
        документа маркером "_id:<word>" (точное соответствие через BM25).

    Это моделирует два реальных пути нахождения документа и позволяет бенчмарку
    объективно оценивать как семантический, так и keyword каналы поиска.
    """
    gen = SemanticMockEmbeddingGenerator()
    q_emb = gen.get_embedding(query)
    active_concepts = {_CONCEPT_BY_INDEX[i] for i, v in enumerate(q_emb) if v > 0}

    # Идентификаторы в запросе — слова, не попавшие ни в одно понятие.
    q_tokens = set(_tokenize(query))
    recognized = set()
    for words in _CONCEPT_LEXICON.values():
        recognized.update(words)
    query_identifiers = q_tokens - recognized

    judgments: dict[str, int] = {}
    for doc_id, _text, rel_concepts in corpus:
        rel_concept_set = set(rel_concepts)
        doc_identifiers = {c[len("_id:"):] for c in rel_concepts if c.startswith("_id:")}
        concept_match = bool(active_concepts & rel_concept_set)
        id_match = bool(query_identifiers & doc_identifiers)
        judgments[doc_id] = 1 if (concept_match or id_match) else 0
    return judgments
