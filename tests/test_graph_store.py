"""Тесты для GraphStore (рёбра в SQLite + NetworkX-кеш).

Узлы графа — документы из DocumentStore: перед add_edge узлы создаются
через doc_store.add(). Сохранение автоматическое (SQL), метод search()
у графа удалён.
"""

import pytest

from src.document_store import Database, DocumentStore
from src.graph_store import GraphStore


@pytest.fixture
def db(tmp_path):
    """SQLite Database во временной директории; close() в teardown (Windows)."""
    database = Database(str(tmp_path / "store.db"))
    yield database
    database.close()


@pytest.fixture
def docs(db):
    return DocumentStore(db)


@pytest.fixture
def graph(db, docs):
    return GraphStore(db, docs)


def add_docs(docs: DocumentStore, *doc_ids: str) -> None:
    """Создать документы-узлы с текстом-заглушкой."""
    for doc_id in doc_ids:
        docs.add(doc_id, f"Text of {doc_id}")


class TestGraphStore:
    """Тесты для класса GraphStore."""

    def test_init_empty(self, graph):
        """Тест инициализации пустого графа."""
        stats = graph.stats()
        assert stats["total_nodes"] == 0
        assert stats["total_edges"] == 0
        assert stats["relation_types"] == []

    def test_add_edge_and_get_related(self, graph, docs):
        """Тест добавления рёбер и получения связанных узлов."""
        add_docs(docs, "node1", "node2", "node3")

        graph.add_edge("node1", "node2", "related_to", 1.0)
        graph.add_edge("node1", "node3", "similar_to", 0.8)

        related = graph.get_related("node1", max_depth=1)
        assert len(related) == 2

        relations = [(s, t, r) for s, t, r, w, d in related]
        assert ("node1", "node2", "related_to") in relations
        assert ("node1", "node3", "similar_to") in relations

    def test_delete_document_cascades_edges(self, graph, docs, db):
        """Удаление документа каскадно удаляет его рёбра (SQL + кеш).

        Как в RAGSystem.delete_document: DocumentStore.delete() удаляет
        строки graph_edges через FK CASCADE, GraphStore.delete_document()
        убирает узел из NetworkX-кеша.
        """
        add_docs(docs, "node1", "node2", "node3")
        graph.add_edge("node1", "node2", "related_to", 1.0)
        graph.add_edge("node2", "node3", "similar_to", 0.8)

        assert docs.delete("node2") is True
        graph.delete_document("node2")

        # SQL: оба ребра удалены каскадом
        rows = db.execute("SELECT COUNT(*) FROM graph_edges")
        assert rows[0][0] == 0

        # Кеш: рёбер нет, узлы = оставшиеся документы
        stats = graph.stats()
        assert stats["total_edges"] == 0
        assert stats["total_nodes"] == 2
        assert graph.get_related("node1") == []

    def test_remove_edge(self, graph, docs):
        """Тест удаления ребра."""
        add_docs(docs, "node1", "node2")
        graph.add_edge("node1", "node2", "related_to", 1.0)

        assert graph.stats()["total_edges"] == 1

        graph.remove_edge("node1", "node2", "related_to")
        assert graph.stats()["total_edges"] == 0

        # Удаление несуществующего ребра - silent
        graph.remove_edge("node1", "node2", "related_to")
        assert graph.stats()["total_edges"] == 0

    def test_delete_all(self, graph, docs):
        """Тест очистки графа и документов."""
        add_docs(docs, "node1", "node2")
        graph.add_edge("node1", "node2", "related_to", 1.0)

        graph.delete_all()
        docs.clear()

        stats = graph.stats()
        assert stats["total_nodes"] == 0
        assert stats["total_edges"] == 0

    def test_get_relation_types(self, graph, docs):
        """Тест получения типов отношений."""
        add_docs(docs, "node1", "node2", "node3")

        graph.add_edge("node1", "node2", "related_to", 1.0)
        graph.add_edge("node1", "node3", "similar_to", 0.8)
        graph.add_edge("node2", "node3", "related_to", 0.5)

        relation_types = graph.get_relation_types()
        assert relation_types == ["related_to", "similar_to"]

    def test_add_edge_missing_node_raises(self, graph, docs):
        """Тест ValueError при добавлении ребра с несуществующим узлом."""
        add_docs(docs, "node1")

        with pytest.raises(ValueError, match="Target node"):
            graph.add_edge("node1", "node2", "related_to", 1.0)

        with pytest.raises(ValueError, match="Source node"):
            graph.add_edge("node2", "node1", "related_to", 1.0)

    def test_get_related_max_depth(self, graph, docs):
        """Тест BFS с глубиной > 1."""
        add_docs(docs, "node1", "node2", "node3", "node4")

        graph.add_edge("node1", "node2", "related_to", 1.0)
        graph.add_edge("node2", "node3", "related_to", 1.0)
        graph.add_edge("node3", "node4", "related_to", 1.0)

        # max_depth=1: только node1->node2
        related_depth1 = graph.get_related("node1", max_depth=1)
        assert len(related_depth1) == 1
        assert related_depth1[0][1] == "node2"

        # max_depth=2: node1->node2, node2->node3
        related_depth2 = graph.get_related("node1", max_depth=2)
        assert len(related_depth2) == 2

        # max_depth=3: все три ребра
        related_depth3 = graph.get_related("node1", max_depth=3)
        assert len(related_depth3) == 3

    def test_duplicate_edge_updates_weight(self, graph, docs):
        """Тест обновления веса при повторном добавлении ребра (upsert)."""
        add_docs(docs, "node1", "node2")

        graph.add_edge("node1", "node2", "related_to", 1.0)
        assert graph.get_all_edges() == [("node1", "node2", "related_to", 1.0)]

        graph.add_edge("node1", "node2", "related_to", 0.5)
        assert graph.get_all_edges() == [("node1", "node2", "related_to", 0.5)]
        # Ребро одно, не дубликат
        assert graph.stats()["total_edges"] == 1

    def test_degree(self, graph, docs):
        """degree считает входящие и исходящие рёбра узла."""
        add_docs(docs, "a", "b", "c")
        graph.add_edge("a", "b", "related_to", 1.0)
        graph.add_edge("c", "a", "depends_on", 0.5)

        assert graph.degree("a") == 2
        assert graph.degree("b") == 1
        assert graph.degree("nonexistent") == 0

    def test_remove_phantom_edges(self, graph, docs):
        """remove_phantom_edges удаляет рёбра к узлам вне valid_ids."""
        add_docs(docs, "a", "b", "c")
        graph.add_edge("a", "b", "related_to", 1.0)
        graph.add_edge("b", "c", "similar_to", 0.8)

        removed = graph.remove_phantom_edges({"a", "b"})
        assert removed == 1
        assert graph.get_all_edges() == [("a", "b", "related_to", 1.0)]

        # Все id валидны — ничего не удаляется
        assert graph.remove_phantom_edges({"a", "b", "c"}) == 0

    # ── D3: Bidirectional BFS ──────────────────────────────────────────

    def test_get_related_bidirectional(self, graph, docs):
        """D3: get_related находит входящие рёбра (direction='in')."""
        add_docs(docs, "source", "target")
        graph.add_edge("source", "target", "uses", 1.0)

        # Исходящие рёбра от source
        related_out = graph.get_related("source", max_depth=1, direction="out")
        assert len(related_out) == 1
        assert related_out[0][1] == "target"
        assert related_out[0][4] == "out"

        # Входящие рёбра к target (должен найти source → target, но как "in")
        related_in = graph.get_related("target", max_depth=1, direction="in")
        assert len(related_in) == 1, f"Рёбер от target как target: {related_in}"
        assert related_in[0][0] == "source"
        assert related_in[0][1] == "target"
        assert related_in[0][4] == "in"

    def test_get_related_default_is_both(self, graph, docs):
        """D3: По умолчанию direction='both', обход включает оба направления."""
        add_docs(docs, "a", "b", "c")
        graph.add_edge("a", "b", "related_to", 1.0)
        graph.add_edge("c", "a", "depends_on", 0.8)

        # От 'a' — два ребра: a→b (out) и c→a (in)
        related = graph.get_related("a", max_depth=1, direction="both")
        assert len(related) == 2
        directions = {r[4] for r in related}
        assert "out" in directions
        assert "in" in directions

    def test_get_related_depth2_bidirectional(self, graph, docs):
        """D3: BFS глубиной 2 обходит оба направления."""
        add_docs(docs, "a", "b", "c")
        graph.add_edge("a", "b", "related_to", 1.0)
        graph.add_edge("c", "b", "depends_on", 0.8)

        # От 'a': a→b (depth 1), потом c→b (depth 2, входящее к b)
        related = graph.get_related("a", max_depth=2, direction="both")
        assert len(related) == 2
        pairs = [(r[0], r[1]) for r in related]
        assert ("a", "b") in pairs
        assert ("c", "b") in pairs

    def test_get_related_unknown_node_returns_empty(self, graph):
        """get_related для несуществующего узла возвращает [].

        regression: не падает с KeyError.
        """
        assert graph.get_related("nonexistent") == []

    def test_get_related_single_node_no_edges(self, graph, docs):
        """get_related для узла без рёбер возвращает []."""
        add_docs(docs, "lonely")
        assert graph.get_related("lonely") == []

    def test_get_related_metadata_filter(self, graph, docs):
        """metadata_filter применяется к соседнему узлу."""
        docs.add("hub", "Hub", {"type": "hub"})
        docs.add("person", "Person doc", {"type": "person"})
        docs.add("place", "Place doc", {"type": "place"})
        graph.add_edge("hub", "person", "mentions", 1.0)
        graph.add_edge("hub", "place", "mentions", 1.0)

        related = graph.get_related("hub", metadata_filter={"type": "person"})
        assert len(related) == 1
        assert related[0][1] == "person"

        # Список — семантика $in
        related_in = graph.get_related("hub", metadata_filter={"type": ["person", "place"]})
        assert len(related_in) == 2

        # Никто не проходит
        assert graph.get_related("hub", metadata_filter={"type": "event"}) == []

    # ── get_edges_batch ─────────────────────────────────────────────────

    def test_get_edges_batch_empty(self, graph, docs):
        """get_edges_batch с пустым набором или без связей."""
        add_docs(docs, "a")
        result = graph.get_edges_batch(set())
        assert result == {}

        # Узел без рёбер отсутствует в NetworkX-кеше
        result = graph.get_edges_batch({"a"})
        assert result == {}

    def test_get_edges_batch_basic(self, graph, docs):
        """get_edges_batch возвращает связи для нескольких узлов."""
        add_docs(docs, "a", "b", "c")
        graph.add_edge("a", "b", "related_to", 1.0)
        graph.add_edge("a", "c", "similar_to", 0.8)

        result = graph.get_edges_batch({"a"})
        assert "a" in result
        assert "b" in result["a"]
        assert "c" in result["a"]
        assert len(result["a"]["b"]) == 1
        assert result["a"]["b"][0]["relation"] == "related_to"
        assert result["a"]["b"][0]["weight"] == 1.0
        assert result["a"]["b"][0]["direction"] == "out"

    def test_get_edges_batch_multiple_nodes(self, graph, docs):
        """get_edges_batch для нескольких source node."""
        add_docs(docs, "a", "b", "c")
        graph.add_edge("a", "b", "related_to", 1.0)
        graph.add_edge("b", "c", "depends_on", 0.5)

        result = graph.get_edges_batch({"a", "b"})
        assert "a" in result
        assert "b" in result["a"]
        assert "b" in result
        assert "c" in result["b"]

    def test_get_edges_batch_with_depth(self, graph, docs):
        """get_edges_batch с BFS глубиной > 1."""
        add_docs(docs, "a", "b", "c")
        graph.add_edge("a", "b", "related_to", 1.0)
        graph.add_edge("b", "c", "related_to", 1.0)

        result = graph.get_edges_batch({"a"}, max_depth=1)
        assert "b" in result["a"]
        assert "c" not in result["a"]

        result = graph.get_edges_batch({"a"}, max_depth=2)
        assert "b" in result["a"]
        assert "c" in result["a"]

    def test_get_edges_batch_type_filter(self, graph, docs):
        """get_edges_batch фильтрует по типу связи."""
        add_docs(docs, "a", "b", "c")
        graph.add_edge("a", "b", "related_to", 1.0)
        graph.add_edge("a", "c", "similar_to", 0.8)

        result = graph.get_edges_batch({"a"}, relation_type_filter=["related_to"])
        assert "b" in result["a"]
        assert "c" not in result["a"]

    def test_get_edges_batch_metadata_filter(self, graph, docs):
        """get_edges_batch фильтрует соседей по метаданным."""
        docs.add("hub", "Hub", {"type": "hub"})
        docs.add("person", "Person doc", {"type": "person"})
        docs.add("place", "Place doc", {"type": "place"})
        graph.add_edge("hub", "person", "mentions", 1.0)
        graph.add_edge("hub", "place", "mentions", 1.0)

        result = graph.get_edges_batch({"hub"}, metadata_filter={"type": "person"})
        assert "person" in result["hub"]
        assert "place" not in result["hub"]

    def test_get_edges_batch_unknown_node(self, graph):
        """get_edges_batch для несуществующего узла."""
        result = graph.get_edges_batch({"nonexistent"})
        assert result == {}
