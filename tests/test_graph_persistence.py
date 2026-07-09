"""Тесты персистентности графа: рёбра переживают переоткрытие Database.

Сохранение автоматическое (каждая мутация — SQL-запись), «перезапуск»
имитируется закрытием Database и созданием новых Database/DocumentStore/
GraphStore на том же SQLite-файле: GraphStore перечитывает рёбра из
таблицы graph_edges (_build_graph).

Windows: Database держит файл открытым — обязательно close() до конца
теста (tmp_path). WAL создаёт -wal/-shm файлы рядом с .db — это норма.
"""

import pytest

from src.document_store import Database, DocumentStore
from src.graph_store import GraphStore


@pytest.fixture
def open_store(tmp_path):
    """Фабрика: открыть Database/DocumentStore/GraphStore на общем .db файле.

    Все открытые Database закрываются в teardown (повторный close безопасен).
    """
    db_path = str(tmp_path / "store.db")
    opened: list[Database] = []

    def _open():
        db = Database(db_path)
        opened.append(db)
        docs = DocumentStore(db)
        graph = GraphStore(db, docs)
        return db, docs, graph

    yield _open
    for db in opened:
        try:
            db.close()
        except Exception:
            pass


class TestGraphPersistence:
    """Тесты сохранения и загрузки графа через переоткрытие SQLite-файла."""

    def test_reopen_empty(self, open_store):
        """Переоткрытие пустой базы: граф пуст."""
        db, docs, graph = open_store()
        assert graph.stats()["total_edges"] == 0
        db.close()

        db2, docs2, graph2 = open_store()
        assert graph2.stats()["total_nodes"] == 0
        assert graph2.stats()["total_edges"] == 0

    def test_documents_persist(self, open_store):
        """Документы-узлы переживают переоткрытие."""
        db, docs, graph = open_store()
        docs.add("node1", "Python is a programming language", {"lang": "python"})
        docs.add("node2", "Django is a web framework", {"framework": "django"})
        db.close()

        db2, docs2, graph2 = open_store()
        assert graph2.stats()["total_nodes"] == 2

        doc = docs2.get("node1")
        assert doc is not None
        assert doc["text"] == "Python is a programming language"
        assert doc["metadata"]["lang"] == "python"

    def test_edges_persist(self, open_store):
        """Рёбра с весами переживают переоткрытие."""
        db, docs, graph = open_store()
        for node in ("node1", "node2", "node3"):
            docs.add(node, f"Text of {node}")
        graph.add_edge("node1", "node2", "related_to", 1.0)
        graph.add_edge("node1", "node3", "similar_to", 0.8)
        db.close()

        db2, docs2, graph2 = open_store()
        assert graph2.stats()["total_edges"] == 2

        edges = {(s, t, r): w for s, t, r, w in graph2.get_all_edges()}
        assert edges[("node1", "node2", "related_to")] == 1.0
        assert edges[("node1", "node3", "similar_to")] == 0.8

    def test_add_after_reopen(self, open_store):
        """Добавление документов и рёбер после переоткрытия работает."""
        db, docs, graph = open_store()
        docs.add("node1", "Python programming")
        db.close()

        db2, docs2, graph2 = open_store()
        docs2.add("node2", "Java programming")
        graph2.add_edge("node1", "node2", "similar_to", 1.0)
        db2.close()

        db3, docs3, graph3 = open_store()
        assert graph3.stats()["total_nodes"] == 2
        assert graph3.stats()["total_edges"] == 1

    def test_delete_all_persists(self, open_store):
        """delete_all + clear очищают базу насовсем."""
        db, docs, graph = open_store()
        docs.add("node1", "Text 1")
        docs.add("node2", "Text 2")
        graph.add_edge("node1", "node2", "related_to", 1.0)

        graph.delete_all()
        docs.clear()
        db.close()

        db2, docs2, graph2 = open_store()
        assert graph2.stats()["total_nodes"] == 0
        assert graph2.stats()["total_edges"] == 0

    def test_get_related_after_reopen(self, open_store):
        """BFS работает после переоткрытия."""
        db, docs, graph = open_store()
        for node in ("node1", "node2", "node3"):
            docs.add(node, f"Text of {node}")
        graph.add_edge("node1", "node2", "related_to", 1.0)
        graph.add_edge("node2", "node3", "related_to", 1.0)
        db.close()

        db2, docs2, graph2 = open_store()
        related = graph2.get_related("node1", max_depth=2)
        assert len(related) == 2
        pairs = [(r[0], r[1]) for r in related]
        assert ("node1", "node2") in pairs
        assert ("node2", "node3") in pairs

    def test_get_relation_types_after_reopen(self, open_store):
        """Типы отношений доступны после переоткрытия."""
        db, docs, graph = open_store()
        for node in ("node1", "node2", "node3"):
            docs.add(node, f"Text of {node}")
        graph.add_edge("node1", "node2", "related_to", 1.0)
        graph.add_edge("node1", "node3", "similar_to", 0.8)
        db.close()

        db2, docs2, graph2 = open_store()
        assert graph2.get_relation_types() == ["related_to", "similar_to"]

    def test_delete_document_cascade_persists(self, open_store):
        """Каскадное удаление рёбер (FK CASCADE) видно после переоткрытия."""
        db, docs, graph = open_store()
        docs.add("node1", "Text 1")
        docs.add("node2", "Text 2")
        graph.add_edge("node1", "node2", "related_to", 1.0)

        # Удаляем документ только через DocumentStore — SQL-состояние
        # проверяем переоткрытием (кеш второго GraphStore строится из таблицы)
        assert docs.delete("node1") is True
        db.close()

        db2, docs2, graph2 = open_store()
        assert docs2.get("node1") is None
        assert graph2.stats()["total_nodes"] == 1
        assert graph2.stats()["total_edges"] == 0
        assert graph2.get_related("node2") == []

    def test_remove_edge_persists(self, open_store):
        """Удаление ребра переживает переоткрытие."""
        db, docs, graph = open_store()
        docs.add("node1", "Text 1")
        docs.add("node2", "Text 2")
        graph.add_edge("node1", "node2", "related_to", 1.0)
        graph.remove_edge("node1", "node2", "related_to")
        db.close()

        db2, docs2, graph2 = open_store()
        assert graph2.stats()["total_edges"] == 0

    def test_edge_weight_update_persists(self, open_store):
        """Upsert веса ребра переживает переоткрытие."""
        db, docs, graph = open_store()
        docs.add("node1", "Text 1")
        docs.add("node2", "Text 2")
        graph.add_edge("node1", "node2", "related_to", 1.0)
        graph.add_edge("node1", "node2", "related_to", 0.5)
        db.close()

        db2, docs2, graph2 = open_store()
        assert graph2.get_all_edges() == [("node1", "node2", "related_to", 0.5)]

    def test_db_file_created(self, tmp_path):
        """Database создаёт SQLite-файл по указанному пути."""
        db_path = tmp_path / "sub" / "store.db"
        db = Database(str(db_path))
        try:
            assert db_path.exists()
        finally:
            db.close()
