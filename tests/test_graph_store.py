"""Тесты для графовой базы знаний."""

import pytest
from src.graph_store import GraphKnowledgeBase


class TestGraphKnowledgeBase:
    """Тесты для класса GraphKnowledgeBase."""

    def test_init_empty(self):
        """Тест инициализации пустого графа."""
        gkb = GraphKnowledgeBase()
        stats = gkb.stats()
        assert stats["total_nodes"] == 0
        assert stats["total_edges"] == 0
        assert stats["relation_types"] == []

    def test_add_and_search_basic(self):
        """Тест добавления узлов и поиска по тексту."""
        gkb = GraphKnowledgeBase()
        gkb.add_node("node1", "Python is a programming language")
        gkb.add_node("node2", "Django is a web framework")

        results = gkb.search("Python programming", k=2)
        assert len(results) == 2
        assert results[0][0] == "node1"  # node1 должен быть первым
        assert "Python" in results[0][1]

    def test_add_edge_and_get_related(self):
        """Тест добавления рёбер и получения связанных узлов."""
        gkb = GraphKnowledgeBase()
        gkb.add_node("node1", "Text 1")
        gkb.add_node("node2", "Text 2")
        gkb.add_node("node3", "Text 3")

        gkb.add_edge("node1", "node2", "related_to", 1.0)
        gkb.add_edge("node1", "node3", "similar_to", 0.8)

        related = gkb.get_related("node1", max_depth=1)
        assert len(related) == 2

        relations = [(s, t, r) for s, t, r, w in related]
        assert ("node1", "node2", "related_to") in relations
        assert ("node1", "node3", "similar_to") in relations

    def test_search_with_relation_expansion(self):
        """Тест поиска с расширением на связанные узлы."""
        gkb = GraphKnowledgeBase()
        gkb.add_node("node1", "Python programming language")
        gkb.add_node("node2", "Django web framework")
        gkb.add_edge("node1", "node2", "related_to", 1.0)

        # Поиск без расширения
        results_no_expand = gkb.search("Python", k=1, expand_relations=False)
        assert len(results_no_expand) == 1
        assert results_no_expand[0][0] == "node1"

        # Поиск с расширением
        results_expand = gkb.search("Python", k=1, expand_relations=True)
        assert len(results_expand) >= 1
        node_ids = [r[0] for r in results_expand]
        assert "node1" in node_ids
        # node2 должен быть добавлен через расширение
        assert "node2" in node_ids

    def test_remove_node_cascades_edges(self):
        """Тест удаления узла и каскадного удаления рёбер."""
        gkb = GraphKnowledgeBase()
        gkb.add_node("node1", "Text 1")
        gkb.add_node("node2", "Text 2")
        gkb.add_node("node3", "Text 3")

        gkb.add_edge("node1", "node2", "related_to", 1.0)
        gkb.add_edge("node2", "node3", "similar_to", 0.8)

        gkb.remove_node("node2")

        assert "node2" not in gkb._nodes
        # Оба ребра должны быть удалены
        stats = gkb.stats()
        assert stats["total_edges"] == 0

    def test_remove_edge(self):
        """Тест удаления ребра."""
        gkb = GraphKnowledgeBase()
        gkb.add_node("node1", "Text 1")
        gkb.add_node("node2", "Text 2")
        gkb.add_edge("node1", "node2", "related_to", 1.0)

        assert gkb.stats()["total_edges"] == 1

        gkb.remove_edge("node1", "node2", "related_to")
        assert gkb.stats()["total_edges"] == 0

        # Удаление несуществующего ребра - silent
        gkb.remove_edge("node1", "node2", "related_to")
        assert gkb.stats()["total_edges"] == 0

    def test_clear(self):
        """Тест очистки графа."""
        gkb = GraphKnowledgeBase()
        gkb.add_node("node1", "Text 1")
        gkb.add_node("node2", "Text 2")
        gkb.add_edge("node1", "node2", "related_to", 1.0)

        gkb.clear()

        stats = gkb.stats()
        assert stats["total_nodes"] == 0
        assert stats["total_edges"] == 0

    def test_get_relation_types(self):
        """Тест получения типов отношений."""
        gkb = GraphKnowledgeBase()
        gkb.add_node("node1", "Text 1")
        gkb.add_node("node2", "Text 2")
        gkb.add_node("node3", "Text 3")

        gkb.add_edge("node1", "node2", "related_to", 1.0)
        gkb.add_edge("node1", "node3", "similar_to", 0.8)
        gkb.add_edge("node2", "node3", "related_to", 0.5)

        relation_types = gkb.get_relation_types()
        assert set(relation_types) == {"related_to", "similar_to"}

    def test_add_edge_missing_node_raises(self):
        """Тест ValueError при добавлении ребра с несуществующим узлом."""
        gkb = GraphKnowledgeBase()
        gkb.add_node("node1", "Text 1")

        with pytest.raises(ValueError, match="Target node"):
            gkb.add_edge("node1", "node2", "related_to", 1.0)

        with pytest.raises(ValueError, match="Source node"):
            gkb.add_edge("node2", "node1", "related_to", 1.0)

    def test_search_empty(self):
        """Тест поиска в пустом графе."""
        gkb = GraphKnowledgeBase()
        results = gkb.search("query", k=5)
        assert results == []

    def test_get_related_max_depth(self):
        """Тест BFS с глубиной > 1."""
        gkb = GraphKnowledgeBase()
        gkb.add_node("node1", "Text 1")
        gkb.add_node("node2", "Text 2")
        gkb.add_node("node3", "Text 3")
        gkb.add_node("node4", "Text 4")

        gkb.add_edge("node1", "node2", "related_to", 1.0)
        gkb.add_edge("node2", "node3", "related_to", 1.0)
        gkb.add_edge("node3", "node4", "related_to", 1.0)

        # max_depth=1: только node1->node2
        related_depth1 = gkb.get_related("node1", max_depth=1)
        assert len(related_depth1) == 1
        assert related_depth1[0][1] == "node2"

        # max_depth=2: node1->node2, node2->node3
        related_depth2 = gkb.get_related("node1", max_depth=2)
        assert len(related_depth2) == 2

        # max_depth=3: все три ребра
        related_depth3 = gkb.get_related("node1", max_depth=3)
        assert len(related_depth3) == 3

    def test_add_node_updates_existing(self):
        """Тест обновления существующего узла."""
        gkb = GraphKnowledgeBase()
        gkb.add_node("node1", "Original text", {"key": "value1"})

        gkb.add_node("node1", "Updated text", {"key": "value2"})

        assert gkb._nodes["node1"]["text"] == "Updated text"
        assert gkb._nodes["node1"]["metadata"]["key"] == "value2"

    def test_duplicate_edge_updates_weight(self):
        """Тест обновления веса при повторном добавлении ребра."""
        gkb = GraphKnowledgeBase()
        gkb.add_node("node1", "Text 1")
        gkb.add_node("node2", "Text 2")

        gkb.add_edge("node1", "node2", "related_to", 1.0)
        assert gkb._edges["node1::related_to::node2"]["weight"] == 1.0

        gkb.add_edge("node1", "node2", "related_to", 0.5)
        assert gkb._edges["node1::related_to::node2"]["weight"] == 0.5
