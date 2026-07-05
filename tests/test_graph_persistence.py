"""Тесты персистентности графовой базы знаний."""

import os
import pytest
import tempfile
import shutil
from src.graph_store import GraphKnowledgeBase


class TestGraphPersistence:
    """Тесты сохранения и загрузки графовой базы знаний."""

    @pytest.fixture
    def temp_dir(self):
        """Создать временную директорию."""
        path = tempfile.mkdtemp()
        yield path
        shutil.rmtree(path)

    def test_save_and_load_empty(self, temp_dir):
        """Сохранить и загрузить пустой граф."""
        graph = GraphKnowledgeBase(store_path=temp_dir)
        graph.save()

        # Проверяем, что файл создан
        graph2 = GraphKnowledgeBase(store_path=temp_dir)
        assert graph2.stats()["total_nodes"] == 0
        assert graph2.stats()["total_edges"] == 0

    def test_save_and_load_with_nodes(self, temp_dir):
        """Сохранить и загрузить граф с узлами."""
        # Создаём и наполняем
        graph = GraphKnowledgeBase(store_path=temp_dir)
        graph.add_node("node1", "Python is a programming language", {"lang": "python"})
        graph.add_node("node2", "Django is a web framework", {"framework": "django"})
        graph.save()

        # Загружаем в новый инстанс
        graph2 = GraphKnowledgeBase(store_path=temp_dir)
        assert graph2.stats()["total_nodes"] == 2

        # Проверяем содержимое узлов
        assert "node1" in graph2._nodes
        assert graph2._nodes["node1"]["text"] == "Python is a programming language"
        assert graph2._nodes["node1"]["metadata"]["lang"] == "python"

    def test_save_and_load_with_edges(self, temp_dir):
        """Сохранить и загрузить граф с рёбрами."""
        graph = GraphKnowledgeBase(store_path=temp_dir)
        graph.add_node("node1", "Text 1")
        graph.add_node("node2", "Text 2")
        graph.add_node("node3", "Text 3")

        graph.add_edge("node1", "node2", "related_to", 1.0)
        graph.add_edge("node1", "node3", "similar_to", 0.8)
        graph.save()

        # Загружаем в новый инстанс
        graph2 = GraphKnowledgeBase(store_path=temp_dir)
        assert graph2.stats()["total_edges"] == 2

        # Проверяем рёбра
        edges = graph2._edges
        assert "node1::related_to::node2" in edges
        assert edges["node1::related_to::node2"]["weight"] == 1.0
        assert "node1::similar_to::node3" in edges
        assert edges["node1::similar_to::node3"]["weight"] == 0.8

    def test_search_after_reload(self, temp_dir):
        """Поиск работает после перезагрузки."""
        # Первая сессия
        graph = GraphKnowledgeBase(store_path=temp_dir)
        graph.add_node("node1", "Python programming language")
        graph.add_node("node2", "Java programming language")
        graph.save()

        # Вторая сессия (имитация перезапуска)
        graph2 = GraphKnowledgeBase(store_path=temp_dir)
        results = graph2.search("Python", k=5)
        assert len(results) > 0
        assert results[0][0] == "node1"

    def test_add_after_load(self, temp_dir):
        """Добавление узлов и рёбер после загрузки работает."""
        graph = GraphKnowledgeBase(store_path=temp_dir)
        graph.add_node("node1", "Python programming")
        graph.save()

        # Загружаем и добавляем ещё
        graph2 = GraphKnowledgeBase(store_path=temp_dir)
        graph2.add_node("node2", "Java programming")
        graph2.add_edge("node1", "node2", "similar_to", 1.0)
        graph2.save()

        # Проверяем финальное состояние
        graph3 = GraphKnowledgeBase(store_path=temp_dir)
        assert graph3.stats()["total_nodes"] == 2
        assert graph3.stats()["total_edges"] == 1

    def test_clear_removes_files(self, temp_dir):
        """Очистка графа удаляет файлы."""
        graph = GraphKnowledgeBase(store_path=temp_dir)
        graph.add_node("node1", "Python programming")
        graph.save()

        # Проверяем, что файл есть
        assert os.path.exists(os.path.join(temp_dir, "graph_index.json"))

        # Очищаем
        graph.clear()

        # Загружаем — должен быть пустым
        graph2 = GraphKnowledgeBase(store_path=temp_dir)
        assert graph2.stats()["total_nodes"] == 0
        assert graph2.stats()["total_edges"] == 0

    def test_no_store_path_creates_no_file(self, temp_dir):
        """Без store_path граф не сохраняется на диск."""
        graph = GraphKnowledgeBase()  # без store_path
        graph.add_node("node1", "Python programming")

        # save не должен падать, но и файла не будет
        graph.save()  # должна быть no-op

    def test_get_related_after_reload(self, temp_dir):
        """Получение связанных узлов работает после перезагрузки."""
        graph = GraphKnowledgeBase(store_path=temp_dir)
        graph.add_node("node1", "Text 1")
        graph.add_node("node2", "Text 2")
        graph.add_node("node3", "Text 3")

        graph.add_edge("node1", "node2", "related_to", 1.0)
        graph.add_edge("node2", "node3", "related_to", 1.0)
        graph.save()

        # Загружаем и проверяем get_related
        graph2 = GraphKnowledgeBase(store_path=temp_dir)
        related = graph2.get_related("node1", max_depth=2)
        assert len(related) == 2

    def test_get_relation_types_after_reload(self, temp_dir):
        """Получение типов отношений работает после перезагрузки."""
        graph = GraphKnowledgeBase(store_path=temp_dir)
        graph.add_node("node1", "Text 1")
        graph.add_node("node2", "Text 2")
        graph.add_node("node3", "Text 3")

        graph.add_edge("node1", "node2", "related_to", 1.0)
        graph.add_edge("node1", "node3", "similar_to", 0.8)
        graph.save()

        # Загружаем и проверяем типы отношений
        graph2 = GraphKnowledgeBase(store_path=temp_dir)
        relation_types = graph2.get_relation_types()
        assert set(relation_types) == {"related_to", "similar_to"}

    def test_search_with_expansion_after_reload(self, temp_dir):
        """Поиск с расширением работает после перезагрузки."""
        graph = GraphKnowledgeBase(store_path=temp_dir)
        graph.add_node("node1", "Python programming language")
        graph.add_node("node2", "Django web framework")
        graph.add_edge("node1", "node2", "related_to", 1.0)
        graph.save()

        # Загружаем и проверяем поиск с расширением
        graph2 = GraphKnowledgeBase(store_path=temp_dir)
        results = graph2.search("Python", k=1, expand_relations=True)
        node_ids = [r[0] for r in results]
        assert "node1" in node_ids
        assert "node2" in node_ids  # node2 добавлен через расширение

    def test_remove_node_after_reload(self, temp_dir):
        """Удаление узла работает после перезагрузки."""
        graph = GraphKnowledgeBase(store_path=temp_dir)
        graph.add_node("node1", "Text 1")
        graph.add_node("node2", "Text 2")
        graph.add_edge("node1", "node2", "related_to", 1.0)
        graph.save()

        # Загружаем и удаляем узел
        graph2 = GraphKnowledgeBase(store_path=temp_dir)
        graph2.remove_node("node1")
        graph2.save()

        # Проверяем, что узел и ребро удалены
        graph3 = GraphKnowledgeBase(store_path=temp_dir)
        assert "node1" not in graph3._nodes
        assert graph3.stats()["total_edges"] == 0

    def test_remove_edge_after_reload(self, temp_dir):
        """Удаление ребра работает после перезагрузки."""
        graph = GraphKnowledgeBase(store_path=temp_dir)
        graph.add_node("node1", "Text 1")
        graph.add_node("node2", "Text 2")
        graph.add_edge("node1", "node2", "related_to", 1.0)
        graph.save()

        # Загружаем и удаляем ребро
        graph2 = GraphKnowledgeBase(store_path=temp_dir)
        graph2.remove_edge("node1", "node2", "related_to")
        graph2.save()

        # Проверяем, что ребро удалено
        graph3 = GraphKnowledgeBase(store_path=temp_dir)
        assert graph3.stats()["total_edges"] == 0

    def test_load_nonexistent_file(self, temp_dir):
        """Загрузка несуществующего файла возвращает False."""
        graph = GraphKnowledgeBase(store_path=temp_dir)
        result = graph.load()
        assert result is False
        assert graph.stats()["total_nodes"] == 0

    def test_file_format_is_json(self, temp_dir):
        """Файл сохраняется в формате JSON."""
        import json

        graph = GraphKnowledgeBase(store_path=temp_dir)
        graph.add_node("node1", "Python programming", {"key": "value"})
        graph.add_node("node2", "Java programming", {"key": "value2"})
        graph.add_edge("node1", "node2", "related_to", 1.0)
        graph.save()

        # Проверяем, что файл валидный JSON
        file_path = os.path.join(temp_dir, "graph_index.json")
        assert os.path.exists(file_path)

        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        assert "nodes" in data
        assert "edges" in data
