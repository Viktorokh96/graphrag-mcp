"""Тесты визуализации графа (Фаза 9) на новом GraphStore + DocumentStore.

Проверяют, что рендереры работают через адаптер _VizGraph (старый интерфейс
graph._nodes/_edges поверх нового стора).
"""

import json

import pytest

from src.graph_viz import _VizGraph, render_graph_viz


LONG = "Document about topic {} with alpha beta gamma content for indexing tests."


@pytest.fixture
def graph_rag(rag):
    ids = [rag.add_document(LONG.format(i), {"grp": i % 2}) for i in range(5)]
    rag.add_relation(ids[0], ids[1], "related_to", 0.9)
    rag.add_relation(ids[1], ids[2], "depends_on", 0.5)
    rag.add_relation(ids[0], ids[3], "sibling")
    return rag, ids


class TestVizGraphAdapter:
    def test_snapshot_nodes_and_edges(self, graph_rag):
        rag, ids = graph_rag
        vg = _VizGraph(rag.graph_kb)
        assert set(vg._nodes.keys()) == set(ids)
        assert len(vg._edges) == 3
        # каждый узел несёт text и metadata
        assert all("text" in n and "metadata" in n for n in vg._nodes.values())

    def test_delegates_methods(self, graph_rag):
        rag, ids = graph_rag
        vg = _VizGraph(rag.graph_kb)
        # get_related делегируется обёрнутому стору
        rel = vg.get_related(ids[0])
        assert any(r[1] == ids[1] for r in rel)


class TestRenderFormats:
    def test_html(self, graph_rag, tmp_path):
        rag, _ = graph_rag
        out = tmp_path / "g.html"
        render_graph_viz(rag.graph_kb, output_path=str(out), output_format="html")
        content = out.read_text(encoding="utf-8")
        assert out.stat().st_size > 0
        assert "vis" in content.lower()

    def test_dot(self, graph_rag, tmp_path):
        rag, _ = graph_rag
        out = tmp_path / "g.dot"
        render_graph_viz(rag.graph_kb, output_path=str(out), output_format="dot")
        content = out.read_text(encoding="utf-8")
        assert content.startswith("digraph")
        assert "->" in content

    def test_json(self, graph_rag, tmp_path):
        rag, _ = graph_rag
        out = tmp_path / "g.json"
        render_graph_viz(rag.graph_kb, output_path=str(out), output_format="json")
        data = json.loads(out.read_text(encoding="utf-8"))
        assert len(data["nodes"]) == 5
        assert len(data["edges"]) == 3

    def test_ascii(self, graph_rag, tmp_path):
        rag, _ = graph_rag
        out = tmp_path / "g.txt"
        render_graph_viz(rag.graph_kb, output_path=str(out), output_format="ascii")
        content = out.read_text(encoding="utf-8")
        assert "nodes" in content and "edges" in content

    def test_focus_subgraph(self, graph_rag, tmp_path):
        rag, ids = graph_rag
        out = tmp_path / "focus.json"
        render_graph_viz(
            rag.graph_kb, output_path=str(out), output_format="json",
            focus_node=ids[0], max_depth=1,
        )
        data = json.loads(out.read_text(encoding="utf-8"))
        node_ids = {n["id"] for n in data["nodes"]}
        # фокус на ids[0]: он сам и прямые соседи ids[1], ids[3]
        assert ids[0] in node_ids
        assert ids[1] in node_ids and ids[3] in node_ids
        assert ids[4] not in node_ids

    def test_relation_type_filter(self, graph_rag, tmp_path):
        rag, _ = graph_rag
        out = tmp_path / "filtered.json"
        render_graph_viz(
            rag.graph_kb, output_path=str(out), output_format="json",
            relation_type=["related_to"],
        )
        data = json.loads(out.read_text(encoding="utf-8"))
        assert {e["relation"] for e in data["edges"]} == {"related_to"}

    def test_max_nodes_limit(self, graph_rag, tmp_path):
        rag, _ = graph_rag
        out = tmp_path / "limited.json"
        render_graph_viz(
            rag.graph_kb, output_path=str(out), output_format="json", max_nodes=3,
        )
        data = json.loads(out.read_text(encoding="utf-8"))
        assert len(data["nodes"]) <= 3

    def test_unknown_format_raises(self, graph_rag, tmp_path):
        rag, _ = graph_rag
        with pytest.raises(ValueError):
            render_graph_viz(rag.graph_kb, output_path=str(tmp_path / "x"), output_format="pdf")
