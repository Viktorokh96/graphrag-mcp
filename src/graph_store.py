"""GraphStore — граф знаний: рёбра в SQLite/Postgres + NetworkX-кеш в памяти.

Узлы графа — это документы (таблица documents, см. document_store.py).
Рёбра живут в таблице graph_edges c FK ON DELETE CASCADE: удаление документа
автоматически удаляет его рёбра. NetworkX MultiDiGraph строится из таблицы
при старте и поддерживается инкрементально — BFS-обходы идут по памяти.
"""

from collections import deque
from typing import Optional

import networkx as nx

from src._meta_filter import matches_metadata_filter
from src.document_store import Database, DocumentStore


class GraphStore:
    """Графовая база: рёбра в SQL, обход через NetworkX."""

    def __init__(self, db: Database, doc_store: DocumentStore):
        self._db = db
        self._docs = doc_store
        self._graph = nx.MultiDiGraph()
        self._build_graph()

    def _build_graph(self) -> None:
        """Построить NetworkX-кеш из таблицы graph_edges."""
        self._graph.clear()
        rows = self._db.execute("SELECT source_id, target_id, relation, weight FROM graph_edges")
        for source, target, relation, weight in rows:
            self._graph.add_edge(source, target, key=relation, weight=weight)

    # -- mutation ---------------------------------------------------------

    def add_edge(self, source_id: str, target_id: str, relation: str, weight: float = 1.0) -> None:
        if self._docs.get(source_id) is None:
            raise ValueError(f"Source node '{source_id}' does not exist")
        if self._docs.get(target_id) is None:
            raise ValueError(f"Target node '{target_id}' does not exist")
        self._db.execute(
            "INSERT INTO graph_edges (source_id, target_id, relation, weight) VALUES (?, ?, ?, ?) "
            "ON CONFLICT (source_id, target_id, relation) DO UPDATE SET weight=excluded.weight",
            (source_id, target_id, relation, weight),
        )
        self._graph.add_edge(source_id, target_id, key=relation, weight=weight)

    def remove_edge(self, source_id: str, target_id: str, relation: str) -> None:
        self._db.execute(
            "DELETE FROM graph_edges WHERE source_id = ? AND target_id = ? AND relation = ?",
            (source_id, target_id, relation),
        )
        if self._graph.has_edge(source_id, target_id, key=relation):
            self._graph.remove_edge(source_id, target_id, key=relation)

    def delete_document(self, doc_id: str) -> None:
        """Убрать узел из кеша (строки graph_edges удаляет FK CASCADE)."""
        if self._graph.has_node(doc_id):
            self._graph.remove_node(doc_id)

    def delete_all(self) -> None:
        self._db.execute("DELETE FROM graph_edges")
        self._graph.clear()

    # -- traversal --------------------------------------------------------

    def get_related(
        self,
        node_id: str,
        max_depth: int = 1,
        direction: str = "both",
        metadata_filter: Optional[dict] = None,
    ) -> list[tuple[str, str, str, float, str]]:
        """BFS от узла. Возвращает [(source, target, relation, weight, 'out'|'in')].

        metadata_filter применяется к соседнему узлу: рёбра к узлам,
        не проходящим фильтр, исключаются.
        """
        if not self._graph.has_node(node_id):
            return []

        meta_cache: dict[str, Optional[dict]] = {}

        def neighbor_passes(neighbor: str) -> bool:
            if not metadata_filter:
                return True
            if neighbor not in meta_cache:
                doc = self._docs.get(neighbor)
                meta_cache[neighbor] = doc["metadata"] if doc else None
            return matches_metadata_filter(meta_cache[neighbor], metadata_filter)

        result = []
        visited_edges: set[tuple[str, str, str]] = set()
        queue = deque([(node_id, 0)])
        seen_depth = {node_id: 0}

        while queue:
            current, depth = queue.popleft()
            if depth >= max_depth:
                continue

            candidates = []
            if direction in ("out", "both"):
                for _, target, relation, data in self._graph.out_edges(current, keys=True, data=True):
                    candidates.append((current, target, relation, data.get("weight", 1.0), "out", target))
            if direction in ("in", "both"):
                for source, _, relation, data in self._graph.in_edges(current, keys=True, data=True):
                    candidates.append((source, current, relation, data.get("weight", 1.0), "in", source))

            for source, target, relation, weight, edge_dir, neighbor in candidates:
                edge_key = (source, target, relation)
                if edge_key in visited_edges:
                    continue
                if not neighbor_passes(neighbor):
                    continue
                visited_edges.add(edge_key)
                result.append((source, target, relation, weight, edge_dir))
                if neighbor not in seen_depth or seen_depth[neighbor] > depth + 1:
                    seen_depth[neighbor] = depth + 1
                    queue.append((neighbor, depth + 1))

        return result

    def get_edges_batch(
        self,
        node_ids: set[str],
        max_depth: int = 1,
        relation_type_filter: Optional[list[str]] = None,
        metadata_filter: Optional[dict] = None,
    ) -> dict[str, dict[str, list[dict]]]:
        """Связи для набора узлов: {node_id: {neighbor_id: [{relation, weight, direction}]}}."""
        result: dict[str, dict[str, list[dict]]] = {}
        for node_id in node_ids:
            if not self._graph.has_node(node_id):
                continue
            links: dict[str, list[dict]] = {}
            for source, target, relation, weight, edge_dir in self.get_related(
                node_id, max_depth=max_depth, metadata_filter=metadata_filter
            ):
                if relation_type_filter is not None and relation not in relation_type_filter:
                    continue
                neighbor = target if edge_dir == "out" else source
                links.setdefault(neighbor, []).append(
                    {"relation": relation, "weight": weight, "direction": edge_dir}
                )
            result[node_id] = links
        return result

    # -- compatibility shims for graph_viz.py --------------------------------

    @property
    def _nodes(self) -> dict:
        """Совместимость с graph_viz: dict {node_id: {text, metadata}}."""
        result: dict = {}
        for nid in self._graph.nodes():
            doc = self._docs.get(nid)
            if doc:
                result[nid] = {"text": doc["text"], "metadata": doc.get("metadata", {})}
        return result

    @property
    def _edges(self) -> dict:
        """Совместимость с graph_viz: dict {idx: {source, target, relation, weight}}."""
        result: dict = {}
        for i, (s, t, r, w) in enumerate(self.get_all_edges()):
            result[i] = {"source": s, "target": t, "relation": r, "weight": w}
        return result

    # -- maintenance --------------------------------------------------------

    def remove_phantom_edges(self, valid_ids: set[str]) -> int:
        """Удалить рёбра, ссылающиеся на несуществующие документы.

        При FK CASCADE фантомы невозможны, но метод сохранён для sync-логики
        (например, после ручного вмешательства в базу).
        """
        rows = self._db.execute("SELECT source_id, target_id, relation FROM graph_edges")
        removed = 0
        for source, target, relation in rows:
            if source not in valid_ids or target not in valid_ids:
                self.remove_edge(source, target, relation)
                removed += 1
        return removed

    def get_all_edges(self) -> list[tuple[str, str, str, float]]:
        return [
            (s, t, r, d.get("weight", 1.0))
            for s, t, r, d in self._graph.edges(keys=True, data=True)
        ]

    def get_relation_types(self) -> list[str]:
        return sorted({r for _, _, r in self._graph.edges(keys=True)})

    def degree(self, node_id: str) -> int:
        if not self._graph.has_node(node_id):
            return 0
        return self._graph.degree(node_id)

    def stats(self) -> dict:
        """Статистика: узлы = документы с рёбрами + все документы стора."""
        return {
            "total_nodes": self._docs.count(),
            "total_edges": self._graph.number_of_edges(),
            "relation_types": self.get_relation_types(),
        }
