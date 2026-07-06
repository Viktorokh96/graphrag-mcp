"""Графовая база знаний для RAG системы."""

import json
from collections import deque
from pathlib import Path
from typing import Optional


class GraphKnowledgeBase:
    """Графовая база знаний для хранения документов и связей между ними с поддержкой персистентности."""

    def __init__(self, store_path: Optional[str] = None):
        """
        Инициализация графовой базы знаний.

        Args:
            store_path: путь к директории для сохранения графа (опционально)
        """
        self._nodes: dict[str, dict] = {}
        self._edges: dict[str, dict] = {}
        self._store_path = store_path
        
        # Загружаем граф с диска если store_path указан
        if store_path:
            self.load()

    def add_node(self, node_id: str, text: str, metadata: Optional[dict] = None) -> None:
        """
        Добавить или обновить узел в графе.

        Args:
            node_id: уникальный идентификатор узла
            text: текст узла
            metadata: метаданные узла
        """
        self._nodes[node_id] = {
            "text": text,
            "metadata": metadata or {},
            "tokens": text.split()
        }
        
        # Сохраняем граф на диск если store_path указан
        if self._store_path:
            self.save()

    def add_edge(self, source_id: str, target_id: str, relation: str, weight: float = 1.0) -> None:
        """
        Добавить или обновить ребро между узлами.

        Args:
            source_id: идентификатор исходного узла
            target_id: идентификатор целевого узла
            relation: тип отношения
            weight: вес ребра

        Raises:
            ValueError: если один из узлов не существует
        """
        if source_id not in self._nodes:
            raise ValueError(f"Source node '{source_id}' does not exist")
        if target_id not in self._nodes:
            raise ValueError(f"Target node '{target_id}' does not exist")

        edge_key = f"{source_id}::{relation}::{target_id}"
        self._edges[edge_key] = {
            "source": source_id,
            "target": target_id,
            "relation": relation,
            "weight": weight
        }
        
        # Сохраняем граф на диск если store_path указан
        if self._store_path:
            self.save()

    def search(self, query: str, k: int = 5, expand_relations: bool = True) -> list[tuple[str, str, float, dict]]:
        """
        Поиск по тексту узлов с BM25-like scoring.

        Args:
            query: поисковый запрос
            k: количество результатов
            expand_relations: расширять ли на связанные узлы

        Returns:
            список кортежей (node_id, text, score, metadata)
        """
        query_tokens = query.split()
        if not query_tokens:
            return []

        # Вычисляем BM25-like score для каждого узла
        scores = []
        for node_id, node_data in self._nodes.items():
            tokens = node_data["tokens"]
            if not tokens:
                scores.append((node_id, 0.0))
                continue

            # TF-IDF like scoring: считаем частоту терминов
            score = 0.0
            for token in query_tokens:
                token_count = tokens.count(token)
                if token_count > 0:
                    # Простой BM25-like score
                    tf = token_count / len(tokens)
                    score += tf

            scores.append((node_id, score))

        # Сортируем по score
        scores.sort(key=lambda x: x[1], reverse=True)

        # Берем top-k
        top_k = scores[:k]
        results = []

        score = 0.0  # инициализируем для LSP; будет перезаписан в цикле ниже
        for node_id, score in top_k:
            node_data = self._nodes[node_id]
            results.append((node_id, node_data["text"], score, node_data["metadata"]))

        # Расширяем на связанные узлы
        if expand_relations and results:
            seen_ids = {node_id for node_id, _, _, _ in results}
            # Собираем явные score для каждого node_id
            node_scores = {node_id: sc for node_id, sc in top_k}
            for node_id, _, _, _ in results:
                related = self.get_related(node_id, max_depth=1)
                for source, target, relation, weight, _dir in related:
                    if target not in seen_ids and target in self._nodes:
                        target_data = self._nodes[target]
                        # Score для связанных узлов - уменьшаем на вес ребра
                        node_score = node_scores.get(node_id, score)
                        related_score = node_score * weight * 0.5
                        results.append((target, target_data["text"], related_score, target_data["metadata"]))
                        seen_ids.add(target)

        # Сортируем финальные результаты по score
        results.sort(key=lambda x: x[2], reverse=True)

        return results

    def get_related(self, node_id: str, max_depth: int = 1, direction: str = "both") -> list[tuple[str, str, str, float, str]]:
        """
        Получить связанные узлы через BFS (двунаправленный).

        Args:
            node_id: идентификатор узла
            max_depth: максимальная глубина обхода
            direction: "out" — только исходящие, "in" — только входящие,
                       "both" — оба направления (по умолчанию)

        Returns:
            список кортежей (source_id, target_id, relation, weight, direction)
        """
        if node_id not in self._nodes:
            return []

        result = []
        visited_edges = set()

        # BFS
        queue = deque([(node_id, 0)])  # (current_node, depth)
        visited_nodes_at_depth = {node_id: 0}

        while queue:
            current, depth = queue.popleft()

            if depth >= max_depth:
                continue

            # Ищем все рёбра от current
            for edge_key, edge_data in self._edges.items():
                edge_dir = None
                neighbor = None

                # Исходящее ребро: current -> target
                if direction in ("out", "both") and edge_data["source"] == current:
                    edge_dir = "out"
                    neighbor = edge_data["target"]

                # Входящее ребро: source -> current (т.е. current = target)
                if direction in ("in", "both") and edge_data["target"] == current:
                    edge_dir = "in"
                    neighbor = edge_data["source"]

                if edge_dir is not None and edge_key not in visited_edges:
                    visited_edges.add(edge_key)
                    result.append((
                        edge_data["source"],
                        edge_data["target"],
                        edge_data["relation"],
                        edge_data["weight"],
                        edge_dir,
                    ))

                    # Добавляем neighbour в очередь для дальнейшего обхода
                    if neighbor not in visited_nodes_at_depth or visited_nodes_at_depth[neighbor] > depth + 1:
                        visited_nodes_at_depth[neighbor] = depth + 1
                        queue.append((neighbor, depth + 1))

        return result

    def get_all_node_ids(self) -> set[str]:
        """Получить множество всех node_id в графе."""
        return set(self._nodes.keys())

    def remove_phantom_edges(self, valid_ids: set[str]) -> int:
        """Удалить рёбра, чьи source или target не входят в valid_ids.

        Args:
            valid_ids: множество валидных идентификаторов узлов

        Returns:
            количество удалённых рёбер
        """
        keys_to_remove = []
        for edge_key, edge_data in self._edges.items():
            if edge_data["source"] not in valid_ids or edge_data["target"] not in valid_ids:
                keys_to_remove.append(edge_key)
        for key in keys_to_remove:
            del self._edges[key]
        if keys_to_remove and self._store_path:
            self.save()
        return len(keys_to_remove)

    def remove_phantom_nodes(self, valid_ids: set[str]) -> int:
        """Удалить узлы, не входящие в valid_ids.

        Args:
            valid_ids: множество валидных идентификаторов

        Returns:
            количество удалённых узлов
        """
        keys_to_remove = [nid for nid in self._nodes if nid not in valid_ids]
        for nid in keys_to_remove:
            self.remove_node(nid)
        return len(keys_to_remove)

    def get_relation_types(self) -> list[str]:
        """
        Получить список уникальных типов отношений.

        Returns:
            список уникальных relation
        """
        relations = set()
        for edge_data in self._edges.values():
            relations.add(edge_data["relation"])
        return list(relations)

    def stats(self) -> dict:
        """
        Получить статистику графа.

        Returns:
            dict с total_nodes, total_edges, relation_types
        """
        return {
            "total_nodes": len(self._nodes),
            "total_edges": len(self._edges),
            "relation_types": self.get_relation_types()
        }

    def remove_node(self, node_id: str) -> None:
        """
        Удалить узел и все связанные с ним рёбра.

        Args:
            node_id: идентификатор узла
        """
        if node_id not in self._nodes:
            return

        # Удаляем узел
        del self._nodes[node_id]

        # Удаляем все рёбра, связанные с этим узлом
        edges_to_remove = []
        for edge_key, edge_data in self._edges.items():
            if edge_data["source"] == node_id or edge_data["target"] == node_id:
                edges_to_remove.append(edge_key)

        for edge_key in edges_to_remove:
            del self._edges[edge_key]
        
        # Сохраняем граф на диск если store_path указан
        if self._store_path:
            self.save()

    def remove_edge(self, source_id: str, target_id: str, relation: str) -> None:
        """
        Удалить конкретное ребро.

        Args:
            source_id: идентификатор исходного узла
            target_id: идентификатор целевого узла
            relation: тип отношения
        """
        edge_key = f"{source_id}::{relation}::{target_id}"
        if edge_key in self._edges:
            del self._edges[edge_key]
        
        # Сохраняем граф на диск если store_path указан
        if self._store_path:
            self.save()

    def save(self) -> None:
        """
        Сохранить граф на диск в формате JSON.

        Сохраняет:
        - nodes: узлы графа
        - edges: рёбра графа
        """
        if not self._store_path:
            return
        
        # Создаем директорию если не существует
        store_path = Path(self._store_path)
        store_path.mkdir(parents=True, exist_ok=True)
        
        # Путь к файлу графа
        graph_file = store_path / "graph_index.json"
        
        # Данные для сохранения
        data = {
            "nodes": self._nodes,
            "edges": self._edges
        }
        
        # Сохраняем в JSON
        with open(graph_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def load(self) -> bool:
        """
        Загрузить граф с диска.

        Returns:
            True если успешно загружено, False если файла нет
        """
        if not self._store_path:
            return False
        
        # Путь к файлу графа
        graph_file = Path(self._store_path) / "graph_index.json"
        
        if not graph_file.exists():
            return False
        
        # Загружаем из JSON
        with open(graph_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        self._nodes = data.get("nodes", {})
        self._edges = data.get("edges", {})
        
        return True

    def clear(self) -> None:
        """
        Очистить все узлы и рёбра.
        
        Если store_path указан — удаляет файл с диска.
        """
        self._nodes.clear()
        self._edges.clear()
        
        # Удаляем файл с диска если store_path указан
        if self._store_path:
            graph_file = Path(self._store_path) / "graph_index.json"
            if graph_file.exists():
                graph_file.unlink()
