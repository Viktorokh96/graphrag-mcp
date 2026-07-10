"""Тесты HTTP REST API (Фаза 3) через FastAPI TestClient.

RAGSystem инициализируется с мок-эмбеддером и tmp-хранилищем (реальная модель
BGE-M3 не грузится). Lifespan приложения переопределяется, чтобы использовать
этот инстанс.
"""

import pytest
from fastapi.testclient import TestClient

from src import http_api
from src.config import RAGConfig
from src.rag import RAGSystem
from tests.conftest import HashEmbeddingGenerator


@pytest.fixture
def client(tmp_path, monkeypatch):
    """TestClient с RAGSystem на мок-эмбеддере и tmp-хранилище."""
    cfg = RAGConfig()
    cfg.store_path = str(tmp_path / "store")
    rag = RAGSystem(config=cfg, embedding_generator=HashEmbeddingGenerator())

    def fake_init(self, config=None):
        self.instance = rag
        return rag

    monkeypatch.setattr(http_api.RAGHolder, "init", fake_init)
    with TestClient(http_api.app) as c:
        yield c
    rag.close()


LONG = "This is a sufficiently long document about Python programming and testing. " * 2


class TestHealthAndStats:
    def test_health(self, client):
        r = client.get("/health")
        assert r.status_code == 200
        assert r.json()["status"] == "ok"

    def test_stats_empty(self, client):
        r = client.get("/stats")
        assert r.status_code == 200
        body = r.json()
        assert body["total_documents"] == 0
        assert body["dimension"] == 64

    def test_graph_stats_empty(self, client):
        r = client.get("/graph/stats")
        assert r.status_code == 200
        body = r.json()
        assert body["total_nodes"] == 0
        assert body["total_edges"] == 0


class TestDocuments:
    def test_add_and_get(self, client):
        r = client.post("/documents", json={"text": LONG, "meta": {"source": "test"}})
        assert r.status_code == 200
        doc_id = r.json()["doc_id"]
        assert r.json()["duplicate"] is False

        r2 = client.get(f"/documents/{doc_id}")
        assert r2.status_code == 200
        assert r2.json()["metadata"]["source"] == "test"
        assert r2.json()["total_chars"] == len(LONG)

    def test_add_duplicate(self, client):
        client.post("/documents", json={"text": LONG})
        r = client.post("/documents", json={"text": LONG})
        assert r.json()["duplicate"] is True

    def test_get_missing_404(self, client):
        r = client.get("/documents/nonexistent")
        assert r.status_code == 404

    def test_meta_json_string(self, client):
        r = client.post("/documents", json={"text": LONG, "meta": '{"k": "v"}'})
        doc_id = r.json()["doc_id"]
        r2 = client.get(f"/documents/{doc_id}")
        assert r2.json()["metadata"]["k"] == "v"

    def test_list_pagination(self, client):
        for i in range(5):
            client.post("/documents", json={"text": f"{LONG} variant {i}"})
        r = client.get("/documents", params={"limit": 2, "offset": 0})
        body = r.json()
        assert body["total"] == 5
        assert len(body["documents"]) == 2

    def test_list_metadata_filter(self, client):
        client.post("/documents", json={"text": LONG + " one", "meta": {"kind": "a"}})
        client.post("/documents", json={"text": LONG + " two", "meta": {"kind": "b"}})
        r = client.get("/documents", params={"metadata_filter": '{"kind": "a"}'})
        body = r.json()
        assert body["total"] == 1

    def test_delete(self, client):
        doc_id = client.post("/documents", json={"text": LONG}).json()["doc_id"]
        r = client.delete(f"/documents/{doc_id}")
        assert r.json()["deleted"] is True
        r2 = client.delete(f"/documents/{doc_id}")
        assert r2.json()["deleted"] is False

    def test_get_pagination_offset_limit(self, client):
        doc_id = client.post("/documents", json={"text": LONG}).json()["doc_id"]
        r = client.get(f"/documents/{doc_id}", params={"offset": 5, "limit": 4})
        assert r.json()["text"] == LONG[5:9]


class TestSearch:
    def _seed(self, client):
        client.post("/documents", json={"text": "Python programming language for data science and ML.", "meta": {"t": "py"}})
        client.post("/documents", json={"text": "FastAPI is a web framework for building HTTP APIs in Python.", "meta": {"t": "web"}})
        client.post("/documents", json={"text": "Кулинарный рецепт борща со свёклой и капустой на бульоне.", "meta": {"t": "ru"}})

    def test_bm25_search(self, client):
        self._seed(client)
        r = client.post("/search", json={"query": "Python programming", "mode": "bm25", "k": 3})
        assert r.status_code == 200
        assert len(r.json()["results"]) >= 1

    def test_hybrid_search(self, client):
        self._seed(client)
        r = client.post("/search", json={"query": "web framework", "mode": "hybrid", "k": 3})
        assert r.status_code == 200
        assert "results" in r.json()

    def test_semantic_search(self, client):
        self._seed(client)
        r = client.post("/search", json={"query": "Python", "mode": "semantic", "k": 2})
        assert r.status_code == 200

    def test_search_metadata_filter(self, client):
        self._seed(client)
        r = client.post("/search", json={"query": "свёкла капуста", "mode": "bm25", "metadata_filter": {"t": "ru"}})
        results = r.json()["results"]
        assert all(d["metadata"].get("t") == "ru" for d in results)

    def test_search_max_chars(self, client):
        self._seed(client)
        r = client.post("/search", json={"query": "Python", "mode": "bm25", "max_chars": 10})
        for d in r.json()["results"]:
            assert len(d["text"]) <= 10


class TestRelations:
    def test_add_and_get_relation(self, client):
        a = client.post("/documents", json={"text": LONG + " AAA"}).json()["doc_id"]
        b = client.post("/documents", json={"text": LONG + " BBB"}).json()["doc_id"]
        r = client.post("/relations", json={"source_id": a, "target_id": b, "relation": "related_to", "weight": 0.9})
        assert r.json()["status"] == "ok"
        r2 = client.get(f"/relations/{a}")
        rels = r2.json()["relations"]
        assert len(rels) == 1
        assert rels[0]["target"] == b
        assert rels[0]["relation"] == "related_to"

    def test_links_in_search(self, client):
        a = client.post("/documents", json={"text": LONG + " alpha node"}).json()["doc_id"]
        b = client.post("/documents", json={"text": LONG + " beta node"}).json()["doc_id"]
        client.post("/relations", json={"source_id": a, "target_id": b, "relation": "x"})
        r = client.get(f"/documents/{a}", params={"relations_load_depth": 1})
        assert b in r.json()["links"]


class TestClearAndStructured:
    def test_clear(self, client):
        client.post("/documents", json={"text": LONG})
        assert client.get("/stats").json()["total_documents"] == 1
        r = client.post("/clear")
        assert r.json()["status"] == "ok"
        assert client.get("/stats").json()["total_documents"] == 0

    def test_structured_repomix(self, client):
        import json as _json
        content = _json.dumps({
            "repository": "demo-project-with-a-reasonably-long-name",
            "structure": ["src/module_alpha.py", "src/module_beta.py", "tests/test_alpha.py"],
            "files": {
                "src/module_alpha.py": {"content": "def a():\n    return 1  # " + "x" * 60, "language": "python"},
                "src/module_beta.py": {"content": "def b():\n    return 2  # " + "y" * 60, "language": "python"},
            },
        })
        r = client.post("/structured", json={"content": content})
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "ok"
        assert body["files_count"] == 2
        assert body["structure_doc_id"] is not None

    def test_structured_tiny_repo_no_crash(self, client):
        # Дерево короче 50 символов не должно ронять эндпоинт (structure_doc_id=None)
        import json as _json
        content = _json.dumps({
            "repository": "x",
            "structure": ["a.py"],
            "files": {"a.py": {"content": "def a():\n    return 1  # " + "z" * 60, "language": "python"}},
        })
        r = client.post("/structured", json={"content": content})
        assert r.status_code == 200
        assert r.json()["structure_doc_id"] is None
        assert r.json()["files_count"] == 1

    def test_root_redirects_to_ui(self, client):
        r = client.get("/", follow_redirects=False)
        assert r.status_code in (307, 308)
        assert r.headers["location"] == "/ui"
