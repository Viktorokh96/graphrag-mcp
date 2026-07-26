"""Тесты безопасности HTTP API: аутентификация, CORS, валидация входа."""

import pytest
from fastapi.testclient import TestClient

from src import http_api
from src.cli import _is_loopback
from src.config import RAGConfig
from src.graph_viz import _json_for_script
from src.rag import RAGSystem
from tests.conftest import HashEmbeddingGenerator


@pytest.fixture
def client(tmp_path, monkeypatch):
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


class TestAuth:
    def test_no_token_configured_allows_requests(self, client, monkeypatch):
        monkeypatch.delenv("API_TOKEN", raising=False)
        assert client.get("/stats").status_code == 200

    def test_token_required_when_configured(self, client, monkeypatch):
        monkeypatch.setenv("API_TOKEN", "s3cret")
        assert client.get("/stats").status_code == 401
        assert client.post("/clear").status_code == 401

    def test_valid_token_accepted(self, client, monkeypatch):
        monkeypatch.setenv("API_TOKEN", "s3cret")
        assert client.get("/stats", headers={"Authorization": "Bearer s3cret"}).status_code == 200
        assert client.get("/stats", headers={"X-API-Key": "s3cret"}).status_code == 200

    def test_wrong_token_rejected(self, client, monkeypatch):
        monkeypatch.setenv("API_TOKEN", "s3cret")
        assert client.get("/stats", headers={"Authorization": "Bearer nope"}).status_code == 401

    def test_health_is_public(self, client, monkeypatch):
        monkeypatch.setenv("API_TOKEN", "s3cret")
        assert client.get("/health").status_code == 200


class TestCORS:
    def test_no_wildcard_origin_by_default(self, client):
        r = client.get("/health", headers={"Origin": "https://evil.example"})
        assert "access-control-allow-origin" not in {k.lower() for k in r.headers}

    def test_origins_configurable(self, monkeypatch):
        monkeypatch.setenv("CORS_ALLOW_ORIGINS", "http://a.example, http://b.example")
        assert RAGConfig().resolve_cors_origins() == ["http://a.example", "http://b.example"]


class TestInputValidation:
    def test_search_rejects_out_of_range_k(self, client):
        assert client.post("/search", json={"query": "x", "k": 10_000}).status_code == 422

    def test_search_rejects_unknown_mode(self, client):
        assert client.post("/search", json={"query": "x", "mode": "drop"}).status_code == 422

    def test_search_rejects_out_of_range_alpha(self, client):
        assert client.post("/search", json={"query": "x", "alpha": 5.0}).status_code == 422

    def test_relations_depth_bounded(self, client):
        assert client.get("/documents", params={"relations_load_depth": 99}).status_code == 422


class TestSecretsAndEscaping:
    def test_env_preview_masks_api_key(self):
        cfg = RAGConfig(embedding_api_key="sk-abcdef123456")
        assert "sk-abcdef123456" not in cfg.to_env_preview()
        assert "***3456" in cfg.to_env_preview()

    def test_script_json_escapes_tags(self):
        assert "</script>" not in _json_for_script({"text": "</script><img onerror=alert(1)>"})


class TestLoopbackDetection:
    @pytest.mark.parametrize("host,expected", [
        ("127.0.0.1", True), ("localhost", True), ("::1", True),
        ("0.0.0.0", False), ("10.0.0.5", False),
    ])
    def test_is_loopback(self, host, expected):
        assert _is_loopback(host) is expected
