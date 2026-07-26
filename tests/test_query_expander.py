"""Тесты для QueryExpander."""

from unittest.mock import patch, MagicMock

import pytest


class TestQueryExpander:
    def test_import(self):
        from src.query_expander import QueryExpander
        assert QueryExpander is not None

    def test_expand_returns_query_when_count_zero(self):
        from src.query_expander import QueryExpander

        e = QueryExpander(provider="ollama", model="mock", count=0)
        result = e.expand("hello world")
        assert result == ["hello world"]

    @patch("httpx.Client")
    def test_expand_parses_llm_response(self, mock_client_class):
        from src.query_expander import QueryExpander

        mock_client = MagicMock()
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "response": "variant one\nvariant two\nvariant three"
        }
        mock_client.post.return_value = mock_resp
        mock_client_class.return_value = mock_client

        e = QueryExpander(provider="ollama", model="mock", count=3)
        result = e.expand("original query")
        assert result == ["original query", "variant one", "variant two", "variant three"]
        mock_client.post.assert_called_once()
        call_kwargs = mock_client.post.call_args[1]
        assert call_kwargs["json"]["model"] == "mock"

    @patch("httpx.Client")
    def test_expand_with_extra_lines(self, mock_client_class):
        from src.query_expander import QueryExpander

        mock_client = MagicMock()
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"response": "line1\n\nline2\n\n\nline3"}
        mock_client.post.return_value = mock_resp
        mock_client_class.return_value = mock_client

        e = QueryExpander(provider="ollama", model="mock", count=3)
        result = e.expand("q")
        assert result == ["q", "line1", "line2", "line3"]

    @patch("httpx.Client")
    def test_expand_with_short_response(self, mock_client_class):
        from src.query_expander import QueryExpander

        mock_client = MagicMock()
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"response": "only one"}
        mock_client.post.return_value = mock_resp
        mock_client_class.return_value = mock_client

        e = QueryExpander(provider="ollama", model="mock", count=5)
        result = e.expand("q")
        assert result == ["q", "only one"]
        assert len(result) == 2

    @patch("httpx.Client")
    def test_expand_empty_response(self, mock_client_class):
        from src.query_expander import QueryExpander

        mock_client = MagicMock()
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"response": ""}
        mock_client.post.return_value = mock_resp
        mock_client_class.return_value = mock_client

        e = QueryExpander(provider="ollama", model="mock", count=3)
        result = e.expand("q")
        assert result == ["q"]


class TestQueryExpanderProviders:
    """Провайдеры openai-compatible / anthropic и валидация конструктора."""

    def _mock_client_class(self, mock_client_class, payload):
        client = MagicMock()
        response = MagicMock()
        response.json.return_value = payload
        client.post.return_value = response
        mock_client_class.return_value = client
        return client

    @patch("httpx.Client")
    def test_openai_request_shape_and_parsing(self, mock_client_class):
        from src.query_expander import QueryExpander

        client = self._mock_client_class(
            mock_client_class,
            {"choices": [{"message": {"content": "alt one\nalt two"}}]},
        )
        e = QueryExpander(provider="openai-compatible", model="gpt", base_url="http://host/v1/", api_key="sk-1")
        assert e.expand("q", count=2) == ["q", "alt one", "alt two"]

        call = client.post.call_args
        assert call[0][0] == "http://host/v1/chat/completions"
        assert call[1]["headers"]["Authorization"] == "Bearer sk-1"
        assert call[1]["json"]["model"] == "gpt"
        assert call[1]["json"]["messages"][0]["content"].startswith("Generate 2 alternative phrasings")

    @patch("httpx.Client")
    def test_openai_without_api_key_has_no_auth_header(self, mock_client_class):
        from src.query_expander import QueryExpander

        client = self._mock_client_class(
            mock_client_class, {"choices": [{"message": {"content": "alt"}}]}
        )
        QueryExpander(provider="openai-compatible", base_url="http://host/v1").expand("q")
        assert "Authorization" not in client.post.call_args[1]["headers"]

    @patch("httpx.Client")
    def test_openai_truncates_to_requested_count(self, mock_client_class):
        from src.query_expander import QueryExpander

        self._mock_client_class(
            mock_client_class, {"choices": [{"message": {"content": "a\nb\nc\nd"}}]}
        )
        e = QueryExpander(provider="openai-compatible", count=2)
        assert e.expand("q") == ["q", "a", "b"]

    @patch("httpx.Client")
    def test_anthropic_request_shape_and_parsing(self, mock_client_class):
        from src.query_expander import QueryExpander

        client = self._mock_client_class(
            mock_client_class, {"content": [{"text": "alt one\nalt two"}]}
        )
        e = QueryExpander(provider="anthropic", model="claude", base_url="https://api.anthropic.com/v1", api_key="k")
        assert e.expand("q") == ["q", "alt one", "alt two"]

        call = client.post.call_args
        assert call[0][0] == "https://api.anthropic.com/v1/messages"
        assert call[1]["headers"]["x-api-key"] == "k"
        assert call[1]["headers"]["anthropic-version"] == "2023-06-01"
        assert call[1]["json"]["model"] == "claude"

    @patch("httpx.Client")
    def test_http_error_propagates(self, mock_client_class):
        from src.query_expander import QueryExpander

        client = self._mock_client_class(mock_client_class, {})
        client.post.return_value.raise_for_status.side_effect = RuntimeError("503")
        with pytest.raises(RuntimeError, match="503"):
            QueryExpander(provider="ollama").expand("q")

    def test_sentence_transformer_provider_not_implemented(self):
        from src.query_expander import QueryExpander

        with pytest.raises(NotImplementedError):
            QueryExpander(provider="sentence_transformer")

    def test_unknown_provider_raises(self):
        from src.query_expander import QueryExpander

        with pytest.raises(ValueError, match="Unknown expansion provider"):
            QueryExpander(provider="magic")

    @patch("httpx.Client")
    def test_expand_dispatches_to_unknown_provider(self, mock_client_class):
        from src.query_expander import QueryExpander

        e = QueryExpander(provider="ollama")
        e.provider = "changed-later"
        with pytest.raises(NotImplementedError, match="changed-later"):
            e.expand("q")

    @patch("httpx.Client")
    def test_negative_count_returns_original_query(self, mock_client_class):
        from src.query_expander import QueryExpander

        e = QueryExpander(provider="ollama", count=3)
        assert e.expand("q", count=-1) == ["q"]
        mock_client_class.return_value.post.assert_not_called()

    @patch("httpx.Client")
    def test_close_closes_client(self, mock_client_class):
        from src.query_expander import QueryExpander

        e = QueryExpander(provider="ollama")
        e.close()
        mock_client_class.return_value.close.assert_called_once()

    @patch("httpx.Client")
    def test_base_url_trailing_slash_stripped(self, mock_client_class):
        from src.query_expander import QueryExpander

        client = self._mock_client_class(mock_client_class, {"response": ""})
        QueryExpander(provider="ollama", base_url="http://host:11434///").expand("q")
        assert client.post.call_args[0][0] == "http://host:11434/api/generate"
