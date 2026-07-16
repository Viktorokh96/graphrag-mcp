"""Тесты для QueryExpander."""

from unittest.mock import patch, MagicMock


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
