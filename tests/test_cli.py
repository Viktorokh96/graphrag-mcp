"""Тесты для CLI интерфейса RAG."""

from unittest.mock import patch, MagicMock


class TestCLI:
    """Тесты для CLI команд."""

    def test_import(self):
        """CLI модуль импортируется."""
        from src.cli import main
        assert main is not None

    @patch("src.cli.RAGSystem")
    def test_add_document_command(self, mock_rag):
        """Команда add-document."""
        from src.cli import main

        mock_rag_instance = MagicMock()
        mock_rag_instance.add_document.return_value = "doc-123"
        mock_rag.return_value = mock_rag_instance

        exit_code = main(["add-document", "--text", "Hello world"])
        assert exit_code == 0
        mock_rag_instance.add_document.assert_called_once_with("Hello world", None)

    @patch("src.cli.RAGSystem")
    def test_add_document_with_meta(self, mock_rag):
        """Команда add-document с метаданными."""
        from src.cli import main

        mock_rag_instance = MagicMock()
        mock_rag_instance.add_document.return_value = "doc-456"
        mock_rag.return_value = mock_rag_instance

        exit_code = main(["add-document", "--text", "Test", "--meta", '{"source":"test"}'])
        assert exit_code == 0
        mock_rag_instance.add_document.assert_called_once()
        args, _ = mock_rag_instance.add_document.call_args
        assert args[0] == "Test"
        assert args[1] == {"source": "test"}

    @patch("src.cli.RAGSystem")
    def test_add_file_command(self, mock_rag):
        """Команда add-file."""
        from src.cli import main

        mock_rag_instance = MagicMock()
        mock_rag_instance.add_file.return_value = "doc-file-1"
        mock_rag.return_value = mock_rag_instance

        exit_code = main(["add-file", "--path", "/tmp/test.txt"])
        assert exit_code == 0
        mock_rag_instance.add_file.assert_called_once_with("/tmp/test.txt", None)

    @patch("src.cli.RAGSystem")
    def test_search_command(self, mock_rag):
        """Команда search."""
        from src.cli import main

        mock_rag_instance = MagicMock()
        mock_rag_instance.search.return_value = [
            ("doc1", "Python text", 0.95, {}),
            ("doc2", "Another text", 0.80, {}),
        ]
        mock_rag.return_value = mock_rag_instance

        exit_code = main(["search", "--query", "python", "--k", "2"])
        assert exit_code == 0
        mock_rag_instance.search.assert_called_once_with("python", 2, metadata_filter=None)

    @patch("src.cli.RAGSystem")
    def test_bm25_search_command(self, mock_rag):
        """Команда bm25-search."""
        from src.cli import main

        mock_rag_instance = MagicMock()
        mock_rag_instance.bm25_search.return_value = [
            ("doc1", "python programming", 0.8, {}),
        ]
        mock_rag.return_value = mock_rag_instance

        exit_code = main(["bm25-search", "--query", "python"])
        assert exit_code == 0
        mock_rag_instance.bm25_search.assert_called_once_with("python", 5, metadata_filter=None)

    @patch("src.cli.RAGSystem")
    def test_hybrid_search_command(self, mock_rag):
        """Команда hybrid-search."""
        from src.cli import main

        mock_rag_instance = MagicMock()
        mock_rag_instance.search_hybrid.return_value = [
            ("doc1", "python text", 0.9, {}),
        ]
        mock_rag.return_value = mock_rag_instance

        exit_code = main(["hybrid-search", "--query", "python", "--k", "3", "--alpha", "0.7"])
        assert exit_code == 0
        mock_rag_instance.search_hybrid.assert_called_once_with("python", 3, 0.7, metadata_filter=None)

    @patch("src.cli.RAGSystem")
    def test_stats_command(self, mock_rag):
        """Команда stats."""
        from src.cli import main

        mock_rag_instance = MagicMock()
        mock_rag_instance.stats.return_value = {
            "total_documents": 5,
            "store_path": "./rag_data",
            "dimension": 768,
        }
        mock_rag.return_value = mock_rag_instance

        exit_code = main(["stats"])
        assert exit_code == 0
        mock_rag_instance.stats.assert_called_once()

    @patch("src.cli.RAGSystem")
    def test_clear_command(self, mock_rag):
        """Команда clear."""
        from src.cli import main

        mock_rag_instance = MagicMock()
        mock_rag.return_value = mock_rag_instance

        exit_code = main(["clear"])
        assert exit_code == 0
        mock_rag_instance.clear.assert_called_once()

    @patch("src.cli.RAGSystem")
    def test_unknown_command(self, mock_rag):
        """Неизвестная команда выдаёт ошибку."""
        from src.cli import main

        exit_code = main(["unknown-command"])
        assert exit_code == 1

    @patch("src.cli.RAGSystem")
    def test_missing_text_for_add_document(self, mock_rag):
        """add-document без --text выдаёт ошибку."""
        from src.cli import main

        exit_code = main(["add-document"])
        assert exit_code == 1

    @patch("src.cli.RAGSystem")
    def test_cli_uses_custom_store_path(self, mock_rag):
        """Параметр --store передаётся в RAGSystem."""
        from src.cli import main

        mock_rag_instance = MagicMock()
        mock_rag_instance.search.return_value = []
        mock_rag.return_value = mock_rag_instance

        main(["--store", "/custom/path", "search", "--query", "test"])
        mock_rag.assert_called_once()
        _, kwargs = mock_rag.call_args
        assert "store_path" in kwargs
        assert kwargs["store_path"] == "/custom/path"

    @patch("src.cli.RAGSystem")
    def test_cli_uses_custom_api_key(self, mock_rag):
        """Параметр --key передаётся в RAGSystem."""
        from src.cli import main

        mock_rag_instance = MagicMock()
        mock_rag_instance.search.return_value = []
        mock_rag.return_value = mock_rag_instance

        main(["--key", "sk-my-key", "search", "--query", "test"])
        mock_rag.assert_called_once()
        _, kwargs = mock_rag.call_args
        assert "api_key" in kwargs
        assert kwargs["api_key"] == "sk-my-key"
