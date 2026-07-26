"""Repomix-style индексация структурированного кода.

Принимает JSON в формате repomix:
{
  "repository": "project-name",
  "structure": ["src/main.py", "src/utils.py", ...],
  "files": {
    "src/main.py": {"content": "...", "language": "python", "size": 1024},
    ...
  }
}

Пайплайн:
1. Дерево папок → документ с типом `structure`
2. Каждый файл → один или несколько чанков с типом `file`
3. Авто-связи: sibling для файлов одной папки
"""

import json
import logging
import os

from src.rag import RAGSystem

logger = logging.getLogger(__name__)


def _detect_language(filepath: str) -> str:
    ext = os.path.splitext(filepath)[1].lower()
    mapping = {
        ".py": "python",
        ".js": "javascript",
        ".ts": "typescript",
        ".tsx": "typescriptreact",
        ".jsx": "javascriptreact",
        ".rs": "rust",
        ".go": "go",
        ".java": "java",
        ".kt": "kotlin",
        ".swift": "swift",
        ".c": "c",
        ".cpp": "cpp",
        ".h": "c",
        ".hpp": "cpp",
        ".rb": "ruby",
        ".php": "php",
        ".sh": "bash",
        ".yaml": "yaml",
        ".yml": "yaml",
        ".json": "json",
        ".toml": "toml",
        ".md": "markdown",
        ".sql": "sql",
        ".dockerfile": "dockerfile",
        ".html": "html",
        ".css": "css",
        ".scss": "scss",
    }
    return mapping.get(ext, "text")


def _get_dir(filepath: str) -> str:
    return os.path.dirname(filepath) or "/"


def _parent_dirs(filepath: str) -> list[str]:
    parts = filepath.split("/")
    return ["/".join(parts[:i]) or "/" for i in range(1, len(parts))]


class StructuredIndexer:
    """Индексация repomix-формата."""

    def __init__(self, rag: RAGSystem, extract_graph: bool = False):
        self.rag = rag
        self.extract_graph = extract_graph

    def index(
        self,
        content: str,
    ) -> dict:
        """Проиндексировать repomix JSON.

        Args:
            content: JSON-строка в формате repomix
            extract_graph: извлекать граф из каждого файла (через GraphExtractor)

        Returns:
            {status, structure_doc_id, file_doc_ids, files_count, errors, error_details}

        Raises:
            ValueError: content — не валидный repomix JSON
        """
        try:
            data = json.loads(content)
        except json.JSONDecodeError as e:
            raise ValueError(f"content is not valid JSON: {e}") from e
        if not isinstance(data, dict):
            raise ValueError(f"repomix content must be a JSON object, got {type(data).__name__}")
        repo = data.get("repository", "unknown")
        structure = data.get("structure", [])
        files = data.get("files", {})
        if not isinstance(files, dict):
            raise ValueError(f"repomix 'files' must be an object, got {type(files).__name__}")

        # 1. Структура дерева — один документ. Для крошечных репозиториев дерево
        # может быть короче MIN_CONTENT_LENGTH — тогда пропускаем его, а не роняем
        # всю индексацию (структуру всё равно можно восстановить из file-документов).
        tree_text = f"Repository: {repo}\n\nStructure:\n" + "\n".join(structure)
        try:
            tree_doc_id = self.rag.add_document(
                tree_text,
                metadata={"source": repo, "type": "structure"},
            )
        except ValueError as e:
            logger.info("Structure document for %s skipped: %s", repo, e)
            tree_doc_id = None

        # 2. Каждый файл → документ
        file_doc_ids: dict[str, str] = {}
        file_dirs: dict[str, set[str]] = {}  # dir → set of doc_ids
        error_details: list[dict] = []

        for filepath, info in files.items():
            if not isinstance(info, dict):
                error_details.append(
                    {"path": filepath, "error": f"file entry must be an object, got {type(info).__name__}"}
                )
                continue
            file_content = info.get("content", "")
            if len(file_content.strip()) < self.rag.MIN_CONTENT_LENGTH:
                continue

            lang = info.get("language") or _detect_language(filepath)
            meta = {
                "source": repo,
                "path": filepath,
                "language": lang,
                "type": "file",
            }
            try:
                doc_id = self.rag.add_document(file_content, metadata=meta, extract_graph=self.extract_graph)
                file_doc_ids[filepath] = doc_id
                d = _get_dir(filepath)
                file_dirs.setdefault(d, set()).add(doc_id)
            except Exception as e:
                # Один битый файл не должен ронять всю индексацию, но причина
                # возвращается вызывающему в error_details, а не теряется.
                logger.warning("Indexing of %s failed: %s", filepath, e, exc_info=True)
                error_details.append({"path": filepath, "error": str(e)})

        # 3. Авто-связи: sibling для файлов одной папки
        for dirpath, ids in file_dirs.items():
            ids_list = list(ids)
            for i in range(len(ids_list)):
                for j in range(i + 1, len(ids_list)):
                    self.rag.add_relation(ids_list[i], ids_list[j], "sibling", weight=0.8)

        return {
            "status": "ok",
            "structure_doc_id": tree_doc_id,
            "file_doc_ids": file_doc_ids,
            "files_count": len(file_doc_ids),
            "errors": len(error_details),
            "error_details": error_details,
        }
