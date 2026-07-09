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
import os

from src.rag import RAGSystem


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
            {status, doc_ids, files_count}
        """
        data = json.loads(content)
        repo = data.get("repository", "unknown")
        structure = data.get("structure", [])
        files = data.get("files", {})

        # 1. Структура дерева — один документ
        tree_text = f"Repository: {repo}\n\nStructure:\n" + "\n".join(structure)
        tree_doc_id = self.rag.add_document(
            tree_text,
            metadata={"source": repo, "type": "structure"},
        )

        # 2. Каждый файл → документ
        file_doc_ids: dict[str, str] = {}
        file_dirs: dict[str, set[str]] = {}  # dir → set of doc_ids
        errors = 0

        for filepath, info in files.items():
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
            except ValueError:
                errors += 1

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
            "errors": errors,
        }
