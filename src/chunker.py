"""Чанкование документов: semantic splitting с fallback.

Стратегия (paragraph → sentence → token):
  1. Текст режется на параграфы (по ``\\n\\n``).
  2. Параграф длиннее chunk_size токенов → режется на предложения (``. ! ?``).
  3. Предложение длиннее chunk_size → жёсткий срез по токенам.
  4. Соседние куски склеиваются, пока не превысят chunk_size.

Overlap реализован переносом хвоста предыдущего чанка (последние
``chunk_overlap`` токенов) в начало следующего.

Токенизация — tiktoken cl100k_base (быстрая, без torch); количество токенов
BGE-M3 отличается, но для целей нарезки это несущественно.
"""

import re
import uuid
from typing import Optional

_PARAGRAPH_RE = re.compile(r"\n\s*\n")
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")


def _get_encoder():
    import tiktoken

    return tiktoken.get_encoding("cl100k_base")


class Chunker:
    """Нарезка текста на чанки фиксированного токен-бюджета с overlap."""

    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 64):
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self._encoder = None

    def _enc(self):
        if self._encoder is None:
            self._encoder = _get_encoder()
        return self._encoder

    def count_tokens(self, text: str) -> int:
        return len(self._enc().encode(text))

    def needs_chunking(self, text: str) -> bool:
        return self.count_tokens(text) > self.chunk_size

    def _unit_budget(self) -> int:
        """Токен-бюджет нового контента в чанке.

        Инвариант: любой чанк ≤ chunk_size токенов ВКЛЮЧАЯ overlap-хвост
        предыдущего чанка и разделитель, поэтому единицы нарезки ограничены
        chunk_size - chunk_overlap - стоимость разделителя.
        """
        return max(1, self.chunk_size - self.chunk_overlap - self.count_tokens("\n\n"))

    # -- splitting ----------------------------------------------------------

    def _split_units(self, text: str) -> list[str]:
        """Разбить текст на единицы, каждая ≤ _unit_budget() токенов.

        Порядок: параграфы → предложения → жёсткий срез по токенам.
        """
        budget = self._unit_budget()
        units: list[str] = []
        for paragraph in _PARAGRAPH_RE.split(text):
            paragraph = paragraph.strip()
            if not paragraph:
                continue
            if self.count_tokens(paragraph) <= budget:
                units.append(paragraph)
                continue
            for sentence in _SENTENCE_RE.split(paragraph):
                sentence = sentence.strip()
                if not sentence:
                    continue
                if self.count_tokens(sentence) <= budget:
                    units.append(sentence)
                else:
                    units.extend(self._split_by_tokens(sentence, budget))
        return units

    def _split_by_tokens(self, text: str, budget: int) -> list[str]:
        enc = self._enc()
        tokens = enc.encode(text)
        parts = []
        for start in range(0, len(tokens), budget):
            parts.append(enc.decode(tokens[start:start + budget]).strip())
        return [p for p in parts if p]

    def _tail_tokens(self, text: str) -> str:
        """Последние chunk_overlap токенов текста (для overlap)."""
        if self.chunk_overlap == 0:
            return ""
        enc = self._enc()
        tokens = enc.encode(text)
        if not tokens:
            return ""
        return enc.decode(tokens[-self.chunk_overlap:]).strip()

    # -- public API ------------------------------------------------------------

    def chunk(
        self,
        text: str,
        doc_id: Optional[str] = None,
        metadata: Optional[dict] = None,
    ) -> list[dict]:
        """Нарезать текст на чанки.

        Args:
            text: исходный текст
            doc_id: идентификатор родительского документа (uuid4 по умолчанию)
            metadata: метаданные родителя — копируются в каждый чанк

        Returns:
            [{doc_id, parent_doc_id, chunk_index, text, metadata}]
            Если текст помещается в один чанк — один элемент с
            parent_doc_id=None и chunk_index=None (документ не считается
            чанкованным).
        """
        parent_id = doc_id or str(uuid.uuid4())
        base_meta = dict(metadata or {})

        if not self.needs_chunking(text):
            return [{
                "doc_id": parent_id,
                "parent_doc_id": None,
                "chunk_index": None,
                "text": text,
                "metadata": base_meta,
            }]

        units = self._split_units(text)
        sep_tokens = self.count_tokens("\n\n")
        budget = self._unit_budget()
        chunks: list[str] = []
        current: list[str] = []   # единицы нового контента текущего чанка (без overlap)
        current_tokens = 0
        prefix = ""               # overlap-хвост предыдущего чанка

        def flush():
            nonlocal prefix
            parts = ([prefix] if prefix else []) + current
            chunks.append("\n\n".join(parts))
            prefix = self._tail_tokens(chunks[-1])

        for unit in units:
            unit_tokens = self.count_tokens(unit)
            extra = unit_tokens + (sep_tokens if current else 0)
            if current and current_tokens + extra > budget:
                flush()
                current, current_tokens = [], 0
                extra = unit_tokens
            current.append(unit)
            current_tokens += extra
        if current:
            flush()

        result = []
        for i, chunk_text in enumerate(chunks):
            meta = dict(base_meta)
            meta["parent_doc_id"] = parent_id
            meta["chunk_index"] = i
            result.append({
                "doc_id": f"{parent_id}#{i}",
                "parent_doc_id": parent_id,
                "chunk_index": i,
                "text": chunk_text,
                "metadata": meta,
            })
        return result
