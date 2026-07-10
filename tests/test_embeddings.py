"""Тесты для модуля EmbeddingGenerator (TF-IDF векторизация)"""

import pytest
import numpy as np
from src.embeddings import EmbeddingGenerator


class TestEmbeddingGenerator:
    """Класс тестов для генератора эмбеддингов."""

    @pytest.fixture
    def generator(self):
        """Фикстура: базовый генератор."""
        return EmbeddingGenerator()

    def test_fit_and_transform_single_text(self, generator):
        """Должен вернуть вектор фиксированной размерности для одного текста."""
        texts = ["hello world"]
        vectors = generator.fit_transform(texts)
        assert len(vectors) == 1
        assert isinstance(vectors[0], list)
        # Размерность должна быть константной (количество уникальных слов)
        assert len(vectors[0]) > 0

    def test_fit_and_transform_multiple_texts(self, generator):
        """Должен вернуть матрицу для нескольких текстов."""
        texts = [
            "the cat sat on the mat",
            "the dog chased the cat",
            "the bird flew over the mat",
        ]
        vectors = generator.fit_transform(texts)
        assert len(vectors) == 3
        for v in vectors:
            assert isinstance(v, list)

    def test_transform_new_text(self, generator):
        """Должен векторизовать новый текст после fit."""
        texts = ["cat dog bird"]
        generator.fit(texts)
        vec = generator.transform(["new animal"])
        assert len(vec) == 1
        assert isinstance(vec[0], list)

    def test_semantic_similarity(self, generator):
        """Похожие тексты должны иметь более близкие векторы."""
        texts = ["python programming language", "java programming language", "quantum physics theory"]
        vectors = generator.fit_transform(texts)
        # Косинусная близость python vs java
        v0, v1, v2 = np.array(vectors[0]), np.array(vectors[1]), np.array(vectors[2])
        cos_py_java = np.dot(v0, v1) / (
            np.linalg.norm(v0) * np.linalg.norm(v1)
        )
        # Косинусная близость python vs quantum
        cos_py_quant = np.dot(v0, v2) / (
            np.linalg.norm(v0) * np.linalg.norm(v2)
        )
        assert cos_py_java > cos_py_quant, (
            f"Python-Java ({cos_py_java:.3f}) должно быть > Python-Quantum ({cos_py_quant:.3f})"
        )

    def test_embedding_dimension_consistency(self, generator):
        """Размерность эмбеддинга должна быть одинаковой для fit и transform."""
        texts = ["one", "two", "three"]
        generator.fit(texts)
        vec_train = generator.transform(["one"])[0]
        # Добавляем новое слово — размерность не меняется (используется vocabulary fit)
        vec_new = generator.transform(["four"])[0]
        assert len(vec_train) == len(vec_new), (
            f"Размерности не совпадают: {len(vec_train)} vs {len(vec_new)}"
        )

    def test_empty_text(self, generator):
        """Пустой текст должен давать нулевой вектор."""
        generator.fit(["valid text"])
        vec = generator.transform([""])[0]
        assert all(v == 0 for v in vec), "Пустой текст должен давать нулевой вектор"

    def test_unknown_words_return_zero_vector(self, generator):
        """Слова вне словаря должны игнорироваться (давать нулевой вектор)."""
        generator.fit(["hello"])
        vec = generator.transform(["world"])[0]  # 'world' не в словаре
        assert all(v == 0 for v in vec), "Неизвестное слово должно давать нулевой вектор"
