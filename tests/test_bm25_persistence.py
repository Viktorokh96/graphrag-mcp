"""Тесты персистентности BM25 индекса."""

import os
import pytest
import tempfile
import shutil
from src.bm25_index import BM25Index


class TestBM25Persistence:
    """Тесты сохранения и загрузки BM25 индекса."""

    @pytest.fixture
    def temp_dir(self):
        """Создать временную директорию."""
        path = tempfile.mkdtemp()
        yield path
        shutil.rmtree(path)

    def test_save_and_load_empty(self, temp_dir):
        """Сохранить и загрузить пустой индекс."""
        index = BM25Index(store_path=temp_dir)
        index.save()
        
        # Проверяем, что файл создан
        index2 = BM25Index(store_path=temp_dir)
        assert index2.stats()["total_documents"] == 0

    def test_save_and_load_with_documents(self, temp_dir):
        """Сохранить и загрузить индекс с документами."""
        # Создаём и наполняем
        index = BM25Index(store_path=temp_dir)
        index.add_document("doc1", "python programming language")
        index.add_document("doc2", "machine learning algorithms")
        index.save()
        
        # Загружаем в новый инстанс
        index2 = BM25Index(store_path=temp_dir)
        assert index2.stats()["total_documents"] == 2
        
        # Поиск должен работать
        results = index2.search("python", k=5)
        assert len(results) > 0
        assert results[0][0] == "doc1"

    def test_search_after_reload(self, temp_dir):
        """Поиск работает после перезагрузки."""
        # Первая сессия
        index = BM25Index(store_path=temp_dir)
        index.add_document("doc1", "python is a programming language")
        index.add_document("doc2", "java is also a programming language")
        index.save()
        
        # Вторая сессия (имитация перезапуска)
        index2 = BM25Index(store_path=temp_dir)
        results = index2.search("python", k=5)
        assert len(results) == 1
        assert results[0][0] == "doc1"

    def test_add_after_load(self, temp_dir):
        """Добавление документов после загрузки работает."""
        index = BM25Index(store_path=temp_dir)
        index.add_document("doc1", "python programming")
        index.save()
        
        # Загружаем и добавляем ещё
        index2 = BM25Index(store_path=temp_dir)
        index2.add_document("doc2", "java programming")
        index2.save()
        
        # Проверяем финальное состояние
        index3 = BM25Index(store_path=temp_dir)
        assert index3.stats()["total_documents"] == 2
        results = index3.search("java", k=5)
        assert len(results) == 1
        assert results[0][0] == "doc2"

    def test_clear_removes_files(self, temp_dir):
        """Очистка индекса удаляет файлы."""
        index = BM25Index(store_path=temp_dir)
        index.add_document("doc1", "python programming")
        index.save()
        
        # Проверяем, что файл есть
        assert os.path.exists(os.path.join(temp_dir, "bm25_index.json"))
        
        # Очищаем
        index.clear()
        # Файл должен быть удалён или обнулён
        index.save()
        
        # Загружаем — должен быть пустым
        index2 = BM25Index(store_path=temp_dir)
        assert index2.stats()["total_documents"] == 0

    def test_no_store_path_creates_no_file(self, temp_dir):
        """Без store_path индекс не сохраняется на диск."""
        index = BM25Index()  # без store_path
        index.add_document("doc1", "python programming")
        
        # save не должен падать, но и файла не будет
        index.save()  # должна быть no-op

    def test_tokenization_improved(self):
        """Токенизация использует regex и lower case."""
        index = BM25Index()
        # Текст с пунктуацией и заглавными
        index.add_document("doc1", "Python — мощный язык программирования!")
        # Проверяем, что токены нормальные (без пунктуации)
        assert "python" in index._documents[0]
        assert "мощный" in index._documents[0]
        # "—" и "!" не должны быть в токенах
        assert "—" not in index._documents[0]
        assert "!" not in index._documents[0]

    def test_search_case_insensitive(self):
        """Поиск нечувствителен к регистру."""
        index = BM25Index()
        index.add_document("doc1", "Python Programming Language")
        
        results_lower = index.search("python", k=5)
        results_upper = index.search("PYTHON", k=5)
        
        assert len(results_lower) == 1
        assert len(results_upper) == 1
        assert results_lower[0][0] == "doc1"
        assert results_upper[0][0] == "doc1"