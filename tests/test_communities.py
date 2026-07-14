"""Тесты для rag_find_communities, rag_set_community_names, rag_get_communities.

TDD: тесты написаны ДО реализации. Все должны падать при запуске.
"""


class TestFindCommunities:
    """Интеграционные тесты RAGSystem.find_communities."""

    def test_returns_communities(self, rag):
        """Должен вернуть непустой список сообществ при наличии документов."""
        # Добавляем документы с разной семантикой
        rag.add_document("python programming language for general purpose scripting and automation tasks")
        rag.add_document("java programming language for enterprise software development and scaling")
        rag.add_document("javascript web development frontend backend node.js react angular")
        rag.add_document("database design sql postgresql mysql indexing optimization queries")
        rag.add_document("machine learning python tensorflow pytorch neural networks training")
        rag.add_document("deep learning neural networks convolutional recurrent architectures")

        result = rag.find_communities()
        assert "communities" in result
        assert "total_communities" in result
        assert "total_nodes" in result
        assert result["total_nodes"] == 6
        assert result["total_communities"] >= 1

    def test_communities_have_correct_structure(self, rag):
        """Каждое сообщество должно иметь id, size, members."""
        rag.add_document("python programming language for general purpose scripting and automation tasks")
        rag.add_document("java programming language for enterprise software development and scaling")
        rag.add_document("database design sql postgresql mysql indexing optimization queries")

        result = rag.find_communities()
        for community in result["communities"]:
            assert "id" in community
            assert "size" in community
            assert "members" in community
            assert isinstance(community["id"], int)
            assert isinstance(community["size"], int)
            assert isinstance(community["members"], list)
            assert community["size"] == len(community["members"])

    def test_all_documents_in_some_community(self, rag):
        """Каждый документ должен попадать в какое-то сообщество."""
        rag.add_document("python programming language for general purpose scripting and automation tasks")
        rag.add_document("java programming language for enterprise software development and scaling")
        rag.add_document("database design sql postgresql mysql indexing optimization queries")

        result = rag.find_communities()
        all_members = set()
        for c in result["communities"]:
            all_members.update(c["members"])

        # Все doc_id должны быть в каком-то сообществе
        items, _ = rag.doc_store.list(limit=100)
        all_doc_ids = {d["doc_id"] for d in items}
        assert all_doc_ids == all_members

    def test_resolution_affects_granularity(self, rag):
        """Большее resolution → больше сообществ (мелких)."""
        for i in range(10):
            rag.add_document(f"document about topic number {i} with unique keywords for testing community detection algorithm")

        low = rag.find_communities(resolution=0.5)
        high = rag.find_communities(resolution=2.0)
        # Высокое resolution обычно даёт больше сообществ
        assert high["total_communities"] >= low["total_communities"]

    def test_with_existing_relations(self, rag):
        """Существующие рёбра графа должны влиять на сообщества."""
        d1 = rag.add_document("python programming language for general purpose scripting and automation tasks")
        d2 = rag.add_document("java programming language for enterprise software development and scaling")
        rag.add_document("database design sql postgresql mysql indexing optimization queries")

        # Связываем d1 и d2 — они должны быть в одном сообществе
        rag.add_relation(d1, d2, "related_to")

        result = rag.find_communities()
        # Найти сообщества содержащие d1 и d2
        d1_community = None
        d2_community = None
        for c in result["communities"]:
            if d1 in c["members"]:
                d1_community = c["id"]
            if d2 in c["members"]:
                d2_community = c["id"]
        assert d1_community == d2_community

    def test_empty_store(self, rag):
        """Пустой стор — не должно падать."""
        result = rag.find_communities()
        assert result["total_communities"] == 0
        assert result["total_nodes"] == 0
        assert result["communities"] == []

    def test_single_document(self, rag):
        """Один документ — одно сообщество."""
        rag.add_document("python programming language for general purpose scripting and automation tasks")
        result = rag.find_communities()
        assert result["total_communities"] == 1
        assert result["communities"][0]["size"] == 1

    def test_caches_result(self, rag):
        """Результат кешируется для использования в set_community_names."""
        rag.add_document("python programming language for general purpose scripting and automation tasks")
        rag.add_document("java programming language for enterprise software development and scaling")

        rag.find_communities()
        # Кеш должен содержать сообщества
        assert len(rag._communities) >= 1


class TestSetCommunityNames:
    """Тесты RAGSystem.set_community_names."""

    def test_set_names(self, rag):
        """Должен установить имена для сообществ."""
        rag.add_document("python programming language for general purpose scripting and automation tasks")
        rag.add_document("java programming language for enterprise software development and scaling")
        rag.add_document("database design sql postgresql mysql indexing optimization queries")

        rag.find_communities()
        community_ids = [c["id"] for c in rag._communities]

        names = {community_ids[0]: "Programming Languages"}
        result = rag.set_community_names(names)
        assert result["status"] == "ok"
        assert result["updated"] == 1

    def test_names_appear_in_get_communities(self, rag):
        """Имена должны отображаться в get_communities."""
        rag.add_document("python programming language for general purpose scripting and automation tasks")
        rag.add_document("java programming language for enterprise software development and scaling")

        rag.find_communities()
        community_ids = [c["id"] for c in rag._communities]

        rag.set_community_names({community_ids[0]: "Languages"})

        communities = rag.get_communities()
        named = [c for c in communities if c["name"] == "Languages"]
        assert len(named) == 1

    def test_overwrite_name(self, rag):
        """Повторный set перезаписывает имя."""
        rag.add_document("python programming language for general purpose scripting and automation tasks")
        rag.find_communities()
        community_ids = [c["id"] for c in rag._communities]

        rag.set_community_names({community_ids[0]: "First Name"})
        rag.set_community_names({community_ids[0]: "Second Name"})

        communities = rag.get_communities()
        found = [c for c in communities if c["id"] == community_ids[0]]
        assert found[0]["name"] == "Second Name"

    def test_multiple_names(self, rag):
        """Можно задать имена нескольким сообществам сразу."""
        for i in range(6):
            rag.add_document(f"document about topic number {i} with unique keywords for testing")

        rag.find_communities()
        names = {c["id"]: f"Community {c['id']}" for c in rag._communities}
        result = rag.set_community_names(names)
        assert result["updated"] == len(names)


class TestGetCommunities:
    """Тесты RAGSystem.get_communities."""

    def test_empty_before_find(self, rag):
        """До find_communities — пустой список."""
        result = rag.get_communities()
        assert result == []

    def test_after_find(self, rag):
        """После find — список сообществ без имен."""
        rag.add_document("python programming language for general purpose scripting and automation tasks")
        rag.add_document("java programming language for enterprise software development and scaling")

        rag.find_communities()
        communities = rag.get_communities()
        assert len(communities) >= 1
        for c in communities:
            assert "id" in c
            assert "name" in c
            assert "size" in c
            assert "members" in c

    def test_with_names(self, rag):
        """С именами — name заполнен."""
        rag.add_document("python programming language for general purpose scripting and automation tasks")
        rag.find_communities()
        community_ids = [c["id"] for c in rag._communities]

        rag.set_community_names({community_ids[0]: "Test"})
        communities = rag.get_communities()
        named = [c for c in communities if c["id"] == community_ids[0]]
        assert named[0]["name"] == "Test"


class TestMCPFindCommunities:
    """Тесты MCP-хэндлеров через handle_tool_call."""

    def test_find_communities(self, rag):
        from src.mcp_server import handle_tool_call
        rag.add_document("python programming language for general purpose scripting and automation tasks")
        rag.add_document("java programming language for enterprise software development and scaling")

        result = handle_tool_call(rag, "rag_find_communities", {})
        assert "communities" in result
        assert result["total_nodes"] == 2

    def test_find_communities_with_params(self, rag):
        from src.mcp_server import handle_tool_call
        for i in range(5):
            rag.add_document(f"document about topic number {i} with unique keywords for testing")

        result = handle_tool_call(rag, "rag_find_communities", {"resolution": 1.5, "k_nn": 3})
        assert "communities" in result

    def test_set_community_names(self, rag):
        from src.mcp_server import handle_tool_call
        rag.add_document("python programming language for general purpose scripting and automation tasks")
        rag.add_document("java programming language for enterprise software development and scaling")

        find_result = handle_tool_call(rag, "rag_find_communities", {})
        community_ids = [c["id"] for c in find_result["communities"]]

        result = handle_tool_call(rag, "rag_set_community_names", {
            "names": {str(community_ids[0]): "Languages"}
        })
        assert result["status"] == "ok"
        assert result["updated"] == 1

    def test_get_communities(self, rag):
        from src.mcp_server import handle_tool_call
        rag.add_document("python programming language for general purpose scripting and automation tasks")

        handle_tool_call(rag, "rag_find_communities", {})
        result = handle_tool_call(rag, "rag_get_communities", {})
        assert isinstance(result, list)
        assert len(result) == 1


class TestNewToolDefs:
    """Проверяем наличие новых инструментов в TOOL_DEFS."""

    def test_find_communities_tool_exists(self):
        from src.mcp_server import TOOL_DEFS
        names = {t.name for t in TOOL_DEFS}
        assert "rag_find_communities" in names

    def test_set_community_names_tool_exists(self):
        from src.mcp_server import TOOL_DEFS
        names = {t.name for t in TOOL_DEFS}
        assert "rag_set_community_names" in names

    def test_get_communities_tool_exists(self):
        from src.mcp_server import TOOL_DEFS
        names = {t.name for t in TOOL_DEFS}
        assert "rag_get_communities" in names

    def test_find_communities_schema(self):
        from src.mcp_server import TOOL_DEFS
        tool = next(t for t in TOOL_DEFS if t.name == "rag_find_communities")
        schema = tool.inputSchema
        assert "resolution" in schema["properties"]
        assert "k_nn" in schema["properties"]

    def test_set_community_names_schema(self):
        from src.mcp_server import TOOL_DEFS
        tool = next(t for t in TOOL_DEFS if t.name == "rag_set_community_names")
        schema = tool.inputSchema
        assert "names" in schema["properties"]
        assert "names" in schema["required"]


class TestCommunityCacheInvalidation:
    """Инвалидация кеша сообществ при изменении сторов."""

    def test_clear_resets_communities(self, rag):
        """После clear() кеш сообществ должен быть пуст."""
        rag.add_document("python programming language for general purpose scripting and automation tasks")
        rag.add_document("java programming language for enterprise software development and scaling")
        rag.find_communities()
        assert len(rag._communities) > 0

        rag.clear()
        assert rag._communities == []
        assert rag._community_names == {}

    def test_delete_document_resets_communities(self, rag):
        """После delete_document() кеш сообществ должен быть пуст."""
        rag.add_document("python programming language for general purpose scripting and automation tasks")
        rag.add_document("java programming language for enterprise software development and scaling")
        rag.find_communities()
        assert len(rag._communities) > 0

        items, _ = rag.doc_store.list(limit=1)
        rag.delete_document(items[0]["doc_id"])
        assert rag._communities == []
        assert rag._community_names == {}

    def test_update_document_resets_communities(self, rag):
        """После update_document() кеш сообществ должен быть пуст."""
        doc_id = rag.add_document("python programming language for general purpose scripting and automation tasks")
        rag.add_document("java programming language for enterprise software development and scaling")
        rag.find_communities()
        assert len(rag._communities) > 0

        rag.update_document(doc_id, text="updated content for this test document to verify text update works correctly and thoroughly")
        assert rag._communities == []
        assert rag._community_names == {}


class TestCommunityCachePersistence:
    """Персистентность кеша сообществ через JSON-файл."""

    def test_cache_file_created_after_find(self, make_rag):
        """После find_communities() создаётся JSON-файл на диске."""
        rag = make_rag()
        rag.add_document("python programming language for general purpose scripting and automation tasks")
        rag.add_document("java programming language for enterprise software development and scaling")
        rag.find_communities()
        assert rag._community_cache_path.exists()
        assert rag._community_cache_path.stat().st_size > 0

    def test_communities_restored_after_restart(self, make_rag):
        """После перезапуска (новый RAGSystem с тем же store_path) кеш восстанавливается."""
        import copy

        # Первый запуск: find + имена
        rag1 = make_rag(subdir="persist_test")
        rag1.add_document("python programming language for general purpose scripting and automation tasks")
        rag1.add_document("java programming language for enterprise software development and scaling")
        rag1.find_communities()
        community_ids = [c["id"] for c in rag1._communities]
        rag1.set_community_names({community_ids[0]: "Languages"})
        saved_communities = copy.deepcopy(rag1._communities)
        saved_names = dict(rag1._community_names)
        store_path = rag1.store_path
        rag1.close()

        # Второй запуск: кеш должен восстановиться
        from src.config import RAGConfig
        from src.rag import RAGSystem
        from tests.conftest import HashEmbeddingGenerator
        cfg = RAGConfig()
        cfg.store_path = store_path
        rag2 = RAGSystem(config=cfg, embedding_generator=HashEmbeddingGenerator())
        try:
            assert rag2._communities == saved_communities
            assert rag2._community_names == saved_names
            # get_communities должен вернуть то же самое
            result = rag2.get_communities()
            assert len(result) == len(saved_communities)
            named = [c for c in result if c["name"] == "Languages"]
            assert len(named) == 1
        finally:
            rag2.close()

    def test_clear_removes_cache_file(self, make_rag):
        """После clear() JSON-файл удаляется."""
        rag = make_rag()
        rag.add_document("python programming language for general purpose scripting and automation tasks")
        rag.find_communities()
        assert rag._community_cache_path.exists()

        rag.clear()
        assert not rag._community_cache_path.exists()

    def test_names_persist_after_restart(self, make_rag):
        """Имена сохраняются через перезапуск."""

        rag1 = make_rag(subdir="names_test")
        rag1.add_document("python programming language for general purpose scripting and automation tasks")
        rag1.add_document("java programming language for enterprise software development and scaling")
        rag1.find_communities()
        community_ids = [c["id"] for c in rag1._communities]
        rag1.set_community_names({community_ids[0]: "Programming", community_ids[1] if len(community_ids) > 1 else community_ids[0]: "Other"})
        saved_names = dict(rag1._community_names)
        store_path = rag1.store_path
        rag1.close()

        from src.config import RAGConfig
        from src.rag import RAGSystem
        from tests.conftest import HashEmbeddingGenerator
        cfg = RAGConfig()
        cfg.store_path = store_path
        rag2 = RAGSystem(config=cfg, embedding_generator=HashEmbeddingGenerator())
        try:
            assert rag2._community_names == saved_names
        finally:
            rag2.close()
