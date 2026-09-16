import tempfile
from pathlib import Path
from src.memory.vector_store import VectorStore
from src.memory.memory_manager import MemoryManager


def test_vector_store_sparse_search():
    with tempfile.TemporaryDirectory() as tmp_dir:
        store_file = Path(tmp_dir) / "test_store.json"
        store = VectorStore(storage_path=str(store_file))

        chunks = [
            {"id": "c1", "content": "人工智能正在深刻改变内容创作的底层范式与生产关系。", "source": "文章A", "type": "argument"},
            {"id": "c2", "content": "周末去野外露营烤肉，空气清新，心情十分惬意放松。", "source": "文章B", "type": "argument"},
            {"id": "c3", "content": "写作者的独特个人风格是机器无法轻易抹杀的灵魂印记。", "source": "文章A", "type": "quote"},
        ]
        store.add_chunks(chunks)

        # 搜索与“创作、风格”相关的语料
        results = store.search("内容创作与风格", top_k=2)
        assert len(results) == 2
        # c1 或 c3 应该优先排在前面，而不是烤肉的 c2
        matched_ids = [r["id"] for r in results]
        assert "c1" in matched_ids or "c3" in matched_ids


def test_memory_manager_ingest_and_retrieve():
    with tempfile.TemporaryDirectory() as tmp_dir:
        store_file = Path(tmp_dir) / "test_memory.json"
        store = VectorStore(storage_path=str(store_file))
        mgr = MemoryManager(vector_store=store)

        article = """这是开篇第一段引言，用来吸引读者的注意力。

说白了，很多人忽略了底层逻辑的重要性，盲目跟风。

我们在论证第二点的时候，需要提供充分翔实的案例作为支撑。

最后这是结尾段落，给出一个耐人寻味的总结。"""

        added = mgr.ingest_article("深度好文", article)
        assert added >= 3

        stats = mgr.get_memory_stats()
        assert stats["total_chunks"] >= 3
        assert "深度好文" in stats["sources"]

        # 动态检索
        few_shots = mgr.retrieve_dynamic_few_shots("底层逻辑与盲目跟风", top_k=1)
        assert len(few_shots) == 1
        assert "底层逻辑" in few_shots[0]
