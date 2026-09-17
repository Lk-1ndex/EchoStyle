import tempfile
from pathlib import Path
from src.memory.vector_store import VectorStore


def test_rrf_reciprocal_rank_fusion_math():
    """验证 RRF 算法在无 Dense API 下基于稀疏排名的融合数学计算"""
    with tempfile.TemporaryDirectory() as tmp_dir:
        store_file = Path(tmp_dir) / "rrf_store.json"
        store = VectorStore(storage_path=str(store_file))

        chunks = [
            {"id": "doc_1", "content": "人工智能是深刻影响内容创作与思想表达的生产工具。", "source": "A", "type": "argument"},
            {"id": "doc_2", "content": "今天中午在公司食堂吃红烧牛肉面，味道一般般。", "source": "B", "type": "argument"},
            {"id": "doc_3", "content": "写作者的独特个人风格是机器无法取代的灵魂印记。", "source": "C", "type": "quote"},
        ]
        store.add_chunks(chunks)

        # 针对包含"创作、风格"的查询执行 RRF 搜索
        results = store.hybrid_search("内容创作与写作者风格", top_k=2)
        assert len(results) == 2

        # 检查是否附加了 rrf_score
        assert "rrf_score" in results[0]
        assert results[0]["rrf_score"] > 0

        # 排名前两位的应当是 doc_1 或 doc_3，而不是牛肉面的 doc_2
        top_ids = [r["id"] for r in results]
        assert "doc_1" in top_ids or "doc_3" in top_ids
        assert "doc_2" not in top_ids

        # 验证 RRF 公式: 1 / (60 + rank)
        # 第一名 rank=1 时，理论得分为 1 / 61 ≈ 0.016393
        expected_top_score = round(1.0 / (60 + 1), 6)
        assert abs(results[0]["rrf_score"] - expected_top_score) < 0.001


def test_dense_search_with_embeddings():
    """验证纯密集向量检索 dense_search 依据余弦相似度排序并记录 dense_score"""
    from unittest.mock import patch

    with tempfile.TemporaryDirectory() as tmp_dir:
        store_file = Path(tmp_dir) / "dense_store.json"
        store = VectorStore(storage_path=str(store_file))

        store.chunks = [
            {"id": "doc_1", "content": "技术架构与分布式系统", "embedding": [1.0, 0.0, 0.0], "metadata": {"type": "argument"}},
            {"id": "doc_2", "content": "文学随笔与人生感悟", "embedding": [0.0, 1.0, 0.0], "metadata": {"type": "quote"}},
        ]

        # 模拟 query 向量偏向 doc_1
        with patch.object(store.model_provider, "get_embeddings", return_value=[[0.95, 0.05, 0.0]]):
            results = store.dense_search("技术与架构演进", top_k=2)
            assert len(results) == 2
            assert results[0]["id"] == "doc_1"
            assert "dense_score" in results[0]
            assert results[0]["dense_score"] > results[1]["dense_score"]

        # 验证带 type_filter
        with patch.object(store.model_provider, "get_embeddings", return_value=[[0.95, 0.05, 0.0]]):
            filtered = store.dense_search("技术与架构演进", top_k=2, type_filter="quote")
            assert len(filtered) == 1
            assert filtered[0]["id"] == "doc_2"

            # 验证不存在的类型返回空列表，绝不静默降级为其他类型
            non_existent = store.dense_search("技术与架构演进", top_k=2, type_filter="non_existent")
            assert non_existent == []

            # 验证 hybrid_search 不存在类型也返回空列表
            non_existent_hybrid = store.hybrid_search("技术与架构演进", top_k=2, type_filter="non_existent")
            assert non_existent_hybrid == []

