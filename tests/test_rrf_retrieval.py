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


def test_fail_closed_on_missing_embeddings():
    """验证 P0 缺陷修复：Dense RAG 在 Embedding 缺失或失败时 Fail-Closed，严禁静默退化"""
    import pytest
    from unittest.mock import patch
    from src.core.exceptions import EmbeddingUnavailableError

    with tempfile.TemporaryDirectory() as tmp_dir:
        store_file = Path(tmp_dir) / "fail_closed_store.json"
        store = VectorStore(storage_path=str(store_file))

        # 候选切片没有向量
        store.chunks = [
            {"id": "doc_1", "content": "没有向量切片A", "embedding": None, "metadata": {"type": "argument"}},
            {"id": "doc_2", "content": "没有向量切片B", "embedding": None, "metadata": {"type": "quote"}},
        ]

        # 1. dense_search 在无向量时必须坚决抛出异常阻断，绝不能静默返回 candidates[:top_k]
        with patch.object(store.model_provider, "get_embeddings", return_value=None):
            with pytest.raises(EmbeddingUnavailableError) as exc:
                store.dense_search("测试查询", top_k=2)
            assert "Dense 语义检索不可用" in str(exc.value)

        # 2. 即使 query 获得了向量，但库中切片均无向量，dense_search 仍必须 fail-closed
        with patch.object(store.model_provider, "get_embeddings", return_value=[[0.1, 0.2]]):
            with pytest.raises(EmbeddingUnavailableError) as exc:
                store.dense_search("测试查询", top_k=2)
            assert "Fail-Closed" in str(exc.value)

        # 3. hybrid_search 在 require_dense=True 时同样必须 fail-closed
        with patch.object(store.model_provider, "get_embeddings", return_value=None):
            with pytest.raises(EmbeddingUnavailableError) as exc:
                store.hybrid_search("测试查询", top_k=2, require_dense=True)
            assert "Hybrid 检索 Dense 通道不可用" in str(exc.value)

        # 4. hybrid_search 在生产宽松模式下 (require_dense=False) 仍允许降级为单通道稀疏排序
        with patch.object(store.model_provider, "get_embeddings", return_value=None):
            results = store.hybrid_search("向量", top_k=2, require_dense=False)
            assert len(results) > 0
            assert "rrf_score" in results[0]


def test_fail_closed_on_partial_missing_and_dimension_mismatch():
    """验证 P0 缺陷修复：候选切片部分缺失向量或维度不匹配时严格阻断，防止 zip 截断与不公平比对"""
    import pytest
    from unittest.mock import patch
    from src.core.exceptions import EmbeddingUnavailableError, EmbeddingDimensionMismatchError

    with tempfile.TemporaryDirectory() as tmp_dir:
        store_file = Path(tmp_dir) / "partial_dim_store.json"
        store = VectorStore(storage_path=str(store_file))

        # 场景 1：候选集中只有部分切片具有向量 (doc_1 有，doc_2 为 None)
        store.chunks = [
            {"id": "doc_1", "content": "有向量切片", "embedding": [0.5, 0.5, 0.5], "metadata": {"type": "argument"}},
            {"id": "doc_2", "content": "无向量残缺切片", "embedding": None, "metadata": {"type": "quote"}},
        ]

        # dense_search 必须 Fail-Closed，严禁将 doc_2 赋 0 分强行参与排序
        with patch.object(store.model_provider, "get_embeddings", return_value=[[0.5, 0.5, 0.5]]):
            with pytest.raises(EmbeddingUnavailableError) as exc_dense:
                store.dense_search("测试查询", top_k=2)
            assert "候选切片存在部分缺少 embedding 向量" in str(exc_dense.value)

        # hybrid_search(require_dense=True) 也必须 Fail-Closed
        with patch.object(store.model_provider, "get_embeddings", return_value=[[0.5, 0.5, 0.5]]):
            with pytest.raises(EmbeddingUnavailableError) as exc_hybrid:
                store.hybrid_search("测试查询", top_k=2, require_dense=True)
            assert "候选切片存在部分缺少 embedding 向量" in str(exc_hybrid.value)

        # 场景 2：向量维度不匹配 (Query 维度 3，切片维度 2)
        store.chunks = [
            {"id": "doc_3", "content": "二维向量切片", "embedding": [0.5, 0.5], "metadata": {"type": "argument"}},
        ]
        with patch.object(store.model_provider, "get_embeddings", return_value=[[0.5, 0.5, 0.5]]):
            with pytest.raises(EmbeddingDimensionMismatchError) as exc_dim:
                store.dense_search("测试查询", top_k=1)
            assert "向量维度不匹配" in str(exc_dim.value)
            assert exc_dim.value.query_dim == 3
            assert exc_dim.value.chunk_dim == 2

        # 场景 3：add_chunks 返回向量长度不匹配或维度不一
        chunks_to_add = [{"content": "文本1"}, {"content": "文本2"}]
        # 返回数量不符
        with patch.object(store.model_provider, "get_embeddings", return_value=[[0.1, 0.2]]):
            with pytest.raises(EmbeddingUnavailableError) as exc_len:
                store.add_chunks(chunks_to_add)
            assert "向量生成数量不匹配" in str(exc_len.value)

        # 返回部分 None
        with patch.object(store.model_provider, "get_embeddings", return_value=[[0.1, 0.2], None]):
            with pytest.raises(EmbeddingUnavailableError) as exc_none:
                store.add_chunks(chunks_to_add)
            assert "拒绝残缺向量入库" in str(exc_none.value)

        # 返回维度不一致
        with patch.object(store.model_provider, "get_embeddings", return_value=[[0.1, 0.2], [0.1, 0.2, 0.3]]):
            with pytest.raises(EmbeddingDimensionMismatchError) as exc_diff:
                store.add_chunks(chunks_to_add)
            assert "向量维度不匹配" in str(exc_diff.value)


def test_add_chunks_rejects_dimension_mismatch_with_existing_store():
    """验证 add_chunks 跨批次追加切片时，若新批次维度与库内已有向量维度不一致，严格阻断抛出异常"""
    import pytest
    from unittest.mock import patch
    from src.core.exceptions import EmbeddingDimensionMismatchError

    with tempfile.TemporaryDirectory() as tmp_dir:
        store_file = Path(tmp_dir) / "cross_batch_dim_store.json"
        store = VectorStore(storage_path=str(store_file))

        # 第一批：写入 3 维向量切片
        batch1 = [{"id": "b1", "content": "第一批3维切片"}]
        with patch.object(store.model_provider, "get_embeddings", return_value=[[0.1, 0.2, 0.3]]):
            store.add_chunks(batch1)

        assert len(store.chunks) == 1
        assert len(store.chunks[0]["embedding"]) == 3

        # 第二批：试图写入 2 维向量切片，必须 Fail-Closed 阻断，拒绝污染数据库
        batch2 = [{"id": "b2", "content": "第二批2维切片"}]
        with patch.object(store.model_provider, "get_embeddings", return_value=[[0.1, 0.2]]):
            with pytest.raises(EmbeddingDimensionMismatchError) as exc_info:
                store.add_chunks(batch2)
            assert exc_info.value.query_dim == 3
            assert exc_info.value.chunk_dim == 2


def test_hybrid_search_unembedded_chunks_do_not_receive_dense_rank():
    """验证在宽松模式下 (require_dense=False)，缺失向量的切片绝不获得 Dense Rank，防止其凭空被赋予 RRF 倒数排名加分"""
    from unittest.mock import patch

    with tempfile.TemporaryDirectory() as tmp_dir:
        store_file = Path(tmp_dir) / "unembedded_dense_rank_store.json"
        store = VectorStore(storage_path=str(store_file))

        # doc_1 具有向量，doc_2 没有向量 (embedding: None)
        # 且 doc_2 的文本与 query 不重叠，doc_1 与 query 也不重叠
        store.chunks = [
            {"id": "doc_1", "content": "无关内容甲", "embedding": [1.0, 0.0], "metadata": {}},
            {"id": "doc_2", "content": "无关内容乙", "embedding": None, "metadata": {}},
        ]

        # 模拟 query 向量偏向 doc_1
        with patch.object(store.model_provider, "get_embeddings", return_value=[[1.0, 0.0]]):
            results = store.hybrid_search("测试查询", top_k=2, require_dense=False)
            res_dict = {r["id"]: r for r in results}

            # doc_1 有向量，获得 dense rank = 1，其 rrf_score 应包含 1 / (60 + 1)
            # doc_2 没有向量，dense_ranks 中必须无其键，绝不应获得 dense rank (即不应获得 1/(60 + 2))
            assert "doc_1" in res_dict
            assert "doc_2" in res_dict

            # 验证 doc_1 的 RRF 分数包含双通道贡献 (dense + sparse ≈ 0.0328)
            assert res_dict["doc_1"]["rrf_score"] > 0.030
            # 验证 doc_2 因缺少向量只获得单通道 sparse 贡献 (≈ 0.0161)，绝未叠加 1/(60+2) 的错误 dense 加分 (若叠加则会超过 0.030)
            assert res_dict["doc_2"]["rrf_score"] < 0.020
            expected_sparse_only = round(1.0 / (60 + 2), 6)
            assert abs(res_dict["doc_2"]["rrf_score"] - expected_sparse_only) < 1e-4




