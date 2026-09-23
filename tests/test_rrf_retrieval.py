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

        # doc_1 具有向量，但文本与 query 无交集；doc_2 没有向量 (embedding: None)，但文本与 query 有交集
        store.chunks = [
            {"id": "doc_1", "content": "无关内容甲", "embedding": [1.0, 0.0], "metadata": {}},
            {"id": "doc_2", "content": "测试查询相关内容乙", "embedding": None, "metadata": {}},
        ]

        # 模拟 query 向量偏向 doc_1
        with patch.object(store.model_provider, "get_embeddings", return_value=[[1.0, 0.0]]):
            results = store.hybrid_search("测试查询", top_k=2, require_dense=False)
            res_dict = {r["id"]: r for r in results}

            # doc_1 有向量，获得 dense rank = 1，其 rrf_score 应包含 1 / (60 + 1)
            # doc_2 没有向量，dense_ranks 中必须无其键，绝不应获得 dense rank (即不应获得 1/(60 + 2))
            assert "doc_1" in res_dict
            assert "doc_2" in res_dict

            # doc_1 获得单通道 dense 贡献 (≈ 0.016393)，文本无交集因此无 sparse 贡献
            assert abs(res_dict["doc_1"]["rrf_score"] - round(1.0 / (60 + 1), 6)) < 1e-4
            # doc_2 因缺少向量只获得单通道 sparse 贡献 (≈ 0.016393)，绝未叠加 dense 加分
            assert abs(res_dict["doc_2"]["rrf_score"] - round(1.0 / (60 + 1), 6)) < 1e-4


def test_rrf_zero_similarity_no_sparse_rank_or_insertion_bias():
    """验证 P0 缺陷修复：零词汇交集文档绝不获得 Sparse 排名，且绝不通过填充 top_k 凭入库顺序混入检索结果"""
    with tempfile.TemporaryDirectory() as tmp_dir:
        store_file = Path(tmp_dir) / "zero_sim_store.json"
        store = VectorStore(storage_path=str(store_file))

        # 入库 3 篇文档，前两篇入库更早但与 query 完全无词汇交集
        chunks = [
            {"id": "doc_early_1", "content": "太阳从东边升起，西边落下。", "source": "A"},
            {"id": "doc_early_2", "content": "今天在公园散步，天气非常晴朗。", "source": "B"},
            {"id": "doc_matched", "content": "深度学习大模型文风迁移与智能写作技术。", "source": "C"},
        ]
        store.add_chunks(chunks)

        # 纯稀疏/混合检索 (无向量时退化为单通道稀疏)
        results = store.hybrid_search("深度学习大模型", top_k=3, require_dense=False)
        result_ids = [r["id"] for r in results]

        # 核心断言 1：只有具有真实正向相关性的文档被召回 (len 为 1，绝不硬凑 top_k 3 个)
        assert len(results) == 1
        assert results[0]["id"] == "doc_matched"
        assert results[0]["rrf_score"] > 0.016

        # 核心断言 2：doc_early_1 和 doc_early_2 与 query 完全无交集，其实际 sparse score = 0，
        # 绝不能因为入库顺序更早而获得排名，更严禁混入检索结果中！
        assert "doc_early_1" not in result_ids
        assert "doc_early_2" not in result_ids


def test_rrf_insertion_order_invariance_and_top_n_window():
    """验证 RRF 各通道按有效 Top-N 窗口召回，且彻底消除入库顺序对召回与排序的扰动"""
    with tempfile.TemporaryDirectory() as tmp_dir:
        store_file1 = Path(tmp_dir) / "order1.json"
        store_file2 = Path(tmp_dir) / "order2.json"
        s1 = VectorStore(storage_path=str(store_file1))
        s2 = VectorStore(storage_path=str(store_file2))

        doc_match1 = {"id": "match_1", "content": "深度学习大模型架构设计与优化"}
        doc_match2 = {"id": "match_2", "content": "深度学习大模型训练策略与实践"}
        doc_irrelevant1 = {"id": "irr_1", "content": "红烧牛肉面的家常做法与配方"}
        doc_irrelevant2 = {"id": "irr_2", "content": "周末野外露营烧烤实用技巧指南"}

        # 顺序 1：无关文档在前，相关文档在后
        s1.add_chunks([doc_irrelevant1, doc_irrelevant2, doc_match1, doc_match2])
        # 顺序 2：相关文档在前，无关文档在后
        s2.add_chunks([doc_match2, doc_match1, doc_irrelevant2, doc_irrelevant1])

        res1 = s1.hybrid_search("深度学习大模型", top_k=2, require_dense=False)
        res2 = s2.hybrid_search("深度学习大模型", top_k=2, require_dense=False)

        # 两者的召回集合与排序应当完全一致，不受插入顺序影响，无关文档均被排除
        assert [r["id"] for r in res1] == [r["id"] for r in res2]
        assert "irr_1" not in [r["id"] for r in res1]
        assert "irr_2" not in [r["id"] for r in res1]


def test_dense_search_tie_breaking_by_id():
    """验证 dense_search 在候选切片余弦相似度完全相同时，依据 chunk ID 升序确定性破平，杜绝入库顺序偏差"""
    from unittest.mock import patch

    with tempfile.TemporaryDirectory() as tmp_dir:
        store_file1 = Path(tmp_dir) / "dense_tie1.json"
        store_file2 = Path(tmp_dir) / "dense_tie2.json"
        s1 = VectorStore(storage_path=str(store_file1))
        s2 = VectorStore(storage_path=str(store_file2))

        # 两个切片具有完全相同的 embedding 向量 (相似度相同)
        chunk_a = {"id": "chunk_aaa", "content": "内容相同切片 A", "embedding": [1.0, 0.0, 0.0]}
        chunk_b = {"id": "chunk_bbb", "content": "内容相同切片 B", "embedding": [1.0, 0.0, 0.0]}

        # 逆向入库
        s1.chunks = [dict(chunk_b), dict(chunk_a)]
        s2.chunks = [dict(chunk_a), dict(chunk_b)]

        with patch.object(s1.model_provider, "get_embeddings", return_value=[[1.0, 0.0, 0.0]]), \
             patch.object(s2.model_provider, "get_embeddings", return_value=[[1.0, 0.0, 0.0]]):
            r1 = s1.dense_search("测试查询", top_k=2)
            r2 = s2.dense_search("测试查询", top_k=2)

            # 无论切片在库中的先后顺序如何，排序结果必须完全一致（按 ID 升序破平，chunk_aaa 优先）
            assert [c["id"] for c in r1] == ["chunk_aaa", "chunk_bbb"]
            assert [c["id"] for c in r2] == ["chunk_aaa", "chunk_bbb"]


def test_hybrid_search_and_add_chunks_without_explicit_id():
    """验证 add_chunks 和 hybrid_search 在切片缺少 explicit id 字段时能健壮处理，绝不抛出 KeyError"""
    with tempfile.TemporaryDirectory() as tmp_dir:
        store_file = Path(tmp_dir) / "no_id_store.json"
        s = VectorStore(storage_path=str(store_file))

        # 传入缺少 id 的切片字典
        raw_chunks = [
            {"content": "深度学习大模型文风迁移与智能写作技术。"},
            {"content": "红烧牛肉面的家常做法与配方。"},
        ]
        added = s.add_chunks(raw_chunks)
        assert added == 2
        # 验证自动赋予了 id
        assert "id" in s.chunks[0]
        assert "id" in s.chunks[1]

        # 验证 hybrid_search 正常检索出匹配切片
        results = s.hybrid_search("深度学习大模型", top_k=1, require_dense=False)
        assert len(results) == 1
        assert "深度学习" in results[0]["content"]


def test_dense_search_and_hybrid_search_share_identical_dense_channel():
    """验证 P0 缺陷修复：dense_search 与 hybrid_search 共享统一的 _dense_rank_candidates 准入机制与破平规则"""
    from unittest.mock import patch

    with tempfile.TemporaryDirectory() as tmp_dir:
        store_file = Path(tmp_dir) / "shared_dense_store.json"
        store = VectorStore(storage_path=str(store_file))

        # 构造候选切片：doc_pos 相似度为正，doc_zero 相似度为 0，doc_neg 相似度为负
        store.chunks = [
            {"id": "doc_pos", "content": "正向语义相关切片", "embedding": [1.0, 0.0, 0.0]},
            {"id": "doc_zero", "content": "正交零相似切片", "embedding": [0.0, 1.0, 0.0]},
            {"id": "doc_neg", "content": "完全负相关切片", "embedding": [-1.0, 0.0, 0.0]},
        ]

        # Query 向量与 doc_pos 完全重合
        with patch.object(store.model_provider, "get_embeddings", return_value=[[1.0, 0.0, 0.0]]):
            # 1. dense_search 只接纳 score > 0 的切片，doc_zero 和 doc_neg 均被严格过滤
            dense_res = store.dense_search("测试查询", top_k=3)
            assert len(dense_res) == 1
            assert dense_res[0]["id"] == "doc_pos"
            assert dense_res[0]["dense_score"] == 1.0

            # 2. _dense_rank_candidates 返回相同的准入候选集合
            rank_window = max(3 * 3, 20)
            ranked = store._dense_rank_candidates(store.chunks, [1.0, 0.0, 0.0], rank_window=rank_window, require_dense=True)
            assert len(ranked) == 1
            assert ranked[0][1] == "doc_pos"

            # 3. hybrid_search 的 Dense 通道同样只接纳 doc_pos
            hybrid_res = store.hybrid_search("测试查询", top_k=3, require_dense=True)
            assert any(r["id"] == "doc_pos" for r in hybrid_res)
            assert not any(r["id"] == "doc_zero" for r in hybrid_res)
            assert not any(r["id"] == "doc_neg" for r in hybrid_res)


def test_stable_chunk_id_across_rebuilds_eliminates_random_tie_bias():
    """验证 P1 缺陷修复：MemoryManager 与 VectorStore 采用确定性哈希 ID，同一批文章多次建库 ID 完全相同，杜绝破平随机偏差"""
    from src.memory.memory_manager import MemoryManager

    article_title = "思考的深度"
    article_content = """说白了，很多人在互联网上搞的内容输出，本质上不过是高级的信息搬运工。

别闹了。真正的思考从来不是拼图游戏，而是带着偏见的价值判断。

写作这门手艺，最忌讳的就是四平八稳。保持尖锐，这是我们唯一能守住的阵地！"""

    with tempfile.TemporaryDirectory() as tmp_dir:
        s1 = VectorStore(storage_path=str(Path(tmp_dir) / "store1.json"))
        s2 = VectorStore(storage_path=str(Path(tmp_dir) / "store2.json"))
        mgr1 = MemoryManager(vector_store=s1)
        mgr2 = MemoryManager(vector_store=s2)

        mgr1.ingest_article(article_title, article_content)
        mgr2.ingest_article(article_title, article_content)

        assert len(s1.chunks) == len(s2.chunks)
        ids1 = [c["id"] for c in s1.chunks]
        ids2 = [c["id"] for c in s2.chunks]

        # 核心断言：两次跨重建生成的切片 ID 100% 确定性完全一致
        assert ids1 == ids2
        assert all(isinstance(cid, str) and len(cid) == 16 for cid in ids1)


def test_sparse_stopword_filtering_prevents_pseudo_relevance():
    """验证 P1 缺陷修复：高频停用字/词（是、在、工、作、的、人）被有效过滤，防止无关文档仅因虚词重叠产生伪正相关"""
    with tempfile.TemporaryDirectory() as tmp_dir:
        store_file = Path(tmp_dir) / "stopword_store.json"
        store = VectorStore(storage_path=str(store_file))

        # 文档虽然包含“在、是、工、作、的、人”等常见汉字，但与查询的核心实词毫无语义关联
        chunks = [
            {"id": "doc_stopwords_only", "content": "人在不同的工作岗位上是需要付出努力的。", "source": "A"},
            {"id": "doc_semantic_match", "content": "分布式系统架构设计与高并发微服务实战演进。", "source": "B"},
        ]
        store.add_chunks(chunks)

        # 针对包含停用字但核心概念为“架构设计”的查询
        results = store.hybrid_search("在工作中关于架构设计的探讨", top_k=2, require_dense=False)
        result_ids = [r["id"] for r in results]

        # 核心断言：doc_semantic_match 因命中“架构设计”被正确召回，而 doc_stopwords_only 绝不因包含虚词而获得虚假高分或混入
        assert "doc_semantic_match" in result_ids
        assert "doc_stopwords_only" not in result_ids


def test_bm25_corpus_idf_suppresses_pseudo_relevance():
    """验证 P1 缺陷修复：BM25L 与 IDF 算法有效压制非停用词的泛用单字撞车（如'化'、'际'），防止无关文风文档混入检索"""
    with tempfile.TemporaryDirectory() as tmp_dir:
        store_file = Path(tmp_dir) / "bm25_store.json"
        store = VectorStore(storage_path=str(store_file))

        # 构造两篇切片：一篇为餐饮相关，另一篇为包含'口语化'、'实际上'等单字'化'、'际'但与餐饮毫无关系的文风文章
        chunks = [
            {"id": "doc_catering", "content": "现代都市连锁餐饮与外卖快节奏生活变迁，外卖标准化重塑大众餐饮习惯。"},
            {"id": "doc_writing_style", "content": "我们的中文互联网充斥着一种极其恶劣的文风。实际上，工具越是强大，人类越要警惕被机器同化。保持口语化。"},
        ]
        store.add_chunks(chunks)

        # 查询包含多词搭配与二元词组
        results = store.hybrid_search("现代都市便利化餐饮：快节奏生活下的餐饮形态与人际社交变迁", top_k=2, require_dense=False)
        result_ids = [r["id"] for r in results]

        # 核心断言：doc_catering 凭借高 IDF 的'餐饮'、'现代都市'、'快节奏'等二元词命中并排在首位
        # doc_writing_style 绝不能因为仅靠单字'化'、'际'弱匹配而进入 Sparse ranking
        assert "doc_catering" in result_ids
        assert "doc_writing_style" not in result_ids


def test_chinese_trigram_and_short_phrase_tokens_are_weighted():
    tokens = VectorStore._tokenize("非厄米谱结构")

    assert tokens["非厄米"] > tokens["非厄"] > tokens["非"]
    assert tokens["非厄米谱结构"] > 0
    assert not VectorStore._is_chinese_ngram("architecture")
    assert VectorStore._is_chinese_ngram("谱结构")


def test_discourse_classifier_uses_deterministic_scoring():
    from src.memory.memory_manager import MemoryManager

    assert MemoryManager._classify_function("你有没有想过，问题究竟在哪里？", 0, 5, "opening") == "hook"
    assert MemoryManager._classify_function("比如，这个案例展示了具体路径。", 2, 5, "body") == "example"
    assert MemoryManager._classify_function("说白了，判断从来不是信息搬运！", 2, 5, "body") == "quote"
    assert MemoryManager._classify_function("中间部分展开机制与边界条件。", 2, 5, "body") == "argument"
    assert MemoryManager._classify_function("归根结底，这是我们唯一能守住的原则。", 4, 5, "ending") == "conclusion"


def test_add_chunks_with_none_id_generates_stable_id():
    """验证 P1 缺陷修复：当切片显式传入 id 为 None 时，正确自动分配确定性稳定哈希 ID，杜绝 id 塌缩为 'None' 导致切片覆盖或破平失效"""
    with tempfile.TemporaryDirectory() as tmp_dir:
        store_file = Path(tmp_dir) / "none_id_store.json"
        store = VectorStore(storage_path=str(store_file))

        # 传入两个 id 为 None 的切片
        chunks = [
            {"id": None, "content": "第一篇测试切片，内容关于系统高并发架构设计。"},
            {"id": None, "content": "第二篇测试切片，内容关于系统分布式存储演进。"},
        ]
        store.add_chunks(chunks)

        # 核心断言 1：每个切片均获得有效非空的确定性稳定哈希 ID，绝不为 None 或 'None'
        assert len(store.chunks) == 2
        id1 = store.chunks[0]["id"]
        id2 = store.chunks[1]["id"]
        assert id1 is not None and id1 != "None" and len(id1) == 16
        assert id2 is not None and id2 != "None" and len(id2) == 16
        assert id1 != id2

        # 核心断言 2：hybrid_search 能分别索引并返回它们，字典键绝不会相互覆盖
        results = store.hybrid_search("系统", top_k=2, require_dense=False)
        assert len(results) == 2
        returned_ids = [r["id"] for r in results]
        assert id1 in returned_ids
        assert id2 in returned_ids


def test_dense_search_require_dense_parameter():
    """验证 dense_search 与 hybrid_search 对称支持 require_dense 参数"""
    from unittest.mock import patch
    import pytest
    from src.core.exceptions import EmbeddingUnavailableError

    with tempfile.TemporaryDirectory() as tmp_dir:
        store_file = Path(tmp_dir) / "dense_param_store.json"
        store = VectorStore(storage_path=str(store_file))
        store.chunks = [{"id": "d1", "content": "内容", "embedding": None}]

        with patch.object(store.model_provider, "get_embeddings", return_value=None):
            # require_dense=True 时 Fail-Closed
            with pytest.raises(EmbeddingUnavailableError):
                store.dense_search("查询", top_k=1, require_dense=True)

            # require_dense=False 时返回空列表，不抛出异常
            res = store.dense_search("查询", top_k=1, require_dense=False)
            assert res == []


def test_hybrid_search_falls_back_to_sparse_when_embedding_provider_is_down():
    from unittest.mock import patch

    import pytest

    from src.core.exceptions import EmbeddingUnavailableError

    with tempfile.TemporaryDirectory() as tmp_dir:
        store = VectorStore(storage_path=str(Path(tmp_dir) / "fallback_store.json"))
        store.chunks = [
            {
                "id": "matching",
                "content": "边界条件决定非厄米系统的谱结构。",
                "embedding": [1.0, 0.0],
            },
            {
                "id": "unrelated",
                "content": "红烧牛肉面的家常做法。",
                "embedding": [0.0, 1.0],
            },
        ]

        outage = EmbeddingUnavailableError("provider down")
        with patch.object(store.model_provider, "get_embeddings", side_effect=outage):
            results = store.hybrid_search("非厄米谱结构", top_k=1, require_dense=False)
            assert [item["id"] for item in results] == ["matching"]

        with patch.object(store.model_provider, "get_embeddings", side_effect=outage):
            with pytest.raises(EmbeddingUnavailableError, match="provider down"):
                store.hybrid_search("非厄米谱结构", top_k=1, require_dense=True)
