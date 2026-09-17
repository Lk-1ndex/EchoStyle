import json
import math
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from src.core.config import EmbeddingConfig, LLMConfig
from src.core.exceptions import EmbeddingUnavailableError, EmbeddingDimensionMismatchError
from src.core.model_provider import ModelProvider


class VectorStore:
    """
    工业级 RRF (Reciprocal Rank Fusion) 混合检索记忆库：
    采用 SIGIR 标准倒数排名融合算法，消除密集向量与稀疏词频的数值量纲偏差：
    RRF(d) = 1 / (60 + rank_dense(d)) + 1 / (60 + rank_sparse(d))
    """

    RRF_K: int = 60  # RRF 经典平滑常数

    def __init__(
        self,
        storage_path: str = "./profiles/style_memory.json",
        embedding_config: Optional[EmbeddingConfig] = None,
        llm_config: Optional[LLMConfig] = None,
    ):
        self.storage_path = Path(storage_path)
        self.embedding_config = embedding_config or EmbeddingConfig()
        self.llm_config = llm_config or LLMConfig()
        self.model_provider = ModelProvider(self.llm_config, self.embedding_config)
        self.chunks: List[Dict[str, Any]] = []
        self._load()

    def add_chunks(self, chunks: List[Dict[str, Any]]) -> int:
        if not chunks:
            return 0

        # 获取向量
        embeddings = self.model_provider.get_embeddings([c["content"] for c in chunks])
        if embeddings is not None:
            # 严格实验/生产环境校验：若向量接口返回非空，必须完整且维度一致
            if len(embeddings) != len(chunks):
                raise EmbeddingUnavailableError(
                    f"向量生成数量不匹配 (Fail-Closed): 期望 {len(chunks)} 个向量，实际返回 {len(embeddings)} 个。"
                )
            if any(e is None for e in embeddings):
                raise EmbeddingUnavailableError(
                    "向量服务返回了部分 None 空向量，拒绝残缺向量入库 (Fail-Closed)！"
                )
            first_dim = len(embeddings[0]) if embeddings else 0
            for i, emb in enumerate(embeddings):
                if len(emb) != first_dim:
                    raise EmbeddingDimensionMismatchError(
                        query_dim=first_dim, chunk_dim=len(emb), chunk_id=chunks[i].get("id")
                    )

            # 跨批次维度一致性强校验：若已有切片具有向量，新入库向量必须与已有向量维度严格相等
            existing_with_emb = [c for c in self.chunks if c.get("embedding") is not None]
            if existing_with_emb and embeddings:
                existing_dim = len(existing_with_emb[0]["embedding"])
                if first_dim != existing_dim:
                    raise EmbeddingDimensionMismatchError(
                        query_dim=existing_dim, chunk_dim=first_dim, chunk_id=chunks[0].get("id")
                    )

        added = 0
        for i, chunk in enumerate(chunks):
            chunk["embedding"] = embeddings[i] if embeddings and i < len(embeddings) else None
            self.chunks.append(chunk)
            added += 1

        self._save()
        return added

    def dense_search(
        self,
        query: str,
        top_k: int = 3,
        type_filter: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        纯密集向量语义检索 (Dense Semantic Search)：
        仅依据稠密向量余弦相似度排序，不执行稀疏词频 (Sparse/BM25) 统计，不经过 RRF 倒数排名融合。
        用于消融实验 Standard Semantic RAG (Condition C1a) 严格控制变量。
        Fail-Closed 机制：若无法获取 query 向量或候选切片缺少 embedding 向量，直接抛出异常，严禁静默退化！
        """
        if not self.chunks:
            return []

        candidates = self.chunks
        if type_filter:
            candidates = [
                c for c in candidates
                if c.get("metadata", {}).get("type") == type_filter or c.get("type") == type_filter
            ]
            if not candidates:
                return []

        query_embs = self.model_provider.get_embeddings([query])
        query_dense = query_embs[0] if query_embs else None

        if not query_dense:
            raise EmbeddingUnavailableError(
                "Dense 语义检索不可用: 无法获取 query 向量。"
                "基准消融实验场景下严格禁止静默退化为未排序切片 (Fail-Closed)！"
            )

        # 严格 Fail-Closed：要求所有候选切片均必须具有有效向量，严禁 partial None 造成静默不公平排序
        if not all(c.get("embedding") is not None for c in candidates):
            missing_ids = [c.get("id", "unknown") for c in candidates if c.get("embedding") is None]
            raise EmbeddingUnavailableError(
                f"Dense 语义检索不可用: 候选切片存在部分缺少 embedding 向量的情况 (缺失切片: {missing_ids})。"
                "基准消融实验场景下严格禁止部分切片参与排序 (Fail-Closed)！"
            )

        dense_scores = []
        for c in candidates:
            emb = c["embedding"]
            score = self._cosine_similarity(query_dense, emb, chunk_id=c.get("id"))
            dense_scores.append((score, c))
        dense_scores.sort(key=lambda x: x[0], reverse=True)
        results = []
        for score, c in dense_scores[:top_k]:
            item = dict(c)
            item["dense_score"] = round(score, 6)
            results.append(item)
        return results

    def hybrid_search(
        self,
        query: str,
        top_k: int = 3,
        type_filter: Optional[str] = None,
        require_dense: bool = False,
    ) -> List[Dict[str, Any]]:
        """
        标准 RRF (Reciprocal Rank Fusion) 倒数排名融合检索
        :param require_dense: 若为 True (严格基准对照模式)，当 Dense 向量不可用时直接抛出 EmbeddingUnavailableError，
                              拒绝单通道降级污染实验条件；生产普通模式下为 False，允许优雅降级为单通道稀疏排序。
        """
        if not self.chunks:
            return []

        # 候选集类型过滤
        candidates = self.chunks
        if type_filter:
            candidates = [
                c for c in candidates
                if c.get("metadata", {}).get("type") == type_filter or c.get("type") == type_filter
            ]
            if not candidates:
                return []

        # 1. 密集向量检索通道 (Dense Retrieval)
        query_embs = self.model_provider.get_embeddings([query])
        query_dense = query_embs[0] if query_embs else None
        has_chunk_embs = any(c.get("embedding") is not None for c in candidates)
        all_chunk_embs = all(c.get("embedding") is not None for c in candidates)

        if require_dense:
            if not query_dense:
                raise EmbeddingUnavailableError(
                    "Hybrid 检索 Dense 通道不可用: 无法获取 query 向量。"
                    "严格基准对照模式下拒绝退化为纯稀疏排序 (Fail-Closed)！"
                )
            if not all_chunk_embs:
                missing_ids = [c.get("id", "unknown") for c in candidates if c.get("embedding") is None]
                raise EmbeddingUnavailableError(
                    f"Hybrid 检索 Dense 通道不可用: 候选切片存在部分缺少 embedding 向量的情况 (缺失切片: {missing_ids})。"
                    "严格基准对照模式下拒绝退化为纯稀疏排序 (Fail-Closed)！"
                )

        dense_ranks: Dict[str, int] = {}
        if query_dense and has_chunk_embs:
            dense_scores = []
            for c in candidates:
                emb = c.get("embedding")
                if emb is not None:
                    score = self._cosine_similarity(query_dense, emb, chunk_id=c.get("id"))
                    dense_scores.append((score, c["id"]))
            # 按相似度降序排序，赋予绝对排名 (1-indexed)
            dense_scores.sort(key=lambda x: x[0], reverse=True)
            for rank_idx, (_, cid) in enumerate(dense_scores, start=1):
                dense_ranks[cid] = rank_idx


        # 2. 稀疏词汇检索通道 (Sparse Retrieval - Token Overlap / BM25 变体)
        query_tokens = self._tokenize(query)
        sparse_scores = []
        for c in candidates:
            c_tokens = self._tokenize(c["content"])
            score = self._sparse_similarity(query_tokens, c_tokens)
            sparse_scores.append((score, c["id"]))
        # 按稀疏分数降序排序，赋予绝对排名
        sparse_scores.sort(key=lambda x: x[0], reverse=True)
        sparse_ranks: Dict[str, int] = {}
        for rank_idx, (_, cid) in enumerate(sparse_scores, start=1):
            sparse_ranks[cid] = rank_idx

        # 3. 执行 RRF 排名倒数融合计算
        rrf_scores: List[Tuple[float, Dict[str, Any]]] = []
        for c in candidates:
            cid = c["id"]
            dense_rank = dense_ranks.get(cid)
            sparse_rank = sparse_ranks.get(cid, len(candidates) + 1)

            # 标准公式: 1 / (60 + rank_dense) + 1 / (60 + rank_sparse)
            rrf_val = 0.0
            if dense_rank is not None:
                rrf_val += 1.0 / (self.RRF_K + dense_rank)
            rrf_val += 1.0 / (self.RRF_K + sparse_rank)

            c["rrf_score"] = round(rrf_val, 6)
            rrf_scores.append((rrf_val, c))

        # 按 RRF 得分从高到低排列
        rrf_scores.sort(key=lambda x: x[0], reverse=True)
        return [c for score, c in rrf_scores[:top_k]]

    def search(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        return self.hybrid_search(query, top_k=top_k)

    def get_all(self) -> List[Dict[str, Any]]:
        return self.chunks

    def clear(self):
        self.chunks = []
        if self.storage_path.exists():
            self.storage_path.unlink()

    def _load(self):
        if self.storage_path.exists():
            try:
                with open(self.storage_path, "r", encoding="utf-8") as f:
                    self.chunks = json.load(f)
            except Exception:
                self.chunks = []

    def _save(self):
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.storage_path, "w", encoding="utf-8") as f:
            json.dump(self.chunks, f, ensure_ascii=False, indent=2)

    @staticmethod
    def _cosine_similarity(vec1: List[float], vec2: List[float], chunk_id: Optional[str] = None) -> float:
        if len(vec1) != len(vec2):
            raise EmbeddingDimensionMismatchError(
                query_dim=len(vec1),
                chunk_dim=len(vec2),
                chunk_id=chunk_id,
            )
        dot = sum(a * b for a, b in zip(vec1, vec2))
        norm1 = math.sqrt(sum(a * a for a in vec1))
        norm2 = math.sqrt(sum(b * b for b in vec2))
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return dot / (norm1 * norm2)

    @staticmethod
    def _tokenize(text: str) -> Dict[str, int]:
        text = text.lower().strip()
        tokens: Dict[str, int] = {}
        words = re.findall(r"[\u4e00-\u9fa5]{1,2}|[a-zA-Z0-9]+", text)
        for w in words:
            tokens[w] = tokens.get(w, 0) + 1
        return tokens

    @staticmethod
    def _sparse_similarity(tf1: Dict[str, int], tf2: Dict[str, int]) -> float:
        if not tf1 or not tf2:
            return 0.0
        common_keys = set(tf1.keys()) & set(tf2.keys())
        dot = sum(tf1[k] * tf2[k] for k in common_keys)
        norm1 = math.sqrt(sum(v * v for v in tf1.values()))
        norm2 = math.sqrt(sum(v * v for v in tf2.values()))
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return dot / (norm1 * norm2)
