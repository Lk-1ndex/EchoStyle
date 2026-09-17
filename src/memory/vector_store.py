import json
import math
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from src.core.config import EmbeddingConfig, LLMConfig
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
        用于消融实验 Standard Semantic RAG (Condition C1) 严格控制变量。
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

        if query_dense and any(c.get("embedding") is not None for c in candidates):
            dense_scores = []
            for c in candidates:
                emb = c.get("embedding")
                score = self._cosine_similarity(query_dense, emb) if emb else 0.0
                dense_scores.append((score, c))
            dense_scores.sort(key=lambda x: x[0], reverse=True)
            results = []
            for score, c in dense_scores[:top_k]:
                item = dict(c)
                item["dense_score"] = round(score, 6)
                results.append(item)
            return results
        else:
            return candidates[:top_k]

    def hybrid_search(
        self,
        query: str,
        top_k: int = 3,
        type_filter: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        标准 RRF (Reciprocal Rank Fusion) 倒数排名融合检索
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

        dense_ranks: Dict[str, int] = {}
        if query_dense and any(c.get("embedding") is not None for c in candidates):
            dense_scores = []
            for c in candidates:
                emb = c.get("embedding")
                score = self._cosine_similarity(query_dense, emb) if emb else 0.0
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
    def _cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
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
