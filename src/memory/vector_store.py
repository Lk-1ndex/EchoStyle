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
    Hybrid RAG 混合检索记忆库：
    结合 Dense Embedding 深度语义向量 (0.65) 与 Sparse BM25/N-Gram 词汇精准度 (0.35)，
    并支持按段落类型 (hook/argument/quote/conclusion) 与主题进行 Metadata 过滤。
    """

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

    def hybrid_search(
        self,
        query: str,
        top_k: int = 3,
        type_filter: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Hybrid 混合检索：Dense 语义相似度 + Sparse 稀疏词汇相似度
        """
        if not self.chunks:
            return []

        # 候选过滤
        candidates = self.chunks
        if type_filter:
            candidates = [c for c in candidates if c.get("metadata", {}).get("type") == type_filter]
            if not candidates:
                candidates = self.chunks  # 降级回退

        query_embs = self.model_provider.get_embeddings([query])
        query_dense = query_embs[0] if query_embs else None
        query_tokens = self._tokenize(query)

        scored_candidates: List[Tuple[float, Dict[str, Any]]] = []

        for c in candidates:
            # 1. 密集向量得分 (Dense Score)
            dense_score = 0.0
            if query_dense and c.get("embedding"):
                dense_score = max(0.0, self._cosine_similarity(query_dense, c["embedding"]))

            # 2. 稀疏词汇得分 (Sparse Score)
            sparse_score = self._sparse_similarity(query_tokens, self._tokenize(c["content"]))

            # 3. 加权混合计算
            if query_dense:
                hybrid_score = 0.65 * dense_score + 0.35 * sparse_score
            else:
                hybrid_score = sparse_score  # 无向量 API 时纯走本地 BM25/TF-IDF

            scored_candidates.append((hybrid_score, c))

        # 降序排列
        scored_candidates.sort(key=lambda x: x[0], reverse=True)
        return [c for score, c in scored_candidates[:top_k]]

    def search(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """保持接口兼容的标准搜索"""
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
