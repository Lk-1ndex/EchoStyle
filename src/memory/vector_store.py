import hashlib
import json
import math
import os
import re
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from filelock import FileLock
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
        storage_path: str = "./profiles/style_memory_v2.json",
        embedding_config: Optional[EmbeddingConfig] = None,
        llm_config: Optional[LLMConfig] = None,
    ):
        self.storage_path = Path(storage_path)
        self.embedding_config = embedding_config or EmbeddingConfig()
        self.llm_config = llm_config or LLMConfig()
        self.model_provider = ModelProvider(self.llm_config, self.embedding_config)
        self.chunks: List[Dict[str, Any]] = []
        self._file_lock = FileLock(f"{self.storage_path}.lock", timeout=60)
        self._loaded_signature: Optional[Tuple[int, int, int]] = None
        self._load()

    def add_chunks(self, chunks: List[Dict[str, Any]]) -> int:
        if not chunks:
            return 0

        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        with self._file_lock:
            self._refresh_from_disk(locked=True)
            existing = {
                (self._profile_id(c), str(c["id"])): c
                for c in self.chunks if c.get("id")
            }
            new_chunks = []
            for raw_chunk in chunks:
                chunk = dict(raw_chunk)
                if not chunk.get("id"):
                    meta = chunk.get("metadata") or {}
                    source = meta.get("source", chunk.get("source", ""))
                    pos_idx = meta.get("position_pct", chunk.get("paragraph_index", 0))
                    stable_seed = f"{source}_{pos_idx}_{chunk.get('content', '')}"
                    chunk["id"] = hashlib.sha256(stable_seed.encode("utf-8")).hexdigest()[:16]
                key = (self._profile_id(chunk), str(chunk["id"]))
                if key in existing:
                    if existing[key].get("content") != chunk.get("content"):
                        raise ValueError(f"切片 ID 冲突且内容不同: {key}")
                    continue
                existing[key] = chunk
                new_chunks.append(chunk)

            if not new_chunks:
                return 0

            embeddings = self.model_provider.get_embeddings([c["content"] for c in new_chunks])
            if embeddings is not None:
                if len(embeddings) != len(new_chunks):
                    raise EmbeddingUnavailableError(
                        f"向量生成数量不匹配 (Fail-Closed): 期望 {len(new_chunks)} 个向量，实际返回 {len(embeddings)} 个。"
                    )
                if any(e is None for e in embeddings):
                    raise EmbeddingUnavailableError("向量服务返回了部分 None 空向量，拒绝残缺向量入库 (Fail-Closed)！")
                first_dim = len(embeddings[0])
                for i, emb in enumerate(embeddings):
                    if len(emb) != first_dim:
                        raise EmbeddingDimensionMismatchError(
                            query_dim=first_dim, chunk_dim=len(emb), chunk_id=new_chunks[i]["id"]
                        )
                for chunk in new_chunks:
                    existing_with_emb = next(
                        (c for c in self.chunks if self._profile_id(c) == self._profile_id(chunk) and c.get("embedding") is not None),
                        None,
                    )
                    if existing_with_emb and len(existing_with_emb["embedding"]) != first_dim:
                        raise EmbeddingDimensionMismatchError(
                            query_dim=len(existing_with_emb["embedding"]), chunk_dim=first_dim, chunk_id=chunk["id"]
                        )

            for i, chunk in enumerate(new_chunks):
                chunk["embedding"] = embeddings[i] if embeddings is not None else None
            self.chunks.extend(new_chunks)
            try:
                self._save()
            except Exception:
                self._load()
                raise
            return len(new_chunks)

    @staticmethod
    def _profile_id(chunk: Dict[str, Any]) -> Optional[str]:
        return (chunk.get("metadata") or {}).get("profile_id")

    def _candidates(self, profile_id: Optional[str], type_filter: Optional[str]) -> List[Dict[str, Any]]:
        self._refresh_from_disk()
        return [
            c for c in self.chunks
            if self._profile_id(c) == profile_id
            and (not type_filter or (c.get("metadata") or {}).get("type") == type_filter or c.get("type") == type_filter)
        ]

    def _dense_rank_candidates(
        self,
        candidates: List[Dict[str, Any]],
        query_dense: Optional[List[float]],
        rank_window: int,
        require_dense: bool = False,
        channel_name: str = "Dense 语义检索",
    ) -> List[Tuple[float, str, Dict[str, Any]]]:
        """
        统一密集向量候选排序通道：
        为 C1a (dense_search) 与 C1b (hybrid_search) 提供完全一致的候选集准入规则、
        余弦相似度计算、正相关过滤 (score > 0)、确定性破平与 Top-N 候选窗口截取。
        """
        has_chunk_embs = any(c.get("embedding") is not None for c in candidates)
        all_chunk_embs = all(c.get("embedding") is not None for c in candidates)

        if require_dense:
            if not query_dense:
                suffix = "严格基准对照模式下拒绝退化为纯稀疏排序 (Fail-Closed)！" if "Hybrid" in channel_name else "基准消融实验场景下严格禁止静默退化为未排序切片 (Fail-Closed)！"
                raise EmbeddingUnavailableError(
                    f"{channel_name}不可用: 无法获取 query 向量。{suffix}"
                )
            if not all_chunk_embs:
                missing_ids = [c.get("id", "unknown") for c in candidates if c.get("embedding") is None]
                suffix = "严格基准对照模式下拒绝退化为纯稀疏排序 (Fail-Closed)！" if "Hybrid" in channel_name else "基准消融实验场景下严格禁止部分切片参与排序 (Fail-Closed)！"
                raise EmbeddingUnavailableError(
                    f"{channel_name}不可用: 候选切片存在部分缺少 embedding 向量的情况 (缺失切片: {missing_ids})。{suffix}"
                )

        if not query_dense or not has_chunk_embs:
            return []

        dense_scores = []
        for idx, c in enumerate(candidates):
            emb = c.get("embedding")
            cid = str(c.get("id") or idx)
            if emb is not None:
                score = self._cosine_similarity(query_dense, emb, chunk_id=cid)
                if score > 0:
                    dense_scores.append((score, cid, c))

        # 确定性破平：按相似度降序排序，得分相同时按 chunk ID 升序破平，杜绝入库顺序偏差
        dense_scores.sort(key=lambda x: (-x[0], str(x[1])))
        return dense_scores[:rank_window]

    def dense_search(
        self,
        query: str,
        top_k: int = 3,
        type_filter: Optional[str] = None,
        require_dense: bool = True,
        profile_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        纯密集向量语义检索 (Dense Semantic Search - Condition C1a)：
        与 hybrid_search 中的 Dense 通道完全共享同一套 score threshold (score > 0)、
        确定性破平规则与 rank_window 准入机制 (_dense_rank_candidates)，
        直接截取其中 top_k 结果，确保 C1a 到 C1b 的消融差异纯粹为 Sparse + RRF 融合效应。
        Fail-Closed 机制：若 require_dense=True 且无法获取 query 向量或候选切片缺少 embedding 向量，直接抛出异常，严禁静默退化！
        """
        candidates = self._candidates(profile_id, type_filter)
        if not candidates:
            return []

        query_dense = self._query_embedding(query, require_dense=require_dense)

        rank_window = max(top_k * 3, 20)
        ranked = self._dense_rank_candidates(
            candidates=candidates,
            query_dense=query_dense,
            rank_window=rank_window,
            require_dense=require_dense,
            channel_name="Dense 语义检索",
        )

        results = []
        for score, cid, c in ranked[:top_k]:
            item = dict(c)
            item["dense_score"] = round(score, 6)
            results.append(item)
        return results

    def _bm25_search_candidates(
        self,
        candidates: List[Dict[str, Any]],
        query: str,
        rank_window: int,
    ) -> List[Tuple[float, str]]:
        """
        基于 BM25L 算法与 Robertson-Spärck Jones 非负 IDF 的稀疏检索通道：
        1. 过滤高频虚词与停用词；
        2. 中文二元词组 (Bigram) 2.0x 加权优先保障搭配语义，一元字 (Unigram) 降权抑制字频假阳性；
        3. 计算语料库级 IDF 惩罚泛用词，并通过 BM25L 饱和项 (delta=0.5) 避免长文档过罚；
        4. 门控机制：当 Query 包含二元词搭配时，拒绝仅凭单一孤立单字微弱重叠的无关文档准入 (防止伪正相关)；
        5. 确定性破平：得分相同时按 chunk ID 升序破平，杜绝入库顺序偏差。
        """
        q_tokens = self._tokenize(query)
        if not q_tokens or not candidates:
            return []

        cand_tokens = [self._tokenize(c.get("content", "")) for c in candidates]
        doc_lens = [sum(t.values()) for t in cand_tokens]
        N = len(candidates)
        avgdl = sum(doc_lens) / max(1, N)
        has_query_bigrams = any(len(t) >= 2 for t in q_tokens)

        scores: List[Tuple[float, str]] = []
        k1 = 1.5
        b = 0.75
        delta = 0.5  # BM25L 下界修正参数

        for idx, c in enumerate(candidates):
            cid = str(c.get("id") or idx)
            c_tok = cand_tokens[idx]
            dlen = doc_lens[idx]
            if dlen == 0:
                continue

            score = 0.0
            matched_bigrams = 0
            for t, q_w in q_tokens.items():
                if t in c_tok:
                    if len(t) >= 2:
                        matched_bigrams += 1
                    df = sum(1 for tok in cand_tokens if t in tok)
                    idf = math.log((N - df + 0.5) / (df + 0.5) + 1.0)
                    tf = c_tok[t]
                    denom = 1.0 - b + b * (dlen / avgdl) if avgdl > 0 else 1.0
                    tf_norm = tf / denom if denom > 0 else tf
                    bm25_term = (tf_norm + delta) / (k1 + tf_norm + delta)
                    score += q_w * idf * bm25_term

            # 伪正相关抑制门控：若查询包含二元词，但切片未命中任何二元词，仅靠孤立单字弱相关者拒绝准入
            if has_query_bigrams and matched_bigrams == 0:
                score = 0.0

            if score > 0:
                scores.append((round(score, 6), cid))

        scores.sort(key=lambda x: (-x[0], str(x[1])))
        return scores[:rank_window]

    def hybrid_search(
        self,
        query: str,
        top_k: int = 3,
        type_filter: Optional[str] = None,
        require_dense: bool = False,
        profile_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        标准 RRF (Reciprocal Rank Fusion) 倒数排名融合检索 (Condition C1b)
        :param require_dense: 若为 True (严格基准对照模式)，当 Dense 向量不可用时直接抛出 EmbeddingUnavailableError，
                              拒绝单通道降级污染实验条件；生产普通模式下为 False，允许优雅降级为单通道稀疏排序。
        """
        candidates = self._candidates(profile_id, type_filter)
        if not candidates:
            return []

        rank_window = max(top_k * 3, 20)

        # 1. 密集向量检索通道 (Dense Retrieval) - 共享 _dense_rank_candidates 准入与排序
        query_dense = self._query_embedding(query, require_dense=require_dense)

        dense_ranked = self._dense_rank_candidates(
            candidates=candidates,
            query_dense=query_dense,
            rank_window=rank_window,
            require_dense=require_dense,
            channel_name="Hybrid 检索 Dense 通道",
        )
        dense_ranks: Dict[str, int] = {}
        for rank_idx, (score, cid, c) in enumerate(dense_ranked, start=1):
            dense_ranks[cid] = rank_idx

        # 2. 稀疏词汇检索通道 (Sparse Retrieval - BM25L + IDF)
        # 严格过滤 score > 0 的有效文档，仅截取 top_N 赋予稀疏绝对排名
        sparse_scores = self._bm25_search_candidates(candidates, query, rank_window)
        sparse_ranks: Dict[str, int] = {}
        for rank_idx, (_, cid) in enumerate(sparse_scores, start=1):
            sparse_ranks[cid] = rank_idx

        # 3. 执行 RRF 排名倒数融合计算
        # 仅对在至少一个检索通道进入 top_N 有效候选集的切片执行 RRF 融合计算
        # 零相关文档与未入围文档绝对不参与 RRF 融合，更绝不进入最终召回列表，彻底根除入库顺序偏差 (Insertion-Order Bias)
        candidate_cids = set(dense_ranks.keys()) | set(sparse_ranks.keys())
        if not candidate_cids:
            return []

        cid_to_chunk = {str(c.get("id") or idx): c for idx, c in enumerate(candidates)}
        rrf_scores: List[Tuple[float, Dict[str, Any]]] = []
        for cid in candidate_cids:
            c = cid_to_chunk.get(cid)
            if not c:
                continue
            dense_rank = dense_ranks.get(cid)
            sparse_rank = sparse_ranks.get(cid)

            # 标准公式: 1 / (60 + rank_dense) + 1 / (60 + rank_sparse)
            rrf_val = 0.0
            if dense_rank is not None:
                rrf_val += 1.0 / (self.RRF_K + dense_rank)
            if sparse_rank is not None:
                rrf_val += 1.0 / (self.RRF_K + sparse_rank)

            item = dict(c)
            item["rrf_score"] = round(rrf_val, 6)
            rrf_scores.append((rrf_val, item))

        # 按 RRF 得分降序排列，得分相同者按 id 确定性升序排序（彻底根除依赖列表入库顺序造成的隐式偏置）
        rrf_scores.sort(key=lambda x: (-x[0], str(x[1].get("id", ""))))
        return [c for score, c in rrf_scores[:top_k]]

    def _query_embedding(self, query: str, require_dense: bool) -> Optional[List[float]]:
        """Fetch a query vector, preserving sparse fallback in production mode."""
        try:
            query_embs = self.model_provider.get_embeddings([query])
        except EmbeddingUnavailableError:
            if require_dense:
                raise
            return None
        return query_embs[0] if query_embs else None


    def search(self, query: str, top_k: int = 3, profile_id: Optional[str] = None) -> List[Dict[str, Any]]:
        return self.hybrid_search(query, top_k=top_k, profile_id=profile_id)

    def get_all(self, profile_id: Optional[str] = None) -> List[Dict[str, Any]]:
        return list(self._candidates(profile_id, type_filter=None))

    def clear(self, profile_id: Optional[str] = None):
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        with self._file_lock:
            self._refresh_from_disk(locked=True)
            remaining = [c for c in self.chunks if self._profile_id(c) != profile_id]
            if len(remaining) == len(self.chunks):
                return
            self.chunks = remaining
            self._save()

    def _load(self):
        if self.storage_path.exists():
            with open(self.storage_path, "r", encoding="utf-8") as f:
                stat = os.fstat(f.fileno())
                loaded = json.load(f)
            if not isinstance(loaded, list) or any(not isinstance(c, dict) for c in loaded):
                raise ValueError(f"记忆库格式损坏: {self.storage_path}")
            self.chunks = loaded
            self._loaded_signature = (stat.st_ino, stat.st_size, stat.st_mtime_ns)
        else:
            self.chunks = []
            self._loaded_signature = None

    def _disk_signature(self) -> Optional[Tuple[int, int, int]]:
        try:
            stat = self.storage_path.stat()
        except FileNotFoundError:
            return None
        return (stat.st_ino, stat.st_size, stat.st_mtime_ns)

    def _refresh_from_disk(self, locked: bool = False):
        if locked:
            self._load()
        elif self._disk_signature() != self._loaded_signature:
            with self._file_lock:
                if self._disk_signature() != self._loaded_signature:
                    self._load()

    def _save(self):
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = None
        try:
            with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=self.storage_path.parent, suffix=".tmp", delete=False) as f:
                temp_path = Path(f.name)
                json.dump(self.chunks, f, ensure_ascii=False, indent=2)
                f.flush()
                os.fsync(f.fileno())
            os.replace(temp_path, self.storage_path)
            self._loaded_signature = self._disk_signature()
        finally:
            if temp_path is not None and temp_path.exists():
                temp_path.unlink()

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

    CHINESE_STOPWORDS = {
        "的", "了", "在", "是", "我", "有", "和", "就", "不", "人", "都", "一", "一个",
        "上", "也", "很", "到", "说", "要", "去", "你", "会", "着", "没有", "看",
        "好", "自己", "这", "这个", "那", "那个", "个", "与", "及", "等", "为", "之", "以", "其",
        "并", "而", "或", "于", "中", "把", "让", "给", "从", "对", "但", "向",
        "被", "更", "又", "过", "得", "它", "他", "她", "吗", "呢", "吧", "啊",
        "做", "作", "工", "按", "由", "此", "该", "各", "每", "所", "将",
        "即", "便", "若", "虽", "因", "故", "且", "乃", "只", "某", "如何", "怎么",
        "什么", "哪", "哪里", "为什么", "怎样", "多少", "无论", "不管", "虽然", "但是",
        "因为", "所以", "如果", "不仅", "而且", "不仅如此", "与此同时", "此外", "另外",
        "这种", "那种", "这样", "那样", "以及", "从而", "通过", "进行", "关于", "对于"
    }
    ENGLISH_STOPWORDS = {
        "a", "an", "the", "in", "on", "at", "to", "is", "are", "of", "and", "or",
        "for", "with", "by", "as", "from", "it", "this", "that"
    }

    @classmethod
    def _tokenize(cls, text: str) -> Dict[str, float]:
        text = text.lower().strip()
        tokens: Dict[str, float] = {}
        # 英文与数字单词（过滤停用词与单字符）
        for w in re.findall(r"[a-zA-Z0-9]+", text):
            if w not in cls.ENGLISH_STOPWORDS and len(w) > 1:
                tokens[w] = tokens.get(w, 0.0) + 1.0
        # 中文短语提取（按标点符号断句，避免跨句生成无效 bigram）
        cn_phrases = re.findall(r"[\u4e00-\u9fa5]+", text)
        for phrase in cn_phrases:
            if len(phrase) == 1:
                if phrase[0] not in cls.CHINESE_STOPWORDS:
                    tokens[phrase[0]] = tokens.get(phrase[0], 0.0) + 0.5
            else:
                # 1-gram 单字降权至 0.3，抑制高频单字伪正相关
                for ch in phrase:
                    if ch not in cls.CHINESE_STOPWORDS:
                        tokens[ch] = tokens.get(ch, 0.0) + 0.3
                # 2-gram 重叠二元字组赋权 2.0，优先保障词汇搭配与实体语义
                for i in range(len(phrase) - 1):
                    bg = phrase[i : i + 2]
                    if bg[0] in cls.CHINESE_STOPWORDS and bg[1] in cls.CHINESE_STOPWORDS:
                        continue
                    tokens[bg] = tokens.get(bg, 0.0) + 2.0
        return tokens

    @staticmethod
    def _sparse_similarity(tf1: Dict[str, float], tf2: Dict[str, float]) -> float:
        if not tf1 or not tf2:
            return 0.0
        common_keys = set(tf1.keys()) & set(tf2.keys())
        dot = sum(tf1[k] * tf2[k] for k in common_keys)
        norm1 = math.sqrt(sum(v * v for v in tf1.values()))
        norm2 = math.sqrt(sum(v * v for v in tf2.values()))
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return dot / (norm1 * norm2)
