import re
import uuid
from typing import Any, Dict, List, Optional
from src.core.config import EmbeddingConfig, LLMConfig
from src.core.models import ChunkMetadata
from .vector_store import VectorStore


class MemoryManager:
    """
    风格感知记忆管理器 (Style-Aware Memory Manager)：
    不仅仅根据主题找答案，更根据【篇章结构与行文风格】(hook/quote/argument/conclusion)
    与情感色彩进行定向精准召回。
    """

    def __init__(
        self,
        vector_store: Optional[VectorStore] = None,
        embedding_config: Optional[EmbeddingConfig] = None,
        llm_config: Optional[LLMConfig] = None,
    ):
        self.vector_store = vector_store or VectorStore(embedding_config=embedding_config, llm_config=llm_config)

    def ingest_article(self, title: str, content: str) -> int:
        chunks = self._chunk_article(title, content)
        return self.vector_store.add_chunks(chunks)

    def retrieve_style_aware(
        self,
        query: str,
        target_type: Optional[str] = None,
        top_k: int = 3,
        require_dense: bool = False,
    ) -> List[str]:
        """
        风格感知定向检索：
        :param query: 主题与意图
        :param target_type: 可选 'hook'(开篇痛点), 'quote'(犀利金句), 'argument'(核心论据), 'conclusion'(收尾)
        :param top_k: 召回数量
        :param require_dense: 是否强制要求 Dense 通道有效 (Fail-Closed)
        """
        results = self.vector_store.hybrid_search(
            query=query,
            top_k=top_k,
            type_filter=target_type,
            require_dense=require_dense,
        )
        return [r["content"] for r in results]

    def retrieve_dense(
        self,
        query: str,
        top_k: int = 3,
        target_type: Optional[str] = None,
    ) -> List[str]:
        """
        纯密集向量语义检索 (Standard Semantic RAG):
        不使用词频匹配，不经过 RRF 倒数排名融合，用于消融对照组严格控制变量。
        Fail-Closed：若向量不可用直接报错。
        """
        results = self.vector_store.dense_search(
            query=query,
            top_k=top_k,
            type_filter=target_type
        )
        return [r["content"] for r in results]

    def retrieve_dynamic_few_shots(
        self,
        query: str,
        top_k: int = 3,
        type_filter: Optional[str] = None,
        require_dense: bool = False,
    ) -> List[str]:
        """
        风格感知定向篇章结构组合召回 (Style-Aware Few-shot Retrieval)：
        若指定 type_filter 或 top_k < 3，执行指定类型或单条语义召回；
        若未指定 type_filter 且 top_k >= 3，定向均衡覆盖 hook (开篇痛点) + quote (警策金句) + argument (核心论据)，
        确保 WriterAgent 生成链路 (Condition D) 与消融实验 Condition C2 检索逻辑完全一致。
        """
        if type_filter or top_k < 3:
            return self.retrieve_style_aware(query, target_type=type_filter, top_k=top_k, require_dense=require_dense)

        hooks = self.retrieve_style_aware(query, target_type="hook", top_k=1, require_dense=require_dense)
        quotes = self.retrieve_style_aware(query, target_type="quote", top_k=1, require_dense=require_dense)
        args_s = self.retrieve_style_aware(query, target_type="argument", top_k=1, require_dense=require_dense)
        combined = hooks + quotes + args_s

        if len(combined) < top_k:
            fallback = self.vector_store.hybrid_search(query, top_k=top_k, require_dense=require_dense)
            for f in fallback:
                if f["content"] not in combined and len(combined) < top_k:
                    combined.append(f["content"])
        return combined[:top_k]

    def get_memory_stats(self) -> Dict[str, Any]:
        all_chunks = self.vector_store.get_all()
        types_count = {}
        for c in all_chunks:
            meta = c.get("metadata", {})
            t = meta.get("type", "argument")
            types_count[t] = types_count.get(t, 0) + 1

        return {
            "total_chunks": len(all_chunks),
            "type_breakdown": types_count,
            "sources": list(set(c.get("metadata", {}).get("source", "未知") for c in all_chunks)),
        }

    def clear_memory(self):
        self.vector_store.clear()

    def _chunk_article(self, title: str, content: str) -> List[Dict[str, Any]]:
        raw_paras = [p.strip() for p in content.split("\n\n") if len(p.strip()) >= 10]
        chunks = []
        total_paras = len(raw_paras)

        for idx, p in enumerate(raw_paras):
            if p.startswith("#") and len(p.split("\n")) == 1:
                continue

            # 篇章物理位置计算
            pos_pct = round(idx / max(1, total_paras - 1), 2)
            if pos_pct <= 0.2:
                position = "opening"
            elif pos_pct >= 0.8:
                position = "ending"
            else:
                position = "body"

            # 篇章结构功能语义显式判定 (Discourse Function Classification)
            if idx == 0 or (position == "opening" and ("？" in p or "！" in p or "说白了" in p or "别闹" in p)):
                func = "hook"
            elif idx == total_paras - 1 or (position == "ending" and ("守住" in p or "这是" in p or "唯一" in p or "总结" in p)):
                func = "conclusion"
            elif len(p) <= 120 and ("比如" in p or "例如" in p or "看到一篇" in p or "扒了一份" in p or "案例" in p):
                func = "example"
            elif len(p) <= 100 and ("说白了" in p or "本质上" in p or "必须" in p or "从来不是" in p or "绝不" in p or "！" in p):
                func = "quote"
            else:
                func = "argument"

            emotion = "客观分析"
            if "？" in p or "难道" in p:
                emotion = "设问反思"
            elif "！" in p or "绝不" in p or "别闹了" in p or "懦弱" in p:
                emotion = "犀利强烈"

            chunk_id = str(uuid.uuid4())[:8]
            meta = ChunkMetadata(
                chunk_id=chunk_id,
                source=title,
                position=position,
                position_pct=pos_pct,
                function=func,
                type=func,  # 兼容旧版 type 字段
                topic="通用",
                emotion=emotion,
                style="深度思考",
                char_length=len(p),
            )

            chunks.append({
                "id": chunk_id,
                "content": p,
                "metadata": meta.model_dump(),
            })

        return chunks
