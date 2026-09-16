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
    ) -> List[str]:
        """
        风格感知定向检索：
        :param query: 主题与意图
        :param target_type: 可选 'hook'(开篇痛点), 'quote'(犀利金句), 'argument'(核心论据), 'conclusion'(收尾)
        :param top_k: 召回数量
        """
        results = self.vector_store.hybrid_search(
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
    ) -> List[str]:
        """组合多风格类型的均衡 Few-shot 召回"""
        return self.retrieve_style_aware(query, target_type=type_filter, top_k=top_k)

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

            chunk_type = "argument"
            if idx == 0:
                chunk_type = "hook"
            elif idx == total_paras - 1:
                chunk_type = "conclusion"
            elif len(p) < 90 and ("说白了" in p or "本质上" in p or "其实" in p or "！" in p or "必须" in p):
                chunk_type = "quote"

            emotion = "客观分析"
            if "？" in p or "难道" in p:
                emotion = "设问反思"
            elif "！" in p or "绝不" in p or "别闹了" in p:
                emotion = "犀利强烈"

            chunk_id = str(uuid.uuid4())[:8]
            meta = ChunkMetadata(
                chunk_id=chunk_id,
                source=title,
                type=chunk_type,
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
