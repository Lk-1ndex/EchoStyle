import re
import uuid
from typing import Any, Dict, List, Optional
from src.core.config import EmbeddingConfig, LLMConfig
from src.core.models import ChunkMetadata
from .vector_store import VectorStore


class MemoryManager:
    """
    长期风格记忆管理器 (Style Memory RAG)：
    实现历史文章的多维切片、富元数据（Rich Metadata）打标与 Hybrid 混合检索。
    """

    def __init__(
        self,
        vector_store: Optional[VectorStore] = None,
        embedding_config: Optional[EmbeddingConfig] = None,
        llm_config: Optional[LLMConfig] = None,
    ):
        self.vector_store = vector_store or VectorStore(embedding_config=embedding_config, llm_config=llm_config)

    def ingest_article(self, title: str, content: str) -> int:
        """将文章切片、打标并摄入记忆库"""
        chunks = self._chunk_article(title, content)
        return self.vector_store.add_chunks(chunks)

    def retrieve_dynamic_few_shots(
        self,
        query: str,
        top_k: int = 3,
        type_filter: Optional[str] = None,
    ) -> List[str]:
        """
        语义动态召回最匹配的原文高光片段
        """
        results = self.vector_store.hybrid_search(query, top_k=top_k, type_filter=type_filter)
        return [r["content"] for r in results]

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

            # 段落类型感知
            chunk_type = "argument"
            if idx == 0:
                chunk_type = "hook"
            elif idx == total_paras - 1:
                chunk_type = "conclusion"
            elif len(p) < 90 and ("说白了" in p or "本质上" in p or "其实" in p or "！" in p or "必须" in p):
                chunk_type = "quote"

            # 情绪/风格初级启发式推断
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
