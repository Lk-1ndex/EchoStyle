from typing import List, Optional
from uuid import uuid4
from .base import BaseAgent
from .state import AgentState, AgentStatus
from src.core.config import LLMConfig
from src.core.models import DeepStyleProfile
from src.analyzer.stylometrics import StylometricsAnalyzer
from src.analyzer.distiller import StyleDistiller
from src.memory.memory_manager import MemoryManager
from src.memory.profile_store import ProfileStore


class AnalystAgent(BaseAgent):
    """
    风格建模与记忆蒸馏智能体：
    双驱分析（TTR/统计语言学 + 质性解构），并将高价值语料切片持久化至 Style Memory。
    """

    def __init__(
        self,
        llm_config: LLMConfig,
        memory_manager: MemoryManager,
        profile_store: Optional[ProfileStore] = None,
    ):
        super().__init__(name="AnalystAgent", description="负责作者文风统计建模、质性特征解构及语料向量化记忆")
        self.llm_config = llm_config
        self.memory_manager = memory_manager
        self.profile_store = profile_store
        self.distiller = StyleDistiller(llm_config)

    def run(self, state: AgentState, sample_articles: List[dict], profile_name: str = "深度文风档案") -> DeepStyleProfile:
        state.transition_to(AgentStatus.MODELING, f"启动深度文风建模，正在分析 {len(sample_articles)} 篇样文...")

        full_corpus = "\n\n".join([a["content"] for a in sample_articles])

        # 1. 统计语言学客观建模 (含 TTR)
        state.transition_to(AgentStatus.MODELING, "计算统计指标（句长均值/方差、TTR词汇丰富度、标点熵、转折词密度）...")
        quantitative_metrics = StylometricsAnalyzer.analyze(full_corpus)
        state.transition_to(
            AgentStatus.MODELING,
            f"客观特征计算完成: 平均句长 {quantitative_metrics.avg_sentence_length:.1f} 字, 离散度 {quantitative_metrics.sentence_length_std:.1f}, 词汇丰富度 TTR {quantitative_metrics.ttr:.3f}"
        )

        # 2. 质性特征逆向解构 (LLM 分析)
        state.transition_to(AgentStatus.MODELING, "调用 LLM 进行五维文风指纹逆向工程（视角、人设、口癖、篇章逻辑）...")
        qualitative_profile = self.distiller.distill(
            [a["content"] for a in sample_articles],
            profile_name=profile_name
        )
        if self.distiller.last_retry_used:
            state.execution_logs.append("[RECOVERY] 首次画像 JSON 无效，已通过分段 Schema 重建并恢复。")

        # 3. 融合为 DeepStyleProfile
        deep_profile = DeepStyleProfile(
            name=profile_name,
            profile_id=str(uuid4()),
            qualitative=qualitative_profile,
            quantitative=quantitative_metrics
        )

        # 4. 先构建全部切片并单次提交 Style Memory，再以画像保存作为可见提交标记。
        state.transition_to(AgentStatus.MODELING, "将样文段落进行多维切片并存入长期风格记忆库 (Style Memory)...")
        committed_chunk_ids = self.memory_manager.ingest_articles(
            sample_articles,
            profile_id=deep_profile.profile_id,
        )
        total_chunks_added = len(committed_chunk_ids)
        if self.profile_store is not None:
            try:
                self.profile_store.save(deep_profile)
            except Exception as save_error:
                try:
                    self.memory_manager.remove_chunks(
                        deep_profile.profile_id,
                        committed_chunk_ids,
                    )
                except Exception as rollback_error:
                    rollback_note = f"风格记忆补偿回滚失败: {rollback_error}"
                    add_note = getattr(save_error, "add_note", None)
                    if add_note is not None:
                        add_note(rollback_note)
                    else:
                        notes = getattr(save_error, "__notes__", None)
                        if notes is None:
                            notes = []
                            setattr(save_error, "__notes__", notes)
                        notes.append(rollback_note)
                raise
        if self.memory_manager.vector_store.model_provider.has_embedding_credentials():
            result_message = f"成功向量化入库 {total_chunks_added} 个风格片段，并持久化文风档案！"
        else:
            result_message = (
                f"Embedding 未配置，已按稀疏检索模式保存 {total_chunks_added} 个风格片段，并持久化文风档案。"
            )
        state.transition_to(AgentStatus.MODELING, result_message)

        return deep_profile
