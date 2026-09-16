from typing import Any, Dict, List, Optional, Tuple
from src.core.config import AppConfig
from src.core.models import DeepStyleProfile, EvaluationReport
from src.memory.memory_manager import MemoryManager
from .base import BaseAgent
from .state import AgentState, AgentStatus
from .extractor_agent import ExtractorAgent
from .analyst_agent import AnalystAgent
from .writer_agent import WriterAgent
from .critic_agent import CriticAgent


class CoordinatorAgent(BaseAgent):
    """
    中央调度与状态机编排智能体：
    统筹各专业 Agent 的生命周期流转，驱动状态机闭环与错误恢复。
    """

    def __init__(self, config: AppConfig, memory_manager: Optional[MemoryManager] = None):
        super().__init__(name="CoordinatorAgent", description="调度中枢，负责任务编排、状态机生命周期与反思重构闭环")
        self.config = config
        self.memory_manager = memory_manager or MemoryManager(
            embedding_config=config.embedding,
            llm_config=config.llm
        )

        # 初始化各专业 Agent 节点
        self.extractor_agent = ExtractorAgent(mineru_cmd=config.extractor.mineru_command)
        self.analyst_agent = AnalystAgent(config.llm, self.memory_manager)
        self.writer_agent = WriterAgent(config.llm, self.memory_manager)
        self.critic_agent = CriticAgent(config.llm, quality_threshold=config.agent.quality_threshold)

    def extract_sources(self, sources: List[str], state: Optional[AgentState] = None) -> List[Dict[str, Any]]:
        st = state or AgentState()
        st.transition_to(AgentStatus.PARSING, f"开始感知与提取 {len(sources)} 个输入源...")

        articles = []
        for src in sources:
            try:
                res = self.extractor_agent.run(st, source=src, force_engine=self.config.extractor.pdf_engine)
                articles.append(res)
            except Exception as e:
                st.record_error(f"提取源 [{src}] 异常: {str(e)}")
        return articles

    def build_style(self, sample_articles: List[Dict[str, Any]], profile_name: str = "深度文风档案", state: Optional[AgentState] = None) -> DeepStyleProfile:
        st = state or AgentState()
        st.transition_to(AgentStatus.MODELING, "调度 AnalystAgent 开启双驱建模...")
        return self.analyst_agent.run(st, sample_articles=sample_articles, profile_name=profile_name)

    def generate_article(
        self,
        profile: DeepStyleProfile,
        topic: str,
        key_points: str = "",
        word_count: int = 1500,
        target_audience: str = "大众读者",
        state: Optional[AgentState] = None,
    ) -> Tuple[str, EvaluationReport, AgentState]:
        st = state or AgentState()
        st.topic = topic
        st.key_points = key_points
        st.word_count = word_count
        st.target_audience = target_audience
        st.max_retries = self.config.agent.max_reflections

        st.transition_to(AgentStatus.DRAFTING, f"=== 任务启动: 主题 [{topic}] ===")

        # 1. 首轮创作
        draft = self.writer_agent.run(st, profile=profile)

        # 2. 首轮质检
        report = self.critic_agent.run(st, draft=draft, profile=profile)

        # 3. 驱动反思重写状态机 (Self-Reflection Loop)
        while (report.overall_score < self.config.agent.quality_threshold or len(report.detected_cliches) > 0) and st.retry_count < st.max_retries:
            st.retry_count += 1
            st.transition_to(
                AgentStatus.REFLECTING,
                f"=== 触发第 {st.retry_count}/{st.max_retries} 轮反思重写 (当前评分 {report.overall_score:.1f} < 阈值 {self.config.agent.quality_threshold} 分) ==="
            )

            feedback_content = f"{report.feedback}\n违规套话: {report.detected_cliches}" if report.detected_cliches else report.feedback

            # Writer 结合 Critic 反馈执行重构
            draft = self.writer_agent.run(
                st,
                profile=profile,
                feedback=feedback_content,
                previous_draft=draft,
            )

            # Critic 再次严审
            report = self.critic_agent.run(st, draft=draft, profile=profile)

        # 4. 终态判定
        if report.overall_score >= self.config.agent.quality_threshold and len(report.detected_cliches) == 0:
            st.transition_to(AgentStatus.COMPLETED, f"终审达成！综合得分: {report.overall_score:.1f} 分，文章符合所有文风与去AI味红线。")
        else:
            if st.retry_count >= st.max_retries:
                st.transition_to(AgentStatus.COMPLETED, f"已达最大反思轮次 ({st.max_retries} 轮)，输出当前最优版本 (得分: {report.overall_score:.1f})。")
            else:
                st.transition_to(AgentStatus.FAILED, "任务未能通过质检验收。")

        return draft, report, st

    def run(self, state: AgentState, **kwargs) -> Any:
        pass
