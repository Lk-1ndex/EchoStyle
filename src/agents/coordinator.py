from typing import Any, Dict, List, Optional, Tuple
from src.core.config import AppConfig
from src.core.models import DeepStyleProfile, EvaluationReport
from src.memory.memory_manager import MemoryManager
from .base import BaseAgent
from .state import AgentState, AgentStatus
from .tools import Tool, ToolRegistry, CritiqueAction
from .extractor_agent import ExtractorAgent
from .analyst_agent import AnalystAgent
from .writer_agent import WriterAgent
from .critic_agent import CriticAgent


class CoordinatorAgent(BaseAgent):
    """
    中央调度与动态规划智能体 (Autonomous Planning & State Machine Coordinator)：
    1. 基于 ToolRegistry 统一管理并动态调度各专业工具；
    2. 具备 Checkpoint 快照与真回滚 (Rollback) 能力；
    3. 完整实现 run() 抽象接口，驱动有限状态机全生命周期流转。
    """

    def __init__(self, config: AppConfig, memory_manager: Optional[MemoryManager] = None):
        super().__init__(name="CoordinatorAgent", description="智能体调度中枢，负责意图理解、工具动态编排与有限状态机驱动")
        self.config = config
        self.memory_manager = memory_manager or MemoryManager(
            embedding_config=config.embedding,
            llm_config=config.llm
        )

        # 初始化专业智能体节点
        self.extractor_agent = ExtractorAgent(mineru_cmd=config.extractor.mineru_command)
        self.analyst_agent = AnalystAgent(config.llm, self.memory_manager)
        self.writer_agent = WriterAgent(config.llm, self.memory_manager)
        self.critic_agent = CriticAgent(config.llm, quality_threshold=config.agent.quality_threshold)

        # 初始化工具注册表 (Tool Registry)
        self.tool_registry = ToolRegistry()
        self._register_tools()

    def _register_tools(self):
        """将专业智能体能力包装注册为可编排的工具"""
        self.tool_registry.register(Tool(
            name="extract_document_tool",
            description="自适应排版探测与样文解析抽取工具",
            func=self.extractor_agent.run
        ))
        self.tool_registry.register(Tool(
            name="analyze_style_tool",
            description="统计语言学与语义质性文风建模工具",
            func=self.analyst_agent.run
        ))
        self.tool_registry.register(Tool(
            name="retrieve_memory_tool",
            description="基于 RRF 算法的风格记忆语义召回工具",
            func=self.memory_manager.retrieve_dynamic_few_shots
        ))
        self.tool_registry.register(Tool(
            name="draft_tool",
            description="记忆驱动的文风受控写作与反思重写工具",
            func=self.writer_agent.run
        ))
        self.tool_registry.register(Tool(
            name="critique_tool",
            description="EchoEval 多维质量盲审与结构化行动决策工具",
            func=self.critic_agent.evaluate_action
        ))

    def run(
        self,
        state: AgentState,
        profile: Optional[DeepStyleProfile] = None,
        sample_sources: Optional[List[str]] = None,
        topic: str = "",
        key_points: str = "",
        word_count: int = 1500,
        target_audience: str = "大众读者",
        **kwargs
    ) -> Tuple[str, EvaluationReport, AgentState]:
        """
        实现 BaseAgent 统一契约规范的核心执行入口 (受控工作流调度与工具编排)
        """
        state.topic = topic or state.topic
        state.key_points = key_points or state.key_points
        state.word_count = word_count or state.word_count
        state.target_audience = target_audience or state.target_audience
        state.max_retries = self.config.agent.max_reflections

        state.execution_logs.append(f"[WORKFLOW] 启动任务工作流调度，目标主题: [{state.topic}]")

        # 动态分支 1：检测是否需要解析样文与建模
        active_profile = profile
        if active_profile is None:
            if not sample_sources:
                raise ValueError("执行失败：未提供现成的文风档案 (profile)，也未提供样文输入源 (sample_sources)！")

            # 动态调用提取工具
            extracted_articles = []
            for src in sample_sources:
                art = self.tool_registry.get("extract_document_tool").execute(state, source=src)
                extracted_articles.append(art)

            # 动态调用建模工具
            active_profile = self.tool_registry.get("analyze_style_tool").execute(
                state, sample_articles=extracted_articles, profile_name="动态提炼文风"
            )

        # 动态分支 2：创建初稿前检查点 (CheckPoint)
        state.create_checkpoint("pre_draft")

        # 动态调用创作工具
        draft = self.tool_registry.get("draft_tool").execute(state, profile=active_profile)
        state.create_checkpoint("first_draft")

        # 动态分支 3：调用质检决策工具
        critique_action: CritiqueAction = self.tool_registry.get("critique_tool").execute(state, draft=draft, profile=active_profile)
        state.create_checkpoint("best_version")

        # 动态自省与状态机闭环 (Self-Reflection & Dynamic Recovery Loop)
        while critique_action.decision != "ACCEPT" and state.retry_count < state.max_retries:
            state.retry_count += 1
            prev_score = critique_action.score

            if critique_action.decision == "REJECT":
                # 若初稿发生严重文风偏离，执行回滚至 pre_draft 重新构思
                state.execution_logs.append("[DECISION] 评审判定严重不合格，执行快照回滚并重新撰写...")
                state.rollback_to("pre_draft")
                draft = self.tool_registry.get("draft_tool").execute(
                    state, profile=active_profile, feedback="严格注意语气人设，禁止大话套话！"
                )
            else:
                # REVISE 润色模式
                state.execution_logs.append(f"[DECISION] 触发第 {state.retry_count} 轮反思润色重构...")
                feedback_str = f"{critique_action.actionable_feedback}\n捕获八股词: {critique_action.detected_cliches}"
                draft = self.tool_registry.get("draft_tool").execute(
                    state, profile=active_profile, feedback=feedback_str, previous_draft=draft
                )

            # 重新质检验收
            critique_action = self.tool_registry.get("critique_tool").execute(state, draft=draft, profile=active_profile)

            # 状态自愈判断：若新版本明显退步（分数暴跌），执行回滚保护
            if critique_action.score < (prev_score - 15.0):
                state.execution_logs.append("[RECOVERY] 检测到反思重构版本评分退步，启动回滚至前一优选检查点...")
                state.rollback_to("best_version")
                draft = state.current_draft
                break
            else:
                state.create_checkpoint("best_version")

        # 终态判定
        final_report = critique_action.detailed_report or state.latest_report
        if critique_action.decision == "ACCEPT":
            state.transition_to(AgentStatus.COMPLETED, f"终审圆满通过！综合得分: {final_report.overall_score:.1f} 分。")
        else:
            state.transition_to(
                AgentStatus.COMPLETED_WITH_WARNING,
                f"反思重试轮次已满但未达ACCEPT阈值，降级输出当前最高质量版本 (得分: {final_report.overall_score:.1f})。"
            )

        return draft, final_report, state

    def generate_article(
        self,
        profile: DeepStyleProfile,
        topic: str,
        key_points: str = "",
        word_count: int = 1500,
        target_audience: str = "大众读者",
        state: Optional[AgentState] = None,
    ) -> Tuple[str, EvaluationReport, AgentState]:
        """保持原有方法名兼容的统一包装"""
        st = state or AgentState()
        return self.run(
            state=st,
            profile=profile,
            topic=topic,
            key_points=key_points,
            word_count=word_count,
            target_audience=target_audience,
        )

    def extract_sources(self, sources: List[str], state: Optional[AgentState] = None) -> List[Dict[str, Any]]:
        st = state or AgentState()
        articles = []
        for src in sources:
            try:
                res = self.tool_registry.get("extract_document_tool").execute(st, source=src, force_engine=self.config.extractor.pdf_engine)
                articles.append(res)
            except Exception as e:
                st.record_error(f"提取源 [{src}] 异常: {str(e)}")
        return articles

    def build_style(self, sample_articles: List[Dict[str, Any]], profile_name: str = "深度文风档案", state: Optional[AgentState] = None) -> DeepStyleProfile:
        st = state or AgentState()
        return self.tool_registry.get("analyze_style_tool").execute(st, sample_articles=sample_articles, profile_name=profile_name)
