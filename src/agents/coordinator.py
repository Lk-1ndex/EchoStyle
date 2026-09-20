from typing import Any, Dict, List, Optional, Tuple
from src.core.config import AppConfig
from src.core.models import DeepStyleProfile, EvaluationReport
from src.memory.memory_manager import MemoryManager
from src.memory.profile_store import ProfileStore
from .base import BaseAgent
from .state import AgentState, AgentStatus
from .tools import Tool, ToolRegistry, CritiqueAction
from .extractor_agent import ExtractorAgent
from .analyst_agent import AnalystAgent
from .writer_agent import WriterAgent
from .critic_agent import CriticAgent


class CoordinatorAgent(BaseAgent):
    """
    受控工作流调度与状态机编排中枢 (Controlled Workflow & State Machine Coordinator)：
    1. 基于 ToolRegistry 统一管理并确定性编排各专业智能体工具；
    2. 具备内容快照 (DraftCheckpoint) 与有限状态机 (FSM) 失败回滚能力；
    3. 完整实现 run() 契约规范，驱动写作、质检、反思重构全生命周期流转。
    """

    def __init__(
        self,
        config: AppConfig,
        memory_manager: Optional[MemoryManager] = None,
        profile_store: Optional[ProfileStore] = None,
    ):
        super().__init__(name="CoordinatorAgent", description="工作流调度中枢，负责受控工具编排与有限状态机驱动")
        self.config = config
        self.memory_manager = memory_manager or MemoryManager(
            embedding_config=config.embedding,
            llm_config=config.llm
        )
        self.profile_store = profile_store

        # 初始化专业智能体节点
        self.extractor_agent = ExtractorAgent(
            mineru_cmd=config.extractor.mineru_command,
            mineru_tier=config.extractor.mineru_tier,
            mineru_timeout=config.extractor.mineru_timeout,
            mineru_intra_op_num_threads=config.extractor.mineru_intra_op_num_threads,
            mineru_inter_op_num_threads=config.extractor.mineru_inter_op_num_threads,
            mineru_pdf_render_threads=config.extractor.mineru_pdf_render_threads,
            mineru_malloc_trim=config.extractor.mineru_malloc_trim,
        )
        self.analyst_agent = AnalystAgent(config.llm, self.memory_manager, profile_store=self.profile_store)
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
        word_count: Optional[int] = None,
        target_audience: Optional[str] = None,
        initial_draft: Optional[str] = None,
        revision_instruction: Optional[str] = None,
        **kwargs
    ) -> Tuple[str, EvaluationReport, AgentState]:
        """
        实现 BaseAgent 统一契约规范的核心执行入口 (受控工作流调度与工具编排)
        """
        state.topic = topic or state.topic
        state.key_points = key_points or state.key_points
        if word_count is not None:
            state.word_count = word_count
        elif initial_draft is not None and revision_instruction:
            state.word_count = max(1, len("".join(initial_draft.split())))
        if target_audience is not None:
            state.target_audience = target_audience
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
        if not active_profile.profile_id:
            raise ValueError("旧版文风档案没有 profile_id，请重新建模后再写作。")

        # 动态分支 2：创建初稿前检查点 (CheckPoint)
        state.create_checkpoint("pre_draft")

        # 动态调用创作工具。消融实验可直接复用初稿；对话模式可先按用户要求定向修改。
        if initial_draft is not None:
            if state.current_status == AgentStatus.IDLE:
                state.transition_to(AgentStatus.DRAFTING, "加载外部初稿 (单变量严格对照)")
            state.record_draft("UserDraft" if revision_instruction else "WriterAgent", initial_draft)
            if revision_instruction:
                state.transition_to(AgentStatus.CRITIQUING, "收到用户对现有稿件的定向修改要求。")
                draft = self.tool_registry.get("draft_tool").execute(
                    state,
                    profile=active_profile,
                    feedback=revision_instruction,
                    previous_draft=initial_draft,
                )
            else:
                draft = initial_draft
        else:
            draft = self.tool_registry.get("draft_tool").execute(state, profile=active_profile)
        state.create_checkpoint("first_draft")

        # 动态分支 3：调用质检决策工具
        critique_action: CritiqueAction = self.tool_registry.get("critique_tool").execute(state, draft=draft, profile=active_profile)
        best_score = critique_action.score
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

            # 状态自愈判断：若新版本明显退步（相较前一轮评分暴跌超过 15 分），启动回滚保护至历史最高分检查点
            if critique_action.score < (prev_score - 15.0):
                state.execution_logs.append(
                    f"[RECOVERY] 检测到反思重构版本评分退步 (当前 {critique_action.score:.1f} < 前轮 {prev_score:.1f} - 15.0)，"
                    f"启动回滚至历史最优检查点 'best_version' (历史最高分: {best_score:.1f})..."
                )
                state.rollback_to("best_version")
                draft = state.current_draft
                break
            else:
                # 核心修复 P1-8：只有当新版本得分严格超越历史最优得分时，才更新 best_version 快照！
                # 避免次优版本（如轻微下滑未触发15分熔断的版本）将真正的历史最高分版本覆盖
                if critique_action.score > best_score:
                    best_score = critique_action.score
                    state.create_checkpoint("best_version")

        # 终态判定：若循环结束且未 ACCEPT，且当前版本不如历史最优，自动回退至最优检查点
        if critique_action.decision != "ACCEPT" and critique_action.score < best_score:
            state.execution_logs.append(
                f"[RECOVERY] 反思重试结束，当前版本得分 ({critique_action.score:.1f}) 低于历史最优 ({best_score:.1f})，"
                f"自动回退至历史最优检查点 'best_version'..."
            )
            state.rollback_to("best_version")
            draft = state.current_draft

        # 终态判定：若触发回滚保护，优先以回滚后 state.latest_report 为准
        final_report = state.latest_report or critique_action.detailed_report
        final_score = final_report.overall_score if final_report else critique_action.score
        threshold = getattr(self.critic_agent, "quality_threshold", 80.0)
        has_cliches = bool(final_report.detected_cliches) if final_report else bool(critique_action.detected_cliches)
        is_accepted = (final_score >= threshold and not has_cliches)

        if is_accepted:
            state.transition_to(AgentStatus.COMPLETED, f"终审圆满通过！综合得分: {final_score:.1f} 分。")
        else:
            state.transition_to(
                AgentStatus.COMPLETED_WITH_WARNING,
                f"反思重试轮次已满但未达ACCEPT阈值，降级输出当前最高质量版本 (得分: {final_score:.1f})。"
            )

        return draft, final_report, state

    def generate_article(
        self,
        profile: DeepStyleProfile,
        topic: str,
        key_points: str = "",
        word_count: Optional[int] = None,
        target_audience: str = "大众读者",
        state: Optional[AgentState] = None,
        initial_draft: Optional[str] = None,
        revision_instruction: Optional[str] = None,
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
            initial_draft=initial_draft,
            revision_instruction=revision_instruction,
        )

    def extract_sources(self, sources: List[str], state: Optional[AgentState] = None) -> List[Dict[str, Any]]:
        st = state or AgentState()
        articles = []
        last_error = None
        for src in sources:
            try:
                res = self.tool_registry.get("extract_document_tool").execute(st, source=src, force_engine=self.config.extractor.pdf_engine)
                articles.append(res)
            except Exception as e:
                st.record_error(f"提取源 [{src}] 异常: {str(e)}")
                last_error = e
        if not articles and last_error is not None:
            raise last_error
        return articles

    def build_style(self, sample_articles: List[Dict[str, Any]], profile_name: str = "深度文风档案", state: Optional[AgentState] = None) -> DeepStyleProfile:
        st = state or AgentState()
        return self.tool_registry.get("analyze_style_tool").execute(st, sample_articles=sample_articles, profile_name=profile_name)
