from .base import BaseAgent
from .state import AgentState, AgentStatus
from .tools import CritiqueAction
from src.core.config import LLMConfig
from src.core.models import DeepStyleProfile, EvaluationReport
from src.core.exceptions import EvaluationUnavailableError
from src.evaluation.judge import LLMJudge


class CriticAgent(BaseAgent):
    """
    质量把关与自查智能体：
    负责进行严格的去 AI 味、风格一致性量化质检，并输出结构化行动决策 (ACCEPT, REVISE, REJECT)。
    """

    def __init__(self, llm_config: LLMConfig, quality_threshold: float = 80.0):
        super().__init__(name="CriticAgent", description="负责多维度质量盲评审校、打分与生成结构化决策动作")
        self.quality_threshold = quality_threshold
        self.judge = LLMJudge(llm_config)

    def evaluate_action(self, state: AgentState, draft: str, profile: DeepStyleProfile) -> CritiqueAction:
        """评估并返回结构化决策动作"""
        state.transition_to(AgentStatus.CRITIQUING, "启动 EchoEval 多维质量测评与决策推断...")

        try:
            report = self.judge.evaluate(draft, profile)
        except EvaluationUnavailableError as exc:
            state.latest_report = None
            state.transition_to(AgentStatus.FAILED, f"裁判服务不可用，评测失败: {exc}")
            raise
        state.latest_report = report

        # 更新最新草稿节点
        if state.draft_chain:
            state.draft_chain[-1].score = report.overall_score
            state.draft_chain[-1].detected_cliches = report.detected_cliches
            state.draft_chain[-1].critic_feedback = report.feedback

        # 结构化决策推断 (Decision Inference)
        if report.overall_score >= self.quality_threshold and len(report.detected_cliches) == 0:
            decision = "ACCEPT"
        elif report.overall_score >= 50.0:
            decision = "REVISE"
        else:
            decision = "REJECT"

        action = CritiqueAction(
            decision=decision,
            score=report.overall_score,
            detected_cliches=report.detected_cliches,
            actionable_feedback=report.feedback,
            detailed_report=report,
        )

        cliche_msg = f"，违规八股词: {report.detected_cliches}" if report.detected_cliches else ""
        state.transition_to(
            AgentStatus.CRITIQUING,
            f"测评出炉: 得分 {report.overall_score:.1f} -> 结构化行动决策: [{decision}]{cliche_msg}"
        )
        return action

    def evaluate(self, draft: str, profile: DeepStyleProfile, state: AgentState | None = None) -> EvaluationReport:
        """独立评估草稿并返回 EvaluationReport"""
        if state is None:
            state = AgentState()
        try:
            return self.judge.evaluate(draft, profile)
        except EvaluationUnavailableError as exc:
            state.latest_report = None
            state.transition_to(AgentStatus.FAILED, f"裁判服务不可用，评测失败: {exc}")
            raise

    def run(self, state: AgentState, draft: str, profile: DeepStyleProfile) -> EvaluationReport:
        """兼容原有调用的基础接口"""
        action = self.evaluate_action(state, draft, profile)
        return action.detailed_report
