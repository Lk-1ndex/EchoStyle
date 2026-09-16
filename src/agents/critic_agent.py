from .base import BaseAgent
from .state import AgentState, AgentStatus
from src.core.config import LLMConfig
from src.core.models import DeepStyleProfile, EvaluationReport
from src.evaluation.judge import LLMJudge


class CriticAgent(BaseAgent):
    """
    质量把关与自查智能体：
    负责进行严格的去 AI 味、风格一致性量化质检，更新状态机并向系统给出裁决。
    """

    def __init__(self, llm_config: LLMConfig, quality_threshold: float = 80.0):
        super().__init__(name="CriticAgent", description="负责多维度质量盲评审校、打分与生成反思建议")
        self.quality_threshold = quality_threshold
        self.judge = LLMJudge(llm_config)

    def run(self, state: AgentState, draft: str, profile: DeepStyleProfile) -> EvaluationReport:
        state.transition_to(AgentStatus.CRITIQUING, "启动 EchoEval 多维质量测评 (加权综合裁决 + 去AI味检查)...")

        report = self.judge.evaluate(draft, profile)
        state.latest_report = report

        # 更新最新草稿节点的评分信息
        if state.draft_chain:
            state.draft_chain[-1].score = report.overall_score
            state.draft_chain[-1].detected_cliches = report.detected_cliches
            state.draft_chain[-1].critic_feedback = report.feedback

        is_passed = report.overall_score >= self.quality_threshold and len(report.detected_cliches) == 0
        status_str = "【合格通过】" if is_passed else "【未达标，触发反思重构】"
        cliche_msg = f"，违规八股词: {report.detected_cliches}" if report.detected_cliches else ""

        state.transition_to(
            AgentStatus.CRITIQUING,
            f"测评出炉: 综合得分 {report.overall_score:.1f} (神似度 {report.style_fidelity:.1f}, 读者好感 {report.llm_judge_score:.1f}, 统计拟合 {report.stylometric_similarity:.1f}) -> 裁决: {status_str}{cliche_msg}"
        )

        return report
