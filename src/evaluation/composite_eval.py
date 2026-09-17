from typing import Dict, Any, Tuple
from pydantic import BaseModel, Field

from src.core.models import StatisticalMetrics, DeepStyleProfile
from .lexical_metrics import LexicalEvaluator
from .rhythm_metrics import RhythmEvaluator
from .discourse_metrics import DiscourseEvaluator


class CompositeEvaluationResult(BaseModel):
    """EchoStyle 统一综合评测卡 (Unified Optimization Card)"""
    echo_score: float = Field(..., description="最终综合优化目标得分 (0-100)")
    style_fidelity: float = Field(..., description="文风神似度与读者主观好感 (权重 35%)")
    discourse_fit: float = Field(..., description="篇章宏观推进与结构拟合分 (权重 25%)")
    rhythm_match: float = Field(..., description="客观行文节奏与句长方差吻合分 (权重 20%)")
    lexical_authenticity: float = Field(..., description="标准化 STTR 用词质感分 (权重 20%)")
    cliche_penalty: float = Field(..., description="违规 AI 套话惩罚扣分")
    detected_cliches: list[str] = Field(default_factory=list)
    breakdowns: Dict[str, Any] = Field(default_factory=dict)


class CompositeEvaluator:
    """
    统一优化目标计算器 (Unified Objective Evaluator)：
    明确定义端到端唯一优化目标 EchoScore，使所有对比实验与消融实验均对齐于统一的可视化公式。
    """

    @classmethod
    def calculate_echoscore(
        cls,
        generated_text: str,
        target_metrics: StatisticalMetrics,
        target_profile: DeepStyleProfile,
        llm_fidelity_score: float = 85.0,
    ) -> CompositeEvaluationResult:
        # 1. 篇章结构拟合 (Discourse Fit)
        discourse_score, disc_breakdown = DiscourseEvaluator.evaluate_discourse_fit(generated_text, target_profile)

        # 2. 客观节奏吻合度 (Rhythm Match)
        rhythm_score, rhythm_breakdown = RhythmEvaluator.evaluate_rhythm_fit(generated_text, target_metrics)

        # 3. 词汇标准化丰富度 (Lexical Authenticity)
        target_sttr = (
            target_metrics.sttr
            if (target_metrics and target_metrics.sttr > 0)
            else (target_metrics.ttr if (target_metrics and target_metrics.ttr > 0) else 0.70)
        )
        lexical_score, lex_breakdown = LexicalEvaluator.evaluate_lexical_authenticity(generated_text, target_sttr)

        # 4. 违规套话扣分 (Cliche Penalty)
        forbidden = (
            target_profile.qualitative.anti_patterns.forbidden_words
            if (target_profile and target_profile.qualitative and target_profile.qualitative.anti_patterns)
            else None
        )
        _, cliche_penalty, detected = LexicalEvaluator.evaluate_cliches(
            generated_text, custom_forbidden=forbidden
        )

        # 5. 综合加权最终统一目标 (Unified Optimization Objective)
        # EchoScore = 0.35 * Fidelity + 0.25 * Discourse + 0.20 * Rhythm + 0.20 * Lexical - ClichePenalty
        base_score = (
            0.35 * llm_fidelity_score +
            0.25 * discourse_score +
            0.20 * rhythm_score +
            0.20 * lexical_score
        )
        final_echoscore = max(0.0, min(100.0, round(base_score - cliche_penalty, 1)))

        return CompositeEvaluationResult(
            echo_score=final_echoscore,
            style_fidelity=round(llm_fidelity_score, 1),
            discourse_fit=round(discourse_score, 1),
            rhythm_match=round(rhythm_score, 1),
            lexical_authenticity=round(lexical_score, 1),
            cliche_penalty=round(cliche_penalty, 1),
            detected_cliches=detected,
            breakdowns={
                "discourse": disc_breakdown,
                "rhythm": rhythm_breakdown,
                "lexical": lex_breakdown,
            }
        )
