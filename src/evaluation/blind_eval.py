import math
import random
from typing import Dict, List, Any, Tuple, Optional
from pydantic import BaseModel, Field

from src.core.model_provider import ModelProvider
from src.core.config import LLMConfig


class PairwiseComparisonResult(BaseModel):
    task_id: str
    topic: str
    winner: str  # 'candidate', 'baseline', or 'tie'
    candidate_score: float
    baseline_score: float
    rationale: str


class BlindEvaluationSummary(BaseModel):
    total_trials: int
    candidate_wins: int
    baseline_wins: int
    ties: int
    win_rate: float
    ci_lower: float  # 95% Wilson Score CI Lower
    ci_upper: float  # 95% Wilson Score CI Upper
    trials: List[PairwiseComparisonResult]


class BlindEvaluator:
    """
    规范化双盲对照评测器 (Blind Pairwise Comparison Evaluator)：
    - 随机打乱候选项展示顺序 (Blind A/B)，杜绝位置偏置 (Position Bias)
    - 评判维度：声音辨识度 (Voice)、观点张力 (Sharpness)、篇章推进 (Discourse)、去AI味 (Anti-AI)
    - 统计学输出：真实胜率与 95% Wilson Score 置信区间 (95% CI)
    """

    @classmethod
    def calculate_wilson_ci(cls, successes: float, total: int, confidence: float = 0.95) -> Tuple[float, float]:
        """计算 95% Wilson Score 连续性置信区间"""
        if total == 0:
            return 0.0, 0.0

        z = 1.95996  # 95% 置信度 z 统计量
        p_hat = successes / total

        denominator = 1 + (z ** 2) / total
        center = (p_hat + (z ** 2) / (2 * total)) / denominator
        spread = (z * math.sqrt((p_hat * (1 - p_hat) / total) + ((z ** 2) / (4 * (total ** 2))))) / denominator

        lower = max(0.0, round(center - spread, 4))
        upper = min(1.0, round(center + spread, 4))
        return lower, upper

    @classmethod
    def evaluate_pair(
        cls,
        topic: str,
        text_candidate: str,
        text_baseline: str,
        author_reference: str,
        provider: Optional[ModelProvider] = None,
    ) -> PairwiseComparisonResult:
        """
        执行单次双盲裁决。
        如果未提供真实 LLM provider，采用规则与语言学统计进行客观盲测仲裁。
        """
        # 随机分配 A 和 B，避免固定顺序带来的位置偏好
        is_candidate_first = random.choice([True, False])
        sample_a = text_candidate if is_candidate_first else text_baseline
        sample_b = text_baseline if is_candidate_first else text_candidate

        if provider and provider.llm_config.api_key:
            judge_prompt = f"""你是一名严格的文学总编辑与盲评裁判。
请对比【样本 A】与【样本 B】，判断哪一篇在【行文风格、思想锐利度、篇章自然呼吸感、去 AI 八股味】上更契合【原作者参考范文】。

【原作者参考范文风格基准】
{author_reference[:600]}

【评测选题】
{topic}

【样本 A】
{sample_a[:600]}

【样本 B】
{sample_b[:600]}

请严格按以下 JSON 格式输出打分与裁决：
{{
  "score_a": <0-100分>,
  "score_b": <0-100分>,
  "winner": "A" | "B" | "TIE",
  "rationale": "<详细评价依据>"
}}"""
            try:
                raw_json = provider.chat(
                    system_prompt="你是一位极其挑剔的文学总编辑盲评审裁者，只输出合法 JSON。",
                    user_prompt=judge_prompt,
                    temperature=0.2,
                    json_mode=True
                )
                import json
                data = json.loads(raw_json)
                score_a = float(data.get("score_a", 75.0))
                score_b = float(data.get("score_b", 75.0))
                raw_winner = data.get("winner", "TIE").upper()
                rationale = data.get("rationale", "")

                if raw_winner == "A":
                    winner = "candidate" if is_candidate_first else "baseline"
                elif raw_winner == "B":
                    winner = "baseline" if is_candidate_first else "candidate"
                else:
                    winner = "tie"

                c_score = score_a if is_candidate_first else score_b
                b_score = score_b if is_candidate_first else score_a

                return PairwiseComparisonResult(
                    task_id=f"task_{random.randint(100, 999)}",
                    topic=topic,
                    winner=winner,
                    candidate_score=c_score,
                    baseline_score=b_score,
                    rationale=rationale
                )
            except Exception:
                pass

        # 离线客观规则仲裁 (Fallback)
        from src.evaluation.metrics import MetricEvaluator
        from src.analyzer.stylometrics import StylometricsAnalyzer

        target_m = StylometricsAnalyzer.analyze(author_reference)
        fit_c, _ = MetricEvaluator.evaluate_stylometric_fit(text_candidate, target_m)
        anti_c, _, _ = MetricEvaluator.evaluate_anti_ai(text_candidate)
        disc_c, _ = MetricEvaluator.evaluate_discourse_fit(text_candidate)
        total_c = fit_c * 0.35 + anti_c * 0.35 + disc_c * 0.30

        fit_b, _ = MetricEvaluator.evaluate_stylometric_fit(text_baseline, target_m)
        anti_b, _, _ = MetricEvaluator.evaluate_anti_ai(text_baseline)
        disc_b, _ = MetricEvaluator.evaluate_discourse_fit(text_baseline)
        total_b = fit_b * 0.35 + anti_b * 0.35 + disc_b * 0.30

        if total_c > total_b + 3.0:
            winner = "candidate"
        elif total_b > total_c + 3.0:
            winner = "baseline"
        else:
            winner = "tie"

        return PairwiseComparisonResult(
            task_id=f"task_{random.randint(100, 999)}",
            topic=topic,
            winner=winner,
            candidate_score=round(total_c, 1),
            baseline_score=round(total_b, 1),
            rationale=f"客观文风与篇章拟合度对比: 候选方案得分 {total_c:.1f}, 对照组得分 {total_b:.1f}"
        )

    @classmethod
    def aggregate_results(cls, results: List[PairwiseComparisonResult]) -> BlindEvaluationSummary:
        total = len(results)
        if total == 0:
            return BlindEvaluationSummary(
                total_trials=0, candidate_wins=0, baseline_wins=0, ties=0,
                win_rate=0.0, ci_lower=0.0, ci_upper=0.0, trials=[]
            )

        c_wins = sum(1 for r in results if r.winner == "candidate")
        b_wins = sum(1 for r in results if r.winner == "baseline")
        ties = sum(1 for r in results if r.winner == "tie")

        # 胜率折算 (按学术标准平局折算 0.5 胜场)
        effective_wins = c_wins + 0.5 * ties
        win_rate = round(effective_wins / total, 3)
        ci_lower, ci_upper = cls.calculate_wilson_ci(effective_wins, total)

        return BlindEvaluationSummary(
            total_trials=total,
            candidate_wins=c_wins,
            baseline_wins=b_wins,
            ties=ties,
            win_rate=win_rate,
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            trials=results
        )
