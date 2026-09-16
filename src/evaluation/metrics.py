"""
EchoStyle 评测模块门面 (Evaluation Facade)：
整合词汇、节奏偏离度、篇章结构拟合与综合优化目标，统一对外暴露纯净接口。
"""
from typing import Any, Dict, List, Tuple, Optional
from src.core.models import StatisticalMetrics
from src.analyzer.stylometrics import StylometricsAnalyzer

# 导入解耦后的子模块
from .lexical_metrics import LexicalEvaluator
from .rhythm_metrics import RhythmEvaluator
from .discourse_metrics import DiscourseEvaluator, DISCOURSE_GOLD_BENCHMARK
from .composite_eval import CompositeEvaluator, CompositeEvaluationResult
from .judge import MultiPersonaJudge, EVALUATOR_PANEL


class MetricEvaluator:
    """
    量化评测门面类 (Facade)：保持向后兼容性，代理至各解耦子模块。
    """

    DEFAULT_AI_CLICHES = LexicalEvaluator.DEFAULT_AI_CLICHES

    @classmethod
    def evaluate_anti_ai(cls, text: str, custom_forbidden: Optional[List[str]] = None) -> Tuple[float, float, List[str]]:
        """计算【去 AI 味得分】与综合违规惩罚分"""
        return LexicalEvaluator.evaluate_cliches(text, custom_forbidden)

    @classmethod
    def evaluate_discourse_fit(cls, generated_text: str, target_profile: Any = None) -> Tuple[float, Dict[str, Any]]:
        """篇章结构与行文逻辑拟合度 (Discourse Style Fit)"""
        return DiscourseEvaluator.evaluate_discourse_fit(generated_text, target_profile)

    @classmethod
    def evaluate_stylometric_fit(cls, generated_text: str, target_metrics: StatisticalMetrics) -> Tuple[float, Dict[str, float]]:
        """
        计算【统计语言学拟合度】：
        结合 RhythmEvaluator (节奏偏离度) 与 LexicalEvaluator (STTR 标准化词汇丰富度)。
        """
        if not target_metrics or target_metrics.total_sentences == 0:
            return 85.0, {"sentence_len_match": 85.0, "rhythm_std_match": 85.0, "ttr_match": 85.0, "sttr_match": 85.0}

        rhythm_score, rhythm_breakdown = RhythmEvaluator.evaluate_rhythm_fit(generated_text, target_metrics)
        target_sttr = target_metrics.sttr if target_metrics.sttr > 0 else target_metrics.ttr
        lex_score, lex_breakdown = LexicalEvaluator.evaluate_lexical_authenticity(generated_text, target_sttr)

        overall_fit = round(rhythm_score * 0.70 + lex_score * 0.30, 1)

        breakdown = {
            "sentence_len_match": rhythm_breakdown["sentence_len_score"],
            "rhythm_std_match": rhythm_breakdown["std_score"],
            "ttr_match": lex_breakdown["sttr_score"],
            "sttr_match": lex_breakdown["sttr_score"],
            "entropy_match": rhythm_breakdown["entropy_score"],
            "gen_avg_len": rhythm_breakdown["gen_avg_len"],
            "target_avg_len": rhythm_breakdown["target_avg_len"],
            "gen_sttr": lex_breakdown["gen_sttr"],
            "target_sttr": lex_breakdown["target_sttr"],
        }
        return overall_fit, breakdown

    @classmethod
    def calculate_stylometric_deviation(
        cls,
        generated_text: str,
        target_metrics: StatisticalMetrics,
        target_profile: Any = None,
    ) -> Dict[str, Any]:
        """
        计算生成文本相对于目标作者基准的客观文风偏离度 (Delta Stylometrics)
        结合 Rhythm Deviation、STTR 偏离、篇章结构偏离与 AI 套话惩罚。
        """
        gen_metrics = StylometricsAnalyzer.analyze(generated_text)
        _, total_penalty, detected_cliches = cls.evaluate_anti_ai(generated_text)
        discourse_score, discourse_breakdown = cls.evaluate_discourse_fit(generated_text, target_profile)
        _, rhythm_breakdown = RhythmEvaluator.evaluate_rhythm_fit(generated_text, target_metrics)

        delta_avg_len = abs(gen_metrics.avg_sentence_length - target_metrics.avg_sentence_length)
        delta_std = abs(gen_metrics.sentence_length_std - target_metrics.sentence_length_std)
        target_sttr = target_metrics.sttr if target_metrics.sttr > 0 else target_metrics.ttr
        delta_sttr = abs(gen_metrics.sttr - target_sttr)
        delta_entropy = abs(gen_metrics.punctuation_entropy - target_metrics.punctuation_entropy)
        delta_short_ratio = abs(gen_metrics.short_sentence_ratio - target_metrics.short_sentence_ratio)

        # 综合偏离惩罚指数 (Composite Stylometric Deviation)
        composite_dev = round(
            delta_avg_len * 1.0 +
            delta_std * 1.5 +
            delta_sttr * 50.0 +
            delta_short_ratio * 20.0 +
            total_penalty * 0.4 +
            max(0.0, 100.0 - discourse_score) * 0.15,
            2
        )

        return {
            "gen_avg_len": round(gen_metrics.avg_sentence_length, 2),
            "target_avg_len": round(target_metrics.avg_sentence_length, 2),
            "delta_avg_len": round(delta_avg_len, 2),
            "gen_std": round(gen_metrics.sentence_length_std, 2),
            "target_std": round(target_metrics.sentence_length_std, 2),
            "delta_std": round(delta_std, 2),
            "gen_sttr": round(gen_metrics.sttr, 3),
            "target_sttr": round(target_sttr, 3),
            "delta_sttr": round(delta_sttr, 3),
            "gen_ttr": round(gen_metrics.ttr, 3),
            "target_ttr": round(target_metrics.ttr, 3),
            "delta_ttr": round(abs(gen_metrics.ttr - target_metrics.ttr), 3),
            "gen_short_ratio": round(gen_metrics.short_sentence_ratio, 3),
            "target_short_ratio": round(target_metrics.short_sentence_ratio, 3),
            "delta_short_ratio": round(delta_short_ratio, 3),
            "gen_entropy": round(gen_metrics.punctuation_entropy, 2),
            "target_entropy": round(target_metrics.punctuation_entropy, 2),
            "delta_entropy": round(delta_entropy, 2),
            "cliche_count": len(detected_cliches),
            "detected_cliches": detected_cliches,
            "ai_penalty": total_penalty,
            "discourse_score": discourse_score,
            "discourse_breakdown": discourse_breakdown,
            "composite_deviation_score": composite_dev,
        }
