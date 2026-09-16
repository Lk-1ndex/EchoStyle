import math
from typing import Dict, Any, Tuple
from src.core.models import StatisticalMetrics
from src.analyzer.stylometrics import StylometricsAnalyzer


class RhythmEvaluator:
    """
    行文节奏与句法离散度评估器 (Rhythm & Cadence Metrics)：
    核心原则：尊重作者自身节奏特征，计算与作者真实节奏的偏离度 (Rhythm Deviation)，
    而非盲目假设'方差越大越好'（避免误伤长句多、节奏稳重的学术或严肃型写作者）。
    """

    @classmethod
    def evaluate_rhythm_fit(
        cls,
        generated_text: str,
        target_metrics: StatisticalMetrics,
    ) -> Tuple[float, Dict[str, Any]]:
        """
        计算节奏与句长拟合得分 (0-100)：
        - 句长均值偏离惩罚: |AvgLen_gen - AvgLen_target|
        - 句长离散度(波长起伏)偏离惩罚: |Std_gen - Std_target|
        - 爆发力短句率偏离惩罚: |ShortRatio_gen - ShortRatio_target|
        - 标点多样性熵偏离惩罚: |Entropy_gen - Entropy_target|
        """
        gen_m = StylometricsAnalyzer.analyze(generated_text)
        if not target_metrics or target_metrics.total_sentences == 0:
            return 85.0, {
                "sentence_len_score": 85.0,
                "std_score": 85.0,
                "short_ratio_score": 85.0,
                "entropy_score": 85.0,
                "delta_avg_len": 0.0,
                "delta_std": 0.0,
            }

        # 1. 均值句长偏离度
        delta_len = abs(gen_m.avg_sentence_length - target_metrics.avg_sentence_length)
        len_score = max(30.0, 100.0 - (delta_len * 3.5))

        # 2. 节奏波动离散度偏离惩罚 (客观比对目标作者真实标准差)
        delta_std = abs(gen_m.sentence_length_std - target_metrics.sentence_length_std)
        std_score = max(30.0, 100.0 - (delta_std * 3.5))

        # 3. 极短句爆发力比率偏离惩罚
        delta_short = abs(gen_m.short_sentence_ratio - target_metrics.short_sentence_ratio)
        short_score = max(40.0, 100.0 - (delta_short * 80.0))

        # 4. 标点符号熵偏离惩罚
        delta_entropy = abs(gen_m.punctuation_entropy - target_metrics.punctuation_entropy)
        entropy_score = max(40.0, 100.0 - (delta_entropy * 25.0))

        overall_rhythm = round(
            len_score * 0.40 +
            std_score * 0.35 +
            short_score * 0.15 +
            entropy_score * 0.10,
            1
        )

        breakdown = {
            "gen_avg_len": round(gen_m.avg_sentence_length, 2),
            "target_avg_len": round(target_metrics.avg_sentence_length, 2),
            "delta_avg_len": round(delta_len, 2),
            "sentence_len_score": round(len_score, 1),
            "gen_std": round(gen_m.sentence_length_std, 2),
            "target_std": round(target_metrics.sentence_length_std, 2),
            "delta_std": round(delta_std, 2),
            "std_score": round(std_score, 1),
            "gen_short_ratio": round(gen_m.short_sentence_ratio, 3),
            "target_short_ratio": round(target_metrics.short_sentence_ratio, 3),
            "delta_short_ratio": round(delta_short, 3),
            "short_ratio_score": round(short_score, 1),
            "gen_entropy": round(gen_m.punctuation_entropy, 2),
            "target_entropy": round(target_metrics.punctuation_entropy, 2),
            "delta_entropy": round(delta_entropy, 2),
            "entropy_score": round(entropy_score, 1),
        }

        return overall_rhythm, breakdown
