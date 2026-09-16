from typing import Dict, List, Tuple
from src.analyzer.stylometrics import StatisticalMetrics, StylometricsAnalyzer


class MetricEvaluator:
    """
    量化评估计算器：
    纯规则与统计指标层，不消耗 Token，毫秒级得出客观指标。
    """

    DEFAULT_AI_CLICHES = [
        "总而言之", "不可否认", "值得一提的是", "宛如", "由此可见",
        "显而易见", "纵观历史", "综上所述", "不难看出", "深入探讨",
        "双刃剑", "画卷", "推向新的高度", "注入了新的活力", "不言而喻"
    ]

    @classmethod
    def evaluate_anti_ai(cls, text: str, custom_forbidden: List[str] = None) -> Tuple[float, float, List[str]]:
        """
        计算【去 AI 味得分】与违规八股惩罚分
        :return: (anti_ai_score 0-100, ai_penalty, detected_cliches)
        """
        forbidden_list = set(cls.DEFAULT_AI_CLICHES)
        if custom_forbidden:
            forbidden_list.update(custom_forbidden)

        detected = []
        for word in forbidden_list:
            if word in text:
                detected.append(word)

        # 每一个违规词扣 10 分
        ai_penalty = len(detected) * 10.0
        anti_ai_score = max(0.0, 100.0 - ai_penalty)
        return anti_ai_score, ai_penalty, detected

    @classmethod
    def evaluate_stylometric_fit(cls, generated_text: str, target_metrics: StatisticalMetrics) -> Tuple[float, Dict[str, float]]:
        """
        计算【统计语言学拟合度】：
        比对生成文本的真实句长均值、标准差（节奏波动）、TTR词汇丰富度与目标指标的吻合程度。
        """
        gen_metrics = StylometricsAnalyzer.analyze(generated_text)
        if not target_metrics or target_metrics.total_sentences == 0:
            return 85.0, {"sentence_len_match": 85.0, "rhythm_std_match": 85.0, "ttr_match": 85.0}

        # 1. 句长偏离惩罚
        len_diff = abs(gen_metrics.avg_sentence_length - target_metrics.avg_sentence_length)
        len_score = max(40.0, 100.0 - (len_diff * 3.5))

        # 2. 节奏波动（标准差）偏离惩罚
        std_diff = abs(gen_metrics.sentence_length_std - target_metrics.sentence_length_std)
        std_score = max(40.0, 100.0 - (std_diff * 3.0))

        # 3. 词汇丰富度 TTR 吻合度
        ttr_diff = abs(gen_metrics.ttr - target_metrics.ttr)
        ttr_score = max(40.0, 100.0 - (ttr_diff * 100.0))

        # 4. 标点熵相似度
        entropy_diff = abs(gen_metrics.punctuation_entropy - target_metrics.punctuation_entropy)
        entropy_score = max(50.0, 100.0 - (entropy_diff * 20.0))

        overall_fit = round(len_score * 0.40 + std_score * 0.35 + ttr_score * 0.15 + entropy_score * 0.10, 1)

        breakdown = {
            "sentence_len_match": round(len_score, 1),
            "rhythm_std_match": round(std_score, 1),
            "ttr_match": round(ttr_score, 1),
            "entropy_match": round(entropy_score, 1),
            "gen_avg_len": gen_metrics.avg_sentence_length,
            "target_avg_len": target_metrics.avg_sentence_length,
            "gen_ttr": gen_metrics.ttr,
            "target_ttr": target_metrics.ttr,
        }
        return overall_fit, breakdown
