import re
from typing import Dict, List, Tuple, Any, Optional
from src.analyzer.stylometrics import StylometricsAnalyzer


class LexicalEvaluator:
    """
    词汇与负向套话量化计算器 (Lexical & Anti-AI Metrics)：
    纯规则计算，零 Token 消耗，毫秒级得出词汇丰富度与去 AI 味纯净度。
    """

    DEFAULT_AI_CLICHES = [
        "总而言之", "不可否认", "值得一提的是", "宛如", "由此可见",
        "显而易见", "纵观历史", "综上所述", "不难看出", "深入探讨",
        "双刃剑", "画卷", "推向新的高度", "注入了新的活力", "不言而喻"
    ]

    @classmethod
    def evaluate_cliches(cls, text: str, custom_forbidden: Optional[List[str]] = None) -> Tuple[float, float, List[str]]:
        """
        计算违规八股套话扣分
        :return: (anti_ai_score 0-100, cliche_penalty, detected_cliches)
        """
        forbidden_list = set(cls.DEFAULT_AI_CLICHES)
        if custom_forbidden:
            forbidden_list.update(custom_forbidden)

        detected = []
        for word in forbidden_list:
            if word in text:
                detected.append(word)

        # 每一个违规词扣 10 分
        cliche_penalty = len(detected) * 10.0
        anti_ai_score = max(0.0, 100.0 - cliche_penalty)
        return anti_ai_score, cliche_penalty, detected

    @classmethod
    def evaluate_lexical_authenticity(
        cls,
        text: str,
        target_sttr: float,
        target_unique_ratio: float = 0.65
    ) -> Tuple[float, Dict[str, Any]]:
        """
        计算用词独立性与标准化词汇丰富度拟合分 (0-100)
        """
        metrics = StylometricsAnalyzer.analyze(text)
        current_sttr = metrics.sttr if metrics.sttr > 0 else metrics.ttr
        effective_target = target_sttr if target_sttr > 0 else 0.70

        sttr_diff = abs(current_sttr - effective_target)
        # STTR 偏离惩罚
        sttr_score = max(40.0, 100.0 - (sttr_diff * 120.0))

        breakdown = {
            "gen_sttr": current_sttr,
            "target_sttr": effective_target,
            "delta_sttr": round(sttr_diff, 3),
            "sttr_score": round(sttr_score, 1),
            "total_tokens": metrics.total_chars,
            "unique_words": metrics.unique_words_count,
        }
        return round(sttr_score, 1), breakdown
