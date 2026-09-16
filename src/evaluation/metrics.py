from typing import Any, Dict, List, Tuple
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
        计算【去 AI 味得分】与综合违规惩罚分：
        涵盖八股词频、句式单调对称性 (Rhythm Monotony) 与典型教科书模板。
        """
        forbidden_list = set(cls.DEFAULT_AI_CLICHES)
        if custom_forbidden:
            forbidden_list.update(custom_forbidden)

        detected = []
        for word in forbidden_list:
            if word in text:
                detected.append(word)

        # 1. 违规词扣分 (每个扣 10 分)
        cliche_penalty = len(detected) * 10.0

        # 2. 句式单调性惩罚 (AI 文风特征：标准差极小，句子长度过于平庸匀称)
        gen_metrics = StylometricsAnalyzer.analyze(text)
        monotony_penalty = 0.0
        if gen_metrics.sentence_length_std < 8.0 and gen_metrics.total_sentences >= 4:
            monotony_penalty = round((8.0 - gen_metrics.sentence_length_std) * 2.5, 1)

        # 3. 典型三段论教科书结构特征惩罚
        template_penalty = 0.0
        if "首先" in text and "其次" in text:
            template_penalty += 10.0
        if "第一，" in text and "第二，" in text:
            template_penalty += 10.0
        if "从宏观来看" in text or "从长远来看" in text:
            template_penalty += 5.0

        total_penalty = round(cliche_penalty + monotony_penalty + template_penalty, 1)
        anti_ai_score = max(0.0, 100.0 - total_penalty)
        return anti_ai_score, total_penalty, detected

    @classmethod
    def evaluate_discourse_fit(cls, generated_text: str, target_profile: Any = None) -> Tuple[float, Dict[str, Any]]:
        """
        篇章结构与行文逻辑拟合度 (Discourse Style Fit)：
        检测开篇切入模式、论证推进张力与收尾结语模式，防范标准 AI 的【定义->解释->总结】八股。
        """
        paragraphs = [p.strip() for p in generated_text.split("\n") if len(p.strip()) > 5]
        if not paragraphs:
            return 50.0, {"opening_type": "empty", "tension_score": 50.0, "ending_type": "empty"}

        opening_para = paragraphs[0]
        ending_para = paragraphs[-1]

        # 1. 开篇结构检测
        opening_score = 80.0
        opening_type = "natural"
        # 惩罚典型的 AI 定义式开篇
        if any(marker in opening_para[:40] for marker in ["是指", "作为一种", "在当今", "扮演着重要角色", "不可忽视"]):
            opening_score = 40.0
            opening_type = "ai_definition_formula"
        elif any(marker in opening_para for marker in ["？", "难道", "说白了", "别", "为什么", "很多人"]):
            opening_score = 95.0
            opening_type = "sharp_hook"

        # 2. 论证张力与立场鲜明度 (Tension & Stance)
        body_text = "\n".join(paragraphs[1:-1]) if len(paragraphs) > 2 else generated_text
        tension_markers = ["必须", "本质上", "绝不", "从来不是", "退一步讲", "恰恰相反", "其实", "刺痛", "懦弱", "偏见"]
        matched_markers = [m for m in tension_markers if m in body_text]
        tension_score = min(100.0, 60.0 + len(matched_markers) * 6.0)

        # 3. 收尾结语结构检测
        ending_score = 80.0
        ending_type = "natural"
        if any(marker in ending_para for marker in ["综上所述", "总而言之", "总的来说", "只要我们", "迎来更加美好的"]):
            ending_score = 35.0
            ending_type = "ai_empty_summary"
        elif any(marker in ending_para for marker in ["！", "唯一", "守住", "阵地", "？", "这就是"]):
            ending_score = 95.0
            ending_type = "punchy_aphorism"

        overall_discourse = round(opening_score * 0.35 + tension_score * 0.40 + ending_score * 0.25, 1)
        breakdown = {
            "opening_score": opening_score,
            "opening_type": opening_type,
            "tension_score": round(tension_score, 1),
            "tension_markers_count": len(matched_markers),
            "ending_score": ending_score,
            "ending_type": ending_type,
        }
        return overall_discourse, breakdown

    @classmethod
    def evaluate_stylometric_fit(cls, generated_text: str, target_metrics: StatisticalMetrics) -> Tuple[float, Dict[str, float]]:
        """
        计算【统计语言学拟合度】：
        比对生成文本的真实句长均值、标准差（节奏波动）、STTR标准化词汇丰富度与目标指标的吻合程度。
        """
        gen_metrics = StylometricsAnalyzer.analyze(generated_text)
        if not target_metrics or target_metrics.total_sentences == 0:
            return 85.0, {"sentence_len_match": 85.0, "rhythm_std_match": 85.0, "ttr_match": 85.0, "sttr_match": 85.0}

        # 1. 句长偏离惩罚
        len_diff = abs(gen_metrics.avg_sentence_length - target_metrics.avg_sentence_length)
        len_score = max(40.0, 100.0 - (len_diff * 3.5))

        # 2. 节奏波动（标准差）偏离惩罚
        std_diff = abs(gen_metrics.sentence_length_std - target_metrics.sentence_length_std)
        std_score = max(40.0, 100.0 - (std_diff * 3.0))

        # 3. 标准化词汇丰富度 STTR 吻合度 (替代易受长度衰减的原始 TTR)
        target_sttr = target_metrics.sttr if target_metrics.sttr > 0 else target_metrics.ttr
        sttr_diff = abs(gen_metrics.sttr - target_sttr)
        sttr_score = max(40.0, 100.0 - (sttr_diff * 120.0))

        # 4. 标点熵相似度
        entropy_diff = abs(gen_metrics.punctuation_entropy - target_metrics.punctuation_entropy)
        entropy_score = max(50.0, 100.0 - (entropy_diff * 20.0))

        overall_fit = round(len_score * 0.35 + std_score * 0.35 + sttr_score * 0.20 + entropy_score * 0.10, 1)

        breakdown = {
            "sentence_len_match": round(len_score, 1),
            "rhythm_std_match": round(std_score, 1),
            "ttr_match": round(sttr_score, 1),
            "sttr_match": round(sttr_score, 1),
            "entropy_match": round(entropy_score, 1),
            "gen_avg_len": gen_metrics.avg_sentence_length,
            "target_avg_len": target_metrics.avg_sentence_length,
            "gen_sttr": gen_metrics.sttr,
            "target_sttr": target_sttr,
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
        结合微观统计指标与宏观篇章结构指标 (Discourse Fit)。
        """
        gen_metrics = StylometricsAnalyzer.analyze(generated_text)
        _, total_penalty, detected_cliches = cls.evaluate_anti_ai(generated_text)
        discourse_score, discourse_breakdown = cls.evaluate_discourse_fit(generated_text, target_profile)

        delta_avg_len = abs(gen_metrics.avg_sentence_length - target_metrics.avg_sentence_length)
        delta_std = abs(gen_metrics.sentence_length_std - target_metrics.sentence_length_std)
        target_sttr = target_metrics.sttr if target_metrics.sttr > 0 else target_metrics.ttr
        delta_sttr = abs(gen_metrics.sttr - target_sttr)
        delta_entropy = abs(gen_metrics.punctuation_entropy - target_metrics.punctuation_entropy)
        delta_short_ratio = abs(gen_metrics.short_sentence_ratio - target_metrics.short_sentence_ratio)

        # 综合偏离惩罚指数 (Composite Stylometric Deviation)
        # 句长偏离 * 1.0 + 节奏差 * 1.5 + STTR偏离 * 50.0 + 短句率偏离 * 20.0 + AI违规总惩罚 * 0.5 + 篇章偏离(100-score)*0.2
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

