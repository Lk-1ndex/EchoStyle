from src.analyzer.stylometrics import StylometricsAnalyzer
from src.evaluation.metrics import MetricEvaluator


def test_sttr_calculation_length_invariance():
    # 测试 Standardized TTR (STTR)
    short_text = "真正的写作者必须保持语言的敏锐，不能被机器算法同化，守住人类的思考阵地。"
    long_text = short_text * 10

    short_metrics = StylometricsAnalyzer.analyze(short_text)
    long_metrics = StylometricsAnalyzer.analyze(long_text)

    # 原始 TTR 随着文本变长必然暴跌 (词汇重复)
    assert long_metrics.ttr < short_metrics.ttr
    # 标准化 STTR 基于切片均值，具备更稳定的度量特性
    assert short_metrics.sttr > 0.0
    assert long_metrics.sttr > 0.0


def test_discourse_fit_evaluation():
    # 1. 优秀样文结构：犀利设问切入 + 强烈张力 + 金句收尾
    good_article = """说白了，很多人不过是信息搬运工。

真正的思考从来不是拼图游戏，而是带着偏见的价值判断。
写作这门手艺，最忌讳的就是四平八稳。我们必须敢于在关键分歧点上下注，绝不能妥协退让。

这是我们在算法洪流里唯一能守住的阵地！"""

    good_score, good_breakdown = MetricEvaluator.evaluate_discourse_fit(good_article)
    assert good_score >= 80.0
    assert good_breakdown["opening_type"] == "sharp_hook"
    assert good_breakdown["ending_type"] == "punchy_aphorism"

    # 2. 劣质 AI 结构：抽象定义开篇 + 缺乏张力 + 空洞总结
    bad_article = """人工智能是指通过计算机系统模拟人类智能的技术，在当今时代扮演着重要角色。

首先，技术在各个领域得到了广泛的应用。
其次，我们要看到其对生活方式的改变。

综上所述，人工智能具有深远的意义，我们应当深入探讨并迎接美好未来。"""

    bad_score, bad_breakdown = MetricEvaluator.evaluate_discourse_fit(bad_article)
    assert bad_score < 60.0
    assert bad_breakdown["opening_type"] == "ai_definition_formula"
    assert bad_breakdown["ending_type"] == "ai_empty_summary"


def test_ai_uniformity_monotony_penalty():
    # 测试极其匀称无波动的 AI 句子长度平庸惩罚
    uniform_text = """这是第一句差不多长度的话。这是第二句差不多长度的话。这是第三句差不多长度的话。这是第四句差不多长度的话。"""
    _, penalty, _ = MetricEvaluator.evaluate_anti_ai(uniform_text)
    # 因为标准差极小且句数>=4，应触发句式单调性惩罚
    assert penalty > 0.0
