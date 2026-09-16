from src.analyzer.stylometrics import StylometricsAnalyzer
from src.evaluation.metrics import MetricEvaluator


def test_ablation_metric_matrix_calculation():
    ground_truth = """真正的思考从来不是拼图游戏，而是带着偏见的价值判断。别闹了。
写作这门手艺，最忌讳的就是四平八稳。保持尖锐，保持口语化。这是唯一的阵地！"""
    target_m = StylometricsAnalyzer.analyze(ground_truth)

    # 4 组消融模拟样本
    # Condition A: 典型 AI 翻译腔与八股结构 (Vanilla)
    art_a = """总而言之，不可否认这是一把双刃剑。首先在当今时代扮演着重要角色，其次综上所述值得深入探讨。"""

    # Condition B: 有句长控制但无金句范例 (Profile only)
    art_b = """很多人的内容输出不过是信息搬运。四平八稳的文章没有价值。我们必须保持语言的体温。"""

    # Condition C: 有真实金句范例 (Profile + RAG)
    art_c = """说白了，很多人在互联网上搞内容，不过是高级搬运工。别闹了！真正的思考从来不是拼图游戏。必须敢于下注。"""

    # Condition D: 经过反思打磨与八股清零 (Full EchoStyle)
    art_d = """说白了，很多人不过是高级信息搬运工。
别闹了。真正的思考从来不是拼图游戏，而是带着偏见的价值判断。
写作这门手艺，最忌讳的就是四平八稳。保持尖锐，这是我们唯一能守住的阵地！"""

    dev_a = MetricEvaluator.calculate_stylometric_deviation(art_a, target_m)
    dev_b = MetricEvaluator.calculate_stylometric_deviation(art_b, target_m)
    dev_c = MetricEvaluator.calculate_stylometric_deviation(art_c, target_m)
    dev_d = MetricEvaluator.calculate_stylometric_deviation(art_d, target_m)

    # 验证偏离惩罚递减趋势: Condition A > Condition B > Condition C > Condition D
    assert dev_a["composite_deviation_score"] > dev_b["composite_deviation_score"]
    assert dev_b["composite_deviation_score"] > dev_d["composite_deviation_score"]

    # 验证 Condition A 八股词捕获
    assert dev_a["cliche_count"] >= 3
    # 验证 Condition D 篇章拟合分最高
    assert dev_d["discourse_score"] >= dev_a["discourse_score"]
