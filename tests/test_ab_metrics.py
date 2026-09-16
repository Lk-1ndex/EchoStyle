from src.analyzer.stylometrics import StatisticalMetrics, StylometricsAnalyzer
from src.evaluation.metrics import MetricEvaluator


def test_calculate_stylometric_deviation():
    # 模拟目标作者的统计基准
    target_text = """真正的思考从来不是拼图游戏，而是带着偏见的价值判断。
别闹了。
写作这门手艺，最忌讳的就是四平八稳。
保持尖锐，保持口语化，保持那种带点自嘲却绝不妥协的语言质感。"""
    target_metrics = StylometricsAnalyzer.analyze(target_text)

    # 1. 拟合良好的仿写文本
    good_text = """说白了，很多人不过是高级信息搬运工。
别装了。
敢于下注的人太少了。
我们必须警惕被平庸的机器算法同化。"""
    good_dev = MetricEvaluator.calculate_stylometric_deviation(good_text, target_metrics)
    assert "delta_avg_len" in good_dev
    assert "delta_std" in good_dev
    assert "delta_ttr" in good_dev
    assert good_dev["cliche_count"] == 0
    assert good_dev["composite_deviation_score"] >= 0.0

    # 2. 充斥 AI 八股和超长官话的文本
    bad_ai_text = """总而言之，不可否认这是一把双刃剑。
纵观历史的长河，值得一提的是，人工智能的发展无疑为传统产业注入了新的活力并推向了新的高度。
由此可见，我们应当深入探讨其内涵，综上所述其意义显而易见且不言而喻。"""
    bad_dev = MetricEvaluator.calculate_stylometric_deviation(bad_ai_text, target_metrics)

    # 验证八股词捕获
    assert bad_dev["cliche_count"] >= 4
    assert "总而言之" in bad_dev["detected_cliches"]
    assert "不可否认" in bad_dev["detected_cliches"]
    assert "双刃剑" in bad_dev["detected_cliches"]

    # 差质量文本的偏离惩罚得分应明显高于好文本
    assert bad_dev["composite_deviation_score"] > good_dev["composite_deviation_score"]
