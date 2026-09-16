from src.analyzer.stylometrics import StylometricsAnalyzer


def test_stylometrics_basic_stats():
    text = """
说白了，技术从来不是什么壁垒，认知才是。
很多人总以为掌握了一个工具就能改变命运，其实不然！
退一步讲，工具越便宜，人的独立思考越珍贵。你觉得呢？
"""
    metrics = StylometricsAnalyzer.analyze(text)

    assert metrics.total_sentences >= 3
    assert metrics.avg_sentence_length > 0
    assert metrics.sentence_length_std >= 0
    assert metrics.punctuation_entropy > 0
    # 验证捕获到的标志性转折词
    assert "说白了" in metrics.transition_words
    assert "其实" in metrics.transition_words
    assert "退一步讲" in metrics.transition_words
    assert metrics.transition_density > 0


def test_stylometrics_empty():
    metrics = StylometricsAnalyzer.analyze("")
    assert metrics.total_sentences == 0
    assert metrics.avg_sentence_length == 0.0
