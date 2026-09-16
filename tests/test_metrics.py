from src.evaluation.metrics import MetricEvaluator
from src.analyzer.stylometrics import StylometricsAnalyzer


def test_anti_ai_score():
    # 含有两个违规词
    bad_text = "总而言之，这一现象不可否认是时代发展的产物。"
    score, penalty, detected = MetricEvaluator.evaluate_anti_ai(bad_text)
    assert "总而言之" in detected
    assert "不可否认" in detected
    assert penalty == 20.0
    assert score == 80.0  # 100 - 10*2 = 80.0

    # 纯净的人类表达
    good_text = "说白了，这就是一次纯粹的观念碰撞，别想得太复杂。"
    clean_score, clean_penalty, clean_detected = MetricEvaluator.evaluate_anti_ai(good_text)
    assert clean_score == 100.0
    assert clean_penalty == 0.0
    assert len(clean_detected) == 0


def test_stylometric_similarity():
    target = StylometricsAnalyzer.analyze("短句好。节奏快。不啰嗦。很有力。")
    generated = "短句好。节奏快。很有力。继续写。"
    fit_score, breakdown = MetricEvaluator.evaluate_stylometric_fit(generated, target)
    assert fit_score > 70.0
    assert "sentence_len_match" in breakdown
    assert "ttr_match" in breakdown
