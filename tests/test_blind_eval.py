from src.evaluation.blind_eval import BlindEvaluator, PairwiseComparisonResult


def test_wilson_score_ci():
    # 8 次胜场 / 10 次试验
    lower, upper = BlindEvaluator.calculate_wilson_ci(8, 10)
    assert 0.40 <= lower <= 0.60
    assert 0.85 <= upper <= 0.98

    # 边界情况
    zero_l, zero_u = BlindEvaluator.calculate_wilson_ci(0, 0)
    assert zero_l == 0.0 and zero_u == 0.0


def test_evaluate_pair_offline_fallback():
    topic = "论独立思考的价值"
    author_ref = """说白了，很多人在搞高级信息搬运。真正的思考是带着偏见的价值判断。别闹了。"""

    # Candidate: 拟合原作者短句与张力
    cand = """别闹了。
很多人不过是假装专业。真正的写作必须刺痛别人的伪善。这是唯一的阵地！"""

    # Baseline: 充满典型 AI 翻译腔与八股词
    base = """总而言之，不可否认这是一把双刃剑。
人工智能作为一种新兴技术，在当今时代扮演着重要角色，综上所述值得深入探讨。"""

    res = BlindEvaluator.evaluate_pair(topic, cand, base, author_ref, provider=None)
    assert isinstance(res, PairwiseComparisonResult)
    # Candidate 应当由于无八股词且句式拟合度高而胜出
    assert res.winner == "candidate"
    assert res.candidate_score > res.baseline_score


def test_aggregate_results():
    results = [
        PairwiseComparisonResult(task_id="1", topic="T1", winner="candidate", candidate_score=90, baseline_score=60, rationale=""),
        PairwiseComparisonResult(task_id="2", topic="T2", winner="candidate", candidate_score=85, baseline_score=70, rationale=""),
        PairwiseComparisonResult(task_id="3", topic="T3", winner="tie", candidate_score=80, baseline_score=80, rationale=""),
        PairwiseComparisonResult(task_id="4", topic="T4", winner="baseline", candidate_score=70, baseline_score=85, rationale=""),
    ]

    summary = BlindEvaluator.aggregate_results(results)
    assert summary.total_trials == 4
    assert summary.candidate_wins == 2
    assert summary.baseline_wins == 1
    assert summary.ties == 1
    # 胜率: (2 + 0.5 * 1) / 4 = 2.5 / 4 = 0.625
    assert summary.win_rate == 0.625
    assert 0.0 <= summary.ci_lower <= summary.win_rate <= summary.ci_upper <= 1.0


def test_multi_persona_judge_offline_objective():
    """验证 MultiPersonaJudge 离线仲裁不再死板返回固定 winner A，而是进行语言学特征客观仲裁"""
    from src.evaluation.judge import MultiPersonaJudge, EVALUATOR_PANEL

    persona = EVALUATOR_PANEL["GENERAL_READER"]
    ref = "说白了，很多人在搞高级信息搬运。真正的思考是带着偏见的价值判断。别闹了。"
    good_sample = "别闹了。很多人不过是假装专业。真正的写作必须刺痛别人的伪善。这是唯一的阵地！"
    bad_sample = "总而言之，不可否认这是一把双刃剑。人工智能作为一种新兴技术，在当今时代扮演着重要角色，综上所述值得深入探讨。"

    # good as sample A, bad as sample B -> winner should be A
    res_a_wins = MultiPersonaJudge.evaluate_pair_with_persona(
        "独立思考", good_sample, bad_sample, ref, persona, provider=None
    )
    assert res_a_wins["winner"] == "A"
    assert res_a_wins["score_a"] > res_a_wins["score_b"]

    # good as sample B, bad as sample A -> winner should be B (proves it's not hardcoded to A!)
    res_b_wins = MultiPersonaJudge.evaluate_pair_with_persona(
        "独立思考", bad_sample, good_sample, ref, persona, provider=None
    )
    assert res_b_wins["winner"] == "B"
    assert res_b_wins["score_b"] > res_b_wins["score_a"]
