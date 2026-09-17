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


def test_hierarchical_statistics_calculation():
    """验证 P1-7 缺陷修复：5 topics x 5 repeats 分层统计严格计算跨主题均值、Student-t CI 与组间/组内方差"""
    from experiments.ablation_study import _calc_hierarchical_stats, HierarchicalStat

    # 构造 5 个 topic，每个 topic 5 个采样值
    # Topic 1: [80, 80, 80, 80, 80] -> mean 80, within_var 0
    # Topic 2: [82, 82, 82, 82, 82] -> mean 82, within_var 0
    # Topic 3: [84, 84, 84, 84, 84] -> mean 84, within_var 0
    # Topic 4: [86, 86, 86, 86, 86] -> mean 86, within_var 0
    # Topic 5: [88, 88, 88, 88, 88] -> mean 88, within_var 0
    topic_runs = [
        [80.0, 80.0, 80.0, 80.0, 80.0],
        [82.0, 82.0, 82.0, 82.0, 82.0],
        [84.0, 84.0, 84.0, 84.0, 84.0],
        [86.0, 86.0, 86.0, 86.0, 86.0],
        [88.0, 88.0, 88.0, 88.0, 88.0],
    ]

    stat = _calc_hierarchical_stats(topic_runs)
    assert isinstance(stat, HierarchicalStat)
    # 跨主题均值应为 (80 + 82 + 84 + 86 + 88) / 5 = 84.0
    assert stat.mean == 84.0
    # 组内采样方差应为 0.0
    assert stat.within_std == 0.0
    # 组间标准差 (sample std of [80, 82, 84, 86, 88]):
    # sum of squares = 16 + 4 + 0 + 4 + 16 = 40 / (5 - 1) = 10 -> sqrt(10) ≈ 3.162
    assert abs(stat.std - 3.16) <= 0.02
    # Student-t (df = 4, t = 2.776): SE = 3.162 / sqrt(5) ≈ 1.414, CI = 2.776 * 1.414 ≈ 3.93
    assert abs(stat.ci95 - 3.93) <= 0.05


def test_independent_evaluator_blind_deterministic_evaluation():
    """验证 P0-3 缺陷修复：IndependentEvaluator 使用 temperature=0.0 进行确定性双盲终审裁决"""
    from unittest.mock import MagicMock
    from src.core.config import LLMConfig
    from src.evaluation.judge import IndependentEvaluator
    from src.core.models import (
        DeepStyleProfile, StyleProfile, TonePersona, CadenceSyntax,
        LexiconRhetoric, DiscourseArchitecture, AntiPatterns, EvaluationReport
    )

    llm_conf = LLMConfig()
    evaluator = IndependentEvaluator(llm_conf)

    profile = DeepStyleProfile(
        name="测试盲审文风",
        qualitative=StyleProfile(
            name="测试",
            tone_persona=TonePersona(perspective="第一人称", emotional_tone="直白"),
            cadence_syntax=CadenceSyntax(sentence_style="短句", paragraph_habit="紧凑"),
            lexicon_rhetoric=LexiconRhetoric(catchphrases=["说白了"], metaphor_style="生活化", vocabulary_richness="通俗"),
            discourse=DiscourseArchitecture(opening_hook="破空设问", body_progression="层层递进", ending_style="金句收尾"),
            anti_patterns=AntiPatterns(forbidden_words=["不可否认"])
        )
    )

    mock_json_response = """{
        "style_fidelity": 88.0,
        "logic_depth": 85.0,
        "human_preference": 90.0,
        "radar": {"tone": 88.0, "lexicon": 88.0, "discourse": 85.0},
        "critique_feedback": "独立第三方盲审合格"
    }"""

    captured_kwargs = {}

    def mock_chat_completion(**kwargs):
        captured_kwargs.update(kwargs)
        return mock_json_response

    evaluator.model_provider.chat_completion = MagicMock(side_effect=mock_chat_completion)

    article = "说白了，现代内容输出不应当变成信息搬运。必须保持独立思考。"
    report = evaluator.evaluate(article, profile)

    assert isinstance(report, EvaluationReport)
    assert report.style_fidelity == 88.0
    # 核心断言：裁判推断必须使用 temperature=0.0 确定性推断
    assert captured_kwargs.get("temperature") == 0.0
    assert captured_kwargs.get("json_mode") is True
    # 核心断言：使用独立盲审 Prompt
    assert "双盲" in captured_kwargs.get("system_prompt", "")


def test_memory_manager_retrieve_hybrid_without_type_filter():
    """验证 P0-1 缺陷修复：MemoryManager.retrieve_hybrid 支持无结构过滤的 RRF 混合召回"""
    import tempfile
    from pathlib import Path
    from unittest.mock import patch
    from src.memory.vector_store import VectorStore
    from src.memory.memory_manager import MemoryManager

    with tempfile.TemporaryDirectory() as tmp_dir:
        store_file = Path(tmp_dir) / "hybrid_mem_store.json"
        vstore = VectorStore(storage_path=str(store_file))
        vstore.chunks = [
            {"id": "c1", "content": "职场汇报与业务交付", "embedding": [0.1, 0.2], "metadata": {"type": "hook"}},
            {"id": "c2", "content": "做题家与标准答案考核", "embedding": [0.2, 0.3], "metadata": {"type": "quote"}},
        ]
        mgr = MemoryManager(vector_store=vstore)

        with patch.object(vstore.model_provider, "get_embeddings", return_value=[[0.1, 0.2]]):
            # retrieve_hybrid 不传 target_type 时，hook 和 quote 均可正常被 RRF 检索召回
            results = mgr.retrieve_hybrid("职场汇报交付", top_k=2, require_dense=True)
            assert len(results) >= 1
            assert "职场汇报与业务交付" in results


def test_run_ablation_study_simulation_pipeline_end_to_end():
    """验证消融实验离线模拟流水线端到端执行与自定义报告路径输出"""
    import tempfile
    from pathlib import Path
    from experiments.ablation_study import run_ablation_study

    with tempfile.TemporaryDirectory() as tmp_dir:
        custom_report = Path(tmp_dir) / "test_ablation_report.md"
        # 运行 2 个主题 × 2 次重复采样离线模拟
        run_ablation_study(
            topics_count=2,
            repeats=2,
            simulate=True,
            output_path=str(custom_report)
        )

        assert custom_report.exists()
        content = custom_report.read_text(encoding="utf-8")
        assert "SIMULATION MODE / NOT A REAL BENCHMARK" in content
        assert "A (Vanilla Base)" in content
        assert "C1a (+Dense RAG)" in content
        assert "C1b (+Hybrid RRF RAG)" in content
        assert "C2 (+Style-Aware RAG)" in content
        assert "D (Full EchoStyle)" in content


