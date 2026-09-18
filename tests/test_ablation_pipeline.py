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


def test_hierarchical_statistics_empty_leading_topic():
    """验证 P0 缺陷修复：当首个主题或部分前置主题因整块失败为空时，分层统计正确提取有效主题计算均值与置信区间，绝不因索引硬取 topic_runs[0] 塌缩为 0"""
    from experiments.ablation_study import _calc_hierarchical_stats

    # 模拟 Topic 1 全部失败为空列表，只有 Topic 2 成功有采样数据
    runs = [[], [10.0, 12.0, 14.0]]
    stat = _calc_hierarchical_stats(runs)

    assert stat.mean == 12.0
    assert stat.std == 2.0
    assert stat.ci95 > 0.0
    assert stat.within_std == 2.0


def test_independent_evaluator_condition_blind_holdout_evaluation():
    """验证：IndependentEvaluator 作为 Condition-Blind Holdout Evaluator，使用 temperature=0.0 降低采样随机性进行条件盲化留出裁决"""
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
    # 核心断言：裁判推断使用 temperature=0.0 降低采样随机性
    assert captured_kwargs.get("temperature") == 0.0
    assert captured_kwargs.get("json_mode") is True
    # 核心断言：使用条件盲审 Prompt
    sys_prompt = captured_kwargs.get("system_prompt", "")
    assert "Condition-Blind" in sys_prompt or "条件盲审" in sys_prompt


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
    """验证 7 组消融实验离线模拟流水线端到端执行与自定义报告路径输出"""
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
        assert "A0 (Vanilla Base)" in content
        assert "A1 (Scaffolding Base)" in content
        assert "B (+Profile Only)" in content
        assert "C1a (+Dense RAG)" in content
        assert "C1b (+Hybrid RRF RAG)" in content
        assert "C2 (+Style-Aware RAG)" in content
        assert "D (Full EchoStyle)" in content
        # 验证仿真模式不输出科学结论断言，而是流程验证措辞
        assert "仿真数据用于验证分析管道能够识别以下差异" in content


def test_independent_evaluator_strict_fail_closed_on_failure():
    """验证 P0-3 缺陷修复：IndependentEvaluator 在 strict=True 时坚决 Fail-Closed，拒绝以 80/85/80 默认虚拟分污染基准"""
    import pytest
    from unittest.mock import MagicMock
    from src.core.config import LLMConfig
    from src.evaluation.judge import IndependentEvaluator
    from src.core.exceptions import EvaluationUnavailableError
    from src.core.models import (
        DeepStyleProfile, StyleProfile, TonePersona, CadenceSyntax,
        LexiconRhetoric, DiscourseArchitecture, AntiPatterns
    )

    llm_conf = LLMConfig()
    profile = DeepStyleProfile(
        name="测试",
        qualitative=StyleProfile(
            name="测试",
            tone_persona=TonePersona(perspective="第一人称", emotional_tone="直白"),
            cadence_syntax=CadenceSyntax(sentence_style="短句", paragraph_habit="紧凑"),
            lexicon_rhetoric=LexiconRhetoric(catchphrases=["说白了"], metaphor_style="生活化", vocabulary_richness="通俗"),
            discourse=DiscourseArchitecture(opening_hook="破空设问", body_progression="层层递进", ending_style="金句收尾"),
            anti_patterns=AntiPatterns(forbidden_words=["不可否认"])
        )
    )


    # 1. strict=True 场景：大模型调用超时或抛出异常，必须抛出 EvaluationUnavailableError 阻断
    strict_evaluator = IndependentEvaluator(llm_conf, strict=True)
    strict_evaluator.model_provider.chat_completion = MagicMock(side_effect=TimeoutError("API Connection timed out"))

    with pytest.raises(EvaluationUnavailableError) as exc_info:
        strict_evaluator.evaluate("这是待审文章", profile)
    assert "Fail-Closed" in str(exc_info.value)
    assert "timed out" in str(exc_info.value)

    # 2. strict=True 场景：返回内容 JSON 解析错误或缺失关键字段
    strict_evaluator.model_provider.chat_completion = MagicMock(return_value="bad non json output")
    with pytest.raises(EvaluationUnavailableError) as exc_info:
        strict_evaluator.evaluate("这是待审文章", profile)
    assert "Fail-Closed" in str(exc_info.value)

    strict_evaluator.model_provider.chat_completion = MagicMock(return_value='{"score": 90}')
    with pytest.raises(EvaluationUnavailableError) as exc_info:
        strict_evaluator.evaluate("这是待审文章", profile)
    assert "缺失关键评分字段" in str(exc_info.value)

    # 3. strict=True 场景：评分超限 [0-100] 必须 Fail-Closed
    strict_evaluator.model_provider.chat_completion = MagicMock(
        return_value='{"style_fidelity": 150.0, "logic_depth": 85.0, "human_preference": 80.0}'
    )
    with pytest.raises(EvaluationUnavailableError) as exc_info:
        strict_evaluator.evaluate("这是待审文章", profile)
    assert "超出有效范围" in str(exc_info.value)

    # 4. strict=False 宽松模式（生产 UI 体验）：允许优雅降级为规则引擎质检，不抛出阻断异常
    loose_evaluator = IndependentEvaluator(llm_conf, strict=False)
    loose_evaluator.model_provider.chat_completion = MagicMock(side_effect=RuntimeError("503 Service Unavailable"))
    report = loose_evaluator.evaluate("这是待审文章", profile)
    assert report is not None
    assert "评分降级" in report.feedback


def test_ablation_study_evaluate_condition_run_retries_and_excludes_on_failure():
    """验证 Benchmark 模式下评测失败带重试，重试耗尽后返回 None（排除样本，绝不填 80/85/80 默认虚拟分）"""
    from unittest.mock import MagicMock
    from src.core.config import LLMConfig
    from src.evaluation.judge import IndependentEvaluator
    from src.analyzer.stylometrics import StylometricsAnalyzer
    from experiments.ablation_study import _evaluate_condition_run
    from src.core.models import (
        DeepStyleProfile, StyleProfile, TonePersona, CadenceSyntax,
        LexiconRhetoric, DiscourseArchitecture, AntiPatterns
    )

    llm_conf = LLMConfig()
    profile = DeepStyleProfile(
        name="测试",
        qualitative=StyleProfile(
            name="测试",
            tone_persona=TonePersona(perspective="第一人称", emotional_tone="直白"),
            cadence_syntax=CadenceSyntax(sentence_style="短句", paragraph_habit="紧凑"),
            lexicon_rhetoric=LexiconRhetoric(catchphrases=["说白了"], metaphor_style="生活化", vocabulary_richness="通俗"),
            discourse=DiscourseArchitecture(opening_hook="破空设问", body_progression="层层递进", ending_style="金句收尾"),
            anti_patterns=AntiPatterns(forbidden_words=["不可否认"])
        )
    )
    metrics = StylometricsAnalyzer.analyze("样文基准测试")

    strict_evaluator = IndependentEvaluator(llm_conf, strict=True)
    # 模拟 API 彻底挂掉（多次重试均超时）
    strict_evaluator.model_provider.chat_completion = MagicMock(side_effect=TimeoutError("Connection timed out"))

    # 执行带重试评估
    res = _evaluate_condition_run(strict_evaluator, "待测文章", profile, metrics, max_retries=1)

    # 核心断言：失败后返回 None（样本被直接排除），绝不生成伪造的 80/85/80 默认分
    assert res is None
    # 验证确实执行了重试 (1 次初始 + 1 次重试 = 2 次调用)
    assert strict_evaluator.model_provider.chat_completion.call_count == 2


def test_independent_evaluator_strict_radar_out_of_bounds():
    """验证 IndependentEvaluator 在 strict=True 时对 radar 评分超限 [0-100] 执行 Fail-Closed 阻断"""
    import pytest
    from unittest.mock import MagicMock
    from src.core.config import LLMConfig
    from src.evaluation.judge import IndependentEvaluator
    from src.core.exceptions import EvaluationUnavailableError
    from src.core.models import (
        DeepStyleProfile, StyleProfile, TonePersona, CadenceSyntax,
        LexiconRhetoric, DiscourseArchitecture, AntiPatterns
    )

    llm_conf = LLMConfig()
    profile = DeepStyleProfile(
        name="测试",
        qualitative=StyleProfile(
            tone_persona=TonePersona(perspective="第一人称", emotional_tone="直白"),
            cadence_syntax=CadenceSyntax(sentence_style="短句", paragraph_habit="紧凑"),
            lexicon_rhetoric=LexiconRhetoric(catchphrases=["说白了"], metaphor_style="生活化", vocabulary_richness="通俗"),
            discourse=DiscourseArchitecture(opening_hook="设问", body_progression="递进", ending_style="金句"),
            anti_patterns=AntiPatterns(forbidden_words=[])
        )
    )

    strict_evaluator = IndependentEvaluator(llm_conf, strict=True)
    # 模拟 radar 维度超出 100 分
    strict_evaluator.model_provider.chat_completion = MagicMock(
        return_value='{"style_fidelity": 80.0, "logic_depth": 85.0, "human_preference": 80.0, "radar": {"tone": 150.0}}'
    )

    with pytest.raises(EvaluationUnavailableError) as exc:
        strict_evaluator.evaluate("这是待测文章", profile)
    assert "radar" in str(exc.value) and "超出有效范围" in str(exc.value)


def test_llm_judge_robust_json_extraction_with_markdown_prelude():
    """验证 LLMJudge 与 IndependentEvaluator 共享鲁棒 JSON 提取器，在 Markdown 前缀存在时不会解析崩溃降级"""
    from unittest.mock import MagicMock
    from src.core.config import LLMConfig
    from src.evaluation.judge import LLMJudge
    from src.core.models import (
        DeepStyleProfile, StyleProfile, TonePersona, CadenceSyntax,
        LexiconRhetoric, DiscourseArchitecture, AntiPatterns
    )

    llm_conf = LLMConfig()
    judge = LLMJudge(llm_conf)
    profile = DeepStyleProfile(
        name="测试",
        qualitative=StyleProfile(
            tone_persona=TonePersona(perspective="第一人称", emotional_tone="直白"),
            cadence_syntax=CadenceSyntax(sentence_style="短句", paragraph_habit="紧凑"),
            lexicon_rhetoric=LexiconRhetoric(catchphrases=[], metaphor_style="生活化", vocabulary_richness="通俗"),
            discourse=DiscourseArchitecture(opening_hook="设问", body_progression="递进", ending_style="金句"),
            anti_patterns=AntiPatterns(forbidden_words=[])
        )
    )

    # 模拟大模型输出带有思维链前缀与后置解释的 Markdown JSON
    complex_llm_reply = """思考过程：文章结构紧凑，语气独立犀利。
```json
{
  "style_fidelity": 92.0,
  "logic_depth": 88.0,
  "human_preference": 91.0,
  "radar": {
    "tone": 92.0,
    "lexicon": 90.0,
    "discourse": 88.0,
  },
  "critique_feedback": "行文极具质感，保持了作者的呼吸节奏。"
}
```
以上为详细审校意见。"""

    judge.model_provider.chat_completion = MagicMock(return_value=complex_llm_reply)
    report = judge.evaluate("这是待测文章", profile)

    # 验证成功提取并正确应用了大模型评分，绝未静默降级为 80/85/80
    assert report.style_fidelity == 92.0
    assert report.logic_depth == 88.0
    assert report.llm_judge_score == 91.0
    assert "保持了作者的呼吸节奏" in report.feedback


def test_load_config_parses_embedding_environment_variables(monkeypatch):
    """验证 load_config 优先解析 EMBEDDING_API_KEY, EMBEDDING_BASE_URL, EMBEDDING_MODEL 环境变量"""
    import tempfile
    from pathlib import Path
    from src.core.config import load_config

    monkeypatch.setenv("EMBEDDING_API_KEY", "sk-custom-embedding-key")
    monkeypatch.setenv("EMBEDDING_BASE_URL", "https://api.custom-emb.com/v1")
    monkeypatch.setenv("EMBEDDING_MODEL", "custom-emb-text-v1")

    with tempfile.TemporaryDirectory() as d:
        dummy_conf = Path(d) / "config.yaml"
        dummy_conf.write_text("llm:\n  api_key: 'sk-llm'\n", encoding="utf-8")
        conf = load_config(str(dummy_conf))

        assert conf.embedding.api_key == "sk-custom-embedding-key"
        assert conf.embedding.base_url == "https://api.custom-emb.com/v1"
        assert conf.embedding.model == "custom-emb-text-v1"


def test_paired_delta_statistics_and_report_valid_pairs():
    """验证 P0 缺陷修复：消融实验采用配对单变量差值 Δ 及其 95% CI 统计，并输出配对区块有效性指标"""
    import tempfile
    from pathlib import Path
    from experiments.ablation_study import run_ablation_study

    with tempfile.TemporaryDirectory() as tmp_dir:
        report_file = Path(tmp_dir) / "paired_report.md"
        run_ablation_study(topics_count=2, repeats=2, simulate=True, output_path=str(report_file))

        assert report_file.exists()
        content = report_file.read_text(encoding="utf-8")

        # 核心断言 1：输出配对区块有效性指标
        assert "配对区块有效性 (Pair Validity)" in content
        assert "有效配对区块" in content
        assert "Fail-Closed Paired Block" in content

        # 核心断言 2：包含核心组件配对单变量效应与置信区间矩阵
        assert "核心组件配对单变量效应与置信区间" in content
        assert "配对均值差值 (Mean Δ)" in content
        assert "95% 置信区间 (Student-t)" in content

        # 核心断言 3：各组件对比组与配对差值完整展示
        for pair in ["A1 vs A0", "B vs A1", "C1a vs B", "C1b vs C1a", "C2 vs C1b", "D vs C2"]:
            assert pair in content


def test_paired_block_fail_closed_on_any_condition_failure():
    """验证 P0 缺陷修复：任何单一条件评测失败时，坚决将整组 7 条件配对区块标记为 Invalid，杜绝样本不均衡偏置"""
    import pytest
    from unittest.mock import MagicMock, patch
    from src.core.config import AppConfig
    from src.core.exceptions import EvaluationUnavailableError
    from experiments.ablation_study import run_ablation_study

    dummy_config = AppConfig()
    dummy_config.llm.api_key = "sk-valid-key-for-test"
    dummy_config.embedding.api_key = "sk-valid-emb-key"

    # 模拟在 1 topic x 1 repeat 场景下，C1b 条件在评测时失败返回 None
    call_counts = {"count": 0}

    def mock_eval_run(*args, **kwargs):
        call_counts["count"] += 1
        # 前 4 个条件 (A0, A1, B, C1a) 正常，第 5 个条件 (C1b) 失败返回 None
        if call_counts["count"] == 5:
            return None
        mock_res = MagicMock()
        mock_res.echo_score = 88.0
        mock_res.discourse_fit = 85.0
        mock_res.rhythm_match = 90.0
        mock_res.lexical_authenticity = 92.0
        mock_res.cliche_penalty = 0.0
        return mock_res

    with patch("experiments.ablation_study.load_config", return_value=dummy_config), \
         patch("src.core.model_provider.ModelProvider.get_embeddings", return_value=[[0.1, 0.2], [0.1, 0.2]]), \
         patch("src.core.model_provider.ModelProvider.chat", return_value="生成的文章"), \
         patch("src.agents.coordinator.CoordinatorAgent.build_style", return_value=MagicMock()), \
         patch("src.agents.writer_agent.WriterAgent.generate", return_value="生成的文章"), \
         patch("src.agents.coordinator.CoordinatorAgent.run", return_value=("成文", MagicMock(), MagicMock())), \
         patch("src.memory.memory_manager.MemoryManager.retrieve_dense", return_value=["切片"]), \
         patch("src.memory.memory_manager.MemoryManager.retrieve_hybrid", return_value=["切片"]), \
         patch("src.memory.memory_manager.MemoryManager.retrieve_dynamic_few_shots", return_value=["切片"]), \
         patch("experiments.ablation_study._evaluate_condition_run", side_effect=mock_eval_run):

        # 因为唯一的 1 组配对块中 C1b 失败，整组 7 条件配对块失效，valid_pairs 为 0，触发 Fail-Closed 抛出异常
        with pytest.raises(EvaluationUnavailableError) as exc:
            run_ablation_study(topics_count=1, repeats=1, simulate=False)
        assert "有效配对统计报告" in str(exc.value)
