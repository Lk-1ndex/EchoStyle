from unittest.mock import patch
from src.core.config import AppConfig
from src.core.models import (
    DeepStyleProfile,
    StyleProfile,
    TonePersona,
    CadenceSyntax,
    LexiconRhetoric,
    DiscourseArchitecture,
    AntiPatterns,
    StatisticalMetrics,
)
from src.agents.coordinator import CoordinatorAgent
from src.agents.state import AgentState, AgentStatus


def test_coordinator_workflow_integration():
    """工作流集成测试：验证 Coordinator 工具编排、状态机流转、检查点创建与质检决策闭环 (使用 Mock 隔离外部 LLM HTTP 边界)"""
    config = AppConfig()
    coordinator = CoordinatorAgent(config)
    state = AgentState(
        topic="在AI时代坚守个人独特文风",
        key_points="拒绝八股；保持尖锐；文风是人类的数字指纹",
        word_count=800,
    )

    profile = DeepStyleProfile(
        name="测试深度文风",
        qualitative=StyleProfile(
            name="测试",
            tone_persona=TonePersona(perspective="第一人称'我'", emotional_tone="犀利直白", persona_traits=["独立思考者"]),
            cadence_syntax=CadenceSyntax(sentence_style="短句密集", paragraph_habit="简短有力", punctuation_habits=["喜欢破折号"]),
            lexicon_rhetoric=LexiconRhetoric(catchphrases=["说白了"], metaphor_style="生活琐事", vocabulary_richness="通俗口语"),
            discourse=DiscourseArchitecture(opening_hook="直击痛点", body_progression="层层递进", ending_style="留白反思"),
            anti_patterns=AntiPatterns(forbidden_words=["总而言之", "不可否认"])
        ),
        quantitative=StatisticalMetrics(
            total_sentences=10,
            avg_sentence_length=18.5,
            sentence_length_std=8.2,
            ttr=0.85,
            punctuation_entropy=2.1,
            transition_density=12.0
        )
    )

    # 模拟 Writer 生成纯净文章，模拟 Critic 评分为通过 (score 85)
    mock_draft = "说白了，写作者的独特个人风格是机器无法取代的灵魂印记。不要成为算法的复读机。"
    mock_critic_json = '{"style_fidelity": 88, "logic_depth": 85, "human_preference": 86, "radar": {"tone": 88, "cadence": 85, "lexicon": 88, "discourse": 85, "anti_ai": 100}, "critique_feedback": "行文干脆利落，无AI八股。"}'

    with patch.object(coordinator.writer_agent.model_provider, "chat_completion", return_value=mock_draft), \
         patch.object(coordinator.critic_agent.judge.model_provider, "chat_completion", return_value=mock_critic_json):

        # 执行 run()
        final_draft, report, final_state = coordinator.run(
            state=state,
            profile=profile,
        )

        # 1. 验证最终状态正确收敛到 COMPLETED
        assert final_state.current_status == AgentStatus.COMPLETED

        # 2. 验证成文与评测报告有效生成
        assert final_draft == mock_draft
        assert report.overall_score >= 80.0
        assert len(report.detected_cliches) == 0

        # 3. 验证检查点成功创建
        assert "pre_draft" in final_state.checkpoints
        assert "first_draft" in final_state.checkpoints
        assert "best_version" in final_state.checkpoints

        # 4. 验证草稿版本链完整性
        assert len(final_state.draft_chain) >= 1
        assert final_state.draft_chain[0].author_agent == "WriterAgent"
        assert final_state.draft_chain[0].score == report.overall_score


# 保持旧命名兼容
test_coordinator_dynamic_run_e2e = test_coordinator_workflow_integration


def test_condition_d_strict_single_variable_e2e():
    """验证消融实验 Condition D 严格单变量实验机制：复用 C2 稿件与 memory_snapshot，唯一自变量为 Critic 反思重写"""
    config = AppConfig()
    coordinator = CoordinatorAgent(config)

    profile = DeepStyleProfile(
        name="消融测试文风",
        qualitative=StyleProfile(
            name="消融",
            tone_persona=TonePersona(perspective="第一人称", emotional_tone="直白"),
            cadence_syntax=CadenceSyntax(sentence_style="短句", paragraph_habit="紧凑"),
            lexicon_rhetoric=LexiconRhetoric(catchphrases=["说白了"], metaphor_style="生活", vocabulary_richness="通俗"),
            discourse=DiscourseArchitecture(opening_hook="痛点", body_progression="递进", ending_style="金句"),
            anti_patterns=AntiPatterns()
        )
    )

    # 1. 模拟 C2 生成产物
    state_c2 = AgentState(topic="职场表演艺术", key_points="交付结果优先", word_count=500)
    state_c2.memory_snapshot = [{"content": "切片1：不要表演，要结果"}, {"content": "切片2：金句：说白了都是借口"}]
    art_c2 = "说白了，很多人在职场中只是表演型工作，而不是交付真实结果。"

    # 2. 模拟 Condition D 严控单变量：复用 state_c2.memory_snapshot 与 art_c2
    state_d = AgentState(topic="职场表演艺术", key_points="交付结果优先", word_count=500)
    state_d.memory_snapshot = state_c2.memory_snapshot

    # 模拟 Critic 判定首轮需反思 REVISE，第二轮通过 ACCEPT
    critic_report_1 = '{"style_fidelity": 72, "logic_depth": 70, "human_preference": 75, "radar": {"tone": 72, "cadence": 70, "lexicon": 72, "discourse": 70, "anti_ai": 90}, "detected_cliches": ["毋庸讳言"], "critique_feedback": "请删除毋庸讳言"}'
    critic_report_2 = '{"style_fidelity": 88, "logic_depth": 85, "human_preference": 88, "radar": {"tone": 88, "cadence": 85, "lexicon": 88, "discourse": 85, "anti_ai": 100}, "detected_cliches": [], "critique_feedback": "修改到位，通过"}'
    rewritten_draft = "说白了，真正的职场人靠交付结果说话，摒弃虚妄表演。"

    with patch.object(coordinator.critic_agent.judge.model_provider, "chat_completion", side_effect=[critic_report_1, critic_report_2]), \
         patch.object(coordinator.writer_agent.model_provider, "chat_completion", return_value=rewritten_draft):

        art_d, report_d, final_st = coordinator.run(
            state=state_d,
            profile=profile,
            initial_draft=art_c2,
        )

        assert final_st.current_status == AgentStatus.COMPLETED
        assert final_st.retry_count == 1
        assert art_d == rewritten_draft
        assert report_d.overall_score >= 80.0
        # 验证 memory_snapshot 始终为 C2 提供的同一个快照
        assert final_st.memory_snapshot == state_c2.memory_snapshot
        # 验证 draft_chain 中首个版本来自 initial_draft
        assert final_st.draft_chain[0].content == art_c2
        assert final_st.draft_chain[1].content == rewritten_draft


import os
import pytest
import tempfile
from pathlib import Path
from src.memory.vector_store import VectorStore
from src.memory.memory_manager import MemoryManager
from src.analyzer.stylometrics import StylometricsAnalyzer
from src.evaluation.composite_eval import CompositeEvaluator


def test_component_pipeline_smoke():
    """
    真实组件端到端 Smoke Test (无网络隔离依赖)：
    贯通 StylometricsAnalyzer -> 真实 VectorStore 切片入库 -> 
    Dense 语义与 Style-Aware 定向召回 -> Task-Aware Token Budget 装配 -> CompositeEvaluator 评测。
    使用 Mock 向量满足 Fail-Closed 校验，验证除外连网络外的全部真实内部数据通路。
    """
    sample_text = """# 别把信息搬运当成深度思考

说白了，很多人在互联网上搞的内容输出，本质上不过是高级的信息搬运工。

别闹了。真正的思考从来不是拼图游戏，而是带着偏见的价值判断。

写作这门手艺，最忌讳的就是四平八稳。我们必须守住这个阵地！"""

    with tempfile.TemporaryDirectory() as tmp_dir:
        store_path = Path(tmp_dir) / "smoke_store.json"
        vstore = VectorStore(storage_path=str(store_path))
        mem_mgr = MemoryManager(vector_store=vstore)

        # 1. 真实入库与特征建模 (提供 mock 向量以满足 fail-closed 严选约束)
        with patch.object(vstore.model_provider, "get_embeddings", return_value=[[0.1] * 128, [0.2] * 128, [0.3] * 128]):
            added = mem_mgr.ingest_article("样例样文", sample_text)
            assert added >= 2

        metrics = StylometricsAnalyzer.analyze(sample_text)
        assert metrics.total_sentences >= 2
        assert metrics.avg_sentence_length > 0

        # 2. 真实纯 Dense 召回与 Style-Aware 召回校验
        with patch.object(vstore.model_provider, "get_embeddings", return_value=[[0.1] * 128]):
            dense_shots = mem_mgr.retrieve_dense("深度思考与信息搬运", top_k=2)
            assert len(dense_shots) > 0

            few_shots = mem_mgr.retrieve_dynamic_few_shots("深度思考与信息搬运", top_k=3)
            assert len(few_shots) > 0

        # 3. 真实 Token Budget 启发式预算装配链路
        config = AppConfig()
        profile = DeepStyleProfile(
            name="SmokeProfile",
            qualitative=StyleProfile(
                name="SmokeQual",
                tone_persona=TonePersona(perspective="第一人称", emotional_tone="直白", persona_traits=["独立思考"]),
                cadence_syntax=CadenceSyntax(sentence_style="短句", paragraph_habit="紧凑", punctuation_habits=["叹号"]),
                lexicon_rhetoric=LexiconRhetoric(catchphrases=["说白了"], metaphor_style="硬朗", vocabulary_richness="通俗"),
                discourse=DiscourseArchitecture(opening_hook="痛点破空", body_progression="层层递进", ending_style="金句收尾"),
                anti_patterns=AntiPatterns(forbidden_words=["总而言之", "不可否认"])
            ),
            quantitative=metrics
        )

        sys_prompt, user_prompt = vstore.model_provider.assemble_budgeted_prompt(
            system_persona="独立思考写作者",
            style_dna=profile.to_system_prompt(dynamic_few_shots=[]),
            memory_exemplars=few_shots,
            user_task="围绕现代人思考懒惰创作一篇文章",
            task_mode="write",
            retry_count=0
        )
        assert "独立思考" in sys_prompt
        assert len(user_prompt) > 0

        # 4. 真实综合指标评估器执行
        eval_res = CompositeEvaluator.calculate_echoscore(
            generated_text=sample_text,
            target_metrics=metrics,
            target_profile=profile,
            llm_fidelity_score=85.0
        )
        assert 0.0 <= eval_res.echo_score <= 100.0
        assert eval_res.rhythm_match > 0.0


@pytest.mark.integration
def test_coordinator_real_network_integration():
    """
    真正走真实网络的端到端在线集成测试：
    在存在环境变量 ECHOSTYLE_INTEGRATION_API_KEY 时，向真实在线 API 发送请求并驱动 CoordinatorAgent.run() 全链路；
    若未提供该环境变量，则自动 pytest.skip() 跳过。
    """
    api_key = os.getenv("ECHOSTYLE_INTEGRATION_API_KEY")
    if not api_key:
        pytest.skip("未配置 ECHOSTYLE_INTEGRATION_API_KEY，跳过真实在线网络与大模型集成测试")

    config = AppConfig()
    config.llm.api_key = api_key
    base_url = os.getenv("ECHOSTYLE_INTEGRATION_BASE_URL", config.llm.base_url)
    config.llm.base_url = base_url
    coordinator = CoordinatorAgent(config)

    profile = DeepStyleProfile(
        name="RealNetworkIntegrationProfile",
        qualitative=StyleProfile(
            name="在线集成文风",
            tone_persona=TonePersona(perspective="第一人称", emotional_tone="直白犀利", persona_traits=["深刻洞察"]),
            cadence_syntax=CadenceSyntax(sentence_style="短句有力", paragraph_habit="紧凑递进", punctuation_habits=["叹号"]),
            lexicon_rhetoric=LexiconRhetoric(catchphrases=["说白了"], metaphor_style="生活化", vocabulary_richness="丰富"),
            discourse=DiscourseArchitecture(opening_hook="破空设问", body_progression="层层递进", ending_style="金句收尾"),
            anti_patterns=AntiPatterns(forbidden_words=["总而言之", "不可否认"])
        )
    )

    state = AgentState(
        topic="真实在线网络测试主题：数字时代独立思考的价值",
        key_points="拒绝从众；保持真实；敢于发声",
        word_count=500,
    )
    draft, report, final_state = coordinator.run(state=state, profile=profile)
    assert len(draft) > 0
    assert final_state.current_status in [AgentStatus.COMPLETED, AgentStatus.COMPLETED_WITH_WARNING]


def test_coordinator_network_integration_flow_with_mocked_network():
    """验证即使在无外部公网 API 的测试环境下，注入 Profile 后的在线网络集成流程闭环，不发生未传 profile 的 ValueError"""
    from unittest.mock import patch
    from src.core.config import AppConfig

    config = AppConfig()
    config.llm.api_key = "mock-api-key-for-unit-test"
    coordinator = CoordinatorAgent(config)

    profile = DeepStyleProfile(
        name="MockNetworkIntegrationProfile",
        qualitative=StyleProfile(
            name="模拟文风",
            tone_persona=TonePersona(perspective="第一人称", emotional_tone="直白犀利"),
            cadence_syntax=CadenceSyntax(sentence_style="短句有力", paragraph_habit="紧凑递进"),
            lexicon_rhetoric=LexiconRhetoric(catchphrases=["说白了"], metaphor_style="生活化", vocabulary_richness="丰富"),
            discourse=DiscourseArchitecture(opening_hook="破空设问", body_progression="层层递进", ending_style="金句收尾"),
            anti_patterns=AntiPatterns(forbidden_words=["总而言之"])
        )
    )

    state = AgentState(
        topic="网络流程模拟测试：技术与人性",
        key_points="保持理性；拒绝盲从",
        word_count=500,
    )

    mock_article = "说白了，技术的发展不应当让人性退化。我们必须保持独立清醒的思考。"
    mock_judge_json = """{
        "style_fidelity": 88.0,
        "logic_depth": 85.0,
        "human_preference": 90.0,
        "radar": {"tone": 88.0, "lexicon": 88.0, "discourse": 85.0},
        "critique_feedback": "通过验收"
    }"""

    with patch.object(coordinator.writer_agent.model_provider, "chat_completion", return_value=mock_article):
        with patch.object(coordinator.critic_agent.judge.model_provider, "chat_completion", return_value=mock_judge_json):
            draft, report, final_state = coordinator.run(state=state, profile=profile)

            assert len(draft) > 0
            assert draft == mock_article
            assert report.overall_score >= 80.0
            assert final_state.current_status == AgentStatus.COMPLETED


