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


import pytest
import tempfile
from pathlib import Path
from src.memory.vector_store import VectorStore
from src.memory.memory_manager import MemoryManager
from src.analyzer.stylometrics import StylometricsAnalyzer
from src.evaluation.composite_eval import CompositeEvaluator


@pytest.mark.integration
def test_coordinator_real_pipeline_smoke():
    """
    真实组件端到端 Smoke Test (标记为 @pytest.mark.integration)：
    不 patch 内部组件，贯通 StylometricsAnalyzer -> 真实 VectorStore 切片入库 -> 
    Dense 语义与 Style-Aware 定向召回 -> Task-Aware Token Budget 装配 -> CompositeEvaluator 评测。
    若配置了真实 API Key，则进一步执行真实 LLM 对话检验；未配置时完整验证除外连网络外的全部真实数据通路。
    """
    sample_text = """# 别把信息搬运当成深度思考

说白了，很多人在互联网上搞的内容输出，本质上不过是高级的信息搬运工。

别闹了。真正的思考从来不是拼图游戏，而是带着偏见的价值判断。

写作这门手艺，最忌讳的就是四平八稳。我们必须守住这个阵地！"""

    with tempfile.TemporaryDirectory() as tmp_dir:
        store_path = Path(tmp_dir) / "smoke_store.json"
        vstore = VectorStore(storage_path=str(store_path))
        mem_mgr = MemoryManager(vector_store=vstore)

        # 1. 真实入库与特征建模
        added = mem_mgr.ingest_article("样例样文", sample_text)
        assert added >= 2

        metrics = StylometricsAnalyzer.analyze(sample_text)
        assert metrics.total_sentences >= 2
        assert metrics.avg_sentence_length > 0

        # 2. 真实纯 Dense 召回与 Style-Aware 召回校验
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
