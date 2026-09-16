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


def test_coordinator_dynamic_run_e2e():
    """验证 Coordinator 动态任务规划、工具调用、状态流转与质检闭环"""
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
