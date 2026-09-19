import json
from unittest.mock import MagicMock

from src.agents.conversation_agent import ConversationAction, ConversationAgent
from src.agents.coordinator import CoordinatorAgent
from src.agents.state import AgentState
from src.core.config import AppConfig
from src.core.models import (
    AntiPatterns,
    CadenceSyntax,
    DeepStyleProfile,
    DiscourseArchitecture,
    EvaluationReport,
    LexiconRhetoric,
    StyleProfile,
    TonePersona,
)
from src.memory.memory_manager import MemoryManager
from src.memory.vector_store import VectorStore


def make_profile(profile_id: str = "chat-profile") -> DeepStyleProfile:
    return DeepStyleProfile(
        name="对话画像",
        profile_id=profile_id,
        qualitative=StyleProfile(
            tone_persona=TonePersona(perspective="第一人称", emotional_tone="冷静直接"),
            cadence_syntax=CadenceSyntax(sentence_style="长短句交替", paragraph_habit="紧凑"),
            lexicon_rhetoric=LexiconRhetoric(metaphor_style="具体", vocabulary_richness="丰富"),
            discourse=DiscourseArchitecture(opening_hook="设问", body_progression="递进", ending_style="留白"),
            anti_patterns=AntiPatterns(),
        ),
    )


def route_payload(action: str, **overrides) -> str:
    payload = {
        "action": action,
        "topic": "",
        "key_points": "",
        "profile_name": "对话创建的文风档案",
        "word_count": 1500,
        "target_audience": "大众读者",
        "use_active_profile": True,
        "use_documents_as_style": False,
        "use_documents_as_sources": False,
    }
    payload.update(overrides)
    return json.dumps(payload, ensure_ascii=False)


def test_document_question_routes_and_uses_grounded_context():
    provider = MagicMock()
    provider.chat_completion.side_effect = [
        route_payload("document_qa", topic="文章的核心结论是什么", use_documents_as_sources=True),
        "核心结论是边界条件决定响应。[论文A]",
    ]
    agent = ConversationAgent(AppConfig(), MagicMock(), model_provider=provider)
    documents = [
        {
            "title": "论文A",
            "content": "边界条件会改变系统的谱结构。\n\n实验结果表明，非厄米响应由边界条件主导。",
        }
    ]

    result = agent.respond("这篇论文的核心结论是什么？", documents)

    assert result.action == ConversationAction.DOCUMENT_QA
    assert "[论文A]" in result.content
    answer_call = provider.chat_completion.call_args_list[1]
    assert "边界条件" in answer_call.kwargs["user_prompt"]
    assert "不可信数据" in answer_call.kwargs["system_prompt"]


def test_build_style_calls_existing_analyst_workflow():
    provider = MagicMock()
    provider.chat_completion.return_value = route_payload(
        "build_style",
        profile_name="论文作者画像",
        use_documents_as_style=True,
    )
    coordinator = MagicMock()
    profile = make_profile()
    coordinator.build_style.return_value = profile
    agent = ConversationAgent(AppConfig(), coordinator, model_provider=provider)
    documents = [{"title": "样文", "content": "有足够长度的样本文本，用于分析作者风格。"}]

    result = agent.respond("分析这些文章的文风，命名为论文作者画像", documents)

    assert result.action == ConversationAction.BUILD_STYLE
    assert result.profile == profile
    assert "已根据 1 篇样文" in result.content
    assert coordinator.build_style.call_args.kwargs["profile_name"] == "论文作者画像"
    assert isinstance(coordinator.build_style.call_args.kwargs["state"], AgentState)


def test_write_uses_active_profile_and_preserves_report():
    provider = MagicMock()
    provider.chat_completion.return_value = route_payload(
        "write",
        topic="技术与判断力",
        key_points="工具不能替代判断",
        word_count=900,
    )
    coordinator = MagicMock()
    profile = make_profile()
    report = EvaluationReport(overall_score=88.0, anti_ai_score=95.0)
    coordinator.generate_article.return_value = ("最终文章", report, AgentState())
    agent = ConversationAgent(AppConfig(), coordinator, model_provider=provider)

    result = agent.respond("写一篇关于技术与判断力的文章，900字", [], profile=profile)

    assert result.action == ConversationAction.WRITE
    assert result.article == "最终文章"
    assert result.report.overall_score == 88.0
    assert coordinator.generate_article.call_args.kwargs["profile"] == profile
    assert coordinator.generate_article.call_args.kwargs["word_count"] == 900


def test_write_passes_compacted_context_into_controlled_writer():
    provider = MagicMock()
    provider.chat_completion.return_value = route_payload(
        "write",
        topic="技术与判断力",
        key_points="工具不能替代判断",
    )
    coordinator = MagicMock()
    coordinator.generate_article.return_value = ("最终文章", EvaluationReport(overall_score=88.0), AgentState())
    agent = ConversationAgent(AppConfig(), coordinator, model_provider=provider)

    agent.respond(
        "继续写这篇文章",
        [],
        profile=make_profile(),
        history=[{"role": "user", "content": "面向中文读者，语气要克制。"}],
        context_summary="必须保留边界条件这一论点。",
    )

    key_points = coordinator.generate_article.call_args.kwargs["key_points"]
    assert "必须保留边界条件这一论点" in key_points
    assert "面向中文读者" in key_points


def test_revision_passes_user_instruction_to_controlled_workflow():
    provider = MagicMock()
    provider.chat_completion.return_value = route_payload("revise", key_points="第二段更通俗")
    coordinator = MagicMock()
    profile = make_profile()
    report = EvaluationReport(overall_score=91.0)
    coordinator.generate_article.return_value = ("修改后的文章", report, AgentState())
    agent = ConversationAgent(AppConfig(), coordinator, model_provider=provider)

    result = agent.respond(
        "把第二段改得更通俗一些",
        [],
        profile=profile,
        last_article="原文章",
    )

    assert result.action == ConversationAction.REVISE
    assert result.article == "修改后的文章"
    call = coordinator.generate_article.call_args.kwargs
    assert call["initial_draft"] == "原文章"
    assert call["revision_instruction"] == "把第二段改得更通俗一些"


def test_revision_passes_compacted_context_into_controlled_writer():
    provider = MagicMock()
    provider.chat_completion.return_value = route_payload("revise")
    coordinator = MagicMock()
    coordinator.generate_article.return_value = ("修改后的文章", EvaluationReport(overall_score=91.0), AgentState())
    agent = ConversationAgent(AppConfig(), coordinator, model_provider=provider)

    agent.respond(
        "继续修改",
        [],
        profile=make_profile(),
        last_article="原文章",
        context_summary="保留原文的三个论证层次。",
    )

    call = coordinator.generate_article.call_args.kwargs
    assert "保留原文的三个论证层次" in call["revision_instruction"]
    assert "保留原文的三个论证层次" in call["key_points"]


def test_invalid_router_output_falls_back_to_local_intent_rules():
    provider = MagicMock()
    provider.chat_completion.return_value = "not-json"
    coordinator = MagicMock()
    coordinator.build_style.return_value = make_profile()
    agent = ConversationAgent(AppConfig(), coordinator, model_provider=provider)

    result = agent.respond(
        "请分析这些文章的文风并建立画像",
        [{"title": "样文", "content": "一段用于风格分析的正文。"}],
    )

    assert result.action == ConversationAction.BUILD_STYLE
    assert any("本地规则" in log for log in result.logs)
    coordinator.build_style.assert_called_once()


def test_coordinator_applies_revision_before_quality_review(tmp_path):
    config = AppConfig()
    manager = MemoryManager(vector_store=VectorStore(storage_path=str(tmp_path / "memory.json")))
    coordinator = CoordinatorAgent(config, memory_manager=manager)
    coordinator.writer_agent.model_provider.chat_completion = MagicMock(return_value="修改后的完整文章")
    coordinator.critic_agent.judge.evaluate = MagicMock(
        return_value=EvaluationReport(overall_score=92.0, detected_cliches=[])
    )

    article, report, state = coordinator.generate_article(
        profile=make_profile(),
        topic="修改当前稿件",
        state=AgentState(),
        initial_draft="原始文章",
        revision_instruction="把第二段改得更通俗",
    )

    assert article == "修改后的完整文章"
    assert report.overall_score == 92.0
    assert [version.author_agent for version in state.draft_chain] == ["UserDraft", "WriterAgent"]
    assert state.draft_chain[0].content == "原始文章"
    assert "把第二段改得更通俗" in coordinator.writer_agent.model_provider.chat_completion.call_args.kwargs["user_prompt"]
