import json
from unittest.mock import MagicMock, patch

import pytest

from src.agents.conversation_agent import ConversationAction, ConversationAgent
from src.agents.coordinator import CoordinatorAgent
from src.agents.state import AgentState
from src.core.config import AppConfig
from src.core.model_provider import ModelProvider
from src.core.exceptions import IncompleteGenerationError
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


def test_chat_passes_more_than_six_recent_messages_when_the_model_has_room():
    provider = MagicMock()
    provider.input_token_budget.return_value = 100_000
    provider.count_tokens.side_effect = lambda text: len(text)
    provider.truncate_tokens.side_effect = lambda text, max_tokens: text[:max_tokens]
    provider.chat_completion.side_effect = [route_payload("chat"), "继续回答"]
    agent = ConversationAgent(AppConfig(), MagicMock(), model_provider=provider)
    history = [
        {"role": "user" if index % 2 == 0 else "assistant", "content": f"history-turn-{index}"}
        for index in range(20)
    ]

    agent.respond("继续", [], history=history)

    answer_prompt = provider.chat_completion.call_args_list[1].kwargs["user_prompt"]
    assert "history-turn-0" in answer_prompt
    assert "history-turn-19" in answer_prompt


def test_document_context_can_expand_beyond_the_old_character_cap():
    provider = MagicMock()
    provider.count_tokens.side_effect = lambda text: len(text)
    provider.truncate_tokens.side_effect = lambda text, max_tokens: text[:max_tokens]
    agent = ConversationAgent(AppConfig(), MagicMock(), model_provider=provider)
    content = ("long-document-section " * 3_000) + "TAIL_SENTINEL"

    context = agent._build_document_context(
        [{"title": "长文档", "content": content}],
        "请查找文档内容",
        max_tokens=100_000,
    )

    assert len(context) > 48_000
    assert "TAIL_SENTINEL" in context


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


def test_explicit_ten_thousand_character_request_overrides_router_default():
    provider = MagicMock()
    provider.chat_completion.return_value = route_payload("write", topic="城市治理", word_count=1500)
    coordinator = MagicMock()
    coordinator.generate_article.return_value = ("完整文章", EvaluationReport(overall_score=90), AgentState())
    agent = ConversationAgent(AppConfig(), coordinator, model_provider=provider)

    agent.respond("写一篇城市治理文章，1w字左右", [], profile=make_profile())

    assert coordinator.generate_article.call_args.kwargs["word_count"] == 10_000


def test_fallback_router_keeps_explicit_long_word_count():
    provider = MagicMock()
    provider.chat_completion.return_value = "not-json"
    coordinator = MagicMock()
    coordinator.generate_article.return_value = ("完整文章", EvaluationReport(overall_score=90), AgentState())
    agent = ConversationAgent(AppConfig(), coordinator, model_provider=provider)

    agent.respond("请写一篇一万字的文章", [], profile=make_profile())

    assert coordinator.generate_article.call_args.kwargs["word_count"] == 10_000


def test_writer_continues_before_critic_reviews_long_draft(tmp_path):
    config = AppConfig()
    manager = MemoryManager(vector_store=VectorStore(storage_path=str(tmp_path / "memory.json")))
    coordinator = CoordinatorAgent(config, memory_manager=manager)
    provider = coordinator.writer_agent.model_provider
    chunks = ["甲" * 3000, "乙" * 3000, "丙" * 3000, "丁" * 1000]

    def complete(**kwargs):
        provider.last_finish_reason = "length" if len(chunks) > 1 else "stop"
        return chunks.pop(0)

    state = AgentState(memory_snapshot=[])
    with patch.object(provider, "chat_completion", side_effect=complete) as chat, \
         patch.object(coordinator.critic_agent.judge, "evaluate", return_value=EvaluationReport(overall_score=90)) as judge:
        article, _, _ = coordinator.generate_article(
            profile=make_profile(), topic="城市治理", word_count=10_000, state=state
        )

    assert len(article.replace("\n", "")) == 10_000
    assert chat.call_count == 4
    assert judge.call_args.args[0] == article
    assert "甲" * 3000 in chat.call_args_list[1].kwargs["user_prompt"]


def test_incomplete_long_draft_is_not_sent_to_critic(tmp_path):
    config = AppConfig()
    manager = MemoryManager(vector_store=VectorStore(storage_path=str(tmp_path / "memory.json")))
    coordinator = CoordinatorAgent(config, memory_manager=manager)
    provider = coordinator.writer_agent.model_provider
    with patch.object(provider, "chat_completion", return_value="同一段正文"), \
         patch.object(coordinator.critic_agent.judge, "evaluate") as judge:
        with pytest.raises(IncompleteGenerationError) as error:
            coordinator.generate_article(
                profile=make_profile(), topic="城市治理", word_count=10_000,
                state=AgentState(memory_snapshot=[]),
            )

    assert error.value.partial_text == "同一段正文"
    judge.assert_not_called()


def test_long_form_continues_after_an_early_natural_stop():
    provider = ModelProvider(AppConfig().llm)
    chunks = ["甲" * 4000, "乙" * 5500]

    def complete(**kwargs):
        provider.last_finish_reason = "stop"
        return chunks.pop(0)

    with patch.object(provider, "chat_completion", side_effect=complete) as chat:
        article = provider.generate_long_form("system", "user", target_chars=10_000)

    assert len(article.replace("\n", "")) >= 9_500
    assert chat.call_count == 2
    assert "甲" * 4000 in chat.call_args_list[1].kwargs["user_prompt"]


@pytest.mark.parametrize("target_chars", [500, 1000, 1500])
def test_long_form_enforces_short_targets_after_natural_stop(target_chars):
    provider = ModelProvider(AppConfig().llm)
    chunks = ["甲" * 100, "乙" * (target_chars - 100)]

    def complete(**kwargs):
        provider.last_finish_reason = "stop"
        return chunks.pop(0)

    with patch.object(provider, "chat_completion", side_effect=complete) as chat:
        article = provider.generate_long_form("system", "user", target_chars=target_chars)

    assert len("".join(article.split())) >= int(target_chars * 0.95)
    assert chat.call_count == 2


def test_long_form_requires_natural_finish_after_length_cutoff():
    provider = ModelProvider(AppConfig().llm)
    chunks = ["甲" * 500, "完整结尾。"]

    def complete(**kwargs):
        provider.last_finish_reason = "length" if len(chunks) == 2 else "stop"
        return chunks.pop(0)

    with patch.object(provider, "chat_completion", side_effect=complete) as chat:
        article = provider.generate_long_form("system", "user", target_chars=500)

    assert article.endswith("完整结尾。")
    assert chat.call_count == 2


def test_long_form_removes_a_repeated_full_draft_prefix():
    provider = ModelProvider(AppConfig().llm)
    first_part = "甲" * 4000
    chunks = [first_part, first_part + "乙" * 5500]

    def complete(**kwargs):
        provider.last_finish_reason = "stop"
        return chunks.pop(0)

    with patch.object(provider, "chat_completion", side_effect=complete):
        article = provider.generate_long_form("system", "user", target_chars=10_000)

    assert article.count(first_part) == 1
    assert len("".join(article.split())) == 9_500


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


def test_profile_write_passes_uploaded_documents_as_grounded_sources():
    provider = MagicMock()
    provider.chat_completion.return_value = route_payload(
        "write",
        topic="非厄米系统",
        key_points="解释实验结论",
        use_active_profile=True,
        use_documents_as_sources=True,
    )
    coordinator = MagicMock()
    coordinator.generate_article.return_value = (
        "最终文章",
        EvaluationReport(overall_score=88.0),
        AgentState(),
    )
    agent = ConversationAgent(AppConfig(), coordinator, model_provider=provider)
    documents = [{"title": "论文A", "content": "SOURCE_SENTINEL：边界条件决定谱结构。"}]

    agent.respond(
        "依据上传论文，按当前文风写一篇文章",
        documents,
        profile=make_profile(),
    )

    key_points = coordinator.generate_article.call_args.kwargs["key_points"]
    assert "SOURCE_SENTINEL" in key_points
    assert "不可信数据" in key_points


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
