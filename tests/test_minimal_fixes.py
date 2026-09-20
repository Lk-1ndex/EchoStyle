import json
from pathlib import Path
from unittest.mock import patch
from uuid import UUID

import pytest

from src.agents.analyst_agent import AnalystAgent
from src.agents.conversation_agent import ConversationAction, ConversationAgent, ConversationResult
from src.agents.coordinator import CoordinatorAgent
from src.agents.critic_agent import CriticAgent
from src.agents.state import AgentState, AgentStatus
from src.core.config import AppConfig, LLMConfig
from src.core.exceptions import EvaluationUnavailableError
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
from src.evaluation.judge import LLMJudge
from src.memory.memory_manager import MemoryManager
from src.memory.profile_store import ProfileStore
from src.memory.vector_store import VectorStore


def make_profile(profile_id="profile-a"):
    return DeepStyleProfile(
        name="Test profile",
        profile_id=profile_id,
        qualitative=StyleProfile(
            tone_persona=TonePersona(perspective="first person", emotional_tone="direct"),
            cadence_syntax=CadenceSyntax(sentence_style="short", paragraph_habit="compact"),
            lexicon_rhetoric=LexiconRhetoric(metaphor_style="concrete", vocabulary_richness="plain"),
            discourse=DiscourseArchitecture(opening_hook="question", body_progression="argument", ending_style="open"),
            anti_patterns=AntiPatterns(),
        ),
    )


def test_new_profile_id_round_trip_and_legacy_profile_rejected(tmp_path):
    store = VectorStore(storage_path=str(tmp_path / "memory.json"))
    manager = MemoryManager(vector_store=store)
    profile_store = ProfileStore(str(tmp_path / "profiles.json"))
    analyst = AnalystAgent(LLMConfig(), manager, profile_store=profile_store)
    analyst.distiller.distill = lambda *args, **kwargs: make_profile().qualitative

    profile = analyst.run(
        AgentState(),
        [{"title": "Sample", "content": "A complete paragraph with enough words for memory ingestion."}],
    )
    assert str(UUID(profile.profile_id)) == profile.profile_id
    assert DeepStyleProfile.model_validate_json(profile.model_dump_json()).profile_id == profile.profile_id
    assert manager.get_memory_stats(profile_id=profile.profile_id)["total_chunks"] == 1
    assert ProfileStore(str(tmp_path / "profiles.json")).get_active().profile_id == profile.profile_id

    old_data = profile.model_dump(exclude={"profile_id"})
    old_profile = DeepStyleProfile.model_validate(old_data)
    assert old_profile.profile_id is None
    coordinator = CoordinatorAgent(AppConfig(), memory_manager=manager)
    with pytest.raises(ValueError, match="profile_id"):
        coordinator.run(AgentState(topic="test"), profile=old_profile, initial_draft="Draft")


def test_profile_scoped_dense_hybrid_stats_and_clear(tmp_path):
    store = VectorStore(storage_path=str(tmp_path / "memory.json"))
    manager = MemoryManager(vector_store=store)
    with patch.object(store.model_provider, "get_embeddings", side_effect=lambda texts: [[1.0, 0.0] for _ in texts]):
        assert manager.ingest_article("Alpha", "Architecture design belongs to the alpha author alone.", profile_id="alpha") == 1
        assert manager.ingest_article("Beta", "Architecture design belongs to the beta author alone.", profile_id="beta") == 1
        assert manager.ingest_article("Unscoped", "Architecture design in an unscoped example.") == 1

        for profile_id, expected in [("alpha", "alpha"), ("beta", "beta")]:
            dense = manager.retrieve_dense("architecture design", top_k=3, profile_id=profile_id)
            hybrid = manager.retrieve_hybrid("architecture design", top_k=3, profile_id=profile_id)
            dynamic = manager.retrieve_dynamic_few_shots("architecture design", profile_id=profile_id)
            assert len(dense) == len(hybrid) == len(dynamic) == 1
            assert expected in dense[0] and expected in hybrid[0] and expected in dynamic[0]
            assert manager.get_memory_stats(profile_id=profile_id)["total_chunks"] == 1

        assert manager.get_memory_stats()["total_chunks"] == 1
        assert "unscoped" in manager.retrieve_hybrid("architecture design", top_k=3)[0]
        manager.clear_memory(profile_id="alpha")
        assert manager.get_memory_stats(profile_id="alpha")["total_chunks"] == 0
        assert manager.get_memory_stats(profile_id="beta")["total_chunks"] == 1
        assert manager.get_memory_stats()["total_chunks"] == 1
        manager.clear_memory()
        assert manager.get_memory_stats()["total_chunks"] == 0
        assert manager.get_memory_stats(profile_id="beta")["total_chunks"] == 1


def test_duplicate_ids_and_interleaved_instances_do_not_lose_data(tmp_path):
    storage_path = str(tmp_path / "memory.json")
    first = VectorStore(storage_path=storage_path)
    second = VectorStore(storage_path=storage_path)
    a = {"id": "a", "content": "First author content", "metadata": {"profile_id": "alpha"}}
    b = {"id": "a", "content": "Second author content", "metadata": {"profile_id": "beta"}}

    assert first.add_chunks([a]) == 1
    assert first.add_chunks([a, a]) == 0
    with pytest.raises(ValueError, match="ID"):
        first.add_chunks([{**a, "content": "Different content"}])
    assert second.add_chunks([b]) == 1
    assert len(first.get_all(profile_id="alpha")) == 1
    assert len(first.get_all(profile_id="beta")) == 1

    first.clear(profile_id="alpha")
    assert second.add_chunks([{**a, "content": "New alpha content"}]) == 1
    assert [c["content"] for c in first.get_all(profile_id="alpha")] == ["New alpha content"]
    assert len(first.get_all(profile_id="beta")) == 1


def test_unchanged_vector_store_is_not_reparsed_on_read(tmp_path):
    storage_path = str(tmp_path / "memory.json")
    first = VectorStore(storage_path=storage_path)
    second = VectorStore(storage_path=storage_path)
    chunk = {"id": "a", "content": "Original", "metadata": {"profile_id": "alpha"}}
    first.add_chunks([chunk])

    with patch.object(first, "_load", wraps=first._load) as load:
        assert len(first.get_all(profile_id="alpha")) == 1
        assert len(first.get_all(profile_id="alpha")) == 1
        assert load.call_count == 0

        second.add_chunks([{"id": "b", "content": "New", "metadata": {"profile_id": "alpha"}}])
        assert len(first.get_all(profile_id="alpha")) == 2
        assert load.call_count == 1
        assert len(first.get_all(profile_id="alpha")) == 2
        assert load.call_count == 1


def test_corrupt_store_raises_without_overwriting_it(tmp_path):
    storage_path = tmp_path / "memory.json"
    storage_path.write_text("{broken json", encoding="utf-8")
    with pytest.raises(json.JSONDecodeError):
        VectorStore(storage_path=str(storage_path))
    assert storage_path.read_text(encoding="utf-8") == "{broken json"

    storage_path.write_text("[]", encoding="utf-8")
    store = VectorStore(storage_path=str(storage_path))
    storage_path.write_text("{broken again", encoding="utf-8")
    with pytest.raises(json.JSONDecodeError):
        store.get_all()
    with pytest.raises(json.JSONDecodeError):
        store.add_chunks([{"id": "new", "content": "Never persisted"}])
    assert storage_path.read_text(encoding="utf-8") == "{broken again"


def test_old_memory_file_is_not_loaded_or_changed(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    old_file = tmp_path / "profiles" / "style_memory.json"
    old_file.parent.mkdir()
    old_file.write_text('[{"id":"legacy","content":"private old text"}]', encoding="utf-8")
    store = VectorStore()
    assert store.storage_path.name == "style_memory_v2.json"
    assert store.get_all() == []
    assert "private old text" in old_file.read_text(encoding="utf-8")


@pytest.mark.parametrize(
    "response",
    [
        "{}",
        '{"style_fidelity":"80","logic_depth":85,"human_preference":80}',
        '{"style_fidelity":true,"logic_depth":85,"human_preference":80}',
        '{"style_fidelity":101,"logic_depth":85,"human_preference":80}',
        '{"style_fidelity":NaN,"logic_depth":85,"human_preference":80}',
        '{"style_fidelity":80,"logic_depth":85,"human_preference":80,"radar":{"tone":-1}}',
    ],
)
def test_judge_rejects_missing_and_invalid_scores(response):
    judge = LLMJudge(LLMConfig())
    judge._call_judge = lambda *args: response
    with pytest.raises(EvaluationUnavailableError):
        judge.evaluate("A clean draft with a substantive argument.", make_profile())


def test_judge_failure_marks_critic_and_coordinator_failed():
    profile = make_profile()
    critic = CriticAgent(LLMConfig())
    critic.judge._call_judge = lambda *args: (_ for _ in ()).throw(RuntimeError("offline"))
    state = AgentState(topic="test")
    state.transition_to(AgentStatus.DRAFTING, "Draft")
    state.record_draft("WriterAgent", "A draft")
    state.latest_report = EvaluationReport(overall_score=90.0)
    with pytest.raises(EvaluationUnavailableError, match="offline"):
        critic.evaluate_action(state, "A draft", profile)
    assert state.current_status == AgentStatus.FAILED
    assert state.latest_report is None

    direct_state = AgentState()
    with pytest.raises(EvaluationUnavailableError, match="offline"):
        critic.evaluate("A draft", profile, state=direct_state)
    assert direct_state.current_status == AgentStatus.FAILED

    coordinator = CoordinatorAgent(AppConfig())
    coordinator.critic_agent.judge = critic.judge
    task_state = AgentState(topic="test")
    with pytest.raises(EvaluationUnavailableError):
        coordinator.run(task_state, profile=profile, initial_draft="A draft")
    assert task_state.current_status == AgentStatus.FAILED
    assert task_state.latest_report is None


def test_run_preserves_state_and_extract_surfaces_original_error(tmp_path):
    coordinator = CoordinatorAgent(AppConfig())
    coordinator.critic_agent.judge.evaluate = lambda draft, profile: EvaluationReport(overall_score=90.0)
    state = AgentState(topic="test", word_count=800, target_audience="special readers")
    coordinator.run(state, profile=make_profile(), initial_draft="A draft")
    assert (state.word_count, state.target_audience) == (800, "special readers")

    missing = str(tmp_path / "missing.txt")
    with pytest.raises(FileNotFoundError, match="missing.txt"):
        coordinator.extract_sources([missing])
    good = tmp_path / "good.txt"
    good.write_text("A useful sample article.", encoding="utf-8")
    assert len(coordinator.extract_sources([missing, str(good)])) == 1


def test_online_ablation_uses_private_store_and_cleans_on_failure(tmp_path, monkeypatch):
    from experiments.ablation_study import run_ablation_study

    monkeypatch.chdir(tmp_path)
    profile_dir = tmp_path / "profiles"
    profile_dir.mkdir()
    old_file = profile_dir / "style_memory.json"
    current_file = profile_dir / "style_memory_v2.json"
    old_file.write_text("legacy sentinel", encoding="utf-8")
    current_file.write_text("current sentinel", encoding="utf-8")
    config = AppConfig()
    config.llm.api_key = "sk-test-key"
    private_paths = []

    def stop_at_coordinator(config, memory_manager):
        private_paths.append(memory_manager.vector_store.storage_path)
        memory_manager.vector_store.add_chunks([{"id": "private", "content": "Only for this experiment"}])
        assert private_paths[0].exists()
        raise RuntimeError("stop after private write")

    with patch("experiments.ablation_study.load_config", return_value=config), \
         patch("experiments.ablation_study.ModelProvider.get_embeddings", side_effect=lambda texts: [[1.0] for _ in texts]), \
         patch("experiments.ablation_study.CoordinatorAgent", side_effect=stop_at_coordinator):
        with pytest.raises(RuntimeError, match="stop after private write"):
            run_ablation_study(topics_count=1, repeats=1)

    assert len(private_paths) == 1 and not private_paths[0].exists()
    assert old_file.read_text(encoding="utf-8") == "legacy sentinel"
    assert current_file.read_text(encoding="utf-8") == "current sentinel"


def test_web_chat_build_write_and_profile_restore_without_network(tmp_path, monkeypatch):
    from streamlit.testing.v1 import AppTest

    monkeypatch.chdir(tmp_path)
    config = AppConfig()
    config.llm.api_key = "test-key"
    with patch("src.core.config.load_config", return_value=config):
        app = AppTest.from_file(Path(__file__).resolve().parents[1] / "src/web/app.py", default_timeout=10).run()
    assert not app.exception
    assert len(app.chat_input) == 1
    assert not any(item.label == "当前文风画像" for item in app.selectbox)
    assert app.session_state["messages"] == []
    assert any("What should we write?" in item.value for item in app.markdown)

    effort_selector = next(item for item in app.select_slider if item.label == "模型思考强度")
    effort_selector.set_value("low").run()
    assert not app.exception
    assert app.session_state["config"].llm.thinking_effort == "low"
    assert app.session_state["conversation_agent"].model_provider.llm_config.thinking_effort == "low"

    sample = {"title": "Sample", "content": "A source paragraph with enough text.", "engine_used": "mock", "char_count": 36}
    app.session_state["documents"] = [sample]

    app.slider[0].set_value(95.0).run()
    assert app.session_state["coordinator"].critic_agent.quality_threshold == 95.0

    profile = make_profile()
    build_result = ConversationResult(
        content="Profile ready",
        action=ConversationAction.BUILD_STYLE,
        profile=profile,
    )
    with patch.object(ConversationAgent, "respond", return_value=build_result) as respond:
        app.chat_input[0].set_value("分析这些文件的文风").run()
    assert not app.exception
    assert respond.call_args.kwargs["documents"] == [sample]
    assert app.session_state["deep_profile"].profile_id == profile.profile_id
    assert app.session_state["profile_store"].get(profile.profile_id).name == profile.name

    with patch("src.core.config.load_config", return_value=config):
        restored_app = AppTest.from_file(
            Path(__file__).resolve().parents[1] / "src/web/app.py",
            default_timeout=10,
        ).run()
    assert not restored_app.exception
    assert restored_app.session_state["deep_profile"].profile_id == profile.profile_id

    second_profile = make_profile("profile-b")
    restored_app.session_state["profile_store"].save(second_profile, make_active=False)
    restored_app.run()
    assert sum(item.label == "当前文风画像" for item in restored_app.selectbox) == 1
    profile_selector = restored_app.selectbox[0]
    profile_selector.select_index(1).run()
    assert not restored_app.exception
    assert restored_app.session_state["deep_profile"].profile_id == second_profile.profile_id
    assert restored_app.session_state["profile_store"].get_active().profile_id == second_profile.profile_id

    final_state = AgentState(topic="A new article topic")
    write_result = ConversationResult(
        content="Final article",
        action=ConversationAction.WRITE,
        profile=profile,
        article="Final article",
        report=EvaluationReport(overall_score=90.0),
        logs=final_state.execution_logs,
    )
    with patch.object(ConversationAgent, "respond", return_value=write_result) as write:
        app.chat_input[0].set_value("写一篇新文章").run()
    assert not app.exception
    assert write.call_args is not None
    assert app.session_state["last_article"] == "Final article"
    assert app.session_state["messages"][-1]["report"]["overall_score"] == 90.0

    with patch.object(ConversationAgent, "respond", side_effect=EvaluationUnavailableError("offline")):
        app.chat_input[0].set_value("再写一篇").run()
    assert not app.exception
    assert app.session_state["last_article"] == "Final article"
    assert "offline" in app.session_state["messages"][-1]["content"]

    next(button for button in app.button if button.label == "新建对话").click().run()
    assert not app.exception
    assert app.session_state["messages"] == []
    assert any("What should we write?" in item.value for item in app.markdown)


def test_web_clear_documents_does_not_clear_style_memory(tmp_path, monkeypatch):
    from streamlit.testing.v1 import AppTest

    monkeypatch.chdir(tmp_path)
    config = AppConfig()
    with patch("src.core.config.load_config", return_value=config):
        app = AppTest.from_file(Path(__file__).resolve().parents[1] / "src/web/app.py", default_timeout=10).run()
    store = app.session_state["memory_mgr"].vector_store
    with patch.object(store.model_provider, "get_embeddings", return_value=None):
        store.add_chunks([
            {"id": "a", "content": "Current profile data", "metadata": {"profile_id": "alpha"}},
            {"id": "b", "content": "Another profile data", "metadata": {"profile_id": "beta"}},
        ])

    app.session_state["deep_profile"] = make_profile("alpha")
    app.session_state["documents"] = [{"title": "Temporary", "content": "Session document"}]
    app.session_state["document_hashes"] = {"temporary-hash"}
    app.run()
    next(b for b in app.button if "清空文件" in b.label).click().run()
    assert not app.exception
    assert app.session_state["documents"] == []
    assert app.session_state["document_hashes"] == set()
    assert len(store.get_all(profile_id="alpha")) == 1
    assert len(store.get_all(profile_id="beta")) == 1
