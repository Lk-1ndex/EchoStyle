import pytest
from src.agents.state import AgentState, AgentStatus
from src.core.exceptions import InvalidStateTransitionError, CheckpointNotFoundError


def test_fsm_illegal_transition_blocked():
    """验证有限状态机非法跳变被严格拦截"""
    state = AgentState(topic="测试FSM")
    assert state.current_status == AgentStatus.IDLE

    # 合法跳变 IDLE -> PARSING
    state.transition_to(AgentStatus.PARSING, "正常解析")
    assert state.current_status == AgentStatus.PARSING

    # 非法跳变：PARSING 不能直接跳到 COMPLETED (必须经由 DRAFTING -> CRITIQUING 等)
    with pytest.raises(InvalidStateTransitionError) as exc_info:
        state.transition_to(AgentStatus.COMPLETED, "试图直接完成")

    assert "非法状态迁移拒绝" in str(exc_info.value)
    assert exc_info.value.current_status == "PARSING"
    assert exc_info.value.target_status == "COMPLETED"


def test_state_checkpoint_and_rollback():
    """验证状态快照与回滚机制"""
    state = AgentState(topic="测试回滚")
    state.transition_to(AgentStatus.DRAFTING, "初稿撰写")
    state.record_draft("WriterAgent", "初版优秀稿件，文风自然", score=88.0)

    # 1. 保存快照
    state.create_checkpoint("good_version")
    assert "good_version" in state.checkpoints

    # 2. 模拟后续反思重写失误，稿件恶化
    state.transition_to(AgentStatus.CRITIQUING, "质检中")
    state.transition_to(AgentStatus.REFLECTING, "反思修改")
    state.record_draft("WriterAgent", "改烂了的第二版草稿，充斥八股", score=60.0, cliches=["不可否认"])

    assert state.current_draft == "改烂了的第二版草稿，充斥八股"
    assert len(state.draft_chain) == 2

    # 3. 触发真实回滚
    success = state.rollback_to("good_version")
    assert success is True

    # 4. 验证状态与草稿完全自愈恢复
    assert state.current_draft == "初版优秀稿件，文风自然"
    assert state.current_status == AgentStatus.DRAFTING
    assert len(state.draft_chain) == 1

    # 5. 回滚不存在的检查点应抛异常
    with pytest.raises(CheckpointNotFoundError):
        state.rollback_to("non_existent_checkpoint")


def test_rollback_preserves_retry_count_preventing_infinite_loop():
    """验证 P0 缺陷修复：rollback_to 不回滚 workflow control state (retry_count)，杜绝无限循环"""
    state = AgentState(topic="测试重试计数不回滚", max_retries=2)
    state.transition_to(AgentStatus.DRAFTING, "初稿前保存检查点")
    
    # 在 retry_count = 0 时创建 pre_draft 检查点
    assert state.retry_count == 0
    state.create_checkpoint("pre_draft")

    # 模拟 Critic 判定 REJECT，Coordinator 递增重试计数并执行 rollback_to("pre_draft")
    state.retry_count += 1
    assert state.retry_count == 1
    state.rollback_to("pre_draft")

    # 核心断言：retry_count 绝对不能被重置回 0！
    assert state.retry_count == 1

    # 模拟第 2 轮再次 REJECT -> rollback
    state.retry_count += 1
    assert state.retry_count == 2
    state.rollback_to("pre_draft")
    assert state.retry_count == 2

    # 验证达到 max_retries 时循环能够正常跳出，不会因回滚陷入无限死循环
    assert state.retry_count >= state.max_retries

    # 验证显式要求恢复重试计数时的行为 (restore_retry_count=True)
    state.rollback_to("pre_draft", restore_retry_count=True)
    assert state.retry_count == 0


def test_draft_checkpoint_creation_and_subscript():
    """验证 DraftCheckpoint 实体创建、索引兼容性与状态回滚"""
    from src.agents.state import DraftCheckpoint
    state = AgentState(topic="测试快照模型")
    state.transition_to(AgentStatus.DRAFTING, "初稿保存")
    state.record_draft("WriterAgent", "测试初稿内容", score=82.0)

    ckpt = state.create_draft_checkpoint("v1")
    assert isinstance(ckpt, DraftCheckpoint)
    assert state.checkpoints["v1"] == ckpt

    # 验证字段属性访问与字典下标访问均兼容
    assert ckpt.status == AgentStatus.DRAFTING
    assert ckpt["status"] == AgentStatus.DRAFTING
    assert ckpt.current_draft == "测试初稿内容"
    assert ckpt["current_draft"] == "测试初稿内容"

    # 修改状态后回滚
    state.current_draft = "篡改内容"
    state.rollback_to("v1")
    assert state.current_draft == "测试初稿内容"


def test_coordinator_initial_draft_fsm_and_rollback_recovery():
    """验证 Coordinator 在注入 initial_draft 时的 FSM 状态合法性，以及评分退步回滚时报告恢复正确"""
    from unittest.mock import MagicMock
    from src.core.config import AppConfig
    from src.agents.coordinator import CoordinatorAgent
    from src.agents.tools import CritiqueAction
    from src.core.models import (
        DeepStyleProfile, StyleProfile, TonePersona, CadenceSyntax,
        LexiconRhetoric, DiscourseArchitecture, AntiPatterns, EvaluationReport
    )

    profile = DeepStyleProfile(
        name="测试文风",
        qualitative=StyleProfile(
            name="测试",
            tone_persona=TonePersona(perspective="第一人称", emotional_tone="冷静"),
            cadence_syntax=CadenceSyntax(sentence_style="短句", paragraph_habit="简短"),
            lexicon_rhetoric=LexiconRhetoric(metaphor_style="通俗", vocabulary_richness="丰富"),
            discourse=DiscourseArchitecture(opening_hook="破空设问", body_progression="层层递进", ending_style="金句收尾"),
            anti_patterns=AntiPatterns()
        )
    )

    config = AppConfig()
    coordinator = CoordinatorAgent(config)

    # 1. 验证 initial_draft 正常 ACCEPT 链路 (防止 IDLE -> CRITIQUING 非法跃迁，且顺畅流转至 COMPLETED)
    state1 = AgentState(topic="单变量测试")
    initial_text = "这是 C2 生成的高质量初稿"
    report_accept = EvaluationReport(overall_score=85.0, detected_cliches=[])

    coordinator.critic_agent.judge.evaluate = MagicMock(return_value=report_accept)
    draft1, rep1, final_st1 = coordinator.run(state=state1, profile=profile, initial_draft=initial_text)

    assert draft1 == initial_text
    assert rep1.overall_score == 85.0
    assert final_st1.current_status == AgentStatus.COMPLETED

    # 2. 验证评分退步 (score < prev_score - 15) 回滚保护：必须恢复前一优选版本的报告，而非退步后的坏报告
    state2 = AgentState(topic="回滚报告测试")
    report_v1 = EvaluationReport(overall_score=75.0, detected_cliches=[], feedback="需进一步润色")
    report_degraded = EvaluationReport(overall_score=55.0, detected_cliches=["老生常谈"], feedback="严重倒退")

    coordinator.critic_agent.judge.evaluate = MagicMock(side_effect=[report_v1, report_degraded])
    coordinator.writer_agent.model_provider.chat_completion = MagicMock(return_value="改坏了的文本")

    draft2, rep2, final_st2 = coordinator.run(state=state2, profile=profile, initial_draft="优选初始草稿")
    assert draft2 == "优选初始草稿"
    assert rep2.overall_score == 75.0  # 核心断言：必须恢复为 best_version 的 75.0，绝不能是 55.0
    assert final_st2.current_status == AgentStatus.COMPLETED_WITH_WARNING


