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
