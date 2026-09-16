from src.agents.state import AgentState, AgentStatus
from src.analyzer.stylometrics import StylometricsAnalyzer
from src.evaluation.metrics import MetricEvaluator


def test_agent_state_transitions():
    state = AgentState(topic="测试主题")
    assert state.current_status == AgentStatus.IDLE

    # 1. IDLE -> DRAFTING
    state.transition_to(AgentStatus.DRAFTING, "开始撰写初稿")
    assert state.current_status == AgentStatus.DRAFTING
    assert len(state.execution_logs) == 1

    state.record_draft(agent_name="WriterAgent", draft="这是第一版草稿", score=75.0, cliches=["总而言之"])
    assert len(state.draft_chain) == 1
    assert state.draft_chain[0].version == 1
    assert state.draft_chain[0].score == 75.0
    assert "总而言之" in state.draft_chain[0].detected_cliches

    # 2. DRAFTING -> CRITIQUING
    state.transition_to(AgentStatus.CRITIQUING, "质量审校中")
    assert state.current_status == AgentStatus.CRITIQUING

    # 3. CRITIQUING -> REFLECTING
    state.transition_to(AgentStatus.REFLECTING, "触发反思重写")
    assert state.current_status == AgentStatus.REFLECTING

    state.record_draft(agent_name="WriterAgent", draft="这是第二版重构草稿", score=88.0, cliches=[])
    assert len(state.draft_chain) == 2
    assert state.draft_chain[1].version == 2
    assert state.draft_chain[1].score == 88.0

    # 4. REFLECTING -> CRITIQUING -> COMPLETED
    state.transition_to(AgentStatus.CRITIQUING, "终审质检")
    state.transition_to(AgentStatus.COMPLETED, "任务顺利完成")
    assert state.current_status == AgentStatus.COMPLETED


def test_stylometrics_ttr():
    text = "写作者在思考技术，写作者也在思考人生的意义。写作是一场孤独的旅行。"
    metrics = StylometricsAnalyzer.analyze(text)
    assert metrics.ttr > 0.0
    assert metrics.unique_words_count > 0
    assert metrics.total_chars > 0


def test_metrics_evaluation_formula():
    text = "总而言之，这一现象不可否认是时代发展的必然趋势。"
    anti_ai_score, penalty, cliches = MetricEvaluator.evaluate_anti_ai(text)
    assert len(cliches) == 2
    assert penalty == 20.0
    assert anti_ai_score == 80.0
