import time
import uuid
import copy
from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field
from src.core.models import EvaluationReport
from src.core.exceptions import InvalidStateTransitionError, CheckpointNotFoundError


class AgentStatus(str, Enum):
    IDLE = "IDLE"
    PARSING = "PARSING"           # 样文格式与排版感知抽取
    MODELING = "MODELING"         # 统计建模与记忆切片
    DRAFTING = "DRAFTING"         # 记忆召回与初稿撰写
    CRITIQUING = "CRITIQUING"     # 质量严审与 EchoEval 评测
    REFLECTING = "REFLECTING"     # 反思自查与针对性重构
    COMPLETED = "COMPLETED"       # 终审通过，任务圆满达成
    COMPLETED_WITH_WARNING = "COMPLETED_WITH_WARNING"  # 重试已满但未通过ACCEPT阈值，降级输出最佳版本
    FAILED = "FAILED"             # 任务失败或超重试上限


# 严格有限状态机合法转移拓扑表 (FSM Transition Guard)
ALLOWED_TRANSITIONS: Dict[AgentStatus, List[AgentStatus]] = {
    AgentStatus.IDLE: [AgentStatus.PARSING, AgentStatus.MODELING, AgentStatus.DRAFTING, AgentStatus.FAILED],
    AgentStatus.PARSING: [AgentStatus.MODELING, AgentStatus.DRAFTING, AgentStatus.FAILED],
    AgentStatus.MODELING: [AgentStatus.DRAFTING, AgentStatus.FAILED],
    AgentStatus.DRAFTING: [AgentStatus.CRITIQUING, AgentStatus.FAILED],
    AgentStatus.CRITIQUING: [AgentStatus.REFLECTING, AgentStatus.COMPLETED, AgentStatus.COMPLETED_WITH_WARNING, AgentStatus.FAILED],
    AgentStatus.REFLECTING: [AgentStatus.DRAFTING, AgentStatus.CRITIQUING, AgentStatus.FAILED],
    AgentStatus.COMPLETED: [AgentStatus.IDLE],  # 允许重置任务
    AgentStatus.COMPLETED_WITH_WARNING: [AgentStatus.IDLE],  # 允许重置任务
    AgentStatus.FAILED: [AgentStatus.IDLE],     # 允许失败后重置
}


class DraftVersion(BaseModel):
    """草稿版本溯源节点"""
    version: int
    author_agent: str
    content: str
    timestamp: float = Field(default_factory=time.time)
    score: float = 0.0
    detected_cliches: List[str] = Field(default_factory=list)
    critic_feedback: str = ""


class AgentState(BaseModel):
    """
    真正具备有限状态机 (FSM) 约束与快照回滚 (Rollback) 能力的 Agent 状态容器
    """
    task_id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    current_status: AgentStatus = AgentStatus.IDLE
    topic: str = ""
    key_points: str = ""
    word_count: int = 1500
    target_audience: str = "大众读者"

    # 记忆与上下文快照 (None 表示未初始化，空列表 [] 表示显式禁用检索)
    memory_snapshot: Optional[List[Dict[str, Any]]] = Field(default=None)

    # 版本演进链与当前稿件
    draft_chain: List[DraftVersion] = Field(default_factory=list)
    current_draft: str = ""
    latest_report: Optional[EvaluationReport] = None

    # 反思重试控制
    retry_count: int = 0
    max_retries: int = 2

    # 执行审计日志与错误记录
    execution_logs: List[str] = Field(default_factory=list)
    error_history: List[str] = Field(default_factory=list)

    # 状态快照仓库 (Checkpoints)
    checkpoints: Dict[str, Dict[str, Any]] = Field(default_factory=dict)

    def transition_to(self, new_status: AgentStatus, log_msg: str):
        """
        显式状态转移驱动（含 FSM 严格校验守卫）
        """
        allowed = ALLOWED_TRANSITIONS.get(self.current_status, [])
        if new_status not in allowed and new_status != self.current_status:
            err = InvalidStateTransitionError(
                current_status=self.current_status.value,
                target_status=new_status.value,
                allowed=[s.value for s in allowed]
            )
            self.record_error(str(err))
            raise err

        prev_status = self.current_status
        self.current_status = new_status
        timestamp_str = time.strftime("%H:%M:%S")
        entry = f"[{timestamp_str}] [STATE: {prev_status.value} -> {new_status.value}] {log_msg}"
        self.execution_logs.append(entry)
        print(entry)

    def create_checkpoint(self, checkpoint_name: str):
        """创建当前完整状态快照"""
        self.checkpoints[checkpoint_name] = {
            "status": self.current_status,
            "current_draft": self.current_draft,
            "draft_chain": copy.deepcopy(self.draft_chain),
            "latest_report": copy.deepcopy(self.latest_report),
            "retry_count": self.retry_count,
            "timestamp": time.time(),
        }
        self.execution_logs.append(f"[CHECKPOINT] 已保存检查点: '{checkpoint_name}' (稿件字数: {len(self.current_draft)})")

    def rollback_to(self, checkpoint_name: str) -> bool:
        """从检查点回滚恢复状态"""
        if checkpoint_name not in self.checkpoints:
            raise CheckpointNotFoundError(f"未找到检查点快照: '{checkpoint_name}'")

        ckpt = self.checkpoints[checkpoint_name]
        self.current_status = ckpt["status"]
        self.current_draft = ckpt["current_draft"]
        self.draft_chain = copy.deepcopy(ckpt["draft_chain"])
        self.latest_report = copy.deepcopy(ckpt["latest_report"])
        self.retry_count = ckpt["retry_count"]

        entry = f"[ROLLBACK SUCCESS] 状态成功回滚至检查点: '{checkpoint_name}'，已恢复稿件版本与评估分"
        self.execution_logs.append(entry)
        print(entry)
        return True

    def record_draft(self, agent_name: str, draft: str, score: float = 0.0, cliches: List[str] = None, feedback: str = ""):
        self.current_draft = draft
        new_version = len(self.draft_chain) + 1
        dv = DraftVersion(
            version=new_version,
            author_agent=agent_name,
            content=draft,
            score=score,
            detected_cliches=cliches or [],
            critic_feedback=feedback,
        )
        self.draft_chain.append(dv)

    def record_error(self, err_msg: str):
        entry = f"[ERROR RECOVERY] {err_msg}"
        self.error_history.append(entry)
        self.execution_logs.append(entry)
        print(entry)
