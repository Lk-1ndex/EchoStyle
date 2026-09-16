import time
import uuid
from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field
from src.core.models import EvaluationReport


class AgentStatus(str, Enum):
    IDLE = "IDLE"
    PARSING = "PARSING"           # 文档感知与提取中
    MODELING = "MODELING"         # 统计建模与记忆切片中
    DRAFTING = "DRAFTING"         # 记忆召回与草稿撰写中
    CRITIQUING = "CRITIQUING"     # 质量严审与 EchoEval 测评中
    REFLECTING = "REFLECTING"     # 反思自查与针对性重构中
    COMPLETED = "COMPLETED"       # 终审通过，任务圆满达成
    FAILED = "FAILED"             # 发生不可恢复异常或超重试上限


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
    工业级 Agent 状态机与上下文全生命周期容器：
    记录完整的任务目标、记忆快照、草稿版本链、重试计数、异常恢复记录与状态迁移轨迹。
    """
    task_id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    current_status: AgentStatus = AgentStatus.IDLE
    topic: str = ""
    key_points: str = ""
    word_count: int = 1500
    target_audience: str = "大众读者"

    # 记忆与上下文快照
    memory_snapshot: List[Dict[str, Any]] = Field(default_factory=list, description="本轮检索召回的 Few-shot 记忆快照")

    # 版本演进链与当前稿件
    draft_chain: List[DraftVersion] = Field(default_factory=list)
    current_draft: str = ""
    latest_report: Optional[EvaluationReport] = None

    # 反思重试控制
    retry_count: int = 0
    max_retries: int = 2

    # 执行审计日志与错误恢复
    execution_logs: List[str] = Field(default_factory=list)
    error_history: List[str] = Field(default_factory=list)

    def transition_to(self, new_status: AgentStatus, log_msg: str):
        """显式状态迁移驱动器"""
        prev_status = self.current_status
        self.current_status = new_status
        timestamp_str = time.strftime("%H:%M:%S")
        entry = f"[{timestamp_str}] [STATE: {prev_status.value} -> {new_status.value}] {log_msg}"
        self.execution_logs.append(entry)
        print(entry)

    def record_draft(self, agent_name: str, draft: str, score: float = 0.0, cliches: List[str] = None, feedback: str = ""):
        """记录新版本草稿节点"""
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
        """异常记录与恢复标记"""
        entry = f"[ERROR RECOVERY] {err_msg}"
        self.error_history.append(entry)
        self.execution_logs.append(entry)
        print(entry)
