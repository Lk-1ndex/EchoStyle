import copy
from typing import Any, Callable, Dict, List, Optional
from pydantic import BaseModel, Field
from src.core.models import DeepStyleProfile, EvaluationReport


class CritiqueAction(BaseModel):
    """Critic 输出的结构化决策动作"""
    decision: str = Field(..., description="ACCEPT (合格放行), REVISE (需要反思润色), REJECT (严重违背文风需推倒重来)")
    score: float = 0.0
    detected_cliches: List[str] = Field(default_factory=list)
    actionable_feedback: str = ""
    detailed_report: Optional[EvaluationReport] = None


class Tool:
    """Agent 可调用的工具定义"""
    def __init__(self, name: str, description: str, func: Callable):
        self.name = name
        self.description = description
        self.func = func

    def execute(self, *args, **kwargs) -> Any:
        return self.func(*args, **kwargs)


class ToolRegistry:
    """
    统一工具注册表 (Tool Registry)：
    提供工作流受控编排、按需调用的工具容器。
    """
    def __init__(self):
        self._tools: Dict[str, Tool] = {}

    def register(self, tool: Tool):
        self._tools[tool.name] = tool

    def get(self, name: str) -> Optional[Tool]:
        return self._tools.get(name)

    def list_tools(self) -> List[Dict[str, str]]:
        return [{"name": t.name, "description": t.description} for t in self._tools.values()]
