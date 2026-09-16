from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field
from .state import AgentState, AgentStatus


class AgentMessage(BaseModel):
    sender: str
    recipient: str
    action: str
    content: Any
    metadata: Dict[str, Any] = Field(default_factory=dict)


# 保持向后兼容的别名
AgentContext = AgentState


class BaseAgent(ABC):
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description

    @abstractmethod
    def run(self, state: AgentState, **kwargs) -> Any:
        pass
