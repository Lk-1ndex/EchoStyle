import os
import yaml
from pathlib import Path
from pydantic import BaseModel, Field


class LLMConfig(BaseModel):
    api_key: str = Field(default="", description="大模型 API Key")
    base_url: str = Field(default="https://api.deepseek.com/v1", description="OpenAI 兼容 API 基础路径")
    model: str = Field(default="deepseek-chat", description="模型名称")
    temperature: float = Field(default=0.7)
    max_tokens: int = Field(default=4096)


class EmbeddingConfig(BaseModel):
    api_key: str = Field(default="", description="向量 Embedding API Key (留空则复用 LLM)")
    base_url: str = Field(default="", description="Embedding 基础路径 (留空则复用 LLM)")
    model: str = Field(default="text-embedding-3-small", description="向量模型名称")


class ExtractorConfig(BaseModel):
    pdf_engine: str = Field(default="auto", description="auto (智能探测自适应), markitdown, 或 mineru")
    mineru_command: str = Field(default="magic-pdf")
    clean_noise: bool = Field(default=True)
    repair_linebreaks: bool = Field(default=True)


class AgentConfig(BaseModel):
    max_reflections: int = Field(default=2, description="Critic Agent 触发重写的最大自省反思轮次")
    quality_threshold: float = Field(default=80.0, description="成品合格质量分阈值 (低于则触发反思重写)")


class AppConfig(BaseModel):
    llm: LLMConfig = Field(default_factory=LLMConfig)
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    extractor: ExtractorConfig = Field(default_factory=ExtractorConfig)
    agent: AgentConfig = Field(default_factory=AgentConfig)


def load_config(config_path: str = "config.yaml") -> AppConfig:
    path = Path(config_path)
    if not path.exists():
        example_path = Path("config.example.yaml")
        if example_path.exists():
            path = example_path
        else:
            return AppConfig()

    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    # 环境变量优先覆盖
    if os.getenv("OPENAI_API_KEY"):
        if "llm" not in data:
            data["llm"] = {}
        data["llm"]["api_key"] = os.getenv("OPENAI_API_KEY")
    if os.getenv("OPENAI_BASE_URL"):
        if "llm" not in data:
            data["llm"] = {}
        data["llm"]["base_url"] = os.getenv("OPENAI_BASE_URL")

    return AppConfig(**data)
