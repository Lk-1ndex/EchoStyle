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


class EvaluatorConfig(BaseModel):
    api_key: str = Field(default="", description="Holdout 独立评测裁判 API Key (留空则复用 LLM)")
    base_url: str = Field(default="", description="Holdout 独立评测裁判 基础路径 (留空则复用 LLM)")
    model: str = Field(default="", description="Holdout 独立评测裁判 模型名称 (留空则复用 LLM)")
    temperature: float = Field(default=0.0, description="评测温度，默认 0.0 尽可能压低采样随机性")
    max_tokens: int = Field(default=4096)

    def get_effective_llm_config(self, fallback: LLMConfig) -> LLMConfig:
        return LLMConfig(
            api_key=self.api_key.strip() if self.api_key and self.api_key.strip() else fallback.api_key,
            base_url=self.base_url.strip() if self.base_url and self.base_url.strip() else fallback.base_url,
            model=self.model.strip() if self.model and self.model.strip() else fallback.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens or fallback.max_tokens,
        )


class AppConfig(BaseModel):
    llm: LLMConfig = Field(default_factory=LLMConfig)
    evaluator: EvaluatorConfig = Field(default_factory=EvaluatorConfig)
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    extractor: ExtractorConfig = Field(default_factory=ExtractorConfig)
    agent: AgentConfig = Field(default_factory=AgentConfig)

    def get_evaluator_config(self) -> LLMConfig:
        """获取用于独立盲评的有效 LLM 配置（支持独立模型或复用生成模型）"""
        return self.evaluator.get_effective_llm_config(self.llm)


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

    # 独立 Evaluator 环境变量
    if os.getenv("EVALUATOR_API_KEY"):
        if "evaluator" not in data:
            data["evaluator"] = {}
        data["evaluator"]["api_key"] = os.getenv("EVALUATOR_API_KEY")
    if os.getenv("EVALUATOR_BASE_URL"):
        if "evaluator" not in data:
            data["evaluator"] = {}
        data["evaluator"]["base_url"] = os.getenv("EVALUATOR_BASE_URL")
    if os.getenv("EVALUATOR_MODEL"):
        if "evaluator" not in data:
            data["evaluator"] = {}
        data["evaluator"]["model"] = os.getenv("EVALUATOR_MODEL")

    return AppConfig(**data)

