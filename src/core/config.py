import os
import yaml
from pathlib import Path
from typing import Literal, Optional
from pydantic import BaseModel, Field, field_validator, model_validator


class LLMConfig(BaseModel):
    api_key: str = Field(default="", description="大模型 API Key")
    base_url: str = Field(default="https://api.deepseek.com", description="OpenAI 兼容 API 基础路径")
    model: str = Field(default="deepseek-flash", description="模型名称")
    temperature: float = Field(default=0.7)
    max_tokens: int = Field(default=4096)
    context_window: Optional[int] = Field(
        default=None,
        ge=8192,
        description="模型上下文窗口；留空时根据模型名自动识别",
    )
    thinking_effort: Literal["auto", "off", "low", "high", "max"] = Field(
        default="auto", description="DeepSeek 思考强度；auto 使用模型默认设置"
    )


class EmbeddingConfig(BaseModel):
    mode: Literal["sparse", "dense"] = Field(
        default="sparse",
        description="检索向量模式；sparse 不发起 Embedding 请求，dense 使用独立配置",
    )
    api_key: str = Field(default="", description="独立向量 Embedding API Key")
    base_url: str = Field(default="", description="独立 Embedding API 基础路径")
    model: str = Field(default="", description="向量模型名称")
    batch_size: int = Field(default=32, ge=1, le=256, description="单次 Embedding 请求的最大文本数")
    timeout: float = Field(default=60.0, gt=0, description="单批 Embedding 请求超时秒数")
    max_retries: int = Field(default=3, ge=0, le=10, description="Embedding 临时故障最大重试次数")
    retry_base_delay: float = Field(default=1.0, ge=0, description="Embedding 指数退避基础秒数")

    @field_validator("api_key", "base_url", "model", mode="before")
    @classmethod
    def strip_strings(cls, value: object) -> object:
        return value.strip() if isinstance(value, str) else value

    @model_validator(mode="after")
    def validate_dense_configuration(self) -> "EmbeddingConfig":
        if self.mode == "dense":
            missing = [
                name
                for name in ("api_key", "base_url", "model")
                if not getattr(self, name)
            ]
            if missing:
                raise ValueError(f"embedding.mode=dense 时必须配置: {', '.join(missing)}")
        return self


class ExtractorConfig(BaseModel):
    pdf_engine: str = Field(
        default="mineru",
        description="PDF 统一使用 MinerU；markitdown 仅作为显式诊断模式或失败回退",
    )
    mineru_command: str = Field(default="mineru-kit")
    mineru_tier: str = Field(default="basic", description="MinerU 本地模型档位: basic 或 standard")
    mineru_timeout: int = Field(default=900, ge=1, description="单个 PDF 的 MinerU 最大运行秒数")
    mineru_intra_op_num_threads: int = Field(default=2, ge=1, description="MinerU ONNX 算子线程数")
    mineru_inter_op_num_threads: int = Field(default=1, ge=1, description="MinerU ONNX 算子间线程数")
    mineru_pdf_render_threads: int = Field(default=1, ge=1, description="PDF 渲染线程数")
    mineru_malloc_trim: bool = Field(default=True, description="解析结束时启用内存回收提示")
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
            context_window=fallback.context_window,
            thinking_effort=fallback.thinking_effort,
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

    # 向量 Embedding 环境变量
    if os.getenv("EMBEDDING_MODE"):
        if "embedding" not in data:
            data["embedding"] = {}
        data["embedding"]["mode"] = os.getenv("EMBEDDING_MODE")
    if os.getenv("EMBEDDING_API_KEY"):
        if "embedding" not in data:
            data["embedding"] = {}
        data["embedding"]["api_key"] = os.getenv("EMBEDDING_API_KEY")
    if os.getenv("EMBEDDING_BASE_URL"):
        if "embedding" not in data:
            data["embedding"] = {}
        data["embedding"]["base_url"] = os.getenv("EMBEDDING_BASE_URL")
    if os.getenv("EMBEDDING_MODEL"):
        if "embedding" not in data:
            data["embedding"] = {}
        data["embedding"]["model"] = os.getenv("EMBEDDING_MODEL")

    return AppConfig(**data)
