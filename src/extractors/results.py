"""Structured per-source extraction results."""

from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ExtractionStatus(str, Enum):
    SUCCEEDED = "succeeded"
    FAILED = "failed"


class ExtractionFailure(BaseModel):
    error_type: str
    message: str


class SourceExtractionResult(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    source_index: int = Field(ge=0)
    source: str
    status: ExtractionStatus
    article: dict[str, Any] | None = None
    error: ExtractionFailure | None = None
    logs: list[str] = Field(default_factory=list)
    exception: Exception | None = Field(default=None, exclude=True, repr=False)

    @model_validator(mode="after")
    def validate_payload(self) -> "SourceExtractionResult":
        if self.status == ExtractionStatus.SUCCEEDED and self.article is None:
            raise ValueError("成功的提取结果必须包含 article")
        if self.status == ExtractionStatus.FAILED and self.error is None:
            raise ValueError("失败的提取结果必须包含 error")
        return self


class ExtractionBatchResult(BaseModel):
    outcomes: list[SourceExtractionResult] = Field(default_factory=list)

    @property
    def successes(self) -> list[SourceExtractionResult]:
        return [item for item in self.outcomes if item.status == ExtractionStatus.SUCCEEDED]

    @property
    def failures(self) -> list[SourceExtractionResult]:
        return [item for item in self.outcomes if item.status == ExtractionStatus.FAILED]

    @property
    def articles(self) -> list[dict[str, Any]]:
        return [item.article for item in self.successes if item.article is not None]
