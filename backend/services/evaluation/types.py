"""评测系统类型定义"""

from typing import Any
from pydantic import BaseModel, ConfigDict, Field, field_validator


class LLMUsage(BaseModel):
    """LLM 调用统计

    从 LLMCallStats.to_dict() 输出格式定义
    """
    model_config = ConfigDict(extra="allow")  # 允许未来扩展字段

    llm_calls: int
    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None
    usage_reported_calls: int | None = None
    observed_input_tokens: int | None = None
    observed_output_tokens: int | None = None
    observed_total_tokens: int | None = None
    models: list[str] = Field(default_factory=list)
    task_calls: dict[str, int] = Field(default_factory=dict)

    @field_validator("llm_calls")
    @classmethod
    def validate_calls(cls, v: int) -> int:
        if v < 0:
            raise ValueError("llm_calls must be non-negative")
        return v


class EvalError(BaseModel):
    """评测错误信息"""
    code: str = Field(min_length=1)
    stage: str = Field(min_length=1)
    error_type: str | None = None

    @field_validator("code", "stage")
    @classmethod
    def reject_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("字段不能为空白")
        return v.strip()
