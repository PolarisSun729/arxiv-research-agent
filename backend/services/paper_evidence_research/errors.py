from __future__ import annotations

from typing import Any


class PaperEvidenceResearchError(RuntimeError):
    """证据研究的结构化系统异常；正常拒答不能使用该异常表达。"""

    def __init__(
        self,
        *,
        code: str,
        stage: str,
        message: str,
        detail: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.stage = stage
        self.detail = dict(detail or {})

    def to_payload(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "stage": self.stage,
            "message": str(self),
            "detail": dict(self.detail),
        }
