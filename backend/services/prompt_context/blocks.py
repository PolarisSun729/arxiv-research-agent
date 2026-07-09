from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from typing import Any, Dict


@dataclass
class PromptBlock:
    """进入 token-aware prompt 预算系统的最小可选择单元。"""

    block_id: str
    section: str
    block_type: str
    source: str
    text: str
    priority: int = 100
    score: float = 0.0
    original_rank: int = 0
    context_role: str = ""
    token_count: int = 0
    original_token_count: int = 0
    budget_group: str = ""
    protected: bool = False
    droppable: bool = True
    compactable: bool = True
    evidence_origin: str = "raw"
    source_id: str = ""
    original_source_id: str = ""
    can_support_numeric_claim: bool = False
    can_support_direct_quote: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)

    def with_updates(self, **changes: Any) -> "PromptBlock":
        return replace(self, **changes)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "PromptBlock":
        allowed = set(cls.__dataclass_fields__.keys())
        data = {key: value for key, value in dict(payload or {}).items() if key in allowed}
        return cls(**data)
