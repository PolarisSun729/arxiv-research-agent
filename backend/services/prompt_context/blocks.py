from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from typing import Any, Dict


@dataclass
class PromptBlock:
    """进入 token-aware prompt 预算系统的最小可选择单元。

    一个 block 对应 prompt 中一段可以被整体保留、压缩或丢弃的文本，例如一条 RAG 证据、
    一轮历史对话、会话摘要或用户画像。预算规划器（PromptBudgetPlanner）以 block 为粒度
    做取舍，渲染器（PromptRenderer）再按 section 把选中的 block 拼成最终 prompt。

    block 在链路中是“不可变”使用的：压缩、补 token 数等操作都通过 ``with_updates``
    生成新对象，避免多个阶段共享同一对象时互相污染。
    """

    # ---- 身份与归属 ----
    block_id: str  # 全局唯一 ID，预算规划器用它判重；重复 ID 的后一个 block 会被视为已选中而跳过。
    section: str  # 所属 prompt section，必须是 budget_planner.SECTION_ORDER 中的值才会被渲染。
    block_type: str  # 细分类型（如 table_final_evidence / figure_evidence / recent_turn），决定规则压缩策略。
    source: str  # 产生该 block 的上游模块，仅用于诊断。
    text: str  # 实际进入 prompt 的文本；压缩后会被替换为压缩文本。

    # ---- 排序信号 ----
    priority: int = 100  # 越小越优先；预算不足时先保留小 priority 的 block。
    score: float = 0.0  # 同 priority 下的相关性分数，越大越优先。
    original_rank: int = 0  # 上游原始顺序；既用于同分时的稳定排序，也决定渲染时 section 内的先后。
    context_role: str = ""  # RAG 证据角色（anchor_evidence / sibling_context / section_context ...），决定压缩力度。

    # ---- token 统计 ----
    token_count: int = 0  # 当前 text 的 token 数；0 表示尚未计数，由压缩器/规划器补齐。
    original_token_count: int = 0  # 压缩前的 token 数，用于诊断压缩收益。
    budget_group: str = ""  # 上游给出的预算分组标签（如 rag_core / rag_expansion），目前只透传到输出供诊断。

    # ---- 预算约束 ----
    protected: bool = False  # True 时无条件进入 prompt，不参与软预算竞争。
    droppable: bool = True  # False 时若放不进预算，会以 budget_error 形式暴露在 dropped_blocks 中。
    compactable: bool = True  # False 时规则压缩与 LLM 压缩都会跳过该 block。

    # ---- 证据可信度 ----
    evidence_origin: str = "raw"  # raw / rule_compacted / llm_compacted，标记文本是否被改写过。
    source_id: str = ""  # 回答时引用的证据 ID。
    original_source_id: str = ""  # 证据被拆分/合并前的原始 ID，供引用回溯。
    can_support_numeric_claim: bool = False  # 是否允许模型基于该 block 给出数值结论（LLM 压缩后一律置 False）。
    can_support_direct_quote: bool = True  # 是否允许直接引用原文（LLM 压缩后文本已被改写，置 False）。

    # 结构化附加信息：表格证据、会话摘要原始 payload、压缩诊断等都放在这里。
    metadata: Dict[str, Any] = field(default_factory=dict)

    def with_updates(self, **changes: Any) -> "PromptBlock":
        """返回修改了指定字段的新 block，原 block 保持不变。"""
        return replace(self, **changes)

    def to_dict(self) -> Dict[str, Any]:
        """序列化为普通 dict，便于写入 debug 输出或跨模块传递。"""
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "PromptBlock":
        """从 dict 反序列化，自动忽略未知字段，兼容上游多带字段的情况。"""
        allowed = set(cls.__dataclass_fields__.keys())
        data = {key: value for key, value in dict(payload or {}).items() if key in allowed}
        return cls(**data)
