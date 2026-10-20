"""计划里流动的语义类型（设计见 notes/agent-redesign.md D4-1）。

工具契约的 input_schema / output_schema 只有键名和 dict/list，代码判断不了「这一步要的东西，
前面有没有人产出」：同一种东西在不同工具里叫不同的名字（paper_ref / paper_reference），
「已解析的论文」和「已建索引的论文」也都是 dict。这里给每种东西一个统一的名字和结构：

- 每个类型自带不变量：拿到 ResolvedPaper，arxiv_id 一定已确定；拿到 PaperList，列表一定非空。
  做不到就是产出者失败，交给恢复层，不能产出一个「空的」实例。
- 类型只放下游真正要读的字段；resolution_debug 这类调试信息进 trace，不进类型。
- message、user_id、question 等是「环境输入」，始终可用，不算语义类型。

工具契约用 consumes / produces 引用这里的类型名，闭合检查见 plan_closure.py。
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Literal, Mapping, Optional, Tuple

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from .schemas import ArxivSearchSpec
from .state import AgentState


PaperOrigin = Literal["search", "candidates", "context"]


def _non_empty_arxiv_id(value: str) -> str:
    value = str(value or "").strip()
    if not value:
        raise ValueError("arxiv_id must not be empty")
    return value


class PaperRecord(BaseModel):
    """论文的共用结构，不是语义类型；其余字段沿用 search_arxiv 返回的论文 dict。"""

    model_config = ConfigDict(extra="allow")

    arxiv_id: str
    title: Optional[str] = None
    rank_reason: Optional[str] = None  # personalize_paper_results 写入

    _require_arxiv_id = field_validator("arxiv_id")(_non_empty_arxiv_id)


SearchSpec = ArxivSearchSpec


class PaperList(BaseModel):
    """不变量：papers 非空。"""

    papers: List[PaperRecord] = Field(min_length=1)
    origin: PaperOrigin
    search_spec: Optional[SearchSpec] = None  # origin=search 时有
    relaxation: Dict[str, Any] = Field(default_factory=dict)  # 阶梯检索记录
    personalized: bool = False


class ResolvedPaper(BaseModel):
    """不变量：arxiv_id 已确定。"""

    arxiv_id: str
    title: Optional[str] = None
    resolved_by: Literal["arxiv_id", "list_index", "selected", "candidate_id", "title_search"]
    paper: Optional[PaperRecord] = None

    _require_arxiv_id = field_validator("arxiv_id")(_non_empty_arxiv_id)


class IndexedPaper(BaseModel):
    """不变量：索引已就绪。从不由 state 投影，每轮都要 check，避免索引过期。"""

    paper: ResolvedPaper
    checked_at: datetime


class QAResult(BaseModel):
    paper: ResolvedPaper
    question: str
    mode: Literal["qa", "summary", "detail"]
    answer: str  # 可为空，由 assess_paper_qa_quality 判断
    sources: List[Dict[str, Any]] = Field(default_factory=list)
    cited_source_ids: List[str] = Field(default_factory=list)
    outcome: Optional[str] = None
    qa_observation: Dict[str, Any] = Field(default_factory=dict)


class CheckedQAResult(BaseModel):
    """不变量：可以直接给用户。"""

    result: QAResult
    decision: Literal["finalize", "finalize_with_degradation"]
    degraded_stages: List[str] = Field(default_factory=list)
    reason: str


class UserProfile(BaseModel):
    """不变量：画像或记忆摘要至少一个非空。"""

    user_id: str
    research_profile: Dict[str, Any] = Field(default_factory=dict)
    memory_summary: Any = None

    @model_validator(mode="after")
    def _require_profile(self) -> "UserProfile":
        if not self.research_profile and self.memory_summary in (None, "", [], {}):
            raise ValueError("user profile must not be empty")
        return self


class RecommendedPaper(PaperRecord):
    reason: str


class Recommendations(BaseModel):
    """不变量：items 非空。"""

    items: List[RecommendedPaper] = Field(min_length=1)
    candidates_origin: PaperOrigin


class PreferenceTarget(BaseModel):
    paper: ResolvedPaper
    action: str  # 取值沿用 update_preference_store 现有 action


class PreferenceUpdate(BaseModel):
    """不变量：写入已成功。"""

    paper: ResolvedPaper
    action: str


class AnswerDraft(BaseModel):
    text: str
    kind: Literal["search", "recommendation", "preference", "clarification", "fallback"]
    evidence_refs: List[str] = Field(default_factory=list)  # 引用到的 arxiv_id


SEMANTIC_TYPES: Dict[str, type[BaseModel]] = {
    "SearchSpec": SearchSpec,
    "PaperList": PaperList,
    "ResolvedPaper": ResolvedPaper,
    "IndexedPaper": IndexedPaper,
    "QAResult": QAResult,
    "CheckedQAResult": CheckedQAResult,
    "UserProfile": UserProfile,
    "Recommendations": Recommendations,
    "PreferenceTarget": PreferenceTarget,
    "PreferenceUpdate": PreferenceUpdate,
    "AnswerDraft": AnswerDraft,
}

# 每个类型一句说明，后续同时放进规划 prompt。
SEMANTIC_TYPE_DESCRIPTIONS: Dict[str, str] = {
    "SearchSpec": "一次 arXiv 结构化检索的参数（主题概念、分类、时间范围、数量、排序）。",
    "PaperList": "非空的论文列表，来自检索、候选论文或上一轮结果；个性化排序后仍是 PaperList。",
    "ResolvedPaper": "已确定 arxiv_id 的单篇目标论文。",
    "IndexedPaper": "全文索引已就绪的论文，只能由本轮的索引检查或建索引产出。",
    "QAResult": "对单篇论文的问答 / 总结结果，尚未经过质量判断，不能直接给用户。",
    "CheckedQAResult": "经过质量判断、可以给用户的论文问答结果。",
    "UserProfile": "非空的用户研究画像或记忆摘要。",
    "Recommendations": "非空的推荐论文列表，每篇带推荐理由。",
    "PreferenceTarget": "偏好写入的目标论文和动作，写入前由执行器请求用户确认。",
    "PreferenceUpdate": "已成功写入的偏好变更。",
    "AnswerDraft": "给用户的最终回复文本。",
}

# 能作为最终回复交付的类型。
FINAL_SEMANTIC_TYPES: Tuple[str, ...] = ("AnswerDraft", "CheckedQAResult")


@dataclass(frozen=True)
class TypeInput:
    """工具契约里的一条 consumes 声明：哪个语义类型、由哪些输入键承载。

    现有工具对同一类型用了不同键名（如 synthesize_arxiv_response 的 ranked_papers 和
    arxiv_results 都是 PaperList），所以一个类型可以对应多个键；required 的类型至少要有一个键被绑定。
    """

    semantic_type: str
    input_keys: Tuple[str, ...]
    required: bool = True


def needs(semantic_type: str, *input_keys: str) -> TypeInput:
    return TypeInput(semantic_type=semantic_type, input_keys=tuple(input_keys))


def accepts(semantic_type: str, *input_keys: str) -> TypeInput:
    return TypeInput(semantic_type=semantic_type, input_keys=tuple(input_keys), required=False)


@dataclass(frozen=True)
class ProjectedType:
    """state 中已有的一个语义类型实例及其来源。"""

    semantic_type: str
    source: str
    value: BaseModel


def project_state_types(state: AgentState, *, paper_ref: Optional[Mapping[str, Any]] = None) -> Dict[str, ProjectedType]:
    """根据 state 算出本轮开始时已经有哪些语义类型，结果同时给闭合检查和规划 prompt。

    每个类型都按自身模型校验，不满足不变量（比如上一轮论文缺 arxiv_id）就视为没有。
    paper_ref 是理解层给出的论文指代（B-3），形如 {"kind": "arxiv_id" | "list_index" | "selected", "value": ...}；
    只有能由代码直接定位的指代才投影成 ResolvedPaper，标题搜索和歧义仍交给 resolve_paper。
    IndexedPaper 从不投影：每轮都要 check，避免索引过期。
    """
    context = state.context if isinstance(state.context, Mapping) else {}
    projected: Dict[str, ProjectedType] = {}

    def _add(semantic_type: str, source: str, payload: Any) -> None:
        if semantic_type in projected:
            return
        try:
            value = SEMANTIC_TYPES[semantic_type].model_validate(payload)
        except (ValidationError, TypeError, ValueError):
            return
        projected[semantic_type] = ProjectedType(semantic_type=semantic_type, source=source, value=value)

    if state.search_spec is not None:
        _add("SearchSpec", "state.search_spec", state.search_spec.model_dump())

    last_papers = _paper_dicts(context.get("last_papers"))
    if last_papers:
        _add("PaperList", "context.last_papers", {"papers": last_papers, "origin": "context"})

    resolved = _resolve_paper_ref(paper_ref, context=context, last_papers=last_papers)
    if resolved is not None:
        _add("ResolvedPaper", f"paper_ref.{resolved['resolved_by']}", resolved)
    selected = context.get("selected_paper")
    if isinstance(selected, Mapping):
        _add("ResolvedPaper", "context.selected_paper", _resolved_payload(selected, resolved_by="selected"))

    research_profile = context.get("research_profile") if isinstance(context.get("research_profile"), Mapping) else {}
    memory_summary = context.get("user_memory_summary")
    if research_profile or memory_summary not in (None, "", [], {}):
        source = "context.research_profile" if research_profile else "context.user_memory_summary"
        _add(
            "UserProfile",
            source,
            {"user_id": str(state.user_id or ""), "research_profile": dict(research_profile), "memory_summary": memory_summary},
        )
    return projected


def _paper_dicts(value: Any) -> List[Dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [dict(item) for item in value if isinstance(item, Mapping)]


def _resolved_payload(paper: Mapping[str, Any], *, resolved_by: str) -> Dict[str, Any]:
    return {
        "arxiv_id": paper.get("arxiv_id"),
        "title": paper.get("title"),
        "resolved_by": resolved_by,
        "paper": dict(paper),
    }


def _resolve_paper_ref(
    paper_ref: Optional[Mapping[str, Any]],
    *,
    context: Mapping[str, Any],
    last_papers: List[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    """只处理代码能直接定位的指代：显式 arxiv_id、上一轮列表序号（从 1 开始）、当前选中论文。"""
    if not isinstance(paper_ref, Mapping):
        return None
    kind = str(paper_ref.get("kind") or "").strip()
    value = paper_ref.get("value")
    if kind == "arxiv_id" and str(value or "").strip():
        arxiv_id = str(value).strip()
        known = next((paper for paper in last_papers if str(paper.get("arxiv_id") or "").strip() == arxiv_id), None)
        return _resolved_payload(known or {"arxiv_id": arxiv_id}, resolved_by="arxiv_id")
    if kind == "list_index":
        try:
            index = int(value)
        except (TypeError, ValueError):
            return None
        if 1 <= index <= len(last_papers):
            return _resolved_payload(last_papers[index - 1], resolved_by="list_index")
        return None
    if kind == "selected" and isinstance(context.get("selected_paper"), Mapping):
        return _resolved_payload(context["selected_paper"], resolved_by="selected")
    return None


__all__ = [
    "AnswerDraft",
    "CheckedQAResult",
    "FINAL_SEMANTIC_TYPES",
    "IndexedPaper",
    "PaperList",
    "PaperRecord",
    "PreferenceTarget",
    "PreferenceUpdate",
    "ProjectedType",
    "QAResult",
    "RecommendedPaper",
    "Recommendations",
    "ResolvedPaper",
    "SEMANTIC_TYPES",
    "SEMANTIC_TYPE_DESCRIPTIONS",
    "SearchSpec",
    "TypeInput",
    "UserProfile",
    "accepts",
    "needs",
    "project_state_types",
]
