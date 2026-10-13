from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

from .actions import ResearchAction
from .contracts import PaperEvidenceResearchRequest, PaperEvidenceResearchResult


class EvidenceNeed(BaseModel):
    need_id: str
    description: str
    importance: Literal["core", "supporting"] = "supporting"
    status: Literal["provisional", "open", "satisfied", "blocked", "superseded", "abandoned"] = "provisional"
    directly_required_by_question: bool = False
    parent_need_id: str | None = None
    supporting_claim_ids: list[str] = Field(default_factory=list)
    verified_evidence_ids: list[str] = Field(default_factory=list)
    blocking_reason: str | None = None


class EvidenceCandidate(BaseModel):
    candidate_id: str
    content: str
    chunk_type: str = "text"
    chunk_id: str = ""
    parent_chunk_id: str = ""
    original_chunk_id: str = ""
    section_path: str = ""
    page_number: int | None = None
    table_id: str = ""
    matched_need_ids: list[str] = Field(default_factory=list)
    appearance_count: int = 1
    # 图表资产：起草和决策时由 ContextPackBuilder 据此附原图、渲染图表摘要。
    # asset_abs_path 是服务器本地路径，只在研究链路内部流转，不能写进轨迹或对外结果。
    # 不带 table_evidence：ContextPackBuilder 会给结构化表格另起 table-evidence-* 引用 ID，与候选 ID 对不上。
    asset_kind: str = ""
    asset_path: str = ""
    asset_abs_path: str = ""
    asset_summary: str = ""
    asset_preview_text: str = ""
    asset_section_match_type: str = ""
    asset_section_match_confidence: float = 0.0
    asset_section_match_allow_embedding: bool = False


class EvidenceContextPack(BaseModel):
    selected_need_ids: list[str] = Field(default_factory=list)
    candidates: list[EvidenceCandidate] = Field(default_factory=list)
    total_context_chars: int = 0


class DraftAnswer(BaseModel):
    draft_id: str
    version: int
    answer: str
    declared_claims: list[dict[str, Any]] = Field(default_factory=list)
    used_candidate_ids: list[str] = Field(default_factory=list)


class AnswerClaim(BaseModel):
    claim_id: str
    text: str
    importance: Literal["core", "supporting", "background"] = "supporting"
    addressed_need_ids: list[str] = Field(default_factory=list)
    citation_ids: list[str] = Field(default_factory=list)
    # 一期没有 FigureVerifier；该标记阻止依赖图像本体读取的主张被文本校验结果误放行。
    requires_visual_verification: bool = False


class ClaimAssessment(BaseModel):
    claim_id: str
    verdict: Literal["supported", "partially_supported", "unsupported", "conflicting", "citation_mismatch"]
    supporting_evidence_ids: list[str] = Field(default_factory=list)
    missing_facets: list[str] = Field(default_factory=list)


class DraftRevisionNote(BaseModel):
    """上一版草稿里没通过校验的一句话，以及校验器给出的修改线索。"""

    claim_text: str
    reason_code: str
    reason: str
    # 校验器认为能支持这句话的证据；引用挂错时照此改挂即可。
    suggested_citation_ids: list[str] = Field(default_factory=list)
    missing_facets: list[str] = Field(default_factory=list)


class DraftGenerationRequest(BaseModel):
    version: int
    research_question: str
    addressed_need_ids: list[str]
    context_pack: EvidenceContextPack
    preferred_answer_style: str = ""
    # 重写时带上上一版草稿与逐句校验结论；为空表示首次起草。不带这些，重写只是同样输入再生成一遍。
    previous_answer: str = ""
    revision_notes: list[DraftRevisionNote] = Field(default_factory=list)

    @property
    def candidates(self) -> list[EvidenceCandidate]:
        """兼容脚本适配器的候选读取方式，对外仍由上下文包持有选择结果。"""

        return self.context_pack.candidates


class ClaimExtractionRequest(BaseModel):
    research_question: str
    answer: str
    draft_version: int
    # 主张必须绑定证据需求账本才能参与覆盖投影；提取器读不到完整研究状态，由图显式传入。
    evidence_needs: list[EvidenceNeed] = Field(default_factory=list)
    # 草稿声明使用的候选及其 matched_need_ids 是主张归属的最强信号：为某需求检索回来的
    # 证据被某句引用，该句大概率在回应这个需求。图只传草稿引用过的候选，不暴露整个池。
    candidates: list[EvidenceCandidate] = Field(default_factory=list)


class ClaimVerificationRequest(BaseModel):
    research_question: str
    draft_version: int
    claims: list[AnswerClaim]
    context_pack: EvidenceContextPack

    @property
    def candidates(self) -> list[EvidenceCandidate]:
        """校验器只读取本轮上下文包，不能绕过选择边界访问完整候选池。"""

        return self.context_pack.candidates


class NeedEvidenceDigest(BaseModel):
    """需求到已召回证据的对应关系；证据全文在 ResearchDecisionContext.evidence 里，这里不重复。"""

    need_id: str
    candidate_count: int = 0
    candidate_ids: list[str] = Field(default_factory=list)


class ResearchDecisionContext(BaseModel):
    research_question: str
    evidence_needs: list[EvidenceNeed]
    candidate_count: int
    current_draft_version: int | None
    claim_assessments: list[ClaimAssessment]
    retrievals_remaining: int
    drafts_remaining: int
    # 规则兜底策略所需的最小进度信息：已消耗的检索轮数与已有候选覆盖到的需求集合。
    retrievals_used: int = 0
    candidate_need_ids: list[str] = Field(default_factory=list)
    # 证据池满后新候选无法入池，继续检索必然无进展；决策器需要据此转向起草。
    evidence_pool_capacity: int = 0
    evidence_pool_full: bool = False
    need_evidence: list[NeedEvidenceDigest] = Field(default_factory=list)
    # 池中全部候选的全文（按入池顺序，即召回排名顺序）。只看摘要的决策器会把对题证据误判为
    # 无关而弃答（0a82ae71），所以决策时必须读原文；图片候选由策略层据资产字段附原图。
    evidence: list[EvidenceCandidate] = Field(default_factory=list)


class VerifiedDraftSnapshot(BaseModel):
    """某版草稿校验后的完整快照；重写草稿可能比上一版更差，终止时要能退回最好的那版。"""

    draft: DraftAnswer
    claims: list[AnswerClaim]
    claim_assessments: list[ClaimAssessment]
    evidence_needs: list[EvidenceNeed]
    satisfied_core_need_count: int
    supported_claim_count: int


class PaperEvidenceResearchState(BaseModel):
    request: PaperEvidenceResearchRequest
    research_question: str = ""
    evidence_needs: list[EvidenceNeed] = Field(default_factory=list)
    evidence_candidates: dict[str, EvidenceCandidate] = Field(default_factory=dict)
    current_draft: DraftAnswer | None = None
    claims: list[AnswerClaim] = Field(default_factory=list)
    claim_assessments: list[ClaimAssessment] = Field(default_factory=list)
    pending_action: ResearchAction | None = None
    action_accepted: bool = False
    action_rejection_code: str | None = None
    retrieval_count: int = 0
    # 按需求保存尚未恢复的技术故障；轨迹可以裁剪，终态判断仍必须可恢复且确定。
    retrieval_failures: dict[str, str] = Field(default_factory=dict)
    draft_attempt_count: int = 0
    verification_count: int = 0
    citation_repair_count: int = 0
    no_progress_count: int = 0
    # 连续无进展后检索通道关闭：后续只允许基于现有证据池起草、定稿或弃答。
    retrieval_stalled: bool = False
    best_verified_draft: VerifiedDraftSnapshot | None = None
    # 决策器判定无关、已移出证据池的候选；后续检索再召回它们时不再入池。
    discarded_candidate_ids: list[str] = Field(default_factory=list)
    invalid_action_count: int = 0
    trace_events: list[dict[str, Any]] = Field(default_factory=list)
    result: PaperEvidenceResearchResult | None = None
