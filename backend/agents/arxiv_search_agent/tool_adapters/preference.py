from __future__ import annotations

from typing import Any, Callable, Dict, Mapping, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .base import BaseToolAdapter, backend_tool_error
from .models import ToolExecutionResult


class ResolvePreferenceTargetInput(BaseModel):
    model_config = ConfigDict(extra="allow")

    message: str = ""
    selected_paper: Optional[Dict[str, Any]] = None
    context: Dict[str, Any] = Field(default_factory=dict)


class PreferenceTargetOutput(BaseModel):
    model_config = ConfigDict(extra="allow")

    arxiv_id: Optional[str] = None
    title: Optional[str] = None
    query: Optional[str] = None
    matched_by: Optional[str] = None
    source: Optional[str] = None


class UpdatePreferenceStoreInput(BaseModel):
    model_config = ConfigDict(extra="allow")

    paper_reference: Optional[Dict[str, Any]] = None
    message: str = ""
    user_id: Optional[str] = None

    @model_validator(mode="after")
    def _require_target(self) -> "UpdatePreferenceStoreInput":
        if not self.arxiv_id:
            raise ValueError("preference_target_missing")
        return self

    @property
    def arxiv_id(self) -> str:
        return str((self.paper_reference or {}).get("arxiv_id") or "").strip() if isinstance(self.paper_reference, Mapping) else ""


class PreferenceActionOutput(BaseModel):
    model_config = ConfigDict(extra="allow")

    status: str = "failed"
    action: Optional[str] = None
    label: Optional[str] = None
    arxiv_id: Optional[str] = None
    liked: Optional[bool] = None
    title: str = ""
    message: str = ""
    paper: Optional[Dict[str, Any]] = None
    error: Optional[Any] = None
    tool_result: Optional[Dict[str, Any]] = None


class VerifyPreferenceUpdateInput(BaseModel):
    model_config = ConfigDict(extra="allow")

    preference_action_result: Optional[Dict[str, Any]] = None


class VerifiedPreferenceOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ok: bool
    detail: Optional[Dict[str, Any]] = None


class SynthesizePreferenceResponseInput(BaseModel):
    model_config = ConfigDict(extra="allow")

    verified_preference_update: Optional[Dict[str, Any]] = None


class PreferenceResponseOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    final_answer: str


class ResolvePreferenceTargetAdapter(BaseToolAdapter[ResolvePreferenceTargetInput, PreferenceTargetOutput]):
    tool_name = "resolve_preference_target"
    input_model = ResolvePreferenceTargetInput
    output_model = PreferenceTargetOutput

    def __init__(self, resolve_paper_adapter: Any) -> None:
        self.resolve_paper_adapter = resolve_paper_adapter

    def _run(self, tool_input: ResolvePreferenceTargetInput) -> PreferenceTargetOutput:
        result = self.resolve_paper_adapter.execute(tool_input)
        if result.ok and isinstance(result.data, BaseModel):
            return PreferenceTargetOutput.model_validate(result.data.model_dump())
        return PreferenceTargetOutput(query=tool_input.message)


class UpdatePreferenceStoreAdapter(BaseToolAdapter[UpdatePreferenceStoreInput, PreferenceActionOutput]):
    tool_name = "update_preference_store"
    input_model = UpdatePreferenceStoreInput
    output_model = PreferenceActionOutput

    def __init__(self, invoke_backend_tool: Callable[..., Dict[str, Any]]) -> None:
        self.invoke_backend_tool = invoke_backend_tool

    def execute(self, tool_input: UpdatePreferenceStoreInput) -> ToolExecutionResult:
        result = super().execute(tool_input)
        if result.ok and isinstance(result.data, PreferenceActionOutput) and result.data.tool_result and not bool(result.data.tool_result.get("ok", False)):
            backend_result = result.data.tool_result
            return result.model_copy(
                update={
                    "ok": False,
                    "error": backend_tool_error(
                        error_code=((backend_result.get("error") or {}).get("code") if isinstance(backend_result.get("error"), Mapping) else None) or "preference_write_failed",
                        message=str((backend_result.get("error") or {}).get("message") if isinstance(backend_result.get("error"), Mapping) else backend_result.get("summary") or "偏好写入失败"),
                        detail={"backend_error": backend_result.get("error")},
                        suggested_recovery="ask_clarification",
                        safe_debug={"backend_tool_name": "record/remove_paper_preference"},
                    ),
                }
            )
        return result

    def _run(self, tool_input: UpdatePreferenceStoreInput) -> PreferenceActionOutput:
        paper_reference = tool_input.paper_reference or {}
        message = str(tool_input.message or "")
        lowered = message.lower()
        remove_scope = None
        if any(token in lowered for token in ("取消喜欢", "取消不喜欢", "撤销喜欢", "撤销不喜欢", "unlike", "remove like", "remove dislike")):
            remove_scope = "disliked" if any(token in lowered for token in ("取消不喜欢", "撤销不喜欢", "remove dislike")) else "liked"

        # 只把真实偏好写入或删除委托给后端工具；兴趣画像/向量重建不在这里伪装同步。
        if remove_scope:
            tool_result = self.invoke_backend_tool("remove_paper_preference", user_id=tool_input.user_id or "", arxiv_id=tool_input.arxiv_id, remove_scope=remove_scope)
            tool_data = (tool_result or {}).get("data") if isinstance(tool_result, Mapping) else {}
            return PreferenceActionOutput(
                status="success" if bool((tool_result or {}).get("ok")) else "failed",
                action="remove",
                label="none",
                arxiv_id=tool_input.arxiv_id,
                title=str((paper_reference or {}).get("title") or "").strip(),
                message=str((tool_data or {}).get("message") or (tool_result or {}).get("summary") or "已取消偏好标记"),
                paper=dict(paper_reference),
                error=None if bool((tool_result or {}).get("ok")) else ((tool_result or {}).get("error") or {}).get("message") if isinstance((tool_result or {}).get("error"), Mapping) else None,
                tool_result=dict(tool_result or {}),
            )

        liked = not any(token in lowered for token in ("不喜欢", "dislike", "thumbs down"))
        tool_result = self.invoke_backend_tool("record_paper_preference", user_id=tool_input.user_id or "", arxiv_id=tool_input.arxiv_id, liked=liked, paper=dict(paper_reference))
        tool_data = (tool_result or {}).get("data") if isinstance(tool_result, Mapping) else {}
        return PreferenceActionOutput(
            status="success" if bool((tool_result or {}).get("ok")) else "failed",
            action="like" if liked else "dislike",
            label="liked" if liked else "disliked",
            arxiv_id=tool_input.arxiv_id,
            liked=liked,
            title=str((paper_reference or {}).get("title") or "").strip(),
            message=str((tool_data or {}).get("message") or (tool_result or {}).get("summary") or "偏好已更新"),
            paper=(tool_data or {}).get("paper") or dict(paper_reference),
            error=None if bool((tool_result or {}).get("ok")) else ((tool_result or {}).get("error") or {}).get("message") if isinstance((tool_result or {}).get("error"), Mapping) else None,
            tool_result=dict(tool_result or {}),
        )


class VerifyPreferenceUpdateAdapter(BaseToolAdapter[VerifyPreferenceUpdateInput, VerifiedPreferenceOutput]):
    tool_name = "verify_preference_update"
    input_model = VerifyPreferenceUpdateInput
    output_model = VerifiedPreferenceOutput

    def _run(self, tool_input: VerifyPreferenceUpdateInput) -> VerifiedPreferenceOutput:
        action_result = tool_input.preference_action_result
        return VerifiedPreferenceOutput(
            ok=isinstance(action_result, Mapping)
            and bool(action_result.get("arxiv_id"))
            and str(action_result.get("status") or "success") == "success",
            detail=dict(action_result or {}) if isinstance(action_result, Mapping) else None,
        )


class SynthesizePreferenceResponseAdapter(BaseToolAdapter[SynthesizePreferenceResponseInput, PreferenceResponseOutput]):
    tool_name = "synthesize_preference_response"
    input_model = SynthesizePreferenceResponseInput
    output_model = PreferenceResponseOutput

    def _run(self, tool_input: SynthesizePreferenceResponseInput) -> PreferenceResponseOutput:
        verified = tool_input.verified_preference_update if isinstance(tool_input.verified_preference_update, Mapping) else {}
        detail = verified.get("detail") if isinstance(verified.get("detail"), Mapping) else {}
        if not verified.get("ok"):
            return PreferenceResponseOutput(final_answer="偏好更新未成功，请确认目标论文后重试。")
        action = str(detail.get("action") or "").strip()
        if action == "remove":
            return PreferenceResponseOutput(final_answer=f"已取消论文偏好：{detail.get('arxiv_id') or '目标论文'}。")
        return PreferenceResponseOutput(final_answer=f"已更新论文偏好：{detail.get('arxiv_id') or '目标论文'}。")
