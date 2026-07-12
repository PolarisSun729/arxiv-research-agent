import hashlib
import json
from typing import Any, Dict, Literal

from pydantic import BaseModel, ConfigDict


ApprovalGrantStatus = Literal["approved", "consumed", "revoked", "expired"]


def arguments_fingerprint(arguments: Dict[str, Any]) -> str:
    """为最终绑定参数生成稳定指纹，使授权只放行用户实际确认的那次调用。"""

    canonical = json.dumps(arguments, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)
    return f"sha256:{hashlib.sha256(canonical.encode('utf-8')).hexdigest()}"


class ApprovalGrant(BaseModel):
    """记录一次与 plan、step、工具和最终参数绑定的副作用授权。"""

    model_config = ConfigDict(extra="forbid")

    grant_id: str
    interaction_id: str
    user_id: str
    session_id: str
    thread_id: str
    plan_id: str
    step_id: str
    tool_name: str
    arguments_fingerprint: str
    status: ApprovalGrantStatus = "approved"
    approved_at: str
    consumed_at: str | None = None
    revoked_at: str | None = None
