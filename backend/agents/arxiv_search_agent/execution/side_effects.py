from typing import Any
from uuid import uuid4

from services.storage.sqlite.stores import ApprovalGrantStore


class SideEffectInvocationService:
    """封装授权消费与副作用 invocation 生命周期，执行器不直接访问持久化 store。"""

    def __init__(self, store: ApprovalGrantStore | None) -> None:
        self.store = store

    def prepare(
        self,
        *,
        grant_id: str,
        plan_id: str,
        step_id: str,
        tool_name: str,
        arguments_fingerprint: str,
    ) -> str:
        if self.store is None:
            raise RuntimeError("approval store is required for side-effect execution")
        invocation_id = str(uuid4())
        # 授权消费与 prepared invocation 必须在同一事务完成，避免崩溃后重复调用外部副作用。
        self.store.consume_and_prepare_invocation(
            grant_id=grant_id,
            invocation_id=invocation_id,
            plan_id=plan_id,
            step_id=step_id,
            tool_name=tool_name,
            arguments_fingerprint=arguments_fingerprint,
        )
        return invocation_id

    def mark_indeterminate(self, invocation_id: str, error: BaseException) -> None:
        if self.store is not None:
            self.store.mark_invocation(invocation_id, status="indeterminate", error_code=error.__class__.__name__)

    def mark_terminal(self, invocation_id: str, *, succeeded: bool, result_summary: Any, error_code: str = "") -> None:
        if self.store is not None:
            self.store.mark_invocation(
                invocation_id,
                status="succeeded" if succeeded else "failed",
                result_summary=result_summary,
                error_code=error_code,
            )
