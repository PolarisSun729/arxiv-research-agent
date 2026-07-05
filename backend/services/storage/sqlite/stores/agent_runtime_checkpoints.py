from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional

from services.storage.sqlite.base import BaseSqliteStore
from services.storage.sqlite.shared import DEFAULT_USER_ID, logger


class AgentRuntimeCheckpointStore(BaseSqliteStore):
    """维护 Agent 业务 runtime checkpoint，负责确认真源、原子消费和过期清理。"""

    def _row_to_agent_runtime_checkpoint(self, row: Any) -> Dict[str, Any]:
        return {
            'checkpoint_id': row[0],
            'user_id': row[1],
            'session_id': row[2],
            'thread_id': row[3],
            'runtime_state': self._deserialize_json_field(row[4]) or None,
            'graph_state': self._deserialize_json_field(row[5]) or None,
            'pending_confirmation': self._deserialize_json_field(row[6]) or None,
            'current_node': row[7] or '',
            'next_route': row[8] or '',
            'status': row[9] or 'running',
            'error_summary': row[10] or '',
            'created_at': row[11],
            'updated_at': row[12],
            'expires_at': row[13],
        }

    def _runtime_checkpoint_approved_step_ids(self, raw_runtime_state: Any) -> List[str]:
        payload = self._deserialize_json_field(raw_runtime_state)
        if not isinstance(payload, dict):
            return []
        raw_step_ids = payload.get("approved_step_ids")
        if not isinstance(raw_step_ids, (list, tuple, set)):
            return []
        approved_step_ids: List[str] = []
        for item in raw_step_ids:
            step_id = str(item).strip()
            if step_id and step_id not in approved_step_ids:
                approved_step_ids.append(step_id)
        return approved_step_ids

    def _pending_confirmation_step_id(self, raw_pending_confirmation: Any) -> str:
        pending_confirmation = self._deserialize_json_field(raw_pending_confirmation)
        if not isinstance(pending_confirmation, dict):
            return ""
        return str(pending_confirmation.get("step_id") or "").strip()

    def _runtime_state_pending_confirmation_step_id(self, raw_runtime_state: Any) -> str:
        runtime_state = self._deserialize_json_field(raw_runtime_state)
        if not isinstance(runtime_state, dict):
            return ""
        return self._pending_confirmation_step_id(runtime_state.get("pending_confirmation"))

    def _runtime_state_plan_id(self, raw_runtime_state: Any) -> str:
        runtime_state = self._deserialize_json_field(raw_runtime_state)
        if not isinstance(runtime_state, dict):
            return ""
        plan = runtime_state.get("plan") if isinstance(runtime_state.get("plan"), dict) else {}
        return str(plan.get("plan_id") or runtime_state.get("plan_id") or "").strip()

    def _runtime_checkpoint_plan_matches(self, existing_runtime_state: Any, incoming_runtime_state: Any) -> bool:
        existing_plan_id = self._runtime_state_plan_id(existing_runtime_state)
        incoming_plan_id = self._runtime_state_plan_id(incoming_runtime_state)
        # 历史 checkpoint 或单元测试可能没有 plan_id；只有明确不一致时才关闭 replay 保护。
        return not existing_plan_id or not incoming_plan_id or existing_plan_id == incoming_plan_id

    def _is_consumed_agent_runtime_confirmation_replay(
        self,
        *,
        existing_checkpoint: Optional[Dict[str, Any]],
        incoming_runtime_state: Any,
        incoming_pending_confirmation: Any,
        incoming_status: str,
    ) -> bool:
        if not isinstance(existing_checkpoint, dict):
            return False
        if str(existing_checkpoint.get("status") or "").strip() != "running":
            return False
        if existing_checkpoint.get("pending_confirmation"):
            return False
        pending_step_id = (
            self._pending_confirmation_step_id(incoming_pending_confirmation)
            or self._runtime_state_pending_confirmation_step_id(incoming_runtime_state)
        )
        if not pending_step_id:
            return False
        incoming_is_waiting = str(incoming_status or "").strip() == "waiting_confirmation"
        incoming_has_pending = bool(self._deserialize_json_field(incoming_pending_confirmation)) or bool(
            self._runtime_state_pending_confirmation_step_id(incoming_runtime_state)
        )
        if not incoming_is_waiting and not incoming_has_pending:
            return False
        existing_runtime_state = existing_checkpoint.get("runtime_state")
        approved_step_ids = set(self._runtime_checkpoint_approved_step_ids(existing_runtime_state))
        if pending_step_id not in approved_step_ids:
            return False
        if not self._runtime_checkpoint_plan_matches(existing_runtime_state, incoming_runtime_state):
            return False
        return True

    def _merge_runtime_checkpoint_approved_step_ids(
        self,
        incoming_runtime_state: Any,
        *,
        existing_checkpoint: Optional[Dict[str, Any]],
    ) -> Any:
        if not isinstance(existing_checkpoint, dict):
            return incoming_runtime_state
        if str(existing_checkpoint.get("status") or "").strip() != "running":
            return incoming_runtime_state
        if existing_checkpoint.get("pending_confirmation"):
            return incoming_runtime_state
        existing_runtime_state = existing_checkpoint.get("runtime_state")
        if not self._runtime_checkpoint_plan_matches(existing_runtime_state, incoming_runtime_state):
            return incoming_runtime_state
        existing_step_ids = self._runtime_checkpoint_approved_step_ids(existing_runtime_state)
        if not existing_step_ids:
            return incoming_runtime_state

        runtime_state = self._deserialize_json_field(incoming_runtime_state)
        if isinstance(runtime_state, dict):
            runtime_state = dict(runtime_state)
        else:
            # 上游 running 快照偶尔缺少 runtime_state；这里复用已消费现场，避免批准态被空快照擦掉。
            existing_payload = self._deserialize_json_field(existing_runtime_state)
            runtime_state = dict(existing_payload) if isinstance(existing_payload, dict) else {}
        incoming_step_ids = self._runtime_checkpoint_approved_step_ids(runtime_state)
        merged_step_ids = list(dict.fromkeys([*existing_step_ids, *incoming_step_ids]))
        if merged_step_ids == incoming_step_ids:
            return incoming_runtime_state
        runtime_state["approved_step_ids"] = merged_step_ids
        return runtime_state

    def upsert_agent_runtime_checkpoint(
        self,
        *,
        user_id: str = DEFAULT_USER_ID,
        session_id: str,
        thread_id: str,
        runtime_state: Optional[Dict[str, Any]] = None,
        graph_state: Optional[Dict[str, Any]] = None,
        pending_confirmation: Optional[Dict[str, Any]] = None,
        current_node: Optional[str] = None,
        next_route: Optional[str] = None,
        status: str = 'running',
        error_summary: Optional[str] = None,
        expires_at: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """写入 Agent 执行现场 checkpoint。

        这张表记录的是可恢复执行现场，不是前端展示镜像；pending_action 仍由 agent_sessions 保存，
        但 resume 校验必须以这里的 pending_confirmation/status 为准。
        """
        try:
            normalized_user_id = str(user_id or DEFAULT_USER_ID).strip() or DEFAULT_USER_ID
            normalized_session_id = str(session_id or '').strip()
            normalized_thread_id = str(thread_id or normalized_session_id).strip()
            if not normalized_session_id or not normalized_thread_id:
                return None

            checkpoint_id = f"{normalized_user_id}:{normalized_session_id}:{normalized_thread_id}"
            normalized_status = str(status or 'running').strip() or 'running'
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    '''
                    SELECT checkpoint_id, user_id, session_id, thread_id, runtime_state_json,
                           graph_state_json, pending_confirmation_json, current_node, next_route,
                           status, error_summary, created_at, updated_at, expires_at
                    FROM agent_runtime_checkpoints
                    WHERE user_id = ? AND session_id = ? AND thread_id = ?
                    ''',
                    (normalized_user_id, normalized_session_id, normalized_thread_id),
                )
                existing_row = cursor.fetchone()
                existing_checkpoint = self._row_to_agent_runtime_checkpoint(existing_row) if existing_row else None
                if self._is_consumed_agent_runtime_confirmation_replay(
                    existing_checkpoint=existing_checkpoint,
                    incoming_runtime_state=runtime_state,
                    incoming_pending_confirmation=pending_confirmation,
                    incoming_status=normalized_status,
                ):
                    # LangGraph resume 可能重放中断前 waiting 快照；一旦 DB 已记录批准态，就不能回写旧 pending。
                    logger.debug(
                        "Skip stale agent runtime confirmation replay: session_id=%s thread_id=%s status=%s",
                        normalized_session_id,
                        normalized_thread_id,
                        normalized_status,
                    )
                    return existing_checkpoint
                runtime_state_to_write = self._merge_runtime_checkpoint_approved_step_ids(
                    runtime_state,
                    existing_checkpoint=existing_checkpoint,
                )
                cursor.execute(
                    '''
                    INSERT INTO agent_runtime_checkpoints (
                        checkpoint_id, user_id, session_id, thread_id, runtime_state_json,
                        graph_state_json, pending_confirmation_json, current_node, next_route,
                        status, error_summary, expires_at
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(user_id, session_id, thread_id) DO UPDATE SET
                        runtime_state_json = excluded.runtime_state_json,
                        graph_state_json = excluded.graph_state_json,
                        pending_confirmation_json = excluded.pending_confirmation_json,
                        current_node = excluded.current_node,
                        next_route = excluded.next_route,
                        status = excluded.status,
                        error_summary = excluded.error_summary,
                        expires_at = excluded.expires_at,
                        updated_at = CURRENT_TIMESTAMP
                    ''',
                    (
                        checkpoint_id,
                        normalized_user_id,
                        normalized_session_id,
                        normalized_thread_id,
                        self._serialize_json_field(runtime_state_to_write),
                        self._serialize_json_field(graph_state),
                        self._serialize_json_field(pending_confirmation),
                        str(current_node or '').strip(),
                        str(next_route or '').strip(),
                        normalized_status,
                        str(error_summary or '').strip(),
                        expires_at,
                    ),
                )
                conn.commit()
            return self.get_agent_runtime_checkpoint(
                user_id=normalized_user_id,
                session_id=normalized_session_id,
                thread_id=normalized_thread_id,
            )
        except Exception as e:
            logger.error(f"Error upserting agent runtime checkpoint: {str(e)}")
            return None

    def get_agent_runtime_checkpoint(
        self,
        *,
        user_id: str = DEFAULT_USER_ID,
        session_id: str,
        thread_id: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        try:
            normalized_user_id = str(user_id or DEFAULT_USER_ID).strip() or DEFAULT_USER_ID
            normalized_session_id = str(session_id or '').strip()
            normalized_thread_id = str(thread_id or normalized_session_id).strip()
            if not normalized_session_id or not normalized_thread_id:
                return None
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    '''
                    SELECT checkpoint_id, user_id, session_id, thread_id, runtime_state_json,
                           graph_state_json, pending_confirmation_json, current_node, next_route,
                           status, error_summary, created_at, updated_at, expires_at
                    FROM agent_runtime_checkpoints
                    WHERE user_id = ? AND session_id = ? AND thread_id = ?
                    ''',
                    (normalized_user_id, normalized_session_id, normalized_thread_id),
                )
                row = cursor.fetchone()
                return self._row_to_agent_runtime_checkpoint(row) if row else None
        except Exception as e:
            logger.error(f"Error getting agent runtime checkpoint: {str(e)}")
            return None

    def list_agent_runtime_checkpoints_by_thread(
        self,
        *,
        session_id: str,
        thread_id: Optional[str] = None,
        limit: int = 20,
    ) -> List[Dict[str, Any]]:
        """按 session/thread 列出候选 runtime checkpoint，供恢复态丢失 user_id 时二次校验。"""
        try:
            normalized_session_id = str(session_id or '').strip()
            normalized_thread_id = str(thread_id or normalized_session_id).strip()
            if not normalized_session_id or not normalized_thread_id:
                return []
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    '''
                    SELECT checkpoint_id, user_id, session_id, thread_id, runtime_state_json,
                           graph_state_json, pending_confirmation_json, current_node, next_route,
                           status, error_summary, created_at, updated_at, expires_at
                    FROM agent_runtime_checkpoints
                    WHERE session_id = ? AND thread_id = ?
                    ORDER BY updated_at DESC, created_at DESC
                    LIMIT ?
                    ''',
                    (normalized_session_id, normalized_thread_id, max(int(limit or 1), 1)),
                )
                rows = cursor.fetchall()
                return [self._row_to_agent_runtime_checkpoint(row) for row in rows]
        except Exception as e:
            logger.error(f"Error listing agent runtime checkpoints by thread: {str(e)}")
            return []

    def get_agent_runtime_checkpoint_by_thread(
        self,
        *,
        session_id: str,
        thread_id: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """按 session/thread 找回唯一 runtime checkpoint，兼容旧调用方的单记录接口。"""
        try:
            checkpoints = self.list_agent_runtime_checkpoints_by_thread(
                session_id=session_id,
                thread_id=thread_id,
                limit=2,
            )
            # 旧接口只在原始候选唯一时返回，避免缺失 user_id 时误读其他用户现场。
            if len(checkpoints) != 1:
                return None
            return checkpoints[0]
        except Exception as e:
            logger.error(f"Error getting agent runtime checkpoint by thread: {str(e)}")
            return None

    def _runtime_state_without_pending_confirmation(
        self,
        raw_runtime_state: Any,
        *,
        decision: Optional[str] = None,
        step_id: Optional[str] = None,
    ) -> str:
        """清理 runtime_state_json 内嵌的确认真源，避免查库或后续诊断看到旧 pending。

        关键修复：在批准决策时，必须把 pending_confirmation 中记录的真正目标 step_id 添加到
        approved_step_ids，而不仅仅是传入的 step_id。这对于缺索引补丁链等桥接确认场景至关重要。
        """
        payload = self._deserialize_json_field(raw_runtime_state)
        if not isinstance(payload, dict):
            return self._serialize_json_field(payload)

        previous_pending = payload.get("pending_confirmation") if isinstance(payload.get("pending_confirmation"), dict) else {}
        # 桥接确认场景里，resume payload 和 runtime pending_confirmation 都指向目标副作用 step；
        # recovery_strategy.step_id 才能告诉我们哪个桥接 step 已经完成，避免它继续停在 waiting_confirmation。
        recovery_strategy = payload.get("recovery_strategy") if isinstance(payload.get("recovery_strategy"), dict) else {}
        bridge_step_id = str(recovery_strategy.get("step_id") or "").strip()
        pending_step_id = str(previous_pending.get("step_id") or "").strip()
        requested_step_id = str(step_id or "").strip()
        normalized_step_id = pending_step_id or requested_step_id
        normalized_decision = str(decision or "").strip().lower()
        payload["pending_confirmation"] = None

        # confirmation 已消费后，request_confirmation 恢复策略不再代表可恢复现场；拒绝分支保留 skip 语义。
        if normalized_decision == "reject":
            payload["recovery_strategy"] = {
                "type": "skip_step",
                "reason": "confirmation_rejected",
                **({"step_id": normalized_step_id} if normalized_step_id else {}),
            }
        elif normalized_decision == "approve" or str(recovery_strategy.get("type") or "").strip() == "request_confirmation":
            payload["recovery_strategy"] = None

        if normalized_decision == "approve" and normalized_step_id:
            approved_step_ids = [str(item).strip() for item in list(payload.get("approved_step_ids") or []) if str(item).strip()]
            if normalized_step_id not in approved_step_ids:
                approved_step_ids.append(normalized_step_id)
            payload["approved_step_ids"] = approved_step_ids

        step_status = payload.get("step_status") if isinstance(payload.get("step_status"), dict) else None
        if step_status is not None and normalized_step_id and step_status.get(normalized_step_id) == "waiting_confirmation":
            step_status[normalized_step_id] = "skipped" if normalized_decision == "reject" else "pending"
            payload["step_status"] = step_status
        if (
            step_status is not None
            and normalized_decision == "approve"
            and bridge_step_id
            and bridge_step_id != normalized_step_id
            and step_status.get(bridge_step_id) == "waiting_confirmation"
        ):
            # 批准的是目标副作用 step，桥接确认 step 的职责已经结束；如果继续保留 waiting，
            # LangGraph resume 后会把旧确认现场重新持久化，进而再次触发同一个确认门。
            step_status[bridge_step_id] = "success"
            payload["step_status"] = step_status
        if str(payload.get("turn_status") or "").strip() == "waiting_confirmation":
            payload["turn_status"] = None
        return self._serialize_json_field(payload)

    def mark_agent_runtime_checkpoint_status(
        self,
        *,
        user_id: str = DEFAULT_USER_ID,
        session_id: str,
        thread_id: Optional[str] = None,
        status: str,
        error_summary: Optional[str] = None,
        clear_pending_confirmation: bool = False,
    ) -> bool:
        """更新执行现场终态或过期状态，避免旧 confirmation 被重复 resume。"""
        try:
            normalized_user_id = str(user_id or DEFAULT_USER_ID).strip() or DEFAULT_USER_ID
            normalized_session_id = str(session_id or '').strip()
            normalized_thread_id = str(thread_id or normalized_session_id).strip()
            if not normalized_session_id or not normalized_thread_id:
                return False
            assignments = ['status = ?', 'error_summary = ?', 'updated_at = CURRENT_TIMESTAMP']
            values: List[Any] = [str(status or '').strip(), str(error_summary or '').strip()]
            with self._get_connection() as conn:
                cursor = conn.cursor()
                if clear_pending_confirmation:
                    cursor.execute(
                        '''
                        SELECT runtime_state_json
                        FROM agent_runtime_checkpoints
                        WHERE user_id = ? AND session_id = ? AND thread_id = ?
                        ''',
                        (normalized_user_id, normalized_session_id, normalized_thread_id),
                    )
                    row = cursor.fetchone()
                    cleaned_runtime_state = self._runtime_state_without_pending_confirmation(row[0] if row else None)
                    assignments.extend(["pending_confirmation_json = ''", "runtime_state_json = ?", "expires_at = NULL"])
                    values.append(cleaned_runtime_state)
                cursor.execute(
                    f'''
                    UPDATE agent_runtime_checkpoints
                    SET {", ".join(assignments)}
                    WHERE user_id = ? AND session_id = ? AND thread_id = ?
                    ''',
                    values + [normalized_user_id, normalized_session_id, normalized_thread_id],
                )
                conn.commit()
                return cursor.rowcount > 0
        except Exception as e:
            logger.error(f"Error marking agent runtime checkpoint status: {str(e)}")
            return False

    def consume_agent_runtime_pending_confirmation(
        self,
        *,
        user_id: str = DEFAULT_USER_ID,
        session_id: str,
        thread_id: Optional[str] = None,
        next_route: str = "running",
        decision: Optional[str] = None,
        step_id: Optional[str] = None,
        tool_name: Optional[str] = None,
        pending_action_id: Optional[str] = None,
    ) -> bool:
        """原子消费等待确认的业务 checkpoint。

        快速连续点击确认时，多个请求可能同时读到同一张确认卡片；这里用
        status='waiting_confirmation' 作为抢占条件，只有第一个请求能清空
        pending_confirmation 并进入 running，后续请求会因为 rowcount=0 被拒绝恢复。
        """
        try:
            normalized_user_id = str(user_id or DEFAULT_USER_ID).strip() or DEFAULT_USER_ID
            normalized_session_id = str(session_id or '').strip()
            normalized_thread_id = str(thread_id or normalized_session_id).strip()
            if not normalized_session_id or not normalized_thread_id:
                return False
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    '''
                    SELECT runtime_state_json, pending_confirmation_json
                    FROM agent_runtime_checkpoints
                    WHERE user_id = ?
                      AND session_id = ?
                      AND thread_id = ?
                      AND status = 'waiting_confirmation'
                      AND COALESCE(NULLIF(pending_confirmation_json, ''), '') <> ''
                    ''',
                    (normalized_user_id, normalized_session_id, normalized_thread_id),
                )
                row = cursor.fetchone()
                if not row:
                    return False
                pending_payload = self._deserialize_json_field(row[1]) if row else None
                pending_step_id = pending_payload.get("step_id") if isinstance(pending_payload, dict) else None
                pending_tool_name = pending_payload.get("tool_name") if isinstance(pending_payload, dict) else None
                stored_pending_action_id = pending_payload.get("pending_action_id") if isinstance(pending_payload, dict) else None
                normalized_step_id = str(step_id or "").strip()
                normalized_tool_name = str(tool_name or "").strip()
                normalized_pending_action_id = str(pending_action_id or "").strip()
                if normalized_step_id and pending_step_id and normalized_step_id != str(pending_step_id).strip():
                    # 消费动作必须绑定当前确认任务本身，避免旧按钮把新的 waiting 任务误消费。
                    return False
                if normalized_tool_name and pending_tool_name and normalized_tool_name != str(pending_tool_name).strip():
                    return False
                if (
                    normalized_pending_action_id
                    and stored_pending_action_id
                    and normalized_pending_action_id != str(stored_pending_action_id).strip()
                ):
                    return False
                cleaned_runtime_state = self._runtime_state_without_pending_confirmation(
                    row[0],
                    decision=decision,
                    step_id=normalized_step_id or pending_step_id,
                )
                cursor.execute(
                    '''
                    UPDATE agent_runtime_checkpoints
                    SET status = 'running',
                        pending_confirmation_json = '',
                        runtime_state_json = ?,
                        next_route = ?,
                        error_summary = '',
                        expires_at = NULL,
                        updated_at = CURRENT_TIMESTAMP
                    WHERE user_id = ?
                      AND session_id = ?
                      AND thread_id = ?
                      AND status = 'waiting_confirmation'
                      AND COALESCE(NULLIF(pending_confirmation_json, ''), '') <> ''
                    ''',
                    (
                        cleaned_runtime_state,
                        str(next_route or 'running').strip() or 'running',
                        normalized_user_id,
                        normalized_session_id,
                        normalized_thread_id,
                    ),
                )
                conn.commit()
                return cursor.rowcount > 0
        except Exception as e:
            logger.error(f"Error consuming agent runtime pending confirmation: {str(e)}")
            return False

    def expire_agent_runtime_checkpoints(self, *, now: Optional[str] = None) -> int:
        """把超过 expires_at 的等待现场标记为 expired，而不是直接删除。"""
        try:
            now_text = now or datetime.now(timezone.utc).isoformat()
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    '''
                    UPDATE agent_runtime_checkpoints
                    SET status = 'expired',
                        error_summary = COALESCE(NULLIF(error_summary, ''), 'checkpoint_expired'),
                        pending_confirmation_json = '',
                        updated_at = CURRENT_TIMESTAMP
                    WHERE status = 'waiting_confirmation'
                      AND expires_at IS NOT NULL
                      AND expires_at <= ?
                    ''',
                    (now_text,),
                )
                conn.commit()
                return int(cursor.rowcount or 0)
        except Exception as e:
            logger.error(f"Error expiring agent runtime checkpoints: {str(e)}")
            return 0

    def cleanup_agent_runtime_checkpoints(self, *, retention_days: int = 7) -> int:
        """删除已终止且超过保留期的 runtime checkpoint，避免持久化表无限增长。"""
        try:
            cutoff = datetime.now(timezone.utc) - timedelta(days=max(int(retention_days or 0), 1))
            cutoff_text = cutoff.isoformat()
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    '''
                    DELETE FROM agent_runtime_checkpoints
                    WHERE status IN ('completed', 'cancelled', 'failed', 'expired')
                      AND updated_at <= ?
                    ''',
                    (cutoff_text,),
                )
                conn.commit()
                return int(cursor.rowcount or 0)
        except Exception as e:
            logger.error(f"Error cleaning agent runtime checkpoints: {str(e)}")
            return 0

    def cleanup_langgraph_checkpoints_for_terminal_runtime(self, *, retention_days: int = 7) -> Dict[str, int]:
        """按业务 runtime checkpoint 生命周期同步清理 LangGraph 原始 checkpoint。

        runtime checkpoint 是恢复语义的权威记录；只有它进入 completed/failed/expired 等终态并超过保留期后，
        才删除同 thread_id 下的原始 graph checkpoint 与 writes，避免误删仍可恢复的确认现场。
        """
        result = {"threads": 0, "checkpoints": 0, "writes": 0}
        try:
            cutoff = datetime.now(timezone.utc) - timedelta(days=max(int(retention_days or 0), 1))
            cutoff_text = cutoff.isoformat()
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    '''
                    SELECT DISTINCT thread_id
                    FROM agent_runtime_checkpoints
                    WHERE status IN ('completed', 'cancelled', 'failed', 'expired')
                      AND updated_at <= ?
                    ''',
                    (cutoff_text,),
                )
                thread_ids = [str(row[0] or '').strip() for row in cursor.fetchall() if str(row[0] or '').strip()]
                for thread_id in thread_ids:
                    cursor.execute("DELETE FROM langgraph_checkpoint_writes WHERE thread_id = ?", (thread_id,))
                    result["writes"] += int(cursor.rowcount or 0)
                    cursor.execute("DELETE FROM langgraph_checkpoints WHERE thread_id = ?", (thread_id,))
                    result["checkpoints"] += int(cursor.rowcount or 0)
                conn.commit()
                result["threads"] = len(thread_ids)
            return result
        except Exception as e:
            logger.error(f"Error cleaning langgraph checkpoints: {str(e)}")
            return result

    def get_context_lifecycle_stats(
        self,
        *,
        user_id: str = DEFAULT_USER_ID,
        paper_session_id: Optional[str] = None,
        agent_session_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """汇总上下文相关表的轻量健康度，不读取大字段正文。"""
        normalized_user_id = str(user_id or DEFAULT_USER_ID).strip() or DEFAULT_USER_ID
        stats: Dict[str, Any] = {
            "user_id": normalized_user_id,
            "paper_chat": {},
            "agent_runtime_checkpoint": {},
            "langgraph_checkpoint": {},
        }
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                if paper_session_id:
                    cursor.execute(
                        '''
                        SELECT s.message_count, COUNT(m.message_id), s.summary_updated_at,
                               s.summary_turn_count, s.summary_last_turn_id,
                               LENGTH(COALESCE(s.summary_json, ''))
                        FROM paper_chat_sessions s
                        LEFT JOIN paper_chat_messages m ON m.session_id = s.session_id
                        WHERE s.session_id = ? AND s.user_id = ?
                        GROUP BY s.session_id
                        ''',
                        (paper_session_id, normalized_user_id),
                    )
                    row = cursor.fetchone()
                    stats["paper_chat"] = {
                        "session_id": paper_session_id,
                        "message_count": int((row or [0])[0] or 0) if row else 0,
                        "stored_message_count": int((row or [0, 0])[1] or 0) if row else 0,
                        "summary_loaded": bool(row and row[2]),
                        "summary_updated_at": row[2] if row else None,
                        "summary_turn_count": int(row[3] or 0) if row else 0,
                        "summary_last_turn_id": row[4] if row else "",
                        "summary_chars": int(row[5] or 0) if row else 0,
                    }
                if agent_session_id:
                    cursor.execute(
                        '''
                        SELECT status, current_node, next_route, expires_at, updated_at,
                               LENGTH(COALESCE(runtime_state_json, '')),
                               LENGTH(COALESCE(graph_state_json, '')),
                               LENGTH(COALESCE(pending_confirmation_json, ''))
                        FROM agent_runtime_checkpoints
                        WHERE user_id = ? AND session_id = ? AND thread_id = ?
                        ''',
                        (normalized_user_id, agent_session_id, agent_session_id),
                    )
                    row = cursor.fetchone()
                    stats["agent_runtime_checkpoint"] = {
                        "session_id": agent_session_id,
                        "exists": bool(row),
                        "status": row[0] if row else None,
                        "current_node": row[1] if row else "",
                        "next_route": row[2] if row else "",
                        "expires_at": row[3] if row else None,
                        "updated_at": row[4] if row else None,
                        "runtime_state_chars": int(row[5] or 0) if row else 0,
                        "graph_state_chars": int(row[6] or 0) if row else 0,
                        "pending_confirmation_chars": int(row[7] or 0) if row else 0,
                    }
                    cursor.execute("SELECT COUNT(*) FROM langgraph_checkpoints WHERE thread_id = ?", (agent_session_id,))
                    checkpoint_count = int((cursor.fetchone() or [0])[0] or 0)
                    cursor.execute("SELECT COUNT(*) FROM langgraph_checkpoint_writes WHERE thread_id = ?", (agent_session_id,))
                    write_count = int((cursor.fetchone() or [0])[0] or 0)
                    stats["langgraph_checkpoint"] = {
                        "thread_id": agent_session_id,
                        "checkpoint_count": checkpoint_count,
                        "write_count": write_count,
                    }
            return stats
        except Exception as e:
            logger.error(f"Error getting context lifecycle stats: {str(e)}")
            stats["error"] = str(e)
            return stats
