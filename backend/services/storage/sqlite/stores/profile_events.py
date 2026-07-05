import uuid
from typing import Any, Dict, List, Optional

from services.storage.sqlite.base import BaseSqliteStore
from services.storage.sqlite.shared import (
    DEFAULT_USER_ID,
    PROFILE_EXTRACTOR_VERSION,
    PROFILE_NORMALIZER_VERSION,
    logger,
)


PROFILE_EVENT_STRENGTHS = {
    "liked": 1.0,
    "disliked": -0.7,
    "favorite": 0.85,
    "later": 0.35,
    "read": 0.15,
    "not_interested": -0.45,
    "note_saved": 0.9,
    "qa_asked": 0.25,
    "manual_topic_added": 1.0,
    "manual_topic_removed": -1.0,
    "manual_topic_pinned": 1.2,
    "manual_topic_hidden": -1.2,
    "manual_style_updated": 0.4,
}

PROFILE_EVENT_ACTION_ALIASES = {
    "like": "liked",
    "liked": "liked",
    "dislike": "disliked",
    "disliked": "disliked",
    "favorite": "favorite",
    "later": "later",
    "read": "read",
    "not_interested": "not_interested",
    "note_included": "note_saved",
    "note_updated_in_profile": "note_saved",
}


class ProfileEventStore(BaseSqliteStore):
    """维护画像事件流和用户信号水位；具体业务动作仍由调用方决定是否写事件。"""

    def _ensure_user_profile_event_columns(self, conn):
        # 画像事件表从辅助审计升级为主证据流；旧库启动时补齐新列，避免手工迁移数据库。
        required_columns = {
            "event_type": "TEXT",
            "action_strength": "REAL DEFAULT 0",
            "source": "TEXT",
            "note_id": "TEXT",
            "session_id": "TEXT",
            "metadata_json": "TEXT",
            "include_in_profile": "INTEGER DEFAULT 1",
            "consumed_by_job_id": "TEXT",
            "consumed_at": "TIMESTAMP",
            "dedupe_key": "TEXT",
            "profile_dirty": "INTEGER DEFAULT 1",
        }
        cursor = conn.cursor()
        cursor.execute("PRAGMA table_info(user_profile_events)")
        existing_columns = {row[1] for row in cursor.fetchall()}
        for column_name, column_definition in required_columns.items():
            if column_name not in existing_columns:
                cursor.execute(
                    f"ALTER TABLE user_profile_events ADD COLUMN {column_name} {column_definition}"
                )
        cursor.execute(
            '''
            UPDATE user_profile_events
            SET event_type = COALESCE(event_type, action_type),
                source = COALESCE(source, source_type),
                metadata_json = COALESCE(metadata_json, payload_json),
                include_in_profile = COALESCE(include_in_profile, 1),
                profile_dirty = COALESCE(profile_dirty, 1)
            WHERE event_type IS NULL OR source IS NULL OR metadata_json IS NULL
            '''
        )
        conn.commit()

    def _record_preference_profile_event(self, conn, user_id: str, arxiv_id: str, event_type: str) -> None:
        """强偏好不再写入 paper-action 表；画像只通过独立事件流消费这类显式信号。"""
        self.record_user_profile_event(
            user_id=user_id,
            event_type=event_type,
            source_type="paper_preference",
            source_id=arxiv_id,
            action_type=event_type,
            arxiv_id=arxiv_id,
            metadata={"source": "explicit_preference"},
            include_in_profile=True,
            conn=conn,
        )

    def _deactivate_profile_events_for_paper(self, conn, user_id: str, arxiv_id: str, event_types: List[str]) -> None:
        """偏好或弱行为被撤销时停用对应画像证据，避免旧事件在重建时继续影响画像。"""
        normalized_types = [self._normalize_profile_event_type(item) for item in event_types if str(item or "").strip()]
        if not normalized_types:
            return
        placeholders = ", ".join("?" for _ in normalized_types)
        conn.execute(
            f'''
            UPDATE user_profile_events
            SET include_in_profile = 0, profile_dirty = 1
            WHERE user_id = ? AND arxiv_id = ? AND event_type IN ({placeholders})
            ''',
            [user_id, arxiv_id, *normalized_types],
        )

    def _record_profile_signal_removed_event(self, conn, user_id: str, arxiv_id: str, removed_event_type: str) -> None:
        """写入低权重删除事件，专门用于触发画像重建时重新计算已撤销的用户信号。"""
        normalized_removed = self._normalize_profile_event_type(removed_event_type)
        self.record_user_profile_event(
            user_id=user_id,
            event_type=f"{normalized_removed}_removed",
            source_type="paper_signal_removal",
            source_id=arxiv_id,
            action_type=f"{normalized_removed}_removed",
            arxiv_id=arxiv_id,
            metadata={"removed_event_type": normalized_removed},
            include_in_profile=True,
            conn=conn,
        )

    def record_user_profile_event(
        self,
        *,
        user_id: str,
        event_type: Optional[str] = None,
        source_type: str,
        action_type: str,
        source_id: Optional[str] = None,
        arxiv_id: Optional[str] = None,
        note_id: Optional[str] = None,
        session_id: Optional[str] = None,
        action_strength: Optional[float] = None,
        source: Optional[str] = None,
        payload: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        include_in_profile: bool = True,
        dedupe_key: Optional[str] = None,
        conn=None,
    ) -> str:
        event_id = str(uuid.uuid4())
        normalized_event_type = self._normalize_profile_event_type(event_type or action_type)
        normalized_source = str(source or source_type or "unknown").strip() or "unknown"
        normalized_arxiv_id = str(arxiv_id or "").strip() or None
        normalized_note_id = str(note_id or "").strip() or None
        normalized_session_id = str(session_id or "").strip() or None
        resolved_source_id = str(source_id or normalized_arxiv_id or normalized_note_id or normalized_session_id or "").strip() or None
        resolved_strength = float(action_strength if action_strength is not None else PROFILE_EVENT_STRENGTHS.get(normalized_event_type, 0.0))
        normalized_metadata = dict(metadata or payload or {})
        resolved_dedupe_key = dedupe_key or self._build_profile_event_dedupe_key(
            normalized_event_type,
            arxiv_id=normalized_arxiv_id,
            note_id=normalized_note_id,
            session_id=normalized_session_id,
            source_id=resolved_source_id,
            metadata=normalized_metadata,
        )
        def _write_event(connection) -> None:
            connection.execute(
                '''
                INSERT INTO user_profile_events (
                    event_id, user_id, event_type, source_type, source_id, action_type, action_strength,
                    source, arxiv_id, note_id, session_id, payload_json, metadata_json, include_in_profile,
                    consumed_by_job_id, consumed_at, dedupe_key, profile_dirty, extractor_version, normalizer_version
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, NULL, NULL, ?, ?, ?, ?)
                ON CONFLICT(user_id, dedupe_key) WHERE dedupe_key IS NOT NULL DO UPDATE SET
                    action_strength = excluded.action_strength,
                    source = excluded.source,
                    source_type = excluded.source_type,
                    action_type = excluded.action_type,
                    payload_json = excluded.payload_json,
                    metadata_json = excluded.metadata_json,
                    include_in_profile = excluded.include_in_profile,
                    consumed_by_job_id = NULL,
                    consumed_at = NULL,
                    profile_dirty = 1,
                    created_at = CURRENT_TIMESTAMP
                ''',
                (
                    event_id,
                    user_id,
                    normalized_event_type,
                    source_type,
                    resolved_source_id,
                    action_type,
                    resolved_strength,
                    normalized_source,
                    normalized_arxiv_id,
                    normalized_note_id,
                    normalized_session_id,
                    self._serialize_json_field(payload or {}),
                    self._serialize_json_field(normalized_metadata),
                    1 if include_in_profile else 0,
                    resolved_dedupe_key,
                    1,
                    PROFILE_EXTRACTOR_VERSION,
                    PROFILE_NORMALIZER_VERSION,
                ),
            )

        try:
            if conn is not None:
                # 调用方传入连接时说明外层正在维护事务；这里只追加事件，不抢先提交。
                _write_event(conn)
            else:
                with self._get_connection() as connection:
                    _write_event(connection)
                    connection.commit()
        except Exception as e:
            logger.error(f"Error recording profile event: {str(e)}")
        return event_id

    @staticmethod
    def _normalize_profile_event_type(value: Any) -> str:
        normalized = str(value or "").strip().lower()
        normalized = normalized.replace("-", "_")
        return PROFILE_EVENT_ACTION_ALIASES.get(normalized, normalized or "unknown")

    @staticmethod
    def _build_profile_event_dedupe_key(
        event_type: str,
        *,
        arxiv_id: Optional[str] = None,
        note_id: Optional[str] = None,
        session_id: Optional[str] = None,
        source_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        if event_type.startswith("manual_topic_"):
            topic = str((metadata or {}).get("topic") or "").strip().lower()
            return f"{event_type}:topic:{topic}" if topic else f"{event_type}:{source_id or 'manual'}"
        if event_type == "manual_style_updated":
            return "manual_style_updated"
        if note_id:
            return f"{event_type}:note:{note_id}"
        if arxiv_id:
            return f"{event_type}:paper:{arxiv_id}"
        if session_id:
            return f"{event_type}:session:{session_id}"
        return f"{event_type}:{source_id or uuid.uuid4()}"

    def list_user_profile_events(
        self,
        user_id: str = DEFAULT_USER_ID,
        *,
        event_types: Optional[List[str]] = None,
        include_consumed: bool = True,
        include_in_profile_only: bool = True,
        dirty_only: bool = False,
        since: Optional[str] = None,
        until: Optional[str] = None,
        limit: int = 500,
    ) -> List[Dict[str, Any]]:
        filters = ["user_id = ?"]
        values: List[Any] = [user_id]
        normalized_types = [self._normalize_profile_event_type(item) for item in (event_types or []) if str(item or "").strip()]
        if normalized_types:
            placeholders = ", ".join("?" for _ in normalized_types)
            filters.append(f"event_type IN ({placeholders})")
            values.extend(normalized_types)
        if not include_consumed:
            filters.append("consumed_by_job_id IS NULL")
        if include_in_profile_only:
            filters.append("include_in_profile = 1")
        if dirty_only:
            filters.append("profile_dirty = 1")
        if since:
            filters.append("created_at >= ?")
            values.append(since)
        if until:
            filters.append("created_at <= ?")
            values.append(until)
        values.append(max(1, int(limit or 500)))
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    f'''
                    SELECT event_id, user_id, event_type, source_type, source_id, action_type, action_strength,
                           source, arxiv_id, note_id, session_id, payload_json, metadata_json, include_in_profile,
                           consumed_by_job_id, consumed_at, dedupe_key, profile_dirty, created_at
                    FROM user_profile_events
                    WHERE {' AND '.join(filters)}
                    ORDER BY created_at DESC
                    LIMIT ?
                    ''',
                    values,
                )
                return [
                    {
                        "event_id": row[0],
                        "user_id": row[1],
                        "event_type": row[2] or row[5],
                        "source_type": row[3],
                        "source_id": row[4],
                        "action_type": row[5],
                        "action_strength": float(row[6] or 0.0),
                        "source": row[7],
                        "arxiv_id": row[8],
                        "note_id": row[9],
                        "session_id": row[10],
                        "payload": self._deserialize_json_field(row[11]) or {},
                        "metadata": self._deserialize_json_field(row[12]) or {},
                        "include_in_profile": bool(row[13]),
                        "consumed_by_job_id": row[14],
                        "consumed_at": row[15],
                        "dedupe_key": row[16],
                        "profile_dirty": bool(row[17]),
                        "created_at": row[18],
                    }
                    for row in cursor.fetchall()
                ]
        except Exception as e:
            logger.error(f"Error listing profile events: {str(e)}")
            return []

    def mark_user_profile_events_consumed(self, user_id: str, job_id: str, event_ids: List[str]) -> None:
        normalized_event_ids = [str(item or "").strip() for item in event_ids if str(item or "").strip()]
        if not normalized_event_ids:
            return
        placeholders = ", ".join("?" for _ in normalized_event_ids)
        try:
            with self._get_connection() as conn:
                conn.execute(
                    f'''
                    UPDATE user_profile_events
                    SET consumed_by_job_id = ?, consumed_at = CURRENT_TIMESTAMP, profile_dirty = 0
                    WHERE user_id = ? AND event_id IN ({placeholders})
                    ''',
                    [job_id, user_id, *normalized_event_ids],
                )
                conn.commit()
        except Exception as e:
            logger.error(f"Error marking profile events consumed: {str(e)}")

    def get_user_profile_dirty_event_count(self, user_id: str = DEFAULT_USER_ID) -> int:
        """返回尚未被画像构建消费的事件数，供调度器判断是否需要排队重建。"""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    '''
                    SELECT COUNT(*)
                    FROM user_profile_events
                    WHERE user_id = ? AND include_in_profile = 1 AND profile_dirty = 1
                    ''',
                    (user_id,),
                )
                row = cursor.fetchone()
                return int(row[0] or 0) if row else 0
        except Exception as e:
            logger.error(f"Error counting dirty profile events: {str(e)}")
            return 0

    def get_latest_user_preference_timestamp(self, user_id: str = DEFAULT_USER_ID) -> Optional[str]:
        """返回显式偏好表的最新写入时间，用于判断兴趣向量是否需要刷新。"""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    '''
                    SELECT MAX(latest_at) FROM (
                        SELECT created_at AS latest_at FROM user_liked_papers WHERE user_id = ?
                        UNION ALL
                        SELECT created_at AS latest_at FROM user_disliked_papers WHERE user_id = ?
                    )
                    ''',
                    (user_id, user_id),
                )
                row = cursor.fetchone()
                return row[0] if row and row[0] else None
        except Exception as e:
            logger.error(f"Error getting latest preference timestamp: {str(e)}")
            return None

    def get_latest_user_signal_timestamp(self, user_id: str = DEFAULT_USER_ID) -> Optional[str]:
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                # 这是画像/推荐重建的用户信号水位，不是单表事件查询；跨表 UNION 保留旧的脏数据判断语义。
                cursor.execute(
                    '''
                    SELECT MAX(latest_at) FROM (
                        SELECT created_at AS latest_at FROM user_liked_papers WHERE user_id = ?
                        UNION ALL
                        SELECT created_at AS latest_at FROM user_disliked_papers WHERE user_id = ?
                        UNION ALL
                        SELECT updated_at AS latest_at FROM user_paper_actions WHERE user_id = ?
                        UNION ALL
                        SELECT updated_at AS latest_at FROM user_research_profiles WHERE user_id = ?
                        UNION ALL
                        SELECT updated_at AS latest_at FROM user_manual_profiles WHERE user_id = ?
                        UNION ALL
                        SELECT updated_at AS latest_at FROM user_generated_profiles WHERE user_id = ?
                        UNION ALL
                        SELECT updated_at AS latest_at FROM user_effective_profiles WHERE user_id = ?
                        UNION ALL
                        SELECT created_at AS latest_at FROM user_profile_events WHERE user_id = ?
                    )
                    ''',
                    (user_id, user_id, user_id, user_id, user_id, user_id, user_id, user_id),
                )
                row = cursor.fetchone()
                return row[0] if row and row[0] else None
        except Exception as e:
            logger.error(f"Error getting latest user signal timestamp: {str(e)}")
            return self.get_latest_user_preference_timestamp(user_id)
