import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from services.storage.sqlite.base import BaseSqliteStore
from services.storage.sqlite.shared import DEFAULT_USER_ID, logger


class PaperChatSessionStore(BaseSqliteStore):
    """维护论文问答会话的基础 CRUD；消息写入和 QA turn 事务仍留在消息流程。"""

    def _ensure_paper_chat_session_summary_columns(self, conn):
        # 会话摘要是运行时压缩视图，旧库启动时补列；完整消息仍保留在 paper_chat_messages。
        required_columns = {
            "summary_json": "TEXT",
            "summary_updated_at": "TIMESTAMP",
            "summary_turn_count": "INTEGER DEFAULT 0",
            "summary_last_turn_id": "TEXT",
        }
        cursor = conn.cursor()
        cursor.execute("PRAGMA table_info(paper_chat_sessions)")
        existing_columns = {row[1] for row in cursor.fetchall()}
        for column_name, column_definition in required_columns.items():
            if column_name not in existing_columns:
                cursor.execute(
                    f"ALTER TABLE paper_chat_sessions ADD COLUMN {column_name} {column_definition}"
                )
        conn.commit()

    def _row_to_paper_chat_session(self, row: Any) -> Dict[str, Any]:
        return {
            'session_id': row[0],
            'user_id': row[1],
            'arxiv_id': row[2],
            'title': row[3] or '',
            'created_at': row[4],
            'updated_at': row[5],
            'message_count': row[6] or 0,
            'status': row[7] or 'active',
            'summary': (self._deserialize_json_field(row[8]) or None) if len(row) > 8 else None,
            'summary_updated_at': row[9] if len(row) > 9 else None,
            'summary_turn_count': (row[10] or 0) if len(row) > 10 else 0,
            'summary_last_turn_id': (row[11] or '') if len(row) > 11 else '',
        }

    def create_paper_chat_session(
        self,
        arxiv_id: str,
        user_id: str = DEFAULT_USER_ID,
        title: Optional[str] = None,
        status: str = 'active',
        session_id: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        try:
            normalized_session_id = str(session_id or uuid.uuid4())
            normalized_title = str(title or '').strip()
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    '''
                    INSERT INTO paper_chat_sessions (session_id, user_id, arxiv_id, title, status)
                    VALUES (?, ?, ?, ?, ?)
                    ''',
                    (normalized_session_id, user_id, arxiv_id, normalized_title, status),
                )
                conn.commit()
            return self.get_paper_chat_session(normalized_session_id, user_id=user_id)
        except Exception as e:
            logger.error(f"Error creating paper chat session: {str(e)}")
            return None

    def get_paper_chat_session(self, session_id: str, user_id: str = DEFAULT_USER_ID) -> Optional[Dict[str, Any]]:
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    '''
                    SELECT session_id, user_id, arxiv_id, title, created_at, updated_at, message_count, status,
                           summary_json, summary_updated_at, summary_turn_count, summary_last_turn_id
                    FROM paper_chat_sessions
                    WHERE session_id = ? AND user_id = ?
                    ''',
                    (session_id, user_id),
                )
                row = cursor.fetchone()
                return self._row_to_paper_chat_session(row) if row else None
        except Exception as e:
            logger.error(f"Error getting paper chat session: {str(e)}")
            return None

    def list_paper_chat_sessions(
        self,
        arxiv_id: str,
        user_id: str = DEFAULT_USER_ID,
        limit: int = 20,
    ) -> List[Dict[str, Any]]:
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    '''
                    SELECT session_id, user_id, arxiv_id, title, created_at, updated_at, message_count, status,
                           summary_json, summary_updated_at, summary_turn_count, summary_last_turn_id
                    FROM paper_chat_sessions
                    WHERE arxiv_id = ? AND user_id = ?
                    ORDER BY updated_at DESC, created_at DESC
                    LIMIT ?
                    ''',
                    (arxiv_id, user_id, limit),
                )
                return [self._row_to_paper_chat_session(row) for row in cursor.fetchall()]
        except Exception as e:
            logger.error(f"Error listing paper chat sessions: {str(e)}")
            return []

    def get_recent_paper_chat_session(self, arxiv_id: str, user_id: str = DEFAULT_USER_ID) -> Optional[Dict[str, Any]]:
        sessions = self.list_paper_chat_sessions(arxiv_id=arxiv_id, user_id=user_id, limit=1)
        return sessions[0] if sessions else None

    def update_paper_chat_session(self, session_id: str, user_id: str = DEFAULT_USER_ID, **kwargs) -> bool:
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                update_fields = []
                update_values = []
                for field_name in ('title', 'status', 'summary_updated_at', 'summary_turn_count', 'summary_last_turn_id'):
                    if field_name in kwargs:
                        update_fields.append(f'{field_name} = ?')
                        update_values.append(kwargs[field_name])
                if 'summary' in kwargs:
                    update_fields.append('summary_json = ?')
                    update_values.append(self._serialize_json_field(kwargs.get('summary')))
                update_fields.append('updated_at = CURRENT_TIMESTAMP')
                update_values.extend([session_id, user_id])
                cursor.execute(
                    f'''
                    UPDATE paper_chat_sessions
                    SET {", ".join(update_fields)}
                    WHERE session_id = ? AND user_id = ?
                    ''',
                    update_values,
                )
                conn.commit()
                return cursor.rowcount > 0
        except Exception as e:
            logger.error(f"Error updating paper chat session: {str(e)}")
            return False

    def update_paper_chat_session_summary(
        self,
        session_id: str,
        user_id: str = DEFAULT_USER_ID,
        *,
        summary: Optional[Dict[str, Any]],
        summary_turn_count: int,
        summary_last_turn_id: Optional[str],
        summary_updated_at: Optional[str] = None,
    ) -> bool:
        """更新会话摘要压缩视图；完整消息历史不受影响。"""
        return self.update_paper_chat_session(
            session_id,
            user_id=user_id,
            summary=dict(summary or {}),
            summary_turn_count=max(0, int(summary_turn_count or 0)),
            summary_last_turn_id=str(summary_last_turn_id or "").strip() or None,
            summary_updated_at=summary_updated_at or datetime.now(timezone.utc).isoformat(),
        )

    def delete_paper_chat_session(self, session_id: str, user_id: str = DEFAULT_USER_ID) -> bool:
        try:
            normalized_session_id = str(session_id or "").strip()
            normalized_user_id = str(user_id or DEFAULT_USER_ID).strip() or DEFAULT_USER_ID
            if not normalized_session_id:
                return False
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    '''
                    SELECT 1 FROM paper_chat_sessions
                    WHERE session_id = ? AND user_id = ?
                    ''',
                    (normalized_session_id, normalized_user_id),
                )
                if not cursor.fetchone():
                    return False

                # 不依赖连接级 foreign_keys 开关：会话删除必须稳定清掉消息，并保留笔记内容本身。
                cursor.execute(
                    '''
                    DELETE FROM paper_chat_messages
                    WHERE session_id = ?
                    ''',
                    (normalized_session_id,),
                )
                cursor.execute(
                    '''
                    UPDATE paper_notes
                    SET session_id = NULL
                    WHERE session_id = ? AND user_id = ?
                    ''',
                    (normalized_session_id, normalized_user_id),
                )
                cursor.execute(
                    '''
                    DELETE FROM paper_chat_sessions
                    WHERE session_id = ? AND user_id = ?
                    ''',
                    (normalized_session_id, normalized_user_id),
                )
                deleted = cursor.rowcount > 0
                conn.commit()
                return deleted
        except Exception as e:
            logger.error(f"Error deleting paper chat session: {str(e)}")
            return False
