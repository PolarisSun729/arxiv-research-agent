import uuid
from typing import Any, Dict, List, Optional

from services.storage.sqlite.base import BaseSqliteStore
from services.storage.sqlite.shared import DEFAULT_USER_ID, logger, normalize_user_id

_MESSAGE_COLUMNS = (
    'message_id, session_id, user_id, role, content, response_json, status, resume_run_id, created_at'
)


class AgentChatMessageStore(BaseSqliteStore):
    """Agent 搜索页的对话流水，只用于前端回放历史，不参与 Agent 规划上下文。

    Agent 跨轮记忆仍由 agent_sessions 的结构化字段负责；这里按原样保存每条气泡，
    刷新页面或换设备后可以 1:1 还原对话。所有读取都带 user_id 条件，拿到别人的
    session_id 也读不到内容。
    """

    def _row_to_message(self, row: Any) -> Dict[str, Any]:
        return {
            'message_id': row[0],
            'session_id': row[1],
            'user_id': row[2],
            'role': row[3],
            'content': row[4] or '',
            'response': self._deserialize_json_field(row[5]),
            'status': row[6] or 'completed',
            'resume_run_id': row[7] or None,
            'created_at': row[8],
        }

    def append_agent_chat_message(
        self,
        session_id: str,
        role: str,
        content: str,
        user_id: str = DEFAULT_USER_ID,
        response: Optional[Dict[str, Any]] = None,
        status: str = 'completed',
        resume_run_id: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """追加一条消息；失败时返回 None，调用方不应因此中断 Agent 回复。"""
        try:
            message_id = str(uuid.uuid4())
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    '''
                    INSERT INTO agent_chat_messages (
                        message_id, session_id, user_id, role, content, response_json, status, resume_run_id
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    ''',
                    (
                        message_id,
                        session_id,
                        normalize_user_id(user_id),
                        role,
                        content or '',
                        self._serialize_json_field(response),
                        status,
                        resume_run_id,
                    ),
                )
                conn.commit()
                cursor.execute(
                    f'SELECT {_MESSAGE_COLUMNS} FROM agent_chat_messages WHERE message_id = ?',
                    (message_id,),
                )
                row = cursor.fetchone()
                return self._row_to_message(row) if row else None
        except Exception as e:
            logger.error(f"Error appending agent chat message: {str(e)}")
            return None

    def list_agent_chat_messages(self, session_id: str, user_id: str = DEFAULT_USER_ID) -> List[Dict[str, Any]]:
        """按写入顺序返回会话全部消息；rowid 保证同一秒内写入的问答次序稳定。"""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    f'''
                    SELECT {_MESSAGE_COLUMNS}
                    FROM agent_chat_messages
                    WHERE session_id = ? AND user_id = ?
                    ORDER BY created_at ASC, rowid ASC
                    ''',
                    (session_id, normalize_user_id(user_id)),
                )
                return [self._row_to_message(row) for row in cursor.fetchall()]
        except Exception as e:
            logger.error(f"Error listing agent chat messages: {str(e)}")
            return []

    def list_agent_chat_sessions(self, user_id: str = DEFAULT_USER_ID, *, limit: int = 30) -> List[Dict[str, Any]]:
        """列出用户有对话记录的会话，按最近消息时间倒序；标题取会话第一条用户消息。"""
        normalized_limit = max(1, min(int(limit or 1), 100))
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    '''
                    SELECT m.session_id,
                           (
                               SELECT first_user.content
                               FROM agent_chat_messages first_user
                               WHERE first_user.session_id = m.session_id
                                 AND first_user.user_id = m.user_id
                                 AND first_user.role = 'user'
                               ORDER BY first_user.created_at ASC, first_user.rowid ASC
                               LIMIT 1
                           ) AS title,
                           COUNT(*) AS message_count,
                           MIN(m.created_at) AS created_at,
                           MAX(m.created_at) AS updated_at
                    FROM agent_chat_messages m
                    WHERE m.user_id = ?
                    GROUP BY m.session_id
                    ORDER BY MAX(m.created_at) DESC, MAX(m.rowid) DESC
                    LIMIT ?
                    ''',
                    (normalize_user_id(user_id), normalized_limit),
                )
                return [
                    {
                        'session_id': row[0],
                        'title': row[1] or '',
                        'message_count': int(row[2] or 0),
                        'created_at': row[3],
                        'updated_at': row[4],
                    }
                    for row in cursor.fetchall()
                ]
        except Exception as e:
            logger.error(f"Error listing agent chat sessions: {str(e)}")
            return []
