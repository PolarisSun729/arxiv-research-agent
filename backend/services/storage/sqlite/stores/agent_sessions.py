import uuid
from typing import Any, Dict, List, Optional

from services.storage.sqlite.base import BaseSqliteStore
from services.storage.sqlite.shared import DEFAULT_USER_ID, logger


class AgentSessionStore(BaseSqliteStore):
    """维护 Agent 会话记忆的轻量 CRUD；运行时恢复 checkpoint 仍由主库服务中的状态机方法负责。"""

    def _row_to_agent_session(self, row: Any) -> Dict[str, Any]:
        return {
            'session_id': row[0],
            'user_id': row[1],
            'status': row[2] or 'active',
            'selected_paper': self._deserialize_json_field(row[3]) or None,
            'last_papers': self._deserialize_json_field(row[4]) or [],
            'paper_qa_result': self._deserialize_json_field(row[5]) or None,
            'active_arxiv_id': row[6] or '',
            'active_paper_session_id': row[7] or '',
            'last_intent': row[8] or '',
            'last_tool_calls_summary': self._deserialize_json_field(row[9]) or [],
            'last_response_summary': row[10] or '',
            'created_at': row[11],
            'updated_at': row[12],
        }

    def create_or_get_agent_session(
        self,
        user_id: str = DEFAULT_USER_ID,
        session_id: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        try:
            normalized_user_id = str(user_id or DEFAULT_USER_ID).strip() or DEFAULT_USER_ID
            normalized_session_id = str(session_id or uuid.uuid4()).strip()
            existing = self.get_agent_session(normalized_session_id, user_id=normalized_user_id)
            if existing:
                return existing

            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    '''
                    INSERT INTO agent_sessions (session_id, user_id, status)
                    VALUES (?, ?, ?)
                    ''',
                    (normalized_session_id, normalized_user_id, 'active'),
                )
                conn.commit()
            return self.get_agent_session(normalized_session_id, user_id=normalized_user_id)
        except Exception as e:
            logger.error(f"Error creating or getting agent session: {str(e)}")
            return None

    def get_agent_session(self, session_id: str, user_id: str = DEFAULT_USER_ID) -> Optional[Dict[str, Any]]:
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    '''
                    SELECT session_id, user_id, status, selected_paper_json, last_papers_json,
                           paper_qa_result_json, active_arxiv_id,
                           active_paper_session_id, last_intent, last_tool_calls_summary_json,
                           last_response_summary, created_at, updated_at
                    FROM agent_sessions
                    WHERE session_id = ? AND user_id = ?
                    ''',
                    (session_id, user_id),
                )
                row = cursor.fetchone()
                return self._row_to_agent_session(row) if row else None
        except Exception as e:
            logger.error(f"Error getting agent session: {str(e)}")
            return None

    def update_agent_session(
        self,
        session_id: str,
        user_id: str = DEFAULT_USER_ID,
        memory_patch: Optional[Dict[str, Any]] = None,
    ) -> bool:
        try:
            if not session_id:
                return False

            # Agent 记忆采用 patch 语义，只允许业务层显式声明的字段落库，避免临时运行态误写入长期会话。
            allowed_fields = {
                'status': 'status',
                'selected_paper': 'selected_paper_json',
                'last_papers': 'last_papers_json',
                'paper_qa_result': 'paper_qa_result_json',
                'active_arxiv_id': 'active_arxiv_id',
                'active_paper_session_id': 'active_paper_session_id',
                'last_intent': 'last_intent',
                'last_tool_calls_summary': 'last_tool_calls_summary_json',
                'last_response_summary': 'last_response_summary',
            }
            json_fields = {
                'selected_paper_json',
                'last_papers_json',
                'paper_qa_result_json',
                'last_tool_calls_summary_json',
            }

            normalized_patch = dict(memory_patch or {})
            update_fields = []
            update_values: List[Any] = []
            for patch_key, column_name in allowed_fields.items():
                if patch_key not in normalized_patch:
                    continue
                value = normalized_patch.get(patch_key)
                update_fields.append(f'{column_name} = ?')
                if column_name in json_fields:
                    update_values.append(self._serialize_json_field(value))
                else:
                    update_values.append(None if value is None else str(value).strip())

            if not update_fields:
                return True

            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    '''
                    INSERT OR IGNORE INTO agent_sessions (session_id, user_id, status)
                    VALUES (?, ?, ?)
                    ''',
                    (session_id, user_id, 'active'),
                )
                update_fields.append('updated_at = CURRENT_TIMESTAMP')
                update_values.extend([session_id, user_id])
                cursor.execute(
                    f'''
                    UPDATE agent_sessions
                    SET {", ".join(update_fields)}
                    WHERE session_id = ? AND user_id = ?
                    ''',
                    update_values,
                )
                conn.commit()
                return cursor.rowcount > 0
        except Exception as e:
            logger.error(f"Error updating agent session: {str(e)}")
            return False

    def clear_agent_session(self, session_id: str, user_id: str = DEFAULT_USER_ID) -> bool:
        return self.update_agent_session(
            session_id=session_id,
            user_id=user_id,
            memory_patch={
                'status': 'cleared',
                'selected_paper': None,
                'last_papers': None,
                'paper_qa_result': None,
                'active_arxiv_id': None,
                'active_paper_session_id': None,
                'last_intent': None,
                'last_tool_calls_summary': None,
                'last_response_summary': None,
            },
        )
