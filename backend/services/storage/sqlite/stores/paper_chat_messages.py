import sqlite3
import uuid
from typing import Any, Dict, List, Optional

from services.storage.sqlite.base import BaseSqliteStore
from services.storage.sqlite.shared import DEFAULT_USER_ID, logger
from services.storage.sqlite.stores.paper_chat_sessions import PaperChatSessionStore
from services.storage.sqlite.stores.profile_events import ProfileEventStore


class PaperChatMessageStore(BaseSqliteStore):
    """维护论文问答消息生命周期；双消息 QA turn 的原子写入仍由主流程单独处理。"""

    def __init__(self, connection_provider, session_store: PaperChatSessionStore, profile_event_store: ProfileEventStore) -> None:
        super().__init__(connection_provider)
        self.session_store = session_store
        self.profile_event_store = profile_event_store

    def record_user_profile_event(self, *args, **kwargs):
        """消息写入只负责记录弱信号，具体事件去重和状态流转交给画像事件 Store。"""
        return self.profile_event_store.record_user_profile_event(*args, **kwargs)

    def _row_to_paper_chat_session(self, row: Any) -> Dict[str, Any]:
        return self.session_store._row_to_paper_chat_session(row)

    def _row_to_paper_chat_message(self, row: Any) -> Dict[str, Any]:
        return {
            'message_id': row[0],
            'turn_id': row[1] or '',
            'session_id': row[2],
            'role': row[3],
            'content': row[4] or '',
            'sources': self._deserialize_json_field(row[5]) or [],
            'retrieval_debug_snapshot': self._deserialize_json_field(row[6]),
            'contextualized_question': row[7] or '',
            'question_contextualization': self._deserialize_json_field(row[8]),
            'status': row[9] or 'completed',
            'created_at': row[10],
        }

    def _get_paper_chat_message_in_transaction(self, cursor: sqlite3.Cursor, message_id: str) -> Optional[Dict[str, Any]]:
        cursor.execute(
            '''
            SELECT message_id, turn_id, session_id, role, content, sources,
                   retrieval_debug_snapshot, contextualized_question, question_contextualization,
                   status, created_at
            FROM paper_chat_messages
            WHERE message_id = ?
            ''',
            (message_id,),
        )
        row = cursor.fetchone()
        return self._row_to_paper_chat_message(row) if row else None

    def _get_paper_chat_session_in_transaction(self, cursor: sqlite3.Cursor, session_id: str) -> Optional[Dict[str, Any]]:
        cursor.execute(
            '''
            SELECT session_id, user_id, arxiv_id, title, created_at, updated_at, message_count, status,
                   summary_json, summary_updated_at, summary_turn_count, summary_last_turn_id
            FROM paper_chat_sessions
            WHERE session_id = ?
            ''',
            (session_id,),
        )
        row = cursor.fetchone()
        return self._row_to_paper_chat_session(row) if row else None

    def _refresh_paper_chat_session_stats(self, conn: sqlite3.Connection, session_id: str) -> None:
        cursor = conn.cursor()
        cursor.execute(
            '''
            SELECT COUNT(*), MAX(created_at)
            FROM paper_chat_messages
            WHERE session_id = ?
            ''',
            (session_id,),
        )
        row = cursor.fetchone() or (0, None)
        message_count = int(row[0] or 0)
        latest_created_at = row[1]
        if latest_created_at:
            cursor.execute(
                '''
                UPDATE paper_chat_sessions
                SET message_count = ?, updated_at = ?, status = COALESCE(status, 'active')
                WHERE session_id = ?
                ''',
                (message_count, latest_created_at, session_id),
            )
        else:
            cursor.execute(
                '''
                UPDATE paper_chat_sessions
                SET message_count = 0, updated_at = CURRENT_TIMESTAMP, status = COALESCE(status, 'active')
                WHERE session_id = ?
                ''',
                (session_id,),
            )

    def list_paper_chat_messages(self, session_id: str, user_id: str = DEFAULT_USER_ID) -> List[Dict[str, Any]]:
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    '''
                    SELECT m.message_id, m.turn_id, m.session_id, m.role, m.content, m.sources,
                           m.retrieval_debug_snapshot, m.contextualized_question, m.question_contextualization,
                           m.status, m.created_at
                    FROM paper_chat_messages m
                    JOIN paper_chat_sessions s ON s.session_id = m.session_id
                    WHERE m.session_id = ? AND s.user_id = ?
                    ORDER BY m.created_at ASC, m.message_id ASC
                    ''',
                    (session_id, user_id),
                )
                return [self._row_to_paper_chat_message(row) for row in cursor.fetchall()]
        except Exception as e:
            logger.error(f"Error listing paper chat messages: {str(e)}")
            return []

    def count_paper_chat_messages(self, session_id: str, user_id: str = DEFAULT_USER_ID) -> int:
        """统计会话消息总数，供上下文 debug 区分历史规模和本次实际读取规模。"""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    '''
                    SELECT COUNT(*)
                    FROM paper_chat_messages m
                    JOIN paper_chat_sessions s ON s.session_id = m.session_id
                    WHERE m.session_id = ? AND s.user_id = ?
                    ''',
                    (session_id, user_id),
                )
                row = cursor.fetchone()
                return int((row or [0])[0] or 0)
        except Exception as e:
            logger.error(f"Error counting paper chat messages: {str(e)}")
            return 0

    def list_recent_paper_chat_messages(
        self,
        session_id: str,
        user_id: str = DEFAULT_USER_ID,
        *,
        limit: int = 20,
    ) -> List[Dict[str, Any]]:
        """只读取最近若干条消息并恢复正序，专供模型上下文构造使用。

        前端历史展示仍使用 list_paper_chat_messages；这里刻意限制读取窗口，
        避免长会话在构造 prompt 时先把完整历史加载到 Python 内存。
        """
        normalized_limit = max(1, int(limit or 1))
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    '''
                    SELECT message_id, turn_id, session_id, role, content, sources,
                           retrieval_debug_snapshot, contextualized_question, question_contextualization,
                           status, created_at
                    FROM (
                        SELECT m.rowid AS message_rowid,
                               m.message_id, m.turn_id, m.session_id, m.role, m.content, m.sources,
                               m.retrieval_debug_snapshot, m.contextualized_question, m.question_contextualization,
                               m.status, m.created_at
                        FROM paper_chat_messages m
                        JOIN paper_chat_sessions s ON s.session_id = m.session_id
                        WHERE m.session_id = ? AND s.user_id = ?
                        ORDER BY m.created_at DESC, m.rowid DESC
                        LIMIT ?
                    ) recent_messages
                    ORDER BY created_at ASC, message_rowid ASC
                    ''',
                    (session_id, user_id, normalized_limit),
                )
                return [self._row_to_paper_chat_message(row) for row in cursor.fetchall()]
        except Exception as e:
            logger.error(f"Error listing recent paper chat messages: {str(e)}")
            return []

    def get_paper_chat_message(self, message_id: str, user_id: str = DEFAULT_USER_ID) -> Optional[Dict[str, Any]]:
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    '''
                    SELECT m.message_id, m.turn_id, m.session_id, m.role, m.content, m.sources,
                           m.retrieval_debug_snapshot, m.contextualized_question, m.question_contextualization,
                           m.status, m.created_at
                    FROM paper_chat_messages m
                    JOIN paper_chat_sessions s ON s.session_id = m.session_id
                    WHERE m.message_id = ? AND s.user_id = ?
                    ''',
                    (message_id, user_id),
                )
                row = cursor.fetchone()
                return self._row_to_paper_chat_message(row) if row else None
        except Exception as e:
            logger.error(f"Error getting paper chat message: {str(e)}")
            return None

    def get_paper_chat_message_by_turn(
        self,
        session_id: str,
        turn_id: str,
        *,
        role: str = 'assistant',
        user_id: str = DEFAULT_USER_ID,
    ) -> Optional[Dict[str, Any]]:
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    '''
                    SELECT m.message_id, m.turn_id, m.session_id, m.role, m.content, m.sources,
                           m.retrieval_debug_snapshot, m.contextualized_question, m.question_contextualization,
                           m.status, m.created_at
                    FROM paper_chat_messages m
                    JOIN paper_chat_sessions s ON s.session_id = m.session_id
                    WHERE m.session_id = ? AND m.turn_id = ? AND m.role = ? AND s.user_id = ?
                    ORDER BY m.created_at DESC, m.message_id DESC
                    LIMIT 1
                    ''',
                    (session_id, turn_id, role, user_id),
                )
                row = cursor.fetchone()
                return self._row_to_paper_chat_message(row) if row else None
        except Exception as e:
            logger.error(f"Error getting paper chat message by turn: {str(e)}")
            return None

    def append_paper_chat_message(
        self,
        session_id: str,
        role: str,
        content: str,
        user_id: str = DEFAULT_USER_ID,
        turn_id: Optional[str] = None,
        sources: Optional[Any] = None,
        retrieval_debug_snapshot: Optional[Any] = None,
        contextualized_question: Optional[str] = None,
        question_contextualization: Optional[Any] = None,
        status: str = 'completed',
    ) -> Optional[Dict[str, Any]]:
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    '''
                    SELECT session_id, title, arxiv_id FROM paper_chat_sessions
                    WHERE session_id = ? AND user_id = ?
                    ''',
                    (session_id, user_id),
                )
                session_row = cursor.fetchone()
                if not session_row:
                    return None

                message_id = str(uuid.uuid4())
                normalized_turn_id = str(turn_id or uuid.uuid4())
                cursor.execute(
                    '''
                    INSERT INTO paper_chat_messages (
                        message_id, turn_id, session_id, role, content, sources,
                        retrieval_debug_snapshot, contextualized_question, question_contextualization, status
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ''',
                    (
                        message_id,
                        normalized_turn_id,
                        session_id,
                        role,
                        content,
                        self._serialize_json_field(sources),
                        self._serialize_json_field(retrieval_debug_snapshot),
                        str(contextualized_question or ''),
                        self._serialize_json_field(question_contextualization),
                        status,
                    ),
                )

                if role == 'user' and not str(session_row[1] or '').strip() and content:
                    cursor.execute(
                        '''
                        UPDATE paper_chat_sessions
                        SET title = ?, updated_at = CURRENT_TIMESTAMP
                        WHERE session_id = ?
                        ''',
                        (str(content).strip()[:80], session_id),
                    )

                if role == 'user' and str(content or '').strip():
                    # 单消息写入路径同样记录 qa_asked 事件，和原子 turn 写入保持画像证据边界一致。
                    self.record_user_profile_event(
                        user_id=user_id,
                        event_type="qa_asked",
                        source_type="paper_qa",
                        source_id=message_id,
                        action_type="qa_asked",
                        arxiv_id=str(session_row[2] or "").strip(),
                        session_id=session_id,
                        metadata={"turn_id": normalized_turn_id, "question": str(content or "").strip()},
                        include_in_profile=True,
                        conn=conn,
                    )
                self._refresh_paper_chat_session_stats(conn, session_id)
                conn.commit()

            messages = self.list_paper_chat_messages(session_id=session_id, user_id=user_id)
            return next((item for item in messages if item['message_id'] == message_id), None)
        except Exception as e:
            logger.error(f"Error appending paper chat message: {str(e)}")
            return None

    def clear_paper_chat_session(self, session_id: str, user_id: str = DEFAULT_USER_ID) -> bool:
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    '''
                    DELETE FROM paper_chat_messages
                    WHERE session_id = ?
                      AND session_id IN (
                        SELECT session_id FROM paper_chat_sessions WHERE session_id = ? AND user_id = ?
                      )
                    ''',
                    (session_id, session_id, user_id),
                )
                self._refresh_paper_chat_session_stats(conn, session_id)
                conn.commit()
                return True
        except Exception as e:
            logger.error(f"Error clearing paper chat session: {str(e)}")
            return False

    def delete_paper_chat_session(self, session_id: str, user_id: str = DEFAULT_USER_ID) -> bool:
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    '''
                    DELETE FROM paper_chat_messages
                    WHERE session_id = ?
                    ''',
                    (session_id,),
                )
                cursor.execute(
                    '''
                    DELETE FROM paper_chat_sessions
                    WHERE session_id = ? AND user_id = ?
                    ''',
                    (session_id, user_id),
                )
                conn.commit()
                return cursor.rowcount > 0
        except Exception as e:
            logger.error(f"Error deleting paper chat session: {str(e)}")
            return False
