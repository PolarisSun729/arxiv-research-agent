import sqlite3
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional

from services.storage.sqlite.base import BaseSqliteStore
from services.storage.sqlite.shared import DEFAULT_USER_ID, PaperQATurnPersistenceError, logger
from services.storage.sqlite.stores.paper_chat_sessions import PaperChatSessionStore
from services.storage.sqlite.stores.profile_events import ProfileEventStore


class PaperQATurnStore(BaseSqliteStore):
    """维护用户/助手双消息的原子 QA turn 写入，避免半轮对话污染短期记忆。"""

    def __init__(
        self,
        connection_provider,
        session_store: PaperChatSessionStore,
        profile_event_store: ProfileEventStore,
    ) -> None:
        super().__init__(connection_provider)
        self.session_store = session_store
        self.profile_event_store = profile_event_store

    def record_user_profile_event(self, *args, **kwargs):
        """原子 turn 写入只追加画像事件，事件归一化和消费状态由画像事件 Store 维护。"""
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
        """在同一事务中回读刚写入的消息，确认原子 turn 没有半写成功。"""
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
        """在事务提交前回读会话统计，确保返回给上层的是本轮写入后的状态。"""
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
        """根据消息表重算会话统计，避免双消息写入和会话计数出现不一致。"""
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

    def append_paper_qa_turn(
        self,
        *,
        session_id: str,
        user_id: str = DEFAULT_USER_ID,
        question: str,
        answer: str,
        sources: Optional[Any] = None,
        retrieval_debug_snapshot: Optional[Any] = None,
        contextualized_question: Optional[str] = None,
        question_contextualization: Optional[Any] = None,
        turn_id: Optional[str] = None,
        user_status: str = 'completed',
        assistant_status: str = 'completed',
    ) -> Dict[str, Any]:
        normalized_session_id = str(session_id or '').strip()
        normalized_user_id = str(user_id or DEFAULT_USER_ID).strip() or DEFAULT_USER_ID
        normalized_turn_id = str(turn_id or uuid.uuid4()).strip()
        user_message_id = str(uuid.uuid4())
        assistant_message_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc)
        user_created_at = now.strftime("%Y-%m-%d %H:%M:%S.%f")
        assistant_created_at = (now + timedelta(microseconds=1)).strftime("%Y-%m-%d %H:%M:%S.%f")
        with self._get_connection() as conn:
            try:
                cursor = conn.cursor()
                # 一轮 QA 是业务上的最小一致性单元；显式开启事务，保证 user/assistant/session 统计同进同退。
                cursor.execute('BEGIN IMMEDIATE')
                cursor.execute(
                    '''
                    SELECT session_id, title, arxiv_id FROM paper_chat_sessions
                    WHERE session_id = ? AND user_id = ?
                    ''',
                    (normalized_session_id, normalized_user_id),
                )
                session_row = cursor.fetchone()
                if not session_row:
                    raise PaperQATurnPersistenceError(
                        f"paper chat session not found: session_id={normalized_session_id} user_id={normalized_user_id}"
                    )

                cursor.execute(
                    '''
                    INSERT INTO paper_chat_messages (
                        message_id, turn_id, session_id, role, content, sources,
                        retrieval_debug_snapshot, contextualized_question, question_contextualization,
                        status, created_at
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ''',
                    (
                        user_message_id,
                        normalized_turn_id,
                        normalized_session_id,
                        'user',
                        str(question or ''),
                        '',
                        '',
                        str(contextualized_question or ''),
                        self._serialize_json_field(question_contextualization),
                        user_status,
                        user_created_at,
                    ),
                )

                cursor.execute(
                    '''
                    INSERT INTO paper_chat_messages (
                        message_id, turn_id, session_id, role, content, sources,
                        retrieval_debug_snapshot, contextualized_question, question_contextualization,
                        status, created_at
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ''',
                    (
                        assistant_message_id,
                        normalized_turn_id,
                        normalized_session_id,
                        'assistant',
                        str(answer or ''),
                        self._serialize_json_field(sources),
                        self._serialize_json_field(retrieval_debug_snapshot),
                        str(contextualized_question or ''),
                        self._serialize_json_field(question_contextualization),
                        assistant_status,
                        assistant_created_at,
                    ),
                )

                if not str(session_row[1] or '').strip() and question:
                    # 首轮问题可作为会话标题，但必须和消息写入在同一事务内更新，避免标题和消息状态脱节。
                    cursor.execute(
                        '''
                        UPDATE paper_chat_sessions
                        SET title = ?, updated_at = ?
                        WHERE session_id = ?
                        ''',
                        (str(question).strip()[:80], assistant_created_at, normalized_session_id),
                    )

                self._refresh_paper_chat_session_stats(conn, normalized_session_id)
                user_message = self._get_paper_chat_message_in_transaction(cursor, user_message_id)
                assistant_message = self._get_paper_chat_message_in_transaction(cursor, assistant_message_id)
                refreshed_session = self._get_paper_chat_session_in_transaction(cursor, normalized_session_id)
                if not user_message or not assistant_message or not refreshed_session:
                    raise PaperQATurnPersistenceError(
                        f"paper qa turn write verification failed: session_id={normalized_session_id} turn_id={normalized_turn_id}"
                    )
                if str(question or "").strip():
                    # QA 问题是弱兴趣信号，只追加事件并等待后续构建任务消费，避免问答写入被画像生成耗时拖慢。
                    self.record_user_profile_event(
                        user_id=normalized_user_id,
                        event_type="qa_asked",
                        source_type="paper_qa",
                        source_id=user_message_id,
                        action_type="qa_asked",
                        arxiv_id=str(session_row[2] or "").strip(),
                        session_id=normalized_session_id,
                        metadata={"turn_id": normalized_turn_id, "question": str(question or "").strip()},
                        include_in_profile=True,
                        conn=conn,
                    )
                conn.commit()
                return {
                    'turn_id': normalized_turn_id,
                    'user_message': user_message,
                    'assistant_message': assistant_message,
                    'chat_session': refreshed_session,
                    'refreshed_session': refreshed_session,
                }
            except Exception as exc:
                # rollback 必须在连接关闭前完成，确保任意半轮写入都不会逃出当前事务。
                conn.rollback()
                logger.exception(
                    "Error appending paper QA turn: session_id=%s user_id=%s turn_id=%s",
                    normalized_session_id,
                    normalized_user_id,
                    normalized_turn_id,
                )
                if isinstance(exc, PaperQATurnPersistenceError):
                    raise
                raise PaperQATurnPersistenceError(
                    f"paper qa turn persistence failed: session_id={normalized_session_id} "
                    f"user_id={normalized_user_id} turn_id={normalized_turn_id}: {exc}"
                ) from exc
