import uuid
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional

from services.storage.sqlite.shared import (
    DEFAULT_USER_ID,
    PaperQATurnPersistenceError,
    logger,
    normalize_user_id,
)
from services.storage.sqlite.stores.paper_chat_common import PaperChatStoreBase

# user/assistant 两条消息共用的 INSERT；created_at 由调用方显式给出，保证同一轮内顺序稳定。
_INSERT_TURN_MESSAGE_SQL = '''
    INSERT INTO paper_chat_messages (
        message_id, turn_id, session_id, role, content, sources,
        retrieval_debug_snapshot, contextualized_question, question_contextualization,
        status, created_at
    )
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
'''


class PaperQATurnStore(PaperChatStoreBase):
    """维护用户/助手双消息的原子 QA turn 写入，避免半轮对话污染短期记忆。

    行映射、事务内回读和会话统计重算继承自 PaperChatStoreBase，与消息 Store 共用同一份实现。
    """

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
        """在一个事务内写入“用户问题 + 助手回答”两条消息，并刷新会话统计。

        任一步失败都会整体回滚并抛出 PaperQATurnPersistenceError，
        调用方不会看到只有问题没有回答的半轮对话。
        """
        normalized_session_id = str(session_id or '').strip()
        normalized_user_id = normalize_user_id(user_id)
        normalized_turn_id = str(turn_id or uuid.uuid4()).strip()
        user_message_id = str(uuid.uuid4())
        assistant_message_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc)
        # 助手消息比用户消息晚 1 微秒，按 created_at 排序时问题一定排在回答前面。
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

                # 用户消息不携带检索来源和 debug 快照，这两列写空串。
                cursor.execute(
                    _INSERT_TURN_MESSAGE_SQL,
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
                    _INSERT_TURN_MESSAGE_SQL,
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
                # 提交前在同一事务里回读，确认两条消息和会话统计都已写入，否则视为失败回滚。
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
