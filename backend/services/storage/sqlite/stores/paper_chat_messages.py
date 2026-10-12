import uuid
from typing import Any, Dict, List, Optional

from services.storage.sqlite.shared import DEFAULT_USER_ID, logger
from services.storage.sqlite.stores.paper_chat_common import (
    PAPER_CHAT_MESSAGE_COLUMNS,
    PaperChatStoreBase,
    select_columns,
)

# 带 m. 前缀的消息列，用于和 paper_chat_sessions JOIN 校验会话归属的查询。
_MESSAGE_COLUMNS_M = select_columns(PAPER_CHAT_MESSAGE_COLUMNS, "m")


class PaperChatMessageStore(PaperChatStoreBase):
    """维护论文问答消息生命周期；双消息 QA turn 的原子写入由 PaperQATurnStore 单独处理。

    所有读取都 JOIN paper_chat_sessions 并带上 user_id 条件，
    保证调用方即使拿到别人的 session_id/message_id 也读不到不属于自己的消息。
    """

    def list_paper_chat_messages(self, session_id: str, user_id: str = DEFAULT_USER_ID) -> List[Dict[str, Any]]:
        """按时间正序返回会话的全部消息，供前端展示完整历史。"""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    f'''
                    SELECT {_MESSAGE_COLUMNS_M}
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
                # 内层按时间倒序取最近 N 条，外层再按正序排回去；rowid 作为同一时间戳下的稳定次序。
                cursor.execute(
                    f'''
                    SELECT {select_columns(PAPER_CHAT_MESSAGE_COLUMNS)}
                    FROM (
                        SELECT m.rowid AS message_rowid,
                               {_MESSAGE_COLUMNS_M}
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
        """按 message_id 读取单条消息；消息所属会话不属于该用户时返回 None。"""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    f'''
                    SELECT {_MESSAGE_COLUMNS_M}
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
        """读取某一轮对话中指定角色的最新一条消息（默认取助手回答），用于重试/续写场景定位。"""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    f'''
                    SELECT {_MESSAGE_COLUMNS_M}
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
        """追加单条消息并刷新会话统计；会话不存在或不属于该用户时返回 None。

        这是单消息写入路径（流式输出、旧接口等）；一问一答必须同时成功的场景
        请使用 PaperQATurnStore.append_paper_qa_turn。
        """
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
                    # 会话还没有标题时，用第一条用户问题的前 80 个字符作为标题。
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
        """清空会话内的全部消息但保留会话本身；子查询保证只能清空自己名下的会话。"""
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
        """删除整个会话（消息 + 会话记录），统一委托给会话 Store。

        这里曾经有一份独立实现：先无条件按 session_id 删除消息、再按 user_id 删除会话，
        既会误删他人会话下的消息，也没有把 paper_notes.session_id 置空。
        会话 Store 的实现先校验归属、再在同一事务内完成三步清理，是唯一可信版本。
        """
        return self.session_store.delete_paper_chat_session(session_id, user_id=user_id)
