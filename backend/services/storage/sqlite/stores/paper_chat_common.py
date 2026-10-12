"""论文问答会话 / 消息相关 Store 共享的列定义、行映射和事务内辅助方法。

PaperChatSessionStore、PaperChatMessageStore、PaperQATurnStore 三者读写的是同两张表
（paper_chat_sessions / paper_chat_messages）。过去每个 Store 各自复制了一份 SELECT 列清单、
行到字典的映射以及“重算会话统计”的 SQL，任何一处新增列都需要同步改三份。
这里把它们收敛成单一来源，各 Store 只保留自己的业务写入流程。
"""

import sqlite3
from typing import TYPE_CHECKING, Any, Dict, Optional, Sequence

from services.storage.sqlite.base import BaseSqliteStore
from services.storage.sqlite.serialization import JsonFieldCodec

if TYPE_CHECKING:  # 仅用于类型提示，避免运行期与 Store 模块形成循环导入。
    from services.storage.sqlite.stores.paper_chat_sessions import PaperChatSessionStore
    from services.storage.sqlite.stores.profile_events import ProfileEventStore

# 会话表的标准读取列；顺序必须与 row_to_paper_chat_session 的下标一一对应。
PAPER_CHAT_SESSION_COLUMNS = (
    "session_id",
    "user_id",
    "arxiv_id",
    "title",
    "created_at",
    "updated_at",
    "message_count",
    "status",
    "summary_json",
    "summary_updated_at",
    "summary_turn_count",
    "summary_last_turn_id",
)

# 消息表的标准读取列；顺序必须与 row_to_paper_chat_message 的下标一一对应。
PAPER_CHAT_MESSAGE_COLUMNS = (
    "message_id",
    "turn_id",
    "session_id",
    "role",
    "content",
    "sources",
    "retrieval_debug_snapshot",
    "contextualized_question",
    "question_contextualization",
    "status",
    "created_at",
)


def select_columns(columns: Sequence[str], table_alias: str = "") -> str:
    """把列名元组拼成 SELECT 列表；JOIN 查询可传入表别名（如 "m"）避免列名歧义。"""
    prefix = f"{table_alias}." if table_alias else ""
    return ", ".join(f"{prefix}{column}" for column in columns)


def row_to_paper_chat_session(row: Any) -> Dict[str, Any]:
    """把 paper_chat_sessions 行转换为对外字典。

    摘要相关列是后加的；对只查询了前 8 列的旧 SQL 仍按长度兜底，保证返回结构稳定。
    """
    return {
        'session_id': row[0],
        'user_id': row[1],
        'arxiv_id': row[2],
        'title': row[3] or '',
        'created_at': row[4],
        'updated_at': row[5],
        'message_count': row[6] or 0,
        'status': row[7] or 'active',
        'summary': (JsonFieldCodec._deserialize_json_field(row[8]) or None) if len(row) > 8 else None,
        'summary_updated_at': row[9] if len(row) > 9 else None,
        'summary_turn_count': (row[10] or 0) if len(row) > 10 else 0,
        'summary_last_turn_id': (row[11] or '') if len(row) > 11 else '',
    }


def row_to_paper_chat_message(row: Any) -> Dict[str, Any]:
    """把 paper_chat_messages 行转换为对外字典，JSON 文本列在这里统一反序列化。"""
    return {
        'message_id': row[0],
        'turn_id': row[1] or '',
        'session_id': row[2],
        'role': row[3],
        'content': row[4] or '',
        'sources': JsonFieldCodec._deserialize_json_field(row[5]) or [],
        'retrieval_debug_snapshot': JsonFieldCodec._deserialize_json_field(row[6]),
        'contextualized_question': row[7] or '',
        'question_contextualization': JsonFieldCodec._deserialize_json_field(row[8]),
        'status': row[9] or 'completed',
        'created_at': row[10],
    }


class PaperChatStoreBase(BaseSqliteStore):
    """消息 Store 与 QA turn Store 的公共基类。

    两者都依赖会话 Store（读取会话结构）和画像事件 Store（记录 qa_asked 弱信号），
    并且都需要在同一事务里回读消息/会话、重算会话统计。
    这些方法保持实例方法形式，测试可以继续用 mock.patch.object 单独替换某一步来验证回滚。
    """

    def __init__(
        self,
        connection_provider,
        session_store: "PaperChatSessionStore",
        profile_event_store: "ProfileEventStore",
    ) -> None:
        super().__init__(connection_provider)
        self.session_store = session_store
        self.profile_event_store = profile_event_store

    def record_user_profile_event(self, *args, **kwargs):
        """消息写入只追加画像弱信号；事件去重和状态流转由画像事件 Store 负责。"""
        return self.profile_event_store.record_user_profile_event(*args, **kwargs)

    def _row_to_paper_chat_session(self, row: Any) -> Dict[str, Any]:
        return row_to_paper_chat_session(row)

    def _row_to_paper_chat_message(self, row: Any) -> Dict[str, Any]:
        return row_to_paper_chat_message(row)

    def _get_paper_chat_message_in_transaction(self, cursor: sqlite3.Cursor, message_id: str) -> Optional[Dict[str, Any]]:
        """在调用方的事务内回读消息；用于确认原子写入确实落库，不额外开连接。"""
        cursor.execute(
            f'''
            SELECT {select_columns(PAPER_CHAT_MESSAGE_COLUMNS)}
            FROM paper_chat_messages
            WHERE message_id = ?
            ''',
            (message_id,),
        )
        row = cursor.fetchone()
        return self._row_to_paper_chat_message(row) if row else None

    def _get_paper_chat_session_in_transaction(self, cursor: sqlite3.Cursor, session_id: str) -> Optional[Dict[str, Any]]:
        """在事务提交前回读会话，保证返回给上层的是本轮写入后的统计状态。"""
        cursor.execute(
            f'''
            SELECT {select_columns(PAPER_CHAT_SESSION_COLUMNS)}
            FROM paper_chat_sessions
            WHERE session_id = ?
            ''',
            (session_id,),
        )
        row = cursor.fetchone()
        return self._row_to_paper_chat_session(row) if row else None

    def _refresh_paper_chat_session_stats(self, conn: sqlite3.Connection, session_id: str) -> None:
        """根据消息表重算会话的 message_count 和 updated_at。

        计数始终以消息表为准而不是在会话表上 +1/-1，这样清空、删除、重试等路径
        都不会让会话统计和真实消息数量产生漂移。
        """
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
            # 有消息时把会话更新时间对齐到最后一条消息，列表排序才能反映真实的最近活跃时间。
            cursor.execute(
                '''
                UPDATE paper_chat_sessions
                SET message_count = ?, updated_at = ?, status = COALESCE(status, 'active')
                WHERE session_id = ?
                ''',
                (message_count, latest_created_at, session_id),
            )
        else:
            # 会话被清空后没有消息时间可用，只能退回当前时间。
            cursor.execute(
                '''
                UPDATE paper_chat_sessions
                SET message_count = 0, updated_at = CURRENT_TIMESTAMP, status = COALESCE(status, 'active')
                WHERE session_id = ?
                ''',
                (session_id,),
            )
