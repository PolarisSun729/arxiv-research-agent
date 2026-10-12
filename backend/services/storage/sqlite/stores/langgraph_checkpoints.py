from typing import Any, Dict, List, Optional

from services.storage.sqlite.base import BaseSqliteStore
from services.storage.sqlite.shared import logger


# checkpoint 的标准读取列；_row_to_checkpoint 按这里的顺序取下标。
_CHECKPOINT_COLUMNS = "checkpoint_id, parent_checkpoint_id, checkpoint_json, metadata_json, created_at, updated_at"


class LangGraphCheckpointStore(BaseSqliteStore):
    """保存 LangGraph 原始 checkpoint blob，不承载业务确认语义。

    主键为 (thread_id, checkpoint_ns, checkpoint_id)；checkpoint_ns 为空串表示根图命名空间，
    因此所有入口都把 None 统一为 ''，保证读写落在同一主键上。
    """

    def _row_to_checkpoint(self, row: Any, *, thread_id: str, checkpoint_ns: str) -> Dict[str, Any]:
        """把 checkpoint 行映射为 LangGraph saver 需要的字典，并附带该 checkpoint 的 pending writes。"""
        return {
            'thread_id': thread_id,
            'checkpoint_ns': checkpoint_ns,
            'checkpoint_id': row[0],
            'parent_checkpoint_id': row[1],
            'checkpoint': self._deserialize_json_field(row[2]) or {},
            'metadata': self._deserialize_json_field(row[3]) or {},
            'pending_writes': self.get_langgraph_checkpoint_writes(
                thread_id=thread_id,
                checkpoint_ns=checkpoint_ns,
                checkpoint_id=row[0],
            ),
            'created_at': row[4],
            'updated_at': row[5],
        }

    def put_langgraph_checkpoint(
        self,
        *,
        thread_id: str,
        checkpoint_ns: str,
        checkpoint_id: str,
        checkpoint: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None,
        parent_checkpoint_id: Optional[str] = None,
    ) -> bool:
        """保存 LangGraph 原始 checkpoint，只负责持久化恢复所需快照。"""
        try:
            normalized_thread_id = str(thread_id or '').strip()
            normalized_checkpoint_id = str(checkpoint_id or '').strip()
            if not normalized_thread_id or not normalized_checkpoint_id:
                return False
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    '''
                    INSERT INTO langgraph_checkpoints (
                        thread_id, checkpoint_ns, checkpoint_id, parent_checkpoint_id,
                        checkpoint_json, metadata_json
                    )
                    VALUES (?, ?, ?, ?, ?, ?)
                    ON CONFLICT(thread_id, checkpoint_ns, checkpoint_id) DO UPDATE SET
                        parent_checkpoint_id = excluded.parent_checkpoint_id,
                        checkpoint_json = excluded.checkpoint_json,
                        metadata_json = excluded.metadata_json,
                        updated_at = CURRENT_TIMESTAMP
                    ''',
                    (
                        normalized_thread_id,
                        str(checkpoint_ns or ''),
                        normalized_checkpoint_id,
                        parent_checkpoint_id,
                        self._serialize_json_field(checkpoint),
                        self._serialize_json_field(metadata),
                    ),
                )
                conn.commit()
                return True
        except Exception as e:
            logger.error(f"Error putting langgraph checkpoint: {str(e)}")
            return False

    def get_langgraph_checkpoint(
        self,
        *,
        thread_id: str,
        checkpoint_ns: str = '',
        checkpoint_id: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """读取指定 checkpoint；未指定 checkpoint_id 时返回该线程命名空间下最新的一条。"""
        try:
            normalized_thread_id = str(thread_id or '').strip()
            if not normalized_thread_id:
                return None
            normalized_ns = str(checkpoint_ns or '')
            with self._get_connection() as conn:
                cursor = conn.cursor()
                if checkpoint_id:
                    cursor.execute(
                        f'''
                        SELECT {_CHECKPOINT_COLUMNS}
                        FROM langgraph_checkpoints
                        WHERE thread_id = ? AND checkpoint_ns = ? AND checkpoint_id = ?
                        ''',
                        (normalized_thread_id, normalized_ns, str(checkpoint_id)),
                    )
                else:
                    cursor.execute(
                        f'''
                        SELECT {_CHECKPOINT_COLUMNS}
                        FROM langgraph_checkpoints
                        WHERE thread_id = ? AND checkpoint_ns = ?
                        ORDER BY updated_at DESC, created_at DESC
                        LIMIT 1
                        ''',
                        (normalized_thread_id, normalized_ns),
                    )
                row = cursor.fetchone()
                if not row:
                    return None
                return self._row_to_checkpoint(row, thread_id=normalized_thread_id, checkpoint_ns=normalized_ns)
        except Exception as e:
            logger.error(f"Error getting langgraph checkpoint: {str(e)}")
            return None

    def list_langgraph_checkpoints(
        self,
        *,
        thread_id: str,
        checkpoint_ns: str = '',
        limit: int = 10,
    ) -> List[Dict[str, Any]]:
        """按更新时间倒序列出线程命名空间下的 checkpoint（含各自的 pending writes）。"""
        try:
            normalized_thread_id = str(thread_id or '').strip()
            if not normalized_thread_id:
                return []
            normalized_ns = str(checkpoint_ns or '')
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    f'''
                    SELECT {_CHECKPOINT_COLUMNS}
                    FROM langgraph_checkpoints
                    WHERE thread_id = ? AND checkpoint_ns = ?
                    ORDER BY updated_at DESC, created_at DESC
                    LIMIT ?
                    ''',
                    (normalized_thread_id, normalized_ns, max(int(limit or 1), 1)),
                )
                rows = cursor.fetchall()
                return [
                    self._row_to_checkpoint(row, thread_id=normalized_thread_id, checkpoint_ns=normalized_ns)
                    for row in rows
                ]
        except Exception as e:
            logger.error(f"Error listing langgraph checkpoints: {str(e)}")
            return []

    def get_langgraph_checkpoint_writes(
        self,
        *,
        thread_id: str,
        checkpoint_ns: str,
        checkpoint_id: str,
    ) -> List[Dict[str, Any]]:
        """读取某个 checkpoint 的 pending writes，按 (task_id, idx) 排序以还原写入顺序。"""
        try:
            normalized_thread_id = str(thread_id or '').strip()
            normalized_checkpoint_id = str(checkpoint_id or '').strip()
            if not normalized_thread_id or not normalized_checkpoint_id:
                return []
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    '''
                    SELECT task_id, idx, channel, value_json
                    FROM langgraph_checkpoint_writes
                    WHERE thread_id = ? AND checkpoint_ns = ? AND checkpoint_id = ?
                    ORDER BY task_id ASC, idx ASC
                    ''',
                    (normalized_thread_id, str(checkpoint_ns or ''), normalized_checkpoint_id),
                )
                return [
                    {
                        'task_id': row[0],
                        'idx': row[1],
                        'channel': row[2],
                        'value': self._deserialize_json_field(row[3]),
                    }
                    for row in cursor.fetchall()
                ]
        except Exception as e:
            logger.error(f"Error getting langgraph checkpoint writes: {str(e)}")
            return []

    def put_langgraph_checkpoint_writes(
        self,
        *,
        thread_id: str,
        checkpoint_ns: str,
        checkpoint_id: str,
        task_id: str,
        writes: List[Dict[str, Any]],
    ) -> bool:
        """批量 upsert 某个任务的 pending writes；idx 取 writes 中的位置，重放同一任务时覆盖旧值。"""
        try:
            normalized_thread_id = str(thread_id or '').strip()
            normalized_checkpoint_id = str(checkpoint_id or '').strip()
            normalized_task_id = str(task_id or '').strip()
            if not normalized_thread_id or not normalized_checkpoint_id or not normalized_task_id:
                return False
            with self._get_connection() as conn:
                cursor = conn.cursor()
                for index, write in enumerate(list(writes or [])):
                    channel = str((write or {}).get('channel') or '').strip()
                    cursor.execute(
                        '''
                        INSERT INTO langgraph_checkpoint_writes (
                            thread_id, checkpoint_ns, checkpoint_id, task_id, idx, channel, value_json
                        )
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                        ON CONFLICT(thread_id, checkpoint_ns, checkpoint_id, task_id, idx) DO UPDATE SET
                            channel = excluded.channel,
                            value_json = excluded.value_json,
                            updated_at = CURRENT_TIMESTAMP
                        ''',
                        (
                            normalized_thread_id,
                            str(checkpoint_ns or ''),
                            normalized_checkpoint_id,
                            normalized_task_id,
                            index,
                            channel,
                            self._serialize_json_field((write or {}).get('value')),
                        ),
                    )
                conn.commit()
                return True
        except Exception as e:
            logger.error(f"Error putting langgraph checkpoint writes: {str(e)}")
            return False
