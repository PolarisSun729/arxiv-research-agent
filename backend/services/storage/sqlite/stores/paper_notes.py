import uuid
from typing import Any, Dict, List, Optional

from services.storage.sqlite.base import BaseSqliteStore
from services.storage.sqlite.shared import DEFAULT_USER_ID, PAPER_NOTE_TYPES, logger
from services.storage.sqlite.stores.profile_events import ProfileEventStore


# 笔记的标准读取列；_row_to_paper_note 按这里的顺序取下标。
_PAPER_NOTE_COLUMNS = (
    "note_id, user_id, arxiv_id, session_id, source_message_id, title, content, "
    "note_type, source_chunk_ids, tags, include_in_profile, created_at, updated_at"
)


def _normalize_note_type(note_type: Any) -> str:
    """笔记类型归一化：小写去空白，不在 PAPER_NOTE_TYPES 白名单中的值一律回落为 custom。"""
    normalized = str(note_type or "custom").strip().lower() or "custom"
    return normalized if normalized in PAPER_NOTE_TYPES else "custom"


def _clean_str_list(values: Optional[List[Any]]) -> List[str]:
    """把列表元素转成去空白字符串并丢弃空串，用于 tags 与 source_chunk_ids。"""
    return [str(item).strip() for item in (values or []) if str(item).strip()]


class PaperNoteStore(BaseSqliteStore):
    """paper_notes 存储；只有明确纳入画像的笔记才通过事件 store 进入长期证据流。"""

    def __init__(self, connection_provider, profile_event_store: ProfileEventStore) -> None:
        super().__init__(connection_provider)
        self.profile_event_store = profile_event_store

    def record_user_profile_event(self, *args, **kwargs):
        """转发到画像事件 store；保留该方法便于测试替换，也让调用点不必关心事件 store 的存在。"""
        return self.profile_event_store.record_user_profile_event(*args, **kwargs)

    def _row_to_paper_note(self, row: Any) -> Dict[str, Any]:
        """把 paper_notes 行映射为对外字典；JSON 列反序列化，空值回落为空串/空列表。"""
        return {
            'note_id': row[0],
            'user_id': row[1],
            'arxiv_id': row[2],
            'session_id': row[3],
            'source_message_id': row[4],
            'title': row[5] or '',
            'content': row[6] or '',
            'note_type': row[7] or 'custom',
            'source_chunk_ids': self._deserialize_json_field(row[8]) or [],
            'tags': self._deserialize_json_field(row[9]) or [],
            'include_in_profile': bool(row[10]),
            'created_at': row[11],
            'updated_at': row[12],
        }

    def create_paper_note(
        self,
        *,
        user_id: str = DEFAULT_USER_ID,
        arxiv_id: str,
        session_id: Optional[str] = None,
        source_message_id: Optional[str] = None,
        title: Optional[str] = None,
        content: str = "",
        note_type: str = "custom",
        source_chunk_ids: Optional[List[Any]] = None,
        tags: Optional[List[str]] = None,
        include_in_profile: bool = False,
        note_id: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """新建一条论文笔记并返回完整记录；include_in_profile=True 时在同一事务内写画像事件。"""
        try:
            normalized_note_type = _normalize_note_type(note_type)
            normalized_note_id = str(note_id or uuid.uuid4())
            normalized_title = str(title or "").strip()
            normalized_content = str(content or "").strip()
            normalized_tags = _clean_str_list(tags)
            normalized_chunk_ids = _clean_str_list(source_chunk_ids)

            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    '''
                    INSERT INTO paper_notes (
                        note_id, user_id, arxiv_id, session_id, source_message_id, title, content,
                        note_type, source_chunk_ids, tags, include_in_profile
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ''',
                    (
                        normalized_note_id,
                        user_id,
                        arxiv_id,
                        session_id,
                        source_message_id,
                        normalized_title,
                        normalized_content,
                        normalized_note_type,
                        self._serialize_json_field(normalized_chunk_ids),
                        self._serialize_json_field(normalized_tags),
                        1 if include_in_profile else 0,
                    ),
                )
                if include_in_profile:
                    # 只有用户明确允许进入画像的笔记才写入画像事件流，避免普通私有笔记污染长期偏好证据。
                    self.record_user_profile_event(
                        user_id=user_id,
                        event_type="note_saved",
                        source_type="paper_note",
                        source_id=normalized_note_id,
                        action_type="note_saved",
                        arxiv_id=arxiv_id,
                        note_id=normalized_note_id,
                        payload={
                            "title": title,
                            "note_type": normalized_note_type,
                            "tags": normalized_tags,
                            "source_chunk_ids": normalized_chunk_ids,
                        },
                        conn=conn,
                    )
                conn.commit()

            return self.get_paper_note(normalized_note_id, user_id=user_id)
        except Exception as e:
            logger.error(f"Error creating paper note: {str(e)}")
            return None

    def get_paper_note(self, note_id: str, user_id: str = DEFAULT_USER_ID) -> Optional[Dict[str, Any]]:
        """按 (note_id, user_id) 读取笔记；同时匹配 user_id，防止跨用户读取。"""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    f'''
                    SELECT {_PAPER_NOTE_COLUMNS}
                    FROM paper_notes
                    WHERE note_id = ? AND user_id = ?
                    ''',
                    (note_id, user_id),
                )
                row = cursor.fetchone()
                return self._row_to_paper_note(row) if row else None
        except Exception as e:
            logger.error(f"Error getting paper note: {str(e)}")
            return None

    def list_paper_notes(
        self,
        *,
        arxiv_id: str,
        user_id: str = DEFAULT_USER_ID,
        note_type: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """列出用户在某篇论文下的笔记（可按类型过滤），按更新时间倒序。"""
        filters = ["arxiv_id = ?", "user_id = ?"]
        values: List[Any] = [arxiv_id, user_id]
        if note_type:
            # 过滤时只做大小写/空白归一化，不回落 custom：查询未知类型应返回空列表而不是全部 custom 笔记。
            filters.append("note_type = ?")
            values.append(str(note_type or "").strip().lower())
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    f'''
                    SELECT {_PAPER_NOTE_COLUMNS}
                    FROM paper_notes
                    WHERE {' AND '.join(filters)}
                    ORDER BY updated_at DESC, created_at DESC
                    ''',
                    values,
                )
                return [self._row_to_paper_note(row) for row in cursor.fetchall()]
        except Exception as e:
            logger.error(f"Error listing paper notes: {str(e)}")
            return []

    def list_user_profile_notes(self, user_id: str = DEFAULT_USER_ID) -> List[Dict[str, Any]]:
        """列出用户明确允许进入研究画像的笔记，供画像生成器聚合长期证据。"""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    f'''
                    SELECT {_PAPER_NOTE_COLUMNS}
                    FROM paper_notes
                    WHERE user_id = ? AND include_in_profile = 1
                    ORDER BY updated_at DESC, created_at DESC
                    ''',
                    (user_id,),
                )
                return [self._row_to_paper_note(row) for row in cursor.fetchall()]
        except Exception as e:
            logger.error(f"Error listing user profile notes: {str(e)}")
            return []

    def update_paper_note(
        self,
        note_id: str,
        *,
        user_id: str = DEFAULT_USER_ID,
        title: Optional[str] = None,
        content: Optional[str] = None,
        note_type: Optional[str] = None,
        source_chunk_ids: Optional[List[Any]] = None,
        tags: Optional[List[str]] = None,
        include_in_profile: Optional[bool] = None,
    ) -> Optional[Dict[str, Any]]:
        """部分更新笔记：只写入非 None 的参数；笔记不存在或不属于该用户时返回 None。

        更新后若笔记处于“纳入画像”状态，会重新记录 note_saved 事件，让画像重建看到最新内容；
        若本次把笔记从“纳入画像”改为不纳入，则撤销它已有的 note_saved 事件。
        """
        try:
            update_fields: List[str] = []
            update_values: List[Any] = []

            if title is not None:
                update_fields.append("title = ?")
                update_values.append(str(title or "").strip())
            if content is not None:
                update_fields.append("content = ?")
                update_values.append(str(content or "").strip())
            if note_type is not None:
                update_fields.append("note_type = ?")
                update_values.append(_normalize_note_type(note_type))
            if source_chunk_ids is not None:
                update_fields.append("source_chunk_ids = ?")
                update_values.append(self._serialize_json_field(_clean_str_list(source_chunk_ids)))
            if tags is not None:
                update_fields.append("tags = ?")
                update_values.append(self._serialize_json_field(_clean_str_list(tags)))
            if include_in_profile is not None:
                update_fields.append("include_in_profile = ?")
                update_values.append(1 if include_in_profile else 0)

            if not update_fields:
                return self.get_paper_note(note_id, user_id=user_id)

            update_fields.append("updated_at = CURRENT_TIMESTAMP")
            update_values.extend([note_id, user_id])

            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT arxiv_id, include_in_profile FROM paper_notes WHERE note_id = ? AND user_id = ?",
                    (note_id, user_id),
                )
                previous = cursor.fetchone()
                cursor.execute(
                    f'''
                    UPDATE paper_notes
                    SET {", ".join(update_fields)}
                    WHERE note_id = ? AND user_id = ?
                    ''',
                    update_values,
                )
                if cursor.rowcount <= 0:
                    return None
                if include_in_profile is False and previous and previous[1]:
                    # 从“纳入画像”改为不纳入：与更新同一事务撤销已有事件，避免画像继续使用这条笔记。
                    self.profile_event_store._retract_note_profile_events(conn, user_id, note_id, previous[0])
                conn.commit()

            updated_note = self.get_paper_note(note_id, user_id=user_id)
            if updated_note and updated_note.get("include_in_profile"):
                # 已纳入画像的笔记更新后再次记录事件，让画像重建能追溯最新的用户编辑证据。
                self.record_user_profile_event(
                    user_id=user_id,
                    event_type="note_saved",
                    source_type="paper_note",
                    source_id=note_id,
                    action_type="note_saved",
                    arxiv_id=str(updated_note.get("arxiv_id") or "").strip(),
                    note_id=note_id,
                    payload={
                        "title": updated_note.get("title"),
                        "note_type": updated_note.get("note_type"),
                        "tags": updated_note.get("tags") or [],
                    },
                )
            return updated_note
        except Exception as e:
            logger.error(f"Error updating paper note: {str(e)}")
            return None

    def delete_paper_note(self, note_id: str, user_id: str = DEFAULT_USER_ID) -> bool:
        """删除属于该用户的笔记，返回是否真的删除了记录。

        删除已纳入画像的笔记时，在同一事务内撤销它的 note_saved 画像事件，避免被删笔记继续影响画像重建。
        未纳入画像的笔记没有生效中的事件，不写删除事件，以免无谓地触发画像重建。
        """
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT arxiv_id, include_in_profile FROM paper_notes WHERE note_id = ? AND user_id = ?",
                    (note_id, user_id),
                )
                row = cursor.fetchone()
                if not row:
                    return False
                cursor.execute(
                    '''
                    DELETE FROM paper_notes
                    WHERE note_id = ? AND user_id = ?
                    ''',
                    (note_id, user_id),
                )
                deleted = cursor.rowcount > 0
                if deleted and row[1]:
                    self.profile_event_store._retract_note_profile_events(conn, user_id, note_id, row[0])
                conn.commit()
                return deleted
        except Exception as e:
            logger.error(f"Error deleting paper note: {str(e)}")
            return False
