import uuid
from typing import Any, Dict, List, Optional

from services.storage.database.shared import DEFAULT_USER_ID, PAPER_NOTE_TYPES, logger


class PaperNoteMixin:
    """paper_notes 表的内部实现；对外仍由 DatabaseService 暴露兼容入口。"""

    def _row_to_paper_note(self, row: Any) -> Dict[str, Any]:
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
        try:
            normalized_note_type = str(note_type or "custom").strip().lower() or "custom"
            if normalized_note_type not in PAPER_NOTE_TYPES:
                normalized_note_type = "custom"

            normalized_note_id = str(note_id or uuid.uuid4())
            normalized_title = str(title or "").strip()
            normalized_content = str(content or "").strip()
            normalized_tags = [str(item).strip() for item in (tags or []) if str(item).strip()]
            normalized_chunk_ids = [str(item).strip() for item in (source_chunk_ids or []) if str(item).strip()]

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
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    '''
                    SELECT note_id, user_id, arxiv_id, session_id, source_message_id, title, content,
                           note_type, source_chunk_ids, tags, include_in_profile, created_at, updated_at
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
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                if note_type:
                    normalized_note_type = str(note_type or "").strip().lower()
                    cursor.execute(
                        '''
                        SELECT note_id, user_id, arxiv_id, session_id, source_message_id, title, content,
                               note_type, source_chunk_ids, tags, include_in_profile, created_at, updated_at
                        FROM paper_notes
                        WHERE arxiv_id = ? AND user_id = ? AND note_type = ?
                        ORDER BY updated_at DESC, created_at DESC
                        ''',
                        (arxiv_id, user_id, normalized_note_type),
                    )
                else:
                    cursor.execute(
                        '''
                        SELECT note_id, user_id, arxiv_id, session_id, source_message_id, title, content,
                               note_type, source_chunk_ids, tags, include_in_profile, created_at, updated_at
                        FROM paper_notes
                        WHERE arxiv_id = ? AND user_id = ?
                        ORDER BY updated_at DESC, created_at DESC
                        ''',
                        (arxiv_id, user_id),
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
                    '''
                    SELECT note_id, user_id, arxiv_id, session_id, source_message_id, title, content,
                           note_type, source_chunk_ids, tags, include_in_profile, created_at, updated_at
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
                normalized_note_type = str(note_type or "custom").strip().lower() or "custom"
                if normalized_note_type not in PAPER_NOTE_TYPES:
                    normalized_note_type = "custom"
                update_fields.append("note_type = ?")
                update_values.append(normalized_note_type)
            if source_chunk_ids is not None:
                normalized_chunk_ids = [str(item).strip() for item in source_chunk_ids if str(item).strip()]
                update_fields.append("source_chunk_ids = ?")
                update_values.append(self._serialize_json_field(normalized_chunk_ids))
            if tags is not None:
                normalized_tags = [str(item).strip() for item in tags if str(item).strip()]
                update_fields.append("tags = ?")
                update_values.append(self._serialize_json_field(normalized_tags))
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
                    f'''
                    UPDATE paper_notes
                    SET {", ".join(update_fields)}
                    WHERE note_id = ? AND user_id = ?
                    ''',
                    update_values,
                )
                conn.commit()
                if cursor.rowcount <= 0:
                    return None

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
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    '''
                    DELETE FROM paper_notes
                    WHERE note_id = ? AND user_id = ?
                    ''',
                    (note_id, user_id),
                )
                conn.commit()
                return cursor.rowcount > 0
        except Exception as e:
            logger.error(f"Error deleting paper note: {str(e)}")
            return False
