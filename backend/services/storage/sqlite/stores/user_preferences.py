from typing import Any, Dict, List, Optional

from services.storage.sqlite.base import BaseSqliteStore
from services.storage.sqlite.shared import DEFAULT_USER_ID, logger
from services.storage.sqlite.stores.profile_events import ProfileEventStore
from services.storage.sqlite.stores.research_profiles import ResearchProfileStore
from services.user_behavior_policy import (
    WEAK_PAPER_ACTION_TYPES,
    is_explicit_preference_action,
    normalize_paper_action_type,
)

PAPER_ACTION_TYPES = set(WEAK_PAPER_ACTION_TYPES)


class UserPreferenceStore(BaseSqliteStore):
    """用户论文反馈状态存储；画像事件写入通过显式依赖完成，避免重新形成万能存储对象。"""

    def __init__(
        self,
        connection_provider,
        profile_event_store: ProfileEventStore,
        research_profile_store: ResearchProfileStore,
    ) -> None:
        super().__init__(connection_provider)
        self.profile_event_store = profile_event_store
        self.research_profile_store = research_profile_store

    def _record_preference_profile_event(self, conn, user_id: str, arxiv_id: str, event_type: str) -> None:
        self.profile_event_store._record_preference_profile_event(conn, user_id, arxiv_id, event_type)

    def _deactivate_profile_events_for_paper(self, conn, user_id: str, arxiv_id: str, event_types: List[str]) -> None:
        self.profile_event_store._deactivate_profile_events_for_paper(conn, user_id, arxiv_id, event_types)

    def _record_profile_signal_removed_event(self, conn, user_id: str, arxiv_id: str, removed_event_type: str) -> None:
        self.profile_event_store._record_profile_signal_removed_event(conn, user_id, arxiv_id, removed_event_type)

    def record_user_profile_event(self, *args, **kwargs):
        return self.profile_event_store.record_user_profile_event(*args, **kwargs)

    def _normalize_profile_event_type(self, value: Any) -> str:
        return self.profile_event_store._normalize_profile_event_type(value)

    def get_user_research_profile(self, user_id: str = DEFAULT_USER_ID) -> Dict[str, Any]:
        return self.research_profile_store.get_user_research_profile(user_id)

    @staticmethod
    def _normalize_action_type(action_type: Any) -> str:
        return normalize_paper_action_type(action_type)

    def add_liked_paper(self, user_id: str = DEFAULT_USER_ID, arxiv_id: str = None) -> bool:
        try:
            if arxiv_id is None:
                logger.error("arxiv_id is required")
                return False

            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    DELETE FROM user_disliked_papers WHERE user_id = ? AND arxiv_id = ?
                ''', (user_id, arxiv_id))

                cursor.execute('''
                    INSERT OR IGNORE INTO user_liked_papers (user_id, arxiv_id)
                    VALUES (?, ?)
                ''', (user_id, arxiv_id))

                # 强偏好表是 liked 状态的唯一权威来源；这里只清理历史 action 残留并写画像事件。
                cursor.execute(
                    '''
                    DELETE FROM user_paper_actions
                    WHERE user_id = ? AND arxiv_id = ? AND action_type IN ('like', 'dislike', 'not_interested')
                    ''',
                    (user_id, arxiv_id),
                )
                self._deactivate_profile_events_for_paper(conn, user_id, arxiv_id, ["disliked", "not_interested"])
                self._record_preference_profile_event(conn, user_id, arxiv_id, "liked")
                conn.commit()
                logger.info(f"Paper {arxiv_id} added to liked list for user: {user_id}")
                return True
        except Exception as e:
            logger.error(f"Error adding liked paper: {str(e)}")
            return False

    def remove_liked_paper(self, user_id: str = DEFAULT_USER_ID, arxiv_id: str = None) -> bool:
        try:
            if arxiv_id is None:
                logger.error("arxiv_id is required")
                return False

            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    DELETE FROM user_liked_papers WHERE user_id = ? AND arxiv_id = ?
                ''', (user_id, arxiv_id))
                removed = cursor.rowcount > 0

                if removed:
                    cursor.execute(
                        '''
                        DELETE FROM user_paper_actions WHERE user_id = ? AND arxiv_id = ? AND action_type = 'like'
                        ''',
                        (user_id, arxiv_id),
                    )
                    self._deactivate_profile_events_for_paper(conn, user_id, arxiv_id, ["liked"])
                    self._record_profile_signal_removed_event(conn, user_id, arxiv_id, "liked")
                conn.commit()
                logger.info(f"Paper {arxiv_id} removed from liked list for user: {user_id}")
                return removed
        except Exception as e:
            logger.error(f"Error removing liked paper: {str(e)}")
            return False

    def get_liked_papers(self, user_id: str = DEFAULT_USER_ID) -> List[str]:
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT arxiv_id FROM user_liked_papers WHERE user_id = ? ORDER BY created_at DESC
                ''', (user_id,))

                return [row[0] for row in cursor.fetchall()]
        except Exception as e:
            logger.error(f"Error getting liked papers: {str(e)}")
            return []

    def get_liked_papers_with_details(self, user_id: str = DEFAULT_USER_ID) -> List[Dict[str, Any]]:
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT p.arxiv_id, p.title, p.abstract, p.authors, p.categories, p.published_date
                    FROM user_liked_papers ulp
                    JOIN arxiv_papers p ON ulp.arxiv_id = p.arxiv_id
                    WHERE ulp.user_id = ?
                    ORDER BY ulp.created_at DESC
                ''', (user_id,))

                results = []
                for row in cursor.fetchall():
                    results.append({
                        'arxiv_id': row[0],
                        'title': row[1],
                        'abstract': row[2],
                        'authors': self._deserialize_paper_db_value(row[3]),
                        'categories': self._deserialize_paper_db_value(row[4]),
                        'published_date': row[5]
                    })
                return results
        except Exception as e:
            logger.error(f"Error getting liked papers with details: {str(e)}")
            return []

    def is_liked_paper(self, user_id: str = DEFAULT_USER_ID, arxiv_id: str = None) -> bool:
        try:
            if arxiv_id is None:
                return False

            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT COUNT(*) FROM user_liked_papers WHERE user_id = ? AND arxiv_id = ?
                ''', (user_id, arxiv_id))

                return cursor.fetchone()[0] > 0
        except Exception as e:
            logger.error(f"Error checking liked paper: {str(e)}")
            return False

    def add_disliked_paper(self, user_id: str = DEFAULT_USER_ID, arxiv_id: str = None) -> bool:
        try:
            if arxiv_id is None:
                logger.error("arxiv_id is required")
                return False

            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    DELETE FROM user_liked_papers WHERE user_id = ? AND arxiv_id = ?
                ''', (user_id, arxiv_id))

                cursor.execute('''
                    INSERT OR IGNORE INTO user_disliked_papers (user_id, arxiv_id)
                    VALUES (?, ?)
                ''', (user_id, arxiv_id))

                # disliked 同样只落在强偏好表；历史 action 残留要清掉，避免推荐读取到双套负反馈。
                cursor.execute(
                    '''
                    DELETE FROM user_paper_actions
                    WHERE user_id = ? AND arxiv_id = ? AND action_type IN ('like', 'dislike', 'not_interested')
                    ''',
                    (user_id, arxiv_id),
                )
                self._deactivate_profile_events_for_paper(conn, user_id, arxiv_id, ["liked", "not_interested"])
                self._record_preference_profile_event(conn, user_id, arxiv_id, "disliked")
                conn.commit()
                logger.info(f"Paper {arxiv_id} added to disliked list for user: {user_id}")
                return True
        except Exception as e:
            logger.error(f"Error adding disliked paper: {str(e)}")
            return False

    def remove_disliked_paper(self, user_id: str = DEFAULT_USER_ID, arxiv_id: str = None) -> bool:
        try:
            if arxiv_id is None:
                logger.error("arxiv_id is required")
                return False

            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    DELETE FROM user_disliked_papers WHERE user_id = ? AND arxiv_id = ?
                ''', (user_id, arxiv_id))
                removed = cursor.rowcount > 0

                if removed:
                    cursor.execute(
                        '''
                        DELETE FROM user_paper_actions WHERE user_id = ? AND arxiv_id = ? AND action_type = 'dislike'
                        ''',
                        (user_id, arxiv_id),
                    )
                    self._deactivate_profile_events_for_paper(conn, user_id, arxiv_id, ["disliked"])
                    self._record_profile_signal_removed_event(conn, user_id, arxiv_id, "disliked")
                conn.commit()
                logger.info(f"Paper {arxiv_id} removed from disliked list for user: {user_id}")
                return removed
        except Exception as e:
            logger.error(f"Error removing disliked paper: {str(e)}")
            return False

    def get_disliked_papers(self, user_id: str = DEFAULT_USER_ID) -> List[str]:
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT arxiv_id FROM user_disliked_papers WHERE user_id = ? ORDER BY created_at DESC
                ''', (user_id,))

                return [row[0] for row in cursor.fetchall()]
        except Exception as e:
            logger.error(f"Error getting disliked papers: {str(e)}")
            return []

    def is_disliked_paper(self, user_id: str = DEFAULT_USER_ID, arxiv_id: str = None) -> bool:
        try:
            if arxiv_id is None:
                return False

            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT COUNT(*) FROM user_disliked_papers WHERE user_id = ? AND arxiv_id = ?
                ''', (user_id, arxiv_id))

                return cursor.fetchone()[0] > 0
        except Exception as e:
            logger.error(f"Error checking disliked paper: {str(e)}")
            return False

    def get_user_preferences(self, user_id: str = DEFAULT_USER_ID) -> Dict[str, Any]:
        try:
            paper_actions = self.get_user_paper_action_map(user_id)
            return {
                'user_id': user_id,
                'liked_papers': self.get_liked_papers(user_id),
                'disliked_papers': self.get_disliked_papers(user_id),
                'paper_actions': paper_actions,
                'research_profile': self.get_user_research_profile(user_id),
            }
        except Exception as e:
            logger.error(f"Error getting user preferences: {str(e)}")
            return {
                'user_id': user_id,
                'liked_papers': [],
                'disliked_papers': [],
                'paper_actions': {},
                'research_profile': self._empty_user_research_profile(user_id),
            }

    def record_user_paper_action(
        self,
        user_id: str = DEFAULT_USER_ID,
        arxiv_id: str = None,
        action_type: str = "",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        try:
            if arxiv_id is None:
                logger.error("arxiv_id is required")
                return False

            normalized_action = self._normalize_action_type(action_type)
            if is_explicit_preference_action(action_type):
                logger.error("Explicit preference action must use like/dislike endpoint: %s", action_type)
                return False
            if normalized_action not in PAPER_ACTION_TYPES:
                logger.error("Unsupported paper action type: %s", action_type)
                return False

            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    '''
                    INSERT INTO user_paper_actions (user_id, arxiv_id, action_type, metadata_json)
                    VALUES (?, ?, ?, ?)
                    ON CONFLICT(user_id, arxiv_id, action_type)
                    DO UPDATE SET metadata_json = excluded.metadata_json, updated_at = CURRENT_TIMESTAMP
                    ''',
                    (user_id, arxiv_id, normalized_action, self._serialize_json_field(metadata)),
                )
                # 行为事件层保留画像相关的原始用户动作，后续重建可以解释画像项来自哪些显式行为。
                self.record_user_profile_event(
                    user_id=user_id,
                    event_type=self._normalize_profile_event_type(normalized_action),
                    source_type="paper_action",
                    source_id=arxiv_id,
                    action_type=normalized_action,
                    arxiv_id=arxiv_id,
                    metadata={"metadata": metadata or {}, "source": "paper_action"},
                    conn=conn,
                )
                conn.commit()
                return True
        except Exception as e:
            logger.error(f"Error recording paper action: {str(e)}")
            return False

    def remove_user_paper_action(
        self,
        user_id: str = DEFAULT_USER_ID,
        arxiv_id: str = None,
        action_type: str = "",
    ) -> bool:
        try:
            if arxiv_id is None:
                logger.error("arxiv_id is required")
                return False

            normalized_action = self._normalize_action_type(action_type)
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    '''
                    DELETE FROM user_paper_actions WHERE user_id = ? AND arxiv_id = ? AND action_type = ?
                    ''',
                    (user_id, arxiv_id, normalized_action),
                )
                removed = cursor.rowcount > 0
                if removed:
                    # 删除弱行为时同步停用画像事件，避免 action 表已清理但画像重建仍读取旧证据。
                    self._deactivate_profile_events_for_paper(conn, user_id, arxiv_id, [normalized_action])
                    self._record_profile_signal_removed_event(conn, user_id, arxiv_id, normalized_action)
                conn.commit()
                return removed
        except Exception as e:
            logger.error(f"Error removing paper action: {str(e)}")
            return False

    def get_user_paper_actions(
        self,
        user_id: str = DEFAULT_USER_ID,
        action_type: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                if action_type:
                    normalized_action = self._normalize_action_type(action_type)
                    cursor.execute(
                        '''
                        SELECT user_id, arxiv_id, action_type, metadata_json, created_at, updated_at
                        FROM user_paper_actions
                        WHERE user_id = ? AND action_type = ?
                        ORDER BY updated_at DESC, created_at DESC
                        ''',
                        (user_id, normalized_action),
                    )
                else:
                    cursor.execute(
                        '''
                        SELECT user_id, arxiv_id, action_type, metadata_json, created_at, updated_at
                        FROM user_paper_actions
                        WHERE user_id = ?
                        ORDER BY updated_at DESC, created_at DESC
                        ''',
                        (user_id,),
                    )

                return [
                    {
                        "user_id": row[0],
                        "arxiv_id": row[1],
                        "action_type": row[2],
                        "metadata": self._deserialize_json_field(row[3]) or {},
                        "created_at": row[4],
                        "updated_at": row[5],
                    }
                    for row in cursor.fetchall()
                ]
        except Exception as e:
            logger.error(f"Error getting user paper actions: {str(e)}")
            return []

    def get_user_paper_action_map(self, user_id: str = DEFAULT_USER_ID) -> Dict[str, List[str]]:
        action_map: Dict[str, List[str]] = {}
        for item in self.get_user_paper_actions(user_id=user_id):
            action_key = str(item.get("action_type") or "").strip()
            arxiv_id = str(item.get("arxiv_id") or "").strip()
            if not action_key or not arxiv_id:
                continue
            action_map.setdefault(action_key, []).append(arxiv_id)
        return action_map

    def get_user_paper_action_state(self, user_id: str = DEFAULT_USER_ID, arxiv_id: str = None) -> Dict[str, Any]:
        state = {action_type: False for action_type in PAPER_ACTION_TYPES}
        state["metadata"] = {}
        if not arxiv_id:
            return state

        for item in self.get_user_paper_actions(user_id=user_id):
            if str(item.get("arxiv_id") or "").strip() != str(arxiv_id or "").strip():
                continue
            action_key = str(item.get("action_type") or "").strip()
            if action_key:
                state[action_key] = True
                if item.get("metadata"):
                    state["metadata"][action_key] = item.get("metadata")
        return state

    def get_user_labeled_paper_count(self, user_id: str = DEFAULT_USER_ID) -> int:
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    '''
                    SELECT COUNT(DISTINCT arxiv_id) FROM (
                        SELECT arxiv_id FROM user_liked_papers WHERE user_id = ?
                        UNION
                        SELECT arxiv_id FROM user_disliked_papers WHERE user_id = ?
                    )
                    ''',
                    (user_id, user_id),
                )
                row = cursor.fetchone()
                return int(row[0] or 0) if row else 0
        except Exception as e:
            logger.error(f"Error getting user labeled paper count: {str(e)}")
            return 0

    def get_unlabeled_papers(self, user_id: str = DEFAULT_USER_ID) -> List[Dict[str, Any]]:
        """按用户强偏好排除已标注论文，供画像/推荐流程选择下一批候选。"""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    '''
                    SELECT p.arxiv_id, p.title, p.abstract, p.authors, p.categories, p.published_date, p.url
                    FROM arxiv_papers p
                    LEFT JOIN user_liked_papers ulp ON p.arxiv_id = ulp.arxiv_id AND ulp.user_id = ?
                    LEFT JOIN user_disliked_papers udp ON p.arxiv_id = udp.arxiv_id AND udp.user_id = ?
                    WHERE ulp.arxiv_id IS NULL AND udp.arxiv_id IS NULL AND p.embedding_id IS NOT NULL
                    ORDER BY p.published_date DESC
                    ''',
                    (user_id, user_id),
                )
                return [
                    {
                        "arxiv_id": row[0],
                        "title": row[1],
                        "abstract": row[2],
                        "authors": row[3],
                        "categories": row[4],
                        "published_date": row[5],
                        "url": row[6],
                    }
                    for row in cursor.fetchall()
                ]
        except Exception as e:
            logger.error(f"Error getting unlabeled papers: {str(e)}")
            return []
