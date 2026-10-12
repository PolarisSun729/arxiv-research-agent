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

# 强偏好（喜欢/不喜欢）的表结构配置。liked 与 disliked 互斥：写入一侧时必须清掉另一侧，
# 并停用与之冲突的画像事件。表名只来自这里的固定常量，拼进 SQL 不存在注入风险。
_STRONG_PREFERENCES: Dict[str, Dict[str, Any]] = {
    "liked": {
        "table": "user_liked_papers",
        "opposite_table": "user_disliked_papers",
        "action_type": "like",
        "conflicting_events": ["disliked", "not_interested"],
    },
    "disliked": {
        "table": "user_disliked_papers",
        "opposite_table": "user_liked_papers",
        "action_type": "dislike",
        "conflicting_events": ["liked", "not_interested"],
    },
}

# 弱行为表 user_paper_actions 的标准读取列，顺序与 get_user_paper_actions 的映射一致。
_PAPER_ACTION_COLUMNS = "user_id, arxiv_id, action_type, metadata_json, created_at, updated_at"


class UserPreferenceStore(BaseSqliteStore):
    """用户论文反馈状态存储；画像事件写入通过显式依赖完成，避免重新形成万能存储对象。

    - 强偏好（liked/disliked）落在独立的两张表里，是推荐和画像读取的唯一权威来源；
    - 弱行为（收藏、阅读、不感兴趣等）落在 user_paper_actions；
    - 每次状态变化都会同步写入/停用画像事件，保证画像重建能解释每一条信号的来源。
    """

    def __init__(
        self,
        connection_provider,
        profile_event_store: ProfileEventStore,
        research_profile_store: ResearchProfileStore,
    ) -> None:
        super().__init__(connection_provider)
        self.profile_event_store = profile_event_store
        self.research_profile_store = research_profile_store

    # ---- 以下转发方法把画像相关读写委托给显式注入的 Store，本类只关心偏好表本身 ----

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

    @staticmethod
    def _require_arxiv_id(arxiv_id: Optional[str]) -> bool:
        """所有写接口的公共前置校验；缺 arxiv_id 时记录错误并让调用方直接返回 False。"""
        if arxiv_id is None:
            logger.error("arxiv_id is required")
            return False
        return True

    def _add_strong_preference(self, user_id: str, arxiv_id: Optional[str], preference: str) -> bool:
        """写入一条强偏好（liked/disliked），并在同一事务里完成互斥清理和画像事件记录。"""
        config = _STRONG_PREFERENCES[preference]
        try:
            if not self._require_arxiv_id(arxiv_id):
                return False

            with self._get_connection() as conn:
                cursor = conn.cursor()
                # 喜欢与不喜欢互斥：先删掉另一侧，再幂等插入当前侧。
                cursor.execute(
                    f"DELETE FROM {config['opposite_table']} WHERE user_id = ? AND arxiv_id = ?",
                    (user_id, arxiv_id),
                )
                cursor.execute(
                    f"INSERT OR IGNORE INTO {config['table']} (user_id, arxiv_id) VALUES (?, ?)",
                    (user_id, arxiv_id),
                )

                # 强偏好表是 liked/disliked 状态的唯一权威来源；历史 action 残留要清掉，
                # 避免推荐侧同时读到两套正/负反馈。
                cursor.execute(
                    '''
                    DELETE FROM user_paper_actions
                    WHERE user_id = ? AND arxiv_id = ? AND action_type IN ('like', 'dislike', 'not_interested')
                    ''',
                    (user_id, arxiv_id),
                )
                self._deactivate_profile_events_for_paper(conn, user_id, arxiv_id, list(config["conflicting_events"]))
                self._record_preference_profile_event(conn, user_id, arxiv_id, preference)
                conn.commit()
                logger.info(f"Paper {arxiv_id} added to {preference} list for user: {user_id}")
                return True
        except Exception as e:
            logger.error(f"Error adding {preference} paper: {str(e)}")
            return False

    def _remove_strong_preference(self, user_id: str, arxiv_id: Optional[str], preference: str) -> bool:
        """撤销一条强偏好；只有确实删除了记录时才停用画像事件并写入“信号被移除”事件。"""
        config = _STRONG_PREFERENCES[preference]
        try:
            if not self._require_arxiv_id(arxiv_id):
                return False

            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    f"DELETE FROM {config['table']} WHERE user_id = ? AND arxiv_id = ?",
                    (user_id, arxiv_id),
                )
                removed = cursor.rowcount > 0

                if removed:
                    cursor.execute(
                        "DELETE FROM user_paper_actions WHERE user_id = ? AND arxiv_id = ? AND action_type = ?",
                        (user_id, arxiv_id, config["action_type"]),
                    )
                    self._deactivate_profile_events_for_paper(conn, user_id, arxiv_id, [preference])
                    self._record_profile_signal_removed_event(conn, user_id, arxiv_id, preference)
                conn.commit()
                logger.info(f"Paper {arxiv_id} removed from {preference} list for user: {user_id}")
                return removed
        except Exception as e:
            logger.error(f"Error removing {preference} paper: {str(e)}")
            return False

    def _list_strong_preference_ids(self, user_id: str, preference: str) -> List[str]:
        """按标记时间倒序返回某类强偏好的 arxiv_id 列表。"""
        table = _STRONG_PREFERENCES[preference]["table"]
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    f"SELECT arxiv_id FROM {table} WHERE user_id = ? ORDER BY created_at DESC",
                    (user_id,),
                )
                return [row[0] for row in cursor.fetchall()]
        except Exception as e:
            logger.error(f"Error getting {preference} papers: {str(e)}")
            return []

    def add_liked_paper(self, user_id: str = DEFAULT_USER_ID, arxiv_id: Optional[str] = None) -> bool:
        """标记喜欢；会同时取消该论文的“不喜欢”状态。"""
        return self._add_strong_preference(user_id, arxiv_id, "liked")

    def remove_liked_paper(self, user_id: str = DEFAULT_USER_ID, arxiv_id: Optional[str] = None) -> bool:
        """取消喜欢；返回是否真的删除了一条记录。"""
        return self._remove_strong_preference(user_id, arxiv_id, "liked")

    def get_liked_papers(self, user_id: str = DEFAULT_USER_ID) -> List[str]:
        """按标记时间倒序返回用户喜欢的论文 ID。"""
        return self._list_strong_preference_ids(user_id, "liked")

    def get_liked_papers_with_details(self, user_id: str = DEFAULT_USER_ID) -> List[Dict[str, Any]]:
        """返回喜欢列表并 JOIN 论文目录补齐展示字段；目录中缺失的论文不会出现在结果里。"""
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

    def add_disliked_paper(self, user_id: str = DEFAULT_USER_ID, arxiv_id: Optional[str] = None) -> bool:
        """标记不喜欢；会同时取消该论文的“喜欢”状态。"""
        return self._add_strong_preference(user_id, arxiv_id, "disliked")

    def remove_disliked_paper(self, user_id: str = DEFAULT_USER_ID, arxiv_id: Optional[str] = None) -> bool:
        """取消不喜欢；返回是否真的删除了一条记录。"""
        return self._remove_strong_preference(user_id, arxiv_id, "disliked")

    def get_disliked_papers(self, user_id: str = DEFAULT_USER_ID) -> List[str]:
        """按标记时间倒序返回用户不喜欢的论文 ID。"""
        return self._list_strong_preference_ids(user_id, "disliked")

    def get_user_preferences(self, user_id: str = DEFAULT_USER_ID) -> Dict[str, Any]:
        """聚合用户的强偏好、弱行为和研究画像；任一读取失败时返回结构完整的空结果。"""
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
                # _empty_user_research_profile 定义在 ResearchProfileStore 上（静态方法），
                # 本类并不继承它；此前这里写成 self._... 会在兜底分支再抛 AttributeError。
                'research_profile': ResearchProfileStore._empty_user_research_profile(user_id),
            }

    def record_user_paper_action(
        self,
        user_id: str = DEFAULT_USER_ID,
        arxiv_id: Optional[str] = None,
        action_type: str = "",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """记录一条弱行为（收藏、阅读等），同一 (用户, 论文, 行为) 重复写入只更新 metadata。

        like/dislike 这类显式偏好必须走 add_liked_paper/add_disliked_paper，这里会直接拒绝。
        """
        try:
            if not self._require_arxiv_id(arxiv_id):
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
        arxiv_id: Optional[str] = None,
        action_type: str = "",
    ) -> bool:
        """撤销一条弱行为；返回是否真的删除了记录。"""
        try:
            if not self._require_arxiv_id(arxiv_id):
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
        """返回用户的弱行为记录，按最近更新时间倒序；传 action_type 时只返回该类行为。"""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                where_clause = "user_id = ?"
                params: List[Any] = [user_id]
                if action_type:
                    where_clause += " AND action_type = ?"
                    params.append(self._normalize_action_type(action_type))
                cursor.execute(
                    f'''
                    SELECT {_PAPER_ACTION_COLUMNS}
                    FROM user_paper_actions
                    WHERE {where_clause}
                    ORDER BY updated_at DESC, created_at DESC
                    ''',
                    params,
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
        """把弱行为列表折叠成 {action_type: [arxiv_id, ...]}，方便前端按行为类型渲染状态。"""
        action_map: Dict[str, List[str]] = {}
        for item in self.get_user_paper_actions(user_id=user_id):
            action_key = str(item.get("action_type") or "").strip()
            arxiv_id = str(item.get("arxiv_id") or "").strip()
            if not action_key or not arxiv_id:
                continue
            action_map.setdefault(action_key, []).append(arxiv_id)
        return action_map

    def get_user_labeled_paper_count(self, user_id: str = DEFAULT_USER_ID) -> int:
        """统计用户做过强偏好标记的论文数（喜欢与不喜欢合并去重），用于判断画像冷启动。"""
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
