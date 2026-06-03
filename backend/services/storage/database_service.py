import sqlite3
import os
import json
import uuid
from typing import Dict, Any, List, Optional
import logging
from datetime import datetime
from utils.config import SQLITE_CONFIG, get_default_user_id

logger = logging.getLogger(__name__)

DEFAULT_USER_ID = get_default_user_id()
PAPER_ACTION_TYPES = {
    "like",
    "dislike",
    "favorite",
    "read",
    "later",
    "archived",
    "note_saved",
    "not_interested",
}
PROFILE_LIST_FIELDS = {
    "positive_topics",
    "negative_topics",
    "recent_topics",
    "preferred_categories",
    "common_question_types",
    "representative_papers",
}
PAPER_NOTE_TYPES = {
    "summary",
    "method",
    "experiment",
    "result",
    "limitation",
    "idea",
    "todo",
    "custom",
}

class DatabaseService:
    def __init__(self):
        self.db_path = SQLITE_CONFIG["database_path"]
        self.check_same_thread = SQLITE_CONFIG["check_same_thread"]
        self._ensure_database_directory()
        self._initialize_database()

    def _ensure_database_directory(self):
        db_dir = os.path.dirname(self.db_path)
        if db_dir and not os.path.exists(db_dir):
            os.makedirs(db_dir, exist_ok=True)
            logger.info(f"Created database directory: {db_dir}")

    def _get_connection(self):
        return sqlite3.connect(
            self.db_path,
            check_same_thread=self.check_same_thread
        )

    def _initialize_database(self):
        with self._get_connection() as conn:
            cursor = conn.cursor()
            default_user_id_sql = DEFAULT_USER_ID.replace("'", "''")
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS arxiv_papers (
                    arxiv_id TEXT PRIMARY KEY,
                    title TEXT,
                    authors TEXT,
                    abstract TEXT,
                    categories TEXT,
                    published_date DATE,
                    url TEXT,
                    embedding_id TEXT,
                    embedding_model TEXT,
                    embedded_at TIMESTAMP,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            cursor.execute(f'''
                CREATE TABLE IF NOT EXISTS user_liked_papers (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL DEFAULT '{default_user_id_sql}',
                    arxiv_id TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(user_id, arxiv_id),
                    FOREIGN KEY(arxiv_id) REFERENCES arxiv_papers(arxiv_id)
                )
            ''')
            
            cursor.execute(f'''
                CREATE TABLE IF NOT EXISTS user_disliked_papers (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL DEFAULT '{default_user_id_sql}',
                    arxiv_id TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(user_id, arxiv_id),
                    FOREIGN KEY(arxiv_id) REFERENCES arxiv_papers(arxiv_id)
                )
            ''')
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS user_interest_vectors (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL UNIQUE,
                    vector_data TEXT NOT NULL,
                    paper_count INTEGER NOT NULL,
                    embedding_model TEXT NOT NULL,
                    vector_dimension INTEGER NOT NULL,
                    cluster_count INTEGER DEFAULT 0,
                    profile_mode TEXT DEFAULT 'mean',
                    interest_clusters TEXT,
                    weak_interest_pool TEXT,
                    disliked_vector_data TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS paper_qa_index (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    arxiv_id TEXT NOT NULL UNIQUE,
                    collection_name TEXT,
                    status TEXT DEFAULT 'not_indexed',
                    chunk_count INTEGER DEFAULT 0,
                    embedding_model TEXT,
                    pdf_path TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')

            cursor.execute('''
                CREATE TABLE IF NOT EXISTS paper_index_jobs (
                    job_id TEXT PRIMARY KEY,
                    arxiv_id TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'pending',
                    current_stage TEXT,
                    progress INTEGER NOT NULL DEFAULT 0,
                    error_message TEXT,
                    loading_method TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')

            cursor.execute(f'''
                CREATE TABLE IF NOT EXISTS paper_chat_sessions (
                    session_id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL DEFAULT '{default_user_id_sql}',
                    arxiv_id TEXT NOT NULL,
                    title TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    message_count INTEGER DEFAULT 0,
                    status TEXT DEFAULT 'active',
                    FOREIGN KEY(arxiv_id) REFERENCES arxiv_papers(arxiv_id)
                )
            ''')

            cursor.execute('''
                CREATE TABLE IF NOT EXISTS paper_chat_messages (
                    message_id TEXT PRIMARY KEY,
                    turn_id TEXT,
                    session_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT,
                    sources TEXT,
                    retrieval_debug_snapshot TEXT,
                    contextualized_question TEXT,
                    question_contextualization TEXT,
                    status TEXT DEFAULT 'completed',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY(session_id) REFERENCES paper_chat_sessions(session_id) ON DELETE CASCADE
                )
            ''')

            cursor.execute(f'''
                CREATE TABLE IF NOT EXISTS user_paper_actions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL DEFAULT '{default_user_id_sql}',
                    arxiv_id TEXT NOT NULL,
                    action_type TEXT NOT NULL,
                    metadata_json TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(user_id, arxiv_id, action_type),
                    FOREIGN KEY(arxiv_id) REFERENCES arxiv_papers(arxiv_id)
                )
            ''')

            cursor.execute('''
                CREATE TABLE IF NOT EXISTS user_research_profiles (
                    user_id TEXT PRIMARY KEY,
                    positive_topics TEXT,
                    negative_topics TEXT,
                    recent_topics TEXT,
                    preferred_categories TEXT,
                    preferred_answer_style TEXT,
                    common_question_types TEXT,
                    representative_papers TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')

            cursor.execute(f'''
                CREATE TABLE IF NOT EXISTS paper_notes (
                    note_id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL DEFAULT '{default_user_id_sql}',
                    arxiv_id TEXT NOT NULL,
                    session_id TEXT,
                    source_message_id TEXT,
                    title TEXT,
                    content TEXT,
                    note_type TEXT DEFAULT 'custom',
                    source_chunk_ids TEXT,
                    tags TEXT,
                    include_in_profile INTEGER DEFAULT 0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY(arxiv_id) REFERENCES arxiv_papers(arxiv_id),
                    FOREIGN KEY(session_id) REFERENCES paper_chat_sessions(session_id) ON DELETE SET NULL
                )
            ''')

            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_paper_chat_sessions_user_paper_updated
                ON paper_chat_sessions(user_id, arxiv_id, updated_at DESC)
            ''')

            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_paper_index_jobs_arxiv_updated
                ON paper_index_jobs(arxiv_id, updated_at DESC)
            ''')

            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_paper_index_jobs_status_updated
                ON paper_index_jobs(status, updated_at DESC)
            ''')

            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_paper_chat_messages_session_created
                ON paper_chat_messages(session_id, created_at ASC)
            ''')

            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_user_paper_actions_user_action_updated
                ON user_paper_actions(user_id, action_type, updated_at DESC)
            ''')

            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_user_paper_actions_user_paper
                ON user_paper_actions(user_id, arxiv_id)
            ''')

            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_paper_notes_user_paper_updated
                ON paper_notes(user_id, arxiv_id, updated_at DESC)
            ''')

            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_paper_notes_session_message
                ON paper_notes(session_id, source_message_id)
            ''')

            conn.commit()
            logger.info("Database tables initialized successfully")
            self._ensure_user_interest_vector_columns(conn)

    def _ensure_user_interest_vector_columns(self, conn):
        required_columns = {
            "cluster_count": "INTEGER DEFAULT 0",
            "profile_mode": "TEXT DEFAULT 'mean'",
            "interest_clusters": "TEXT",
            "weak_interest_pool": "TEXT",
            "disliked_vector_data": "TEXT",
        }
        cursor = conn.cursor()
        cursor.execute("PRAGMA table_info(user_interest_vectors)")
        existing_columns = {row[1] for row in cursor.fetchall()}
        for column_name, column_definition in required_columns.items():
            if column_name not in existing_columns:
                cursor.execute(
                    f"ALTER TABLE user_interest_vectors ADD COLUMN {column_name} {column_definition}"
                )
        conn.commit()

    @staticmethod
    def _normalize_action_type(action_type: Any) -> str:
        normalized = str(action_type or "").strip().lower()
        alias_map = {
            "liked": "like",
            "disliked": "dislike",
            "bookmark": "favorite",
            "bookmarked": "favorite",
            "saved": "later",
            "save_for_later": "later",
            "uninterested": "not_interested",
            "note": "note_saved",
        }
        return alias_map.get(normalized, normalized)

    @staticmethod
    def _empty_user_research_profile(user_id: str) -> Dict[str, Any]:
        return {
            "user_id": user_id,
            "positive_topics": [],
            "negative_topics": [],
            "recent_topics": [],
            "preferred_categories": [],
            "preferred_answer_style": "",
            "common_question_types": [],
            "representative_papers": [],
            "created_at": None,
            "updated_at": None,
        }

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
                
                conn.commit()
                self.record_user_paper_action(user_id=user_id, arxiv_id=arxiv_id, action_type="like")
                self.remove_user_paper_action(user_id=user_id, arxiv_id=arxiv_id, action_type="dislike")
                self.remove_user_paper_action(user_id=user_id, arxiv_id=arxiv_id, action_type="not_interested")
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
                
                conn.commit()
                self.remove_user_paper_action(user_id=user_id, arxiv_id=arxiv_id, action_type="like")
                logger.info(f"Paper {arxiv_id} removed from liked list for user: {user_id}")
                return cursor.rowcount > 0
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
                        'authors': row[3],
                        'categories': row[4],
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
                
                conn.commit()
                self.record_user_paper_action(user_id=user_id, arxiv_id=arxiv_id, action_type="dislike")
                self.remove_user_paper_action(user_id=user_id, arxiv_id=arxiv_id, action_type="like")
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
                
                conn.commit()
                self.remove_user_paper_action(user_id=user_id, arxiv_id=arxiv_id, action_type="dislike")
                logger.info(f"Paper {arxiv_id} removed from disliked list for user: {user_id}")
                return cursor.rowcount > 0
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
            if normalized_action not in PAPER_ACTION_TYPES:
                logger.error("Unsupported paper action type: %s", action_type)
                return False

            with self._get_connection() as conn:
                cursor = conn.cursor()
                if normalized_action in {"like", "dislike", "not_interested"}:
                    cursor.execute(
                        '''
                        DELETE FROM user_paper_actions
                        WHERE user_id = ? AND arxiv_id = ? AND action_type IN ('like', 'dislike', 'not_interested') AND action_type != ?
                        ''',
                        (user_id, arxiv_id, normalized_action),
                    )

                cursor.execute(
                    '''
                    INSERT INTO user_paper_actions (user_id, arxiv_id, action_type, metadata_json)
                    VALUES (?, ?, ?, ?)
                    ON CONFLICT(user_id, arxiv_id, action_type)
                    DO UPDATE SET metadata_json = excluded.metadata_json, updated_at = CURRENT_TIMESTAMP
                    ''',
                    (user_id, arxiv_id, normalized_action, self._serialize_json_field(metadata)),
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
                conn.commit()
                return cursor.rowcount > 0
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

    def upsert_user_research_profile(self, user_id: str = DEFAULT_USER_ID, profile: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        payload = dict(profile or {})
        normalized: Dict[str, Any] = {}
        for field_name in PROFILE_LIST_FIELDS:
            value = payload.get(field_name)
            if value is None:
                normalized[field_name] = []
            elif isinstance(value, list):
                normalized[field_name] = [str(item).strip() for item in value if str(item).strip()]
            else:
                normalized[field_name] = [str(value).strip()] if str(value).strip() else []

        normalized["preferred_answer_style"] = str(payload.get("preferred_answer_style", "") or "").strip()

        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    '''
                    INSERT INTO user_research_profiles (
                        user_id,
                        positive_topics,
                        negative_topics,
                        recent_topics,
                        preferred_categories,
                        preferred_answer_style,
                        common_question_types,
                        representative_papers,
                        updated_at
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                    ON CONFLICT(user_id) DO UPDATE SET
                        positive_topics = excluded.positive_topics,
                        negative_topics = excluded.negative_topics,
                        recent_topics = excluded.recent_topics,
                        preferred_categories = excluded.preferred_categories,
                        preferred_answer_style = excluded.preferred_answer_style,
                        common_question_types = excluded.common_question_types,
                        representative_papers = excluded.representative_papers,
                        updated_at = CURRENT_TIMESTAMP
                    ''',
                    (
                        user_id,
                        self._serialize_json_field(normalized["positive_topics"]),
                        self._serialize_json_field(normalized["negative_topics"]),
                        self._serialize_json_field(normalized["recent_topics"]),
                        self._serialize_json_field(normalized["preferred_categories"]),
                        normalized["preferred_answer_style"],
                        self._serialize_json_field(normalized["common_question_types"]),
                        self._serialize_json_field(normalized["representative_papers"]),
                    ),
                )
                conn.commit()
        except Exception as e:
            logger.error(f"Error upserting research profile: {str(e)}")

        return self.get_user_research_profile(user_id)

    def patch_user_research_profile(self, user_id: str = DEFAULT_USER_ID, profile: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        current = self.get_user_research_profile(user_id)
        merged = {**current, **dict(profile or {})}
        return self.upsert_user_research_profile(user_id=user_id, profile=merged)

    def get_user_research_profile(self, user_id: str = DEFAULT_USER_ID) -> Dict[str, Any]:
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    '''
                    SELECT user_id, positive_topics, negative_topics, recent_topics, preferred_categories,
                           preferred_answer_style, common_question_types, representative_papers, created_at, updated_at
                    FROM user_research_profiles WHERE user_id = ?
                    ''',
                    (user_id,),
                )
                row = cursor.fetchone()
                if not row:
                    return self._empty_user_research_profile(user_id)

                return {
                    "user_id": row[0],
                    "positive_topics": self._deserialize_json_field(row[1]) or [],
                    "negative_topics": self._deserialize_json_field(row[2]) or [],
                    "recent_topics": self._deserialize_json_field(row[3]) or [],
                    "preferred_categories": self._deserialize_json_field(row[4]) or [],
                    "preferred_answer_style": str(row[5] or ""),
                    "common_question_types": self._deserialize_json_field(row[6]) or [],
                    "representative_papers": self._deserialize_json_field(row[7]) or [],
                    "created_at": row[8],
                    "updated_at": row[9],
                }
        except Exception as e:
            logger.error(f"Error getting research profile: {str(e)}")
            return self._empty_user_research_profile(user_id)

    def get_latest_user_preference_timestamp(self, user_id: str = DEFAULT_USER_ID) -> Optional[str]:
        """
        返回该用户最新一次偏好的创建时间。
        用于判断兴趣向量是否已经过期。
        """
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    '''
                    SELECT MAX(latest_at) FROM (
                        SELECT created_at AS latest_at FROM user_liked_papers WHERE user_id = ?
                        UNION ALL
                        SELECT created_at AS latest_at FROM user_disliked_papers WHERE user_id = ?
                    )
                    ''',
                    (user_id, user_id),
                )
                row = cursor.fetchone()
                return row[0] if row and row[0] else None
        except Exception as e:
            logger.error(f"Error getting latest preference timestamp: {str(e)}")
            return None

    def get_latest_user_signal_timestamp(self, user_id: str = DEFAULT_USER_ID) -> Optional[str]:
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    '''
                    SELECT MAX(latest_at) FROM (
                        SELECT created_at AS latest_at FROM user_liked_papers WHERE user_id = ?
                        UNION ALL
                        SELECT created_at AS latest_at FROM user_disliked_papers WHERE user_id = ?
                        UNION ALL
                        SELECT updated_at AS latest_at FROM user_paper_actions WHERE user_id = ?
                        UNION ALL
                        SELECT updated_at AS latest_at FROM user_research_profiles WHERE user_id = ?
                    )
                    ''',
                    (user_id, user_id, user_id, user_id),
                )
                row = cursor.fetchone()
                return row[0] if row and row[0] else None
        except Exception as e:
            logger.error(f"Error getting latest user signal timestamp: {str(e)}")
            return self.get_latest_user_preference_timestamp(user_id)

    def add_paper(self, paper: Dict[str, Any]) -> bool:
        try:
            normalized_paper = {
                "arxiv_id": self._serialize_paper_db_value(paper.get("arxiv_id")),
                "title": self._serialize_paper_db_value(paper.get("title")),
                "authors": self._serialize_paper_db_value(paper.get("authors")),
                "abstract": self._serialize_paper_db_value(paper.get("abstract")),
                "categories": self._serialize_paper_db_value(paper.get("categories")),
                "published_date": self._serialize_paper_db_value(paper.get("published_date")),
                "url": self._serialize_paper_db_value(paper.get("url")),
                "embedding_id": self._serialize_paper_db_value(paper.get("embedding_id")),
                "embedding_model": self._serialize_paper_db_value(paper.get("embedding_model")),
            }

            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO arxiv_papers
                    (arxiv_id, title, authors, abstract, categories, published_date, url, embedding_id, embedding_model, embedded_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                    ON CONFLICT(arxiv_id) DO UPDATE SET
                        title = excluded.title,
                        authors = excluded.authors,
                        abstract = excluded.abstract,
                        categories = excluded.categories,
                        published_date = excluded.published_date,
                        url = excluded.url,
                        embedding_id = CASE
                            WHEN NULLIF(excluded.embedding_id, '') IS NOT NULL THEN excluded.embedding_id
                            ELSE arxiv_papers.embedding_id
                        END,
                        embedding_model = CASE
                            WHEN NULLIF(excluded.embedding_id, '') IS NOT NULL THEN excluded.embedding_model
                            ELSE arxiv_papers.embedding_model
                        END,
                        embedded_at = CASE
                            WHEN NULLIF(excluded.embedding_id, '') IS NOT NULL THEN excluded.embedded_at
                            ELSE arxiv_papers.embedded_at
                        END
                ''', (
                    normalized_paper["arxiv_id"],
                    normalized_paper["title"],
                    normalized_paper["authors"],
                    normalized_paper["abstract"],
                    normalized_paper["categories"],
                    normalized_paper["published_date"],
                    normalized_paper["url"],
                    normalized_paper["embedding_id"],
                    normalized_paper["embedding_model"],
                ))
                
                conn.commit()
                logger.info(f"Paper added: {normalized_paper.get('arxiv_id')}")
                return True
        except Exception as e:
            logger.error(f"Error adding paper: {str(e)}")
            return False

    def _serialize_paper_db_value(self, value: Any) -> str:
        if value is None:
            return ""

        if isinstance(value, (list, tuple, set)):
            return json.dumps(list(value), ensure_ascii=False)

        if isinstance(value, dict):
            return json.dumps(value, ensure_ascii=False)

        return str(value).strip()

    def _deserialize_paper_db_value(self, value: Any) -> Any:
        if not isinstance(value, str):
            return value

        text = value.strip()
        if not text:
            return ""

        if text.startswith("[") or text.startswith("{"):
            try:
                return json.loads(text)
            except json.JSONDecodeError:
                return value

        return value

    @staticmethod
    def _serialize_json_field(value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, str):
            return value
        return json.dumps(value, ensure_ascii=False)

    @staticmethod
    def _deserialize_json_field(value: Any) -> Any:
        if not isinstance(value, str):
            return value
        text = value.strip()
        if not text:
            return None
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return value

    def get_paper(self, arxiv_id: str) -> Optional[Dict[str, Any]]:
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT arxiv_id, title, authors, abstract, categories, published_date, url, embedding_id, embedding_model, created_at
                    FROM arxiv_papers WHERE arxiv_id = ?
                ''', (arxiv_id,))
                
                row = cursor.fetchone()
                if row:
                    return {
                        'arxiv_id': row[0],
                        'title': row[1],
                        'authors': self._deserialize_paper_db_value(row[2]),
                        'abstract': row[3],
                        'categories': self._deserialize_paper_db_value(row[4]),
                        'published_date': row[5],
                        'url': row[6],
                        'embedding_id': row[7],
                        'embedding_model': row[8],
                        'created_at': row[9]
                    }
                return None
        except Exception as e:
            logger.error(f"Error getting paper: {str(e)}")
            return None

    def search_papers_by_category(self, category: str) -> List[Dict[str, Any]]:
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT arxiv_id, title, authors, abstract, categories, published_date, url, embedding_id, created_at
                    FROM arxiv_papers WHERE categories LIKE ?
                ''', (f'%{category}%',))
                
                results = []
                for row in cursor.fetchall():
                    results.append({
                        'arxiv_id': row[0],
                        'title': row[1],
                        'authors': self._deserialize_paper_db_value(row[2]),
                        'abstract': row[3],
                        'categories': self._deserialize_paper_db_value(row[4]),
                        'published_date': row[5],
                        'url': row[6],
                        'embedding_id': row[7],
                        'created_at': row[8]
                    })
                return results
        except Exception as e:
            logger.error(f"Error searching papers by category: {str(e)}")
            return []

    def get_all_papers(self) -> List[Dict[str, Any]]:
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT arxiv_id, title, authors, abstract, categories, published_date, url, embedding_id, created_at
                    FROM arxiv_papers ORDER BY published_date DESC
                ''')
                
                results = []
                for row in cursor.fetchall():
                    results.append({
                        'arxiv_id': row[0],
                        'title': row[1],
                        'authors': self._deserialize_paper_db_value(row[2]),
                        'abstract': row[3],
                        'categories': self._deserialize_paper_db_value(row[4]),
                        'published_date': row[5],
                        'url': row[6],
                        'embedding_id': row[7],
                        'created_at': row[8]
                    })
                return results
        except Exception as e:
            logger.error(f"Error getting all papers: {str(e)}")
            return []

    def delete_paper(self, arxiv_id: str) -> bool:
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('DELETE FROM arxiv_papers WHERE arxiv_id = ?', (arxiv_id,))
                conn.commit()
                logger.info(f"Paper deleted: {arxiv_id}")
                return cursor.rowcount > 0
        except Exception as e:
            logger.error(f"Error deleting paper: {str(e)}")
            return False

    def update_paper_embedding(self, arxiv_id: str, embedding_id: int, embedding_model: str = None) -> bool:
        """
        更新论文的embedding_id和embedding_model信息
        
        参数:
            arxiv_id: 论文的arXiv ID
            embedding_id: 向量数据库中的embedding ID
            embedding_model: 使用的嵌入模型名称
            
        返回:
            是否更新成功
        """
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                if embedding_model:
                    cursor.execute('''
                        UPDATE arxiv_papers 
                        SET embedding_id = ?, embedding_model = ?, embedded_at = CURRENT_TIMESTAMP
                        WHERE arxiv_id = ?
                    ''', (str(embedding_id), embedding_model, arxiv_id))
                else:
                    cursor.execute('''
                        UPDATE arxiv_papers 
                        SET embedding_id = ?, embedded_at = CURRENT_TIMESTAMP
                        WHERE arxiv_id = ?
                    ''', (str(embedding_id), arxiv_id))
                
                conn.commit()
                logger.info(f"Paper {arxiv_id} embedding updated with id: {embedding_id}")
                return cursor.rowcount > 0
        except Exception as e:
            logger.error(f"Error updating paper embedding: {str(e)}")
            return False

    def save_user_interest_vector(
        self,
        user_id: str,
        vector_data: List[float],
        paper_count: int,
        embedding_model: str,
        vector_dimension: int,
        cluster_count: int = 0,
        profile_mode: str = "mean",
        interest_clusters: Optional[List[Dict[str, Any]]] = None,
        weak_interest_pool: Optional[Dict[str, Any]] = None,
        disliked_vector_data: Optional[List[float]] = None,
    ) -> bool:
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT OR REPLACE INTO user_interest_vectors 
                    (user_id, vector_data, paper_count, embedding_model, vector_dimension, cluster_count, profile_mode, interest_clusters, weak_interest_pool, disliked_vector_data, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                ''', (
                    user_id,
                    json.dumps(vector_data),
                    paper_count,
                    embedding_model,
                    vector_dimension,
                    cluster_count,
                    profile_mode,
                    json.dumps(interest_clusters) if interest_clusters is not None else None,
                    json.dumps(weak_interest_pool) if weak_interest_pool is not None else None,
                    json.dumps(disliked_vector_data) if disliked_vector_data is not None else None,
                ))
                
                conn.commit()
                logger.info(f"User interest vector saved for user: {user_id}")
                return True
        except Exception as e:
            logger.error(f"Error saving user interest vector: {str(e)}")
            return False

    def get_user_interest_vector(self, user_id: str = DEFAULT_USER_ID) -> Optional[Dict[str, Any]]:
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT user_id, vector_data, paper_count, embedding_model, vector_dimension, cluster_count, profile_mode, interest_clusters, weak_interest_pool, disliked_vector_data, created_at, updated_at
                    FROM user_interest_vectors WHERE user_id = ?
                ''', (user_id,))
                
                row = cursor.fetchone()
                if row:
                    interest_clusters = None
                    weak_interest_pool = None
                    disliked_vector_data = None
                    if row[7]:
                        try:
                            interest_clusters = json.loads(row[7])
                        except (TypeError, ValueError, json.JSONDecodeError):
                            interest_clusters = []
                    if row[8]:
                        try:
                            weak_interest_pool = json.loads(row[8])
                        except (TypeError, ValueError, json.JSONDecodeError):
                            weak_interest_pool = None
                    if row[9]:
                        try:
                            disliked_vector_data = json.loads(row[9])
                        except (TypeError, ValueError, json.JSONDecodeError):
                            disliked_vector_data = None
                    return {
                        'user_id': row[0],
                        'vector_data': json.loads(row[1]),
                        'paper_count': row[2],
                        'embedding_model': row[3],
                        'vector_dimension': row[4],
                        'cluster_count': row[5] or 0,
                        'profile_mode': row[6] or 'mean',
                        'interest_clusters': interest_clusters or [],
                        'weak_interest_pool': weak_interest_pool,
                        'disliked_vector_data': disliked_vector_data,
                        'created_at': row[10],
                        'updated_at': row[11]
                    }
                return None
        except Exception as e:
            logger.error(f"Error getting user interest vector: {str(e)}")
            return None

    def get_unlabeled_papers(self, user_id: str = DEFAULT_USER_ID) -> List[Dict[str, Any]]:
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT p.arxiv_id, p.title, p.abstract, p.authors, p.categories, p.published_date, p.url
                    FROM arxiv_papers p
                    LEFT JOIN user_liked_papers ulp ON p.arxiv_id = ulp.arxiv_id AND ulp.user_id = ?
                    LEFT JOIN user_disliked_papers udp ON p.arxiv_id = udp.arxiv_id AND udp.user_id = ?
                    WHERE ulp.arxiv_id IS NULL AND udp.arxiv_id IS NULL AND p.embedding_id IS NOT NULL
                    ORDER BY p.published_date DESC
                ''', (user_id, user_id))
                
                results = []
                for row in cursor.fetchall():
                    results.append({
                        'arxiv_id': row[0],
                        'title': row[1],
                        'abstract': row[2],
                        'authors': row[3],
                        'categories': row[4],
                        'published_date': row[5],
                        'url': row[6]
                    })
                return results
        except Exception as e:
            logger.error(f"Error getting unlabeled papers: {str(e)}")
            return []

    def get_paper_qa_index(self, arxiv_id: str) -> Optional[Dict[str, Any]]:
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT arxiv_id, collection_name, status, chunk_count, embedding_model, pdf_path, created_at, updated_at
                    FROM paper_qa_index WHERE arxiv_id = ?
                ''', (arxiv_id,))
                
                row = cursor.fetchone()
                if row:
                    return {
                        'arxiv_id': row[0],
                        'collection_name': row[1],
                        'status': row[2],
                        'chunk_count': row[3],
                        'embedding_model': row[4],
                        'pdf_path': row[5],
                        'created_at': row[6],
                        'updated_at': row[7]
                    }
                return None
        except Exception as e:
            logger.error(f"Error getting paper QA index: {str(e)}")
            return None

    @staticmethod
    def _row_to_paper_index_job(row: Any) -> Dict[str, Any]:
        return {
            'job_id': row[0],
            'arxiv_id': row[1],
            'status': row[2],
            'current_stage': row[3],
            'progress': row[4],
            'error_message': row[5],
            'loading_method': row[6],
            'created_at': row[7],
            'updated_at': row[8],
        }

    def create_paper_index_job(self, arxiv_id: str, loading_method: str) -> Optional[Dict[str, Any]]:
        try:
            job_id = str(uuid.uuid4())
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO paper_index_jobs (
                        job_id, arxiv_id, status, current_stage, progress, error_message, loading_method
                    )
                    VALUES (?, ?, 'pending', 'pending', 0, NULL, ?)
                ''', (job_id, arxiv_id, loading_method))

                conn.commit()
                logger.info(f"Paper index job created: {job_id} for {arxiv_id}")

            return self.get_paper_index_job(job_id)
        except Exception as e:
            logger.error(f"Error creating paper index job: {str(e)}")
            return None

    def update_paper_index_job(
        self,
        job_id: str,
        status: Optional[str] = None,
        current_stage: Optional[str] = None,
        progress: Optional[int] = None,
        error_message: Optional[str] = None,
    ) -> bool:
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()

                update_fields = []
                update_values = []

                if status is not None:
                    update_fields.append('status = ?')
                    update_values.append(status)
                if current_stage is not None:
                    update_fields.append('current_stage = ?')
                    update_values.append(current_stage)
                if progress is not None:
                    normalized_progress = max(0, min(100, int(progress)))
                    update_fields.append('progress = ?')
                    update_values.append(normalized_progress)
                if error_message is not None:
                    update_fields.append('error_message = ?')
                    update_values.append(error_message)

                if not update_fields:
                    return False

                update_fields.append('updated_at = CURRENT_TIMESTAMP')
                update_values.append(job_id)

                cursor.execute(f'''
                    UPDATE paper_index_jobs
                    SET {", ".join(update_fields)}
                    WHERE job_id = ?
                ''', update_values)

                conn.commit()
                if cursor.rowcount > 0:
                    logger.info(f"Paper index job updated: {job_id}")
                return cursor.rowcount > 0
        except Exception as e:
            logger.error(f"Error updating paper index job: {str(e)}")
            return False

    def get_paper_index_job(self, job_id: str) -> Optional[Dict[str, Any]]:
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT job_id, arxiv_id, status, current_stage, progress, error_message, loading_method, created_at, updated_at
                    FROM paper_index_jobs
                    WHERE job_id = ?
                ''', (job_id,))

                row = cursor.fetchone()
                if row:
                    return self._row_to_paper_index_job(row)
                return None
        except Exception as e:
            logger.error(f"Error getting paper index job: {str(e)}")
            return None

    def get_latest_paper_index_job(self, arxiv_id: str) -> Optional[Dict[str, Any]]:
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT job_id, arxiv_id, status, current_stage, progress, error_message, loading_method, created_at, updated_at
                    FROM paper_index_jobs
                    WHERE arxiv_id = ?
                    ORDER BY updated_at DESC, created_at DESC
                    LIMIT 1
                ''', (arxiv_id,))

                row = cursor.fetchone()
                if row:
                    return self._row_to_paper_index_job(row)
                return None
        except Exception as e:
            logger.error(f"Error getting latest paper index job: {str(e)}")
            return None

    def list_paper_index_jobs(self, arxiv_id: Optional[str] = None, limit: int = 20) -> List[Dict[str, Any]]:
        try:
            normalized_limit = max(1, int(limit))
            with self._get_connection() as conn:
                cursor = conn.cursor()
                if arxiv_id:
                    cursor.execute('''
                        SELECT job_id, arxiv_id, status, current_stage, progress, error_message, loading_method, created_at, updated_at
                        FROM paper_index_jobs
                        WHERE arxiv_id = ?
                        ORDER BY updated_at DESC, created_at DESC
                        LIMIT ?
                    ''', (arxiv_id, normalized_limit))
                else:
                    cursor.execute('''
                        SELECT job_id, arxiv_id, status, current_stage, progress, error_message, loading_method, created_at, updated_at
                        FROM paper_index_jobs
                        ORDER BY updated_at DESC, created_at DESC
                        LIMIT ?
                    ''', (normalized_limit,))

                return [self._row_to_paper_index_job(row) for row in cursor.fetchall()]
        except Exception as e:
            logger.error(f"Error listing paper index jobs: {str(e)}")
            return []

    def update_paper_qa_index(self, arxiv_id: str, **kwargs) -> bool:
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                
                update_fields = []
                update_values = []
                
                if 'collection_name' in kwargs:
                    update_fields.append('collection_name = ?')
                    update_values.append(kwargs['collection_name'])
                if 'status' in kwargs:
                    update_fields.append('status = ?')
                    update_values.append(kwargs['status'])
                if 'chunk_count' in kwargs:
                    update_fields.append('chunk_count = ?')
                    update_values.append(kwargs['chunk_count'])
                if 'embedding_model' in kwargs:
                    update_fields.append('embedding_model = ?')
                    update_values.append(kwargs['embedding_model'])
                if 'pdf_path' in kwargs:
                    update_fields.append('pdf_path = ?')
                    update_values.append(kwargs['pdf_path'])
                
                update_fields.append('updated_at = CURRENT_TIMESTAMP')
                update_values.append(arxiv_id)
                
                if update_fields:
                    cursor.execute(f'''
                        UPDATE paper_qa_index 
                        SET {", ".join(update_fields)}
                        WHERE arxiv_id = ?
                    ''', update_values)
                
                conn.commit()
                logger.info(f"Paper QA index updated for: {arxiv_id}")
                return cursor.rowcount > 0
        except Exception as e:
            logger.error(f"Error updating paper QA index: {str(e)}")
            return False

    def insert_paper_qa_index(self, arxiv_id: str, **kwargs) -> bool:
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                fields = ['arxiv_id']
                values = [arxiv_id]
                update_fields = []

                for field_name in ['collection_name', 'status', 'chunk_count', 'embedding_model', 'pdf_path']:
                    if field_name in kwargs:
                        fields.append(field_name)
                        values.append(kwargs[field_name])
                        update_fields.append(f"{field_name} = excluded.{field_name}")

                placeholders = ', '.join(['?' for _ in values])
                if update_fields:
                    update_fields.append('updated_at = CURRENT_TIMESTAMP')
                else:
                    update_fields = ['updated_at = CURRENT_TIMESTAMP']

                cursor.execute(f'''
                    INSERT INTO paper_qa_index ({", ".join(fields)})
                    VALUES ({placeholders})
                    ON CONFLICT(arxiv_id) DO UPDATE SET
                        {", ".join(update_fields)}
                ''', values)
                
                conn.commit()
                logger.info(f"Paper QA index inserted for: {arxiv_id}")
                return True
        except Exception as e:
            logger.error(f"Error inserting paper QA index: {str(e)}")
            return False

    @staticmethod
    def _row_to_paper_chat_session(row: Any) -> Dict[str, Any]:
        return {
            'session_id': row[0],
            'user_id': row[1],
            'arxiv_id': row[2],
            'title': row[3] or '',
            'created_at': row[4],
            'updated_at': row[5],
            'message_count': row[6] or 0,
            'status': row[7] or 'active',
        }

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

    def create_paper_chat_session(
        self,
        arxiv_id: str,
        user_id: str = DEFAULT_USER_ID,
        title: Optional[str] = None,
        status: str = 'active',
        session_id: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        try:
            normalized_session_id = str(session_id or uuid.uuid4())
            normalized_title = str(title or '').strip()
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    '''
                    INSERT INTO paper_chat_sessions (session_id, user_id, arxiv_id, title, status)
                    VALUES (?, ?, ?, ?, ?)
                    ''',
                    (normalized_session_id, user_id, arxiv_id, normalized_title, status),
                )
                conn.commit()
            return self.get_paper_chat_session(normalized_session_id, user_id=user_id)
        except Exception as e:
            logger.error(f"Error creating paper chat session: {str(e)}")
            return None

    def get_paper_chat_session(self, session_id: str, user_id: str = DEFAULT_USER_ID) -> Optional[Dict[str, Any]]:
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    '''
                    SELECT session_id, user_id, arxiv_id, title, created_at, updated_at, message_count, status
                    FROM paper_chat_sessions
                    WHERE session_id = ? AND user_id = ?
                    ''',
                    (session_id, user_id),
                )
                row = cursor.fetchone()
                return self._row_to_paper_chat_session(row) if row else None
        except Exception as e:
            logger.error(f"Error getting paper chat session: {str(e)}")
            return None

    def list_paper_chat_sessions(
        self,
        arxiv_id: str,
        user_id: str = DEFAULT_USER_ID,
        limit: int = 20,
    ) -> List[Dict[str, Any]]:
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    '''
                    SELECT session_id, user_id, arxiv_id, title, created_at, updated_at, message_count, status
                    FROM paper_chat_sessions
                    WHERE arxiv_id = ? AND user_id = ?
                    ORDER BY updated_at DESC, created_at DESC
                    LIMIT ?
                    ''',
                    (arxiv_id, user_id, limit),
                )
                return [self._row_to_paper_chat_session(row) for row in cursor.fetchall()]
        except Exception as e:
            logger.error(f"Error listing paper chat sessions: {str(e)}")
            return []

    def get_recent_paper_chat_session(self, arxiv_id: str, user_id: str = DEFAULT_USER_ID) -> Optional[Dict[str, Any]]:
        sessions = self.list_paper_chat_sessions(arxiv_id=arxiv_id, user_id=user_id, limit=1)
        return sessions[0] if sessions else None

    def update_paper_chat_session(self, session_id: str, user_id: str = DEFAULT_USER_ID, **kwargs) -> bool:
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                update_fields = []
                update_values = []
                for field_name in ('title', 'status'):
                    if field_name in kwargs:
                        update_fields.append(f'{field_name} = ?')
                        update_values.append(kwargs[field_name])
                update_fields.append('updated_at = CURRENT_TIMESTAMP')
                update_values.extend([session_id, user_id])
                cursor.execute(
                    f'''
                    UPDATE paper_chat_sessions
                    SET {", ".join(update_fields)}
                    WHERE session_id = ? AND user_id = ?
                    ''',
                    update_values,
                )
                conn.commit()
                return cursor.rowcount > 0
        except Exception as e:
            logger.error(f"Error updating paper chat session: {str(e)}")
            return False

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
                    SELECT session_id, title FROM paper_chat_sessions
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

            return self.get_paper_note(note_id, user_id=user_id)
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
