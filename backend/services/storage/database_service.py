import sqlite3
import os
import json
import uuid
from typing import Dict, Any, List, Optional
import logging
from datetime import datetime, timezone, timedelta
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
PAPER_INDEX_ACTIVE_JOB_STATUSES = ("pending", "running", "retrying")


class PaperQATurnPersistenceError(RuntimeError):
    """QA 单轮对话写入失败时抛出的强语义异常，避免关键写路径静默丢失。"""


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
                    chunk_file TEXT,
                    embedding_file TEXT,
                    loading_method TEXT,
                    chunking_strategy TEXT,
                    current_stage TEXT,
                    failed_stage TEXT,
                    error_message TEXT,
                    artifact_status TEXT DEFAULT 'active',
                    indexed_at TIMESTAMP,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')

            cursor.execute('''
                CREATE TABLE IF NOT EXISTS paper_qa_index_versions (
                    build_id TEXT PRIMARY KEY,
                    arxiv_id TEXT NOT NULL,
                    index_version TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'building',
                    is_active INTEGER NOT NULL DEFAULT 0,
                    collection_name TEXT,
                    chunk_count INTEGER DEFAULT 0,
                    embedding_model TEXT,
                    pdf_path TEXT,
                    chunk_file TEXT,
                    embedding_file TEXT,
                    loading_method TEXT,
                    chunking_strategy TEXT,
                    current_stage TEXT,
                    failed_stage TEXT,
                    error_message TEXT,
                    artifact_status TEXT DEFAULT 'active',
                    indexed_at TIMESTAMP,
                    activated_at TIMESTAMP,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(arxiv_id, index_version)
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
                    idempotency_key TEXT,
                    heartbeat_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
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

            cursor.execute(f'''
                CREATE TABLE IF NOT EXISTS agent_sessions (
                    session_id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL DEFAULT '{default_user_id_sql}',
                    status TEXT DEFAULT 'active',
                    selected_paper_json TEXT,
                    last_papers_json TEXT,
                    pending_action_json TEXT,
                    paper_qa_result_json TEXT,
                    active_arxiv_id TEXT,
                    active_paper_session_id TEXT,
                    last_intent TEXT,
                    last_tool_calls_summary_json TEXT,
                    last_response_summary TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')

            cursor.execute(f'''
                CREATE TABLE IF NOT EXISTS agent_runtime_checkpoints (
                    checkpoint_id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL DEFAULT '{default_user_id_sql}',
                    session_id TEXT NOT NULL,
                    thread_id TEXT NOT NULL,
                    runtime_state_json TEXT,
                    graph_state_json TEXT,
                    pending_confirmation_json TEXT,
                    current_node TEXT,
                    next_route TEXT,
                    status TEXT NOT NULL DEFAULT 'running',
                    error_summary TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    expires_at TIMESTAMP,
                    UNIQUE(user_id, session_id, thread_id)
                )
            ''')

            cursor.execute('''
                CREATE TABLE IF NOT EXISTS langgraph_checkpoints (
                    thread_id TEXT NOT NULL,
                    checkpoint_ns TEXT NOT NULL DEFAULT '',
                    checkpoint_id TEXT NOT NULL,
                    parent_checkpoint_id TEXT,
                    checkpoint_json TEXT NOT NULL,
                    metadata_json TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY(thread_id, checkpoint_ns, checkpoint_id)
                )
            ''')

            cursor.execute('''
                CREATE TABLE IF NOT EXISTS langgraph_checkpoint_writes (
                    thread_id TEXT NOT NULL,
                    checkpoint_ns TEXT NOT NULL DEFAULT '',
                    checkpoint_id TEXT NOT NULL,
                    task_id TEXT NOT NULL,
                    idx INTEGER NOT NULL,
                    channel TEXT NOT NULL,
                    value_json TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY(thread_id, checkpoint_ns, checkpoint_id, task_id, idx)
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
                CREATE INDEX IF NOT EXISTS idx_paper_qa_index_versions_arxiv_updated
                ON paper_qa_index_versions(arxiv_id, updated_at DESC)
            ''')

            cursor.execute('''
                CREATE UNIQUE INDEX IF NOT EXISTS idx_paper_qa_index_versions_one_active
                ON paper_qa_index_versions(arxiv_id)
                WHERE is_active = 1
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

            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_agent_sessions_user_updated
                ON agent_sessions(user_id, updated_at DESC)
            ''')

            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_agent_runtime_checkpoints_session
                ON agent_runtime_checkpoints(user_id, session_id, status, updated_at DESC)
            ''')

            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_agent_runtime_checkpoints_expiry
                ON agent_runtime_checkpoints(status, expires_at)
            ''')

            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_langgraph_checkpoints_thread_updated
                ON langgraph_checkpoints(thread_id, checkpoint_ns, updated_at DESC)
            ''')

            conn.commit()
            logger.info("Database tables initialized successfully")
            self._ensure_user_interest_vector_columns(conn)
            self._ensure_paper_qa_index_columns(conn)
            self._ensure_paper_qa_index_version_rows(conn)
            self._ensure_paper_index_job_columns(conn)

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

    def _ensure_paper_qa_index_columns(self, conn):
        # 旧环境可能已经创建过 paper_qa_index；这里补齐 artifact 字段，确保失败后仍可追踪残留文件和 collection。
        required_columns = {
            "chunk_file": "TEXT",
            "embedding_file": "TEXT",
            "loading_method": "TEXT",
            "chunking_strategy": "TEXT",
            "current_stage": "TEXT",
            "failed_stage": "TEXT",
            "error_message": "TEXT",
            "artifact_status": "TEXT DEFAULT 'active'",
            "indexed_at": "TIMESTAMP",
            "active_index_version": "TEXT",
            "active_build_id": "TEXT",
            "previous_build_id": "TEXT",
        }
        cursor = conn.cursor()
        cursor.execute("PRAGMA table_info(paper_qa_index)")
        existing_columns = {row[1] for row in cursor.fetchall()}
        for column_name, column_definition in required_columns.items():
            if column_name not in existing_columns:
                cursor.execute(
                    f"ALTER TABLE paper_qa_index ADD COLUMN {column_name} {column_definition}"
                )
        conn.commit()

    def _ensure_paper_qa_index_version_rows(self, conn):
        # 旧库只有 paper_qa_index 单行记录；这里把已可用索引补成 active version，避免升级后丢失可问答状态。
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT OR IGNORE INTO paper_qa_index_versions (
                build_id, arxiv_id, index_version, status, is_active, collection_name,
                chunk_count, embedding_model, pdf_path, chunk_file, embedding_file,
                loading_method, chunking_strategy, current_stage, failed_stage,
                error_message, artifact_status, indexed_at, activated_at, created_at, updated_at
            )
            SELECT
                'legacy-' || replace(replace(arxiv_id, '.', '_'), '/', '_'),
                arxiv_id,
                'legacy',
                CASE WHEN status = 'indexed' THEN 'active' ELSE status END,
                CASE WHEN status = 'indexed' THEN 1 ELSE 0 END,
                collection_name,
                chunk_count,
                embedding_model,
                pdf_path,
                chunk_file,
                embedding_file,
                loading_method,
                chunking_strategy,
                current_stage,
                failed_stage,
                error_message,
                artifact_status,
                indexed_at,
                CASE WHEN status = 'indexed' THEN COALESCE(indexed_at, updated_at, created_at, CURRENT_TIMESTAMP) ELSE NULL END,
                created_at,
                updated_at
            FROM paper_qa_index
            WHERE collection_name IS NOT NULL
              AND collection_name != ''
            """
        )
        cursor.execute(
            """
            UPDATE paper_qa_index
            SET active_index_version = COALESCE(active_index_version, 'legacy'),
                active_build_id = COALESCE(active_build_id, 'legacy-' || replace(replace(arxiv_id, '.', '_'), '/', '_'))
            WHERE status = 'indexed'
              AND collection_name IS NOT NULL
              AND collection_name != ''
            """
        )
        conn.commit()

    def _ensure_paper_index_job_columns(self, conn):
        # 旧库可能缺少心跳和幂等键；启动时补齐，避免用户为了恢复僵尸任务而手工删库。
        required_columns = {
            "idempotency_key": "TEXT",
            "heartbeat_at": "TIMESTAMP",
        }
        cursor = conn.cursor()
        cursor.execute("PRAGMA table_info(paper_index_jobs)")
        existing_columns = {row[1] for row in cursor.fetchall()}
        for column_name, column_definition in required_columns.items():
            if column_name not in existing_columns:
                cursor.execute(
                    f"ALTER TABLE paper_index_jobs ADD COLUMN {column_name} {column_definition}"
                )

        cursor.execute(
            """
            UPDATE paper_index_jobs
            SET idempotency_key = arxiv_id || ':' || COALESCE(NULLIF(loading_method, ''), 'docling')
            WHERE idempotency_key IS NULL OR idempotency_key = ''
            """
        )
        cursor.execute(
            """
            UPDATE paper_index_jobs
            SET heartbeat_at = COALESCE(updated_at, created_at, CURRENT_TIMESTAMP)
            WHERE heartbeat_at IS NULL OR heartbeat_at = ''
            """
        )
        cursor.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_paper_index_jobs_idempotency_status
            ON paper_index_jobs(idempotency_key, status, heartbeat_at)
            """
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

    def get_total_paper_count(self) -> int:
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('SELECT COUNT(*) FROM arxiv_papers')
                row = cursor.fetchone()
                return int(row[0] or 0) if row else 0
        except Exception as e:
            logger.error(f"Error getting total paper count: {str(e)}")
            return 0

    def get_today_new_paper_count(self) -> int:
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    '''
                    SELECT COUNT(*)
                    FROM arxiv_papers
                    WHERE date(created_at, 'localtime') = date('now', 'localtime')
                    '''
                )
                row = cursor.fetchone()
                return int(row[0] or 0) if row else 0
        except Exception as e:
            logger.error(f"Error getting today's new paper count: {str(e)}")
            return 0

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
                    SELECT arxiv_id, collection_name, status, chunk_count, embedding_model, pdf_path,
                           chunk_file, embedding_file, loading_method, chunking_strategy, current_stage,
                           failed_stage, error_message, artifact_status, indexed_at, created_at, updated_at,
                           active_index_version, active_build_id, previous_build_id
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
                        'chunk_file': row[6],
                        'embedding_file': row[7],
                        'loading_method': row[8],
                        'chunking_strategy': row[9],
                        'current_stage': row[10],
                        'failed_stage': row[11],
                        'error_message': row[12],
                        'artifact_status': row[13],
                        'indexed_at': row[14],
                        'created_at': row[15],
                        'updated_at': row[16],
                        'active_index_version': row[17],
                        'active_build_id': row[18],
                        'previous_build_id': row[19],
                    }
                return None
        except Exception as e:
            logger.error(f"Error getting paper QA index: {str(e)}")
            return None

    @staticmethod
    def _row_to_paper_qa_index_version(row: Any) -> Dict[str, Any]:
        return {
            "build_id": row[0],
            "arxiv_id": row[1],
            "index_version": row[2],
            "status": row[3],
            "is_active": bool(row[4]),
            "collection_name": row[5],
            "chunk_count": row[6],
            "embedding_model": row[7],
            "pdf_path": row[8],
            "chunk_file": row[9],
            "embedding_file": row[10],
            "loading_method": row[11],
            "chunking_strategy": row[12],
            "current_stage": row[13],
            "failed_stage": row[14],
            "error_message": row[15],
            "artifact_status": row[16],
            "indexed_at": row[17],
            "activated_at": row[18],
            "created_at": row[19],
            "updated_at": row[20],
        }

    @staticmethod
    def _paper_qa_index_version_select_sql() -> str:
        return """
            SELECT build_id, arxiv_id, index_version, status, is_active, collection_name,
                   chunk_count, embedding_model, pdf_path, chunk_file, embedding_file,
                   loading_method, chunking_strategy, current_stage, failed_stage,
                   error_message, artifact_status, indexed_at, activated_at, created_at, updated_at
            FROM paper_qa_index_versions
        """

    @staticmethod
    def _build_paper_qa_index_version_id(arxiv_id: str) -> str:
        safe_arxiv_id = "".join(ch if ch.isalnum() else "_" for ch in str(arxiv_id or "").strip()).strip("_") or "paper"
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S%f")
        suffix = uuid.uuid4().hex[:8]
        return f"{safe_arxiv_id}_{timestamp}_{suffix}"

    def create_paper_qa_index_build(self, arxiv_id: str, loading_method: str) -> Optional[Dict[str, Any]]:
        try:
            index_version = self._build_paper_qa_index_version_id(arxiv_id)
            build_id = str(uuid.uuid4())
            with self._get_connection() as conn:
                cursor = conn.cursor()
                # building version 只记录新构建的临时状态，不覆盖 paper_qa_index 中仍在线的 active 指针。
                cursor.execute(
                    """
                    INSERT INTO paper_qa_index_versions (
                        build_id, arxiv_id, index_version, status, is_active,
                        loading_method, current_stage, artifact_status
                    )
                    VALUES (?, ?, ?, 'building', 0, ?, 'create_build_version', 'active')
                    """,
                    (build_id, arxiv_id, index_version, loading_method),
                )
                cursor.execute(
                    """
                    INSERT INTO paper_qa_index (
                        arxiv_id, status, current_stage, loading_method, artifact_status
                    )
                    VALUES (?, 'not_indexed', 'create_build_version', ?, 'active')
                    ON CONFLICT(arxiv_id) DO UPDATE SET
                        current_stage = excluded.current_stage,
                        loading_method = excluded.loading_method,
                        updated_at = CURRENT_TIMESTAMP
                    """,
                    (arxiv_id, loading_method),
                )
                conn.commit()
            return self.get_paper_qa_index_build(build_id)
        except Exception as e:
            logger.error(f"Error creating paper QA index build: {str(e)}")
            return None

    def get_paper_qa_index_build(self, build_id: str) -> Optional[Dict[str, Any]]:
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    self._paper_qa_index_version_select_sql() + " WHERE build_id = ?",
                    (build_id,),
                )
                row = cursor.fetchone()
                return self._row_to_paper_qa_index_version(row) if row else None
        except Exception as e:
            logger.error(f"Error getting paper QA index build: {str(e)}")
            return None

    def get_active_paper_qa_index_build(self, arxiv_id: str) -> Optional[Dict[str, Any]]:
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    self._paper_qa_index_version_select_sql() + """
                    WHERE arxiv_id = ? AND is_active = 1
                    ORDER BY activated_at DESC, updated_at DESC
                    LIMIT 1
                    """,
                    (arxiv_id,),
                )
                row = cursor.fetchone()
                return self._row_to_paper_qa_index_version(row) if row else None
        except Exception as e:
            logger.error(f"Error getting active paper QA index build: {str(e)}")
            return None

    def get_latest_paper_qa_index_build(
        self,
        arxiv_id: str,
        *,
        statuses: Optional[List[str]] = None,
    ) -> Optional[Dict[str, Any]]:
        try:
            status_values = [str(item) for item in (statuses or []) if str(item).strip()]
            where_parts = ["arxiv_id = ?"]
            values: List[Any] = [arxiv_id]
            if status_values:
                where_parts.append(f"status IN ({','.join('?' for _ in status_values)})")
                values.extend(status_values)
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    self._paper_qa_index_version_select_sql()
                    + f"""
                    WHERE {" AND ".join(where_parts)}
                    ORDER BY updated_at DESC, created_at DESC
                    LIMIT 1
                    """,
                    values,
                )
                row = cursor.fetchone()
                return self._row_to_paper_qa_index_version(row) if row else None
        except Exception as e:
            logger.error(f"Error getting latest paper QA index build: {str(e)}")
            return None

    def list_paper_qa_index_builds(
        self,
        arxiv_id: str,
        *,
        statuses: Optional[List[str]] = None,
        limit: int = 20,
    ) -> List[Dict[str, Any]]:
        try:
            normalized_limit = max(1, int(limit))
            status_values = [str(item) for item in (statuses or []) if str(item).strip()]
            where_parts = ["arxiv_id = ?"]
            values: List[Any] = [arxiv_id]
            if status_values:
                where_parts.append(f"status IN ({','.join('?' for _ in status_values)})")
                values.extend(status_values)
            values.append(normalized_limit)
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    self._paper_qa_index_version_select_sql()
                    + f"""
                    WHERE {" AND ".join(where_parts)}
                    ORDER BY updated_at DESC, created_at DESC
                    LIMIT ?
                    """,
                    values,
                )
                return [self._row_to_paper_qa_index_version(row) for row in cursor.fetchall()]
        except Exception as e:
            logger.error(f"Error listing paper QA index builds: {str(e)}")
            return []

    def update_paper_qa_index_build(self, build_id: str, **kwargs) -> bool:
        try:
            allowed_fields = [
                "status",
                "collection_name",
                "chunk_count",
                "embedding_model",
                "pdf_path",
                "chunk_file",
                "embedding_file",
                "loading_method",
                "chunking_strategy",
                "current_stage",
                "failed_stage",
                "error_message",
                "artifact_status",
                "indexed_at",
            ]
            update_fields = []
            update_values = []
            for field_name in allowed_fields:
                if field_name in kwargs:
                    update_fields.append(f"{field_name} = ?")
                    update_values.append(kwargs[field_name])
            if not update_fields:
                return False
            update_fields.append("updated_at = CURRENT_TIMESTAMP")
            update_values.append(build_id)
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    f"""
                    UPDATE paper_qa_index_versions
                    SET {", ".join(update_fields)}
                    WHERE build_id = ?
                    """,
                    update_values,
                )
                conn.commit()
                return cursor.rowcount > 0
        except Exception as e:
            logger.error(f"Error updating paper QA index build: {str(e)}")
            return False

    def activate_paper_qa_index_build(self, build_id: str) -> bool:
        try:
            with self._get_connection() as conn:
                conn.isolation_level = None
                cursor = conn.cursor()
                cursor.execute("BEGIN IMMEDIATE")
                try:
                    cursor.execute(
                        self._paper_qa_index_version_select_sql() + " WHERE build_id = ?",
                        (build_id,),
                    )
                    row = cursor.fetchone()
                    if not row:
                        conn.rollback()
                        return False
                    build = self._row_to_paper_qa_index_version(row)
                    if build.get("status") not in {"build_success", "ready"}:
                        # 只有已经完成向量写入并校验过的新版本才能切 active，避免半成品被问答链路读到。
                        conn.rollback()
                        return False

                    arxiv_id = build["arxiv_id"]
                    cursor.execute(
                        """
                        SELECT build_id
                        FROM paper_qa_index_versions
                        WHERE arxiv_id = ? AND is_active = 1
                        LIMIT 1
                        """,
                        (arxiv_id,),
                    )
                    old_active_row = cursor.fetchone()
                    old_build_id = old_active_row[0] if old_active_row else None
                    if old_build_id and old_build_id != build_id:
                        # 旧 active 不在激活事务里删除，只标记为 cleanup_pending，给回滚和延迟清理留出空间。
                        cursor.execute(
                            """
                            UPDATE paper_qa_index_versions
                            SET is_active = 0,
                                status = 'cleanup_pending',
                                artifact_status = 'cleanup_pending',
                                updated_at = CURRENT_TIMESTAMP
                            WHERE build_id = ?
                            """,
                            (old_build_id,),
                        )

                    cursor.execute(
                        """
                        UPDATE paper_qa_index_versions
                        SET is_active = 1,
                            status = 'active',
                            artifact_status = 'active',
                            activated_at = CURRENT_TIMESTAMP,
                            updated_at = CURRENT_TIMESTAMP
                        WHERE build_id = ?
                        """,
                        (build_id,),
                    )
                    cursor.execute(
                        """
                        INSERT INTO paper_qa_index (
                            arxiv_id, collection_name, status, chunk_count, embedding_model,
                            pdf_path, chunk_file, embedding_file, loading_method, chunking_strategy,
                            current_stage, failed_stage, error_message, artifact_status, indexed_at,
                            active_index_version, active_build_id, previous_build_id
                        )
                        VALUES (?, ?, 'indexed', ?, ?, ?, ?, ?, ?, ?, 'activate_index', '', '', 'active', ?, ?, ?, ?)
                        ON CONFLICT(arxiv_id) DO UPDATE SET
                            collection_name = excluded.collection_name,
                            status = excluded.status,
                            chunk_count = excluded.chunk_count,
                            embedding_model = excluded.embedding_model,
                            pdf_path = excluded.pdf_path,
                            chunk_file = excluded.chunk_file,
                            embedding_file = excluded.embedding_file,
                            loading_method = excluded.loading_method,
                            chunking_strategy = excluded.chunking_strategy,
                            current_stage = excluded.current_stage,
                            failed_stage = excluded.failed_stage,
                            error_message = excluded.error_message,
                            artifact_status = excluded.artifact_status,
                            indexed_at = excluded.indexed_at,
                            active_index_version = excluded.active_index_version,
                            active_build_id = excluded.active_build_id,
                            previous_build_id = excluded.previous_build_id,
                            updated_at = CURRENT_TIMESTAMP
                        """,
                        (
                            arxiv_id,
                            build.get("collection_name"),
                            build.get("chunk_count") or 0,
                            build.get("embedding_model"),
                            build.get("pdf_path"),
                            build.get("chunk_file"),
                            build.get("embedding_file"),
                            build.get("loading_method"),
                            build.get("chunking_strategy"),
                            build.get("indexed_at") or datetime.now().isoformat(timespec="seconds"),
                            build.get("index_version"),
                            build_id,
                            old_build_id,
                        ),
                    )
                    conn.commit()
                    return True
                except Exception:
                    conn.rollback()
                    raise
        except Exception as e:
            logger.error(f"Error activating paper QA index build: {str(e)}")
            return False

    def count_paper_qa_index_builds(self, arxiv_id: str, *, statuses: Optional[List[str]] = None) -> int:
        try:
            status_values = [str(item) for item in (statuses or []) if str(item).strip()]
            where_parts = ["arxiv_id = ?"]
            values: List[Any] = [arxiv_id]
            if status_values:
                where_parts.append(f"status IN ({','.join('?' for _ in status_values)})")
                values.extend(status_values)
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    f"""
                    SELECT COUNT(*)
                    FROM paper_qa_index_versions
                    WHERE {" AND ".join(where_parts)}
                    """,
                    values,
                )
                row = cursor.fetchone()
                return int(row[0] or 0) if row else 0
        except Exception as e:
            logger.error(f"Error counting paper QA index builds: {str(e)}")
            return 0

    def mark_paper_qa_index_build_deleted(self, build_id: str) -> bool:
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    UPDATE paper_qa_index_versions
                    SET is_active = 0,
                        status = 'deleted',
                        artifact_status = 'deleted',
                        current_stage = 'cleanup_old_artifacts',
                        updated_at = CURRENT_TIMESTAMP
                    WHERE build_id = ?
                    """,
                    (build_id,),
                )
                conn.commit()
                return cursor.rowcount > 0
        except Exception as e:
            logger.error(f"Error marking paper QA index build deleted: {str(e)}")
            return False

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
            'heartbeat_at': row[9] if len(row) > 9 else None,
            'idempotency_key': row[10] if len(row) > 10 else None,
        }

    @staticmethod
    def _build_paper_index_job_idempotency_key(arxiv_id: str, loading_method: str) -> str:
        normalized_method = str(loading_method or "docling").strip().lower() or "docling"
        return f"{str(arxiv_id or '').strip()}:{normalized_method}"

    def create_paper_index_job(self, arxiv_id: str, loading_method: str) -> Optional[Dict[str, Any]]:
        try:
            job_id = str(uuid.uuid4())
            idempotency_key = self._build_paper_index_job_idempotency_key(arxiv_id, loading_method)
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO paper_index_jobs (
                        job_id, arxiv_id, status, current_stage, progress, error_message, loading_method, idempotency_key, heartbeat_at
                    )
                    VALUES (?, ?, 'pending', 'pending', 0, NULL, ?, ?, CURRENT_TIMESTAMP)
                ''', (job_id, arxiv_id, loading_method, idempotency_key))

                conn.commit()
                logger.info(f"Paper index job created: {job_id} for {arxiv_id}")

            return self.get_paper_index_job(job_id)
        except Exception as e:
            logger.error(f"Error creating paper index job: {str(e)}")
            return None

    def acquire_paper_index_job(
        self,
        arxiv_id: str,
        loading_method: str,
        *,
        timeout_seconds: int,
    ) -> Optional[Dict[str, Any]]:
        """在数据库写事务内领取 QA 索引任务，保证多进程下同一论文只产生一个 active job。"""
        job_id = str(uuid.uuid4())
        normalized_timeout = max(1, int(timeout_seconds or 1))
        idempotency_key = self._build_paper_index_job_idempotency_key(arxiv_id, loading_method)
        active_statuses = tuple(PAPER_INDEX_ACTIVE_JOB_STATUSES)
        placeholders = ",".join("?" for _ in active_statuses)
        stale_modifier = f"-{normalized_timeout} seconds"
        try:
            with self._get_connection() as conn:
                # BEGIN IMMEDIATE 会提前获取写锁；并发提交会排队，后来的请求能复用先提交的 active job。
                conn.isolation_level = None
                cursor = conn.cursor()
                cursor.execute("BEGIN IMMEDIATE")
                try:
                    cursor.execute(
                        f"""
                        SELECT job_id
                        FROM paper_index_jobs
                        WHERE arxiv_id = ?
                          AND status IN ({placeholders})
                          AND datetime(COALESCE(heartbeat_at, updated_at, created_at)) <= datetime('now', ?)
                        ORDER BY updated_at DESC, created_at DESC
                        """,
                        (arxiv_id, *active_statuses, stale_modifier),
                    )
                    stale_job_ids = [row[0] for row in cursor.fetchall()]
                    previous_job_id = stale_job_ids[0] if stale_job_ids else None
                    if stale_job_ids:
                        stale_placeholders = ",".join("?" for _ in stale_job_ids)
                        # stale 是可重试终态；这里明确写入原因，前端轮询旧 job 时不会再看到无解释的 running。
                        cursor.execute(
                            f"""
                            UPDATE paper_index_jobs
                            SET status = 'stale',
                                current_stage = 'stale',
                                error_message = CASE
                                    WHEN error_message IS NULL OR error_message = ''
                                    THEN ?
                                    ELSE error_message
                                END,
                                heartbeat_at = CURRENT_TIMESTAMP,
                                updated_at = CURRENT_TIMESTAMP
                            WHERE job_id IN ({stale_placeholders})
                            """,
                            (
                                f"QA index job heartbeat timed out after {normalized_timeout} seconds; submit again to retry.",
                                *stale_job_ids,
                            ),
                        )

                    cursor.execute(
                        f"""
                        SELECT job_id, arxiv_id, status, current_stage, progress, error_message,
                               loading_method, created_at, updated_at, heartbeat_at, idempotency_key
                        FROM paper_index_jobs
                        WHERE arxiv_id = ?
                          AND status IN ({placeholders})
                        ORDER BY updated_at DESC, created_at DESC
                        LIMIT 1
                        """,
                        (arxiv_id, *active_statuses),
                    )
                    existing_row = cursor.fetchone()
                    if existing_row:
                        conn.commit()
                        job = self._row_to_paper_index_job(existing_row)
                        job.update(
                            {
                                "created": False,
                                "previous_job_id": previous_job_id,
                                "recovery_action": "marked_stale_and_reused_active" if previous_job_id else "reused_active",
                            }
                        )
                        return job

                    cursor.execute(
                        """
                        INSERT INTO paper_index_jobs (
                            job_id, arxiv_id, status, current_stage, progress,
                            error_message, loading_method, idempotency_key, heartbeat_at
                        )
                        VALUES (?, ?, 'pending', 'pending', 0, NULL, ?, ?, CURRENT_TIMESTAMP)
                        """,
                        (job_id, arxiv_id, loading_method, idempotency_key),
                    )
                    cursor.execute(
                        """
                        SELECT job_id, arxiv_id, status, current_stage, progress, error_message,
                               loading_method, created_at, updated_at, heartbeat_at, idempotency_key
                        FROM paper_index_jobs
                        WHERE job_id = ?
                        """,
                        (job_id,),
                    )
                    created_row = cursor.fetchone()
                    conn.commit()
                    job = self._row_to_paper_index_job(created_row)
                    job.update(
                        {
                            "created": True,
                            "previous_job_id": previous_job_id,
                            "recovery_action": "marked_stale_and_created" if previous_job_id else "created",
                        }
                    )
                    logger.info(f"Paper index job acquired: {job_id} for {arxiv_id}")
                    return job
                except Exception:
                    conn.rollback()
                    raise
        except Exception as e:
            logger.error(f"Error acquiring paper index job: {str(e)}")
            return None

    def mark_stale_paper_index_jobs(
        self,
        *,
        arxiv_id: Optional[str] = None,
        job_id: Optional[str] = None,
        timeout_seconds: int,
    ) -> int:
        """把超过心跳阈值的 pending/running 任务标记为 stale，供提交和轮询前自愈使用。"""
        normalized_timeout = max(1, int(timeout_seconds or 1))
        stale_modifier = f"-{normalized_timeout} seconds"
        active_statuses = tuple(PAPER_INDEX_ACTIVE_JOB_STATUSES)
        placeholders = ",".join("?" for _ in active_statuses)
        filters = [f"status IN ({placeholders})", "datetime(COALESCE(heartbeat_at, updated_at, created_at)) <= datetime('now', ?)"]
        values: List[Any] = [*active_statuses, stale_modifier]
        if arxiv_id:
            filters.append("arxiv_id = ?")
            values.append(arxiv_id)
        if job_id:
            filters.append("job_id = ?")
            values.append(job_id)
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                # 查询接口也会调用本方法，因此错误信息要足够明确，方便前端展示旧任务已可重试。
                cursor.execute(
                    f"""
                    UPDATE paper_index_jobs
                    SET status = 'stale',
                        current_stage = 'stale',
                        error_message = CASE
                            WHEN error_message IS NULL OR error_message = ''
                            THEN ?
                            ELSE error_message
                        END,
                        heartbeat_at = CURRENT_TIMESTAMP,
                        updated_at = CURRENT_TIMESTAMP
                    WHERE {" AND ".join(filters)}
                    """,
                    [
                        f"QA index job heartbeat timed out after {normalized_timeout} seconds; submit again to retry.",
                        *values,
                    ],
                )
                conn.commit()
                return cursor.rowcount
        except Exception as e:
            logger.error(f"Error marking stale paper index jobs: {str(e)}")
            return 0

    def update_paper_index_job(
        self,
        job_id: str,
        status: Optional[str] = None,
        current_stage: Optional[str] = None,
        progress: Optional[int] = None,
        error_message: Optional[str] = None,
        heartbeat_at: Optional[str] = None,
        refresh_heartbeat: bool = True,
        expected_statuses: Optional[List[str]] = None,
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
                if heartbeat_at is not None:
                    update_fields.append('heartbeat_at = ?')
                    update_values.append(heartbeat_at)

                if not update_fields:
                    return False

                if refresh_heartbeat and heartbeat_at is None:
                    # 任何状态推进都代表后台线程仍活跃，同步刷新心跳用于后续 stale 判定。
                    update_fields.append('heartbeat_at = CURRENT_TIMESTAMP')
                update_fields.append('updated_at = CURRENT_TIMESTAMP')
                update_values.append(job_id)
                expected_status_values = [str(item) for item in (expected_statuses or []) if str(item).strip()]
                expected_clause = ""
                if expected_status_values:
                    # 后台线程可能在 stale 恢复后才继续回写；条件更新能阻止旧线程复活不可恢复任务。
                    expected_clause = f" AND status IN ({','.join('?' for _ in expected_status_values)})"
                    update_values.extend(expected_status_values)

                cursor.execute(f'''
                    UPDATE paper_index_jobs
                    SET {", ".join(update_fields)}
                    WHERE job_id = ?{expected_clause}
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
                    SELECT job_id, arxiv_id, status, current_stage, progress, error_message,
                           loading_method, created_at, updated_at, heartbeat_at, idempotency_key
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
                    SELECT job_id, arxiv_id, status, current_stage, progress, error_message,
                           loading_method, created_at, updated_at, heartbeat_at, idempotency_key
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
                        SELECT job_id, arxiv_id, status, current_stage, progress, error_message,
                               loading_method, created_at, updated_at, heartbeat_at, idempotency_key
                        FROM paper_index_jobs
                        WHERE arxiv_id = ?
                        ORDER BY updated_at DESC, created_at DESC
                        LIMIT ?
                    ''', (arxiv_id, normalized_limit))
                else:
                    cursor.execute('''
                        SELECT job_id, arxiv_id, status, current_stage, progress, error_message,
                               loading_method, created_at, updated_at, heartbeat_at, idempotency_key
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

                allowed_fields = [
                    'collection_name',
                    'status',
                    'chunk_count',
                    'embedding_model',
                    'pdf_path',
                    'chunk_file',
                    'embedding_file',
                    'loading_method',
                    'chunking_strategy',
                    'current_stage',
                    'failed_stage',
                    'error_message',
                    'artifact_status',
                    'indexed_at',
                    'active_index_version',
                    'active_build_id',
                    'previous_build_id',
                ]
                for field_name in allowed_fields:
                    if field_name in kwargs:
                        update_fields.append(f'{field_name} = ?')
                        update_values.append(kwargs[field_name])

                if not update_fields:
                    return False

                update_fields.append('updated_at = CURRENT_TIMESTAMP')
                update_values.append(arxiv_id)

                cursor.execute(f'''
                    UPDATE paper_qa_index
                    SET {", ".join(update_fields)}
                    WHERE arxiv_id = ?
                ''', update_values)

                if kwargs.get("status") == "indexed":
                    cursor.execute(
                        """
                        SELECT arxiv_id, collection_name, status, chunk_count, embedding_model, pdf_path,
                               chunk_file, embedding_file, loading_method, chunking_strategy, current_stage,
                               failed_stage, error_message, artifact_status, indexed_at,
                               active_index_version, active_build_id
                        FROM paper_qa_index
                        WHERE arxiv_id = ?
                        """,
                        (arxiv_id,),
                    )
                    active_row = cursor.fetchone()
                    if active_row and str(active_row[1] or "").strip():
                        legacy_version = active_row[15] or "legacy"
                        legacy_build_id = active_row[16] or f"legacy-{str(arxiv_id).replace('.', '_').replace('/', '_')}"
                        # 旧式 update 成功后也补 active version，保证版本化读取和回滚信息完整。
                        cursor.execute(
                            """
                            INSERT OR IGNORE INTO paper_qa_index_versions (
                                build_id, arxiv_id, index_version, status, is_active, collection_name,
                                chunk_count, embedding_model, pdf_path, chunk_file, embedding_file,
                                loading_method, chunking_strategy, current_stage, failed_stage,
                                error_message, artifact_status, indexed_at, activated_at
                            )
                            VALUES (?, ?, ?, 'active', 1, ?, ?, ?, ?, ?, ?, ?, ?, ?, '', '', ?, ?, CURRENT_TIMESTAMP)
                            """,
                            (
                                legacy_build_id,
                                arxiv_id,
                                legacy_version,
                                active_row[1],
                                active_row[3] or 0,
                                active_row[4],
                                active_row[5],
                                active_row[6],
                                active_row[7],
                                active_row[8],
                                active_row[9],
                                active_row[10] or "legacy_active",
                                active_row[13] or "active",
                                active_row[14],
                            ),
                        )
                        cursor.execute(
                            """
                            UPDATE paper_qa_index
                            SET active_index_version = COALESCE(active_index_version, ?),
                                active_build_id = COALESCE(active_build_id, ?)
                            WHERE arxiv_id = ?
                            """,
                            (legacy_version, legacy_build_id, arxiv_id),
                        )
                
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

                allowed_fields = [
                    'collection_name',
                    'status',
                    'chunk_count',
                    'embedding_model',
                    'pdf_path',
                    'chunk_file',
                    'embedding_file',
                    'loading_method',
                    'chunking_strategy',
                    'current_stage',
                    'failed_stage',
                    'error_message',
                    'artifact_status',
                    'indexed_at',
                    'active_index_version',
                    'active_build_id',
                    'previous_build_id',
                ]
                for field_name in allowed_fields:
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

                if kwargs.get("status") == "indexed" and str(kwargs.get("collection_name") or "").strip():
                    legacy_build_id = kwargs.get("active_build_id") or f"legacy-{str(arxiv_id).replace('.', '_').replace('/', '_')}"
                    legacy_version = kwargs.get("active_index_version") or "legacy"
                    # 兼容测试和旧调用：直接写入 paper_qa_index 的可用记录也补成 active version。
                    cursor.execute(
                        """
                        INSERT OR IGNORE INTO paper_qa_index_versions (
                            build_id, arxiv_id, index_version, status, is_active, collection_name,
                            chunk_count, embedding_model, pdf_path, chunk_file, embedding_file,
                            loading_method, chunking_strategy, current_stage, failed_stage,
                            error_message, artifact_status, indexed_at, activated_at
                        )
                        VALUES (?, ?, ?, 'active', 1, ?, ?, ?, ?, ?, ?, ?, ?, ?, '', '', ?, ?, CURRENT_TIMESTAMP)
                        """,
                        (
                            legacy_build_id,
                            arxiv_id,
                            legacy_version,
                            kwargs.get("collection_name"),
                            kwargs.get("chunk_count") or 0,
                            kwargs.get("embedding_model"),
                            kwargs.get("pdf_path"),
                            kwargs.get("chunk_file"),
                            kwargs.get("embedding_file"),
                            kwargs.get("loading_method"),
                            kwargs.get("chunking_strategy"),
                            kwargs.get("current_stage") or "legacy_active",
                            kwargs.get("artifact_status") or "active",
                            kwargs.get("indexed_at"),
                        ),
                    )
                    cursor.execute(
                        """
                        UPDATE paper_qa_index
                        SET active_index_version = COALESCE(active_index_version, ?),
                            active_build_id = COALESCE(active_build_id, ?)
                        WHERE arxiv_id = ?
                        """,
                        (legacy_version, legacy_build_id, arxiv_id),
                    )
                
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

    def _row_to_agent_session(self, row: Any) -> Dict[str, Any]:
        return {
            'session_id': row[0],
            'user_id': row[1],
            'status': row[2] or 'active',
            'selected_paper': self._deserialize_json_field(row[3]) or None,
            'last_papers': self._deserialize_json_field(row[4]) or [],
            'pending_action': self._deserialize_json_field(row[5]) or None,
            'paper_qa_result': self._deserialize_json_field(row[6]) or None,
            'active_arxiv_id': row[7] or '',
            'active_paper_session_id': row[8] or '',
            'last_intent': row[9] or '',
            'last_tool_calls_summary': self._deserialize_json_field(row[10]) or [],
            'last_response_summary': row[11] or '',
            'created_at': row[12],
            'updated_at': row[13],
        }

    def _row_to_agent_runtime_checkpoint(self, row: Any) -> Dict[str, Any]:
        return {
            'checkpoint_id': row[0],
            'user_id': row[1],
            'session_id': row[2],
            'thread_id': row[3],
            'runtime_state': self._deserialize_json_field(row[4]) or None,
            'graph_state': self._deserialize_json_field(row[5]) or None,
            'pending_confirmation': self._deserialize_json_field(row[6]) or None,
            'current_node': row[7] or '',
            'next_route': row[8] or '',
            'status': row[9] or 'running',
            'error_summary': row[10] or '',
            'created_at': row[11],
            'updated_at': row[12],
            'expires_at': row[13],
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

    def _get_paper_chat_message_in_transaction(self, cursor: sqlite3.Cursor, message_id: str) -> Optional[Dict[str, Any]]:
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
        cursor.execute(
            '''
            SELECT session_id, user_id, arxiv_id, title, created_at, updated_at, message_count, status
            FROM paper_chat_sessions
            WHERE session_id = ?
            ''',
            (session_id,),
        )
        row = cursor.fetchone()
        return self._row_to_paper_chat_session(row) if row else None

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

    def create_or_get_agent_session(
        self,
        user_id: str = DEFAULT_USER_ID,
        session_id: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        try:
            normalized_user_id = str(user_id or DEFAULT_USER_ID).strip() or DEFAULT_USER_ID
            normalized_session_id = str(session_id or uuid.uuid4()).strip()
            existing = self.get_agent_session(normalized_session_id, user_id=normalized_user_id)
            if existing:
                return existing

            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    '''
                    INSERT INTO agent_sessions (session_id, user_id, status)
                    VALUES (?, ?, ?)
                    ''',
                    (normalized_session_id, normalized_user_id, 'active'),
                )
                conn.commit()
            return self.get_agent_session(normalized_session_id, user_id=normalized_user_id)
        except Exception as e:
            logger.error(f"Error creating or getting agent session: {str(e)}")
            return None

    def get_agent_session(self, session_id: str, user_id: str = DEFAULT_USER_ID) -> Optional[Dict[str, Any]]:
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    '''
                    SELECT session_id, user_id, status, selected_paper_json, last_papers_json,
                           pending_action_json, paper_qa_result_json, active_arxiv_id,
                           active_paper_session_id, last_intent, last_tool_calls_summary_json,
                           last_response_summary, created_at, updated_at
                    FROM agent_sessions
                    WHERE session_id = ? AND user_id = ?
                    ''',
                    (session_id, user_id),
                )
                row = cursor.fetchone()
                return self._row_to_agent_session(row) if row else None
        except Exception as e:
            logger.error(f"Error getting agent session: {str(e)}")
            return None

    def upsert_agent_runtime_checkpoint(
        self,
        *,
        user_id: str = DEFAULT_USER_ID,
        session_id: str,
        thread_id: str,
        runtime_state: Optional[Dict[str, Any]] = None,
        graph_state: Optional[Dict[str, Any]] = None,
        pending_confirmation: Optional[Dict[str, Any]] = None,
        current_node: Optional[str] = None,
        next_route: Optional[str] = None,
        status: str = 'running',
        error_summary: Optional[str] = None,
        expires_at: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """写入 Agent 执行现场 checkpoint。

        这张表记录的是可恢复执行现场，不是前端展示镜像；pending_action 仍由 agent_sessions 保存，
        但 resume 校验必须以这里的 pending_confirmation/status 为准。
        """
        try:
            normalized_user_id = str(user_id or DEFAULT_USER_ID).strip() or DEFAULT_USER_ID
            normalized_session_id = str(session_id or '').strip()
            normalized_thread_id = str(thread_id or normalized_session_id).strip()
            if not normalized_session_id or not normalized_thread_id:
                return None

            checkpoint_id = f"{normalized_user_id}:{normalized_session_id}:{normalized_thread_id}"
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    '''
                    INSERT INTO agent_runtime_checkpoints (
                        checkpoint_id, user_id, session_id, thread_id, runtime_state_json,
                        graph_state_json, pending_confirmation_json, current_node, next_route,
                        status, error_summary, expires_at
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(user_id, session_id, thread_id) DO UPDATE SET
                        runtime_state_json = excluded.runtime_state_json,
                        graph_state_json = excluded.graph_state_json,
                        pending_confirmation_json = excluded.pending_confirmation_json,
                        current_node = excluded.current_node,
                        next_route = excluded.next_route,
                        status = excluded.status,
                        error_summary = excluded.error_summary,
                        expires_at = excluded.expires_at,
                        updated_at = CURRENT_TIMESTAMP
                    ''',
                    (
                        checkpoint_id,
                        normalized_user_id,
                        normalized_session_id,
                        normalized_thread_id,
                        self._serialize_json_field(runtime_state),
                        self._serialize_json_field(graph_state),
                        self._serialize_json_field(pending_confirmation),
                        str(current_node or '').strip(),
                        str(next_route or '').strip(),
                        str(status or 'running').strip() or 'running',
                        str(error_summary or '').strip(),
                        expires_at,
                    ),
                )
                conn.commit()
            return self.get_agent_runtime_checkpoint(
                user_id=normalized_user_id,
                session_id=normalized_session_id,
                thread_id=normalized_thread_id,
            )
        except Exception as e:
            logger.error(f"Error upserting agent runtime checkpoint: {str(e)}")
            return None

    def get_agent_runtime_checkpoint(
        self,
        *,
        user_id: str = DEFAULT_USER_ID,
        session_id: str,
        thread_id: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        try:
            normalized_user_id = str(user_id or DEFAULT_USER_ID).strip() or DEFAULT_USER_ID
            normalized_session_id = str(session_id or '').strip()
            normalized_thread_id = str(thread_id or normalized_session_id).strip()
            if not normalized_session_id or not normalized_thread_id:
                return None
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    '''
                    SELECT checkpoint_id, user_id, session_id, thread_id, runtime_state_json,
                           graph_state_json, pending_confirmation_json, current_node, next_route,
                           status, error_summary, created_at, updated_at, expires_at
                    FROM agent_runtime_checkpoints
                    WHERE user_id = ? AND session_id = ? AND thread_id = ?
                    ''',
                    (normalized_user_id, normalized_session_id, normalized_thread_id),
                )
                row = cursor.fetchone()
                return self._row_to_agent_runtime_checkpoint(row) if row else None
        except Exception as e:
            logger.error(f"Error getting agent runtime checkpoint: {str(e)}")
            return None

    def mark_agent_runtime_checkpoint_status(
        self,
        *,
        user_id: str = DEFAULT_USER_ID,
        session_id: str,
        thread_id: Optional[str] = None,
        status: str,
        error_summary: Optional[str] = None,
        clear_pending_confirmation: bool = False,
    ) -> bool:
        """更新执行现场终态或过期态，避免旧 confirmation 被重复 resume。"""
        try:
            normalized_user_id = str(user_id or DEFAULT_USER_ID).strip() or DEFAULT_USER_ID
            normalized_session_id = str(session_id or '').strip()
            normalized_thread_id = str(thread_id or normalized_session_id).strip()
            if not normalized_session_id or not normalized_thread_id:
                return False
            assignments = ['status = ?', 'error_summary = ?', 'updated_at = CURRENT_TIMESTAMP']
            values: List[Any] = [str(status or '').strip(), str(error_summary or '').strip()]
            if clear_pending_confirmation:
                assignments.append("pending_confirmation_json = ''")
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    f'''
                    UPDATE agent_runtime_checkpoints
                    SET {", ".join(assignments)}
                    WHERE user_id = ? AND session_id = ? AND thread_id = ?
                    ''',
                    values + [normalized_user_id, normalized_session_id, normalized_thread_id],
                )
                conn.commit()
                return cursor.rowcount > 0
        except Exception as e:
            logger.error(f"Error marking agent runtime checkpoint status: {str(e)}")
            return False

    def expire_agent_runtime_checkpoints(self, *, now: Optional[str] = None) -> int:
        """把超过 expires_at 的等待现场标记为 expired。

        清理先改状态而不是直接删除，是为了让前端/日志能得到明确“过期”语义。
        """
        try:
            now_text = now or datetime.now(timezone.utc).isoformat()
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    '''
                    UPDATE agent_runtime_checkpoints
                    SET status = 'expired',
                        error_summary = COALESCE(NULLIF(error_summary, ''), 'checkpoint_expired'),
                        pending_confirmation_json = '',
                        updated_at = CURRENT_TIMESTAMP
                    WHERE status = 'waiting_confirmation'
                      AND expires_at IS NOT NULL
                      AND expires_at <= ?
                    ''',
                    (now_text,),
                )
                conn.commit()
                return int(cursor.rowcount or 0)
        except Exception as e:
            logger.error(f"Error expiring agent runtime checkpoints: {str(e)}")
            return 0

    def cleanup_agent_runtime_checkpoints(self, *, retention_days: int = 7) -> int:
        """删除已终止且超过保留期的 runtime checkpoint，避免持久化表无限增长。"""
        try:
            cutoff = datetime.now(timezone.utc) - timedelta(days=max(int(retention_days or 0), 1))
            cutoff_text = cutoff.isoformat()
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    '''
                    DELETE FROM agent_runtime_checkpoints
                    WHERE status IN ('completed', 'cancelled', 'failed', 'expired')
                      AND updated_at <= ?
                    ''',
                    (cutoff_text,),
                )
                conn.commit()
                return int(cursor.rowcount or 0)
        except Exception as e:
            logger.error(f"Error cleaning agent runtime checkpoints: {str(e)}")
            return 0

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
        """保存 LangGraph 原始 checkpoint。

        这里不解释业务语义，只负责把 LangGraph 恢复所需的快照落到 SQLite。
        """
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
        try:
            normalized_thread_id = str(thread_id or '').strip()
            if not normalized_thread_id:
                return None
            with self._get_connection() as conn:
                cursor = conn.cursor()
                if checkpoint_id:
                    cursor.execute(
                        '''
                        SELECT checkpoint_id, parent_checkpoint_id, checkpoint_json, metadata_json, created_at, updated_at
                        FROM langgraph_checkpoints
                        WHERE thread_id = ? AND checkpoint_ns = ? AND checkpoint_id = ?
                        ''',
                        (normalized_thread_id, str(checkpoint_ns or ''), str(checkpoint_id)),
                    )
                else:
                    cursor.execute(
                        '''
                        SELECT checkpoint_id, parent_checkpoint_id, checkpoint_json, metadata_json, created_at, updated_at
                        FROM langgraph_checkpoints
                        WHERE thread_id = ? AND checkpoint_ns = ?
                        ORDER BY updated_at DESC, created_at DESC
                        LIMIT 1
                        ''',
                        (normalized_thread_id, str(checkpoint_ns or '')),
                    )
                row = cursor.fetchone()
                if not row:
                    return None
                pending_writes = self.get_langgraph_checkpoint_writes(
                    thread_id=normalized_thread_id,
                    checkpoint_ns=str(checkpoint_ns or ''),
                    checkpoint_id=row[0],
                )
                return {
                    'thread_id': normalized_thread_id,
                    'checkpoint_ns': str(checkpoint_ns or ''),
                    'checkpoint_id': row[0],
                    'parent_checkpoint_id': row[1],
                    'checkpoint': self._deserialize_json_field(row[2]) or {},
                    'metadata': self._deserialize_json_field(row[3]) or {},
                    'pending_writes': pending_writes,
                    'created_at': row[4],
                    'updated_at': row[5],
                }
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
        try:
            normalized_thread_id = str(thread_id or '').strip()
            if not normalized_thread_id:
                return []
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    '''
                    SELECT checkpoint_id, parent_checkpoint_id, checkpoint_json, metadata_json, created_at, updated_at
                    FROM langgraph_checkpoints
                    WHERE thread_id = ? AND checkpoint_ns = ?
                    ORDER BY updated_at DESC, created_at DESC
                    LIMIT ?
                    ''',
                    (normalized_thread_id, str(checkpoint_ns or ''), max(int(limit or 1), 1)),
                )
                rows = cursor.fetchall()
                checkpoints: List[Dict[str, Any]] = []
                for row in rows:
                    checkpoints.append(
                        {
                            'thread_id': normalized_thread_id,
                            'checkpoint_ns': str(checkpoint_ns or ''),
                            'checkpoint_id': row[0],
                            'parent_checkpoint_id': row[1],
                            'checkpoint': self._deserialize_json_field(row[2]) or {},
                            'metadata': self._deserialize_json_field(row[3]) or {},
                            'pending_writes': self.get_langgraph_checkpoint_writes(
                                thread_id=normalized_thread_id,
                                checkpoint_ns=str(checkpoint_ns or ''),
                                checkpoint_id=row[0],
                            ),
                            'created_at': row[4],
                            'updated_at': row[5],
                        }
                    )
                return checkpoints
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

    def update_agent_session(
        self,
        session_id: str,
        user_id: str = DEFAULT_USER_ID,
        memory_patch: Optional[Dict[str, Any]] = None,
    ) -> bool:
        try:
            if not session_id:
                return False

            allowed_fields = {
                'status': 'status',
                'selected_paper': 'selected_paper_json',
                'last_papers': 'last_papers_json',
                'pending_action': 'pending_action_json',
                'paper_qa_result': 'paper_qa_result_json',
                'active_arxiv_id': 'active_arxiv_id',
                'active_paper_session_id': 'active_paper_session_id',
                'last_intent': 'last_intent',
                'last_tool_calls_summary': 'last_tool_calls_summary_json',
                'last_response_summary': 'last_response_summary',
            }
            json_fields = {
                'selected_paper_json',
                'last_papers_json',
                'pending_action_json',
                'paper_qa_result_json',
                'last_tool_calls_summary_json',
            }

            normalized_patch = dict(memory_patch or {})
            update_fields = []
            update_values: List[Any] = []
            for patch_key, column_name in allowed_fields.items():
                if patch_key not in normalized_patch:
                    continue
                value = normalized_patch.get(patch_key)
                update_fields.append(f'{column_name} = ?')
                if column_name in json_fields:
                    update_values.append(self._serialize_json_field(value))
                else:
                    update_values.append(None if value is None else str(value).strip())

            if not update_fields:
                return True

            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    '''
                    INSERT OR IGNORE INTO agent_sessions (session_id, user_id, status)
                    VALUES (?, ?, ?)
                    ''',
                    (session_id, user_id, 'active'),
                )
                update_fields.append('updated_at = CURRENT_TIMESTAMP')
                update_values.extend([session_id, user_id])
                cursor.execute(
                    f'''
                    UPDATE agent_sessions
                    SET {", ".join(update_fields)}
                    WHERE session_id = ? AND user_id = ?
                    ''',
                    update_values,
                )
                conn.commit()
                return cursor.rowcount > 0
        except Exception as e:
            logger.error(f"Error updating agent session: {str(e)}")
            return False

    def clear_agent_session(self, session_id: str, user_id: str = DEFAULT_USER_ID) -> bool:
        return self.update_agent_session(
            session_id=session_id,
            user_id=user_id,
            memory_patch={
                'status': 'cleared',
                'selected_paper': None,
                'last_papers': None,
                'pending_action': None,
                'paper_qa_result': None,
                'active_arxiv_id': None,
                'active_paper_session_id': None,
                'last_intent': None,
                'last_tool_calls_summary': None,
                'last_response_summary': None,
            },
        )

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
        conn: Optional[sqlite3.Connection] = None
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            # 一轮 QA 是业务上的最小一致性单元，显式开启事务以保证 user/assistant/session 统计同进同退。
            cursor.execute('BEGIN IMMEDIATE')
            cursor.execute(
                '''
                SELECT session_id, title FROM paper_chat_sessions
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
            conn.commit()
            return {
                'turn_id': normalized_turn_id,
                'user_message': user_message,
                'assistant_message': assistant_message,
                'chat_session': refreshed_session,
                'refreshed_session': refreshed_session,
            }
        except Exception as exc:
            if conn is not None:
                # 回滚发生在数据库访问层，调用方只需要处理明确的写入失败语义。
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
        finally:
            if conn is not None:
                conn.close()

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
