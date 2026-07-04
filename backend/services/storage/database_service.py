import sqlite3
import os
import json
import uuid
import re
from typing import Dict, Any, List, Optional
import logging
from datetime import datetime, timezone, timedelta
from services.storage.database.notes import PaperNoteMixin
from services.storage.database.papers import PaperCatalogMixin
from services.storage.database.agent_sessions import AgentSessionMixin
from services.storage.database.paper_profile_evidence import PaperProfileEvidenceMixin
from services.storage.database.paper_chat_sessions import PaperChatSessionMixin
from services.storage.database.paper_chat_messages import PaperChatMessageMixin
from services.storage.database.paper_qa_turns import PaperQATurnMixin
from services.storage.database.preferences import UserPreferenceMixin
from services.storage.database.interest_vectors import InterestVectorMixin
from services.storage.database.profile_build_jobs import ProfileBuildJobMixin
from services.storage.database.profile_events import ProfileEventMixin
from services.storage.database.serialization import JsonFieldMixin
from services.storage.database.shared import (
    DEFAULT_USER_ID,
    PAPER_NOTE_TYPES,
    PaperQATurnPersistenceError,
    PROFILE_BUILD_VERSION,
    PROFILE_EXTRACTOR_VERSION,
    PROFILE_NORMALIZER_VERSION,
)
from utils.config import SQLITE_CONFIG

logger = logging.getLogger(__name__)

PROFILE_LIST_FIELDS = {
    "positive_topics",
    "negative_topics",
    "recent_topics",
    "preferred_categories",
    "common_question_types",
    "representative_papers",
}
PROFILE_CANONICAL_TOPIC_FIELDS = {
    "canonical_topics",
    "canonical_negative_topics",
    "canonical_recent_topics",
}
PAPER_INDEX_ACTIVE_JOB_STATUSES = ("pending", "running", "retrying")
PROFILE_TOPIC_BLOCKLIST = {
    "analysis",
    "approach",
    "framework",
    "method",
    "methods",
    "model",
    "models",
    "paper",
    "papers",
    "system",
    "systems",
}
PROFILE_ARXIV_ID_PATTERN = re.compile(r"^(?:\d{4}\.\d{4,5}(?:v\d+)?|[a-z-]+(?:\.[A-Z]{2})?/\d{7}(?:v\d+)?)$")
PROFILE_ARXIV_CATEGORY_PATTERN = re.compile(r"^[a-z-]+(?:\.[A-Z]{2})?$")
PROFILE_URL_PATTERN = re.compile(r"https?://|www\.", re.IGNORECASE)


class DatabaseService(
    JsonFieldMixin,
    PaperCatalogMixin,
    UserPreferenceMixin,
    InterestVectorMixin,
    ProfileEventMixin,
    ProfileBuildJobMixin,
    PaperProfileEvidenceMixin,
    PaperChatSessionMixin,
    PaperChatMessageMixin,
    PaperQATurnMixin,
    PaperNoteMixin,
    AgentSessionMixin,
):
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
                    disliked_paper_examples TEXT,
                    negative_feedback_stats TEXT,
                    negative_feedback_profile TEXT,
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
                    retrieval_index_file TEXT,
                    retrieval_index_count INTEGER DEFAULT 0,
                    retrieval_index_types TEXT,
                    retrieval_index_version TEXT,
                    sparse_index_dir TEXT,
                    sparse_index_manifest_file TEXT,
                    sparse_index_document_count INTEGER DEFAULT 0,
                    sparse_index_token_count INTEGER DEFAULT 0,
                    sparse_index_backend TEXT,
                    sparse_index_schema_version TEXT,
                    sparse_index_source_file TEXT,
                    sparse_index_source_hash TEXT,
                    sparse_index_avgdl REAL DEFAULT 0,
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
                    retrieval_index_file TEXT,
                    retrieval_index_count INTEGER DEFAULT 0,
                    retrieval_index_types TEXT,
                    retrieval_index_version TEXT,
                    sparse_index_dir TEXT,
                    sparse_index_manifest_file TEXT,
                    sparse_index_document_count INTEGER DEFAULT 0,
                    sparse_index_token_count INTEGER DEFAULT 0,
                    sparse_index_backend TEXT,
                    sparse_index_schema_version TEXT,
                    sparse_index_source_file TEXT,
                    sparse_index_source_hash TEXT,
                    sparse_index_avgdl REAL DEFAULT 0,
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
                    summary_json TEXT,
                    summary_updated_at TIMESTAMP,
                    summary_turn_count INTEGER DEFAULT 0,
                    summary_last_turn_id TEXT,
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

            cursor.execute('''
                CREATE TABLE IF NOT EXISTS user_profile_events (
                    event_id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    event_type TEXT,
                    source_type TEXT NOT NULL,
                    source_id TEXT,
                    action_type TEXT NOT NULL,
                    action_strength REAL DEFAULT 0,
                    source TEXT,
                    arxiv_id TEXT,
                    note_id TEXT,
                    session_id TEXT,
                    payload_json TEXT,
                    metadata_json TEXT,
                    include_in_profile INTEGER DEFAULT 1,
                    consumed_by_job_id TEXT,
                    consumed_at TIMESTAMP,
                    dedupe_key TEXT,
                    profile_dirty INTEGER DEFAULT 1,
                    extractor_version TEXT,
                    normalizer_version TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')

            cursor.execute('''
                CREATE TABLE IF NOT EXISTS paper_profile_evidence (
                    evidence_id TEXT PRIMARY KEY,
                    arxiv_id TEXT NOT NULL,
                    concepts_json TEXT,
                    methods_json TEXT,
                    tasks_json TEXT,
                    objects_json TEXT,
                    applications_json TEXT,
                    categories_json TEXT,
                    raw_payload_json TEXT,
                    extractor_version TEXT,
                    normalizer_version TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(arxiv_id, extractor_version, normalizer_version)
                )
            ''')

            cursor.execute('''
                CREATE TABLE IF NOT EXISTS user_generated_profiles (
                    user_id TEXT PRIMARY KEY,
                    snapshot_id TEXT,
                    profile_json TEXT NOT NULL,
                    evidence_summary_json TEXT,
                    quality_report_json TEXT,
                    build_config_json TEXT,
                    extractor_version TEXT,
                    normalizer_version TEXT,
                    profile_build_version TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')

            cursor.execute('''
                CREATE TABLE IF NOT EXISTS user_manual_profiles (
                    user_id TEXT PRIMARY KEY,
                    profile_json TEXT NOT NULL,
                    pinned_items_json TEXT,
                    blocked_items_json TEXT,
                    deleted_items_json TEXT,
                    source TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')

            cursor.execute('''
                CREATE TABLE IF NOT EXISTS user_effective_profiles (
                    user_id TEXT PRIMARY KEY,
                    profile_json TEXT NOT NULL,
                    generated_snapshot_id TEXT,
                    merge_report_json TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')

            cursor.execute('''
                CREATE TABLE IF NOT EXISTS user_profile_snapshots (
                    snapshot_id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    generated_profile_json TEXT NOT NULL,
                    manual_profile_json TEXT,
                    effective_profile_json TEXT NOT NULL,
                    evidence_summary_json TEXT,
                    quality_report_json TEXT,
                    build_config_json TEXT,
                    extractor_version TEXT,
                    normalizer_version TEXT,
                    profile_build_version TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')

            cursor.execute('''
                CREATE TABLE IF NOT EXISTS user_profile_build_jobs (
                    job_id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'pending',
                    snapshot_id TEXT,
                    current_stage TEXT,
                    progress INTEGER NOT NULL DEFAULT 0,
                    error_message TEXT,
                    metrics_json TEXT,
                    build_config_json TEXT,
                    extractor_version TEXT,
                    normalizer_version TEXT,
                    profile_build_version TEXT,
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
                CREATE INDEX IF NOT EXISTS idx_paper_chat_messages_session_created_desc
                ON paper_chat_messages(session_id, created_at DESC)
            ''')

            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_user_paper_actions_user_action_updated
                ON user_paper_actions(user_id, action_type, updated_at DESC)
            ''')

            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_user_profile_events_user_created
                ON user_profile_events(user_id, created_at DESC)
            ''')

            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_user_profile_events_user_type_created
                ON user_profile_events(user_id, event_type, created_at DESC)
            ''')

            cursor.execute('''
                CREATE UNIQUE INDEX IF NOT EXISTS idx_user_profile_events_dedupe
                ON user_profile_events(user_id, dedupe_key)
                WHERE dedupe_key IS NOT NULL
            ''')

            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_user_profile_events_paper
                ON user_profile_events(user_id, arxiv_id, action_type)
            ''')

            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_user_profile_snapshots_user_created
                ON user_profile_snapshots(user_id, created_at DESC)
            ''')

            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_user_profile_build_jobs_user_updated
                ON user_profile_build_jobs(user_id, updated_at DESC)
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
                CREATE INDEX IF NOT EXISTS idx_agent_runtime_checkpoints_thread
                ON agent_runtime_checkpoints(session_id, thread_id, updated_at DESC)
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
            self._ensure_user_profile_event_columns(conn)
            self._ensure_user_profile_build_job_columns(conn)
            self._ensure_paper_qa_index_columns(conn)
            self._ensure_paper_qa_index_version_columns(conn)
            self._ensure_paper_qa_index_version_rows(conn)
            self._ensure_paper_index_job_columns(conn)
            self._ensure_paper_chat_session_summary_columns(conn)
            self._migrate_legacy_research_profiles(conn)

    def _ensure_paper_qa_index_columns(self, conn):
        # 旧环境可能已经创建过 paper_qa_index；这里补齐 artifact 字段，保留失败后的文件与 collection 追踪。
        required_columns = {
            "chunk_file": "TEXT",
            "retrieval_index_file": "TEXT",
            "retrieval_index_count": "INTEGER DEFAULT 0",
            "retrieval_index_types": "TEXT",
            "retrieval_index_version": "TEXT",
            "sparse_index_dir": "TEXT",
            "sparse_index_manifest_file": "TEXT",
            "sparse_index_document_count": "INTEGER DEFAULT 0",
            "sparse_index_token_count": "INTEGER DEFAULT 0",
            "sparse_index_backend": "TEXT",
            "sparse_index_schema_version": "TEXT",
            "sparse_index_source_file": "TEXT",
            "sparse_index_source_hash": "TEXT",
            "sparse_index_avgdl": "REAL DEFAULT 0",
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

    def _ensure_paper_qa_index_version_columns(self, conn):
        # 版本表和 active 表必须拥有同一组 artifact 字段，否则激活/回滚时会丢失 retrieval index 产物定位。
        required_columns = {
            "retrieval_index_file": "TEXT",
            "retrieval_index_count": "INTEGER DEFAULT 0",
            "retrieval_index_types": "TEXT",
            "retrieval_index_version": "TEXT",
            "sparse_index_dir": "TEXT",
            "sparse_index_manifest_file": "TEXT",
            "sparse_index_document_count": "INTEGER DEFAULT 0",
            "sparse_index_token_count": "INTEGER DEFAULT 0",
            "sparse_index_backend": "TEXT",
            "sparse_index_schema_version": "TEXT",
            "sparse_index_source_file": "TEXT",
            "sparse_index_source_hash": "TEXT",
            "sparse_index_avgdl": "REAL DEFAULT 0",
        }
        cursor = conn.cursor()
        cursor.execute("PRAGMA table_info(paper_qa_index_versions)")
        existing_columns = {row[1] for row in cursor.fetchall()}
        for column_name, column_definition in required_columns.items():
            if column_name not in existing_columns:
                cursor.execute(
                    f"ALTER TABLE paper_qa_index_versions ADD COLUMN {column_name} {column_definition}"
                )
        conn.commit()

    def _ensure_paper_qa_index_version_rows(self, conn):
        # 旧库只有 paper_qa_index 单行记录；启动时补 active version，避免升级后丢失可问答状态。
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT OR IGNORE INTO paper_qa_index_versions (
                build_id, arxiv_id, index_version, status, is_active, collection_name,
                chunk_count, embedding_model, pdf_path, chunk_file,
                retrieval_index_file, retrieval_index_count, retrieval_index_types, retrieval_index_version,
                sparse_index_dir, sparse_index_manifest_file, sparse_index_document_count, sparse_index_token_count, sparse_index_backend,
                sparse_index_schema_version, sparse_index_source_file, sparse_index_source_hash, sparse_index_avgdl,
                embedding_file,
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
                retrieval_index_file,
                COALESCE(retrieval_index_count, 0),
                retrieval_index_types,
                retrieval_index_version,
                sparse_index_dir,
                sparse_index_manifest_file,
                COALESCE(sparse_index_document_count, 0),
                COALESCE(sparse_index_token_count, 0),
                sparse_index_backend,
                sparse_index_schema_version,
                sparse_index_source_file,
                sparse_index_source_hash,
                COALESCE(sparse_index_avgdl, 0),
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
        # 旧库可能缺少心跳和幂等键，启动时补齐，避免用户为了恢复任务而手工删库。
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

    @classmethod
    def _empty_profile_projection(cls, user_id: str) -> Dict[str, Any]:
        return cls._empty_user_research_profile(user_id)

    @staticmethod
    def _normalize_profile_list_value(values: Any, limit: int = 30) -> List[str]:
        if values is None:
            return []
        source = values if isinstance(values, list) else [values]
        normalized: List[str] = []
        for item in source:
            text = str(item or "").strip()
            if text and text not in normalized:
                normalized.append(text)
            if len(normalized) >= limit:
                break
        return normalized

    @staticmethod
    def _looks_like_profile_arxiv_id(value: str) -> bool:
        text = str(value or "").strip()
        if text.startswith(("http://", "https://")):
            text = text.rstrip("/").rsplit("/", 1)[-1]
        return bool(text and PROFILE_ARXIV_ID_PATTERN.match(text))

    @staticmethod
    def _looks_like_profile_arxiv_category(value: str) -> bool:
        text = str(value or "").strip()
        return bool(text and PROFILE_ARXIV_CATEGORY_PATTERN.match(text) and ("." in text or text.startswith("cs.")))

    @staticmethod
    def _looks_like_profile_paper_title(value: str) -> bool:
        text = str(value or "").strip()
        words = [part for part in re.split(r"\s+", text) if part]
        if len(text) > 60 or len(words) > 6:
            return True
        title_joiners = {"for", "with", "of", "using", "via", "towards", "toward", "based"}
        lower_words = {word.strip(".,:;!?()[]{}").lower() for word in words}
        if len(words) >= 4 and lower_words & title_joiners:
            return True
        return any(marker in text for marker in (":", "?", "!", " -- ", " - "))

    @classmethod
    def _normalize_profile_topics(cls, values: Any, limit: int = 30) -> List[str]:
        normalized: List[str] = []
        for item in cls._normalize_profile_list_value(values, limit=limit * 3):
            text = str(item or "").strip()
            if not text:
                continue
            if PROFILE_URL_PATTERN.search(text) or cls._looks_like_profile_arxiv_id(text):
                continue
            if cls._looks_like_profile_arxiv_category(text) or text.lower().startswith("cs."):
                continue
            if cls._looks_like_profile_paper_title(text) or text.lower() in PROFILE_TOPIC_BLOCKLIST:
                continue
            if text not in normalized:
                normalized.append(text)
            if len(normalized) >= limit:
                break
        return normalized

    @classmethod
    def _normalize_profile_categories(cls, values: Any, limit: int = 20) -> List[str]:
        categories: List[str] = []
        source: List[str] = []
        for item in cls._normalize_profile_list_value(values, limit=limit * 4):
            if item.startswith("["):
                try:
                    parsed = json.loads(item)
                except json.JSONDecodeError:
                    parsed = None
                if isinstance(parsed, list):
                    source.extend(cls._normalize_profile_list_value(parsed, limit=limit * 4))
                    continue
            source.extend([part.strip() for part in re.split(r"[,\s]+", item) if part.strip()])
        for item in source:
            if not cls._looks_like_profile_arxiv_category(item):
                continue
            if item not in categories:
                categories.append(item)
            if len(categories) >= limit:
                break
        return categories

    @classmethod
    def _normalize_profile_papers(cls, values: Any, limit: int = 20) -> List[str]:
        normalized: List[str] = []
        for item in cls._normalize_profile_list_value(values, limit=limit * 2):
            text = item.rstrip("/").rsplit("/", 1)[-1] if item.startswith(("http://", "https://")) else item
            if not cls._looks_like_profile_arxiv_id(text):
                continue
            if text not in normalized:
                normalized.append(text)
            if len(normalized) >= limit:
                break
        return normalized

    @classmethod
    def _normalize_canonical_topics(cls, values: Any, limit: int = 30) -> List[Dict[str, Any]]:
        """保留 canonical topic 对象边界，旧字段只从其中投影出可展示 label。"""
        source = values if isinstance(values, list) else []
        normalized: List[Dict[str, Any]] = []
        seen: set[str] = set()
        for item in source:
            if isinstance(item, str):
                item = {"label": item}
            if not isinstance(item, dict):
                continue
            label = cls._normalize_profile_topics([item.get("label")], limit=1)
            if not label:
                continue
            normalized_label = label[0]
            key = normalized_label.lower()
            if key in seen:
                continue
            seen.add(key)
            normalized.append(
                {
                    "label": normalized_label,
                    "aliases": cls._normalize_profile_topics(item.get("aliases"), limit=20),
                    "description": str(item.get("description") or "").strip()[:800],
                    "topic_type": str(item.get("topic_type") or item.get("type") or "technical_concept").strip() or "technical_concept",
                    "merge_confidence": cls._coerce_float(item.get("merge_confidence"), default=0.0),
                    "score": cls._coerce_float(item.get("score"), default=0.0),
                    "source_concepts": cls._normalize_source_concepts(item.get("source_concepts")),
                    "source_papers": cls._normalize_profile_papers(item.get("source_papers"), limit=30),
                    "source_events": cls._normalize_profile_list_value(item.get("source_events"), limit=50),
                    "source": str(item.get("source") or "").strip(),
                    "pinned": bool(item.get("pinned")),
                    "normalizer_version": str(item.get("normalizer_version") or PROFILE_NORMALIZER_VERSION).strip(),
                }
            )
            if len(normalized) >= limit:
                break
        return normalized

    @classmethod
    def _normalize_source_concepts(cls, values: Any, limit: int = 50) -> List[Dict[str, Any]]:
        concepts: List[Dict[str, Any]] = []
        for item in values if isinstance(values, list) else []:
            if not isinstance(item, dict):
                continue
            label = cls._normalize_profile_topics([item.get("label")], limit=1)
            if not label:
                continue
            concepts.append(
                {
                    "label": label[0],
                    "type": str(item.get("type") or "technical_concept").strip() or "technical_concept",
                    "confidence": cls._coerce_float(item.get("confidence"), default=0.0),
                    "score": cls._coerce_float(item.get("score"), default=0.0),
                    "source": str(item.get("source") or "").strip(),
                    "source_papers": cls._normalize_profile_papers(item.get("source_papers"), limit=20),
                    "source_events": cls._normalize_profile_list_value(item.get("source_events"), limit=30),
                    "evidence_texts": cls._normalize_profile_list_value(item.get("evidence_texts"), limit=5),
                }
            )
            if len(concepts) >= limit:
                break
        return concepts

    @staticmethod
    def _coerce_float(value: Any, *, default: float = 0.0) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    @classmethod
    def _topic_keys(cls, values: Any) -> set[str]:
        return {item.lower() for item in cls._normalize_profile_topics(values, limit=100)}

    @classmethod
    def _filter_hidden_topics(cls, values: List[str], hidden_topics: Any, limit: int = 30) -> List[str]:
        hidden_keys = cls._topic_keys(hidden_topics)
        return [item for item in cls._normalize_profile_topics(values, limit=limit * 2) if item.lower() not in hidden_keys][:limit]

    @classmethod
    def _filter_hidden_canonical_topics(cls, values: List[Dict[str, Any]], hidden_topics: Any, limit: int = 30) -> List[Dict[str, Any]]:
        hidden_keys = cls._topic_keys(hidden_topics)
        return [item for item in cls._normalize_canonical_topics(values, limit=limit * 2) if item["label"].lower() not in hidden_keys][:limit]

    @classmethod
    def _normalize_profile_projection(cls, user_id: str, profile: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        payload = dict(profile or {})
        normalized = {
            "user_id": user_id,
            "positive_topics": cls._normalize_profile_topics(payload.get("positive_topics"), limit=30),
            "negative_topics": cls._normalize_profile_topics(payload.get("negative_topics"), limit=30),
            "recent_topics": cls._normalize_profile_topics(payload.get("recent_topics"), limit=30),
            "preferred_categories": cls._normalize_profile_categories(payload.get("preferred_categories"), limit=20),
            "preferred_answer_style": str(payload.get("preferred_answer_style", "") or "").strip(),
            "common_question_types": cls._normalize_profile_list_value(payload.get("common_question_types"), limit=20),
            "representative_papers": cls._normalize_profile_papers(payload.get("representative_papers"), limit=20),
            "created_at": payload.get("created_at"),
            "updated_at": payload.get("updated_at"),
        }
        normalized.update(
            {
                "canonical_topics": cls._normalize_canonical_topics(payload.get("canonical_topics"), limit=30),
                "canonical_negative_topics": cls._normalize_canonical_topics(payload.get("canonical_negative_topics"), limit=30),
                "canonical_recent_topics": cls._normalize_canonical_topics(payload.get("canonical_recent_topics"), limit=30),
                "pinned_topics": cls._normalize_profile_topics(payload.get("pinned_topics"), limit=30),
                "hidden_topics": cls._normalize_profile_topics(payload.get("hidden_topics"), limit=30),
                "normalizer_version": str(payload.get("normalizer_version") or PROFILE_NORMALIZER_VERSION).strip(),
                "normalization_signature": str(payload.get("normalization_signature") or "").strip(),
            }
        )
        for extra_field in ("topic_evidence", "aggregation_report", "review_status", "quality_report", "aggregator_version"):
            value = payload.get(extra_field)
            if value is not None:
                # 解释性字段不参与旧字段归一化，但 snapshot/effective 需要保留它们供调试和前端解释来源。
                normalized[extra_field] = value
        return normalized

    @classmethod
    def _merge_profile_projection(
        cls,
        user_id: str,
        generated_profile: Optional[Dict[str, Any]],
        manual_profile: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        generated = cls._normalize_profile_projection(user_id, generated_profile)
        manual = cls._normalize_profile_projection(user_id, manual_profile)
        hidden_topics = manual["hidden_topics"]
        pinned_topics = manual["pinned_topics"]
        generated_positive_topics = cls._filter_hidden_topics(generated["positive_topics"], hidden_topics, limit=30)
        generated_negative_topics = cls._filter_hidden_topics(generated["negative_topics"], hidden_topics, limit=30)
        generated_recent_topics = cls._filter_hidden_topics(generated["recent_topics"], hidden_topics, limit=30)
        manual_positive_topics = cls._filter_hidden_topics(manual["positive_topics"], hidden_topics, limit=30)
        manual_negative_topics = cls._filter_hidden_topics(manual["negative_topics"], hidden_topics, limit=30)
        manual_recent_topics = cls._filter_hidden_topics(manual["recent_topics"], hidden_topics, limit=30)
        pinned_canonical = cls._canonical_topics_from_manual(pinned_topics)
        generated_canonical = cls._filter_hidden_canonical_topics(generated["canonical_topics"], hidden_topics, limit=30)
        generated_negative_canonical = cls._filter_hidden_canonical_topics(generated["canonical_negative_topics"], hidden_topics, limit=30)
        generated_recent_canonical = cls._filter_hidden_canonical_topics(generated["canonical_recent_topics"], hidden_topics, limit=30)
        # 手动隐藏是用户显式排除项，合并时必须先过滤；手动固定则作为高优先级 canonical topic 保留。
        effective = {
            "user_id": user_id,
            "positive_topics": cls._normalize_profile_list_value([*pinned_topics, *manual_positive_topics, *generated_positive_topics], limit=30),
            "negative_topics": cls._normalize_profile_list_value([*manual_negative_topics, *generated_negative_topics], limit=30),
            "recent_topics": cls._normalize_profile_list_value([*generated_recent_topics, *manual_recent_topics], limit=30),
            "preferred_categories": cls._normalize_profile_list_value([*manual["preferred_categories"], *generated["preferred_categories"]], limit=20),
            "preferred_answer_style": manual["preferred_answer_style"] or generated["preferred_answer_style"],
            "common_question_types": cls._normalize_profile_list_value([*manual["common_question_types"], *generated["common_question_types"]], limit=20),
            "representative_papers": cls._normalize_profile_list_value([*generated["representative_papers"], *manual["representative_papers"]], limit=20),
            "canonical_topics": cls._normalize_canonical_topics([*pinned_canonical, *generated_canonical], limit=30),
            "canonical_negative_topics": generated_negative_canonical,
            "canonical_recent_topics": generated_recent_canonical,
            "pinned_topics": pinned_topics,
            "hidden_topics": hidden_topics,
            "normalizer_version": generated.get("normalizer_version") or PROFILE_NORMALIZER_VERSION,
            "normalization_signature": generated.get("normalization_signature") or "",
        }
        for extra_field in ("topic_evidence", "aggregation_report", "review_status", "quality_report", "aggregator_version"):
            if extra_field in generated:
                # effective profile 是推荐和 Agent 的读取边界，保留解释字段方便下游说明 topic 来源。
                effective[extra_field] = generated[extra_field]
        return effective

    @classmethod
    def _canonical_topics_from_manual(cls, topics: Any) -> List[Dict[str, Any]]:
        canonical_topics: List[Dict[str, Any]] = []
        for topic in cls._normalize_profile_topics(topics, limit=30):
            canonical_topics.append(
                {
                    "label": topic,
                    "aliases": [],
                    "description": "用户手动固定的研究主题。",
                    "topic_type": "manual_topic",
                    "merge_confidence": 1.0,
                    "score": 10.0,
                    "source_concepts": [
                        {
                            "label": topic,
                            "type": "manual_topic",
                            "confidence": 1.0,
                            "score": 10.0,
                            "source": "manual_profile",
                            "source_papers": [],
                            "source_events": [],
                            "evidence_texts": [],
                        }
                    ],
                    "source_papers": [],
                    "source_events": [],
                    "source": "manual_profile",
                    "pinned": True,
                    "normalizer_version": PROFILE_NORMALIZER_VERSION,
                }
            )
        return canonical_topics
    def _upsert_legacy_research_profile_cache(self, conn, user_id: str, profile: Dict[str, Any]) -> None:
        normalized = self._normalize_profile_projection(user_id, profile)
        cursor = conn.cursor()
        cursor.execute(
            '''
            INSERT INTO user_research_profiles (
                user_id, positive_topics, negative_topics, recent_topics, preferred_categories,
                preferred_answer_style, common_question_types, representative_papers, updated_at
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

    def _migrate_legacy_research_profiles(self, conn) -> None:
        cursor = conn.cursor()
        cursor.execute(
            '''
            SELECT user_id, positive_topics, negative_topics, recent_topics, preferred_categories,
                   preferred_answer_style, common_question_types, representative_papers
            FROM user_research_profiles
            '''
        )
        rows = cursor.fetchall()
        for row in rows:
            user_id = str(row[0] or DEFAULT_USER_ID).strip() or DEFAULT_USER_ID
            cursor.execute("SELECT 1 FROM user_manual_profiles WHERE user_id = ?", (user_id,))
            manual_exists = cursor.fetchone() is not None
            cursor.execute("SELECT 1 FROM user_generated_profiles WHERE user_id = ?", (user_id,))
            generated_exists = cursor.fetchone() is not None
            if manual_exists or generated_exists:
                continue

            legacy_profile = {
                "positive_topics": self._deserialize_json_field(row[1]) or [],
                "negative_topics": self._deserialize_json_field(row[2]) or [],
                "recent_topics": self._deserialize_json_field(row[3]) or [],
                "preferred_categories": self._deserialize_json_field(row[4]) or [],
                "preferred_answer_style": row[5] or "",
                "common_question_types": self._deserialize_json_field(row[6]) or [],
                "representative_papers": self._deserialize_json_field(row[7]) or [],
            }
            manual_profile = self._normalize_profile_projection(
                user_id,
                {
                    "positive_topics": legacy_profile.get("positive_topics"),
                    "negative_topics": legacy_profile.get("negative_topics"),
                    "preferred_categories": legacy_profile.get("preferred_categories"),
                    "preferred_answer_style": legacy_profile.get("preferred_answer_style"),
                    "common_question_types": legacy_profile.get("common_question_types"),
                },
            )
            # 旧 topic 没有来源标记，只能以低置信度候选进入 manual；标题、URL、分类和 arXiv ID 会被清洗丢弃。
            cursor.execute(
                '''
                INSERT OR IGNORE INTO user_manual_profiles (
                    user_id, profile_json, pinned_items_json, blocked_items_json, deleted_items_json, source
                )
                VALUES (?, ?, ?, ?, ?, ?)
                ''',
                (
                    user_id,
                    self._serialize_json_field(manual_profile),
                    self._serialize_json_field([]),
                    self._serialize_json_field([]),
                    self._serialize_json_field([]),
                    "legacy_migration_low_confidence",
                ),
            )
            effective = self._merge_profile_projection(user_id, {}, manual_profile)
            cursor.execute(
                '''
                INSERT OR IGNORE INTO user_effective_profiles (user_id, profile_json, merge_report_json)
                VALUES (?, ?, ?)
                ''',
                (
                    user_id,
                    self._serialize_json_field(effective),
                    self._serialize_json_field({"source": "legacy_migration", "legacy_fields": list(legacy_profile.keys())}),
                ),
            )
            self._upsert_legacy_research_profile_cache(conn, user_id, effective)
        conn.commit()

    def upsert_user_research_profile(self, user_id: str = DEFAULT_USER_ID, profile: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        normalized = self._normalize_profile_projection(user_id, profile)
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                # 鏃?upsert 鍏ュ彛鐜板湪鍙啓 manual profile锛沞ffective 鐢?manual/generated 鍚堝苟寰楀埌锛岄伩鍏嶆墜鍔ㄤ繚瀛樿鐩栬嚜鍔ㄧ敾鍍忋€?
                cursor.execute(
                    '''
                    INSERT INTO user_manual_profiles (
                        user_id, profile_json, pinned_items_json, blocked_items_json, deleted_items_json, source, updated_at
                    )
                    VALUES (?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                    ON CONFLICT(user_id) DO UPDATE SET
                        profile_json = excluded.profile_json,
                        pinned_items_json = excluded.pinned_items_json,
                        blocked_items_json = excluded.blocked_items_json,
                        deleted_items_json = excluded.deleted_items_json,
                        source = excluded.source,
                        updated_at = CURRENT_TIMESTAMP
                    ''',
                    (
                        user_id,
                        self._serialize_json_field(normalized),
                        self._serialize_json_field([]),
                        self._serialize_json_field([]),
                        self._serialize_json_field([]),
                        "legacy_manual_upsert",
                    ),
                )
                effective = self._refresh_effective_profile(conn, user_id)
                conn.commit()
        except Exception as e:
            logger.error(f"Error upserting research profile: {str(e)}")

        return self.get_user_research_profile(user_id)

    def patch_user_research_profile(self, user_id: str = DEFAULT_USER_ID, profile: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        current = self.get_user_manual_profile(user_id)
        merged = {**current, **dict(profile or {})}
        return self.upsert_user_research_profile(user_id=user_id, profile=merged)

    def get_user_research_profile(self, user_id: str = DEFAULT_USER_ID) -> Dict[str, Any]:
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    '''
                    SELECT profile_json, created_at, updated_at
                    FROM user_effective_profiles WHERE user_id = ?
                    ''',
                    (user_id,),
                )
                row = cursor.fetchone()
                if row:
                    profile = self._normalize_profile_projection(user_id, self._deserialize_json_field(row[0]) or {})
                    profile["created_at"] = row[1]
                    profile["updated_at"] = row[2]
                    return profile

                cursor.execute(
                    '''
                    SELECT user_id, positive_topics, negative_topics, recent_topics, preferred_categories,
                           preferred_answer_style, common_question_types, representative_papers, created_at, updated_at
                    FROM user_research_profiles WHERE user_id = ?
                    ''',
                    (user_id,),
                )
                legacy_row = cursor.fetchone()
                if legacy_row:
                    legacy_profile = {
                        "user_id": legacy_row[0],
                        "positive_topics": self._deserialize_json_field(legacy_row[1]) or [],
                        "negative_topics": self._deserialize_json_field(legacy_row[2]) or [],
                        "recent_topics": self._deserialize_json_field(legacy_row[3]) or [],
                        "preferred_categories": self._deserialize_json_field(legacy_row[4]) or [],
                        "preferred_answer_style": str(legacy_row[5] or ""),
                        "common_question_types": self._deserialize_json_field(legacy_row[6]) or [],
                        "representative_papers": self._deserialize_json_field(legacy_row[7]) or [],
                        "created_at": legacy_row[8],
                        "updated_at": legacy_row[9],
                    }
                    return self._normalize_profile_projection(user_id, legacy_profile)
                return self._empty_user_research_profile(user_id)
        except Exception as e:
            logger.error(f"Error getting research profile: {str(e)}")
            return self._empty_user_research_profile(user_id)

    def get_user_manual_profile(self, user_id: str = DEFAULT_USER_ID) -> Dict[str, Any]:
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT profile_json, created_at, updated_at FROM user_manual_profiles WHERE user_id = ?",
                    (user_id,),
                )
                row = cursor.fetchone()
                if not row:
                    return self._empty_profile_projection(user_id)
                profile = self._normalize_profile_projection(user_id, self._deserialize_json_field(row[0]) or {})
                profile["created_at"] = row[1]
                profile["updated_at"] = row[2]
                return profile
        except Exception as e:
            logger.error(f"Error getting manual research profile: {str(e)}")
            return self._empty_profile_projection(user_id)

    def get_user_generated_profile(self, user_id: str = DEFAULT_USER_ID) -> Dict[str, Any]:
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    '''
                    SELECT profile_json, snapshot_id, evidence_summary_json, quality_report_json,
                           build_config_json, extractor_version, normalizer_version, profile_build_version,
                           created_at, updated_at
                    FROM user_generated_profiles WHERE user_id = ?
                    ''',
                    (user_id,),
                )
                row = cursor.fetchone()
                if not row:
                    return self._empty_profile_projection(user_id)
                profile = self._normalize_profile_projection(user_id, self._deserialize_json_field(row[0]) or {})
                profile.update(
                    {
                        "snapshot_id": row[1],
                        "evidence_summary": self._deserialize_json_field(row[2]) or {},
                        "quality_report": self._deserialize_json_field(row[3]) or {},
                        "build_config": self._deserialize_json_field(row[4]) or {},
                        "extractor_version": row[5],
                        "normalizer_version": row[6],
                        "profile_build_version": row[7],
                        "created_at": row[8],
                        "updated_at": row[9],
                    }
                )
                return profile
        except Exception as e:
            logger.error(f"Error getting generated research profile: {str(e)}")
            return self._empty_profile_projection(user_id)

    def get_user_profile_layers(self, user_id: str = DEFAULT_USER_ID) -> Dict[str, Any]:
        return {
            "manual_profile": self.get_user_manual_profile(user_id),
            "generated_profile": self.get_user_generated_profile(user_id),
            "effective_profile": self.get_user_research_profile(user_id),
        }

    def list_user_profile_snapshots(self, user_id: str = DEFAULT_USER_ID, limit: int = 20) -> List[Dict[str, Any]]:
        """列出画像快照摘要，避免前端列表一次性拉取完整 profile payload。"""
        try:
            active_snapshot_id = (self.get_user_generated_profile(user_id) or {}).get("snapshot_id")
            with self._get_connection() as conn:
                rows = conn.execute(
                    '''
                    SELECT snapshot_id, user_id, evidence_summary_json, quality_report_json, build_config_json,
                           extractor_version, normalizer_version, profile_build_version, created_at
                    FROM user_profile_snapshots
                    WHERE user_id = ?
                    ORDER BY created_at DESC
                    LIMIT ?
                    ''',
                    (user_id, max(1, int(limit or 20))),
                ).fetchall()
            return [
                {
                    "snapshot_id": row[0],
                    "user_id": row[1],
                    "evidence_summary": self._deserialize_json_field(row[2]) or {},
                    "quality_report": self._deserialize_json_field(row[3]) or {},
                    "build_config": self._deserialize_json_field(row[4]) or {},
                    "extractor_version": row[5],
                    "normalizer_version": row[6],
                    "profile_build_version": row[7],
                    "created_at": row[8],
                    "active": row[0] == active_snapshot_id,
                }
                for row in rows
            ]
        except Exception as e:
            logger.error(f"Error listing profile snapshots: {str(e)}")
            return []

    def get_user_profile_snapshot(self, snapshot_id: str) -> Optional[Dict[str, Any]]:
        """读取完整画像快照，用于证据解释、回滚前预览和问题排查。"""
        try:
            with self._get_connection() as conn:
                row = conn.execute(
                    '''
                    SELECT snapshot_id, user_id, generated_profile_json, manual_profile_json, effective_profile_json,
                           evidence_summary_json, quality_report_json, build_config_json,
                           extractor_version, normalizer_version, profile_build_version, created_at
                    FROM user_profile_snapshots
                    WHERE snapshot_id = ?
                    ''',
                    (snapshot_id,),
                ).fetchone()
            if not row:
                return None
            return {
                "snapshot_id": row[0],
                "user_id": row[1],
                "generated_profile": self._normalize_profile_projection(row[1], self._deserialize_json_field(row[2]) or {}),
                "manual_profile": self._normalize_profile_projection(row[1], self._deserialize_json_field(row[3]) or {}),
                "effective_profile": self._normalize_profile_projection(row[1], self._deserialize_json_field(row[4]) or {}),
                "evidence_summary": self._deserialize_json_field(row[5]) or {},
                "quality_report": self._deserialize_json_field(row[6]) or {},
                "build_config": self._deserialize_json_field(row[7]) or {},
                "extractor_version": row[8],
                "normalizer_version": row[9],
                "profile_build_version": row[10],
                "created_at": row[11],
            }
        except Exception as e:
            logger.error(f"Error getting profile snapshot: {str(e)}")
            return None

    def activate_user_profile_snapshot(self, user_id: str, snapshot_id: str) -> Dict[str, Any]:
        """把历史 snapshot 切换为 active generated profile，并重新合并 effective profile。"""
        snapshot = self.get_user_profile_snapshot(snapshot_id)
        if not snapshot or snapshot.get("user_id") != user_id:
            raise ValueError("profile_snapshot_not_found")
        generated = self._normalize_profile_projection(user_id, snapshot.get("generated_profile") or {})
        try:
            with self._get_connection() as conn:
                conn.execute(
                    '''
                    INSERT INTO user_generated_profiles (
                        user_id, snapshot_id, profile_json, evidence_summary_json, quality_report_json, build_config_json,
                        extractor_version, normalizer_version, profile_build_version, updated_at
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                    ON CONFLICT(user_id) DO UPDATE SET
                        snapshot_id = excluded.snapshot_id,
                        profile_json = excluded.profile_json,
                        evidence_summary_json = excluded.evidence_summary_json,
                        quality_report_json = excluded.quality_report_json,
                        build_config_json = excluded.build_config_json,
                        extractor_version = excluded.extractor_version,
                        normalizer_version = excluded.normalizer_version,
                        profile_build_version = excluded.profile_build_version,
                        updated_at = CURRENT_TIMESTAMP
                    ''',
                    (
                        user_id,
                        snapshot_id,
                        self._serialize_json_field(generated),
                        self._serialize_json_field(snapshot.get("evidence_summary") or {}),
                        self._serialize_json_field(snapshot.get("quality_report") or {}),
                        self._serialize_json_field(snapshot.get("build_config") or {}),
                        snapshot.get("extractor_version") or PROFILE_EXTRACTOR_VERSION,
                        snapshot.get("normalizer_version") or PROFILE_NORMALIZER_VERSION,
                        snapshot.get("profile_build_version") or PROFILE_BUILD_VERSION,
                    ),
                )
                effective = self._refresh_effective_profile(conn, user_id, snapshot_id=snapshot_id)
                conn.commit()
            return self._normalize_profile_projection(user_id, effective)
        except Exception as e:
            logger.error(f"Error activating profile snapshot: {str(e)}")
            raise

    def _refresh_effective_profile(self, conn, user_id: str, snapshot_id: Optional[str] = None) -> Dict[str, Any]:
        cursor = conn.cursor()
        cursor.execute("SELECT profile_json FROM user_manual_profiles WHERE user_id = ?", (user_id,))
        manual_row = cursor.fetchone()
        cursor.execute("SELECT profile_json, snapshot_id FROM user_generated_profiles WHERE user_id = ?", (user_id,))
        generated_row = cursor.fetchone()
        manual = self._deserialize_json_field(manual_row[0]) if manual_row else {}
        generated = self._deserialize_json_field(generated_row[0]) if generated_row else {}
        resolved_snapshot_id = snapshot_id or (generated_row[1] if generated_row else None)
        effective = self._merge_profile_projection(user_id, generated, manual)
        merge_report = {
            "manual_available": bool(manual),
            "generated_available": bool(generated),
            "generated_snapshot_id": resolved_snapshot_id,
        }
        cursor.execute(
            '''
            INSERT INTO user_effective_profiles (user_id, profile_json, generated_snapshot_id, merge_report_json, updated_at)
            VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP)
            ON CONFLICT(user_id) DO UPDATE SET
                profile_json = excluded.profile_json,
                generated_snapshot_id = excluded.generated_snapshot_id,
                merge_report_json = excluded.merge_report_json,
                updated_at = CURRENT_TIMESTAMP
            ''',
            (
                user_id,
                self._serialize_json_field(effective),
                resolved_snapshot_id,
                self._serialize_json_field(merge_report),
            ),
        )
        self._upsert_legacy_research_profile_cache(conn, user_id, effective)
        return effective

    def upsert_user_manual_profile(
        self,
        user_id: str = DEFAULT_USER_ID,
        profile: Optional[Dict[str, Any]] = None,
        *,
        source: str = "manual",
    ) -> Dict[str, Any]:
        previous_manual = self.get_user_manual_profile(user_id)
        normalized = self._normalize_profile_projection(user_id, profile)
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                # 鎵嬪姩鐢诲儚鏄敤鎴锋樉寮忔剰鍥剧殑鍞竴鍐欏叆杈圭晫锛屽悗缁噸寤轰笉浼氫慨鏀硅繖寮犺〃銆?
                cursor.execute(
                    '''
                    INSERT INTO user_manual_profiles (
                        user_id, profile_json, pinned_items_json, blocked_items_json, deleted_items_json, source, updated_at
                    )
                    VALUES (?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                    ON CONFLICT(user_id) DO UPDATE SET
                        profile_json = excluded.profile_json,
                        source = excluded.source,
                        updated_at = CURRENT_TIMESTAMP
                    ''',
                    (
                        user_id,
                        self._serialize_json_field(normalized),
                        self._serialize_json_field(normalized.get("pinned_topics") or []),
                        self._serialize_json_field(normalized.get("hidden_topics") or []),
                        self._serialize_json_field([]),
                        source,
                    ),
                )
                self._record_manual_profile_delta_events(
                    conn=conn,
                    user_id=user_id,
                    previous_profile=previous_manual,
                    next_profile=normalized,
                    source=source,
                )
                effective = self._refresh_effective_profile(conn, user_id)
                conn.commit()
                return self._normalize_profile_projection(user_id, effective)
        except Exception as e:
            logger.error(f"Error upserting manual research profile: {str(e)}")
            return self.get_user_research_profile(user_id)

    def patch_user_manual_profile(
        self,
        user_id: str = DEFAULT_USER_ID,
        profile: Optional[Dict[str, Any]] = None,
        *,
        source: str = "manual_patch",
    ) -> Dict[str, Any]:
        current = self.get_user_manual_profile(user_id)
        merged = {**current, **dict(profile or {})}
        return self.upsert_user_manual_profile(user_id=user_id, profile=merged, source=source)

    def _record_manual_profile_delta_events(
        self,
        *,
        conn,
        user_id: str,
        previous_profile: Dict[str, Any],
        next_profile: Dict[str, Any],
        source: str,
    ) -> None:
        tracked_topic_fields = {
            "positive_topics": ("manual_topic_added", "manual_topic_removed"),
            "negative_topics": ("manual_topic_added", "manual_topic_removed"),
            "recent_topics": ("manual_topic_added", "manual_topic_removed"),
            "preferred_categories": ("manual_topic_added", "manual_topic_removed"),
            "pinned_topics": ("manual_topic_pinned", "manual_topic_removed"),
            "hidden_topics": ("manual_topic_hidden", "manual_topic_removed"),
        }
        for field_name, (added_event_type, removed_event_type) in tracked_topic_fields.items():
            previous_values = set(self._normalize_profile_list_value(previous_profile.get(field_name), limit=100))
            next_values = set(self._normalize_profile_list_value(next_profile.get(field_name), limit=100))
            for topic in sorted(next_values - previous_values):
                # 鎵嬪姩鏂板杩涘叆 manual event锛屽悗缁敱 manual/effective 鍚堝苟灞傚鐞嗭紝涓嶆薄鏌?generated profile銆?
                self.record_user_profile_event(
                    user_id=user_id,
                    event_type=added_event_type,
                    source_type="manual_profile",
                    action_type=added_event_type,
                    source=source,
                    metadata={"topic": topic, "field": field_name},
                    include_in_profile=True,
                    conn=conn,
                )
            for topic in sorted(previous_values - next_values):
                self.record_user_profile_event(
                    user_id=user_id,
                    event_type=removed_event_type,
                    source_type="manual_profile",
                    action_type=removed_event_type,
                    source=source,
                    metadata={"topic": topic, "field": field_name},
                    include_in_profile=True,
                    conn=conn,
                )

        previous_style = str(previous_profile.get("preferred_answer_style") or "").strip()
        next_style = str(next_profile.get("preferred_answer_style") or "").strip()
        if previous_style != next_style:
            self.record_user_profile_event(
                user_id=user_id,
                event_type="manual_style_updated",
                source_type="manual_profile",
                action_type="manual_style_updated",
                source=source,
                metadata={"previous_style": previous_style, "style": next_style},
                include_in_profile=True,
                conn=conn,
            )

    def save_generated_profile_snapshot(
        self,
        user_id: str = DEFAULT_USER_ID,
        *,
        generated_profile: Dict[str, Any],
        evidence_summary: Optional[Dict[str, Any]] = None,
        quality_report: Optional[Dict[str, Any]] = None,
        build_config: Optional[Dict[str, Any]] = None,
        job_id: Optional[str] = None,
        activate: bool = True,
    ) -> Dict[str, Any]:
        snapshot_id = str(uuid.uuid4())
        normalized_generated = self._normalize_profile_projection(user_id, generated_profile)
        evidence_summary = dict(evidence_summary or {})
        quality_report = dict(quality_report or {})
        build_config = dict(build_config or {})
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT profile_json FROM user_manual_profiles WHERE user_id = ?", (user_id,))
                manual_row = cursor.fetchone()
                manual_profile = self._deserialize_json_field(manual_row[0]) if manual_row else {}
                if activate:
                    effective_for_snapshot = self._merge_profile_projection(user_id, normalized_generated, manual_profile)
                else:
                    cursor.execute("SELECT profile_json FROM user_effective_profiles WHERE user_id = ?", (user_id,))
                    effective_row = cursor.fetchone()
                    effective_for_snapshot = self._deserialize_json_field(effective_row[0]) if effective_row else self._normalize_profile_projection(user_id, manual_profile)
                # 每次重建都先落 snapshot；低质量结果也可追溯，但只有审查通过才移动 active 指针。
                cursor.execute(
                    '''
                    INSERT INTO user_profile_snapshots (
                        snapshot_id, user_id, generated_profile_json, manual_profile_json, effective_profile_json,
                        evidence_summary_json, quality_report_json, build_config_json,
                        extractor_version, normalizer_version, profile_build_version
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ''',
                    (
                        snapshot_id,
                        user_id,
                        self._serialize_json_field(normalized_generated),
                        self._serialize_json_field(manual_profile or {}),
                        self._serialize_json_field(effective_for_snapshot),
                        self._serialize_json_field(evidence_summary),
                        self._serialize_json_field(quality_report),
                        self._serialize_json_field(build_config),
                        PROFILE_EXTRACTOR_VERSION,
                        PROFILE_NORMALIZER_VERSION,
                        PROFILE_BUILD_VERSION,
                    ),
                )
                if activate:
                    cursor.execute(
                        '''
                        INSERT INTO user_generated_profiles (
                            user_id, snapshot_id, profile_json, evidence_summary_json, quality_report_json, build_config_json,
                            extractor_version, normalizer_version, profile_build_version, updated_at
                        )
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                        ON CONFLICT(user_id) DO UPDATE SET
                            snapshot_id = excluded.snapshot_id,
                            profile_json = excluded.profile_json,
                            evidence_summary_json = excluded.evidence_summary_json,
                            quality_report_json = excluded.quality_report_json,
                            build_config_json = excluded.build_config_json,
                            extractor_version = excluded.extractor_version,
                            normalizer_version = excluded.normalizer_version,
                            profile_build_version = excluded.profile_build_version,
                            updated_at = CURRENT_TIMESTAMP
                        ''',
                        (
                            user_id,
                            snapshot_id,
                            self._serialize_json_field(normalized_generated),
                            self._serialize_json_field(evidence_summary),
                            self._serialize_json_field(quality_report),
                            self._serialize_json_field(build_config),
                            PROFILE_EXTRACTOR_VERSION,
                            PROFILE_NORMALIZER_VERSION,
                            PROFILE_BUILD_VERSION,
                        ),
                    )
                    effective = self._refresh_effective_profile(conn, user_id, snapshot_id=snapshot_id)
                else:
                    # 质量审查失败时只保留可追溯 snapshot，不移动 active 指针，避免低质量画像污染推荐和 Agent。
                    effective = effective_for_snapshot
                if job_id:
                    next_status = "completed" if activate else "needs_review"
                    next_stage = "completed" if activate else "needs_review"
                    cursor.execute(
                        '''
                        UPDATE user_profile_build_jobs
                        SET status = ?, snapshot_id = ?, current_stage = ?, progress = 100, updated_at = CURRENT_TIMESTAMP
                        WHERE job_id = ?
                        ''',
                        (next_status, snapshot_id, next_stage, job_id),
                    )
                conn.commit()
                return {
                    "snapshot_id": snapshot_id,
                    "generated_profile": normalized_generated,
                    "manual_profile": self._normalize_profile_projection(user_id, manual_profile),
                    "effective_profile": self._normalize_profile_projection(user_id, effective),
                    "evidence_summary": evidence_summary,
                    "quality_report": quality_report,
                    "build_config": build_config,
                    "extractor_version": PROFILE_EXTRACTOR_VERSION,
                    "normalizer_version": PROFILE_NORMALIZER_VERSION,
                    "profile_build_version": PROFILE_BUILD_VERSION,
                }
        except Exception as e:
            logger.error(f"Error saving generated profile snapshot: {str(e)}")
            if job_id:
                self.update_user_profile_build_job(job_id, status="failed", current_stage="failed", error_message=str(e))
            return {
                "snapshot_id": None,
                "generated_profile": normalized_generated,
                "effective_profile": self.get_user_research_profile(user_id),
                "evidence_summary": evidence_summary,
                "quality_report": {"error": str(e), **quality_report},
                "build_config": build_config,
            }

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
                           active_index_version, active_build_id, previous_build_id,
                           retrieval_index_file, retrieval_index_count, retrieval_index_types, retrieval_index_version,
                           sparse_index_dir, sparse_index_manifest_file, sparse_index_document_count, sparse_index_token_count, sparse_index_backend,
                           sparse_index_schema_version, sparse_index_source_file, sparse_index_source_hash, sparse_index_avgdl
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
                        'retrieval_index_file': row[20],
                        'retrieval_index_count': row[21] or 0,
                        'retrieval_index_types': row[22],
                        'retrieval_index_version': row[23],
                        'sparse_index_dir': row[24],
                        'sparse_index_manifest_file': row[25],
                        'sparse_index_document_count': row[26] or 0,
                        'sparse_index_token_count': row[27] or 0,
                        'sparse_index_backend': row[28],
                        'sparse_index_schema_version': row[29],
                        'sparse_index_source_file': row[30],
                        'sparse_index_source_hash': row[31],
                        'sparse_index_avgdl': row[32] or 0,
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
            "retrieval_index_file": row[21] if len(row) > 21 else "",
            "retrieval_index_count": (row[22] if len(row) > 22 else 0) or 0,
            "retrieval_index_types": row[23] if len(row) > 23 else "",
            "retrieval_index_version": row[24] if len(row) > 24 else "",
            "sparse_index_dir": row[25] if len(row) > 25 else "",
            "sparse_index_manifest_file": row[26] if len(row) > 26 else "",
            "sparse_index_document_count": (row[27] if len(row) > 27 else 0) or 0,
            "sparse_index_token_count": (row[28] if len(row) > 28 else 0) or 0,
            "sparse_index_backend": row[29] if len(row) > 29 else "",
            "sparse_index_schema_version": row[30] if len(row) > 30 else "",
            "sparse_index_source_file": row[31] if len(row) > 31 else "",
            "sparse_index_source_hash": row[32] if len(row) > 32 else "",
            "sparse_index_avgdl": (row[33] if len(row) > 33 else 0) or 0,
        }

    @staticmethod
    def _paper_qa_index_version_select_sql() -> str:
        return """
            SELECT build_id, arxiv_id, index_version, status, is_active, collection_name,
                   chunk_count, embedding_model, pdf_path, chunk_file, embedding_file,
                   loading_method, chunking_strategy, current_stage, failed_stage,
                   error_message, artifact_status, indexed_at, activated_at, created_at, updated_at,
                   retrieval_index_file, retrieval_index_count, retrieval_index_types, retrieval_index_version,
                   sparse_index_dir, sparse_index_manifest_file, sparse_index_document_count, sparse_index_token_count, sparse_index_backend,
                   sparse_index_schema_version, sparse_index_source_file, sparse_index_source_hash, sparse_index_avgdl
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
                # building version 鍙褰曟柊鏋勫缓鐨勪复鏃剁姸鎬侊紝涓嶈鐩?paper_qa_index 涓粛鍦ㄧ嚎鐨?active 鎸囬拡銆?
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
                "retrieval_index_file",
                "retrieval_index_count",
                "retrieval_index_types",
                "retrieval_index_version",
                "sparse_index_dir",
                "sparse_index_manifest_file",
                "sparse_index_document_count",
                "sparse_index_token_count",
                "sparse_index_backend",
                "sparse_index_schema_version",
                "sparse_index_source_file",
                "sparse_index_source_hash",
                "sparse_index_avgdl",
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
                        # 鍙湁宸茬粡瀹屾垚鍚戦噺鍐欏叆骞舵牎楠岃繃鐨勬柊鐗堟湰鎵嶈兘鍒?active锛岄伩鍏嶅崐鎴愬搧琚棶绛旈摼璺鍒般€?
                        conn.rollback()
                        return False
                    required_sparse_fields = (
                        "sparse_index_manifest_file",
                        "sparse_index_source_hash",
                        "sparse_index_backend",
                        "sparse_index_schema_version",
                    )
                    if (
                        any(not str(build.get(field_name) or "").strip() for field_name in required_sparse_fields)
                        or int(build.get("sparse_index_document_count") or 0) <= 0
                    ):
                        # active 指针必须同时拥有 dense collection 和 sparse manifest；否则重启后 keyword route 会读到不完整版本。
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
                        # 鏃?active 涓嶅湪婵€娲讳簨鍔￠噷鍒犻櫎锛屽彧鏍囪涓?cleanup_pending锛岀粰鍥炴粴鍜屽欢杩熸竻鐞嗙暀鍑虹┖闂淬€?
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
                            pdf_path, chunk_file,
                            retrieval_index_file, retrieval_index_count, retrieval_index_types, retrieval_index_version,
                            sparse_index_dir, sparse_index_manifest_file, sparse_index_document_count, sparse_index_token_count, sparse_index_backend,
                            sparse_index_schema_version, sparse_index_source_file, sparse_index_source_hash, sparse_index_avgdl,
                            embedding_file, loading_method, chunking_strategy,
                            current_stage, failed_stage, error_message, artifact_status, indexed_at,
                            active_index_version, active_build_id, previous_build_id
                        )
                        VALUES (?, ?, 'indexed', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'activate_index', '', '', 'active', ?, ?, ?, ?)
                        ON CONFLICT(arxiv_id) DO UPDATE SET
                            collection_name = excluded.collection_name,
                            status = excluded.status,
                            chunk_count = excluded.chunk_count,
                            embedding_model = excluded.embedding_model,
                            pdf_path = excluded.pdf_path,
                            chunk_file = excluded.chunk_file,
                            retrieval_index_file = excluded.retrieval_index_file,
                            retrieval_index_count = excluded.retrieval_index_count,
                            retrieval_index_types = excluded.retrieval_index_types,
                            retrieval_index_version = excluded.retrieval_index_version,
                            sparse_index_dir = excluded.sparse_index_dir,
                            sparse_index_manifest_file = excluded.sparse_index_manifest_file,
                            sparse_index_document_count = excluded.sparse_index_document_count,
                            sparse_index_token_count = excluded.sparse_index_token_count,
                            sparse_index_backend = excluded.sparse_index_backend,
                            sparse_index_schema_version = excluded.sparse_index_schema_version,
                            sparse_index_source_file = excluded.sparse_index_source_file,
                            sparse_index_source_hash = excluded.sparse_index_source_hash,
                            sparse_index_avgdl = excluded.sparse_index_avgdl,
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
                            build.get("retrieval_index_file"),
                            build.get("retrieval_index_count") or 0,
                            build.get("retrieval_index_types"),
                            build.get("retrieval_index_version"),
                            build.get("sparse_index_dir"),
                            build.get("sparse_index_manifest_file"),
                            build.get("sparse_index_document_count") or 0,
                            build.get("sparse_index_token_count") or 0,
                            build.get("sparse_index_backend"),
                            build.get("sparse_index_schema_version"),
                            build.get("sparse_index_source_file"),
                            build.get("sparse_index_source_hash"),
                            build.get("sparse_index_avgdl") or 0,
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
                    "SELECT COUNT(*) FROM paper_qa_index_versions WHERE " + " AND ".join(where_parts),
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
        """在数据库写事务内领取 QA 索引任务，保证同一论文只产生一个 active job。"""
        job_id = str(uuid.uuid4())
        normalized_timeout = max(1, int(timeout_seconds or 1))
        idempotency_key = self._build_paper_index_job_idempotency_key(arxiv_id, loading_method)
        active_statuses = tuple(PAPER_INDEX_ACTIVE_JOB_STATUSES)
        placeholders = ",".join("?" for _ in active_statuses)
        stale_modifier = f"-{normalized_timeout} seconds"
        try:
            with self._get_connection() as conn:
                # BEGIN IMMEDIATE 浼氭彁鍓嶈幏鍙栧啓閿侊紱骞跺彂鎻愪氦浼氭帓闃燂紝鍚庢潵鐨勮姹傝兘澶嶇敤鍏堟彁浜ょ殑 active job銆?
                conn.isolation_level = None
                cursor = conn.cursor()
                cursor.execute("BEGIN IMMEDIATE")
                try:
                    cursor.execute(
                        "SELECT job_id FROM paper_index_jobs "
                        "WHERE arxiv_id = ? "
                        f"AND status IN ({placeholders}) "
                        "AND datetime(COALESCE(heartbeat_at, updated_at, created_at)) <= datetime('now', ?) "
                        "ORDER BY updated_at DESC, created_at DESC",
                        (arxiv_id, *active_statuses, stale_modifier),
                    )
                    stale_job_ids = [row[0] for row in cursor.fetchall()]
                    previous_job_id = stale_job_ids[0] if stale_job_ids else None
                    if stale_job_ids:
                        stale_placeholders = ",".join("?" for _ in stale_job_ids)
                        # stale 鏄彲閲嶈瘯缁堟€侊紱杩欓噷鏄庣‘鍐欏叆鍘熷洜锛屽墠绔疆璇㈡棫 job 鏃朵笉浼氬啀鐪嬪埌鏃犺В閲婄殑 running銆?
                        cursor.execute(
                            "UPDATE paper_index_jobs SET status = 'stale', current_stage = 'stale', "
                            "error_message = CASE WHEN error_message IS NULL OR error_message = '' THEN ? ELSE error_message END, "
                            "heartbeat_at = CURRENT_TIMESTAMP, updated_at = CURRENT_TIMESTAMP "
                            f"WHERE job_id IN ({stale_placeholders})",
                            (
                                f"QA index job heartbeat timed out after {normalized_timeout} seconds; submit again to retry.",
                                *stale_job_ids,
                            ),
                        )

                    cursor.execute(
                        "SELECT job_id, arxiv_id, status, current_stage, progress, error_message, "
                        "loading_method, created_at, updated_at, heartbeat_at, idempotency_key "
                        "FROM paper_index_jobs WHERE arxiv_id = ? "
                        f"AND status IN ({placeholders}) "
                        "ORDER BY updated_at DESC, created_at DESC LIMIT 1",
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
                        "INSERT INTO paper_index_jobs (job_id, arxiv_id, status, current_stage, progress, "
                        "error_message, loading_method, idempotency_key, heartbeat_at) "
                        "VALUES (?, ?, 'pending', 'pending', 0, NULL, ?, ?, CURRENT_TIMESTAMP)",
                        (job_id, arxiv_id, loading_method, idempotency_key),
                    )
                    cursor.execute(
                        "SELECT job_id, arxiv_id, status, current_stage, progress, error_message, "
                        "loading_method, created_at, updated_at, heartbeat_at, idempotency_key "
                        "FROM paper_index_jobs WHERE job_id = ?",
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
        """把超过心跳阈值的 pending/running 任务标记为 stale。"""
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
                # 鏌ヨ鎺ュ彛涔熶細璋冪敤鏈柟娉曪紝鍥犳閿欒淇℃伅瑕佽冻澶熸槑纭紝鏂逛究鍓嶇灞曠ず鏃т换鍔″凡鍙噸璇曘€?
                cursor.execute(
                    "UPDATE paper_index_jobs SET status = 'stale', current_stage = 'stale', "
                    "error_message = CASE WHEN error_message IS NULL OR error_message = '' THEN ? ELSE error_message END, "
                    "heartbeat_at = CURRENT_TIMESTAMP, updated_at = CURRENT_TIMESTAMP WHERE " + " AND ".join(filters),
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
                    # 浠讳綍鐘舵€佹帹杩涢兘浠ｈ〃鍚庡彴绾跨▼浠嶆椿璺冿紝鍚屾鍒锋柊蹇冭烦鐢ㄤ簬鍚庣画 stale 鍒ゅ畾銆?
                    update_fields.append('heartbeat_at = CURRENT_TIMESTAMP')
                update_fields.append('updated_at = CURRENT_TIMESTAMP')
                update_values.append(job_id)
                expected_status_values = [str(item) for item in (expected_statuses or []) if str(item).strip()]
                expected_clause = ""
                if expected_status_values:
                    # 鍚庡彴绾跨▼鍙兘鍦?stale 鎭㈠鍚庢墠缁х画鍥炲啓锛涙潯浠舵洿鏂拌兘闃绘鏃х嚎绋嬪娲讳笉鍙仮澶嶄换鍔°€?
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
                    'retrieval_index_file',
                    'retrieval_index_count',
                    'retrieval_index_types',
                    'retrieval_index_version',
                    'sparse_index_dir',
                    'sparse_index_manifest_file',
                    'sparse_index_document_count',
                    'sparse_index_token_count',
                    'sparse_index_backend',
                    'sparse_index_schema_version',
                    'sparse_index_source_file',
                    'sparse_index_source_hash',
                    'sparse_index_avgdl',
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
                               active_index_version, active_build_id,
                               retrieval_index_file, retrieval_index_count, retrieval_index_types, retrieval_index_version,
                               sparse_index_dir, sparse_index_manifest_file, sparse_index_document_count, sparse_index_token_count, sparse_index_backend,
                               sparse_index_schema_version, sparse_index_source_file, sparse_index_source_hash, sparse_index_avgdl
                        FROM paper_qa_index
                        WHERE arxiv_id = ?
                        """,
                        (arxiv_id,),
                    )
                    active_row = cursor.fetchone()
                    if active_row and str(active_row[1] or "").strip():
                        legacy_version = active_row[15] or "legacy"
                        legacy_build_id = active_row[16] or f"legacy-{str(arxiv_id).replace('.', '_').replace('/', '_')}"
                        # 鏃у紡 update 鎴愬姛鍚庝篃琛?active version锛屼繚璇佺増鏈寲璇诲彇鍜屽洖婊氫俊鎭畬鏁淬€?
                        cursor.execute(
                            """
                            INSERT OR IGNORE INTO paper_qa_index_versions (
                                build_id, arxiv_id, index_version, status, is_active, collection_name,
                                chunk_count, embedding_model, pdf_path, chunk_file,
                                retrieval_index_file, retrieval_index_count, retrieval_index_types, retrieval_index_version,
                                sparse_index_dir, sparse_index_manifest_file, sparse_index_document_count, sparse_index_token_count, sparse_index_backend,
                                sparse_index_schema_version, sparse_index_source_file, sparse_index_source_hash, sparse_index_avgdl,
                                embedding_file,
                                loading_method, chunking_strategy, current_stage, failed_stage,
                                error_message, artifact_status, indexed_at, activated_at
                            )
                            VALUES (?, ?, ?, 'active', 1, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, '', '', ?, ?, CURRENT_TIMESTAMP)
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
                                active_row[17],
                                active_row[18] or 0,
                                active_row[19],
                                active_row[20],
                                active_row[21],
                                active_row[22],
                                active_row[23] or 0,
                                active_row[24],
                                active_row[25],
                                active_row[26],
                                active_row[27],
                                active_row[28],
                                active_row[29] or 0,
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
                    'retrieval_index_file',
                    'retrieval_index_count',
                    'retrieval_index_types',
                    'retrieval_index_version',
                    'sparse_index_dir',
                    'sparse_index_manifest_file',
                    'sparse_index_document_count',
                    'sparse_index_token_count',
                    'sparse_index_backend',
                    'sparse_index_schema_version',
                    'sparse_index_source_file',
                    'sparse_index_source_hash',
                    'sparse_index_avgdl',
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
                    # 鍏煎娴嬭瘯鍜屾棫璋冪敤锛氱洿鎺ュ啓鍏?paper_qa_index 鐨勫彲鐢ㄨ褰曚篃琛ユ垚 active version銆?
                    cursor.execute(
                        """
                        INSERT OR IGNORE INTO paper_qa_index_versions (
                            build_id, arxiv_id, index_version, status, is_active, collection_name,
                            chunk_count, embedding_model, pdf_path, chunk_file,
                            retrieval_index_file, retrieval_index_count, retrieval_index_types, retrieval_index_version,
                            sparse_index_dir, sparse_index_manifest_file, sparse_index_document_count, sparse_index_token_count, sparse_index_backend,
                            sparse_index_schema_version, sparse_index_source_file, sparse_index_source_hash, sparse_index_avgdl,
                            embedding_file,
                            loading_method, chunking_strategy, current_stage, failed_stage,
                            error_message, artifact_status, indexed_at, activated_at
                        )
                        VALUES (?, ?, ?, 'active', 1, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, '', '', ?, ?, CURRENT_TIMESTAMP)
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
                            kwargs.get("retrieval_index_file"),
                            kwargs.get("retrieval_index_count") or 0,
                            kwargs.get("retrieval_index_types"),
                            kwargs.get("retrieval_index_version"),
                            kwargs.get("sparse_index_dir"),
                            kwargs.get("sparse_index_manifest_file"),
                            kwargs.get("sparse_index_document_count") or 0,
                            kwargs.get("sparse_index_token_count") or 0,
                            kwargs.get("sparse_index_backend"),
                            kwargs.get("sparse_index_schema_version"),
                            kwargs.get("sparse_index_source_file"),
                            kwargs.get("sparse_index_source_hash"),
                            kwargs.get("sparse_index_avgdl") or 0,
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

    def _runtime_checkpoint_approved_step_ids(self, raw_runtime_state: Any) -> List[str]:
        payload = self._deserialize_json_field(raw_runtime_state)
        if not isinstance(payload, dict):
            return []
        raw_step_ids = payload.get("approved_step_ids")
        if not isinstance(raw_step_ids, (list, tuple, set)):
            return []
        approved_step_ids: List[str] = []
        for item in raw_step_ids:
            step_id = str(item).strip()
            if step_id and step_id not in approved_step_ids:
                approved_step_ids.append(step_id)
        return approved_step_ids

    def _pending_confirmation_step_id(self, raw_pending_confirmation: Any) -> str:
        pending_confirmation = self._deserialize_json_field(raw_pending_confirmation)
        if not isinstance(pending_confirmation, dict):
            return ""
        return str(pending_confirmation.get("step_id") or "").strip()

    def _runtime_state_pending_confirmation_step_id(self, raw_runtime_state: Any) -> str:
        runtime_state = self._deserialize_json_field(raw_runtime_state)
        if not isinstance(runtime_state, dict):
            return ""
        return self._pending_confirmation_step_id(runtime_state.get("pending_confirmation"))

    def _runtime_state_plan_id(self, raw_runtime_state: Any) -> str:
        runtime_state = self._deserialize_json_field(raw_runtime_state)
        if not isinstance(runtime_state, dict):
            return ""
        plan = runtime_state.get("plan") if isinstance(runtime_state.get("plan"), dict) else {}
        return str(plan.get("plan_id") or runtime_state.get("plan_id") or "").strip()

    def _runtime_checkpoint_plan_matches(self, existing_runtime_state: Any, incoming_runtime_state: Any) -> bool:
        existing_plan_id = self._runtime_state_plan_id(existing_runtime_state)
        incoming_plan_id = self._runtime_state_plan_id(incoming_runtime_state)
        # 历史 checkpoint 或单元测试可能没有 plan_id；只有明确不一致时才关闭 replay 保护。
        return not existing_plan_id or not incoming_plan_id or existing_plan_id == incoming_plan_id

    def _is_consumed_agent_runtime_confirmation_replay(
        self,
        *,
        existing_checkpoint: Optional[Dict[str, Any]],
        incoming_runtime_state: Any,
        incoming_pending_confirmation: Any,
        incoming_status: str,
    ) -> bool:
        if not isinstance(existing_checkpoint, dict):
            return False
        if str(existing_checkpoint.get("status") or "").strip() != "running":
            return False
        if existing_checkpoint.get("pending_confirmation"):
            return False
        pending_step_id = (
            self._pending_confirmation_step_id(incoming_pending_confirmation)
            or self._runtime_state_pending_confirmation_step_id(incoming_runtime_state)
        )
        if not pending_step_id:
            return False
        incoming_is_waiting = str(incoming_status or "").strip() == "waiting_confirmation"
        incoming_has_pending = bool(self._deserialize_json_field(incoming_pending_confirmation)) or bool(
            self._runtime_state_pending_confirmation_step_id(incoming_runtime_state)
        )
        if not incoming_is_waiting and not incoming_has_pending:
            return False
        existing_runtime_state = existing_checkpoint.get("runtime_state")
        approved_step_ids = set(self._runtime_checkpoint_approved_step_ids(existing_runtime_state))
        if pending_step_id not in approved_step_ids:
            return False
        if not self._runtime_checkpoint_plan_matches(existing_runtime_state, incoming_runtime_state):
            return False
        return True

    def _merge_runtime_checkpoint_approved_step_ids(
        self,
        incoming_runtime_state: Any,
        *,
        existing_checkpoint: Optional[Dict[str, Any]],
    ) -> Any:
        if not isinstance(existing_checkpoint, dict):
            return incoming_runtime_state
        if str(existing_checkpoint.get("status") or "").strip() != "running":
            return incoming_runtime_state
        if existing_checkpoint.get("pending_confirmation"):
            return incoming_runtime_state
        existing_runtime_state = existing_checkpoint.get("runtime_state")
        if not self._runtime_checkpoint_plan_matches(existing_runtime_state, incoming_runtime_state):
            return incoming_runtime_state
        existing_step_ids = self._runtime_checkpoint_approved_step_ids(existing_runtime_state)
        if not existing_step_ids:
            return incoming_runtime_state

        runtime_state = self._deserialize_json_field(incoming_runtime_state)
        if isinstance(runtime_state, dict):
            runtime_state = dict(runtime_state)
        else:
            # 上游 running 快照偶尔缺少 runtime_state；这里复用已消费现场，避免批准态被空快照擦掉。
            existing_payload = self._deserialize_json_field(existing_runtime_state)
            runtime_state = dict(existing_payload) if isinstance(existing_payload, dict) else {}
        incoming_step_ids = self._runtime_checkpoint_approved_step_ids(runtime_state)
        merged_step_ids = list(dict.fromkeys([*existing_step_ids, *incoming_step_ids]))
        if merged_step_ids == incoming_step_ids:
            return incoming_runtime_state
        runtime_state["approved_step_ids"] = merged_step_ids
        return runtime_state

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
            normalized_status = str(status or 'running').strip() or 'running'
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
                existing_row = cursor.fetchone()
                existing_checkpoint = self._row_to_agent_runtime_checkpoint(existing_row) if existing_row else None
                if self._is_consumed_agent_runtime_confirmation_replay(
                    existing_checkpoint=existing_checkpoint,
                    incoming_runtime_state=runtime_state,
                    incoming_pending_confirmation=pending_confirmation,
                    incoming_status=normalized_status,
                ):
                    # LangGraph resume 可能重放中断前 waiting 快照；一旦 DB 已记录批准态，就不能回写旧 pending。
                    logger.debug(
                        "Skip stale agent runtime confirmation replay: session_id=%s thread_id=%s status=%s",
                        normalized_session_id,
                        normalized_thread_id,
                        normalized_status,
                    )
                    return existing_checkpoint
                runtime_state_to_write = self._merge_runtime_checkpoint_approved_step_ids(
                    runtime_state,
                    existing_checkpoint=existing_checkpoint,
                )
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
                        self._serialize_json_field(runtime_state_to_write),
                        self._serialize_json_field(graph_state),
                        self._serialize_json_field(pending_confirmation),
                        str(current_node or '').strip(),
                        str(next_route or '').strip(),
                        normalized_status,
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

    def list_agent_runtime_checkpoints_by_thread(
        self,
        *,
        session_id: str,
        thread_id: Optional[str] = None,
        limit: int = 20,
    ) -> List[Dict[str, Any]]:
        """按 session/thread 列出候选 runtime checkpoint，供恢复态丢失 user_id 时二次校验。"""
        try:
            normalized_session_id = str(session_id or '').strip()
            normalized_thread_id = str(thread_id or normalized_session_id).strip()
            if not normalized_session_id or not normalized_thread_id:
                return []
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    '''
                    SELECT checkpoint_id, user_id, session_id, thread_id, runtime_state_json,
                           graph_state_json, pending_confirmation_json, current_node, next_route,
                           status, error_summary, created_at, updated_at, expires_at
                    FROM agent_runtime_checkpoints
                    WHERE session_id = ? AND thread_id = ?
                    ORDER BY updated_at DESC, created_at DESC
                    LIMIT ?
                    ''',
                    (normalized_session_id, normalized_thread_id, max(int(limit or 1), 1)),
                )
                rows = cursor.fetchall()
                return [self._row_to_agent_runtime_checkpoint(row) for row in rows]
        except Exception as e:
            logger.error(f"Error listing agent runtime checkpoints by thread: {str(e)}")
            return []

    def get_agent_runtime_checkpoint_by_thread(
        self,
        *,
        session_id: str,
        thread_id: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """按 session/thread 找回唯一 runtime checkpoint，兼容旧调用方的单记录接口。"""
        try:
            checkpoints = self.list_agent_runtime_checkpoints_by_thread(
                session_id=session_id,
                thread_id=thread_id,
                limit=2,
            )
            # 旧接口只在原始候选唯一时返回，避免缺失 user_id 时误读其他用户现场。
            if len(checkpoints) != 1:
                return None
            return checkpoints[0]
        except Exception as e:
            logger.error(f"Error getting agent runtime checkpoint by thread: {str(e)}")
            return None

    def _runtime_state_without_pending_confirmation(
        self,
        raw_runtime_state: Any,
        *,
        decision: Optional[str] = None,
        step_id: Optional[str] = None,
    ) -> str:
        """清理 runtime_state_json 内嵌的确认真源，避免查库或后续诊断看到旧 pending。

        关键修复：在批准决策时，必须把 pending_confirmation 中记录的真正目标 step_id 添加到
        approved_step_ids，而不仅仅是传入的 step_id。这对于缺索引补丁链等桥接确认场景至关重要。
        """
        payload = self._deserialize_json_field(raw_runtime_state)
        if not isinstance(payload, dict):
            return self._serialize_json_field(payload)

        previous_pending = payload.get("pending_confirmation") if isinstance(payload.get("pending_confirmation"), dict) else {}
        # 关键修复：优先使用 pending_confirmation 中记录的目标 step_id，而不是 resume_payload 传入的 step_id
        # 因为在桥接确认场景（如 request_confirmation），真正需要批准的是目标工具，而不是桥接步骤本身
        pending_step_id = str(previous_pending.get("step_id") or "").strip()
        requested_step_id = str(step_id or "").strip()
        normalized_step_id = pending_step_id or requested_step_id
        normalized_decision = str(decision or "").strip().lower()
        payload["pending_confirmation"] = None

        # confirmation 已消费后，request_confirmation 恢复策略不再代表可恢复现场；拒绝分支保留 skip 语义。
        recovery_strategy = payload.get("recovery_strategy") if isinstance(payload.get("recovery_strategy"), dict) else {}
        if normalized_decision == "reject":
            payload["recovery_strategy"] = {
                "type": "skip_step",
                "reason": "confirmation_rejected",
                **({"step_id": normalized_step_id} if normalized_step_id else {}),
            }
        elif normalized_decision == "approve" or str(recovery_strategy.get("type") or "").strip() == "request_confirmation":
            payload["recovery_strategy"] = None

        if normalized_decision == "approve" and normalized_step_id:
            approved_step_ids = [str(item).strip() for item in list(payload.get("approved_step_ids") or []) if str(item).strip()]
            if normalized_step_id not in approved_step_ids:
                approved_step_ids.append(normalized_step_id)
            payload["approved_step_ids"] = approved_step_ids

        step_status = payload.get("step_status") if isinstance(payload.get("step_status"), dict) else None
        if step_status is not None and normalized_step_id and step_status.get(normalized_step_id) == "waiting_confirmation":
            step_status[normalized_step_id] = "skipped" if normalized_decision == "reject" else "pending"
            payload["step_status"] = step_status
        if str(payload.get("turn_status") or "").strip() == "waiting_confirmation":
            payload["turn_status"] = None
        return self._serialize_json_field(payload)

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
        """更新执行现场终态或过期状态，避免旧 confirmation 被重复 resume。"""
        try:
            normalized_user_id = str(user_id or DEFAULT_USER_ID).strip() or DEFAULT_USER_ID
            normalized_session_id = str(session_id or '').strip()
            normalized_thread_id = str(thread_id or normalized_session_id).strip()
            if not normalized_session_id or not normalized_thread_id:
                return False
            assignments = ['status = ?', 'error_summary = ?', 'updated_at = CURRENT_TIMESTAMP']
            values: List[Any] = [str(status or '').strip(), str(error_summary or '').strip()]
            with self._get_connection() as conn:
                cursor = conn.cursor()
                if clear_pending_confirmation:
                    cursor.execute(
                        '''
                        SELECT runtime_state_json
                        FROM agent_runtime_checkpoints
                        WHERE user_id = ? AND session_id = ? AND thread_id = ?
                        ''',
                        (normalized_user_id, normalized_session_id, normalized_thread_id),
                    )
                    row = cursor.fetchone()
                    cleaned_runtime_state = self._runtime_state_without_pending_confirmation(row[0] if row else None)
                    assignments.extend(["pending_confirmation_json = ''", "runtime_state_json = ?", "expires_at = NULL"])
                    values.append(cleaned_runtime_state)
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

    def consume_agent_runtime_pending_confirmation(
        self,
        *,
        user_id: str = DEFAULT_USER_ID,
        session_id: str,
        thread_id: Optional[str] = None,
        next_route: str = "running",
        decision: Optional[str] = None,
        step_id: Optional[str] = None,
        tool_name: Optional[str] = None,
        pending_action_id: Optional[str] = None,
    ) -> bool:
        """原子消费等待确认的业务 checkpoint。

        快速连续点击确认时，多个请求可能同时读到同一张确认卡片；这里用
        status='waiting_confirmation' 作为抢占条件，只有第一个请求能清空
        pending_confirmation 并进入 running，后续请求会因为 rowcount=0 被拒绝恢复。
        """
        try:
            normalized_user_id = str(user_id or DEFAULT_USER_ID).strip() or DEFAULT_USER_ID
            normalized_session_id = str(session_id or '').strip()
            normalized_thread_id = str(thread_id or normalized_session_id).strip()
            if not normalized_session_id or not normalized_thread_id:
                return False
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    '''
                    SELECT runtime_state_json, pending_confirmation_json
                    FROM agent_runtime_checkpoints
                    WHERE user_id = ?
                      AND session_id = ?
                      AND thread_id = ?
                      AND status = 'waiting_confirmation'
                      AND COALESCE(NULLIF(pending_confirmation_json, ''), '') <> ''
                    ''',
                    (normalized_user_id, normalized_session_id, normalized_thread_id),
                )
                row = cursor.fetchone()
                if not row:
                    return False
                pending_payload = self._deserialize_json_field(row[1]) if row else None
                pending_step_id = pending_payload.get("step_id") if isinstance(pending_payload, dict) else None
                pending_tool_name = pending_payload.get("tool_name") if isinstance(pending_payload, dict) else None
                stored_pending_action_id = pending_payload.get("pending_action_id") if isinstance(pending_payload, dict) else None
                normalized_step_id = str(step_id or "").strip()
                normalized_tool_name = str(tool_name or "").strip()
                normalized_pending_action_id = str(pending_action_id or "").strip()
                if normalized_step_id and pending_step_id and normalized_step_id != str(pending_step_id).strip():
                    # 消费动作必须绑定当前确认任务本身，避免旧按钮把新的 waiting 任务误消费。
                    return False
                if normalized_tool_name and pending_tool_name and normalized_tool_name != str(pending_tool_name).strip():
                    return False
                if (
                    normalized_pending_action_id
                    and stored_pending_action_id
                    and normalized_pending_action_id != str(stored_pending_action_id).strip()
                ):
                    return False
                cleaned_runtime_state = self._runtime_state_without_pending_confirmation(
                    row[0],
                    decision=decision,
                    step_id=normalized_step_id or pending_step_id,
                )
                cursor.execute(
                    '''
                    UPDATE agent_runtime_checkpoints
                    SET status = 'running',
                        pending_confirmation_json = '',
                        runtime_state_json = ?,
                        next_route = ?,
                        error_summary = '',
                        expires_at = NULL,
                        updated_at = CURRENT_TIMESTAMP
                    WHERE user_id = ?
                      AND session_id = ?
                      AND thread_id = ?
                      AND status = 'waiting_confirmation'
                      AND COALESCE(NULLIF(pending_confirmation_json, ''), '') <> ''
                    ''',
                    (
                        cleaned_runtime_state,
                        str(next_route or 'running').strip() or 'running',
                        normalized_user_id,
                        normalized_session_id,
                        normalized_thread_id,
                    ),
                )
                conn.commit()
                return cursor.rowcount > 0
        except Exception as e:
            logger.error(f"Error consuming agent runtime pending confirmation: {str(e)}")
            return False

    def expire_agent_runtime_checkpoints(self, *, now: Optional[str] = None) -> int:
        """把超过 expires_at 的等待现场标记为 expired，而不是直接删除。"""
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

    def cleanup_langgraph_checkpoints_for_terminal_runtime(self, *, retention_days: int = 7) -> Dict[str, int]:
        """按业务 runtime checkpoint 生命周期同步清理 LangGraph 原始 checkpoint。

        runtime checkpoint 是恢复语义的权威记录；只有它进入 completed/failed/expired 等终态并超过保留期后，
        才删除同 thread_id 下的原始 graph checkpoint 与 writes，避免误删仍可恢复的确认现场。
        """
        result = {"threads": 0, "checkpoints": 0, "writes": 0}
        try:
            cutoff = datetime.now(timezone.utc) - timedelta(days=max(int(retention_days or 0), 1))
            cutoff_text = cutoff.isoformat()
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    '''
                    SELECT DISTINCT thread_id
                    FROM agent_runtime_checkpoints
                    WHERE status IN ('completed', 'cancelled', 'failed', 'expired')
                      AND updated_at <= ?
                    ''',
                    (cutoff_text,),
                )
                thread_ids = [str(row[0] or '').strip() for row in cursor.fetchall() if str(row[0] or '').strip()]
                for thread_id in thread_ids:
                    cursor.execute("DELETE FROM langgraph_checkpoint_writes WHERE thread_id = ?", (thread_id,))
                    result["writes"] += int(cursor.rowcount or 0)
                    cursor.execute("DELETE FROM langgraph_checkpoints WHERE thread_id = ?", (thread_id,))
                    result["checkpoints"] += int(cursor.rowcount or 0)
                conn.commit()
                result["threads"] = len(thread_ids)
            return result
        except Exception as e:
            logger.error(f"Error cleaning langgraph checkpoints: {str(e)}")
            return result

    def get_context_lifecycle_stats(
        self,
        *,
        user_id: str = DEFAULT_USER_ID,
        paper_session_id: Optional[str] = None,
        agent_session_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """汇总上下文相关表的轻量健康度，不读取大字段正文。"""
        normalized_user_id = str(user_id or DEFAULT_USER_ID).strip() or DEFAULT_USER_ID
        stats: Dict[str, Any] = {
            "user_id": normalized_user_id,
            "paper_chat": {},
            "agent_runtime_checkpoint": {},
            "langgraph_checkpoint": {},
        }
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                if paper_session_id:
                    cursor.execute(
                        '''
                        SELECT s.message_count, COUNT(m.message_id), s.summary_updated_at,
                               s.summary_turn_count, s.summary_last_turn_id,
                               LENGTH(COALESCE(s.summary_json, ''))
                        FROM paper_chat_sessions s
                        LEFT JOIN paper_chat_messages m ON m.session_id = s.session_id
                        WHERE s.session_id = ? AND s.user_id = ?
                        GROUP BY s.session_id
                        ''',
                        (paper_session_id, normalized_user_id),
                    )
                    row = cursor.fetchone()
                    stats["paper_chat"] = {
                        "session_id": paper_session_id,
                        "message_count": int((row or [0])[0] or 0) if row else 0,
                        "stored_message_count": int((row or [0, 0])[1] or 0) if row else 0,
                        "summary_loaded": bool(row and row[2]),
                        "summary_updated_at": row[2] if row else None,
                        "summary_turn_count": int(row[3] or 0) if row else 0,
                        "summary_last_turn_id": row[4] if row else "",
                        "summary_chars": int(row[5] or 0) if row else 0,
                    }
                if agent_session_id:
                    cursor.execute(
                        '''
                        SELECT status, current_node, next_route, expires_at, updated_at,
                               LENGTH(COALESCE(runtime_state_json, '')),
                               LENGTH(COALESCE(graph_state_json, '')),
                               LENGTH(COALESCE(pending_confirmation_json, ''))
                        FROM agent_runtime_checkpoints
                        WHERE user_id = ? AND session_id = ? AND thread_id = ?
                        ''',
                        (normalized_user_id, agent_session_id, agent_session_id),
                    )
                    row = cursor.fetchone()
                    stats["agent_runtime_checkpoint"] = {
                        "session_id": agent_session_id,
                        "exists": bool(row),
                        "status": row[0] if row else None,
                        "current_node": row[1] if row else "",
                        "next_route": row[2] if row else "",
                        "expires_at": row[3] if row else None,
                        "updated_at": row[4] if row else None,
                        "runtime_state_chars": int(row[5] or 0) if row else 0,
                        "graph_state_chars": int(row[6] or 0) if row else 0,
                        "pending_confirmation_chars": int(row[7] or 0) if row else 0,
                    }
                    cursor.execute("SELECT COUNT(*) FROM langgraph_checkpoints WHERE thread_id = ?", (agent_session_id,))
                    checkpoint_count = int((cursor.fetchone() or [0])[0] or 0)
                    cursor.execute("SELECT COUNT(*) FROM langgraph_checkpoint_writes WHERE thread_id = ?", (agent_session_id,))
                    write_count = int((cursor.fetchone() or [0])[0] or 0)
                    stats["langgraph_checkpoint"] = {
                        "thread_id": agent_session_id,
                        "checkpoint_count": checkpoint_count,
                        "write_count": write_count,
                    }
            return stats
        except Exception as e:
            logger.error(f"Error getting context lifecycle stats: {str(e)}")
            stats["error"] = str(e)
            return stats

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
