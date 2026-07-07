from services.storage.sqlite.connection import SqliteConnectionProvider
from services.storage.sqlite.shared import DEFAULT_USER_ID, logger


class StorageSchemaMigrator:
    """集中负责 SQLite schema 初始化和旧库补列，业务 store 不再隐式建表。"""

    def __init__(self, connection_provider: SqliteConnectionProvider) -> None:
        self.connection_provider = connection_provider

    def _ensure_user_interest_vector_columns(self, conn):
        # 旧库可能缺少聚类和负反馈字段；启动时补列，保证推荐画像读写兼容历史 SQLite 文件。
        required_columns = {
            "cluster_count": "INTEGER DEFAULT 0",
            "profile_mode": "TEXT DEFAULT 'mean'",
            "interest_clusters": "TEXT",
            "weak_interest_pool": "TEXT",
            "disliked_vector_data": "TEXT",
            "disliked_paper_examples": "TEXT",
            "negative_feedback_stats": "TEXT",
            "negative_feedback_profile": "TEXT",
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
    def _ensure_user_profile_event_columns(self, conn):
        # 画像事件表从辅助审计升级为主证据流；旧库启动时补齐新列，避免手工迁移数据库。
        required_columns = {
            "event_type": "TEXT",
            "action_strength": "REAL DEFAULT 0",
            "source": "TEXT",
            "note_id": "TEXT",
            "session_id": "TEXT",
            "metadata_json": "TEXT",
            "include_in_profile": "INTEGER DEFAULT 1",
            "consumed_by_job_id": "TEXT",
            "consumed_at": "TIMESTAMP",
            "dedupe_key": "TEXT",
            "profile_dirty": "INTEGER DEFAULT 1",
        }
        cursor = conn.cursor()
        cursor.execute("PRAGMA table_info(user_profile_events)")
        existing_columns = {row[1] for row in cursor.fetchall()}
        for column_name, column_definition in required_columns.items():
            if column_name not in existing_columns:
                cursor.execute(
                    f"ALTER TABLE user_profile_events ADD COLUMN {column_name} {column_definition}"
                )
        cursor.execute(
            '''
            UPDATE user_profile_events
            SET event_type = COALESCE(event_type, action_type),
                source = COALESCE(source, source_type),
                metadata_json = COALESCE(metadata_json, payload_json),
                include_in_profile = COALESCE(include_in_profile, 1),
                profile_dirty = COALESCE(profile_dirty, 1)
            WHERE event_type IS NULL OR source IS NULL OR metadata_json IS NULL
            '''
        )
        conn.commit()

    def _ensure_user_profile_build_job_columns(self, conn):
        # build job 是前端轮询的状态源；旧库补齐 metrics_json 后即可承载细粒度进度，不需要破坏现有列结构。
        required_columns = {
            "metrics_json": "TEXT",
        }
        cursor = conn.cursor()
        cursor.execute("PRAGMA table_info(user_profile_build_jobs)")
        existing_columns = {row[1] for row in cursor.fetchall()}
        for column_name, column_definition in required_columns.items():
            if column_name not in existing_columns:
                cursor.execute(
                    f"ALTER TABLE user_profile_build_jobs ADD COLUMN {column_name} {column_definition}"
                )
        conn.commit()

    def _ensure_paper_chat_session_summary_columns(self, conn):
        # 会话摘要是运行时压缩视图，旧库启动时补列；完整消息仍保留在 paper_chat_messages。
        required_columns = {
            "summary_json": "TEXT",
            "summary_updated_at": "TIMESTAMP",
            "summary_turn_count": "INTEGER DEFAULT 0",
            "summary_last_turn_id": "TEXT",
        }
        cursor = conn.cursor()
        cursor.execute("PRAGMA table_info(paper_chat_sessions)")
        existing_columns = {row[1] for row in cursor.fetchall()}
        for column_name, column_definition in required_columns.items():
            if column_name not in existing_columns:
                cursor.execute(
                    f"ALTER TABLE paper_chat_sessions ADD COLUMN {column_name} {column_definition}"
                )
        conn.commit()

    def ensure_schema(self) -> None:
        with self.connection_provider.connect() as conn:
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
                CREATE TABLE IF NOT EXISTS agent_qa_index_continuations (
                    job_id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL DEFAULT '{default_user_id_sql}',
                    session_id TEXT NOT NULL,
                    arxiv_id TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'waiting_job',
                    pending_action_id TEXT,
                    step_id TEXT,
                    tool_name TEXT,
                    original_question TEXT,
                    resume_payload_json TEXT,
                    pending_action_json TEXT,
                    job_snapshot_json TEXT,
                    error_message TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    completed_at TIMESTAMP,
                    FOREIGN KEY(job_id) REFERENCES paper_index_jobs(job_id)
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
                CREATE INDEX IF NOT EXISTS idx_agent_qa_index_continuations_session
                ON agent_qa_index_continuations(user_id, session_id, status, updated_at DESC)
            ''')

            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_agent_qa_index_continuations_arxiv
                ON agent_qa_index_continuations(arxiv_id, status, updated_at DESC)
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
