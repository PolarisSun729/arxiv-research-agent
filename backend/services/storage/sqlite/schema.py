from services.storage.sqlite.connection import SqliteConnectionProvider
from services.storage.sqlite.shared import DEFAULT_USER_ID, logger


class StorageSchemaMigrator:
    """集中负责 SQLite schema 初始化，业务 store 不再隐式建表。"""

    def __init__(self, connection_provider: SqliteConnectionProvider) -> None:
        self.connection_provider = connection_provider


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
                    active_index_version TEXT,
                    active_build_id TEXT,
                    previous_build_id TEXT,
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
                    recipe_version TEXT NOT NULL DEFAULT 'paper_qa_index_v1',
                    attempt_count INTEGER NOT NULL DEFAULT 0,
                    max_attempts INTEGER NOT NULL DEFAULT 3,
                    worker_id TEXT,
                    lease_acquired_at TIMESTAMP,
                    lease_expires_at TIMESTAMP,
                    last_heartbeat_at TIMESTAMP,
                    result_json TEXT,
                    failure_code TEXT,
                    stage_message TEXT,
                    completed_at TIMESTAMP,
                    heartbeat_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')

            cursor.execute('''
                CREATE TABLE IF NOT EXISTS paper_index_job_attempts (
                    attempt_id TEXT PRIMARY KEY,
                    job_id TEXT NOT NULL,
                    attempt_no INTEGER NOT NULL,
                    worker_id TEXT NOT NULL,
                    status TEXT NOT NULL,
                    started_at TIMESTAMP NOT NULL,
                    heartbeat_at TIMESTAMP,
                    finished_at TIMESTAMP,
                    failed_stage TEXT,
                    error_code TEXT,
                    error_message TEXT,
                    UNIQUE(job_id, attempt_no),
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
                    schema_version INTEGER NOT NULL DEFAULT 2,
                    user_id TEXT NOT NULL DEFAULT '{default_user_id_sql}',
                    session_id TEXT NOT NULL,
                    thread_id TEXT NOT NULL,
                    runtime_state_json TEXT,
                    graph_state_json TEXT,
                    interaction_json TEXT,
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
                CREATE TABLE IF NOT EXISTS approval_grants (
                    grant_id TEXT PRIMARY KEY,
                    interaction_id TEXT NOT NULL UNIQUE,
                    user_id TEXT NOT NULL,
                    session_id TEXT NOT NULL,
                    thread_id TEXT NOT NULL,
                    plan_id TEXT NOT NULL,
                    step_id TEXT NOT NULL,
                    tool_name TEXT NOT NULL,
                    arguments_fingerprint TEXT NOT NULL,
                    status TEXT NOT NULL,
                    approved_at TEXT NOT NULL,
                    consumed_at TEXT,
                    revoked_at TEXT
                )
            ''')

            cursor.execute('''
                CREATE TABLE IF NOT EXISTS side_effect_invocations (
                    invocation_id TEXT PRIMARY KEY,
                    grant_id TEXT NOT NULL UNIQUE,
                    plan_id TEXT NOT NULL,
                    step_id TEXT NOT NULL,
                    tool_name TEXT NOT NULL,
                    arguments_fingerprint TEXT NOT NULL,
                    status TEXT NOT NULL,
                    prepared_at TEXT NOT NULL,
                    started_at TEXT,
                    finished_at TEXT,
                    result_summary_json TEXT,
                    error_code TEXT,
                    FOREIGN KEY(grant_id) REFERENCES approval_grants(grant_id)
                )
            ''')

            cursor.execute('''
                CREATE TABLE IF NOT EXISTS agent_work_continuations (
                    continuation_id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    session_id TEXT NOT NULL,
                    thread_id TEXT NOT NULL,
                    runtime_checkpoint_id TEXT NOT NULL,
                    interaction_id TEXT NOT NULL,
                    grant_id TEXT NOT NULL,
                    invocation_id TEXT NOT NULL UNIQUE,
                    plan_id TEXT NOT NULL,
                    step_id TEXT NOT NULL,
                    tool_name TEXT NOT NULL,
                    arguments_fingerprint TEXT NOT NULL,
                    handler_name TEXT NOT NULL,
                    job_id TEXT,
                    job_idempotency_key TEXT,
                    status TEXT NOT NULL,
                    handler_state_json TEXT,
                    display_summary_json TEXT,
                    validated_result_json TEXT,
                    resume_run_id TEXT,
                    error_code TEXT,
                    error_message TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    ready_at TEXT,
                    expires_at TEXT,
                    terminal_at TEXT,
                    FOREIGN KEY(grant_id) REFERENCES approval_grants(grant_id),
                    FOREIGN KEY(invocation_id) REFERENCES side_effect_invocations(invocation_id)
                )
            ''')

            cursor.execute('''
                CREATE TABLE IF NOT EXISTS agent_resume_runs (
                    resume_run_id TEXT PRIMARY KEY,
                    continuation_id TEXT NOT NULL UNIQUE,
                    user_id TEXT NOT NULL,
                    session_id TEXT NOT NULL,
                    thread_id TEXT NOT NULL,
                    status TEXT NOT NULL,
                    started_at TEXT,
                    finished_at TEXT,
                    final_response_json TEXT,
                    error_code TEXT,
                    error_message TEXT,
                    result_retrieved_at TEXT,
                    FOREIGN KEY(continuation_id) REFERENCES agent_work_continuations(continuation_id)
                )
            ''')

            cursor.execute('''
                CREATE TABLE IF NOT EXISTS agent_work_events (
                    event_id TEXT PRIMARY KEY,
                    event_type TEXT NOT NULL,
                    occurred_at TEXT NOT NULL,
                    continuation_id TEXT,
                    job_id TEXT,
                    attempt_no INTEGER,
                    invocation_id TEXT,
                    resume_run_id TEXT,
                    run_id TEXT,
                    stage TEXT,
                    progress INTEGER,
                    error_code TEXT,
                    safe_metadata_json TEXT
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
                CREATE INDEX IF NOT EXISTS idx_paper_index_jobs_idempotency_status
                ON paper_index_jobs(idempotency_key, status, heartbeat_at)
            ''')

            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_paper_index_jobs_claim
                ON paper_index_jobs(status, lease_expires_at, created_at)
            ''')

            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_paper_index_job_attempts_job
                ON paper_index_job_attempts(job_id, attempt_no DESC)
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
                CREATE INDEX IF NOT EXISTS idx_agent_work_continuations_owner_status
                ON agent_work_continuations(user_id, session_id, status, updated_at DESC)
            ''')

            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_agent_work_continuations_job
                ON agent_work_continuations(job_id, status)
            ''')

            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_agent_work_continuations_submitting
                ON agent_work_continuations(status, job_idempotency_key, updated_at)
            ''')

            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_agent_resume_runs_owner
                ON agent_resume_runs(user_id, session_id, status)
            ''')

            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_agent_work_events_continuation
                ON agent_work_events(continuation_id, occurred_at)
            ''')

            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_langgraph_checkpoints_thread_updated
                ON langgraph_checkpoints(thread_id, checkpoint_ns, updated_at DESC)
            ''')

            conn.commit()
            logger.info("Database tables initialized successfully")

