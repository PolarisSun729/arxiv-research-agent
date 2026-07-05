import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from services.storage.sqlite.base import BaseSqliteStore
from services.storage.sqlite.shared import logger

PAPER_INDEX_ACTIVE_JOB_STATUSES = ("pending", "running", "retrying")


class PaperQAIndexStore(BaseSqliteStore):
    """维护 Paper QA 索引版本、active 指针和后台构建任务的 SQLite 存储。"""

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
                        # 只有已完成向量写入并通过校验的新版本才能成为 active，避免半成品被问答链路读到。
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
                        cursor.execute(
                            """
                            SELECT collection_name, chunk_count, embedding_model, pdf_path, chunk_file,
                                   retrieval_index_file, retrieval_index_count, retrieval_index_types, retrieval_index_version,
                                   sparse_index_dir, sparse_index_manifest_file, sparse_index_document_count, sparse_index_token_count, sparse_index_backend,
                                   sparse_index_schema_version, sparse_index_source_file, sparse_index_source_hash, sparse_index_avgdl,
                                   embedding_file, loading_method, chunking_strategy
                            FROM paper_qa_index
                            WHERE arxiv_id = ?
                            """,
                            (arxiv_id,),
                        )
                        old_artifact_row = cursor.fetchone()
                        if old_artifact_row:
                            # 旧 active 版本只延迟清理；先把当前 active 单行的 artifact 快照回填到版本行，避免 cleanup_pending 丢失文件路径。
                            cursor.execute(
                                """
                                UPDATE paper_qa_index_versions
                                SET collection_name = ?,
                                    chunk_count = ?,
                                    embedding_model = ?,
                                    pdf_path = ?,
                                    chunk_file = ?,
                                    retrieval_index_file = ?,
                                    retrieval_index_count = ?,
                                    retrieval_index_types = ?,
                                    retrieval_index_version = ?,
                                    sparse_index_dir = ?,
                                    sparse_index_manifest_file = ?,
                                    sparse_index_document_count = ?,
                                    sparse_index_token_count = ?,
                                    sparse_index_backend = ?,
                                    sparse_index_schema_version = ?,
                                    sparse_index_source_file = ?,
                                    sparse_index_source_hash = ?,
                                    sparse_index_avgdl = ?,
                                    embedding_file = ?,
                                    loading_method = ?,
                                    chunking_strategy = ?
                                WHERE build_id = ?
                                """,
                                (*old_artifact_row, old_build_id),
                            )
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
