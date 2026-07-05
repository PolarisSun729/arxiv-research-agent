import uuid
from typing import Any, Dict, Optional

from services.storage.sqlite.base import BaseSqliteStore
from services.storage.sqlite.profile_normalization import normalize_profile_categories, normalize_profile_topics
from services.storage.sqlite.shared import (
    PROFILE_EXTRACTOR_VERSION,
    PROFILE_NORMALIZER_VERSION,
    logger,
)


class PaperProfileEvidenceStore(BaseSqliteStore):
    """维护单篇论文的画像证据缓存；画像合并和 snapshot 激活仍留在 profile 主流程。"""

    def upsert_paper_profile_evidence(self, arxiv_id: str, evidence: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        payload = dict(evidence or {})
        # 这里复用现有 profile 归一化 helper，保持证据缓存与最终画像抽取的字段清洗规则一致。
        normalized = {
            "concepts": normalize_profile_topics(
                payload.get("technical_concepts")
                or payload.get("concepts")
                or payload.get("topics")
                or [item.get("label") for item in payload.get("candidate_concepts") or [] if isinstance(item, dict)],
                limit=20,
            ),
            "methods": normalize_profile_topics(payload.get("methods"), limit=20),
            "tasks": normalize_profile_topics(payload.get("tasks"), limit=20),
            "objects": normalize_profile_topics(payload.get("research_objects") or payload.get("objects"), limit=20),
            "applications": normalize_profile_topics(payload.get("application_domains") or payload.get("applications"), limit=20),
            "categories": normalize_profile_categories(payload.get("categories"), limit=20),
        }
        evidence_id = str(uuid.uuid4())
        try:
            with self._get_connection() as conn:
                conn.execute(
                    '''
                    INSERT INTO paper_profile_evidence (
                        evidence_id, arxiv_id, concepts_json, methods_json, tasks_json, objects_json,
                        applications_json, categories_json, raw_payload_json, extractor_version, normalizer_version, updated_at
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                    ON CONFLICT(arxiv_id, extractor_version, normalizer_version) DO UPDATE SET
                        concepts_json = excluded.concepts_json,
                        methods_json = excluded.methods_json,
                        tasks_json = excluded.tasks_json,
                        objects_json = excluded.objects_json,
                        applications_json = excluded.applications_json,
                        categories_json = excluded.categories_json,
                        raw_payload_json = excluded.raw_payload_json,
                        updated_at = CURRENT_TIMESTAMP
                    ''',
                    (
                        evidence_id,
                        arxiv_id,
                        self._serialize_json_field(normalized["concepts"]),
                        self._serialize_json_field(normalized["methods"]),
                        self._serialize_json_field(normalized["tasks"]),
                        self._serialize_json_field(normalized["objects"]),
                        self._serialize_json_field(normalized["applications"]),
                        self._serialize_json_field(normalized["categories"]),
                        self._serialize_json_field(payload),
                        PROFILE_EXTRACTOR_VERSION,
                        PROFILE_NORMALIZER_VERSION,
                    ),
                )
                conn.commit()
        except Exception as e:
            logger.error(f"Error upserting paper profile evidence: {str(e)}")
        return {"arxiv_id": arxiv_id, **normalized}

    def get_paper_profile_evidence(
        self,
        arxiv_id: str,
        *,
        extractor_version: str = PROFILE_EXTRACTOR_VERSION,
        normalizer_version: str = PROFILE_NORMALIZER_VERSION,
    ) -> Optional[Dict[str, Any]]:
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    '''
                    SELECT arxiv_id, concepts_json, methods_json, tasks_json, objects_json,
                           applications_json, categories_json, raw_payload_json,
                           extractor_version, normalizer_version, created_at, updated_at
                    FROM paper_profile_evidence
                    WHERE arxiv_id = ? AND extractor_version = ? AND normalizer_version = ?
                    ''',
                    (arxiv_id, extractor_version, normalizer_version),
                )
                row = cursor.fetchone()
                if not row:
                    return None
                raw_payload = self._deserialize_json_field(row[7]) or {}
                if isinstance(raw_payload, dict) and raw_payload:
                    raw_payload.setdefault("arxiv_id", row[0])
                    raw_payload.setdefault("extractor_version", row[8])
                    raw_payload.setdefault("created_at", row[10])
                    raw_payload.setdefault("updated_at", row[11])
                    return raw_payload
                # 旧库可能只有拆分字段没有完整 raw_payload；返回兼容结构，避免重建任务因老证据缺字段失败。
                return {
                    "arxiv_id": row[0],
                    "technical_concepts": self._deserialize_json_field(row[1]) or [],
                    "methods": self._deserialize_json_field(row[2]) or [],
                    "tasks": self._deserialize_json_field(row[3]) or [],
                    "research_objects": self._deserialize_json_field(row[4]) or [],
                    "application_domains": self._deserialize_json_field(row[5]) or [],
                    "categories": self._deserialize_json_field(row[6]) or [],
                    "candidate_concepts": [],
                    "extractor_version": row[8],
                    "normalizer_version": row[9],
                    "schema_valid": False,
                    "error_message": "legacy_evidence_without_full_card",
                    "created_at": row[10],
                    "updated_at": row[11],
                }
        except Exception as e:
            logger.error(f"Error getting paper profile evidence: {str(e)}")
            return None
