import uuid
from typing import Any, Dict, List, Optional

from services.storage.sqlite.base import BaseSqliteStore
from services.storage.sqlite.profile_normalization import (
    looks_like_profile_arxiv_category,
    looks_like_profile_arxiv_id,
    looks_like_profile_paper_title,
    normalize_profile_categories,
    normalize_profile_list_value,
    normalize_profile_papers,
    normalize_profile_topics,
)
from services.storage.sqlite.shared import (
    DEFAULT_USER_ID,
    PROFILE_BUILD_VERSION,
    PROFILE_EXTRACTOR_VERSION,
    PROFILE_NORMALIZER_VERSION,
    logger,
)
from services.storage.sqlite.stores.profile_build_jobs import ProfileBuildJobStore
from services.storage.sqlite.stores.profile_events import ProfileEventStore


class ResearchProfileStore(BaseSqliteStore):
    """维护 manual/generated/effective 研究画像、快照和 legacy 画像迁移。"""

    def __init__(
        self,
        connection_provider,
        profile_event_store: ProfileEventStore,
        profile_build_job_store: ProfileBuildJobStore,
    ) -> None:
        super().__init__(connection_provider)
        self.profile_event_store = profile_event_store
        self.profile_build_job_store = profile_build_job_store

    def record_user_profile_event(self, *args, **kwargs):
        return self.profile_event_store.record_user_profile_event(*args, **kwargs)

    def update_user_profile_build_job(self, *args, **kwargs):
        return self.profile_build_job_store.update_user_profile_build_job(*args, **kwargs)

    def migrate_legacy_profiles(self) -> None:
        with self._get_connection() as conn:
            self._migrate_legacy_research_profiles(conn)

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
        return normalize_profile_list_value(values, limit=limit)

    @staticmethod
    def _looks_like_profile_arxiv_id(value: str) -> bool:
        return looks_like_profile_arxiv_id(value)

    @staticmethod
    def _looks_like_profile_arxiv_category(value: str) -> bool:
        return looks_like_profile_arxiv_category(value)

    @staticmethod
    def _looks_like_profile_paper_title(value: str) -> bool:
        return looks_like_profile_paper_title(value)

    @classmethod
    def _normalize_profile_topics(cls, values: Any, limit: int = 30) -> List[str]:
        return normalize_profile_topics(values, limit=limit)

    @classmethod
    def _normalize_profile_categories(cls, values: Any, limit: int = 20) -> List[str]:
        return normalize_profile_categories(values, limit=limit)

    @classmethod
    def _normalize_profile_papers(cls, values: Any, limit: int = 20) -> List[str]:
        return normalize_profile_papers(values, limit=limit)

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
