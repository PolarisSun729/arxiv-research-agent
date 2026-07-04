import json
from typing import Any, Dict, List, Optional

from services.storage.database.shared import DEFAULT_USER_ID, logger


class InterestVectorMixin:
    """维护用户兴趣向量存取；推荐算法仍在上层服务中，避免存储层承担排序职责。"""

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
        disliked_paper_examples: Optional[List[Dict[str, Any]]] = None,
        negative_feedback_stats: Optional[Dict[str, Any]] = None,
        negative_feedback_profile: Optional[Dict[str, Any]] = None,
    ) -> bool:
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT OR REPLACE INTO user_interest_vectors
                    (user_id, vector_data, paper_count, embedding_model, vector_dimension, cluster_count, profile_mode, interest_clusters, weak_interest_pool, disliked_vector_data, disliked_paper_examples, negative_feedback_stats, negative_feedback_profile, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
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
                    json.dumps(disliked_paper_examples) if disliked_paper_examples is not None else None,
                    json.dumps(negative_feedback_stats) if negative_feedback_stats is not None else None,
                    json.dumps(negative_feedback_profile) if negative_feedback_profile is not None else None,
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
                    SELECT user_id, vector_data, paper_count, embedding_model, vector_dimension, cluster_count, profile_mode, interest_clusters, weak_interest_pool, disliked_vector_data, disliked_paper_examples, negative_feedback_stats, negative_feedback_profile, created_at, updated_at
                    FROM user_interest_vectors WHERE user_id = ?
                ''', (user_id,))

                row = cursor.fetchone()
                if row:
                    interest_clusters = None
                    weak_interest_pool = None
                    disliked_vector_data = None
                    disliked_paper_examples = []
                    negative_feedback_stats = {
                        "enabled": False,
                        "mode": "none",
                        "total_disliked": 0,
                        "usable_disliked": 0,
                        "unresolved_disliked": 0,
                        "milvus_count": 0,
                        "fallback_count": 0,
                        "stored_examples": 0,
                        "negative_cluster_count": 0,
                        "vector_available": False,
                        "participates_in_main_vector": False,
                    }
                    negative_feedback_profile = {
                        "version": "negative_feedback_profile_v1",
                        "enabled": False,
                        "mode": "none",
                        "hard_exclude_ids": [],
                        "examples": [],
                        "clusters": [],
                        "stats": negative_feedback_stats,
                    }
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
                    if row[10]:
                        try:
                            parsed_examples = json.loads(row[10])
                            disliked_paper_examples = parsed_examples if isinstance(parsed_examples, list) else []
                        except (TypeError, ValueError, json.JSONDecodeError):
                            disliked_paper_examples = []
                    if row[11]:
                        try:
                            parsed_stats = json.loads(row[11])
                            if isinstance(parsed_stats, dict):
                                negative_feedback_stats = {**negative_feedback_stats, **parsed_stats}
                        except (TypeError, ValueError, json.JSONDecodeError):
                            negative_feedback_stats = {**negative_feedback_stats}
                    if row[12]:
                        try:
                            parsed_profile = json.loads(row[12])
                            if isinstance(parsed_profile, dict):
                                negative_feedback_profile = {**negative_feedback_profile, **parsed_profile}
                        except (TypeError, ValueError, json.JSONDecodeError):
                            negative_feedback_profile = {**negative_feedback_profile}
                    elif row[10] or row[11]:
                        # Step 1 画像可能只保存 examples/stats，还没有完整 profile 字段；读取时补成 v1 结构，
                        # 让排序层始终消费同一套负向画像，同时避免把“只有 examples”的旧数据误判为禁用。
                        legacy_examples_available = bool(disliked_paper_examples)
                        legacy_enabled = (
                            bool(negative_feedback_stats.get("enabled"))
                            if row[11]
                            else legacy_examples_available
                        )
                        legacy_mode = str(negative_feedback_stats.get("mode") or "").strip()
                        if legacy_enabled and legacy_examples_available and legacy_mode in ("", "none"):
                            legacy_mode = "examples"
                        elif not legacy_mode:
                            legacy_mode = "none"
                        negative_feedback_stats = {
                            **negative_feedback_stats,
                            "enabled": legacy_enabled,
                            "mode": legacy_mode,
                            "total_disliked": max(
                                int(negative_feedback_stats.get("total_disliked") or 0),
                                len(disliked_paper_examples),
                            ),
                            "usable_disliked": max(
                                int(negative_feedback_stats.get("usable_disliked") or 0),
                                len(disliked_paper_examples),
                            ),
                            "stored_examples": len(disliked_paper_examples),
                            "vector_available": disliked_vector_data is not None,
                        }
                        negative_feedback_profile = {
                            **negative_feedback_profile,
                            "enabled": legacy_enabled,
                            "mode": legacy_mode,
                            "examples": disliked_paper_examples,
                            "clusters": [],
                            "stats": negative_feedback_stats,
                        }
                    else:
                        # 更旧画像没有独立负向字段时按空负反馈处理，避免旧的 disliked_vector_data 继续影响新排序语义。
                        disliked_vector_data = None
                    negative_feedback_profile["stats"] = {
                        **negative_feedback_stats,
                        **dict(negative_feedback_profile.get("stats") or {}),
                    }
                    negative_feedback_stats = dict(negative_feedback_profile["stats"])
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
                        'disliked_paper_examples': disliked_paper_examples,
                        'negative_feedback_stats': negative_feedback_stats,
                        'negative_feedback_profile': negative_feedback_profile,
                        'created_at': row[13],
                        'updated_at': row[14]
                    }
                return None
        except Exception as e:
            logger.error(f"Error getting user interest vector: {str(e)}")
            return None
