import json
from typing import Any, Dict, List, Optional

from services.storage.sqlite.base import BaseSqliteStore
from services.storage.sqlite.shared import DEFAULT_USER_ID, logger


def _dumps_optional_json(value: Any) -> Optional[str]:
    """可选字段序列化：None 原样写入 NULL，以区分“未计算”和“计算结果为空”。"""
    return json.dumps(value) if value is not None else None


def _loads_json_or_default(raw: Any, default: Any = None) -> Any:
    """安全解析 JSON 文本列：空值或内容损坏时返回 default，单列损坏不影响整条画像读取。"""
    if not raw:
        return default
    try:
        return json.loads(raw)
    except (TypeError, ValueError):  # json.JSONDecodeError 是 ValueError 的子类
        return default


def _default_negative_feedback_stats() -> Dict[str, Any]:
    """负反馈统计的默认结构（未启用）；每次返回新字典，避免调用方之间共享可变对象。"""
    return {
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


class InterestVectorStore(BaseSqliteStore):
    """维护用户兴趣向量存取；推荐算法仍在上层服务中，避免存储层承担排序职责。"""

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
        """整行覆盖保存用户兴趣向量（每个用户只保留一条），成功返回 True。"""
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
                    _dumps_optional_json(interest_clusters),
                    _dumps_optional_json(weak_interest_pool),
                    _dumps_optional_json(disliked_vector_data),
                    _dumps_optional_json(disliked_paper_examples),
                    _dumps_optional_json(negative_feedback_stats),
                    _dumps_optional_json(negative_feedback_profile),
                ))

                conn.commit()
                logger.info(f"User interest vector saved for user: {user_id}")
                return True
        except Exception as e:
            logger.error(f"Error saving user interest vector: {str(e)}")
            return False

    def get_user_interest_vector(self, user_id: str = DEFAULT_USER_ID) -> Optional[Dict[str, Any]]:
        """读取用户兴趣向量，并把不同时期写入的负反馈字段统一补齐为 negative_feedback_profile_v1 结构。

        兼容三代数据：
        - 新数据：有完整 negative_feedback_profile，直接合并到默认结构上；
        - Step 1 数据：只有 examples/stats，按其内容推断 enabled/mode 后补齐 profile；
        - 更旧数据：没有任何独立负向字段，视为未启用负反馈并忽略旧的 disliked_vector_data。
        """
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT user_id, vector_data, paper_count, embedding_model, vector_dimension, cluster_count, profile_mode, interest_clusters, weak_interest_pool, disliked_vector_data, disliked_paper_examples, negative_feedback_stats, negative_feedback_profile, created_at, updated_at
                    FROM user_interest_vectors WHERE user_id = ?
                ''', (user_id,))

                row = cursor.fetchone()
                if row:
                    negative_feedback_stats = _default_negative_feedback_stats()
                    negative_feedback_profile = {
                        "version": "negative_feedback_profile_v1",
                        "enabled": False,
                        "mode": "none",
                        "hard_exclude_ids": [],
                        "examples": [],
                        "clusters": [],
                        "stats": negative_feedback_stats,
                    }
                    # 各 JSON 列独立容错：某一列损坏只回落该列的默认值，不影响主向量读取。
                    interest_clusters = _loads_json_or_default(row[7], [])
                    weak_interest_pool = _loads_json_or_default(row[8])
                    disliked_vector_data = _loads_json_or_default(row[9])
                    parsed_examples = _loads_json_or_default(row[10], [])
                    disliked_paper_examples = parsed_examples if isinstance(parsed_examples, list) else []
                    parsed_stats = _loads_json_or_default(row[11])
                    if isinstance(parsed_stats, dict):
                        negative_feedback_stats = {**negative_feedback_stats, **parsed_stats}
                    if row[12]:
                        parsed_profile = _loads_json_or_default(row[12])
                        if isinstance(parsed_profile, dict):
                            negative_feedback_profile = {**negative_feedback_profile, **parsed_profile}
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
