"""检索层共用的数值转换与分数归一化工具。

这些函数原先在 route_retriever / index_hit_aggregator / rerank_service / context_expansion 中
各自复制了一份；集中到这里后，各处的“空值怎么处理、相等分数怎么归一化”保持同一套规则。
"""

import math
from typing import Any, List, Optional


def first_float(*values: Any) -> Optional[float]:
    """按顺序返回第一个能转成 float 的值；全部为空或无法转换时返回 None。

    用于“优先取 A 分数，缺失时退回 B 分数”这类多来源取分场景，空字符串视同缺失。
    全部缺失时不默认成 0.0，由调用方决定是报错还是按缺失处理，避免缺分被静默伪装成 0 分。
    """
    for value in values:
        if value is None or value == "":
            continue
        try:
            return float(value)
        except (TypeError, ValueError):
            continue
    return None


def require_float(value: Any, *, what: str) -> float:
    """把必填分数转成 float；缺失或无法转换时抛 ValueError，what 用于说明是哪个分数缺失。"""
    result = optional_float(value)
    if result is None:
        raise ValueError(f"Missing required score: {what} (got {value!r})")
    return result


def min_max_normalize(scores: List[float]) -> List[float]:
    """把一组分数线性缩放到 [0, 1]。

    所有分数（在浮点误差范围内）相等时无法区分高低，统一返回 1.0，
    表示“这一批候选同样好”，而不是全部归零后被下游阈值过滤掉。
    """
    if not scores:
        return []
    min_score = min(scores)
    max_score = max(scores)
    if math.isclose(min_score, max_score):
        return [1.0 for _ in scores]
    scale = max_score - min_score
    return [(score - min_score) / scale for score in scores]


def optional_int(value: Any) -> Optional[int]:
    """宽松地转成 int；None、空字符串或无法转换时返回 None（区分“缺失”和 0）。"""
    try:
        if value in (None, ""):
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def optional_float(value: Any) -> Optional[float]:
    """宽松地转成 float；None、空字符串或无法转换时返回 None（区分“缺失”和 0.0）。"""
    try:
        if value in (None, ""):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


__all__ = ["first_float", "min_max_normalize", "optional_int", "optional_float", "require_float"]
