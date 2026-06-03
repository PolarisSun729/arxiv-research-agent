from __future__ import annotations

import json
from typing import Any, Dict, List, Mapping, Optional


def _extract_papers_from_tool_result(result: Mapping[str, Any]) -> List[Dict[str, Any]]:
    """从工具返回结果中提取论文列表，兼容不同嵌套结构。"""
    data = result.get("data")
    candidates: Any = data
    if isinstance(data, dict):
        if isinstance(data.get("papers"), list):
            candidates = data["papers"]
        elif isinstance(data.get("result"), dict) and isinstance(data["result"].get("papers"), list):
            candidates = data["result"]["papers"]
    if isinstance(candidates, list):
        return [item for item in candidates if isinstance(item, dict)]
    return []


def _extract_error_message(result: Mapping[str, Any]) -> Optional[str]:
    """从工具结果的 error 字段中提取适合展示的错误文本。"""
    error = result.get("error")
    if not isinstance(error, dict):
        return None
    message = str(error.get("message", "") or "").strip()
    detail = error.get("detail")
    if message and detail:
        return f"{message}: {detail}"
    return message or None


def _extract_exception_detail(exc: Exception) -> str:
    """从异常对象中提取更稳定的 detail 文本，兼容 HTTPException 等结构。"""
    detail = getattr(exc, "detail", None)
    if isinstance(detail, dict):
        message = str(detail.get("message") or detail.get("detail") or detail.get("error") or "").strip()
        if message:
            return message
        return json.dumps(detail, ensure_ascii=False)
    if detail is not None:
        text = str(detail).strip()
        if text:
            return text
    text = str(exc).strip()
    return text or exc.__class__.__name__


def _extract_exception_stage(exc: Exception, default_stage: str) -> str:
    """尽量从异常对象中恢复失败阶段名；恢复不到时返回默认阶段。"""
    stage = str(getattr(exc, "error_stage", "") or getattr(exc, "failed_stage", "") or "").strip()
    if stage:
        return stage
    detail = getattr(exc, "detail", None)
    if isinstance(detail, dict):
        stage = str(detail.get("stage") or detail.get("failed_stage") or "").strip()
        if stage:
            return stage
    return default_stage


def _to_plain_dict(value: Any) -> Dict[str, Any]:
    """把可能是 Pydantic 模型或其他对象的值尽量转成普通 dict。"""
    if value is None:
        return {}
    if isinstance(value, dict):
        return dict(value)
    if hasattr(value, "model_dump"):
        dumped = value.model_dump()
        return dumped if isinstance(dumped, dict) else {}
    if hasattr(value, "dict"):
        dumped = value.dict()
        return dumped if isinstance(dumped, dict) else {}
    return {}


def _result_ok(result: Mapping[str, Any]) -> bool:
    """统一判断工具结果是否表示成功。"""
    ok = result.get("ok")
    if isinstance(ok, bool):
        return ok
    if ok is None:
        return False
    return bool(ok)


def _result_text(result: Mapping[str, Any], key: str) -> Optional[str]:
    """从结果字典中读取指定文本字段，并做基础标准化。"""
    value = result.get(key)
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _result_mapping(result: Mapping[str, Any], key: str) -> Optional[Dict[str, Any]]:
    """从结果字典中安全读取某个子对象字段，仅当其本身为 dict 时返回。"""
    value = result.get(key)
    return value if isinstance(value, dict) else None
