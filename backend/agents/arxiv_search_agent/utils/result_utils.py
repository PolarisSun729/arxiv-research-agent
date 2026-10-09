"""工具返回值和异常对象的统一解析工具。

这个模块的目标是把外部工具层、服务层返回的多样结构收口成稳定的读取方式，
避免 node 层在多个文件里重复处理 ok/error/data/trace 这些字段细节。

它主要解决两类问题：
1. 从嵌套结果里提取真正需要的业务字段；
2. 把异常对象和工具错误转换成适合日志、前端和调试使用的统一文本。
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional


def _extract_papers_from_tool_result(result: Mapping[str, Any]) -> List[Dict[str, Any]]:
    """从工具返回结果中提取论文列表，兼容不同嵌套结构。

    不同工具或调用层可能把论文列表放在 data、data.papers 或 data.result.papers 中，
    这里统一兼容这些常见形态，让上层节点不用反复写结构探测逻辑。
    """
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
    """从工具结果的 error 字段中提取适合展示的错误文本。

    如果同时存在 message 和 detail，会尽量拼成一条更完整的可读错误信息。
    """
    error = result.get("error")
    if not isinstance(error, dict):
        return None
    message = str(error.get("message", "") or "").strip()
    detail = error.get("detail")
    if message and detail:
        return f"{message}: {detail}"
    return message or None


def _to_plain_dict(value: Any) -> Dict[str, Any]:
    """把可能是 Pydantic 模型或其他对象的值尽量转成普通 dict。

    这样后续读取逻辑就可以统一按 Mapping 方式处理，而不用区分 model_dump/dict/原生 dict。
    """
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
    """统一判断工具结果是否表示成功。

    这里显式把 None 当成失败，避免上层把“没有 ok 字段”误当成 truthy 成功结果。
    """
    ok = result.get("ok")
    if isinstance(ok, bool):
        return ok
    if ok is None:
        return False
    return bool(ok)


def _result_text(result: Mapping[str, Any], key: str) -> Optional[str]:
    """从结果字典中读取指定文本字段，并做基础标准化。

    常用于 summary、message 这类可选文本字段的安全读取。
    """
    value = result.get(key)
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _result_mapping(result: Mapping[str, Any], key: str) -> Optional[Dict[str, Any]]:
    """从结果字典中安全读取某个子对象字段，仅当其本身为 dict 时返回。

    这个辅助函数的意义是减少上层 if isinstance(..., dict) 的重复样板代码。
    """
    value = result.get(key)
    return value if isinstance(value, dict) else None
