import json
from typing import Any


class JsonFieldMixin:
    """提供 SQLite JSON 文本字段的兼容编解码，保持原 DatabaseService 私有 helper 调用不变。"""

    @staticmethod
    def _serialize_json_field(value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, str):
            return value
        # SQLite 仍以 TEXT 保存 JSON；ensure_ascii=False 保留旧行为，避免中文内容被转义后影响可读性。
        return json.dumps(value, ensure_ascii=False)

    @staticmethod
    def _deserialize_json_field(value: Any) -> Any:
        if not isinstance(value, str):
            return value
        text = value.strip()
        if not text:
            return None
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            # 历史字段可能存的是普通字符串；解析失败时原样返回，避免迁移期误清空旧值。
            return value
